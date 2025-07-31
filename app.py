"""
Flask + OpenCV + Ultralytics YOLO (웹캠 실시간 스트리밍)
- 네가 준 순수 스크립트를 베이스로, 웹에서 볼 수 있게 MJPEG 스트림(/video_feed)으로 제공
- /webcam 페이지에서 <img>로 스트림 표시
- 좌우반전(flip) 유지, CAP_DSHOW 유지(윈도우 속도 개선), 해상도/FPS 설정 유지
- Ultralytics 버전에 따라 Results.plot()이 없을 수 있어 수동 그리기 폴백 추가
"""

import os
import time
from pathlib import Path

import cv2
import numpy as np
from flask import Flask, Response, render_template, request
from PIL import Image  # (업로드 기능은 안 쓰지만 PIL 설치 유도 겸)
import torch
from ultralytics import YOLO

# -----------------------------
# 기본 설정 (필요시 여기만 수정)
# -----------------------------
MODEL_PATH = "./yolo12x.pt"  # 네가 준 코드 그대로
CAM_INDEX = 0                # 0: 기본 카메라
TARGET_W, TARGET_H = 1280, 720
TARGET_FPS = 60
FLIP_LR = True               # 좌우 반전 유지

# PyTorch 2.6+ 가중치 로드 제한 회피 (환경에 따라 무시될 수 있음)
os.environ.setdefault("TORCH_LOAD_WEIGHTS_ONLY", "0")

# -----------------------------
# Flask 앱
# -----------------------------
app = Flask(__name__, template_folder="templates", static_folder="static")

# -----------------------------
# YOLO 로딩 (GPU 자동 선택)
# -----------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"
model = YOLO(MODEL_PATH).to(device)

# -----------------------------
# 박스 직접 그리기(Results.plot() 폴백)
# -----------------------------
def draw_from_boxes(res, model, canvas: np.ndarray) -> np.ndarray:
    """
    Ultralytics Results 객체에서 boxes를 읽어 수동으로 박스/라벨/신뢰도를 그린다.
    - numpy()를 호출하지 않고 tolist()만 사용 → 'Numpy is not available' 환경에서도 안전
    """
    out = canvas.copy()
    boxes_obj = getattr(res, "boxes", None)
    if boxes_obj is None:
        return out

    def _to_list(t):
        if t is None:
            return None
        try:
            return t.detach().cpu().float().tolist()
        except Exception:
            return None

    xyxy_list = _to_list(getattr(boxes_obj, "xyxy", None))
    conf_list = _to_list(getattr(boxes_obj, "conf", None))
    cls_list  = _to_list(getattr(boxes_obj, "cls", None))

    if not xyxy_list:
        return out

    names = getattr(model, "names", {})
    if isinstance(names, dict):
        def name_of(i):
            i = int(i)
            return names.get(i, str(i))
    else:
        def name_of(i):
            i = int(i)
            if isinstance(names, (list, tuple)) and 0 <= i < len(names):
                return names[i]
            return str(i)

    for i, box in enumerate(xyxy_list):
        x1, y1, x2, y2 = map(int, box)
        label = ""
        if cls_list is not None and i < len(cls_list):
            label = name_of(cls_list[i])
        if conf_list is not None and i < len(conf_list):
            c = conf_list[i]
            label = f"{label} {c:.2f}" if label else f"{c:.2f}"

        cv2.rectangle(out, (x1, y1), (x2, y2), (0, 255, 0), 2)
        if label:
            y_text = max(y1 - 5, 15)
            cv2.putText(
                out, label, (x1, y_text),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2
            )
    return out

# -----------------------------
# 카메라 열기 (윈도우는 CAP_DSHOW 유지)
# -----------------------------
def open_camera(index: int) -> cv2.VideoCapture:
    if os.name == "nt":
        cap = cv2.VideoCapture(cv2.CAP_DSHOW + index)
    else:
        cap = cv2.VideoCapture(index)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  TARGET_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, TARGET_H)
    cap.set(cv2.CAP_PROP_FPS,          TARGET_FPS)
    return cap

# -----------------------------
# 프레임 제너레이터(MJPEG 스트림)
# -----------------------------
def gen_frames():
    cap = open_camera(CAM_INDEX)

    # 실제 적용된 값 로그(서버 콘솔에서 확인용)
    print("RES:", cap.get(cv2.CAP_PROP_FRAME_WIDTH), "x", cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print("FPS:", cap.get(cv2.CAP_PROP_FPS))

    if not cap.isOpened():
        # 카메라 실패 시, 안내 프레임 반복 송출
        while True:
            frame = np.ones((480, 640, 3), dtype=np.uint8) * 255
            cv2.putText(frame, "웹캠을 열 수 없습니다.", (50, 240),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            ok, buf = cv2.imencode(".jpg", frame)
            if not ok:
                break
            yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" +
                   buf.tobytes() + b"\r\n")
        return

    # 렌더 루프
    while True:
        ret, frame = cap.read()
        if not ret:
            time.sleep(0.005)
            continue

        # 좌우 반전
        if FLIP_LR:
            frame = cv2.flip(frame, 1)

        annotated = frame
        try:
            # YOLO 추론
            results = model(annotated, verbose=False)
            res = results[0]

            # 1) 가급적 plot() (네가 준 코드 그대로)
            #    단, 일부 버전/환경에서 plot()이 없거나(또는 numpy 연동 문제) 예외가 나면
            #    2) 수동 박스 그리기(draw_from_boxes)로 폴백
            try:
                if hasattr(res, "plot"):
                    annotated = res.plot()  # BGR ndarray
                else:
                    annotated = draw_from_boxes(res, model, annotated)
            except Exception:
                annotated = draw_from_boxes(res, model, annotated)

        except Exception as e:
            # 추론 실패 시 안내 오버레이
            annotated = frame.copy()
            cv2.putText(annotated, f"Detection error: {type(e).__name__}",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        # JPEG 인코딩 후 스트림 전송
        ok, buf = cv2.imencode(".jpg", annotated)
        if not ok:
            continue
        yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" +
               buf.tobytes() + b"\r\n")

    cap.release()

# -----------------------------
# 라우트
# -----------------------------
@app.route("/")
def index():
    # 간단히 /webcam으로 리다이렉트하는 형태로 구성해도 됨
    return render_template("webcam.html")

@app.route("/webcam")
def webcam_page():
    return render_template("webcam.html")

@app.route("/video_feed")
def video_feed():
    # 브라우저 <img>에서 직접 표시 가능한 MJPEG 스트림
    return Response(gen_frames(), mimetype="multipart/x-mixed-replace; boundary=frame")

@app.route("/healthz")
def healthz():
    return {"status": "ok"}

# -----------------------------
# 실행
# -----------------------------
if __name__ == "__main__":
    # 0.0.0.0 바인딩: 같은 네트워크의 다른 기기도 접속 가능
    app.run(host="0.0.0.0", port=5000, debug=True)
