"""
Flask + OpenCV + Ultralytics YOLO (웹캠 실시간 스트리밍)

이 스크립트는 웹브라우저에서 실시간으로 YOLO 객체 검출 결과를 볼 수 있도록
MJPEG 스트림을 제공합니다. Start Bootstrap의 SB Admin 템플릿을 활용하여
깔끔한 UI를 제공하며, YOLO 추론 과정은 torch.no_grad() 컨텍스트로 감싸
메모리 사용을 최소화합니다. 또한 요청을 스레드로 처리하여 모델 실행이
플라스크 전체를 멈추지 않도록 합니다.

변경 요약:
- 서버 시작 시 백그라운드 스레드에서만 모델을 실행 (웹은 화면만 구독)
- Windows 카메라 백엔드: DSHOW → MSMF → ANY 폴백
- Flask reloader로 인한 이중 실행 방지(use_reloader=False)
- Python 3.9 호환 타입 주석
"""

import os
import time
import threading
from typing import Optional

import cv2
import numpy as np
from flask import Flask, Response, render_template
from PIL import Image  # (업로드 기능은 안 쓰지만 PIL 설치 유도 겸)
import torch
from ultralytics import YOLO

# -----------------------------
# 기본 설정 (필요시 여기만 수정)
# -----------------------------
MODEL_PATH = "./yolo12x.pt"
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
# 글로벌 상태 (백그라운드 추론 스레드가 계속 갱신)
# -----------------------------
output_frame = None  # type: Optional[np.ndarray]
frame_lock = threading.Lock()

# -----------------------------
# 박스 직접 그리기(Results.plot() 폴백)
# -----------------------------
def draw_from_boxes(res, model, canvas: np.ndarray) -> np.ndarray:
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
            i = int(i); return names.get(i, str(i))
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
            cv2.putText(out, label, (x1, y_text),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    return out

# -----------------------------
# 카메라 열기 (Windows는 백엔드 폴백 포함)
# -----------------------------
def open_camera(index: int) -> cv2.VideoCapture:
    """
    Windows: DSHOW 우선 → MSMF → ANY 순으로 시도.
    Linux/기타: 기본 VideoCapture(index).
    성공 시 해상도/FPS 및 MJPG 설정을 시도.
    """
    if os.name == "nt":
        for api in (cv2.CAP_DSHOW, cv2.CAP_MSMF, cv2.CAP_ANY):
            cap = cv2.VideoCapture(index, api)
            if cap.isOpened():
                # 해상도/FPS 설정
                cap.set(cv2.CAP_PROP_FRAME_WIDTH,  TARGET_W)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, TARGET_H)
                cap.set(cv2.CAP_PROP_FPS,          TARGET_FPS)
                # MJPG로 설정(대역폭/지연 개선에 도움이 되는 경우가 많음)
                try:
                    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
                except Exception:
                    pass
                return cap
        # 열기 실패
        return cv2.VideoCapture()  # unopened
    else:
        cap = cv2.VideoCapture(index)
        if cap.isOpened():
            cap.set(cv2.CAP_PROP_FRAME_WIDTH,  TARGET_W)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, TARGET_H)
            cap.set(cv2.CAP_PROP_FPS,          TARGET_FPS)
        return cap

# -----------------------------
# 프레임 제너레이터(MJPEG 스트림)
# -----------------------------
def gen_frames():
    """
    웹 클라이언트에 최신 output_frame만 전송.
    """
    global output_frame
    while True:
        with frame_lock:
            frame = None if output_frame is None else output_frame.copy()
        if frame is None:
            placeholder = np.ones((480, 640, 3), dtype=np.uint8) * 255
            cv2.putText(placeholder, "Initializing stream...", (50, 240),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
            ok, buf = cv2.imencode(".jpg", placeholder)
            if ok:
                yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" +
                       buf.tobytes() + b"\r\n")
            time.sleep(0.05)
            continue

        ok, buf = cv2.imencode(".jpg", frame)
        if ok:
            yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" +
                   buf.tobytes() + b"\r\n")
        # 과부하 방지
        time.sleep(0.01)

# -----------------------------
# 백그라운드 YOLO 추론 루프
# -----------------------------
def yolo_inference_loop():
    """
    서버 시작 시 1회만 실행되는 백그라운드 스레드:
    - 카메라에서 프레임 수집
    - YOLO 추론 및 결과 렌더링
    - 최신 프레임을 output_frame에 보관
    카메라 실패 시 재시도(backoff).
    """
    global output_frame

    retry = 0
    while True:
        cap = open_camera(CAM_INDEX)
        if not cap.isOpened():
            retry += 1
            wait = min(5.0, 0.5 * retry)
            print(f"[YOLO THREAD] Camera open failed. retry={retry} wait={wait:.1f}s")
            time.sleep(wait)
            continue

        print("[YOLO THREAD] RES:", cap.get(cv2.CAP_PROP_FRAME_WIDTH), "x",
              cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print("[YOLO THREAD] FPS:", cap.get(cv2.CAP_PROP_FPS))

        frame_count = 0
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    time.sleep(0.01)
                    continue

                if FLIP_LR:
                    frame = cv2.flip(frame, 1)

                annotated = frame
                try:
                    with torch.no_grad():
                        results = model(annotated, verbose=False)
                    res = results[0]

                    try:
                        if hasattr(res, "plot"):
                            annotated = res.plot()
                        else:
                            annotated = draw_from_boxes(res, model, annotated)
                    except Exception:
                        annotated = draw_from_boxes(res, model, annotated)

                    # GPU 메모리 부담 완화(매 30프레임마다)
                    frame_count += 1
                    if device == "cuda" and (frame_count % 30 == 0):
                        torch.cuda.empty_cache()

                except Exception as e:
                    annotated = frame.copy()
                    cv2.putText(annotated, f"Detection error: {type(e).__name__}",
                                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

                with frame_lock:
                    output_frame = annotated

                # 너무 빠른 루프 방지
                time.sleep(0.001)

        finally:
            try:
                cap.release()
            except Exception:
                pass
            # 카메라가 끊기면 재시도
            retry = 0

# -----------------------------
# 라우트
# -----------------------------
@app.route("/")
def index():
    return render_template("webcam.html")

@app.route("/webcam")
def webcam_page():
    return render_template("webcam.html")

@app.route("/video_feed")
def video_feed():
    return Response(gen_frames(), mimetype="multipart/x-mixed-replace; boundary=frame")

@app.route("/healthz")
def healthz():
    return {"status": "ok"}

# -----------------------------
# 실행
# -----------------------------
if __name__ == "__main__":
    # 백그라운드 YOLO 추론 스레드 시작 (서버 시작 시 1회만)
    threading.Thread(target=yolo_inference_loop, daemon=True).start()

    # reloader로 인한 이중 실행 방지(use_reloader=False)
    app.run(host="0.0.0.0", port=5000, debug=True, threaded=True, use_reloader=False)
