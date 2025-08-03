"""
Flask + OpenCV + Ultralytics YOLO (웹캠 실시간 스트리밍)

- 백그라운드 스레드에서만 모델 실행 (웹은 화면만 구독)
- Windows 카메라 백엔드: DSHOW → MSMF → ANY 폴백
- Flask reloader로 인한 이중 실행 방지(use_reloader=False)
- 해상도/현재 FPS/프레임 인덱스를 /api/stream_stats 로 제공
"""

import os
import time
import threading
from typing import Optional
from collections import deque

import cv2
import numpy as np
from flask import Flask, Response, render_template, jsonify
from PIL import Image  # (업로드 기능은 안 쓰지만 PIL 설치 유도 겸)
import torch
from ultralytics import YOLO
import platform, psutil, subprocess

# -----------------------------
# 기본 설정 (필요시 여기만 수정)
# -----------------------------
MODEL_PATH = "./weights/ppe.pt"
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
# 글로벌 상태
# -----------------------------
output_frame: Optional[np.ndarray] = None
frame_lock = threading.Lock()

# 스트림 통계(웹에서 표시)
stream_stats = {
    "width": None,        # int
    "height": None,       # int
    "fps": 0.0,           # float
    "frame_index": 0      # int
}
stats_lock = threading.Lock()

# 선택: NVML을 쓰면 GPU util%까지 안정적으로 나옵니다.
try:
    import pynvml as nvml
    _NVML = True
except Exception:
    _NVML = False


def get_system_versions():
    py = platform.python_version()
    torch_v = getattr(torch, "__version__", None)
    cuda_v = getattr(torch.version, "cuda", None)
    cudnn_v = None
    try:
        cudnn_v = torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else None
    except Exception:
        cudnn_v = None

    ul_v = None
    try:
        from ultralytics import __version__ as ul_ver
        ul_v = ul_ver
    except Exception:
        pass

    cv_v = None
    try:
        import cv2
        cv_v = cv2.__version__
    except Exception:
        pass

    device_name = "cuda" if torch.cuda.is_available() else "cpu"
    gpus = []
    if device_name == "cuda":
        try:
            for i in range(torch.cuda.device_count()):
                gpus.append(torch.cuda.get_device_name(i))
        except Exception:
            pass

    return {
        "python": py,
        "torch": torch_v,
        "cuda": cuda_v,
        "cudnn": cudnn_v,
        "ultralytics": ul_v,
        "opencv": cv_v,
        "device": device_name,
        "gpus": gpus,
    }

def _try_nvml_usage():
    if not _NVML:
        return None
    try:
        nvml.nvmlInit()
    except Exception:
        return None

    try:
        g_list = []
        count = nvml.nvmlDeviceGetCount()
        for i in range(count):
            h = nvml.nvmlDeviceGetHandleByIndex(i)
            try:
                name = nvml.nvmlDeviceGetName(h).decode()
            except Exception:
                name = "GPU"

            mem = nvml.nvmlDeviceGetMemoryInfo(h)

            try:
                util = nvml.nvmlDeviceGetUtilizationRates(h)  # .gpu, .memory
                sm_util = int(util.gpu)
                mem_util = int(util.memory)
            except Exception:
                sm_util = None
                mem_util = None

            try:
                enc_util, _ = nvml.nvmlDeviceGetEncoderUtilization(h)
            except Exception:
                enc_util = None
            try:
                dec_util, _ = nvml.nvmlDeviceGetDecoderUtilization(h)
            except Exception:
                dec_util = None

            try:
                tx_kbs = nvml.nvmlDeviceGetPcieThroughput(h, nvml.NVML_PCIE_UTIL_TX_BYTES)
                rx_kbs = nvml.nvmlDeviceGetPcieThroughput(h, nvml.NVML_PCIE_UTIL_RX_BYTES)
            except Exception:
                tx_kbs = None
                rx_kbs = None

            g_list.append({
                "index": i,
                "name": name,
                "vram_total": int(mem.total),
                "vram_used": int(mem.used),
                "sm_util": sm_util,
                "mem_util": mem_util,
                "enc_util": enc_util,
                "dec_util": dec_util,
                "pcie_tx_kbs": tx_kbs,
                "pcie_rx_kbs": rx_kbs,
            })
        nvml.nvmlShutdown()
        return g_list
    except Exception:
        try:
            nvml.nvmlShutdown()
        except Exception:
            pass
        return None

def _try_nvidia_smi_usage():
    try:
        out = subprocess.check_output(
            ["nvidia-smi",
             "--query-gpu=name,memory.total,memory.used,utilization.gpu,utilization.memory",
             "--format=csv,noheader,nounits"],
            stderr=subprocess.DEVNULL, text=True
        ).strip()
        g_list = []
        for idx, line in enumerate(out.splitlines()):
            name, total, used, sm, memc = [x.strip() for x in line.split(",")]
            g_list.append({
                "index": idx,
                "name": name,
                "vram_total": int(float(total)) * 1024**2,  # MB -> bytes
                "vram_used":  int(float(used))  * 1024**2,  # MB -> bytes
                "sm_util":    int(float(sm)),
                "mem_util":   int(float(memc)),
                "enc_util":   None,
                "dec_util":   None,
                "pcie_tx_kbs": None,
                "pcie_rx_kbs": None,
            })
        return g_list if g_list else None
    except Exception:
        return None

def _torch_mem_usage_fallback():
    if not torch.cuda.is_available():
        return None
    g_list = []
    try:
        for i in range(torch.cuda.device_count()):
            torch.cuda.set_device(i)
            total = torch.cuda.get_device_properties(i).total_memory
            try:
                free, total2 = torch.cuda.mem_get_info(i)
                used = total2 - free
                total = total2
            except Exception:
                used = torch.cuda.memory_reserved(i)
            g_list.append({
                "index": i,
                "name": torch.cuda.get_device_name(i),
                "vram_total": int(total),
                "vram_used": int(used),
                "util": None,
            })
        return g_list
    except Exception:
        return None

def get_system_usage():
    cpu = psutil.cpu_percent(interval=0.2)
    mem = psutil.virtual_memory()
    usage = {
        "cpu_percent": cpu,
        "mem_total": int(mem.total),
        "mem_used": int(mem.used),
        "mem_percent": float(mem.percent),
        "gpus": [],
    }
    g = _try_nvml_usage() or _try_nvidia_smi_usage() or _torch_mem_usage_fallback()
    if g:
        for gpu in g:
            total = gpu["vram_total"]
            used = gpu["vram_used"]
            percent = round(used / total * 100, 1) if total else None
            usage["gpus"].append({
                **gpu,
                "vram_percent": percent,
            })
    return usage

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
                cap.set(cv2.CAP_PROP_FRAME_WIDTH,  TARGET_W)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, TARGET_H)
                cap.set(cv2.CAP_PROP_FPS,          TARGET_FPS)
                try:
                    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
                except Exception:
                    pass
                return cap
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
        time.sleep(0.01)  # 과부하 방지

# -----------------------------
# 백그라운드 YOLO 추론 루프
# -----------------------------
def yolo_inference_loop():
    """
    - 카메라 프레임 수집
    - YOLO 추론 및 결과 렌더링
    - 최신 프레임을 output_frame에 보관
    - 해상도/현재 FPS/프레임 인덱스 갱신
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
        # FPS 계산용
        fps_counter = 0
        t0 = time.time()

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    time.sleep(0.01)
                    continue

                if FLIP_LR:
                    frame = cv2.flip(frame, 1)

                # 통계 업데이트(해상도/프레임 인덱스)
                h, w = frame.shape[:2]
                with stats_lock:
                    stream_stats["width"] = w
                    stream_stats["height"] = h
                    stream_stats["frame_index"] += 1

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

                # 공유 프레임 교체
                with frame_lock:
                    output_frame = annotated

                # FPS 계산 (1초 창)
                fps_counter += 1
                now = time.time()
                if now - t0 >= 1.0:
                    fps = fps_counter / (now - t0)
                    with stats_lock:
                        stream_stats["fps"] = fps
                    fps_counter = 0
                    t0 = now

                time.sleep(0.001)  # 너무 빠른 루프 방지

        finally:
            try:
                cap.release()
            except Exception:
                pass
            retry = 0  # 카메라가 끊기면 재시도

# ----------------------------
# 라우트
# ----------------------------
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

@app.route("/about")
def about():
    versions = get_system_versions()
    return render_template("about.html", versions=versions)

@app.route("/api/system_metrics")
def api_system_metrics():
    return jsonify(get_system_usage())

@app.route("/api/stream_stats")
def api_stream_stats():
    # 해상도 문자열까지 만들어서 내려줌
    with stats_lock:
        w = stream_stats["width"]
        h = stream_stats["height"]
        fps = stream_stats["fps"]
        idx = stream_stats["frame_index"]
    res_str = None if (w is None or h is None) else f"{w}×{h}"
    return jsonify({
        "width": w,
        "height": h,
        "resolution": res_str,
        "fps": round(float(fps), 1) if fps else 0.0,
        "frame_index": int(idx),
    })

# -----------------------------
# 실행
# -----------------------------
if __name__ == "__main__":
    # 백그라운드 YOLO 추론 스레드 시작 (서버 시작 시 1회만)
    threading.Thread(target=yolo_inference_loop, daemon=True).start()
    app.run(host="0.0.0.0", port=5000, debug=True, threaded=True, use_reloader=False)
