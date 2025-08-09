"""
Flask + OpenCV + Ultralytics YOLO (웹캠 실시간 스트리밍)

추가:
- ./weights/*.pt 목록을 제공(/api/models)하고, 선택한 모델로 런타임 교체(/api/select_model)
- 선택 상태/로드 완료 여부를 /api/models 응답에 함께 포함
- NO-* 라벨을 고려한 PPE 상태 계산
"""

import os
import time
import threading
from typing import Optional, Dict, List
from collections import Counter

import cv2
import numpy as np
from flask import Flask, Response, render_template, jsonify, request, send_file
from PIL import Image
import torch
from ultralytics import YOLO
import platform, psutil, subprocess
from io import BytesIO
from glob import glob
import traceback

# -----------------------------
# 기본 설정
# -----------------------------
WEIGHTS_DIR = "./weights"
DEFAULT_MODEL = os.getenv("MODEL_PATH", "").strip() or None  # 우선순위: 환경변수 > 첫 번째 pt
CAM_INDEX = int(os.getenv("CAM_INDEX", "0"))
TARGET_W, TARGET_H = 1280, 720
TARGET_FPS = 60
FLIP_LR = True
INFER_EVERY = int(os.getenv("INFER_EVERY", "1"))
REQUIRED_PPE = [s.strip().lower() for s in os.getenv("REQUIRED_PPE", "helmet,vest,mask").split(",") if s.strip()]

os.environ.setdefault("TORCH_LOAD_WEIGHTS_ONLY", "0")

app = Flask(__name__, template_folder="templates", static_folder="static")
device = "cuda" if torch.cuda.is_available() else "cpu"

# -----------------------------
# 모델 관리 (스레드 안전)
# -----------------------------
model_lock = threading.Lock()
model: Optional[YOLO] = None
requested_model_path: Optional[str] = None   # 사용자가 선택한 경로
loaded_model_path: Optional[str] = None      # 현재 메모리에 올라간 경로
reload_event = threading.Event()
last_model_error: Optional[str] = None

def list_weight_files() -> List[str]:
    files = sorted(glob(os.path.join(WEIGHTS_DIR, "*.pt")))
    return files

def _load_model(path: str) -> YOLO:
    m = YOLO(path).to(device)
    return m

def ensure_default_model_selected():
    global requested_model_path
    if requested_model_path is None:
        candidates = list_weight_files()
        if DEFAULT_MODEL and os.path.isfile(DEFAULT_MODEL):
            requested_model_path = DEFAULT_MODEL
        elif candidates:
            requested_model_path = candidates[0]

# -----------------------------
# 글로벌 상태 (스트림/객체/PPE/시스템)
# -----------------------------
output_frame: Optional[np.ndarray] = None
frame_lock = threading.Lock()

stream_stats = {"width": None, "height": None, "fps": 0.0, "frame_index": 0}
stats_lock = threading.Lock()

last_objects: Dict[str, int] = {}
last_objects_lock = threading.Lock()

# NVML (옵션)
try:
    import pynvml as nvml
    _NVML = True
except Exception:
    _NVML = False

def get_system_versions():
    py = platform.python_version()
    torch_v = getattr(torch, "__version__", None)
    cuda_v = getattr(torch.version, "cuda", None)
    try:
        cudnn_v = torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else None
    except Exception:
        cudnn_v = None
    try:
        from ultralytics import __version__ as ul_ver
        ul_v = ul_ver
    except Exception:
        ul_v = None
    try:
        import cv2
        cv_v = cv2.__version__
    except Exception:
        cv_v = None

    device_name = "cuda" if torch.cuda.is_available() else "cpu"
    gpus = []
    if device_name == "cuda":
        try:
            for i in range(torch.cuda.device_count()):
                gpus.append(torch.cuda.get_device_name(i))
        except Exception:
            pass

    return {
        "python": py, "torch": torch_v, "cuda": cuda_v, "cudnn": cudnn_v,
        "ultralytics": ul_v, "opencv": cv_v, "device": device_name, "gpus": gpus,
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
                util = nvml.nvmlDeviceGetUtilizationRates(h)
                sm_util = int(util.gpu); mem_util = int(util.memory)
            except Exception:
                sm_util = None; mem_util = None
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
                tx_kbs = None; rx_kbs = None
            g_list.append({
                "index": i, "name": name,
                "vram_total": int(mem.total), "vram_used": int(mem.used),
                "sm_util": sm_util, "mem_util": mem_util,
                "enc_util": enc_util, "dec_util": dec_util,
                "pcie_tx_kbs": tx_kbs, "pcie_rx_kbs": rx_kbs,
            })
        nvml.nvmlShutdown()
        return g_list
    except Exception:
        try: nvml.nvmlShutdown()
        except Exception: pass
        return None

def _try_nvidia_smi_usage():
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name,memory.total,memory.used,utilization.gpu,utilization.memory",
             "--format=csv,noheader,nounits"],
            stderr=subprocess.DEVNULL, text=True
        ).strip()
        g_list = []
        for idx, line in enumerate(out.splitlines()):
            name, total, used, sm, memc = [x.strip() for x in line.split(",")]
            g_list.append({
                "index": idx, "name": name,
                "vram_total": int(float(total)) * 1024**2,
                "vram_used":  int(float(used))  * 1024**2,
                "sm_util":    int(float(sm)),
                "mem_util":   int(float(memc)),
                "enc_util":   None, "dec_util": None,
                "pcie_tx_kbs": None, "pcie_rx_kbs": None,
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
                used = total2 - free; total = total2
            except Exception:
                used = torch.cuda.memory_reserved(i)
            g_list.append({
                "index": i, "name": torch.cuda.get_device_name(i),
                "vram_total": int(total), "vram_used": int(used), "util": None,
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
            total = gpu["vram_total"]; used = gpu["vram_used"]
            percent = round(used / total * 100, 1) if total else None
            usage["gpus"].append({**gpu, "vram_percent": percent})
    return usage

# -----------------------------
# 결과/라벨 도우미 & PPE
# -----------------------------
def draw_from_boxes(res, model, canvas: np.ndarray) -> np.ndarray:
    out = canvas.copy()
    boxes_obj = getattr(res, "boxes", None)
    if boxes_obj is None:
        return out
    def _to_list(t):
        if t is None: return None
        try: return t.detach().cpu().float().tolist()
        except Exception: return None
    xyxy_list = _to_list(getattr(boxes_obj, "xyxy", None))
    conf_list = _to_list(getattr(boxes_obj, "conf", None))
    cls_list  = _to_list(getattr(boxes_obj, "cls", None))
    if not xyxy_list: return out
    names = getattr(model, "names", {})
    if isinstance(names, dict):
        def name_of(i): i=int(i); return names.get(i, str(i))
    else:
        def name_of(i):
            i=int(i)
            if isinstance(names, (list,tuple)) and 0<=i<len(names): return names[i]
            return str(i)
    for i, box in enumerate(xyxy_list):
        x1, y1, x2, y2 = map(int, box)
        label = ""
        if cls_list is not None and i < len(cls_list): label = name_of(cls_list[i])
        if conf_list is not None and i < len(conf_list): label = f"{label} {conf_list[i]:.2f}" if label else f"{conf_list[i]:.2f}"
        cv2.rectangle(out, (x1, y1), (x2, y2), (0,255,0), 2)
        if label:
            y_text = max(y1-5, 15)
            cv2.putText(out, label, (x1, y_text), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)
    return out

def count_objects(res, model) -> Dict[str, int]:
    boxes_obj = getattr(res, "boxes", None)
    if boxes_obj is None: return {}
    try:
        cls = boxes_obj.cls.detach().cpu().tolist()
    except Exception:
        return {}
    names = getattr(model, "names", {})
    def name_of(i):
        if isinstance(names, dict): return names.get(int(i), str(int(i)))
        if isinstance(names, (list, tuple)) and 0 <= int(i) < len(names): return names[int(i)]
        return str(int(i))
    labels = [name_of(i) for i in cls]
    return dict(Counter(labels))

PPE_KEYS = {
    "person": ["person", "worker", "human"],
    "helmet": ["helmet", "hardhat"],
    "vest":   ["vest", "safety vest", "reflective vest"],
    "shoes":  ["boots", "safety boots", "shoe", "shoes"],
    "mask":   ["mask", "face mask"],
    "glasses":["glasses", "goggles", "safety glasses"],
}
PPE_NEG_KEYS = {
    "helmet": ["no-helmet", "no helmet", "no-hardhat", "no hardhat", "without helmet"],
    "vest":   ["no-vest", "no vest", "no-safety vest", "no safety vest", "without vest"],
    "shoes":  ["no-shoes", "no shoes", "no-boots", "no boots", "without shoes"],
    "mask":   ["no-mask", "no mask", "without mask"],
    "glasses":["no-glasses", "no glasses", "no-goggles", "without goggles"],
}

def _match_count(objects: Dict[str,int], keys: list) -> int:
    total = 0
    for k, v in objects.items():
        lk = k.lower().replace("—","-").replace("–","-").replace("_"," ")
        for t in keys:
            if t in lk:
                total += v; break
    return total

def derive_ppe_status(objects: Dict[str,int]) -> Dict:
    has_person = _match_count(objects, PPE_KEYS["person"]) > 0
    present = {}; missing_required = []
    for k in ["helmet","vest","shoes","mask","glasses"]:
        pos = _match_count(objects, PPE_KEYS.get(k, []))
        neg = _match_count(objects, PPE_NEG_KEYS.get(k, []))
        present[k] = (pos > 0) and (neg == 0)
        if has_person and (k in REQUIRED_PPE) and (not present[k]):
            missing_required.append(k)
    if not has_person: code="no_person"; text="NO PERSON"; level=0
    elif missing_required: code="violation"; text="MISSING: " + ", ".join(missing_required).upper(); level=2
    else: code="safe"; text="SAFE"; level=3
    return {"has_person":has_person,"present":present,"required":REQUIRED_PPE,
            "missing_required":missing_required,"status_code":code,"status_text":text,
            "level":level,"objects":objects}

# -----------------------------
# 카메라/스트림
# -----------------------------
def open_camera(index: int) -> cv2.VideoCapture:
    if os.name == "nt":
        for api in (cv2.CAP_DSHOW, cv2.CAP_MSMF, cv2.CAP_ANY):
            cap = cv2.VideoCapture(index, api)
            if cap.isOpened():
                cap.set(cv2.CAP_PROP_FRAME_WIDTH,  TARGET_W)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, TARGET_H)
                cap.set(cv2.CAP_PROP_FPS,          TARGET_FPS)
                try: cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
                except Exception: pass
                return cap
        return cv2.VideoCapture()
    else:
        cap = cv2.VideoCapture(index)
        if cap.isOpened():
            cap.set(cv2.CAP_PROP_FRAME_WIDTH,  TARGET_W)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, TARGET_H)
            cap.set(cv2.CAP_PROP_FPS,          TARGET_FPS)
        return cap

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
                yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + buf.tobytes() + b"\r\n")
            time.sleep(0.05); continue
        ok, buf = cv2.imencode(".jpg", frame)
        if ok:
            yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + buf.tobytes() + b"\r\n")
        time.sleep(0.01)

# -----------------------------
# 추론 스레드
# -----------------------------
def yolo_inference_loop():
    global output_frame, model, loaded_model_path, last_model_error

    ensure_default_model_selected()
    # 최초 로드
    if requested_model_path:
        try:
            with model_lock:
                model = _load_model(requested_model_path)
                loaded_model_path = requested_model_path
                last_model_error = None
        except Exception as e:
            last_model_error = f"Initial model load failed: {e}"
            traceback.print_exc()

    retry = 0
    while True:
        cap = open_camera(CAM_INDEX)
        if not cap.isOpened():
            retry += 1
            wait = min(5.0, 0.5 * retry)
            print(f"[YOLO THREAD] Camera open failed. retry={retry} wait={wait:.1f}s")
            time.sleep(wait)
            continue

        print("[YOLO THREAD] RES:", cap.get(cv2.CAP_PROP_FRAME_WIDTH), "x", cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print("[YOLO THREAD] FPS:", cap.get(cv2.CAP_PROP_FPS))

        frame_count = 0
        fps_counter = 0
        t0 = time.time()

        try:
            while True:
                # 모델 교체 요청 처리
                if reload_event.is_set():
                    try:
                        with model_lock:
                            if requested_model_path and requested_model_path != loaded_model_path:
                                print(f"[YOLO THREAD] Loading model: {requested_model_path}")
                                new_model = _load_model(requested_model_path)
                                model = new_model
                                loaded_model_path = requested_model_path
                                last_model_error = None
                                print(f"[YOLO THREAD] Model loaded.")
                    except Exception as e:
                        last_model_error = f"Model load failed: {e}"
                        traceback.print_exc()
                    finally:
                        reload_event.clear()

                ret, frame = cap.read()
                if not ret:
                    time.sleep(0.01); continue
                if FLIP_LR: frame = cv2.flip(frame, 1)

                h, w = frame.shape[:2]
                with stats_lock:
                    stream_stats["width"] = w
                    stream_stats["height"] = h
                    stream_stats["frame_index"] += 1

                annotated = frame
                do_infer = (INFER_EVERY <= 1) or (stream_stats["frame_index"] % INFER_EVERY == 0)

                try:
                    # 모델이 없으면 그냥 패스(오류 표시는 프레임에 텍스트)
                    if do_infer and model is not None:
                        with model_lock:
                            with torch.no_grad():
                                results = model(annotated, verbose=False)
                        res = results[0]
                        with last_objects_lock:
                            global last_objects
                            last_objects = count_objects(res, model)
                        try:
                            if hasattr(res, "plot"):
                                annotated = res.plot()
                            else:
                                annotated = draw_from_boxes(res, model, annotated)
                        except Exception:
                            annotated = draw_from_boxes(res, model, annotated)
                        if device == "cuda" and ((frame_count % 30) == 0):
                            torch.cuda.empty_cache()
                    elif model is None:
                        annotated = frame.copy()
                        cv2.putText(annotated, "Model not loaded", (10, 30),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

                    frame_count += 1

                except Exception as e:
                    annotated = frame.copy()
                    cv2.putText(annotated, f"Detection error: {type(e).__name__}",
                                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

                with frame_lock:
                    output_frame = annotated

                fps_counter += 1
                now = time.time()
                if now - t0 >= 1.0:
                    fps = fps_counter / (now - t0)
                    with stats_lock:
                        stream_stats["fps"] = fps
                    fps_counter = 0; t0 = now

                time.sleep(0.001)

        finally:
            try: cap.release()
            except Exception: pass
            retry = 0

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

@app.route("/snapshot.jpg")
def snapshot():
    with frame_lock:
        if output_frame is None:
            return ("No frame", 503)
        rgb = cv2.cvtColor(output_frame, cv2.COLOR_BGR2RGB)
        pil = Image.fromarray(rgb)
        buf = BytesIO(); pil.save(buf, format="JPEG", quality=90); buf.seek(0)
        return send_file(buf, mimetype="image/jpeg")

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
    with stats_lock:
        w = stream_stats["width"]; h = stream_stats["height"]
        fps = stream_stats["fps"]; idx = stream_stats["frame_index"]
    res_str = None if (w is None or h is None) else f"{w}×{h}"
    return jsonify({
        "width": w, "height": h, "resolution": res_str,
        "fps": round(float(fps), 1) if fps else 0.0,
        "frame_index": int(idx),
    })

@app.route("/api/objects")
def api_objects():
    with last_objects_lock:
        return jsonify(last_objects)

@app.route("/api/ppe_status")
def api_ppe_status():
    with last_objects_lock:
        objs = dict(last_objects)
    status = derive_ppe_status(objs)
    return jsonify(status)

# ---- 모델 선택 API ----
@app.route("/api/models")
def api_models():
    ensure_default_model_selected()
    files = list_weight_files()
    return jsonify({
        "files": files,
        "requested": requested_model_path,
        "loaded": loaded_model_path,
        "error": last_model_error,
        "device": device,
    })

@app.route("/api/select_model", methods=["POST"])
def api_select_model():
    global requested_model_path
    data = request.get_json(silent=True) or {}
    path = data.get("path")
    if not path:
        return jsonify({"ok": False, "error": "path missing"}), 400
    if not os.path.isabs(path):
        path = os.path.abspath(path)
    if not os.path.isfile(path):
        return jsonify({"ok": False, "error": "file not found"}), 404
    # weights 디렉토리 내부만 허용
    wd = os.path.abspath(WEIGHTS_DIR)
    if not os.path.abspath(path).startswith(wd):
        return jsonify({"ok": False, "error": "path must be under weights/"}), 400

    requested_model_path = path
    reload_event.set()
    return jsonify({"ok": True, "requested": requested_model_path})

# -----------------------------
# 실행
# -----------------------------
if __name__ == "__main__":
    threading.Thread(target=yolo_inference_loop, daemon=True).start()
    app.run(host="0.0.0.0", port=5000, debug=True, threaded=True, use_reloader=False)
