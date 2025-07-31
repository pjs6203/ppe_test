# gen_env_and_requirements.py
import sys, platform
import pkg_resources as pr

print("== Python/OS ==")
print("python :", sys.version.split()[0])
print("platform:", platform.platform())

# ---- 점검: torch / CUDA / cuDNN ----
try:
    import torch
    print("\n== PyTorch ==")
    print("torch :", torch.__version__)
    print("wheel CUDA:", torch.version.cuda)
    print("cuda.is_available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("device:", torch.cuda.get_device_name(0))
    print("cuDNN available:", torch.backends.cudnn.is_available())
    print("cuDNN version  :", torch.backends.cudnn.version())
except Exception as e:
    print("\n[torch 오류]", e)

# ---- 점검: opencv / ultralytics / numpy ----
try:
    import cv2; print("\nopencv-python:", cv2.__version__)
except Exception as e:
    print("\n[opencv 오류]", e)

try:
    import ultralytics as ul; print("ultralytics :", ul.__version__)
except Exception as e:
    print("[ultralytics 오류]", e)

try:
    import numpy as np; print("numpy       :", np.__version__)
except Exception as e:
    print("[numpy 오류]", e)

# ---- requirements.txt 생성 (설치된 정확한 버전 고정) ----
mods = ["flask","ultralytics","torch","opencv-python","pillow","numpy"]
pins=[]
for m in mods:
    try:
        v = pr.get_distribution(m).version
        pins.append(f"{m}=={v}")
    except Exception:
        # 설치 안 된 모듈은 건너뜀
        pass

with open("requirements.txt","w",encoding="utf-8") as f:
    f.write("\n".join(pins))

print("\nrequirements.txt 생성 완료:")
print("\n".join(pins))
