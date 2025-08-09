# PPE_TEST

## 0. 참고사항
본 프로그램은 다음 시스템에서 실행되었습니다. 
```
OS Microsoft Windows 11 Pro
CPU AMD Ryzen 7 7800X3D 8-Core Processor
RAM 32.0GB
GPU NVIDIA GeForce RTX 4080 SUPER
Python 3.9.21
PyTorch 2.7.0+cu128
Ultralytics 8.3.140
OpenCV 4.9.0
CUDA 12.8
cuDNN 90701 (9.7.1)
```

## 1. Anaconda 설치
 [Anaconda 공식 사이트](https://www.anaconda.com/products/distribution)

## 2. CUDA 설치
[CUDA Toolkit 12.8 Downloads](https://developer.nvidia.com/cuda-12-8-0-download-archive)

## 3. cuDNN 설치
[cuDNN 9.7.1 Downloads](https://developer.nvidia.com/cudnn-9-7-1-download-archive)

압축을 해제한 뒤 CUDA 설치 경로에 있는 `include`와 `lib` 폴더에 파일을 복사합니다.

## 4. 가상환경 구성
```
conda create -n ppe_env python=3.10
conda activate ppe_env
```
## 5. 의존성 패키지 설치
```
pip install -r requirements.txt
```

## 6. 실행
```
git clone https://github.com/pjs6203/ppe_test.git
cd ppe_test
python app.py
```

## License

This project is licensed under the **AGPL-3.0**.  
See the [LICENSE](./LICENSE) file for details.

### Third-Party Notices
- **Ultralytics YOLO** — AGPL-3.0  
  Used as the inference/training framework.
- **Start Bootstrap – SB Admin** — MIT  
  Template assets included/modified; MIT notice retained.
- **PPE_detection_YOLO (Vinayakmane47)** — MIT  
  Referenced for structure/implementation ideas; MIT notice retained as applicable.
- **Datasets** — *(if used)*  
  - Example: Construction Site Safety (Roboflow Universe) — **CC BY 4.0** (include author & link)

> If you distribute or provide this project over a network, ensure compliance with **AGPL-3.0**,
> including making the complete corresponding source code available to users.
 