# Drawing VLA: Closed-Loop Visuomotor Action Chunking for Interactive Multi-Stroke Sketch Generation

> **Official repository for the paper:**  
> *"Drawing VLA: Closed-Loop Visuomotor Action Chunking for Interactive Multi-Stroke Sketch Generation from Crowdsourced Demonstrations"*  
> **Author:** Seungjoon Son  
>
> [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.24122154.svg)](https://doi.org/10.5281/zenodo.24122154)
> [![Live Benchmark](https://img.shields.io/badge/Live%20Benchmark-Hand%20vs%20AI-success?logo=vercel)](https://handvsai.vercel.app)
> [![Paper PDF](https://img.shields.io/badge/Paper-PDF-red?logo=adobeacrobatreader)](./Drawing_VLA.pdf)
> [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](./LICENSE)
>
> 🎮 **실시간 웹 벤치마크 플레이:** [https://handvsai.vercel.app](https://handvsai.vercel.app)  
> 📄 **논문 원문 (PDF):** [Drawing_VLA.pdf](./Drawing_VLA.pdf)

---

## 🎨 Overview & Visual Results

Drawing VLA explores closed-loop visuomotor policy learning for precision 2D sketch generation. By predicting 16-step action chunks from dual-channel visual observations (current canvas state and target template), the model generates smooth, multi-stroke sketches while maintaining robust closed-loop error correction.

### 🎥 실시간 자율 드로잉 롤아웃 (Live Closed-Loop Rollouts)

| 가이드 선따기: 사과 (3획) | 다중 스트로크 분리: 새싹 (4획) |
| :---: | :---: |
| <img src="./paper/figures/rollout_apple.gif" width="360" alt="Drawing-VLA 사과 롤아웃" /> | <img src="./paper/figures/rollout_sprout.gif" width="360" alt="Drawing-VLA 새싹 롤아웃" /> |

*Drawing-VLA 통합 정책 모델($K=16$ 액션 청킹, Temporal Ensembling, Catmull-Rom 스플라인 브러시 렌더러)이 캔버스 시각 피드백을 실시간으로 확인하며 자율 드로잉을 수행하는 실제 롤아웃 영상.*

### 1. Rollout Comparison (Action Chunking vs. Single-Step Autoregressive)
![Figure 1: Rollout Comparison](./paper/figures/fig1_rollout_comparison.png)
*Figure 1: Comparison between single-step autoregressive rollout and closed-loop Action Chunking. Action Chunking prevents error accumulation and generates clean stroke separation across multi-stroke drawing targets.*

### 2. Real-Time Disturbance Recovery (+20px Perturbation)
![Figure 4: Perturbation Recovery](./paper/figures/fig4_perturbation_recovery.png)
*Figure 4: Closed-loop visual feedback enables real-time trajectory recovery when perturbed by external shifts (+20px disturbance).*

---

## 📁 프로젝트 폴더 및 파일 구조

```
drawing-vla/
├── train/                   # [AI 모델링] Action Chunking 정책 학습 및 롤아웃
│   ├── dataset.py           # 텐서 변환 및 관측 렌더링 파이프라인
│   ├── policies.py          # ACT / Convolutional 정책 모델 아키텍처
│   ├── train_policy.py      # 모델 학습 스크립트 (CUDA / Intel XPU / CPU)
│   ├── rollout_eval.py      # 폐루프 롤아웃 및 정밀도 평가 엔진
│   ├── brush.py             # 유기적 붓 스트로크 렌더러 (스플라인/필압)
│   └── check_demos.py       # 데이터셋 품질 검사 및 시각화
├── tests/                   # [테스트] 유닛 및 폐루프 평가 테스트
│   ├── test_brush.py        # 붓 렌더러 테스트
│   ├── test_dataset.py      # 데이터 로더 및 관측 규격 테스트
│   ├── test_scoring_b.py    # 신규 B-v1 채점 알고리즘 테스트
│   ├── test_rollout.py      # 폐루프 롤아웃 및 복구 테스트
│   └── test_training.py     # 정책 학습 손실 함수 테스트
├── paper/                   # [논문] LaTeX 소스, 벤치마크 결과 도표 및 수치
├── scoring.py               # [채점] 도안 매칭 채점 (F1, 커버리지, 정밀도, avg_error)
├── templates.json           # [도안] 표준 벤치마크 12종 벡터 패스 좌표
├── targets_128.json         # [기준 데이터] 128x128 도안별 표준 목표 이미지
└── Drawing_VLA.pdf          # 논문 정식 PDF
```

---

## 🚀 빠른 시작 (Quick Start)

### 1. 환경 설정
```bash
git clone https://github.com/5Color/drawing-vla.git
cd drawing-vla
pip install -r requirements.txt
```

### 2. 라이브 벤치마크 웹 플랫폼
실시간 웹 캔버스에서 자신의 드로잉을 AI 모델과 직접 겨뤄보고 벤치마크를 체험할 수 있습니다:
- **공식 웹 플랫폼:** [https://handvsai.vercel.app](https://handvsai.vercel.app) *(태블릿 펜, 터치, 마우스 완벽 지원)*

### 3. 테스트 스위트 실행
```bash
python -m unittest tests.test_scoring_b tests.test_brush tests.test_dataset tests.test_rollout tests.test_training
```

### 4. 모델 가중치 다운로드 및 롤아웃 평가
학습된 모델 체크포인트는 [GitHub Releases v1.0.0](https://github.com/5Color/drawing-vla/releases/tag/v1.0.0)에 공개되어 있습니다:
```bash
# 1) 사전 학습 가중치 다운로드
mkdir -p outputs
curl -L -o outputs/apple_vla_policy.pth https://github.com/5Color/drawing-vla/releases/download/v1.0.0/apple_vla_policy.pth

# 2) 폐루프 롤아웃 평가 (붓 렌더링 및 Temporal Ensembling 포함)
python train/rollout_eval.py --ckpt outputs/apple_vla_policy.pth --exec-steps 12 --ensemble --brush

# 3) (선택) 데이터셋 검증 및 직접 재학습
python train/dataset.py --resample-step 0.005
python train/train_policy.py --head transformer --epochs 60 --goal-drop 0.5
```
