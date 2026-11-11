# Drawing VLA: Closed-Loop Visuomotor Action Chunking for Interactive Multi-Stroke Sketch Generation

> **Official repository for the paper:**  
> *"Drawing VLA: Closed-Loop Visuomotor Action Chunking for Interactive Multi-Stroke Sketch Generation from Crowdsourced Demonstrations"*  
> **Author:** Seungjoon Son  
>
> [![Paper PDF](https://img.shields.io/badge/Paper-PDF-red?logo=adobeacrobatreader)](./Drawing_VLA.pdf)
> [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](./LICENSE)

---

## 🎨 Overview & Visual Results

Drawing VLA explores closed-loop visuomotor policy learning for precision 2D sketch generation. By predicting 16-step action chunks from dual-channel visual observations (current canvas state and target template), the model generates smooth, multi-stroke sketches while maintaining robust closed-loop error correction.

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
├── collector/               # [프론트엔드] 단일 파일 Vanilla Web 수집기/게임
│   ├── index.html           # 메인 웹 앱 (외부 의존성 제로, 오프라인 동작 가능)
│   ├── robots.txt           # 검색 엔진 크롤러 설정
│   └── sitemap.xml          # 사이트맵
├── api/                     # [서버리스] Vercel Python Function 진입점
│   ├── save.py              # 시연 제출 및 채점 저장 핸들러 (/api/save)
│   ├── ticket.py            # 제출 티켓 발급 핸들러 (/api/ticket)
│   ├── leaderboard.py       # 순위표 조회 (/api/leaderboard)
│   ├── stats.py             # 통계 조회 (/api/stats)
│   ├── share.py             # 결과 공유 페이지 (/s)
│   ├── og.py                # 동적 OG 이미지 생성 (/api/og)
│   ├── event.py             # 클라이언트 이벤트 로깅 (/api/event)
│   ├── delete.py            # 기록 삭제 요청 (/api/delete)
│   └── cron.py              # 일일 정기 크론 작업 (/api/cron)
├── docs/                    # [문서] 제품 기획서, 비전 명세서, 평가 보고서
│   ├── README.md            # 문서 가이드 및 인덱스
│   ├── 차세대 라이브...md     # 차세대 라이브 드로잉 AI 제품 기획서 및 비전 명세서
│   ├── hand_vs_ai.md        # Hand vs AI 벤치마크 원본 기획서
│   └── reports/             # 실험 및 평가 보고서
│       └── ai_comparison_report.md
├── database/                # [데이터베이스] 스키마 및 마이그레이션
│   └── schema.sql           # Supabase 테이블, 뷰, RPC 함수 정의
├── tools/                   # [운영/도구] CLI 유틸리티 및 관리 스크립트
│   ├── README.md            # 도구별 사용법 및 CLI 가이드
│   ├── monitor.py           # 실시간 제출 모니터링 도구
│   ├── moderate.py          # 리더보드 상위/의심 기록 승인·숨김 관리
│   ├── backup.py            # 데이터베이스 백업 도구
│   ├── pull_dataset.py      # Supabase 검증 데이터 로컬 추출
│   ├── seed_supabase.py     # 로컬 데이터 Supabase 초기 등록
│   ├── rescore_records.py   # 기존 기록 재채점 및 점수 갱신
│   ├── make_link.py         # 개인 링크 발급 및 환경변수 설정
│   ├── vercel_local.py      # 로컬 Vercel 서버리스 시뮬레이터 (포트 8100)
│   └── eval/                # AI 후보 모델 롤아웃 비교/평가 스크립트
├── train/                   # [AI 모델링] Action Chunking 정책 학습 및 롤아웃
│   ├── dataset.py           # 텐서 변환 및 관측 렌더링 파이프라인
│   ├── policies.py          # MLP / Diffusion 정책 모델 아키텍처
│   ├── train_policy.py      # 모델 학습 스크립트 (XPU / CUDA / CPU)
│   ├── rollout_eval.py      # 폐루프 롤아웃 및 정밀도 평가 엔진
│   ├── brush.py             # 유기적 붓 스트로크 렌더러 (스플라인/필압)
│   └── check_demos.py       # 데이터셋 품질 검사 및 시각화
├── tests/                   # [테스트] 유닛 및 통합 테스트
│   ├── test_benchmark.py    # 핵심 통합 테스트 (채점, API, 보안, 어뷰징 방지)
│   ├── test_brush.py        # 붓 렌더러 테스트
│   ├── test_dataset.py      # 데이터 로더 및 관측 규격 테스트
│   ├── test_scoring_b.py    # 신규 B-v1 채점 알고리즘 테스트
│   └── test_*.py            # 기타 단위 테스트
├── data/                    # [데이터] 시연 궤적 데이터 (gitignore 대상)
│   ├── raw/                 # 검증 완료된 유효 시연 (demo_*.json)
│   ├── rejected/            # 결함 또는 부적합 시연
│   └── qa/                  # 시연 품질 검사용 이미지
├── outputs/                 # [산출물] 모델 체크포인트, 롤아웃 GIF/PNG, 캐시 (gitignore)
├── api_core.py              # [핵심 로직] API 백엔드, 채점 연동, 비즈니스 로직
├── collect_common.py        # [공용] 페이로드 규격, HMAC 서명 유틸
├── db.py                    # [스토리지] Supabase REST 및 SQLite 통합 어댑터
├── scoring.py               # [채점] 도안 매칭 채점 (F1, 커버리지, 정밀도, avg_error)
├── submission.py            # [보안] 제출 검증, 사람 손 판별, 치팅 방지
├── server.py                # [로컬 서버] 단독 로컬 수집 웹 서버 (포트 8000)
├── targets_128.json         # [기준 데이터] 128x128 도안별 표준 목표 이미지
├── vercel.json              # Vercel 배포 및 보안 헤더 설정
└── CLAUDE.md                # AI 페어 프로그래밍 작업 지침서 및 상세 규칙

---

## 🏛️ 아키텍처 및 루트 파일 규칙

루트에 위치한 핵심 파이썬 파일들(`api_core.py`, `db.py`, `scoring.py`, `submission.py`, `collect_common.py`)은 다음의 중요한 역할을 수행합니다:

1. **Vercel Serverless Function 번들링**:
   `vercel.json`에서 `collector/`를 정적 웹 루트(`outputDirectory`)로 지정하고 있으며, `api/*.py` 함수들은 루트 디렉토리를 참조하여 공통 백엔드 모듈을 직접 import합니다.
2. **로컬 개발과 클라우드 배포의 100% 동일한 로직 공유**:
   `server.py`(로컬 수집 서버), `tools/vercel_local.py`(로컬 시뮬레이터), 그리고 실제 Vercel 프로덕션 환경이 동일한 채점(`scoring.py`) 및 검증(`submission.py`) 코드를 공유합니다.

---

## 🚀 빠른 시작 (Quick Start)

### 1. 로컬 수집 서버 실행 (의존성 제로)
Python 표준 라이브러리만으로 로컬에서 태블릿/모바일 수집기를 구동할 수 있습니다:
```bash
python server.py
# 브라우저에서 http://localhost:8000 접속 (동일 Wi-Fi 내 태블릿 IP 주소 터미널에 출력)
```

### 2. Vercel 로컬 시뮬레이터 실행
Supabase 대신 로컬 SQLite를 사용하여 배포 API 전체 동작을 로컬에서 테스트합니다:
```bash
python tools/vercel_local.py
# 브라우저에서 http://127.0.0.1:8100 접속
```

### 3. 전체 테스트 스위트 실행
```bash
python -m unittest discover tests
```

### 4. 모델 학습 및 평가
```bash
# 1) 데이터셋 형태 검증
python train/dataset.py --resample-step 0.005

# 2) Action Chunking 정책 학습
python train/train_policy.py --tag v5b --resample-step 0.005 --epochs 60 --resume

# 3) 폐루프 롤아웃 평가 (붓 렌더링 및 Temporal Ensembling 포함)
python train/rollout_eval.py --ckpt outputs/apple_vla_policy_v5b.pth --exec-steps 12 --ensemble --brush
```
