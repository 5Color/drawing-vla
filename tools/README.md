# Tools Directory Guide

`tools/` 디렉토리는 데이터 수집, 데이터베이스 운영, 배포 관리, 모델 평가에 사용되는 CLI 스크립트 모음입니다.

---

## 1. 운영 및 모니터링 (Ops & Moderation)

| 스크립트 | 용도 | 사용 예시 |
|---|---|---|
| `monitor.py` | 실시간 시연 제출 및 리더보드 통계 모니터링 (새 기록, 통과율, 이상 감지) | `python tools/monitor.py` |
| `moderate.py` | 리더보드 상위/의심 시연 조회 및 승인/숨김 처리 | `python tools/moderate.py --top 10`<br>`python tools/moderate.py --hide <id>` |
| `backup.py` | Supabase / SQLite DB의 시연 및 리더보드 테이블 백업 | `python tools/backup.py` |

---

## 2. 데이터셋 및 채점 파이프라인 (Dataset & Scoring)

| 스크립트 | 용도 | 사용 예시 |
|---|---|---|
| `pull_dataset.py` | Supabase의 검증된 시연을 로컬 `data/raw/`로 다운로드 및 정합성 검사 | `python tools/pull_dataset.py --trusted-only` |
| `seed_supabase.py` | 로컬 `data/raw/`의 시연들을 Supabase 원격 DB로 초기 등록 | `python tools/seed_supabase.py --dry-run` |
| `rescore_records.py` | 채점 규격 변경 시 DB 기존 기록들을 재채점하고 통계 갱신 | `python tools/rescore_records.py`<br>`python tools/rescore_records.py --apply` |

---

## 3. 인증 및 로컬 환경 (Auth & Local Dev)

| 스크립트 | 용도 | 사용 예시 |
|---|---|---|
| `collect_secrets.py` | `.collect-secrets.json` 비밀값 설정 및 Supabase 키 관리 | `python tools/collect_secrets.py --check` |
| `make_link.py` | 개인 링크 HMAC 서명 발급 및 Vercel 환경변수 배포 보조 | `python tools/make_link.py --user 홍길동` |
| `vercel_local.py` | Vercel Serverless Function 로컬 시뮬레이터 (SQLite 백엔드, 127.0.0.1:8100) | `python tools/vercel_local.py` |

---

## 4. AI 모델 배포 및 평가 (AI Evaluation & Export)

| 스크립트 | 용도 | 사용 예시 |
|---|---|---|
| `export_ai_demo.py` | 학습된 모델의 롤아웃 궤적을 `collector/index.html` 내 `AI_DEMOS`로 임베딩 | `python tools/export_ai_demo.py --rollout outputs/rollout_apple.json --template apple` |
| `eval/compare_ai_candidates.py` | 동일 도안/조건에서 여러 체크포인트(v4, v5b 등)의 폐루프 롤아웃 성능 일괄 비교 | `python tools/eval/compare_ai_candidates.py` |
| `eval/prepare_ai_choice.py` | 선정된 AI 롤아웃의 고해상도 붓 렌더링 PNG 및 GIF 생성 | `python tools/eval/prepare_ai_choice.py` |
| `eval/render_ai_comparison.py` | 후보 모델들 간의 비교 패널 이미지(`outputs/ai_comparison_brush.png`) 생성 | `python tools/eval/render_ai_comparison.py` |
| `eval/verify_ai_comparison.py` | 후보 모델 롤아웃의 채점 정합성, F1/오차 점수 및 무결성 검증 | `python tools/eval/verify_ai_comparison.py` |
