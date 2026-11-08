# CLAUDE.md — drawing-vla

## Task
**2D Canvas Trajectory Collection for VLA Action Chunking.**
사람 시연(스타일러스/터치)으로 캔버스 궤적을 수집하고, `(현재 캔버스, 목표 도안)` 관측에서
미래 16스텝 궤적을 예측하는 Action Chunking 정책(Diffusion Policy / ACT 계열)을 학습한다.
목표 시연 수: 30~50회. 수집 기기: Lenovo Yoga Tab (TB710FU, 스타일러스), 터치스크린 노트북.

## Architecture & Vision (2026-11-07 확정)
**최종 목표: 로봇 팔이 아니라 디지털 캔버스 위의 '고속 자율 드로잉 VLA 에이전트'.**
숙련된 일러스트레이터처럼 유려한 선 — 필압, 종 모양 가감속(bell-shaped velocity), 끝선 빼기(tapering)가 살아 있는
유기적 스트로크가 핵심 지표다. 일정한 굵기·꺾임·잔떨림 같은 기계적 트레이싱 느낌은 실패로 본다.

- **두 개의 루프를 분리한다.**
  · 저주파 비전 루프: Action Chunking (16스텝 예측 → 12스텝 실행). 관측은 항상 `render_canvas`(학습과 같은 규격).
  · 고주파 캔버스 렌더러: 실행한 제어점을 `train/brush.py`로 그린다 — 제어점 이동평균(잔떨림 제거) →
    centripetal Catmull-Rom 스플라인(제어점을 지나는 보간) → 끝선 빼기 × 속도(시간 정보가 없으면 곡률로 추정) × 필압 굵기.
    **붓 그림은 보여 주기 전용**이다. 관측(obs ch0)에 넣으면 학습 분포와 달라져 정책이 무너진다.
- **추론 예산:** 단순 MLP 대비 지연 3배 이내 (지금 v4 MLP: XPU에서 1.5ms/청크, p95 2ms → 예산 약 5ms/청크).
  디퓨전·Flow Matching은 4~8스텝 샘플링으로 이 안에 맞출 것. 30~60 FPS 체감은 렌더러가 청크 사이를 보간해서 낸다.
  `rollout_eval.py`가 `[speed]` 줄로 ms/청크를 출력한다.
- **필압 수집:** `strokes[].points[i] = [x, y, t_ms, pressure]` (펜 상태는 '획 안의 점 = 펜 다운'으로 암묵).
  학습 계약 `trajectory = [x, y, pen_down]`은 그대로 두고, 필압·시간은 원시 기록에서 파생한다.
  `meta.has_pressure` = 스타일러스 점이 있고 필압이 변하며 0/0.5/1 기본값만이 아님 (수집기가 기록, 배포 서버가 다시 판정).
  pointerup 이벤트의 pressure는 명세상 항상 0 → 마지막 점은 직전 필압을 이어 받는다.
  지금까지의 시연 68개는 모두 터치(필압 없음). 필압 학습은 Yoga Tab 펜 시연이 모인 뒤.
- **로드맵 (아직 안 한 것):**
  · 필압·속도를 행동에 포함: action `(16, 4~5)` = `[x, y, pen, pressure(, dt)]` 새 버전 — dataset·policies·rollout 함께 바꿀 것
    (기존 `(16, 3)` 체크포인트와 섞지 말고 태그로 구분).
  · 종 모양 가감속 학습: 지금 학습은 호 길이 재샘플(`--resample-step 0.005`)이라 속도 정보가 사라진다.
    시간 기반 표현(스텝마다 dt 또는 속도 채널)을 따로 실험할 것 — 재샘플이 막아 준 '고정점 빠짐' 문제와 함께 볼 것.
  · 정책 출력의 저주파 흔들림(청크 경계에서 방향이 꺾여 물결침): Temporal Ensembling(q=4~6)으로 물결을 절반 이하로 줄임.
    다만 F1이 약 0.01 내려가고 설정에 민감 → 근본 해결은 모델 쪽(청크 간 일관성 학습, 디퓨전 저온 샘플링 등).
- 로컬 롤아웃 = `train/rollout_eval.py` (`run_hybrid.py`는 없음). `--brush`로 붓 PNG·GIF를 함께 저장.

## 절대 금지 (STRICT)
- **No DiffVG**, 미분 렌더링, **베지에 곡선 피팅/세그멘테이션**, 곡률 세그멘테이션, Undo 휴리스틱.
  구 11단계 파이프라인은 폐기됨. 되살리지 말 것.
  (이 금지는 학습 데이터 전처리·학습 목표에 대한 것이다. 롤아웃 '표시'에서 예측 제어점을 지나는 스플라인 보간
  (Catmull-Rom, 동등한 3차 베지에 변환 포함)으로 그리는 것은 허용 — 데이터에 곡선을 맞추는 피팅은 여전히 금지.)
- **No heavy frontend frameworks / build tools**: React, Vite, Node.js, npm, Webpack 금지.
- `collector/index.html`은 **단일 파일 Vanilla HTML/CSS/JS**. 외부 CDN·웹폰트·라이브러리 금지
  (오프라인 브라우저에서 즉시 동작해야 함).
- `server.py`는 **Python 표준 라이브러리만** 사용 (pip 설치 불필요).

## 디렉토리
```
CLAUDE.md
server.py               # [로컬 PC 수집] 정적 호스팅 + POST /save → data/raw/ (+ 채점 결과 응답), GET /api/stats
collector/index.html    # 단일 파일 수집기·벤치마크 (로컬·배포 공용). robots.txt, sitemap.xml도 여기
collect_common.py       # 공용: payload 검증, 파일명, 개인 링크 서명(HMAC)
scoring.py              # 공용 채점 (표준 라이브러리): coverage·precision·F1 → 점수·티어·채택 여부
submission.py           # [배포] 제출 검증: 형식·크기, 서버 궤적 생성, 사람 손 검사, 제출 티켓, 재제출 해시
targets_128.json        # [배포] 도안별 표준 목표 이미지(브라우저 렌더). 저장 시 클라이언트 이미지 대신 이걸 넣는다
api_core.py             # [배포] 저장·통계·리더보드·OG 이미지·공유 페이지 로직 (Vercel 함수와 로컬 시뮬레이터 공유)
db.py                   # [배포] Supabase REST(urllib) / 로컬 SQLite 백엔드
database/schema.sql     # Supabase 테이블 drawing_demos + leaderboard_view + template_counts
api/{save,stats,ticket,leaderboard,og,share}.py   # Vercel 함수 진입점 (얇은 래퍼)
vercel.json, .vercelignore, requirements.txt   # 배포 설정 (화이트리스트 업로드, 웹 공개는 collector/만)
tools/make_link.py      # 비밀값(--init)·Supabase 설정 저장, Vercel 환경변수 명령 출력, 개인 링크 발급
tools/pull_dataset.py   # Supabase의 채택 시연 중 학습 기준(F1·정밀도 ≥0.85, --trusted-only) 통과분 → data/raw/ + 검증
tools/moderate.py       # 리더보드 운영: TOP 10·검토 대기 보기, --approve / --hide
tools/seed_supabase.py  # 로컬 data/raw 검증 시연 → Supabase 초기 등록(닉네임 siat, meta.seeded_from로 중복 방지, --dry-run)
tools/vercel_local.py   # 배포 환경 로컬 시뮬레이터 (SQLite, 127.0.0.1:8100)
tests/test_benchmark.py # python -m unittest tests.test_benchmark
data/raw/demo_*.json    # 시연 원본 (에피소드 1개 = 파일 1개). 결함 시연은 data/rejected/
train/                  # dataset.py, check_demos.py, policies.py, train_policy.py, rollout_eval.py
outputs/                # 체크포인트, 캐시, 롤아웃 GIF (재생성 가능)
.collect-secrets.json   # 비밀값 (collect_secret, supabase_url, supabase_key, url) — 절대 공유·배포 금지
```

## 배포 벤치마크 (Vercel + Supabase)
- 참여: 개인 링크 `/?u=<이름>&k=<HMAC(COLLECT_SECRET, 이름)>` 또는 공개 닉네임(`meta.nickname`, 한글 가능해서 헤더가 아닌 본문으로).
  동의 체크박스 없음: 닉네임 창 버튼 아래 활용 안내 문구만 두고 `meta.notice = "v1"`로 기록. 틀린 링크는 401 → 수집기가 공개 참여로 전환.
- **점수는 서버가 궤적으로 다시 계산한다** (`scoring.py`, TOL 0.02 = check_demos와 같은 기준). 클라이언트 점수는 무시.
  score = 100·F1을 소수 첫째 자리까지 **버림**(반올림 안 함, 예: 98.96 → 98.9). 티어 S ≥99.0 / A ≥85 / B ≥70 / C. `is_qualified` = F1 ≥ 0.70 이고 획 수 1~2·부분수+2.
  S ≥99는 "AI 압도"가 항상 사실이 되도록 AI 기록(98.1)보다 높게 잡은 것. AI 기록이 바뀌면 함께 검토.
- 남용 방지: IP 해시(솔트 SHA-256, 원본 IP 저장 안 함) 기준 1분 6회·1시간 60회 + 전체 1분 600회(초과 시 503, 개인 링크 제외).
  닉네임 NFKC 정규화 → 1~20자 한글·영문·숫자, 금칙어(공백·기호 제거 후 비교), 예약어(siat·관리자 등) 거절.
- **부정행위 방지 (`submission.py`)** — 클라이언트 값은 믿지 않는다:
  · 배포 수집기는 `trajectory`를 보내지 않고, 서버가 `strokes[].points`로 ma3x2 궤적을 직접 만든다(JS와 결과 동일, 테스트로 확인).
  · 허용 필드만 새로 만들어 저장(meta 키 화이트리스트, spec은 서버 값, 숫자 반올림). 점 8000개·획 40개·점 간격 0.25·
    선 길이 합 12·PNG 16KB·본문 512KB 한도 → 조작된 입력의 채점 시간 상한 약 0.1초(이전 최악 9초).
  · 목표 이미지(학습 관측 ch1)는 `targets_128.json`의 표준 이미지로 교체. 도안을 바꾸면 이 파일도 다시 만들 것.
  · 자동 입력 의심 기록은 원시 입력·이미지 없이 meta만 저장(DB 채우기 방지).
  · 사람 손 검사는 **보수적으로**(사람을 잘못 거르지 않게, 사람 시연 측정값의 4~6배 여유): 그린 시간 ≥0.5s,
    평균 속도 ≤4.0/s, 50ms 최고 속도 ≤10/s, 초당 점 5~2000, 평균 오차 ≥0.5px, 시간 역행 1ms 초과 금지. 걸리면 저장은 하되 `is_qualified=false`, `meta.flags`·`meta.checks`에 기록.
  · 제출 티켓(`GET /api/ticket`, HMAC 서명 발급 시각, 6시간): 펜이 닿아 있던 시간(+10초 여유)보다 먼저 발급돼 있어야 하고
    `ticket` 열 고유 → 한 번만. 수집기는 티켓을 `localStorage['vla.ticket']`에 보관(새로고침 후 이어 그리기 대비).
  · `traj_hash`(모양만 본 해시: 0.5px 단위 좌표, 시간·압력 제외) 고유 → 같은 그림 재제출은 409.
  · 시도 횟수는 DB 함수 `hit_rate`(rate_hits·ticket_uses 표)로 원자적으로 센다: 저장 실패·중복 시도도 포함, 동시 요청으로 못 넘김.
    티켓 하나로 5번까지 시도. 함수가 없으면 예전 방식(저장된 행 기준)으로 자동 대체.
  · 평균 오차 < 1.5px(사람 최소 2.0px)은 '검토 대기': 점수는 보여 주고 리더보드·학습에서는 빼 둔다 → `tools/moderate.py --approve`.
  · 학습 데이터는 리더보드보다 엄격하게: `pull_dataset.py` 기본 F1·정밀도 ≥0.85.
  · 잘 만든 봇(사람처럼 흔들리는 선)은 서버만으로 완전히 막을 수 없다. 의심 기록은 `tools/moderate.py --hide <id>`로 내린다.
- 보안: 오류 응답은 오류 번호(`ref`)만, 상세는 `vercel logs`(환경변수 `DEBUG_ERRORS=1`이면 응답에도 한 줄).
  api/*.py는 `class handler(api_core.make_handler(name)): pass` (Vercel CLI는 최상위 `class handler` 정의로만 함수를 인식, `handler = …` 대입은 배포 실패). GET은 정규 주소로만 처리하고 다른 모양(추가 파라미터, 다른 숫자 표기)은 308
  → 쿼리를 바꿔 CDN 캐시를 우회하는 DB·이미지 생성 남용 방지. 허용하지 않는 메서드는 405 JSON, Server 헤더에 파이썬 버전 없음.
  공유 링크 절대 주소는 `SITE_URL` 고정(헤더로 못 바꿈). 공유 카드 티어는 주소 값이 아니라 점수로 계산.
  IP는 `x-vercel-forwarded-for` → `x-real-ip` 순(Vercel이 덮어씀), IPv6는 /64 단위. 개인 링크 `k`는 기억한 뒤 주소창에서 지운다.
  CSP·X-Frame-Options·nosniff·Permissions-Policy는 vercel.json (시뮬레이터도 같은 헤더를 보냄).
- 캐시: 리더보드·공개 통계 CDN 15초(`s-maxage`), OG·공유 페이지 하루. 개인 링크 통계는 `?me=`로 주소를 나눠 캐시 안 함.
  저장 응답에 최신 리더보드(`top`)를 담아 결과 창에 바로 그린다(캐시 우회 없음). Supabase 연결은 스레드별 keep-alive, 저장 경로 DB 조회는 병렬.
- Supabase는 서버만 `service_role` 키(`SUPABASE_KEY`)로 접근한다. RLS 켜고 정책 없음 → anon 키로는 아무것도 못 읽음.
- 리더보드: 도안별 닉네임당 최고 기록. 순위 = 점수 → **평균 오차 `avg_error`**(작을수록 위) → 먼저 달성. 공동 순위 없음.
  `avg_error` = 도안 선 ↔ 그린 선 양방향 평균 거리(0.004 간격 재샘플 선분, TOL에서 자름, 512px 기준 px, 소수 둘째 자리).
  채점 코드를 바꾸면 `python tools/seed_supabase.py --rescore-errors`로 등록된 기록을 다시 계산한다.
  열 추가 전 기록은 `python tools/seed_supabase.py`가 실행될 때 채운다(`missing_avg_error`).
- 공유: `/s?score=&tier=&template=` 페이지가 OG 태그(이미지 `/api/og`, Pillow 1200×630, 영문 표기)를 주고 `/`로 이동.
  결과 카드(1080², 한글)는 브라우저 캔버스로 만들고 Web Share API로 공유(미지원이면 클립보드).
- 공개 주소는 `https://handvsai.vercel.app` (2026-11-07 drawing-vla.vercel.app에서 변경, 옛 주소는 새 주소로 넘김).
  SEO 메타·JSON-LD·robots·sitemap·`api_core.SITE_URL`·OG 이미지 글자·공유 카드가 이 주소를 쓴다. 도메인을 바꾸면 함께 수정.
- 배포 저장에서는 `action_chunks`와 `trajectory`를 받지 않는다(청크는 로드할 때, 궤적은 서버가 파생).
- **스무딩**: `trajectory`는 획마다 3점 이동평균 2회(`meta.smoothing = "ma3x2"`, 양 끝점 유지), 원본은 `strokes[].points`.
  화면 선 = 같은 필터 + 중간점 2차 곡선. 2026-11-07 이전 시연 67개는 스무딩 없이 저장됨(차이는 대부분 1px 미만).
- `meta.input_type` = 점 개수 기준 주 입력 장치(touch/pen/mouse). DB `input_type` 열, 리더보드·카드에 📱/🖊️/🖱️.
- "AI vs 나": `AI_DEMOS`(index.html, `// AI_DEMOS_BEGIN … END`)에 롤아웃 궤적과 **같은 채점기로 매긴 점수**를 넣는다.
  새 모델을 학습하면 `python tools/export_ai_demo.py --rollout … --template …`로 갱신. 지금은 사과만(v4 MLP, 98.1점).
- 뒤로가기 방어: 그림이 있으면 history 한 칸 흡수(3초 안에 다시 뒤로가면 나감) + 획마다 `vla.draft` 자동 보관·복구.
  OS 가장자리 스와이프 자체는 웹에서 완전히 막을 수 없다.
- **획 순서 안내 없음 (2026-11-08~)**: 모바일 테스트에서 번호·화살표가 도안을 가리고 혼란을 줘서 제거했다.
  도안은 반투명 가이드선만, 순서·방향은 자유. `meta.guide_order_marks = false`로 기록(이전 시연은 true인 것이 있음).
  `TEMPLATES[*].order`는 정의만 남아 있다(화면에 안 씀). 채점은 순서·방향과 무관(테스트로 확인).
  학습 데이터에 획 순서가 섞이므로, 순서별로 나눠 학습하려면 `guide_order_marks`·첫 획 위치로 갈래를 구분할 것.
- 캔버스 위 안내: '가이드선을 천천히 정확하게 따라 그릴수록 S급 확률 UP!' 한 줄. 버튼: [모두 지우기](두 번 눌러야) [한 획 지우기] [채점하기].
- **이모지 금지 → 픽셀 아이콘** (2026-11-08): 'AI가 만든 사이트' 느낌을 없애려고 모든 아이콘을 직접 찍은 10×10 도트(`PX`)로 그린다.
  HTML은 `<i data-px="이름">`(시작할 때 SVG로 바뀜), JS는 `pxSvg()`·`pxEl()`, 공유 카드 캔버스는 `drawPx()`. 새 아이콘도 `PX`에 도트로 추가.
- 효과음: Web Audio로 합성한 짧은 8비트 소리(파일 없음). 켜짐이 기본, 헤더 스피커 버튼으로 끔(`localStorage['vla.sound']`).
  획 끝·지우기·채점 시작·점수 카운트업·티어 발표(S가 가장 화려)·AI 승리·공유 복사. iOS는 첫 터치에서 오디오를 연다.
- 결과 창: 점수가 0부터 올라가는 카운트업, 나·AI 그리는 과정 재생(▶ 다시 보기). '동작 줄이기' 설정이면 애니메이션 없이 결과만.
- 캔버스 왼쪽 위: 난이도(`DIFFICULTY`, 별 1~3)와 통과율(70점 이상 비율, 기록 5개 이상일 때).
- 순위 탭: 전체/터치/펜/마우스 — `/api/leaderboard?template=apple&input=pen` (정규 주소 순서 그대로, 모르는 input은 버림).
- 도안 10종 (2026-11-08 하트·별·물고기·집·버섯 추가). 도안을 추가·수정하면 `targets_128.json`도 브라우저 렌더로 다시 만들 것.
- **고르게 모으기:** 배포 모드 기본 = '적은 도안부터'(사용자가 직접 바꾼 경우 `prefs.cycleChosen`만 예외).
  첫 화면과 저장 후 다음 도안은 `pickFewest()` — 가중치 1/(모인 수−최소+1)², 무작위라 동시 접속자가 한 도안에 몰리지 않음.
  (이전 버그: 설정 저장이 기본값 '같은 도안 반복'을 먼저 기록해 배포 기본값이 거의 적용되지 않았음 → 사과 쏠림)
- 리전: Supabase는 `ap-southeast-1`(싱가포르, 무료 플랜이라 변경 불가) → vercel.json `regions: ["sin1"]`로 함수도 싱가포르에.
  저장 한 번에 DB 왕복이 여러 번이라 함수와 DB를 붙여 두는 게 사용자와의 거리보다 중요하다.
- 배포 전에는 `python tools/vercel_local.py`로 같은 로직을 로컬에서 시험한다.

## 텐서 규격 (dataset.py 출력 = 모델 입력 계약)
| key | shape | dtype | 범위 | 의미 |
|---|---|---|---|---|
| `obs` | `(2, 128, 128)` | float32 | [0, 1] | ch0 = 현재 캔버스(`traj[:t]` 렌더), ch1 = 목표 도안. Grayscale, 배경 0 / 잉크 1 |
| `action` | `(16, 3)` | float32 | [0, 1] | `traj[t : t+16]` = `(x, y, pen_down)`, 끝은 마지막 포인트 복제 패딩 |
| `state` | `(3,)` | float32 | [0, 1] | 현재 펜 상태 `traj[t-1]`, `t=0`이면 `HOME_STATE = (0.5, 0.5, 0)` |
| `action_is_pad` | `(16,)` | bool | | 패딩 스텝 여부 (loss 마스킹용) |

- 샘플 1개 = (에피소드, 스텝 `t`), `t ∈ [0, T-1]`. 에피소드당 샘플 수 = `T`.
- **Action chunk는 저장하지 않고 로드 시 파생한다.** JSON의 `trajectory`가 단일 진실 원천.
  `chunk(t)[k] = traj[min(t+k, T-1)]`, k = 0..15 (JS `sliceChunk` ↔ Python `slice_action_chunk` 동일).

## 좌표 / pen_down 규약
- `x` 오른쪽, `y` 아래쪽, 원점 좌상단, 캔버스 한 변으로 정규화, `[0, 1]` clamp, 소수 5자리.
- `pen_down` = **해당 포인트에서의 펜 접촉 상태**. 구간 `i-1 → i`는 두 점 모두 `pen_down=1`일 때만 그려진다.
- 각 획은 펜-업 포인트로 감싼다(bracketing):
  `[s0, 0]` (펜 든 채 시작점으로 이동) → `[s0, 1] … [sn, 1]` (그리기) → `[sn, 0]` (펜 들기).
  연속된 `[이전 끝, 0] → [다음 시작, 0]`이 펜-업 이동(travel)이다.

## 관측 렌더링 (canonical)
- 128×128, 배경 0, 잉크 255(→1.0), 선 두께 **2.0px @128** (= 512 좌표계 8 단위), 둥근 cap/join.
- Python(`render_canvas`): pen-down run마다 4× 슈퍼샘플(PIL) → BOX 다운샘플, run끼리 `max` 합성.
  학습과 배포(추론 시뮬레이터)는 반드시 이 함수 하나로 캔버스를 렌더한다.
- 목표 도안: 브라우저가 같은 두께로 128 해상도 렌더 → 각 JSON의 `target_128_png`.
  템플릿 벡터 정의는 `collector/index.html`의 `TEMPLATES` 한 곳에만 둔다.
- `canvas_128_png`(브라우저 최종 캔버스)는 QA용. `dataset.py` 검증에서 Python 렌더와 IoU 비교.
- 두께·해상도를 바꿀 땐 JS 상수(`OBS_W`, `OBS`)와 Python 상수(`STROKE_WIDTH_PX`, `OBS_SIZE`)를 함께 바꿀 것.

## 시연 JSON 스키마 (`schema_version: 1`)
```jsonc
{
  "schema_version": 1,
  "meta": { "target": "apple", "episode_index": 3, "session_id": "...", "created_at": "ISO8601",
            "num_strokes": 4, "num_steps": 812, "duration_ms": 9340.5, "device": { ... },
            "saved_file": "demo_apple_....json" },          // saved_file은 서버가 추가
  "spec": { "chunk_size": 16, "obs_size": 128, "obs_stroke_width_px": 2, ... },
  "trajectory": [[x, y, pen_down], ...],                    // [T, 3]  ← 학습에 쓰는 유일한 필드
  "strokes": [{ "pointer_type": "pen", "points": [[x, y, t_ms, pressure], ...] }],  // 원시 기록 (meta.has_pressure)
  "target_128_png": "data:image/png;base64,...",
  "canvas_128_png": "data:image/png;base64,...",
  "action_chunks": [[[x, y, p] x16] x T]                    // 선택(기본 꺼짐), 교차 검증용
}
```
파일명: `demo_<target>_<YYYYMMDD-HHMMSS>-<ms>.json` (target은 `[a-z0-9-]`만).

## 수집기 입력 규칙 (펜 지연 최소화)
- Pointer Events만 사용, `pointermove`에서 `e.getCoalescedEvents()` 전부 기록 (서브픽셀 `clientX/Y`, `timeStamp`, `pressure`).
- 캔버스: `touch-action: none`, `setPointerCapture`. 레이어 구조 = 아래 목표 도안(반투명) + 위 투명 잉크 캔버스.
- 잉크 캔버스에 `{ desynchronized: true }` 금지: 일부 안드로이드/GPU에서 투명도가 깨져 도안 레이어를 가린다.
- `pointermove` 안에서는 증분 선분 그리기만. 전체 다시 그리기, `toDataURL`, 레이아웃 읽기 금지 (rect는 pointerdown에서 캐시).
- 멀티터치 무시(활성 pointerId 1개), 펜 감지 시 손가락·손바닥 터치 무시 자동 활성화.

## 명령어
```bash
python server.py                       # 0.0.0.0:8000, 태블릿 접속 URL 출력
python server.py --port 8080 --data-dir data/raw
python train/dataset.py                # shape 검증 (데이터 없으면 합성 시연으로 검증)
python train/dataset.py --resample-step 0.004
python train/check_demos.py            # 내용 품질 검사 → data/qa/contact_<target>.png
python train/train_policy.py --tag v4 --resample-step 0.005 --state-embed 64 --epochs 60 --stride 4 --resume
python train/rollout_eval.py --ckpt outputs/apple_vla_policy_v4.pth --exec-steps 12 --brush   # 폐루프 롤아웃 → GIF, 비교 PNG, 붓 렌더
python train/rollout_eval.py --ckpt outputs/apple_vla_policy_v4.pth --exec-steps 12 --ensemble --brush   # ★ 권장 프리셋
#   --ensemble 기본값 = --query-freq 6 --temporal-ensemble 0.01 (F1 0.973, 화면 물결 1.59°, 추론 2배)
python train/brush.py --demo data/raw/demo_apple_….json   # 사람 시연을 붓 렌더러로 (곡률 추정 vs 실제 속도·필압)
```

## 학습 실험에서 얻은 규칙 (사과, 2026-11-07 기준)
- 현재 최고: **v4 MLP + 롤아웃 12스텝 실행 → F1 0.982** (사람 시연 0.995). 실행 스텝 10~14에서 모두 F1 0.92 이상.
- **궤적은 호 길이로 재샘플(`--resample-step 0.005`)해서 학습한다.** 원본은 사람의 감속·머뭇거림(펜 다운 정지 5.4%) 때문에
  '거의 안 움직이는 청크'가 생기고, 롤아웃이 모서리에서 고정점에 빠진다.
- 펜 상태는 그냥 붙이면 무시된다 → `--state-embed 64`로 키워서 붙인다.
- 롤아웃 실행 스텝 8은 고정점에 빠지기 쉽고, 12가 가장 좋았다.
- **Temporal Ensembling (질의 주기 q, `--ensemble --query-freq q --temporal-ensemble m`)** — v4 MLP, 사과:
  · 매 스텝 질의(q=1)는 손해(F1 0.80~0.91, 모서리를 둥글게 깎아 루프를 못 닫음).
  · **q=6, m=0.01 (= `--ensemble` 기본값, `--brush`와 함께 쓰는 권장 프리셋)**: F1 0.973, 커버리지 99.1%, 화면 물결 3.13°→1.59° (사람 1.21°), 추론 2배.
  · q=4, m=0.1 : F1 0.973, 커버리지 98.8%, 화면 물결 1.35°, 추론 3배(예산 한계).
  · 기준(12스텝 하드 컷): F1 0.982, 커버리지 100%, 물결 3.13°. → 매끈함은 크게 좋아지고 F1은 약 0.01 손해.
  · **설정에 민감**: 같은 q에서 m만 바꿔도 F1 0.89~0.97로 흔들린다(MLP는 결정적이라 시드 평균이 없음).
    새 모델·도안에서는 `te_sweep`처럼 q∈{4,6,8} × m∈{0.01,0.1,-0.1}를 다시 훑을 것.
  · 매끈함 지표(`rollout_eval.smoothness`): 0.005 간격 재샘플 후 평균 꺾임·꺾임 변화(물결), '화면'은 붓 렌더러 다듬기 후.
- 디퓨전(`--head diffusion`): 저온 샘플링 `--noise-scale 0.5` + `--exec-steps 12`에서 평균 F1 0.878. 0 이하로 낮추면 평균화되어 나빠짐.
- 모든 버전이 스스로 끝내지 못하고, '진행 없음' 감지(새 잉크 6회 연속 0)로 멈춘다.
- 평가는 학습과 같은 사과 도안에서의 폐루프 재현이다. 새 도안으로 일반화하는지는 아직 검증하지 않았다.
- 내장 XPU는 장시간 학습 중 `UR_RESULT_ERROR_DEVICE_LOST`로 죽을 수 있다 → 항상 `--resume`으로 실행한다(에폭마다 `outputs/train_state_<tag>.pt` 저장).
- 결함 시연은 지우지 말고 `data/rejected/`로 옮긴다 (로더는 `data/raw/`만 읽음).
- 학습 장치 우선순위: XPU(Intel Arc, `torch==2.15.0+xpu` 설치됨) → CUDA → CPU. FP32만 사용.
- `outputs/cache_<target>.npz`: 진행 중 캔버스 캐시(약 400MB). 지워도 첫 실행 때 약 20초면 다시 만들어진다.
