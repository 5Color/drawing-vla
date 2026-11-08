"""배포 비밀값 파일(.collect-secrets.json) 읽기/쓰기. 이 파일은 .vercelignore로 배포에서 제외된다."""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SECRETS_PATH = ROOT / ".collect-secrets.json"
sys.path.insert(0, str(ROOT))  # collect_common, api_core 등 루트 모듈


def load(required: tuple[str, ...] = ()) -> dict:
    data = json.loads(SECRETS_PATH.read_text(encoding="utf-8")) if SECRETS_PATH.exists() else {}
    missing = [k for k in required if not data.get(k)]
    if missing:
        sys.exit(f"[error] {SECRETS_PATH.name}에 {', '.join(missing)} 값이 없습니다. "
                 f"먼저 python tools/make_link.py --init (주소는 --url)")
    return data


def save(data: dict) -> None:
    SECRETS_PATH.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
