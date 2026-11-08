"""Vercel 함수: GET /s?score=… (vercel.json이 /s를 여기로 연결) — 공유 미리보기 페이지. 로직은 api_core (make_handler)."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import api_core  # noqa: E402

# Vercel은 파일 안의 최상위 `class handler` 정의로 Python 함수를 인식한다 (대입문은 인식 못 함)
class handler(api_core.make_handler("share")):
    pass
