"""Vercel 함수: POST /api/event — 퍼널 통계 (sendBeacon text/plain 이벤트 이름 → 날짜별 횟수, 항상 204). 로직은 api_core (make_handler)."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import api_core  # noqa: E402

# Vercel은 파일 안의 최상위 `class handler` 정의로 Python 함수를 인식한다 (대입문은 인식 못 함)
class handler(api_core.make_handler("event")):
    pass
