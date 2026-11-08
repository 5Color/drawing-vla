"""Vercel 함수: GET /api/trajectory?id=<uuid> → 장인 시연 궤적 반환. 로직은 api_core (make_handler)."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import api_core  # noqa: E402

class handler(api_core.make_handler("trajectory")):
    pass
