"""배포(Vercel + Supabase) 환경 로컬 시뮬레이터 — 배포 전에 벤치마크 전체 흐름을 그대로 시험한다.

Vercel과 같은 경로와 같은 로직(api_core)을 쓰고, DB만 Supabase 대신 로컬 SQLite를 쓴다. 127.0.0.1 전용.
  /  /save  /api/stats  /api/ticket  /api/leaderboard  /api/og  /s (공유 미리보기)  /robots.txt  /sitemap.xml
  POST /api/event (퍼널 통계, 항상 204)  POST /api/delete (X-Owner 기기의 기록 지우기)  GET /api/cron (깨우기)
  퍼널 확인: python tools/monitor.py --sqlite outputs/local_bench.sqlite --once

  python tools/vercel_local.py                        테스트용 비밀값, outputs/local_bench.sqlite
  python tools/vercel_local.py --port 8100 --name tester
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

import collect_secrets

ROOT = collect_secrets.ROOT


def main() -> int:
    ap = argparse.ArgumentParser(description="Vercel+Supabase 배포 환경 로컬 시뮬레이터")
    ap.add_argument("--port", type=int, default=8100)
    ap.add_argument("--db", default=str(ROOT / "outputs" / "local_bench.sqlite"))
    ap.add_argument("--secret", default="local-test-secret", help="COLLECT_SECRET (개인 링크 서명 키)")
    ap.add_argument("--name", default="tester", help="출력할 테스트 개인 링크의 이름")
    args = ap.parse_args()
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="replace")

    os.environ.pop("SUPABASE_URL", None)
    os.environ.pop("SUPABASE_KEY", None)
    os.environ["COLLECT_LOCAL_DB"] = args.db
    os.environ["COLLECT_SECRET"] = args.secret
    import api_core  # 환경변수 설정 후 import
    vj = json.loads((ROOT / "vercel.json").read_text(encoding="utf-8"))
    SITE_HEADERS = [(h["key"], h["value"]) for rule in vj.get("headers", []) if rule["source"] == "/(.*)"
                    for h in rule["headers"] if h["key"].lower() not in {k.lower() for k in api_core.SECURITY_HEADERS}]
    from collect_common import sign_name

    class Handler(SimpleHTTPRequestHandler):
        def __init__(self, *a, **kw):
            super().__init__(*a, directory=str(ROOT / "collector"), **kw)

        def end_headers(self):
            if not any(k.lower() == "cache-control" for k in getattr(self, "_sent", ())):
                self.send_header("Cache-Control", "no-store")
            for k, v in SITE_HEADERS:   # vercel.json의 보안 헤더 (CSP 등)
                self.send_header(k, v)
            super().end_headers()
            self._sent = set()   # keep-alive로 같은 핸들러가 다음 요청을 받을 때를 위해

        def send_header(self, k, v):
            self._sent = getattr(self, "_sent", set()) | {k}
            super().send_header(k, v)

        def do_GET(self):
            path = urlparse(self.path).path
            if path == "/api/cron":   # Vercel Cron이 부르는 깨우기 (정규 주소 리다이렉트 없음)
                return api_core.serve_cron(self)
            routes = {"/api/stats": "stats", "/api/ticket": "ticket", "/api/leaderboard": "leaderboard",
                      "/api/og": "og", "/s": "share", "/api/share": "share"}
            name = routes.get(path)
            if name:
                return api_core.serve_get(self, name)
            return super().do_GET()

        def do_POST(self):
            route = {"/save": api_core.serve_save, "/api/save": api_core.serve_save,
                     "/api/event": api_core.serve_event, "/api/delete": api_core.serve_delete}.get(urlparse(self.path).path)
            if route is None:
                return api_core.respond(self, 404, b'{"ok": false}', "application/json")
            route(self)

        def log_message(self, fmt, *a):
            if not self.path.startswith(("/api/stats", "/api/leaderboard", "/api/event")):
                super().log_message(fmt, *a)

    httpd = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    base = f"http://127.0.0.1:{args.port}"
    print(f"[vercel-local] DB: {args.db}")
    print(f"[vercel-local] 공개 참여(닉네임): {base}/")
    print(f"[vercel-local] 개인 링크 예시:   {base}/?u={args.name}&k={sign_name(args.secret, args.name)}", flush=True)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        httpd.server_close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
