#!/usr/bin/env python3
"""VLA 드로잉 시연 수집 서버 (Python 표준 라이브러리 전용).

  GET  /            collector/index.html 호스팅 (태블릿/노트북 브라우저에서 접속)
  POST /save        시연 JSON → data/raw/demo_<target>_<YYYYMMDD-HHMMSS>-<ms>.json
  GET  /api/stats   저장된 시연 수 (도안별)

실행:  python server.py [--port 8000] [--host 0.0.0.0] [--data-dir data/raw]
"""
from __future__ import annotations

import argparse
import datetime as dt
import ipaddress
import json
import os
import socket
import sys
import threading
from http import HTTPStatus
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import scoring
from collect_common import count_by_target, sanitize_target, validate_payload

ROOT = Path(__file__).resolve().parent
COLLECTOR_DIR = ROOT / "collector"
DEFAULT_DATA_DIR = ROOT / "data" / "raw"
MAX_BODY_BYTES = 64 * 1024 * 1024  # 64 MB

_write_lock = threading.Lock()


# ---------------------------------------------------------------------------
# 네트워크: 태블릿에서 접속할 PC의 LAN IP 감지
# ---------------------------------------------------------------------------
def describe_ip(ip: str) -> tuple[str, bool]:
    """(설명, 공인 IP 여부)."""
    addr = ipaddress.ip_address(ip)
    if addr in ipaddress.ip_network("192.168.137.0/24"):
        return "Windows 모바일 핫스팟 (태블릿을 PC 핫스팟에 연결한 경우)", False
    if addr in ipaddress.ip_network("100.64.0.0/10"):
        return "CGNAT/VPN (예: Tailscale, 같은 VPN에 접속한 기기만)", False
    if addr.is_private:
        return "사설 LAN (같은 Wi-Fi/공유기)", False
    return "공인 IP (같은 회선 기기만 접속 가능, 인터넷 노출 주의)", True


def lan_ipv4_addresses() -> list[str]:
    """기본 경로 IP를 맨 앞에, 그 외 IPv4를 뒤에 반환 (루프백/링크로컬 제외)."""
    primary = None
    try:
        # UDP connect는 패킷을 보내지 않는다. OS 라우팅 테이블로 기본 인터페이스만 고른다.
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.connect(("8.8.8.8", 80))
            primary = s.getsockname()[0]
    except OSError:
        pass

    others: set[str] = set()
    try:
        for info in socket.getaddrinfo(socket.gethostname(), None, socket.AF_INET):
            others.add(info[4][0])
    except socket.gaierror:
        pass

    ips = [primary] if primary else []
    ips += sorted(ip for ip in others if ip != primary)
    return [ip for ip in ips if ip and not ip.startswith(("127.", "169.254.", "0."))]


# ---------------------------------------------------------------------------
# 저장 / 검증 (검증 로직은 Vercel 함수와 공유: collect_common.py)
# ---------------------------------------------------------------------------
def save_demo(payload: dict, data_dir: Path) -> Path:
    target = sanitize_target(payload["meta"].get("target"))
    now = dt.datetime.now()
    stem = f"demo_{target}_{now:%Y%m%d-%H%M%S}-{now.microsecond // 1000:03d}"
    with _write_lock:
        data_dir.mkdir(parents=True, exist_ok=True)
        path = data_dir / f"{stem}.json"
        n = 1
        while path.exists():
            path = data_dir / f"{stem}-{n}.json"
            n += 1
        payload["meta"]["target"] = target
        payload["meta"]["saved_file"] = path.name
        payload["meta"]["server_received_at"] = now.isoformat(timespec="milliseconds")
        tmp = path.with_name(path.name + ".tmp")  # demo_*.json glob에 걸리지 않음
        tmp.write_text(json.dumps(payload, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
        os.replace(tmp, path)  # 원자적 교체: 반쯤 쓰인 파일이 data/raw에 남지 않음
    return path


def collect_stats(data_dir: Path) -> dict:
    names = [f.name for f in data_dir.glob("demo_*.json")] if data_dir.is_dir() else []
    counts = count_by_target(names)
    return {"counts": counts, "total": sum(counts.values())}


# ---------------------------------------------------------------------------
# HTTP 핸들러
# ---------------------------------------------------------------------------
class CollectorHandler(SimpleHTTPRequestHandler):
    server_version = "DrawingVLA/1.0"
    data_dir: Path = DEFAULT_DATA_DIR  # main()에서 설정

    def __init__(self, *args, **kwargs):
        # collector/ 폴더만 노출 (data/, 소스 코드는 서빙하지 않음)
        super().__init__(*args, directory=str(COLLECTOR_DIR), **kwargs)

    def end_headers(self):
        self.send_header("Cache-Control", "no-store")  # index.html 수정이 태블릿에 바로 반영되도록
        super().end_headers()

    def _route(self) -> str:
        # 요청 줄 파싱이 실패하면 self.path가 아직 없다 (예: https:// 접속, 포트 스캐너)
        return (getattr(self, "path", None) or "").split("?", 1)[0]

    _tls_hinted: set[str] = set()

    def send_error(self, code, message=None, explain=None):
        raw = getattr(self, "raw_requestline", b"") or b""
        if raw[:2] == b"\x16\x03":  # TLS ClientHello: 브라우저가 https://로 접속을 시도함
            ip = self.client_address[0]
            if ip not in self._tls_hinted:
                self._tls_hinted.add(ip)
                print(f"[hint] {ip} 가 https:// 로 접속을 시도했습니다. "
                      f"주소창에 http:// 를 붙여 직접 입력하세요.", flush=True)
            self.close_connection = True  # TLS 클라이언트에 HTTP 오류를 보내도 의미 없음
            return
        super().send_error(code, message, explain)

    def _send_json(self, status: int, obj: dict) -> None:
        body = json.dumps(obj, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        route = self._route()
        if route == "/api/stats":
            return self._send_json(HTTPStatus.OK, {"ok": True, **collect_stats(self.data_dir)})
        if route == "/favicon.ico":
            self.send_response(HTTPStatus.NO_CONTENT)
            self.end_headers()
            return None
        return super().do_GET()

    def do_POST(self):
        if self._route() != "/save":
            return self._send_json(HTTPStatus.NOT_FOUND, {"ok": False, "error": "not found"})
        try:
            length = int(self.headers.get("Content-Length") or 0)
        except ValueError:
            length = 0
        if length <= 0:
            return self._send_json(HTTPStatus.LENGTH_REQUIRED, {"ok": False, "error": "빈 요청"})
        if length > MAX_BODY_BYTES:
            return self._send_json(HTTPStatus.REQUEST_ENTITY_TOO_LARGE, {"ok": False, "error": "요청이 너무 큽니다"})
        try:
            payload = json.loads(self.rfile.read(length).decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as e:
            return self._send_json(HTTPStatus.BAD_REQUEST, {"ok": False, "error": f"JSON 파싱 실패: {e}"})

        err = validate_payload(payload)
        if err:
            self.log_message("저장 거부: %s", err)
            return self._send_json(HTTPStatus.BAD_REQUEST, {"ok": False, "error": err})

        # 배포 벤치마크와 같은 채점 (결과 창 표시용, 파일에도 meta.score로 기록)
        target = sanitize_target(payload["meta"].get("target"))
        sc = scoring.score_trajectory(target, payload["trajectory"]) if target in scoring.load_templates() else {}
        if sc:
            payload["meta"]["score"] = sc

        try:
            path = save_demo(payload, self.data_dir)
        except OSError as e:
            return self._send_json(HTTPStatus.INTERNAL_SERVER_ERROR, {"ok": False, "error": f"파일 쓰기 실패: {e}"})

        meta = payload["meta"]
        stats = collect_stats(self.data_dir)
        print(
            f"[saved] {path.name}  target={meta['target']}  steps={len(payload['trajectory'])}"
            f"  strokes={meta.get('num_strokes', '?')}  score={sc.get('score', '-')}"
            f"  from={self.client_address[0]}  (total {stats['total']})",
            flush=True,
        )
        return self._send_json(HTTPStatus.OK, {"ok": True, "file": path.name, "template": target, **sc, **stats})

    def log_request(self, code="-", size="-"):
        # 상태 폴링 / 정상 정적 요청은 조용히, 페이지 접속과 오류만 출력
        route = self._route()
        ok = isinstance(code, int) and code < 400 or str(code).startswith(("2", "3"))
        if ok and (route.startswith("/api/") or route not in ("/", "/index.html")):
            return
        super().log_request(code, size)


# ---------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser(description="VLA 드로잉 시연 수집 서버")
    ap.add_argument("--host", default="0.0.0.0", help="바인딩 주소 (기본 0.0.0.0 = LAN 허용)")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--data-dir", default=str(DEFAULT_DATA_DIR), help="시연 저장 폴더 (기본 data/raw)")
    args = ap.parse_args()

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="replace")

    data_dir = Path(args.data_dir).resolve()
    data_dir.mkdir(parents=True, exist_ok=True)
    CollectorHandler.data_dir = data_dir

    if not (COLLECTOR_DIR / "index.html").is_file():
        print(f"[error] {COLLECTOR_DIR / 'index.html'} 가 없습니다.", file=sys.stderr)
        return 1

    try:
        httpd = ThreadingHTTPServer((args.host, args.port), CollectorHandler)
    except OSError as e:
        print(f"[error] {args.host}:{args.port} 바인딩 실패 ({e}). 다른 포트: --port 8080", file=sys.stderr)
        return 1

    port = args.port
    lan = lan_ipv4_addresses() if args.host in ("0.0.0.0", "") else []
    line = "=" * 64
    print(line)
    print(" VLA Drawing Collector server")
    print(line)
    any_public = False
    if lan:
        print(" 태블릿에서 접속할 주소 (태블릿 네트워크에 맞는 것 선택):")
        for i, ip in enumerate(lan):
            desc, public = describe_ip(ip)
            any_public |= public
            mark = "*" if i == 0 else " "
            print(f"  {mark} http://{ip}:{port}".ljust(30) + f"  {desc}")
        print("   (* = 기본 경로 인터페이스)")
        print(" * 펜 고속 샘플링(getCoalescedEvents)은 보안 컨텍스트(https·localhost)에서만 켜질 수 있습니다.")
        print(f"   USB로 연결한 안드로이드 태블릿은 'adb reverse tcp:{port} tcp:{port}' 후 http://localhost:{port} 로 접속하세요.")
        print("   (저장된 시연의 meta.device.coalesced_events로 실제 지원 여부를 확인할 수 있습니다)")
    elif args.host in ("0.0.0.0", ""):
        print(" Tablet (same Wi-Fi) : LAN IP를 찾지 못했습니다. ipconfig로 IPv4 주소를 확인하세요.")
    else:
        print(f" Bound to {args.host} only (LAN 접속 불가). 태블릿용은 --host 0.0.0.0")
    print(f" This PC             : http://localhost:{port}")
    print(f" Save dir            : {data_dir}  ({collect_stats(data_dir)['total']} demos)")
    print(line)
    print(" * 태블릿에서 열리지 않으면: PC와 같은 Wi-Fi인지, Windows 방화벽에서")
    print("   Python 인바운드(개인 네트워크) 허용인지 확인하세요.")
    if any_public:
        print(" ! 이 PC에 공인 IP가 있습니다. 방화벽 허용 시 '공용 네트워크'는 체크하지 마세요")
        print("   (인증 없는 /save 엔드포인트가 인터넷에 노출됩니다). 수집 중에만 서버를 켜 두세요.")
    print(" * 종료: Ctrl+C")
    print(line, flush=True)

    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n[stop] 서버 종료")
    finally:
        httpd.server_close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
