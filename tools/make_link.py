"""참여자 개인 링크 발급.

  python tools/make_link.py --init                         비밀값 생성(.collect-secrets.json) + Vercel 환경변수 명령 출력
  python tools/make_link.py --url https://xxx.vercel.app   배포 주소 저장
  python tools/make_link.py minsu jiyoung                  사람마다 개인 링크 출력

링크 = <주소>/?u=<이름>&k=<서명>. 서명은 COLLECT_SECRET으로 만든 HMAC이라 이름만 바꿔서는 쓸 수 없다.
이름은 영문 소문자·숫자·하이픈 1~24자 (파일명과 통계에 그대로 쓰임).
"""
from __future__ import annotations

import argparse
import secrets
import re
import sys

import collect_secrets
from collect_common import normalize_name, sign_name


ENV_KEYS = (("COLLECT_SECRET", "collect_secret"), ("SUPABASE_URL", "supabase_url"), ("SUPABASE_KEY", "supabase_key"))


def env_commands(data: dict) -> str:
    """Vercel 환경변수 등록 명령 (값이 있는 것만). 비밀값이 '-'로 시작하면 CLI가 옵션으로 오해하므로 = 형식."""
    return "\n".join(f"vercel env add {k} production --value={data[v]} --sensitive --force -y"
                     for k, v in ENV_KEYS if data.get(v))


def main() -> int:
    ap = argparse.ArgumentParser(description="참여자 개인 링크 발급")
    ap.add_argument("names", nargs="*", help="참여자 이름 (영문 소문자/숫자/하이픈)")
    ap.add_argument("--init", action="store_true", help="비밀값 새로 만들기")
    ap.add_argument("--force", action="store_true", help="--init 시 기존 비밀값 덮어쓰기 (기존 링크 전부 무효화)")
    ap.add_argument("--url", help="배포 주소 저장 (예: https://handvsai.vercel.app)")
    ap.add_argument("--env-commands", action="store_true", help="Vercel 환경변수 등록 명령 다시 출력")
    ap.add_argument("--supabase-url", help="Supabase Project URL 저장 (pull_dataset.py·환경변수 명령용)")
    ap.add_argument("--supabase-key", help="Supabase service_role 키 저장 (서버 전용 비밀값)")
    args = ap.parse_args()
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="replace")

    data = collect_secrets.load()
    if args.init:
        if data.get("collect_secret") and not args.force:
            sys.exit("[error] 이미 비밀값이 있습니다. 바꾸면 발급한 링크가 전부 무효가 됩니다. 정말이면 --force")
        data["collect_secret"] = secrets.token_urlsafe(32)
        collect_secrets.save(data)
        print(f"[init] {collect_secrets.SECRETS_PATH.name} 생성 (배포·공유하지 마세요)\n")
        print("아래 명령으로 Vercel에 등록하세요:\n" + env_commands(data))
    if args.supabase_url or args.supabase_key:
        data = collect_secrets.load()
        if args.supabase_url:
            u = args.supabase_url.strip().rstrip("/").removesuffix("/rest/v1")
            if "<" in u or not re.fullmatch(r"https://[a-z0-9-]+\.supabase\.co", u):
                sys.exit(f"[error] --supabase-url은 https://<프로젝트ID>.supabase.co 형태여야 합니다 (받은 값: {u})")
            data["supabase_url"] = u
        if args.supabase_key:
            k = args.supabase_key.strip()
            if k.startswith("http") or "<" in k or len(k) < 30:
                sys.exit("[error] --supabase-key에는 URL이 아니라 API Keys의 secret(sb_secret_…) 또는 service_role(eyJ…) 키를 넣으세요")
            if k.startswith("sb_publishable_"):
                sys.exit("[error] publishable(공개) 키입니다. secret(sb_secret_…) 또는 service_role 키를 넣으세요")
            data["supabase_key"] = k
        collect_secrets.save(data)
        print(f"[supabase] 저장 완료. 아래 명령으로 Vercel에 등록하세요:\n" + env_commands(data))
    if args.url:
        data = collect_secrets.load()
        data["url"] = args.url.rstrip("/")
        collect_secrets.save(data)
        print(f"[url] {data['url']}")
    if args.env_commands:
        print(env_commands(collect_secrets.load(("collect_secret",))))
    if args.names:
        data = collect_secrets.load(("collect_secret", "url"))
        for raw in args.names:
            name = normalize_name(raw)
            if not name:
                print(f"[skip] {raw!r}: 영문 소문자·숫자·하이픈 1~24자로 적어 주세요 (예: minsu, kim-jy)")
                continue
            print(f"{name:<12} {data['url']}/?u={name}&k={sign_name(data['collect_secret'], name)}")
    if not (args.init or args.url or args.env_commands or args.names or args.supabase_url or args.supabase_key):
        ap.print_help()
    return 0


if __name__ == "__main__":
    sys.exit(main())
