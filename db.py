"""벤치마크 DB 계층 (표준 라이브러리 전용).

  SupabaseDB : Supabase REST(PostgREST)를 http.client로 직접 호출. 환경변수 SUPABASE_URL, SUPABASE_KEY(service_role/secret)
               스레드마다 연결을 재사용(keep-alive)해서 요청마다 TLS 연결을 새로 맺지 않는다.
  SqliteDB   : 로컬 시뮬레이터·테스트용. 환경변수 COLLECT_LOCAL_DB (파일 경로)

두 구현은 같은 메서드를 가진다. 스키마는 database/schema.sql.
  hit_rate(ip, ticket, nick, owner)  시도 1회 기록 + 횟수(IP·전체·닉네임·티켓) + 닉네임 주인 확인 (원자적)
  bump_event(event)                  퍼널 통계: 오늘(KST) 이벤트 횟수 +1
  delete_by_owner(owner)             이 기기(owner = sha256(기기 비밀값))로 저장한 기록 삭제 + 닉네임 반납
  keepalive()                        가장 싼 읽기 1번 (Supabase 무료 플랜 일시정지 방지, /api/cron)
schema.sql을 다시 실행하기 전(새 열·표·함수가 아직 없을 때)에도 저장이 멈추지 않게 예전 방식으로 대체한다.
"""
from __future__ import annotations

import contextlib
import datetime as dt
import http.client
import json
import os
import random
import sqlite3
import ssl
import sys
import threading
import time
import urllib.parse
import uuid

LB_SIZE = 10
COLUMNS = ("id", "nickname", "contributor", "template_name", "score", "tier", "f1", "coverage", "precision",
           "strokes", "is_qualified", "trajectory_data", "ip_hash", "input_type", "avg_error", "traj_hash", "ticket",
           "owner", "created_at")
UNIQUE = ("traj_hash", "ticket")
KST = dt.timezone(dt.timedelta(hours=9))
HIT_KEYS = ("ip_min", "ip_hour", "all_min", "ticket_n", "nick_min", "nick_hour")
# PostgREST/Postgres의 '없음' 오류: 함수(PGRST202·42883)·열(PGRST204·42703)·표(PGRST205·42P01)
MISSING_CODES = ("PGRST202", "PGRST204", "PGRST205", "42883", "42703", "42P01")


def _now() -> dt.datetime:
    return dt.datetime.now(dt.timezone.utc)


def _parse_ts(v: str) -> dt.datetime:
    return dt.datetime.fromisoformat(str(v).replace("Z", "+00:00"))


def kst_today() -> str:
    return dt.datetime.now(KST).date().isoformat()


class DBError(RuntimeError):
    pass


class Duplicate(DBError):
    """고유 키(traj_hash, ticket) 중복 → 같은 그림·같은 티켓 재제출."""


def is_missing(err: Exception, name: str) -> bool:
    """name(함수·열·표)이 아직 없다는 오류인가 (database/schema.sql을 다시 실행하기 전)."""
    s = str(err)
    return name in s and any(code in s for code in MISSING_CODES)


def _warn(msg: str) -> None:
    print(f"[db] {msg}", file=sys.stderr, flush=True)


def _hit_row(res) -> dict:
    """hit_rate 응답 → 횟수 dict. 예전 2개 인자 함수에는 닉네임 열이 없다 → 닉네임 횟수 0, 확인 통과."""
    row = (res[0] if isinstance(res, list) and res else res) or {}
    out = {k: int(row.get(k) or 0) for k in HIT_KEYS}
    out["nick_ok"] = row.get("nick_ok") is not False
    return out


# ---------------------------------------------------------------------------
class SupabaseDB:
    IDLE_RECONNECT_S = 20   # 이보다 오래 쉰 연결은 서버가 닫았을 수 있으니 새로 연다

    def __init__(self, url: str, key: str, timeout: float = 10):
        u = urllib.parse.urlsplit(url.strip().rstrip("/").removesuffix("/rest/v1"))
        self.scheme, self.host = u.scheme, u.netloc
        self.base = f"{u.scheme}://{u.netloc}/rest/v1"
        self.key, self.timeout = key, timeout
        self._local = threading.local()
        self._ssl = ssl.create_default_context() if u.scheme == "https" else None

    # --- 전송 (keep-alive) ---
    def _conn(self, fresh: bool):
        c, last = getattr(self._local, "conn", None), getattr(self._local, "last", 0.0)
        if c is not None and (fresh or time.monotonic() - last > self.IDLE_RECONNECT_S):
            c.close()
            c = None
        if c is None:
            c = (http.client.HTTPSConnection(self.host, timeout=self.timeout, context=self._ssl) if self._ssl
                 else http.client.HTTPConnection(self.host, timeout=self.timeout))
            self._local.conn = c
        return c

    def _send(self, method: str, url: str, headers: dict, data: bytes | None) -> tuple[int, dict, bytes]:
        path = url[len(f"{self.scheme}://{self.host}"):]
        for attempt in (0, 1):
            c = self._conn(fresh=attempt == 1)
            try:
                c.request(method, path, body=data, headers=headers)
                r = c.getresponse()
                body = r.read()
                self._local.last = time.monotonic()
                return r.status, {k.lower(): v for k, v in r.getheaders()}, body
            except (http.client.HTTPException, OSError) as e:
                c.close()
                self._local.conn = None
                if attempt == 1 or method != "GET":   # 쓰기는 두 번 보내지 않는다 (중복 저장 방지)
                    raise DBError(f"Supabase 접속 실패: {type(e).__name__}") from None
        raise DBError("unreachable")

    def _req(self, method: str, path: str, params: dict | None = None, body=None, headers: dict | None = None):
        url = f"{self.base}/{path}"
        if params:
            url += "?" + urllib.parse.urlencode(params, safe=",.*()")
        h = {"apikey": self.key, "Accept": "application/json"}
        if not self.key.startswith("sb_"):   # 예전 service_role(JWT)만 Bearer로도 보낸다. 새 sb_secret_ 키는 apikey 헤더만.
            h["Authorization"] = f"Bearer {self.key}"
        data = None
        if body is not None:
            data = json.dumps(body, ensure_ascii=False).encode("utf-8")
            h["Content-Type"] = "application/json"
        h.update(headers or {})
        status, hdr, raw = self._send(method, url, h, data)
        if status >= 400:
            text = raw.decode("utf-8", "replace")[:300]
            if status == 409 and "23505" in text:
                raise Duplicate(text)
            raise DBError(f"Supabase {method} {path} → HTTP {status}: {text}")
        return (json.loads(raw) if raw else None), hdr

    def _count(self, params: dict) -> int:
        _, hdr = self._req("GET", "drawing_demos", {"select": "id", **params},
                           headers={"Prefer": "count=exact", "Range": "0-0"})
        return int(hdr.get("content-range", "*/0").rsplit("/", 1)[-1] or 0)

    # --- 쓰기 ---
    def insert_demo(self, row: dict) -> str:
        try:
            res, _ = self._req("POST", "drawing_demos", {"select": "id"}, body=row,
                               headers={"Prefer": "return=representation"})
        except Duplicate:
            raise
        except DBError as e:
            # owner 열이 아직 없음(schema.sql 다시 실행 전) → owner 없이 한 번 더 (서버가 거절한 요청이라 중복 저장 위험 없음)
            if "owner" not in row or not is_missing(e, "owner"):
                raise
            _warn("drawing_demos.owner 열이 없어 owner 없이 저장합니다 → database/schema.sql을 다시 실행하세요")
            res, _ = self._req("POST", "drawing_demos", {"select": "id"},
                               body={k: v for k, v in row.items() if k != "owner"},
                               headers={"Prefer": "return=representation"})
        return res[0]["id"]

    def insert_many(self, rows: list[dict]) -> int:
        """PostgREST 배열 POST로 한 번에 등록 (호출하는 쪽에서 적당히 나눠서)."""
        self._req("POST", "drawing_demos", body=rows, headers={"Prefer": "return=minimal"})
        return len(rows)

    def update_demo(self, row_id: str, fields: dict) -> None:
        self._req("PATCH", "drawing_demos", {"id": f"eq.{row_id}"}, body=fields, headers={"Prefer": "return=minimal"})

    def delete_by_owner(self, owner: str) -> int:
        """이 기기로 저장한 기록 전부 삭제 + 이 기기가 가진 닉네임 반납. 지운 행 수. owner 열이 아직 없으면 0."""
        if not owner:
            return 0
        try:
            res, _ = self._req("DELETE", "drawing_demos", {"owner": f"eq.{owner}", "select": "id"},
                               headers={"Prefer": "return=representation"})
        except DBError as e:
            if is_missing(e, "owner"):
                return 0
            raise
        try:   # 기록을 먼저 지운 뒤 반납 (반납만 되고 기록이 남으면 남이 같은 닉네임으로 이어 붙을 수 있음)
            self._req("DELETE", "nick_owners", {"owner": f"eq.{owner}"}, headers={"Prefer": "return=minimal"})
        except DBError as e:
            if not is_missing(e, "nick_owners"):
                raise
        return len(res or [])

    def bump_event(self, event: str) -> None:
        """퍼널 통계: 오늘(KST) event 횟수 +1 (schema.sql의 bump_event 함수)."""
        self._req("POST", "rpc/bump_event", body={"p_event": event})

    def keepalive(self) -> bool:
        """가장 싼 읽기 1번 — 무료 플랜은 일주일 동안 요청이 없으면 프로젝트가 일시정지된다."""
        self._req("GET", "drawing_demos", {"select": "id", "limit": 1})
        return True

    # --- 남용 방지 ---
    def hit_rate(self, ip_hash: str, ticket: str, nick: str | None = None, owner: str | None = None) -> dict:
        """시도 1회 기록 + 현재 횟수 + 닉네임 주인 확인 (database/schema.sql의 hit_rate 함수, 원자적).
        → {ip_min, ip_hour, all_min, ticket_n, nick_min, nick_hour, nick_ok}
        4개 인자 함수가 아직 없으면(schema.sql 다시 실행 전) 예전 2개 인자 함수로 → 그것도 없으면 DBError(호출한 쪽이 대체)."""
        try:
            res, _ = self._req("POST", "rpc/hit_rate",
                               body={"p_ip": ip_hash, "p_ticket": ticket, "p_nick": nick, "p_owner": owner})
        except DBError as e:
            if not is_missing(e, "hit_rate"):
                raise
            _warn("hit_rate(4개 인자) 함수가 없어 예전 함수로 셉니다(닉네임 확인 없음) → database/schema.sql을 다시 실행하세요")
            res, _ = self._req("POST", "rpc/hit_rate", body={"p_ip": ip_hash, "p_ticket": ticket})
        return _hit_row(res)

    def ip_recent(self, ip_hash: str, seconds: int, limit: int) -> list[dt.datetime]:
        """이 IP의 최근 제출 시각 (최신순, 최대 limit개) — 요청 한 번으로 분·시간 한도를 함께 계산."""
        since = (_now() - dt.timedelta(seconds=seconds)).isoformat()
        res, _ = self._req("GET", "drawing_demos", {"select": "created_at", "ip_hash": f"eq.{ip_hash}",
                                                    "created_at": f"gte.{since}", "order": "created_at.desc",
                                                    "limit": limit})
        return [_parse_ts(r["created_at"]) for r in res or []]

    def global_recent_count(self, seconds: int) -> int:
        return self._count({"created_at": f"gte.{(_now() - dt.timedelta(seconds=seconds)).isoformat()}"})

    def recent_count(self, ip_hash: str, seconds: int) -> int:
        since = (_now() - dt.timedelta(seconds=seconds)).isoformat()
        return self._count({"ip_hash": f"eq.{ip_hash}", "created_at": f"gte.{since}"})

    # --- 읽기 ---
    def seeded_files(self) -> set[str]:
        """tools/seed_supabase.py로 올린 원본 파일명 (trajectory_data.meta.seeded_from)."""
        out, offset = set(), 0
        while True:
            res, _ = self._req("GET", "drawing_demos", {
                "select": "seed:trajectory_data->meta->>seeded_from",
                "trajectory_data->meta->>seeded_from": "not.is.null", "offset": offset, "limit": 1000})
            out |= {r["seed"] for r in res or []}
            if not res or len(res) < 1000:
                return out
            offset += 1000

    def contributor_count(self, contributor: str) -> int:
        return self._count({"contributor": f"eq.{contributor}"})

    def rank_of(self, template: str, score: float, avg_error: float | None = None) -> tuple[int, int]:
        """(이 기록보다 앞선 채택 기록 수, 채택 기록 전체 수) — 상위 %를 계산.
        앞선 기록 = 점수가 더 높거나, 점수가 같고 평균 오차가 더 작은 것."""
        q = {"template_name": f"eq.{template}", "is_qualified": "eq.true"}
        if avg_error is None:
            better = {"score": f"gt.{score}"}
        else:
            better = {"or": f"(score.gt.{score},and(score.eq.{score},avg_error.lt.{avg_error}))"}
        return self._count({**q, **better}), self._count(q)

    def missing_avg_error(self, limit: int = 50) -> list[dict]:
        """avg_error가 비어 있는 행 (열 추가 전에 등록된 기록 → 채워 넣기용)."""
        res, _ = self._req("GET", "drawing_demos", {
            "select": "id,template_name,trajectory_data", "avg_error": "is.null", "order": "id.asc", "limit": limit})
        return res or []

    # --- 운영 (tools/moderate.py, seed_supabase.py) ---
    def review_rows(self, limit: int = 100) -> list[dict]:
        """'검토 대기' 기록 (사람보다 훨씬 정밀해서 보류된 것)."""
        res, _ = self._req("GET", "drawing_demos", {
            "select": "id,nickname,contributor,template_name,score,avg_error,input_type,is_qualified,created_at,checks:trajectory_data->meta->checks", "is_qualified": "eq.false",
            "trajectory_data->meta->>review": "eq.true", "order": "created_at.desc", "limit": limit})
        return res or []

    def top_rows(self, template: str, limit: int = 10) -> list[dict]:
        res, _ = self._req("GET", "drawing_demos", {
            "select": "id,nickname,contributor,template_name,score,avg_error,input_type,is_qualified,created_at,checks:trajectory_data->meta->checks", "template_name": f"eq.{template}",
            "is_qualified": "eq.true", "order": "score.desc,avg_error.asc.nullslast,created_at.asc", "limit": limit})
        return res or []

    def rows_for_rescore(self, offset: int = 0, limit: int = 100) -> list[dict]:
        res, _ = self._req("GET", "drawing_demos", {
            "select": "id,template_name,avg_error,trajectory:trajectory_data->trajectory",
            "order": "id.asc", "offset": offset, "limit": limit})
        return res or []

    def leaderboard(self, template: str, limit: int = 5) -> list[dict]:
        return self._leaderboard(template, limit)

    def _leaderboard(self, template: str, limit: int, input_type: str | None = None) -> list[dict]:
        """최고 기록의 추정 여부까지 읽는다. 중복 닉네임이 페이지를 채워도 다음 페이지로 이어 간다."""
        limit = min(limit, LB_SIZE)
        rows, offset = [], 0
        while limit > 0:
            page, _ = self._req("GET", "drawing_demos", {
                "select": "id,nickname,score,avg_error,tier,input_type,created_at,score_meta:trajectory_data->meta->score",
                "template_name": f"eq.{template}", "is_qualified": "eq.true",
                **({"input_type": f"eq.{input_type}"} if input_type else {}),
                "order": "score.desc,avg_error.asc.nullslast,created_at.asc,id.asc", "limit": 100, "offset": offset})
            rows.extend(page or [])
            best = _best_per_nickname(rows, limit)
            if len(best) >= limit or len(page or []) < 100:
                return best
            offset += len(page)
        return []

    def leaderboard_by_input(self, template: str, input_type: str, limit: int = 5) -> list[dict]:
        return self._leaderboard(template, limit, input_type)

    def counts(self) -> dict[str, dict]:
        res, _ = self._req("GET", "template_counts", {"select": "template_name,total,qualified"})
        return {r["template_name"]: {"total": r["total"], "qualified": r["qualified"]} for r in res or []}

    def list_qualified(self, offset: int = 0, limit: int = 100) -> list[dict]:
        res, _ = self._req("GET", "drawing_demos", {
            "select": "id,nickname,contributor,template_name,score,f1,precision,coverage,created_at,trajectory_data",
            "is_qualified": "eq.true", "order": "created_at.asc,id.asc", "offset": offset, "limit": limit})
        return res or []


# ---------------------------------------------------------------------------
class SqliteDB:
    """Supabase와 같은 결과를 내는 로컬 구현 (뷰 로직은 파이썬으로)."""

    def __init__(self, path: str):
        self.path, self.lock = path, threading.Lock()
        self._exec(f"create table if not exists drawing_demos ({', '.join(COLUMNS)})")
        have = {r["name"] for r in self._rows("pragma table_info(drawing_demos)")}
        for col in COLUMNS:   # 예전 로컬 DB 파일에 새 열 추가
            if col not in have:
                self._exec(f"alter table drawing_demos add column {col}")
        for col in UNIQUE:
            self._exec(f"create unique index if not exists drawing_demos_{col}_key on drawing_demos ({col})")
        self._exec("create index if not exists drawing_demos_owner_idx on drawing_demos (owner)")
        self._exec("create table if not exists rate_hits (k text, bucket text, n integer, primary key (k, bucket))")
        self._exec("create table if not exists ticket_uses (ticket text primary key, n integer, at text)")
        self._exec("create table if not exists nick_owners (nick text primary key, owner text not null, created_at text)")
        self._exec("create table if not exists funnel_counts (day text, event text, n integer not null default 0, "
                   "primary key (day, event))")

    def _exec(self, sql: str, args=()) -> None:
        # sqlite3의 with 문은 커밋만 하고 연결을 닫지 않는다 → 명시적으로 닫기 (Windows 파일 잠김 방지)
        with self.lock, contextlib.closing(sqlite3.connect(self.path)) as c, c:
            c.execute(sql, args)

    def _rows(self, sql: str, args=()) -> list[dict]:
        with self.lock, contextlib.closing(sqlite3.connect(self.path)) as c:
            c.row_factory = sqlite3.Row
            return [dict(r) for r in c.execute(sql, args)]

    def insert_demo(self, row: dict) -> str:
        r = {**{k: None for k in COLUMNS}, **row, "id": str(uuid.uuid4()),
             "created_at": row.get("created_at") or _now().isoformat()}
        r["trajectory_data"] = json.dumps(r["trajectory_data"], ensure_ascii=False)
        r["is_qualified"] = int(bool(r["is_qualified"]))
        try:
            self._exec(f"insert into drawing_demos ({', '.join(COLUMNS)}) values ({', '.join('?' * len(COLUMNS))})",
                       [r[k] for k in COLUMNS])
        except sqlite3.IntegrityError as e:
            raise Duplicate(str(e)) from None
        return r["id"]

    def insert_many(self, rows: list[dict]) -> int:
        for row in rows:
            self.insert_demo(row)
        return len(rows)

    def update_demo(self, row_id: str, fields: dict) -> None:
        cols = list(fields)
        self._exec(f"update drawing_demos set {', '.join(c + '=?' for c in cols)} where id=?",
                   [fields[c] for c in cols] + [row_id])

    def hit_rate(self, ip_hash: str, ticket: str, nick: str | None = None, owner: str | None = None) -> dict:
        """Supabase hit_rate 함수와 같은 규칙 (한 트랜잭션). 닉네임은 처음 저장한 기기(owner)가 주인이 되고,
        주인이 아닌 기기의 시도는 닉네임 횟수에 넣지 않는다 (남의 닉네임 한도를 소진시키는 방해 방지)."""
        now = _now()
        m = now.replace(second=0, microsecond=0).isoformat()
        hour_ago = (now - dt.timedelta(hours=1)).isoformat()
        ticket = ticket or ""
        up = "insert into rate_hits values (?, ?, 1) on conflict (k, bucket) do update set n = n + 1"
        with self.lock, contextlib.closing(sqlite3.connect(self.path)) as c, c:
            for k in (f"ip:{ip_hash}", "all"):
                c.execute(up, (k, m))
            c.execute("insert into ticket_uses values (?, 1, ?) on conflict (ticket) do update set n = n + 1",
                      (ticket, now.isoformat()))
            ok = True
            if nick is not None:
                if owner is not None:
                    c.execute("insert into nick_owners values (?, ?, ?) on conflict (nick) do nothing",
                              (nick, owner, now.isoformat()))
                row = c.execute("select owner from nick_owners where nick=?", (nick,)).fetchone()
                ok = row is None or row[0] == (owner or "")
                if ok:
                    c.execute(up, (f"nick:{nick}", m))
            if random.random() < 0.02:   # 오래된 기록 청소 (닉네임 주인·퍼널은 지우지 않음)
                c.execute("delete from rate_hits where bucket<?", ((now - dt.timedelta(hours=2)).isoformat(),))
                c.execute("delete from ticket_uses where at<?", ((now - dt.timedelta(hours=7)).isoformat(),))
            one = lambda q, a: (c.execute(q, a).fetchone() or [0])[0] or 0
            nk = f"nick:{nick}" if nick is not None else None
            return {"ip_min": one("select n from rate_hits where k=? and bucket=?", (f"ip:{ip_hash}", m)),
                    "ip_hour": one("select sum(n) from rate_hits where k=? and bucket>?", (f"ip:{ip_hash}", hour_ago)),
                    "all_min": one("select n from rate_hits where k='all' and bucket=?", (m,)),
                    "ticket_n": one("select n from ticket_uses where ticket=?", (ticket,)),
                    "nick_min": one("select n from rate_hits where k=? and bucket=?", (nk, m)) if nk else 0,
                    "nick_hour": one("select sum(n) from rate_hits where k=? and bucket>?", (nk, hour_ago)) if nk else 0,
                    "nick_ok": ok}

    def bump_event(self, event: str) -> None:
        self._exec("insert into funnel_counts values (?, ?, 1) on conflict (day, event) do update set n = n + 1",
                   (kst_today(), event))

    def delete_by_owner(self, owner: str) -> int:
        if not owner:
            return 0
        with self.lock, contextlib.closing(sqlite3.connect(self.path)) as c, c:
            n = c.execute("delete from drawing_demos where owner=?", (owner,)).rowcount
            c.execute("delete from nick_owners where owner=?", (owner,))
        return n

    def keepalive(self) -> bool:
        self._rows("select 1 n")
        return True

    def ip_recent(self, ip_hash: str, seconds: int, limit: int) -> list[dt.datetime]:
        since = (_now() - dt.timedelta(seconds=seconds)).isoformat()
        rows = self._rows("select created_at from drawing_demos where ip_hash=? and created_at>=? "
                          "order by created_at desc limit ?", (ip_hash, since, limit))
        return [_parse_ts(r["created_at"]) for r in rows]

    def global_recent_count(self, seconds: int) -> int:
        since = (_now() - dt.timedelta(seconds=seconds)).isoformat()
        return self._rows("select count(*) n from drawing_demos where created_at>=?", (since,))[0]["n"]

    def recent_count(self, ip_hash: str, seconds: int) -> int:
        since = (_now() - dt.timedelta(seconds=seconds)).isoformat()
        return self._rows("select count(*) n from drawing_demos where ip_hash=? and created_at>=?", (ip_hash, since))[0]["n"]

    def seeded_files(self) -> set[str]:
        return {r["seed"] for r in self._rows(
            "select json_extract(trajectory_data, '$.meta.seeded_from') seed from drawing_demos where seed is not null")}

    def contributor_count(self, contributor: str) -> int:
        return self._rows("select count(*) n from drawing_demos where contributor=?", (contributor,))[0]["n"]

    def rank_of(self, template: str, score: float, avg_error: float | None = None) -> tuple[int, int]:
        q = "select count(*) n from drawing_demos where template_name=? and is_qualified=1"
        if avg_error is None:
            better = self._rows(q + " and score>?", (template, score))[0]["n"]
        else:
            better = self._rows(q + " and (score>? or (score=? and avg_error<?))",
                                (template, score, score, avg_error))[0]["n"]
        return better, self._rows(q, (template,))[0]["n"]

    def missing_avg_error(self, limit: int = 50) -> list[dict]:
        rows = self._rows("select id, template_name, trajectory_data from drawing_demos where avg_error is null "
                          "order by id limit ?", (limit,))
        for r in rows:
            r["trajectory_data"] = json.loads(r["trajectory_data"])
        return rows

    def leaderboard(self, template: str, limit: int = 5) -> list[dict]:
        rows = self._rows("select id, nickname, score, avg_error, tier, input_type, created_at, "
                          "json_extract(trajectory_data, '$.meta.score') score_meta from drawing_demos "
                          "where template_name=? and is_qualified=1 "
                          "order by score desc, avg_error is null, avg_error asc, created_at asc, id asc", (template,))
        return _best_per_nickname(rows, limit)

    def review_rows(self, limit: int = 100) -> list[dict]:
        rows = self._rows("select id, nickname, contributor, template_name, score, avg_error, input_type, is_qualified, created_at, json_extract(trajectory_data, '$.meta.checks') checks "
                          "from drawing_demos where is_qualified=0 and json_extract(trajectory_data, '$.meta.review') = 1 "
                          "order by created_at desc limit ?", (limit,))
        return [{**r, "checks": json.loads(r["checks"]) if r["checks"] else None} for r in rows]

    def top_rows(self, template: str, limit: int = 10) -> list[dict]:
        rows = self._rows("select id, nickname, contributor, template_name, score, avg_error, input_type, is_qualified, created_at, json_extract(trajectory_data, '$.meta.checks') checks "
                          "from drawing_demos where template_name=? and is_qualified=1 "
                          "order by score desc, avg_error is null, avg_error, created_at limit ?", (template, limit))
        return [{**r, "checks": json.loads(r["checks"]) if r["checks"] else None} for r in rows]

    def rows_for_rescore(self, offset: int = 0, limit: int = 100) -> list[dict]:
        rows = self._rows("select id, template_name, avg_error, json_extract(trajectory_data, '$.trajectory') trajectory "
                          "from drawing_demos order by id limit ? offset ?", (limit, offset))
        return [{**r, "trajectory": json.loads(r["trajectory"]) if r["trajectory"] else None} for r in rows]

    def leaderboard_by_input(self, template: str, input_type: str, limit: int = 5) -> list[dict]:
        rows = self._rows("select id, nickname, score, avg_error, tier, input_type, created_at, "
                          "json_extract(trajectory_data, '$.meta.score') score_meta from drawing_demos "
                          "where template_name=? and is_qualified=1 and input_type=? "
                          "order by score desc, avg_error is null, avg_error asc, created_at asc, id asc",
                          (template, input_type))
        return _best_per_nickname(rows, limit)

    def counts(self) -> dict[str, dict]:
        return {r["template_name"]: {"total": r["total"], "qualified": r["qualified"]} for r in self._rows(
            "select template_name, count(*) total, sum(is_qualified) qualified from drawing_demos group by template_name")}

    def list_qualified(self, offset: int = 0, limit: int = 100) -> list[dict]:
        rows = self._rows("select id, nickname, contributor, template_name, score, f1, precision, coverage, created_at, trajectory_data "
                          "from drawing_demos where is_qualified=1 order by created_at, id limit ? offset ?", (limit, offset))
        for r in rows:
            r["trajectory_data"] = json.loads(r["trajectory_data"])
        return rows


def _best_per_nickname(rows: list[dict], limit: int) -> list[dict]:
    """정렬된 기록 → 닉네임당 첫(최고) 기록만, 순위 번호를 붙여 상위 limit개."""
    best, seen = [], set()
    for r in rows:
        k = r["nickname"].lower()
        if k not in seen:
            seen.add(k)
            best.append(r)
    out = []
    for i, r in enumerate(best[:min(limit, LB_SIZE)], 1):
        meta = r.get("score_meta") or {}
        if isinstance(meta, str):
            try:
                meta = json.loads(meta)
            except ValueError:
                meta = {}
        out.append({**r, "rank": i, "score_estimated": isinstance(meta, dict) and meta.get("estimated") is True})
    return out


def from_env():
    url, key = os.environ.get("SUPABASE_URL", "").strip(), os.environ.get("SUPABASE_KEY", "").strip()
    if url and key:
        return SupabaseDB(url, key)
    local = os.environ.get("COLLECT_LOCAL_DB")
    if local:
        return SqliteDB(local)
    raise DBError("DB 설정이 없습니다: SUPABASE_URL과 SUPABASE_KEY 환경변수를 설정하세요")
