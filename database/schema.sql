-- VLA Drawing Benchmark — Supabase 스키마
-- Supabase 대시보드 → SQL Editor에 통째로 붙여 넣고 Run. 여러 번 실행해도 안전(idempotent).
--
-- 접근 원칙: 이 테이블/뷰는 서버(Vercel 함수)만 service_role 키로 읽고 쓴다.
-- RLS를 켜고 정책을 하나도 만들지 않으므로 anon/authenticated 키로는 아무것도 읽을 수 없다.

create extension if not exists pgcrypto;

create table if not exists public.drawing_demos (
  id              uuid primary key default gen_random_uuid(),
  nickname        text not null check (char_length(nickname) between 1 and 20),
  contributor     text,                         -- 개인 링크 참여자 이름 (공개 참여자는 null)
  template_name   text not null,
  score           real not null check (score between 0 and 100),
  tier            text not null check (tier in ('S', 'A', 'B', 'C')),
  f1              real not null,
  coverage        real not null,
  "precision"     real not null,
  strokes         integer not null,
  is_qualified    boolean not null default false,   -- F1 ≥ 0.70 이고 획 수 정상 → 학습 데이터로 채택
  trajectory_data jsonb not null,                   -- 수집기 payload 전체 (trajectory, strokes, target_128_png …)
  ip_hash         text,                             -- 남용 방지용 (솔트 SHA-256 앞 16자, 원본 IP는 저장 안 함)
  input_type      text,                             -- 주 입력 장치 touch / pen / mouse (학습셋 정제용)
  avg_error       real,                             -- 도안과의 평균 거리 (512px 기준 px). 동점일 때 작은 쪽이 위
  traj_hash       text,                             -- 원시 입력 해시 → 같은 그림 재제출 금지 (고유)
  ticket          text,                             -- 제출 티켓 (서버 서명, 한 번만 사용 → 고유)
  owner           text,                             -- 기기 주인 = sha256(기기 비밀값 vla.owner). 비밀값 자체는 저장 안 함 → '내 기록 지우기'
  created_at      timestamptz not null default now()
);

alter table public.drawing_demos add column if not exists input_type text;   -- 이전 버전 스키마에서 올라올 때
alter table public.drawing_demos add column if not exists avg_error  real;
alter table public.drawing_demos add column if not exists traj_hash  text;
alter table public.drawing_demos add column if not exists ticket     text;
alter table public.drawing_demos add column if not exists owner      text;

create index if not exists drawing_demos_lb_idx   on public.drawing_demos (template_name, score desc) where is_qualified;
create index if not exists drawing_demos_ip_idx   on public.drawing_demos (ip_hash, created_at desc);
create index if not exists drawing_demos_qual_idx on public.drawing_demos (is_qualified, created_at);
create index if not exists drawing_demos_contrib_idx on public.drawing_demos (contributor);
create index if not exists drawing_demos_time_idx on public.drawing_demos (created_at desc);          -- 전체 제출 한도
create index if not exists drawing_demos_best_idx on public.drawing_demos
  (template_name, lower(nickname), score desc, avg_error asc nulls last, created_at asc) where is_qualified;   -- 리더보드
create unique index if not exists drawing_demos_traj_hash_key on public.drawing_demos (traj_hash);
create unique index if not exists drawing_demos_ticket_key    on public.drawing_demos (ticket);
create index if not exists drawing_demos_owner_idx on public.drawing_demos (owner);                 -- 내 기록 지우기

alter table public.drawing_demos enable row level security;
revoke all on public.drawing_demos from anon, authenticated;

-- 도안별 TOP 10 (닉네임당 최고 기록 1개). 순위: 점수 → 평균 오차(작을수록 위) → 먼저 달성
drop view if exists public.leaderboard_view;   -- 열 구성이 바뀌어도 다시 만들 수 있게
create view public.leaderboard_view as
with best as (
  select distinct on (template_name, lower(nickname))
         template_name, nickname, score, avg_error, tier, input_type, created_at
  from public.drawing_demos
  where is_qualified
  order by template_name, lower(nickname), score desc, avg_error asc nulls last, created_at asc
), ranked as (
  select template_name, nickname, score, avg_error, tier, input_type, created_at,
         (rank() over (partition by template_name order by score desc, avg_error asc nulls last, created_at asc))::int as rank
  from best
)
select * from ranked where rank <= 10;

-- 도안별 제출 수 (수집기 화면의 "모두가 모은 시연")
create or replace view public.template_counts as
select template_name,
       count(*)::int                                as total,
       (count(*) filter (where is_qualified))::int  as qualified
from public.drawing_demos
group by template_name;

revoke all on public.leaderboard_view from anon, authenticated;
revoke all on public.template_counts  from anon, authenticated;

-- 시도 횟수 한도 (저장 성공 여부와 무관하게 '시도'를 원자적으로 센다 → 동시 요청·재전송으로 못 넘김)
create table if not exists public.rate_hits (
  k text not null, bucket timestamptz not null, n integer not null default 0, primary key (k, bucket));
create table if not exists public.ticket_uses (
  ticket text primary key, n integer not null default 0, at timestamptz not null default now());
alter table public.rate_hits   enable row level security;
alter table public.ticket_uses enable row level security;
revoke all on public.rate_hits, public.ticket_uses from anon, authenticated;

-- 공개 닉네임의 주인 (처음 그 닉네임으로 저장한 기기). nick = lower(닉네임) = 리더보드가 묶는 기준.
-- 이 기능 전부터 DB에 있던 닉네임은 누군가 그 닉네임으로 저장할 때 주인이 정해진다 (먼저 저장한 기기).
create table if not exists public.nick_owners (
  nick       text primary key,
  owner      text not null,                       -- sha256(기기 비밀값)
  created_at timestamptz not null default now());
create index if not exists nick_owners_owner_idx on public.nick_owners (owner);
alter table public.nick_owners enable row level security;
revoke all on public.nick_owners from anon, authenticated;

-- 퍼널 통계 (방문 → 첫 획 → 채점 → 결과 → 다시/다음/공유 …): 날짜(한국 시간)·이벤트별 횟수만. 사람별 기록 없음.
create table if not exists public.funnel_counts (
  day   date not null,
  event text not null,
  n     integer not null default 0,
  primary key (day, event));
alter table public.funnel_counts enable row level security;
revoke all on public.funnel_counts from anon, authenticated;

-- 시도 1회 기록 + 현재 횟수 + 닉네임 주인 확인 (한 번 호출로 원자적으로).
--   IP·전체·닉네임(p_nick = 공개 참여 닉네임 소문자, 개인 링크는 null)별 1분 버킷, 티켓별 시도 수.
--   닉네임: 주인이 없으면 이 기기(p_owner)가 주인이 된다. 주인이 다른 기기면 nick_ok = false.
--   주인이 아닌 기기의 시도는 닉네임 횟수에 넣지 않는다 (남의 닉네임 한도를 소진시키는 방해 방지).
-- 열 이름은 전부 표 별칭으로 쓴다 (plpgsql 출력 이름 ip_min 등과 열 이름이 섞이면 'column reference is ambiguous').
-- 반환 열을 바꿀 때는 create or replace가 안 되므로 먼저 drop function public.hit_rate(text, text, text, text).
drop function if exists public.hit_rate(text, text);   -- 예전 2개 인자 버전 (서버는 4개 → 없으면 2개 → 저장된 행 기준 순으로 대체)
create or replace function public.hit_rate(p_ip text, p_ticket text, p_nick text, p_owner text)
returns table (ip_min int, ip_hour int, all_min int, ticket_n int, nick_min int, nick_hour int, nick_ok boolean)
language plpgsql security definer set search_path = public as $$
#variable_conflict use_column
declare
  m       timestamptz := date_trunc('minute', now());
  v_owner text;
  v_ok    boolean := true;
begin
  insert into rate_hits as r (k, bucket, n) values ('ip:' || p_ip, m, 1), ('all', m, 1)
    on conflict (k, bucket) do update set n = r.n + 1;
  insert into ticket_uses as t (ticket, n) values (coalesce(p_ticket, ''), 1)
    on conflict (ticket) do update set n = t.n + 1;
  if p_nick is not null then
    if p_owner is not null then
      insert into nick_owners as o (nick, owner) values (p_nick, p_owner)
        on conflict (nick) do nothing;
    end if;
    select o.owner into v_owner from nick_owners o where o.nick = p_nick;
    v_ok := v_owner is null or v_owner = coalesce(p_owner, '');
    if v_ok then
      insert into rate_hits as r (k, bucket, n) values ('nick:' || p_nick, m, 1)
        on conflict (k, bucket) do update set n = r.n + 1;
    end if;
  end if;
  if random() < 0.02 then   -- 오래된 기록 청소 (닉네임 주인·퍼널은 지우지 않음)
    delete from rate_hits r where r.bucket < now() - interval '2 hours';
    delete from ticket_uses t where t.at < now() - interval '7 hours';
  end if;
  return query select
    (select r.n from rate_hits r where r.k = 'ip:' || p_ip and r.bucket = m),
    (select coalesce(sum(r.n), 0)::int from rate_hits r where r.k = 'ip:' || p_ip and r.bucket > now() - interval '1 hour'),
    (select r.n from rate_hits r where r.k = 'all' and r.bucket = m),
    (select t.n from ticket_uses t where t.ticket = coalesce(p_ticket, '')),
    (select coalesce(sum(r.n), 0)::int from rate_hits r
      where p_nick is not null and r.k = 'nick:' || p_nick and r.bucket = m),
    (select coalesce(sum(r.n), 0)::int from rate_hits r
      where p_nick is not null and r.k = 'nick:' || p_nick and r.bucket > now() - interval '1 hour'),
    v_ok;
end $$;
revoke all on function public.hit_rate(text, text, text, text) from public, anon, authenticated;
grant execute on function public.hit_rate(text, text, text, text) to service_role;

-- 퍼널 이벤트 +1 (오늘 = 한국 시간 날짜). 이벤트 이름은 서버(api_core.EVENTS)가 허용한 것만 온다.
create or replace function public.bump_event(p_event text)
returns void
language plpgsql security definer set search_path = public as $$
begin
  if p_event is null or char_length(p_event) > 32 then
    return;
  end if;
  insert into funnel_counts as f (day, event, n)
    values ((now() at time zone 'Asia/Seoul')::date, p_event, 1)
    on conflict (day, event) do update set n = f.n + 1;
end $$;
revoke all on function public.bump_event(text) from public, anon, authenticated;
grant execute on function public.bump_event(text) to service_role;

-- API(PostgREST)가 새 열을 바로 알아보게 스키마 캐시 새로고침
notify pgrst, 'reload schema';
