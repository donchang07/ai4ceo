-- PDF 기반 멀티유저 멀티세션 RAG 챗봇용 초기 스키마
-- 대상 앱: prompt/10.multi-users/multi-users-ref.py
-- 실행 위치: Supabase SQL Editor

create extension if not exists vector;

-- 기존 함수/테이블 정리
drop function if exists public.match_documents(vector, double precision, integer, text);
drop trigger if exists trg_sessions_updated_at on public.sessions;
drop function if exists public.set_updated_at();
drop table if exists public.messages cascade;
drop table if exists public.documents cascade;
drop table if exists public.sessions cascade;

-- 세션 테이블 (사용자별 소유권 분리)
create table public.sessions (
  id uuid primary key default gen_random_uuid(),
  session_id uuid not null unique,
  user_id uuid not null references auth.users (id) on delete cascade,
  title text default 'New Chat',
  created_at timestamptz not null default now(),
  updated_at timestamptz not null default now()
);

-- 메시지 테이블
create table public.messages (
  id bigserial primary key,
  session_id uuid not null references public.sessions (id) on delete cascade,
  role text not null check (role in ('user', 'ai')),
  content text not null,
  created_at timestamptz not null default now()
);

-- 문서 임베딩 테이블 (OpenAI text-embedding-3-small: 1536차원)
create table public.documents (
  id bigserial primary key,
  content text not null,
  metadata jsonb default '{}'::jsonb,
  embedding vector(1536) not null,
  user_id text not null,
  created_at timestamptz not null default now()
);

create index idx_sessions_user_updated on public.sessions (user_id, updated_at desc);
create index idx_messages_session_id on public.messages (session_id);
create index idx_documents_user_id on public.documents (user_id);
create index idx_documents_metadata_session on public.documents ((metadata ->> 'session_id'));
create index idx_documents_metadata_source on public.documents ((metadata ->> 'source'));

create index if not exists idx_documents_embedding
  on public.documents using hnsw (embedding vector_cosine_ops);

create or replace function public.set_updated_at()
returns trigger
language plpgsql
as $$
begin
  new.updated_at = now();
  return new;
end;
$$;

create trigger trg_sessions_updated_at
before update on public.sessions
for each row execute procedure public.set_updated_at();

-- 벡터 유사도 검색 RPC (앱의 SessionRetriever 시그니처와 동일해야 함)
create or replace function public.match_documents(
  query_embedding vector,
  match_threshold double precision default 0.25,
  match_count integer default 30,
  filter_user_id text default null
)
returns table (
  id bigint,
  content text,
  metadata jsonb,
  similarity double precision
)
language sql
stable
as $$
  select
    d.id,
    d.content,
    d.metadata,
    (1 - (d.embedding <=> query_embedding))::double precision as similarity
  from public.documents d
  where (filter_user_id is null or d.user_id = filter_user_id)
    and (1 - (d.embedding <=> query_embedding)) >= match_threshold
  order by d.embedding <=> query_embedding
  limit match_count;
$$;

grant execute on function public.match_documents(vector, double precision, integer, text)
  to anon, authenticated;

-- RLS 활성화
alter table public.sessions enable row level security;
alter table public.messages enable row level security;
alter table public.documents enable row level security;

-- Advisor 경고 대응: public.users가 존재하면 RLS + 최소 접근 정책 적용
do $$
begin
  if exists (
    select 1
    from information_schema.tables
    where table_schema = 'public' and table_name = 'users'
  ) then
    execute 'alter table public.users enable row level security';
    execute 'revoke all on table public.users from anon, authenticated';

    execute 'drop policy if exists "users_select_own" on public.users';
    execute 'drop policy if exists "users_update_own" on public.users';

    if exists (
      select 1
      from information_schema.columns
      where table_schema = 'public'
        and table_name = 'users'
        and column_name = 'id'
        and udt_name = 'uuid'
    ) then
      execute $p$
        create policy "users_select_own"
          on public.users for select to authenticated
          using (id = auth.uid())
      $p$;
      execute $p$
        create policy "users_update_own"
          on public.users for update to authenticated
          using (id = auth.uid())
          with check (id = auth.uid())
      $p$;
    end if;
  end if;
end $$;

-- 기존 정책 삭제
drop policy if exists "sessions_select" on public.sessions;
drop policy if exists "sessions_insert" on public.sessions;
drop policy if exists "sessions_update" on public.sessions;
drop policy if exists "sessions_delete" on public.sessions;
drop policy if exists "messages_select" on public.messages;
drop policy if exists "messages_insert" on public.messages;
drop policy if exists "messages_update" on public.messages;
drop policy if exists "messages_delete" on public.messages;
drop policy if exists "documents_select" on public.documents;
drop policy if exists "documents_insert" on public.documents;
drop policy if exists "documents_update" on public.documents;
drop policy if exists "documents_delete" on public.documents;

-- 세션: 본인 데이터만 접근
create policy "sessions_select_own"
  on public.sessions for select to authenticated
  using (auth.uid() = user_id);

create policy "sessions_insert_own"
  on public.sessions for insert to authenticated
  with check (auth.uid() = user_id);

create policy "sessions_update_own"
  on public.sessions for update to authenticated
  using (auth.uid() = user_id)
  with check (auth.uid() = user_id);

create policy "sessions_delete_own"
  on public.sessions for delete to authenticated
  using (auth.uid() = user_id);

-- 메시지: 세션 소유자만 접근
create policy "messages_select_session_owner"
  on public.messages for select to authenticated
  using (
    exists (
      select 1
      from public.sessions s
      where s.id = messages.session_id
        and s.user_id = auth.uid()
    )
  );

create policy "messages_insert_session_owner"
  on public.messages for insert to authenticated
  with check (
    exists (
      select 1
      from public.sessions s
      where s.id = messages.session_id
        and s.user_id = auth.uid()
    )
  );

create policy "messages_update_session_owner"
  on public.messages for update to authenticated
  using (
    exists (
      select 1
      from public.sessions s
      where s.id = messages.session_id
        and s.user_id = auth.uid()
    )
  )
  with check (
    exists (
      select 1
      from public.sessions s
      where s.id = messages.session_id
        and s.user_id = auth.uid()
    )
  );

create policy "messages_delete_session_owner"
  on public.messages for delete to authenticated
  using (
    exists (
      select 1
      from public.sessions s
      where s.id = messages.session_id
        and s.user_id = auth.uid()
    )
  );

-- 문서: user_id(text)와 auth.uid()를 문자열 비교
create policy "documents_select_own"
  on public.documents for select to authenticated
  using (user_id = auth.uid()::text);

create policy "documents_insert_own"
  on public.documents for insert to authenticated
  with check (user_id = auth.uid()::text);

create policy "documents_update_own"
  on public.documents for update to authenticated
  using (user_id = auth.uid()::text)
  with check (user_id = auth.uid()::text);

create policy "documents_delete_own"
  on public.documents for delete to authenticated
  using (user_id = auth.uid()::text);
