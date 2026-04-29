-- =============================================================================
-- multi-session-ref.sql -> multi-users-ref.py 마이그레이션
-- 대상: 이미 운영 중인 public.sessions/messages/documents 스키마를
--       멀티유저(RLS + user_id 소유권) 구조로 전환
-- =============================================================================

create extension if not exists vector;

-- -----------------------------------------------------------------------------
-- 1) sessions.user_id 추가 (기존 데이터 보존)
-- -----------------------------------------------------------------------------
alter table public.sessions
  add column if not exists user_id uuid;

do $$
begin
  if not exists (
    select 1
    from pg_constraint
    where conname = 'sessions_user_id_fkey'
  ) then
    alter table public.sessions
      add constraint sessions_user_id_fkey
      foreign key (user_id) references auth.users(id) on delete cascade;
  end if;
end $$;

create index if not exists idx_sessions_user_updated
  on public.sessions (user_id, updated_at desc);

comment on column public.sessions.user_id is
  'Supabase Auth 사용자 UUID (multi-users-ref.py 세션 소유자)';

-- -----------------------------------------------------------------------------
-- 2) role 값 정규화 (assistant -> ai)
-- -----------------------------------------------------------------------------
update public.messages
set role = 'ai'
where role = 'assistant';

-- -----------------------------------------------------------------------------
-- 3) 기존 벡터 행 user_id 정리
--    - null/빈값이면 임시 sentinel로 채움 (추후 사용자별 재매핑 권장)
-- -----------------------------------------------------------------------------
update public.documents
set user_id = 'legacy-unassigned'
where user_id is null or btrim(user_id) = '';

-- -----------------------------------------------------------------------------
-- 4) match_documents 함수 재정의 (앱 호출 시그니처 고정)
-- -----------------------------------------------------------------------------
drop function if exists public.match_documents(vector, double precision, integer, text);

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

-- -----------------------------------------------------------------------------
-- 5) 인덱스 보강
-- -----------------------------------------------------------------------------
create index if not exists idx_messages_session_id on public.messages (session_id);
create index if not exists idx_documents_user_id on public.documents (user_id);
create index if not exists idx_documents_metadata_session on public.documents ((metadata ->> 'session_id'));
create index if not exists idx_documents_metadata_source on public.documents ((metadata ->> 'source'));
create index if not exists idx_documents_embedding
  on public.documents using hnsw (embedding vector_cosine_ops);

-- -----------------------------------------------------------------------------
-- 6) updated_at 트리거 보강
-- -----------------------------------------------------------------------------
create or replace function public.set_updated_at()
returns trigger
language plpgsql
as $$
begin
  new.updated_at = now();
  return new;
end;
$$;

drop trigger if exists trg_sessions_updated_at on public.sessions;
create trigger trg_sessions_updated_at
before update on public.sessions
for each row execute procedure public.set_updated_at();

-- -----------------------------------------------------------------------------
-- 7) RLS 정책 멀티유저 기준으로 교체
-- -----------------------------------------------------------------------------
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

-- 기존 정책 삭제 (이름이 달라도 안전하게 여러 후보 삭제)
drop policy if exists "sessions_select" on public.sessions;
drop policy if exists "sessions_insert" on public.sessions;
drop policy if exists "sessions_update" on public.sessions;
drop policy if exists "sessions_delete" on public.sessions;
drop policy if exists "sessions_select_own" on public.sessions;
drop policy if exists "sessions_insert_own" on public.sessions;
drop policy if exists "sessions_update_own" on public.sessions;
drop policy if exists "sessions_delete_own" on public.sessions;

drop policy if exists "messages_select" on public.messages;
drop policy if exists "messages_insert" on public.messages;
drop policy if exists "messages_update" on public.messages;
drop policy if exists "messages_delete" on public.messages;
drop policy if exists "messages_select_session_owner" on public.messages;
drop policy if exists "messages_insert_session_owner" on public.messages;
drop policy if exists "messages_update_session_owner" on public.messages;
drop policy if exists "messages_delete_session_owner" on public.messages;

drop policy if exists "documents_select" on public.documents;
drop policy if exists "documents_insert" on public.documents;
drop policy if exists "documents_update" on public.documents;
drop policy if exists "documents_delete" on public.documents;
drop policy if exists "documents_select_own" on public.documents;
drop policy if exists "documents_insert_own" on public.documents;
drop policy if exists "documents_update_own" on public.documents;
drop policy if exists "documents_delete_own" on public.documents;

-- sessions: 본인 소유 세션만
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

-- messages: 세션 소유자만
create policy "messages_select_session_owner"
  on public.messages for select to authenticated
  using (
    exists (
      select 1 from public.sessions s
      where s.id = messages.session_id
        and s.user_id = auth.uid()
    )
  );

create policy "messages_insert_session_owner"
  on public.messages for insert to authenticated
  with check (
    exists (
      select 1 from public.sessions s
      where s.id = messages.session_id
        and s.user_id = auth.uid()
    )
  );

create policy "messages_update_session_owner"
  on public.messages for update to authenticated
  using (
    exists (
      select 1 from public.sessions s
      where s.id = messages.session_id
        and s.user_id = auth.uid()
    )
  )
  with check (
    exists (
      select 1 from public.sessions s
      where s.id = messages.session_id
        and s.user_id = auth.uid()
    )
  );

create policy "messages_delete_session_owner"
  on public.messages for delete to authenticated
  using (
    exists (
      select 1 from public.sessions s
      where s.id = messages.session_id
        and s.user_id = auth.uid()
    )
  );

-- documents: user_id(text)와 auth.uid() 문자열 비교
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

-- -----------------------------------------------------------------------------
-- 8) 후속 수동 작업 가이드
-- -----------------------------------------------------------------------------
-- (A) 기존 sessions.user_id가 null인 레거시 세션은 앱에서 보이지 않습니다.
--     필요한 경우 아래처럼 특정 사용자에게 소유권 이관하세요.
-- update public.sessions
-- set user_id = '여기에-auth-users-uuid'::uuid
-- where user_id is null;
--
-- (B) 레거시 documents.user_id='legacy-unassigned'는 검색에서 제외됩니다.
--     세션 소유권 이관 후 동일 사용자 uuid 문자열로 함께 이관하세요.
-- update public.documents d
-- set user_id = s.user_id::text
-- from public.sessions s
-- where (d.metadata ->> 'session_id')::uuid = s.id
--   and d.user_id = 'legacy-unassigned';
