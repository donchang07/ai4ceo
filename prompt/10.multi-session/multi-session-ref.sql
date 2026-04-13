-- 확장 설치 (pgvector)
create extension if not exists vector;

-- 기존 테이블 정리
drop table if exists public.messages cascade;
drop table if exists public.documents cascade;
drop table if exists public.sessions cascade;

-- sessions 테이블
create table public.sessions (
  id uuid primary key,
  session_id uuid not null,
  title text,
  created_at timestamp with time zone default now(),
  updated_at timestamp with time zone default now()
);

-- messages 테이블
create table public.messages (
  id bigserial primary key,
  session_id uuid not null references public.sessions(id) on delete cascade,
  role text not null,
  content text not null,
  created_at timestamp with time zone default now()
);

-- documents 테이블 (OpenAI 1536차원 기준)
create table public.documents (
  id bigserial primary key,
  content text not null,
  metadata jsonb,
  embedding vector(1536) not null,
  user_id text not null default 'anon',
  created_at timestamp with time zone default now()
);

-- 인덱스
create index if not exists idx_documents_embedding on public.documents using ivfflat (embedding vector_cosine_ops) with (lists = 100);
create index if not exists idx_documents_metadata_session_id on public.documents ((metadata->>'session_id'));
create index if not exists idx_messages_session_id on public.messages(session_id);
create index if not exists idx_sessions_updated_at on public.sessions(updated_at desc);

-- RLS 활성화
alter table public.sessions enable row level security;
alter table public.messages enable row level security;
alter table public.documents enable row level security;

-- 정책: anon 키도 사용 가능하도록 완전 개방 (필요시 조건 추가)
-- sessions
drop policy if exists "allow anon insert sessions" on public.sessions;
drop policy if exists "allow anon select sessions" on public.sessions;
create policy "allow anon insert sessions"
on public.sessions for insert to anon
with check (true);
create policy "allow anon select sessions"
on public.sessions for select to anon
using (true);

-- messages
drop policy if exists "allow anon insert messages" on public.messages;
drop policy if exists "allow anon select messages" on public.messages;
create policy "allow anon insert messages"
on public.messages for insert to anon
with check (true);
create policy "allow anon select messages"
on public.messages for select to anon
using (true);

-- documents
drop policy if exists "allow anon insert documents" on public.documents;
drop policy if exists "allow anon select documents" on public.documents;
create policy "allow anon insert documents"
on public.documents for insert to anon
with check (true);
create policy "allow anon select documents"
on public.documents for select to anon
using (true);

-- match_documents RPC (코사인 유사도)
create or replace function public.match_documents(
    query_embedding vector,
    match_threshold double precision default 0.7,
    match_count integer default 10,
    filter_user_id text default null
) returns table(
    id bigint,
    content text,
    metadata jsonb,
    similarity double precision
) language plpgsql stable as $$
begin
  return query
    select
      d.id,
      d.content,
      d.metadata,
      1 - (d.embedding <=> query_embedding) as similarity
    from public.documents d
    where (filter_user_id is null or d.user_id = filter_user_id)
      and (1 - (d.embedding <=> query_embedding)) >= match_threshold
    order by d.embedding <=> query_embedding
    limit match_count;
end;
$$;

-- 함수에 대해 anon도 실행 가능하도록 권한 부여
grant execute on function public.match_documents(vector, double precision, integer, text) to anon;