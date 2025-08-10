-- Create table to store AI query telemetry events
create table if not exists public.ai_query_events (
  id uuid primary key,
  created_at timestamptz not null default now(),
  source text not null default 'web', -- 'web' | 'cli' | 'api'
  user_id text null,
  session_id text null,
  question text not null,
  answer text null,
  status text not null default 'ok', -- 'ok' | 'error'
  model text null,
  total_ms integer null,
  timings jsonb null,
  chart jsonb null,
  logs text null,
  error text null,
  metadata jsonb null
);

-- Performance index for common filters
create index if not exists ai_query_events_created_at_idx on public.ai_query_events (created_at desc);
create index if not exists ai_query_events_source_idx on public.ai_query_events (source);
create index if not exists ai_query_events_user_idx on public.ai_query_events (user_id);

-- Enable RLS and allow service role full access; application access can be added later as needed
alter table public.ai_query_events enable row level security;

-- Service role policy (by token) will bypass RLS automatically; add a permissive policy for demonstration reads if needed
do $$ begin
  if not exists (
    select 1 from pg_policies where schemaname = 'public' and tablename = 'ai_query_events' and policyname = 'allow_service_role'
  ) then
    create policy "allow_service_role" on public.ai_query_events
      for all
      using (true)
      with check (true);
  end if;
end $$;


