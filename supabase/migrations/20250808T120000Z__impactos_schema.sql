-- ImpactOS core schema for Supabase (idempotent)
-- Safe to run multiple times; uses IF NOT EXISTS

-- Sources: file provenance
create table if not exists public.sources (
  id bigserial primary key,
  filename text not null,
  file_type text not null,
  upload_timestamp timestamptz default now(),
  file_size_bytes bigint,
  processed_timestamp timestamptz,
  processing_status text default 'pending',
  confidence_score double precision,
  metadata jsonb
);

-- Frameworks catalog
create table if not exists public.frameworks (
  id bigserial primary key,
  name text not null unique,
  version text,
  description text,
  category text,
  created_at timestamptz default now()
);

-- Impact metrics extracted from sources
create table if not exists public.impact_metrics (
  id bigserial primary key,
  source_id bigint not null references public.sources(id) on delete cascade,
  metric_name text not null,
  metric_value double precision not null,
  metric_unit text,
  metric_category text,
  "timestamp" timestamptz,
  extraction_confidence double precision,
  context_description text,
  raw_text text,
  -- precise citation fields
  source_sheet_name text,
  source_column_name text,
  source_column_index integer,
  source_row_index integer,
  source_cell_reference text,
  source_formula text,
  -- verification fields
  verification_status text default 'pending',
  verification_timestamp timestamptz,
  verified_value double precision,
  verification_accuracy double precision,
  verification_notes text,
  created_at timestamptz default now()
);

create index if not exists impact_metrics_category_idx on public.impact_metrics(metric_category);
create index if not exists impact_metrics_name_idx on public.impact_metrics(metric_name);
create index if not exists impact_metrics_verification_idx on public.impact_metrics(verification_status);

-- Commitments (targets, pledges)
create table if not exists public.commitments (
  id bigserial primary key,
  source_id bigint not null references public.sources(id) on delete cascade,
  commitment_text text not null,
  commitment_type text,
  target_value double precision,
  target_unit text,
  target_date date,
  status text default 'active',
  confidence_score double precision,
  created_at timestamptz default now()
);

-- Mapping between metrics and frameworks
create table if not exists public.framework_mappings (
  impact_metric_id bigint references public.impact_metrics(id),
  framework_id bigint references public.frameworks(id),
  category text,
  framework_name text not null,
  framework_category text not null,
  mapping_confidence double precision default 0.8,
  mapping_timestamp timestamptz default now(),
  primary key (impact_metric_id, framework_id)
);

-- Embeddings metadata (vectors remain in FAISS for now)
create table if not exists public.embeddings (
  id bigserial primary key,
  metric_id bigint references public.impact_metrics(id) on delete cascade,
  commitment_id bigint references public.commitments(id) on delete cascade,
  embedding_vector_id text not null,
  text_chunk text not null,
  chunk_type text not null,
  embedding_model text default 'all-MiniLM-L6-v2',
  created_at timestamptz default now(),
  check ((metric_id is not null and commitment_id is null) or (metric_id is null and commitment_id is not null))
);

-- Seed frameworks (no-ops if already present)
insert into public.frameworks (name, version, description, category)
values
  ('UK Social Value Model', '2.0', 'UK Social Value Model for measuring community impact', 'government'),
  ('UN Sustainable Development Goals', '2015', 'United Nations 17 Sustainable Development Goals', 'international'),
  ('TOMs (Themes, Outcomes, Measures)', '3.0', 'National TOMs framework for social value measurement', 'government'),
  ('B Corp Assessment', '6.0', 'B Corporation impact assessment framework', 'certification'),
  ('MAC (Measurement Advisory Council)', '1.0', 'Social value measurement advisory framework', 'advisory')
on conflict (name) do nothing;



