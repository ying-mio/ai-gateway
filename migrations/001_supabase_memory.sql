-- Supabase MVP schema for tenant-isolated persona/memory/conversation storage
-- tenant_id is required in every business table for strict server-side filtering.

create extension if not exists pgcrypto;

create table if not exists personas (
  tenant_id text primary key,
  name text not null default '',
  system_prompt text not null default '',
  created_at timestamptz not null default now(),
  updated_at timestamptz not null default now()
);

create table if not exists memories (
  id bigint generated always as identity primary key,
  tenant_id text not null,
  title text not null,
  content text not null,
  tags jsonb not null default '[]'::jsonb,
  weight integer not null default 0,
  created_at timestamptz not null default now()
);

create index if not exists idx_memories_tenant_weight_created
  on memories (tenant_id, weight desc, created_at desc);

create table if not exists conversations (
  id uuid primary key default gen_random_uuid(),
  tenant_id text not null,
  title text,
  created_at timestamptz not null default now()
);

create index if not exists idx_conversations_tenant_created
  on conversations (tenant_id, created_at desc);

create table if not exists messages (
  id bigint generated always as identity primary key,
  conversation_id uuid not null references conversations(id) on delete cascade,
  tenant_id text not null,
  role text not null check (role in ('user', 'assistant', 'system')),
  content text not null,
  model text,
  created_at timestamptz not null default now(),
  token_usage jsonb
);

create index if not exists idx_messages_conversation_created
  on messages (conversation_id, created_at asc);

create index if not exists idx_messages_tenant_created
  on messages (tenant_id, created_at desc);

create or replace function set_updated_at()
returns trigger as $$
begin
  new.updated_at = now();
  return new;
end;
$$ language plpgsql;

drop trigger if exists trg_personas_updated_at on personas;
create trigger trg_personas_updated_at
before update on personas
for each row execute function set_updated_at();
