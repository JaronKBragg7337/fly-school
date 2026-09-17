-- Fly School: the 24/7 classroom log. One row per trial (teach / exam / rest tick). Written by Fly-Lab-2/school/school.py
-- with the service key; read by the public classroom page. Applied 2026-09-17 via Supabase MCP (Claude Code).
create table if not exists fly_school (
  id bigserial primary key,
  at timestamptz not null default now(),
  fly text not null,                 -- 'school-1'
  version text not null,             -- 'fly-v1' lineage name of the brain taking the lesson
  lesson int not null,               -- lesson number since the store was born (never resets)
  phase text not null,               -- 'teach' | 'exam' | 'rest'
  epoch int,                         -- teach: epoch within the lesson; exam: rep
  symbol text,                       -- what was played ('.', '-', '.-', '-.', '' for rest)
  decoded text,                      -- what DNa01 produced
  correct boolean,
  dopamine int,                      -- +1 / -1 on teach trials, null otherwise
  synapses_hit int,
  dn_counts jsonb,                   -- DNa01 spikes per 10 ms bin
  mb jsonb,                          -- mushroom-body stats after the trial
  note text
);
create index if not exists fly_school_at on fly_school (fly, at desc);
create index if not exists fly_school_exam on fly_school (fly, phase, lesson);
alter table fly_school enable row level security;
drop policy if exists "fly_school public read" on fly_school;
create policy "fly_school public read" on fly_school for select using (true);
