-- Minimal schema DDL: tables + primary keys + foreign keys only
-- Safe to run in Supabase Cloud SQL editor. No triggers, functions, policies, grants, or indexes.

-- Workspace roles
CREATE TABLE IF NOT EXISTS public.workspace_role (
  id uuid DEFAULT gen_random_uuid() PRIMARY KEY,
  name text NOT NULL UNIQUE
);

-- Workspaces
CREATE TABLE IF NOT EXISTS public.workspace (
  id uuid DEFAULT gen_random_uuid() PRIMARY KEY,
  name text NOT NULL,
  description text,
  created_at timestamptz NOT NULL DEFAULT now()
);

-- Experiments
CREATE TABLE IF NOT EXISTS public.experiment (
  id uuid DEFAULT gen_random_uuid() PRIMARY KEY,
  created_at timestamptz NOT NULL DEFAULT now(),
  name text NOT NULL,
  description text,
  hyperparams jsonb[],
  tags text[],
  updated_at timestamptz NOT NULL DEFAULT now()
);

-- Log entries (metrics/results)
CREATE TABLE IF NOT EXISTS public.log (
  id bigserial PRIMARY KEY,
  experiment_id uuid REFERENCES public.experiment(id) ON UPDATE CASCADE ON DELETE CASCADE,
  created_at timestamptz NOT NULL DEFAULT now(),
  name text NOT NULL,
  value numeric NOT NULL,
  step numeric,
  metadata jsonb
);

-- Workspace <-> Experiment mapping
CREATE TABLE IF NOT EXISTS public.workspace_experiments (
  workspace_id uuid NOT NULL REFERENCES public.workspace(id) ON DELETE CASCADE,
  experiment_id uuid NOT NULL REFERENCES public.experiment(id) ON DELETE CASCADE,
  PRIMARY KEY (workspace_id, experiment_id)
);

-- User workspace access
CREATE TABLE IF NOT EXISTS public.user_workspaces (
  id uuid DEFAULT gen_random_uuid() PRIMARY KEY,
  user_id uuid NOT NULL REFERENCES auth.users(id) ON DELETE CASCADE,
  workspace_id uuid NOT NULL REFERENCES public.workspace(id) ON DELETE CASCADE,
  role_id uuid NOT NULL REFERENCES public.workspace_role(id) ON DELETE CASCADE,
  created_at timestamptz DEFAULT now()
);

-- API keys (hashes stored, plain keys never persisted)
CREATE TABLE IF NOT EXISTS public.api_keys (
  id uuid DEFAULT gen_random_uuid() PRIMARY KEY,
  user_id uuid REFERENCES auth.users(id) ON DELETE CASCADE,
  name text NOT NULL,
  key_hash text NOT NULL UNIQUE,
  created_at timestamptz NOT NULL DEFAULT now(),
  last_used timestamptz NOT NULL DEFAULT now(),
  revoked boolean NOT NULL DEFAULT false
);

-- Workspace invitations
CREATE TABLE IF NOT EXISTS public.workspace_invitations (
  id uuid DEFAULT gen_random_uuid() PRIMARY KEY,
  "to" uuid NOT NULL REFERENCES auth.users(id) ON DELETE CASCADE,
  "from" uuid NOT NULL REFERENCES auth.users(id) ON DELETE CASCADE,
  workspace_id uuid NOT NULL REFERENCES public.workspace(id) ON DELETE CASCADE,
  role_id uuid NOT NULL REFERENCES public.workspace_role(id) ON DELETE CASCADE,
  status text NOT NULL,
  created_at timestamptz NOT NULL DEFAULT now()
);

-- Experiment references (graph of related experiments)
CREATE TABLE IF NOT EXISTS public.experiment_references (
  id serial PRIMARY KEY,
  from_experiment uuid NOT NULL REFERENCES public.experiment(id) ON UPDATE CASCADE ON DELETE CASCADE,
  to_experiment uuid NOT NULL REFERENCES public.experiment(id) ON UPDATE CASCADE ON DELETE CASCADE,
  created_at timestamp DEFAULT current_timestamp,
  CONSTRAINT experiment_references_check CHECK (from_experiment <> to_experiment),
  CONSTRAINT experiment_references_from_to_uniq UNIQUE (from_experiment, to_experiment)
);

-- Note:
-- - This script intentionally omits extensions, triggers, RLS policies, grants, and indexes.
-- - Run separately any needed seeds (e.g., insert default workspace roles).
--   Example:
--     INSERT INTO public.workspace_role (name)
--     VALUES ('OWNER'), ('ADMIN'), ('EDITOR'), ('VIEWER')
--     ON CONFLICT (name) DO NOTHING;
