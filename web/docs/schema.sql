

SET statement_timeout = 0;
SET lock_timeout = 0;
SET idle_in_transaction_session_timeout = 0;
SET client_encoding = 'UTF8';
SET standard_conforming_strings = on;
SELECT pg_catalog.set_config('search_path', '', false);
SET check_function_bodies = false;
SET xmloption = content;
SET client_min_messages = warning;
SET row_security = off;


CREATE SCHEMA IF NOT EXISTS "dev1";


ALTER SCHEMA "dev1" OWNER TO "postgres";


CREATE EXTENSION IF NOT EXISTS "pgsodium";






COMMENT ON SCHEMA "public" IS 'standard public schema';



CREATE EXTENSION IF NOT EXISTS "pg_graphql" WITH SCHEMA "graphql";






CREATE EXTENSION IF NOT EXISTS "pg_stat_statements" WITH SCHEMA "extensions";






CREATE EXTENSION IF NOT EXISTS "pg_trgm" WITH SCHEMA "public";






CREATE EXTENSION IF NOT EXISTS "pgcrypto" WITH SCHEMA "extensions";






CREATE EXTENSION IF NOT EXISTS "pgjwt" WITH SCHEMA "extensions";






CREATE EXTENSION IF NOT EXISTS "supabase_vault" WITH SCHEMA "vault";






CREATE EXTENSION IF NOT EXISTS "uuid-ossp" WITH SCHEMA "extensions";






CREATE TYPE "public"."user_experiment_role" AS ENUM (
    'OWNER',
    'EDITOR',
    'VIEWER'
);


ALTER TYPE "public"."user_experiment_role" OWNER TO "postgres";


CREATE TYPE "public"."visibility" AS ENUM (
    'PUBLIC',
    'PRIVATE'
);


ALTER TYPE "public"."visibility" OWNER TO "postgres";


CREATE OR REPLACE FUNCTION "public"."check_experiment_access"("experiment_id" "uuid", "user_id" "uuid") RETURNS TABLE("has_access" boolean)
    LANGUAGE "plpgsql"
    AS $$
  BEGIN
    RETURN QUERY
    SELECT EXISTS (
      SELECT 1 FROM experiment e
      LEFT JOIN user_experiments ue ON e.id = ue.experiment_id
      WHERE e.id = experiment_id
      AND (
        e.visibility = 'PUBLIC' OR
        (ue.user_id = user_id AND ue.role IN ('OWNER', 'COLLABORATOR'))
      )
    ) AS has_access;
  END;
  $$;


ALTER FUNCTION "public"."check_experiment_access"("experiment_id" "uuid", "user_id" "uuid") OWNER TO "postgres";


CREATE OR REPLACE FUNCTION "public"."get_experiment_chain"("target_experiment_id" "uuid") RETURNS TABLE("experiment_id" "uuid", "experiment_created_at" timestamp with time zone, "experiment_updated_at" timestamp with time zone, "experiment_name" "text", "experiment_description" "text", "experiment_hyperparams" "jsonb"[], "experiment_tags" "text"[], "experiment_visibility" "public"."visibility", "depth" integer)
    LANGUAGE "sql"
    AS $$
  WITH RECURSIVE ExperimentChain AS (
    -- Base case
    SELECT
      e.id,
      e.created_at,
      e.updated_at,
      e.name,
      e.description,
      e.hyperparams,
      e.tags,
      e.visibility,
      0 AS depth
    FROM experiment e
    WHERE e.id = target_experiment_id

    UNION ALL

    -- Recursive case
    SELECT
      e.id,
      e.created_at,
      e.updated_at,
      e.name,
      e.description,
      e.hyperparams,
      e.tags,
      e.visibility,
      ec.depth + 1
    FROM experiment e
    JOIN experiment_references er ON e.id = er.to_experiment
    JOIN ExperimentChain ec ON er.from_experiment = ec.id
  )
  SELECT
    id AS experiment_id,
    created_at AS experiment_created_at,
    updated_at AS experiment_updated_at,
    name AS experiment_name,
    description AS experiment_description,
    hyperparams AS experiment_hyperparams,
    tags AS experiment_tags,
    visibility AS experiment_visibility,
    depth
  FROM ExperimentChain
  ORDER BY depth DESC;
$$;


ALTER FUNCTION "public"."get_experiment_chain"("target_experiment_id" "uuid") OWNER TO "postgres";


CREATE OR REPLACE FUNCTION "public"."get_experiments_and_metrics"("experiment_ids" "uuid"[] DEFAULT NULL::"uuid"[]) RETURNS TABLE("id" "uuid", "name" "text", "description" "text", "created_at" timestamp with time zone, "updated_at" timestamp with time zone, "tags" "text"[], "hyperparams" "jsonb"[], "metric_dict" "json", "visibility" "public"."visibility")
    LANGUAGE "plpgsql"
    AS $$
BEGIN
    RETURN QUERY
    SELECT 
        e.id, 
        e.name, 
        e.description, 
        e.created_at, 
        e.updated_at,
        e.tags, 
        e.hyperparams, 
        md.metric_dict,
        e.visibility
    FROM experiment e
    LEFT JOIN (
        SELECT
            grouped.experiment_id,
            JSON_OBJECT_AGG(grouped.name, grouped.values) AS metric_dict
        FROM (
            SELECT
                experiment_id,
                metric.name,
                ARRAY_AGG(metric.value ORDER BY metric.created_at ASC) AS values
            FROM metric
            GROUP BY metric.experiment_id, metric.name
        ) AS grouped
        GROUP BY grouped.experiment_id
    ) md ON e.id = md.experiment_id
    WHERE experiment_ids IS NULL OR e.id = ANY(experiment_ids);
END;
$$;


ALTER FUNCTION "public"."get_experiments_and_metrics"("experiment_ids" "uuid"[]) OWNER TO "postgres";


CREATE OR REPLACE FUNCTION "public"."get_experiments_with_metric_names"("experiment_ids" "uuid"[] DEFAULT NULL::"uuid"[]) RETURNS TABLE("id" "uuid", "name" "text", "description" "text", "created_at" timestamp with time zone, "tags" "text"[], "hyperparams" "jsonb"[], "available_metrics" "text"[], "visibility" "public"."visibility")
    LANGUAGE "plpgsql"
    AS $$
BEGIN
    RETURN QUERY
    SELECT 
        e.id, 
        e.name, 
        e.description, 
        e.created_at, 
        e.tags, 
        e.hyperparams, 
        em.available_metrics,
        e.visibility
    FROM experiment e
    INNER JOIN (
        SELECT experiment_id, ARRAY_AGG(DISTINCT metric.name) AS available_metrics
        FROM metric
        GROUP BY experiment_id
    ) em ON e.id = em.experiment_id
    WHERE experiment_ids IS NULL OR e.id = ANY(experiment_ids);
END;
$$;


ALTER FUNCTION "public"."get_experiments_with_metric_names"("experiment_ids" "uuid"[]) OWNER TO "postgres";


CREATE OR REPLACE FUNCTION "public"."get_user_experiments"("user_id" "uuid", "workspace_id" "uuid" DEFAULT NULL::"uuid") RETURNS TABLE("experiment_id" "uuid", "experiment_user_id" "uuid", "experiment_user_role" "public"."user_experiment_role", "experiment_created_at" timestamp with time zone, "experiment_updated_at" timestamp with time zone, "experiment_name" "text", "experiment_description" "text", "experiment_hyperparams" "jsonb"[], "experiment_tags" "text"[], "experiment_visibility" "public"."visibility", "available_metrics" "text"[])
    LANGUAGE "plpgsql"
    AS $$
BEGIN
  RETURN QUERY
  SELECT DISTINCT
    e.id AS experiment_id,
    ue.user_id AS experiment_user_id,
    ue.role AS experiment_user_role,
    e.created_at AS experiment_created_at,
    e.updated_at AS experiment_updated_at,
    e.name AS experiment_name,
    e.description AS experiment_description,
    e.hyperparams AS experiment_hyperparams,
    e.tags AS experiment_tags,
    e.visibility AS experiment_visibility,
    COALESCE(em.available_metrics, ARRAY[]::TEXT[]) AS available_metrics
  FROM experiment e
  LEFT JOIN user_experiments ue
    ON e.id = ue.experiment_id AND ue.user_id = get_user_experiments.user_id
  LEFT JOIN workspace_experiments we
    ON e.id = we.experiment_id
  LEFT JOIN (
    SELECT
      m.experiment_id,
      ARRAY_AGG(DISTINCT m.name) AS available_metrics
    FROM metric m
    GROUP BY m.experiment_id
  ) em ON e.id = em.experiment_id
  WHERE
    (e.visibility = 'PUBLIC' OR ue.role IN (
      'OWNER'::public.user_experiment_role,
      'EDITOR'::public.user_experiment_role
    ))
    AND (get_user_experiments.workspace_id IS NULL OR we.workspace_id = get_user_experiments.workspace_id);
END;
$$;


ALTER FUNCTION "public"."get_user_experiments"("user_id" "uuid", "workspace_id" "uuid") OWNER TO "postgres";


CREATE OR REPLACE FUNCTION "public"."get_workspace_experiments"("name_filter" "text" DEFAULT ''::"text", "user_id_param" "text" DEFAULT NULL::"text", "workspace_id_param" "uuid" DEFAULT NULL::"uuid") RETURNS TABLE("id" "uuid", "created_at" timestamp with time zone, "name" "text", "description" "text", "hyperparams" "jsonb"[], "tags" "text"[], "visibility" "text", "user_id" "text", "available_metrics" "text"[], "key_metrics" "jsonb")
    LANGUAGE "plpgsql"
    AS $$BEGIN
    IF workspace_id_param IS NOT NULL THEN
      -- Get experiments for a specific workspace
      RETURN QUERY
      SELECT
        e.id,
        e.created_at,
        e.name,
        e.description,
        e.hyperparams,
        e.tags,
        e.visibility::TEXT,
        ue.user_id,
        COALESCE(
          ARRAY(
            SELECT DISTINCT m.name
            FROM metric m
            WHERE m.experiment_id = e.id
          ),
          ARRAY[]::TEXT[]
        ) as available_metrics,
        COALESCE(
          (
            SELECT jsonb_agg(
              jsonb_build_object(
                'name', latest_metrics.name,
                'value', CASE
                  WHEN latest_metrics.value % 1 = 0 THEN latest_metrics.value::INTEGER
                  ELSE ROUND(latest_metrics.value::NUMERIC, 4)
                END
              )
              ORDER BY latest_metrics.created_at DESC
            )
            FROM (
              SELECT DISTINCT ON (m.name)
                m.name,
                m.value,
                m.created_at
              FROM metric m
              WHERE m.experiment_id = e.id
              ORDER BY m.name, m.created_at DESC
              LIMIT 2
            ) latest_metrics
          ),
          '[]'::JSONB
        ) as key_metrics
      FROM experiment e
      INNER JOIN workspace_experiments we ON e.id = we.experiment_id
      LEFT JOIN user_experiments ue ON e.id = ue.experiment_id
      WHERE we.workspace_id = workspace_id_param
        AND e.name ILIKE '%' || name_filter || '%'
      ORDER BY e.created_at DESC;

    ELSIF user_id_param IS NOT NULL THEN
      -- Get public experiments + user's private experiments
      RETURN QUERY
      SELECT
        e.id,
        e.created_at,
        e.name,
        e.description,
        e.hyperparams,
        e.tags,
        e.visibility::TEXT,
        COALESCE(ue.user_id, user_id_param) as user_id,
        COALESCE(
          ARRAY(
            SELECT DISTINCT m.name
            FROM metric m
            WHERE m.experiment_id = e.id
          ),
          ARRAY[]::TEXT[]
        ) as available_metrics,
        COALESCE(
          (
            SELECT jsonb_agg(
              jsonb_build_object(
                'name', latest_metrics.name,
                'value', CASE
                  WHEN latest_metrics.value % 1 = 0 THEN latest_metrics.value::INTEGER
                  ELSE ROUND(latest_metrics.value::NUMERIC, 4)
                END
              )
              ORDER BY latest_metrics.created_at DESC
            )
            FROM (
              SELECT DISTINCT ON (m.name)
                m.name,
                m.value,
                m.created_at
              FROM metric m
              WHERE m.experiment_id = e.id
              ORDER BY m.name, m.created_at DESC
              LIMIT 2
            ) latest_metrics
          ),
          '[]'::JSONB
        ) as key_metrics
      FROM experiment e
      LEFT JOIN user_experiments ue ON e.id = ue.experiment_id AND ue.user_id = user_id_param
      WHERE e.name ILIKE '%' || name_filter || '%'
        AND (e.visibility = 'PUBLIC' OR ue.user_id = user_id_param)
      ORDER BY e.created_at DESC;

    ELSE
      -- Anonymous user - only public experiments
      RETURN QUERY
      SELECT
        e.id,
        e.created_at,
        e.name,
        e.description,
        e.hyperparams,
        e.tags,
        e.visibility::TEXT,
        ue.user_id,
        COALESCE(
          ARRAY(
            SELECT DISTINCT m.name
            FROM metric m
            WHERE m.experiment_id = e.id
          ),
          ARRAY[]::TEXT[]
        ) as available_metrics,
        COALESCE(
          (
            SELECT jsonb_agg(
              jsonb_build_object(
                'name', latest_metrics.name,
                'value', CASE
                  WHEN latest_metrics.value % 1 = 0 THEN latest_metrics.value::INTEGER
                  ELSE ROUND(latest_metrics.value::NUMERIC, 4)
                END
              )
              ORDER BY latest_metrics.created_at DESC
            )
            FROM (
              SELECT DISTINCT ON (m.name)
                m.name,
                m.value,
                m.created_at
              FROM metric m
              WHERE m.experiment_id = e.id
              ORDER BY m.name, m.created_at DESC
              LIMIT 2
            ) latest_metrics
          ),
          '[]'::JSONB
        ) as key_metrics
      FROM experiment e
      LEFT JOIN user_experiments ue ON e.id = ue.experiment_id
      WHERE e.visibility = 'PUBLIC'
        AND e.name ILIKE '%' || name_filter || '%'
      ORDER BY e.created_at DESC;
    END IF;
  END;$$;


ALTER FUNCTION "public"."get_workspace_experiments"("name_filter" "text", "user_id_param" "text", "workspace_id_param" "uuid") OWNER TO "postgres";


CREATE OR REPLACE FUNCTION "public"."set_updated_at"() RETURNS "trigger"
    LANGUAGE "plpgsql"
    AS $$
BEGIN
  NEW.updated_at = now();
  RETURN NEW;
END;
$$;


ALTER FUNCTION "public"."set_updated_at"() OWNER TO "postgres";

SET default_tablespace = '';

SET default_table_access_method = "heap";


CREATE TABLE IF NOT EXISTS "public"."api_keys" (
    "id" "uuid" DEFAULT "gen_random_uuid"() NOT NULL,
    "user_id" "uuid",
    "name" "text" NOT NULL,
    "key_hash" "text" NOT NULL,
    "created_at" timestamp with time zone DEFAULT "now"() NOT NULL,
    "last_used" timestamp with time zone DEFAULT "now"() NOT NULL,
    "revoked" boolean DEFAULT false NOT NULL
);


ALTER TABLE "public"."api_keys" OWNER TO "postgres";


CREATE TABLE IF NOT EXISTS "public"."experiment" (
    "id" "uuid" DEFAULT "gen_random_uuid"() NOT NULL,
    "created_at" timestamp with time zone DEFAULT "now"() NOT NULL,
    "name" "text" NOT NULL,
    "description" "text",
    "hyperparams" "jsonb"[],
    "tags" "text"[],
    "visibility" "public"."visibility" DEFAULT 'PRIVATE'::"public"."visibility" NOT NULL,
    "updated_at" timestamp with time zone DEFAULT "now"() NOT NULL
);


ALTER TABLE "public"."experiment" OWNER TO "postgres";


CREATE TABLE IF NOT EXISTS "public"."experiment_references" (
    "id" integer NOT NULL,
    "from_experiment" "uuid" NOT NULL,
    "to_experiment" "uuid" NOT NULL,
    "created_at" timestamp without time zone DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT "experiment_references_check" CHECK (("from_experiment" <> "to_experiment"))
);


ALTER TABLE "public"."experiment_references" OWNER TO "postgres";


CREATE SEQUENCE IF NOT EXISTS "public"."experiment_references_id_seq"
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


ALTER TABLE "public"."experiment_references_id_seq" OWNER TO "postgres";


ALTER SEQUENCE "public"."experiment_references_id_seq" OWNED BY "public"."experiment_references"."id";



CREATE TABLE IF NOT EXISTS "public"."metric" (
    "id" bigint NOT NULL,
    "experiment_id" "uuid",
    "created_at" timestamp with time zone DEFAULT "now"() NOT NULL,
    "name" "text" NOT NULL,
    "value" numeric NOT NULL,
    "step" numeric,
    "metadata" "jsonb"
);


ALTER TABLE "public"."metric" OWNER TO "postgres";


ALTER TABLE "public"."metric" ALTER COLUMN "id" ADD GENERATED BY DEFAULT AS IDENTITY (
    SEQUENCE NAME "public"."metric_id_seq"
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1
);



CREATE TABLE IF NOT EXISTS "public"."user_experiments" (
    "user_id" "uuid" NOT NULL,
    "experiment_id" "uuid" NOT NULL,
    "role" "public"."user_experiment_role" DEFAULT 'VIEWER'::"public"."user_experiment_role" NOT NULL,
    "added_at" timestamp with time zone DEFAULT "now"() NOT NULL
);


ALTER TABLE "public"."user_experiments" OWNER TO "postgres";


CREATE TABLE IF NOT EXISTS "public"."user_workspaces" (
    "id" "uuid" DEFAULT "gen_random_uuid"() NOT NULL,
    "user_id" "uuid" NOT NULL,
    "workspace_id" "uuid" NOT NULL,
    "role_id" "uuid" NOT NULL,
    "created_at" timestamp with time zone DEFAULT "now"()
);


ALTER TABLE "public"."user_workspaces" OWNER TO "postgres";


CREATE TABLE IF NOT EXISTS "public"."workspace" (
    "id" "uuid" DEFAULT "gen_random_uuid"() NOT NULL,
    "name" "text" NOT NULL,
    "description" "text",
    "created_at" timestamp with time zone DEFAULT "now"() NOT NULL,
    "user_id" "uuid" NOT NULL
);


ALTER TABLE "public"."workspace" OWNER TO "postgres";


COMMENT ON TABLE "public"."workspace" IS 'a container for related experiments, etc.';



CREATE TABLE IF NOT EXISTS "public"."workspace_experiments" (
    "workspace_id" "uuid" NOT NULL,
    "experiment_id" "uuid" NOT NULL
);


ALTER TABLE "public"."workspace_experiments" OWNER TO "postgres";


CREATE TABLE IF NOT EXISTS "public"."workspace_role" (
    "id" "uuid" DEFAULT "gen_random_uuid"() NOT NULL,
    "name" "text" NOT NULL
);


ALTER TABLE "public"."workspace_role" OWNER TO "postgres";


ALTER TABLE ONLY "public"."experiment_references" ALTER COLUMN "id" SET DEFAULT "nextval"('"public"."experiment_references_id_seq"'::"regclass");



ALTER TABLE ONLY "public"."api_keys"
    ADD CONSTRAINT "api_keys_key_hash_key" UNIQUE ("key_hash");



ALTER TABLE ONLY "public"."api_keys"
    ADD CONSTRAINT "api_keys_pkey" PRIMARY KEY ("id");



ALTER TABLE ONLY "public"."experiment"
    ADD CONSTRAINT "experiment_pkey" PRIMARY KEY ("id");



ALTER TABLE ONLY "public"."experiment_references"
    ADD CONSTRAINT "experiment_references_from_experiment_to_experiment_key" UNIQUE ("from_experiment", "to_experiment");



ALTER TABLE ONLY "public"."experiment_references"
    ADD CONSTRAINT "experiment_references_pkey" PRIMARY KEY ("id");



ALTER TABLE ONLY "public"."metric"
    ADD CONSTRAINT "metric_pkey" PRIMARY KEY ("id");



ALTER TABLE ONLY "public"."user_experiments"
    ADD CONSTRAINT "user_experiments_pkey" PRIMARY KEY ("user_id", "experiment_id");



ALTER TABLE ONLY "public"."user_workspaces"
    ADD CONSTRAINT "user_workspaces_pkey" PRIMARY KEY ("id", "user_id", "workspace_id", "role_id");



ALTER TABLE ONLY "public"."workspace_experiments"
    ADD CONSTRAINT "workspace_experiments_pkey" PRIMARY KEY ("workspace_id", "experiment_id");



ALTER TABLE ONLY "public"."workspace"
    ADD CONSTRAINT "workspace_pkey" PRIMARY KEY ("id");



ALTER TABLE ONLY "public"."workspace_role"
    ADD CONSTRAINT "workspace_role_name_key" UNIQUE ("name");



ALTER TABLE ONLY "public"."workspace_role"
    ADD CONSTRAINT "workspace_role_pkey" PRIMARY KEY ("id");



CREATE INDEX "idx_api_keys_user_id" ON "public"."api_keys" USING "btree" ("user_id");



CREATE INDEX "idx_experiment_name_trgm" ON "public"."experiment" USING "gin" ("name" "public"."gin_trgm_ops");



CREATE INDEX "idx_experiment_references_from" ON "public"."experiment_references" USING "btree" ("from_experiment");



CREATE INDEX "idx_experiment_references_from_experiment" ON "public"."experiment_references" USING "btree" ("from_experiment");



CREATE INDEX "idx_experiment_references_to" ON "public"."experiment_references" USING "btree" ("to_experiment");



CREATE INDEX "idx_experiment_references_to_experiment" ON "public"."experiment_references" USING "btree" ("to_experiment");



CREATE INDEX "idx_experiment_tags" ON "public"."experiment" USING "gin" ("tags");



CREATE INDEX "idx_experiment_visibility" ON "public"."experiment" USING "btree" ("visibility");



CREATE INDEX "idx_metric_experiment_id" ON "public"."metric" USING "btree" ("experiment_id");



CREATE INDEX "idx_metric_experiment_name_created" ON "public"."metric" USING "btree" ("experiment_id", "name", "created_at");



CREATE INDEX "idx_user_experiments_experiment_id" ON "public"."user_experiments" USING "btree" ("experiment_id");



CREATE INDEX "idx_user_experiments_user_experiment_role" ON "public"."user_experiments" USING "btree" ("user_id", "experiment_id", "role");



CREATE INDEX "idx_user_experiments_user_id" ON "public"."user_experiments" USING "btree" ("user_id");



CREATE INDEX "idx_workspace_experiments_experiment_id" ON "public"."workspace_experiments" USING "btree" ("experiment_id");



CREATE INDEX "idx_workspace_experiments_workspace_id" ON "public"."workspace_experiments" USING "btree" ("workspace_id");



CREATE INDEX "idx_workspace_user_id" ON "public"."workspace" USING "btree" ("user_id");



CREATE OR REPLACE TRIGGER "trigger_set_updated_at" BEFORE UPDATE ON "public"."experiment" FOR EACH ROW EXECUTE FUNCTION "public"."set_updated_at"();



ALTER TABLE ONLY "public"."api_keys"
    ADD CONSTRAINT "api_keys_user_id_fkey" FOREIGN KEY ("user_id") REFERENCES "auth"."users"("id") ON DELETE CASCADE;



ALTER TABLE ONLY "public"."experiment_references"
    ADD CONSTRAINT "experiment_references_to_experiment_fkey" FOREIGN KEY ("to_experiment") REFERENCES "public"."experiment"("id") ON UPDATE CASCADE ON DELETE CASCADE;



ALTER TABLE ONLY "public"."experiment_references"
    ADD CONSTRAINT "experiment_references_to_experiment_fkey1" FOREIGN KEY ("to_experiment") REFERENCES "public"."experiment"("id") ON UPDATE CASCADE ON DELETE CASCADE;



ALTER TABLE ONLY "public"."user_experiments"
    ADD CONSTRAINT "fk_experiment" FOREIGN KEY ("experiment_id") REFERENCES "public"."experiment"("id") ON DELETE CASCADE;



ALTER TABLE ONLY "public"."user_experiments"
    ADD CONSTRAINT "fk_user" FOREIGN KEY ("user_id") REFERENCES "auth"."users"("id") ON DELETE CASCADE;



ALTER TABLE ONLY "public"."metric"
    ADD CONSTRAINT "metric_experiment_id_fkey" FOREIGN KEY ("experiment_id") REFERENCES "public"."experiment"("id") ON UPDATE CASCADE ON DELETE CASCADE;



ALTER TABLE ONLY "public"."user_workspaces"
    ADD CONSTRAINT "user_workspaces_role_id_fkey" FOREIGN KEY ("role_id") REFERENCES "public"."workspace_role"("id") ON DELETE CASCADE;



ALTER TABLE ONLY "public"."user_workspaces"
    ADD CONSTRAINT "user_workspaces_user_id_fkey" FOREIGN KEY ("user_id") REFERENCES "auth"."users"("id") ON DELETE CASCADE;



ALTER TABLE ONLY "public"."user_workspaces"
    ADD CONSTRAINT "user_workspaces_workspace_id_fkey" FOREIGN KEY ("workspace_id") REFERENCES "public"."workspace"("id") ON DELETE CASCADE;



ALTER TABLE ONLY "public"."workspace_experiments"
    ADD CONSTRAINT "workspace_experiments_experiment_id_fkey" FOREIGN KEY ("experiment_id") REFERENCES "public"."experiment"("id") ON DELETE CASCADE;



ALTER TABLE ONLY "public"."workspace_experiments"
    ADD CONSTRAINT "workspace_experiments_workspace_id_fkey" FOREIGN KEY ("workspace_id") REFERENCES "public"."workspace"("id") ON DELETE CASCADE;



ALTER TABLE ONLY "public"."workspace"
    ADD CONSTRAINT "workspace_user_id_fkey" FOREIGN KEY ("user_id") REFERENCES "auth"."users"("id") ON DELETE CASCADE;



CREATE POLICY "all" ON "public"."api_keys" TO "authenticated", "anon" USING (true);



CREATE POLICY "all" ON "public"."experiment" TO "authenticated", "anon" USING (true);



CREATE POLICY "all" ON "public"."experiment_references" TO "authenticated", "anon" USING (true);



CREATE POLICY "all" ON "public"."metric" TO "authenticated", "anon" USING (true);



CREATE POLICY "all" ON "public"."user_experiments" TO "authenticated", "anon" USING (true);



CREATE POLICY "all" ON "public"."workspace" TO "authenticated", "anon" USING (true);



CREATE POLICY "all" ON "public"."workspace_experiments" TO "authenticated", "anon" USING (true);



ALTER TABLE "public"."api_keys" ENABLE ROW LEVEL SECURITY;


ALTER TABLE "public"."experiment" ENABLE ROW LEVEL SECURITY;


ALTER TABLE "public"."experiment_references" ENABLE ROW LEVEL SECURITY;


ALTER TABLE "public"."metric" ENABLE ROW LEVEL SECURITY;


ALTER TABLE "public"."user_experiments" ENABLE ROW LEVEL SECURITY;


ALTER TABLE "public"."user_workspaces" ENABLE ROW LEVEL SECURITY;


ALTER TABLE "public"."workspace" ENABLE ROW LEVEL SECURITY;


ALTER TABLE "public"."workspace_experiments" ENABLE ROW LEVEL SECURITY;


ALTER TABLE "public"."workspace_role" ENABLE ROW LEVEL SECURITY;




ALTER PUBLICATION "supabase_realtime" OWNER TO "postgres";


GRANT USAGE ON SCHEMA "public" TO "postgres";
GRANT USAGE ON SCHEMA "public" TO "anon";
GRANT USAGE ON SCHEMA "public" TO "authenticated";
GRANT USAGE ON SCHEMA "public" TO "service_role";



GRANT ALL ON FUNCTION "public"."gtrgm_in"("cstring") TO "postgres";
GRANT ALL ON FUNCTION "public"."gtrgm_in"("cstring") TO "anon";
GRANT ALL ON FUNCTION "public"."gtrgm_in"("cstring") TO "authenticated";
GRANT ALL ON FUNCTION "public"."gtrgm_in"("cstring") TO "service_role";



GRANT ALL ON FUNCTION "public"."gtrgm_out"("public"."gtrgm") TO "postgres";
GRANT ALL ON FUNCTION "public"."gtrgm_out"("public"."gtrgm") TO "anon";
GRANT ALL ON FUNCTION "public"."gtrgm_out"("public"."gtrgm") TO "authenticated";
GRANT ALL ON FUNCTION "public"."gtrgm_out"("public"."gtrgm") TO "service_role";




















































































































































































GRANT ALL ON FUNCTION "public"."check_experiment_access"("experiment_id" "uuid", "user_id" "uuid") TO "anon";
GRANT ALL ON FUNCTION "public"."check_experiment_access"("experiment_id" "uuid", "user_id" "uuid") TO "authenticated";
GRANT ALL ON FUNCTION "public"."check_experiment_access"("experiment_id" "uuid", "user_id" "uuid") TO "service_role";



GRANT ALL ON FUNCTION "public"."get_experiment_chain"("target_experiment_id" "uuid") TO "anon";
GRANT ALL ON FUNCTION "public"."get_experiment_chain"("target_experiment_id" "uuid") TO "authenticated";
GRANT ALL ON FUNCTION "public"."get_experiment_chain"("target_experiment_id" "uuid") TO "service_role";



GRANT ALL ON FUNCTION "public"."get_experiments_and_metrics"("experiment_ids" "uuid"[]) TO "anon";
GRANT ALL ON FUNCTION "public"."get_experiments_and_metrics"("experiment_ids" "uuid"[]) TO "authenticated";
GRANT ALL ON FUNCTION "public"."get_experiments_and_metrics"("experiment_ids" "uuid"[]) TO "service_role";



GRANT ALL ON FUNCTION "public"."get_experiments_with_metric_names"("experiment_ids" "uuid"[]) TO "anon";
GRANT ALL ON FUNCTION "public"."get_experiments_with_metric_names"("experiment_ids" "uuid"[]) TO "authenticated";
GRANT ALL ON FUNCTION "public"."get_experiments_with_metric_names"("experiment_ids" "uuid"[]) TO "service_role";



GRANT ALL ON FUNCTION "public"."get_user_experiments"("user_id" "uuid", "workspace_id" "uuid") TO "anon";
GRANT ALL ON FUNCTION "public"."get_user_experiments"("user_id" "uuid", "workspace_id" "uuid") TO "authenticated";
GRANT ALL ON FUNCTION "public"."get_user_experiments"("user_id" "uuid", "workspace_id" "uuid") TO "service_role";



GRANT ALL ON FUNCTION "public"."get_workspace_experiments"("name_filter" "text", "user_id_param" "text", "workspace_id_param" "uuid") TO "anon";
GRANT ALL ON FUNCTION "public"."get_workspace_experiments"("name_filter" "text", "user_id_param" "text", "workspace_id_param" "uuid") TO "authenticated";
GRANT ALL ON FUNCTION "public"."get_workspace_experiments"("name_filter" "text", "user_id_param" "text", "workspace_id_param" "uuid") TO "service_role";



GRANT ALL ON FUNCTION "public"."gin_extract_query_trgm"("text", "internal", smallint, "internal", "internal", "internal", "internal") TO "postgres";
GRANT ALL ON FUNCTION "public"."gin_extract_query_trgm"("text", "internal", smallint, "internal", "internal", "internal", "internal") TO "anon";
GRANT ALL ON FUNCTION "public"."gin_extract_query_trgm"("text", "internal", smallint, "internal", "internal", "internal", "internal") TO "authenticated";
GRANT ALL ON FUNCTION "public"."gin_extract_query_trgm"("text", "internal", smallint, "internal", "internal", "internal", "internal") TO "service_role";



GRANT ALL ON FUNCTION "public"."gin_extract_value_trgm"("text", "internal") TO "postgres";
GRANT ALL ON FUNCTION "public"."gin_extract_value_trgm"("text", "internal") TO "anon";
GRANT ALL ON FUNCTION "public"."gin_extract_value_trgm"("text", "internal") TO "authenticated";
GRANT ALL ON FUNCTION "public"."gin_extract_value_trgm"("text", "internal") TO "service_role";



GRANT ALL ON FUNCTION "public"."gin_trgm_consistent"("internal", smallint, "text", integer, "internal", "internal", "internal", "internal") TO "postgres";
GRANT ALL ON FUNCTION "public"."gin_trgm_consistent"("internal", smallint, "text", integer, "internal", "internal", "internal", "internal") TO "anon";
GRANT ALL ON FUNCTION "public"."gin_trgm_consistent"("internal", smallint, "text", integer, "internal", "internal", "internal", "internal") TO "authenticated";
GRANT ALL ON FUNCTION "public"."gin_trgm_consistent"("internal", smallint, "text", integer, "internal", "internal", "internal", "internal") TO "service_role";



GRANT ALL ON FUNCTION "public"."gin_trgm_triconsistent"("internal", smallint, "text", integer, "internal", "internal", "internal") TO "postgres";
GRANT ALL ON FUNCTION "public"."gin_trgm_triconsistent"("internal", smallint, "text", integer, "internal", "internal", "internal") TO "anon";
GRANT ALL ON FUNCTION "public"."gin_trgm_triconsistent"("internal", smallint, "text", integer, "internal", "internal", "internal") TO "authenticated";
GRANT ALL ON FUNCTION "public"."gin_trgm_triconsistent"("internal", smallint, "text", integer, "internal", "internal", "internal") TO "service_role";



GRANT ALL ON FUNCTION "public"."gtrgm_compress"("internal") TO "postgres";
GRANT ALL ON FUNCTION "public"."gtrgm_compress"("internal") TO "anon";
GRANT ALL ON FUNCTION "public"."gtrgm_compress"("internal") TO "authenticated";
GRANT ALL ON FUNCTION "public"."gtrgm_compress"("internal") TO "service_role";



GRANT ALL ON FUNCTION "public"."gtrgm_consistent"("internal", "text", smallint, "oid", "internal") TO "postgres";
GRANT ALL ON FUNCTION "public"."gtrgm_consistent"("internal", "text", smallint, "oid", "internal") TO "anon";
GRANT ALL ON FUNCTION "public"."gtrgm_consistent"("internal", "text", smallint, "oid", "internal") TO "authenticated";
GRANT ALL ON FUNCTION "public"."gtrgm_consistent"("internal", "text", smallint, "oid", "internal") TO "service_role";



GRANT ALL ON FUNCTION "public"."gtrgm_decompress"("internal") TO "postgres";
GRANT ALL ON FUNCTION "public"."gtrgm_decompress"("internal") TO "anon";
GRANT ALL ON FUNCTION "public"."gtrgm_decompress"("internal") TO "authenticated";
GRANT ALL ON FUNCTION "public"."gtrgm_decompress"("internal") TO "service_role";



GRANT ALL ON FUNCTION "public"."gtrgm_distance"("internal", "text", smallint, "oid", "internal") TO "postgres";
GRANT ALL ON FUNCTION "public"."gtrgm_distance"("internal", "text", smallint, "oid", "internal") TO "anon";
GRANT ALL ON FUNCTION "public"."gtrgm_distance"("internal", "text", smallint, "oid", "internal") TO "authenticated";
GRANT ALL ON FUNCTION "public"."gtrgm_distance"("internal", "text", smallint, "oid", "internal") TO "service_role";



GRANT ALL ON FUNCTION "public"."gtrgm_options"("internal") TO "postgres";
GRANT ALL ON FUNCTION "public"."gtrgm_options"("internal") TO "anon";
GRANT ALL ON FUNCTION "public"."gtrgm_options"("internal") TO "authenticated";
GRANT ALL ON FUNCTION "public"."gtrgm_options"("internal") TO "service_role";



GRANT ALL ON FUNCTION "public"."gtrgm_penalty"("internal", "internal", "internal") TO "postgres";
GRANT ALL ON FUNCTION "public"."gtrgm_penalty"("internal", "internal", "internal") TO "anon";
GRANT ALL ON FUNCTION "public"."gtrgm_penalty"("internal", "internal", "internal") TO "authenticated";
GRANT ALL ON FUNCTION "public"."gtrgm_penalty"("internal", "internal", "internal") TO "service_role";



GRANT ALL ON FUNCTION "public"."gtrgm_picksplit"("internal", "internal") TO "postgres";
GRANT ALL ON FUNCTION "public"."gtrgm_picksplit"("internal", "internal") TO "anon";
GRANT ALL ON FUNCTION "public"."gtrgm_picksplit"("internal", "internal") TO "authenticated";
GRANT ALL ON FUNCTION "public"."gtrgm_picksplit"("internal", "internal") TO "service_role";



GRANT ALL ON FUNCTION "public"."gtrgm_same"("public"."gtrgm", "public"."gtrgm", "internal") TO "postgres";
GRANT ALL ON FUNCTION "public"."gtrgm_same"("public"."gtrgm", "public"."gtrgm", "internal") TO "anon";
GRANT ALL ON FUNCTION "public"."gtrgm_same"("public"."gtrgm", "public"."gtrgm", "internal") TO "authenticated";
GRANT ALL ON FUNCTION "public"."gtrgm_same"("public"."gtrgm", "public"."gtrgm", "internal") TO "service_role";



GRANT ALL ON FUNCTION "public"."gtrgm_union"("internal", "internal") TO "postgres";
GRANT ALL ON FUNCTION "public"."gtrgm_union"("internal", "internal") TO "anon";
GRANT ALL ON FUNCTION "public"."gtrgm_union"("internal", "internal") TO "authenticated";
GRANT ALL ON FUNCTION "public"."gtrgm_union"("internal", "internal") TO "service_role";



GRANT ALL ON FUNCTION "public"."set_limit"(real) TO "postgres";
GRANT ALL ON FUNCTION "public"."set_limit"(real) TO "anon";
GRANT ALL ON FUNCTION "public"."set_limit"(real) TO "authenticated";
GRANT ALL ON FUNCTION "public"."set_limit"(real) TO "service_role";



GRANT ALL ON FUNCTION "public"."set_updated_at"() TO "anon";
GRANT ALL ON FUNCTION "public"."set_updated_at"() TO "authenticated";
GRANT ALL ON FUNCTION "public"."set_updated_at"() TO "service_role";



GRANT ALL ON FUNCTION "public"."show_limit"() TO "postgres";
GRANT ALL ON FUNCTION "public"."show_limit"() TO "anon";
GRANT ALL ON FUNCTION "public"."show_limit"() TO "authenticated";
GRANT ALL ON FUNCTION "public"."show_limit"() TO "service_role";



GRANT ALL ON FUNCTION "public"."show_trgm"("text") TO "postgres";
GRANT ALL ON FUNCTION "public"."show_trgm"("text") TO "anon";
GRANT ALL ON FUNCTION "public"."show_trgm"("text") TO "authenticated";
GRANT ALL ON FUNCTION "public"."show_trgm"("text") TO "service_role";



GRANT ALL ON FUNCTION "public"."similarity"("text", "text") TO "postgres";
GRANT ALL ON FUNCTION "public"."similarity"("text", "text") TO "anon";
GRANT ALL ON FUNCTION "public"."similarity"("text", "text") TO "authenticated";
GRANT ALL ON FUNCTION "public"."similarity"("text", "text") TO "service_role";



GRANT ALL ON FUNCTION "public"."similarity_dist"("text", "text") TO "postgres";
GRANT ALL ON FUNCTION "public"."similarity_dist"("text", "text") TO "anon";
GRANT ALL ON FUNCTION "public"."similarity_dist"("text", "text") TO "authenticated";
GRANT ALL ON FUNCTION "public"."similarity_dist"("text", "text") TO "service_role";



GRANT ALL ON FUNCTION "public"."similarity_op"("text", "text") TO "postgres";
GRANT ALL ON FUNCTION "public"."similarity_op"("text", "text") TO "anon";
GRANT ALL ON FUNCTION "public"."similarity_op"("text", "text") TO "authenticated";
GRANT ALL ON FUNCTION "public"."similarity_op"("text", "text") TO "service_role";



GRANT ALL ON FUNCTION "public"."strict_word_similarity"("text", "text") TO "postgres";
GRANT ALL ON FUNCTION "public"."strict_word_similarity"("text", "text") TO "anon";
GRANT ALL ON FUNCTION "public"."strict_word_similarity"("text", "text") TO "authenticated";
GRANT ALL ON FUNCTION "public"."strict_word_similarity"("text", "text") TO "service_role";



GRANT ALL ON FUNCTION "public"."strict_word_similarity_commutator_op"("text", "text") TO "postgres";
GRANT ALL ON FUNCTION "public"."strict_word_similarity_commutator_op"("text", "text") TO "anon";
GRANT ALL ON FUNCTION "public"."strict_word_similarity_commutator_op"("text", "text") TO "authenticated";
GRANT ALL ON FUNCTION "public"."strict_word_similarity_commutator_op"("text", "text") TO "service_role";



GRANT ALL ON FUNCTION "public"."strict_word_similarity_dist_commutator_op"("text", "text") TO "postgres";
GRANT ALL ON FUNCTION "public"."strict_word_similarity_dist_commutator_op"("text", "text") TO "anon";
GRANT ALL ON FUNCTION "public"."strict_word_similarity_dist_commutator_op"("text", "text") TO "authenticated";
GRANT ALL ON FUNCTION "public"."strict_word_similarity_dist_commutator_op"("text", "text") TO "service_role";



GRANT ALL ON FUNCTION "public"."strict_word_similarity_dist_op"("text", "text") TO "postgres";
GRANT ALL ON FUNCTION "public"."strict_word_similarity_dist_op"("text", "text") TO "anon";
GRANT ALL ON FUNCTION "public"."strict_word_similarity_dist_op"("text", "text") TO "authenticated";
GRANT ALL ON FUNCTION "public"."strict_word_similarity_dist_op"("text", "text") TO "service_role";



GRANT ALL ON FUNCTION "public"."strict_word_similarity_op"("text", "text") TO "postgres";
GRANT ALL ON FUNCTION "public"."strict_word_similarity_op"("text", "text") TO "anon";
GRANT ALL ON FUNCTION "public"."strict_word_similarity_op"("text", "text") TO "authenticated";
GRANT ALL ON FUNCTION "public"."strict_word_similarity_op"("text", "text") TO "service_role";



GRANT ALL ON FUNCTION "public"."word_similarity"("text", "text") TO "postgres";
GRANT ALL ON FUNCTION "public"."word_similarity"("text", "text") TO "anon";
GRANT ALL ON FUNCTION "public"."word_similarity"("text", "text") TO "authenticated";
GRANT ALL ON FUNCTION "public"."word_similarity"("text", "text") TO "service_role";



GRANT ALL ON FUNCTION "public"."word_similarity_commutator_op"("text", "text") TO "postgres";
GRANT ALL ON FUNCTION "public"."word_similarity_commutator_op"("text", "text") TO "anon";
GRANT ALL ON FUNCTION "public"."word_similarity_commutator_op"("text", "text") TO "authenticated";
GRANT ALL ON FUNCTION "public"."word_similarity_commutator_op"("text", "text") TO "service_role";



GRANT ALL ON FUNCTION "public"."word_similarity_dist_commutator_op"("text", "text") TO "postgres";
GRANT ALL ON FUNCTION "public"."word_similarity_dist_commutator_op"("text", "text") TO "anon";
GRANT ALL ON FUNCTION "public"."word_similarity_dist_commutator_op"("text", "text") TO "authenticated";
GRANT ALL ON FUNCTION "public"."word_similarity_dist_commutator_op"("text", "text") TO "service_role";



GRANT ALL ON FUNCTION "public"."word_similarity_dist_op"("text", "text") TO "postgres";
GRANT ALL ON FUNCTION "public"."word_similarity_dist_op"("text", "text") TO "anon";
GRANT ALL ON FUNCTION "public"."word_similarity_dist_op"("text", "text") TO "authenticated";
GRANT ALL ON FUNCTION "public"."word_similarity_dist_op"("text", "text") TO "service_role";



GRANT ALL ON FUNCTION "public"."word_similarity_op"("text", "text") TO "postgres";
GRANT ALL ON FUNCTION "public"."word_similarity_op"("text", "text") TO "anon";
GRANT ALL ON FUNCTION "public"."word_similarity_op"("text", "text") TO "authenticated";
GRANT ALL ON FUNCTION "public"."word_similarity_op"("text", "text") TO "service_role";


















GRANT ALL ON TABLE "public"."api_keys" TO "anon";
GRANT ALL ON TABLE "public"."api_keys" TO "authenticated";
GRANT ALL ON TABLE "public"."api_keys" TO "service_role";



GRANT ALL ON TABLE "public"."experiment" TO "anon";
GRANT ALL ON TABLE "public"."experiment" TO "authenticated";
GRANT ALL ON TABLE "public"."experiment" TO "service_role";



GRANT ALL ON TABLE "public"."experiment_references" TO "anon";
GRANT ALL ON TABLE "public"."experiment_references" TO "authenticated";
GRANT ALL ON TABLE "public"."experiment_references" TO "service_role";



GRANT ALL ON SEQUENCE "public"."experiment_references_id_seq" TO "anon";
GRANT ALL ON SEQUENCE "public"."experiment_references_id_seq" TO "authenticated";
GRANT ALL ON SEQUENCE "public"."experiment_references_id_seq" TO "service_role";



GRANT ALL ON TABLE "public"."metric" TO "anon";
GRANT ALL ON TABLE "public"."metric" TO "authenticated";
GRANT ALL ON TABLE "public"."metric" TO "service_role";



GRANT ALL ON SEQUENCE "public"."metric_id_seq" TO "anon";
GRANT ALL ON SEQUENCE "public"."metric_id_seq" TO "authenticated";
GRANT ALL ON SEQUENCE "public"."metric_id_seq" TO "service_role";



GRANT ALL ON TABLE "public"."user_experiments" TO "anon";
GRANT ALL ON TABLE "public"."user_experiments" TO "authenticated";
GRANT ALL ON TABLE "public"."user_experiments" TO "service_role";



GRANT ALL ON TABLE "public"."user_workspaces" TO "anon";
GRANT ALL ON TABLE "public"."user_workspaces" TO "authenticated";
GRANT ALL ON TABLE "public"."user_workspaces" TO "service_role";



GRANT ALL ON TABLE "public"."workspace" TO "anon";
GRANT ALL ON TABLE "public"."workspace" TO "authenticated";
GRANT ALL ON TABLE "public"."workspace" TO "service_role";



GRANT ALL ON TABLE "public"."workspace_experiments" TO "anon";
GRANT ALL ON TABLE "public"."workspace_experiments" TO "authenticated";
GRANT ALL ON TABLE "public"."workspace_experiments" TO "service_role";



GRANT ALL ON TABLE "public"."workspace_role" TO "anon";
GRANT ALL ON TABLE "public"."workspace_role" TO "authenticated";
GRANT ALL ON TABLE "public"."workspace_role" TO "service_role";



ALTER DEFAULT PRIVILEGES FOR ROLE "postgres" IN SCHEMA "public" GRANT ALL ON SEQUENCES  TO "postgres";
ALTER DEFAULT PRIVILEGES FOR ROLE "postgres" IN SCHEMA "public" GRANT ALL ON SEQUENCES  TO "anon";
ALTER DEFAULT PRIVILEGES FOR ROLE "postgres" IN SCHEMA "public" GRANT ALL ON SEQUENCES  TO "authenticated";
ALTER DEFAULT PRIVILEGES FOR ROLE "postgres" IN SCHEMA "public" GRANT ALL ON SEQUENCES  TO "service_role";






ALTER DEFAULT PRIVILEGES FOR ROLE "postgres" IN SCHEMA "public" GRANT ALL ON FUNCTIONS  TO "postgres";
ALTER DEFAULT PRIVILEGES FOR ROLE "postgres" IN SCHEMA "public" GRANT ALL ON FUNCTIONS  TO "anon";
ALTER DEFAULT PRIVILEGES FOR ROLE "postgres" IN SCHEMA "public" GRANT ALL ON FUNCTIONS  TO "authenticated";
ALTER DEFAULT PRIVILEGES FOR ROLE "postgres" IN SCHEMA "public" GRANT ALL ON FUNCTIONS  TO "service_role";






ALTER DEFAULT PRIVILEGES FOR ROLE "postgres" IN SCHEMA "public" GRANT ALL ON TABLES  TO "postgres";
ALTER DEFAULT PRIVILEGES FOR ROLE "postgres" IN SCHEMA "public" GRANT ALL ON TABLES  TO "anon";
ALTER DEFAULT PRIVILEGES FOR ROLE "postgres" IN SCHEMA "public" GRANT ALL ON TABLES  TO "authenticated";
ALTER DEFAULT PRIVILEGES FOR ROLE "postgres" IN SCHEMA "public" GRANT ALL ON TABLES  TO "service_role";






























RESET ALL;
