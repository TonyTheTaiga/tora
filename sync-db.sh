#!/usr/bin/env bash
set -Eeuo pipefail

# --- Config (override via env vars or flags) ---
REMOTE_DB_URL="${SUPABASE_REMOTE_DB_URL:-postgresql://postgres:PZg5U2BPVZSQWMrt@db.apdjlrqohzgeekvakysc.supabase.co:5432/postgres}"  # e.g. postgresql://postgres:YOUR_PASSWORD@db.xxxxx.supabase.co:5432/postgres?sslmode=require
LOCAL_DB_URL="${LOCAL_DB_URL:-postgres://postgres:postgres@127.0.0.1:54322/postgres}"
SCHEMAS="${SCHEMAS:-public,auth,storage}"    # comma-separated list
EXCLUDE_TABLES="${EXCLUDE_TABLES:-}"         # optional comma-separated fqtns, e.g. "public.some_big_table,auth.audit_log"
CONFIRM="${CONFIRM:-}"                       # set to "yes" to skip prompt

usage() {
  cat <<USAGE
Usage: $0 [--remote <url>] [--local <url>] [--schemas "public,auth,storage"] [--exclude "schema.table,..."] [-y|--yes]

Environment alternatives:
  SUPABASE_REMOTE_DB_URL, LOCAL_DB_URL, SCHEMAS, EXCLUDE_TABLES, CONFIRM=yes

Default local URL targets 'supabase start' Postgres on port 54322.
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --remote) REMOTE_DB_URL="$2"; shift 2 ;;
    --local) LOCAL_DB_URL="$2"; shift 2 ;;
    --schemas) SCHEMAS="$2"; shift 2 ;;
    --exclude) EXCLUDE_TABLES="$2"; shift 2 ;;
    -y|--yes) CONFIRM="yes"; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown arg: $1"; usage; exit 1 ;;
  endcase
done

command -v pg_dump >/dev/null || { echo "pg_dump not found. Install PostgreSQL client tools."; exit 1; }
command -v pg_restore >/dev/null || { echo "pg_restore not found. Install PostgreSQL client tools."; exit 1; }
command -v psql >/dev/null || { echo "psql not found. Install PostgreSQL client tools."; exit 1; }

if [[ -z "$REMOTE_DB_URL" ]]; then
  echo "ERROR: Provide your cloud DB URL via --remote or SUPABASE_REMOTE_DB_URL."
  echo "       (Project Settings → Database → Connection string; include '?sslmode=require')"
  exit 1
fi

WORKDIR="$(mktemp -d -t supa-sync-XXXXXX)"
DUMP="$WORKDIR/data.dump"

echo "==> Dumping data from remote (${SCHEMAS})..."
schema_args=()
IFS=',' read -ra _schemas <<< "$SCHEMAS"
for s in "${_schemas[@]}"; do
  schema_args+=("-n" "$s")
done

exclude_args=()
if [[ -n "$EXCLUDE_TABLES" ]]; then
  IFS=',' read -ra _ex <<< "$EXCLUDE_TABLES"
  for t in "${_ex[@]}"; do
    exclude_args+=("--exclude-table-data=$t")
  done
fi

pg_dump \
  --format=custom \
  --no-owner --no-privileges \
  --data-only \
  "${schema_args[@]}" \
  "${exclude_args[@]}" \
  --file "$DUMP" \
  "$REMOTE_DB_URL"

echo "==> About to TRUNCATE local schemas [$SCHEMAS] and restore from dump into:"
echo "    $LOCAL_DB_URL"
if [[ "$CONFIRM" != "yes" ]]; then
  read -r -p "Proceed? This will OVERWRITE local data. [y/N] " ans
  [[ "${ans:-}" =~ ^[Yy]$ ]] || { echo "Aborted."; exit 1; }
fi

echo "==> Truncating local tables (RESTART IDENTITY, CASCADE)..."
TRUNCATE_SQL=$(cat <<'SQL'
DO $$
DECLARE
  rec RECORD;
BEGIN
  FOR rec IN
    SELECT format('%I.%I', schemaname, tablename) AS fqtn
    FROM pg_tables
    WHERE schemaname = ANY (string_to_array(current_setting('app.truncate_schemas'), ','))
  LOOP
    EXECUTE 'TRUNCATE TABLE '||rec.fqtn||' RESTART IDENTITY CASCADE';
  END LOOP;
END$$;
SQL
)

psql "$LOCAL_DB_URL" -v ON_ERROR_STOP=1 \
  -c "SET app.truncate_schemas TO '$SCHEMAS';" \
  -c "$TRUNCATE_SQL"

echo "==> Restoring data into local (disabling triggers during load)..."
pg_restore \
  --no-owner --no-privileges \
  --data-only \
  --disable-triggers \
  -d "$LOCAL_DB_URL" \
  "$DUMP"

echo "==> Vacuum/Analyze..."
psql "$LOCAL_DB_URL" -v ON_ERROR_STOP=1 -c "VACUUM ANALYZE;"

echo "✅ Done."
echo "   Dump file kept at: $DUMP"
echo "   Temp dir: $WORKDIR"
