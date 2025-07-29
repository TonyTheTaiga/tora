update-dev-db:
	supabase db dump --data-only > seed.sql
	psql "postgresql://postgres:postgres@127.0.0.1:54322/postgres" -f seed.sql
