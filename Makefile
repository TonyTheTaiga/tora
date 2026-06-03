schema-up:
	supabase db pull --schema public,auth --password $(SUPABASE_DB_PASSWORD)

schema-pull:
	supabase db dump --schema public --password $(SUPABASE_DB_PASSWORD) -f supabase/schema.sql

dump:
	supabase db dump --data-only --schema public,auth --password $(SUPABASE_DB_PASSWORD) -f supabase/seed.sql

reset-db:
	supabase db reset

db:
	psql postgresql://postgres:postgres@127.0.0.1:54322/postgres

.PHONY: docs-python
docs-python:
	uvx pdoc --docformat google --no-math --no-mermaid --no-search --no-show-source --favicon /favicon.svg --logo /favicon.svg --footer-text "Tora Python SDK" --output-directory ./web/static/api-docs python/tora
