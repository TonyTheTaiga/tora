schema-up:
	supabase db pull --schema public,auth --password PZg5U2BPVZSQWMrt

schema-pull:
	supabase db dump --schema public --password PZg5U2BPVZSQWMrt -f supabase/schema.sql

dump:
	supabase db dump --data-only --schema public,auth --password PZg5U2BPVZSQWMrt -f supabase/seed.sql

reset-db:
	supabase db reset
