set -x
set -e

echo "Environment variables being passed to container:"
echo "SUPABASE_PASSWORD=$SUPABASE_PASSWORD"
echo "SUPABASE_URL=$SUPABASE_URL"
echo "SUPABASE_API_KEY=$SUPABASE_API_KEY"
echo "SUPABASE_JWT_SECRET=$SUPABASE_JWT_SECRET"
echo ""

docker --debug build -t tora .

docker run --rm -it -p 8080:8080 \
	-e SUPABASE_PASSWORD=$SUPABASE_PASSWORD \
	-e SUPABASE_URL=$SUPABASE_URL \
	-e SUPABASE_API_KEY=$SUPABASE_API_KEY \
	-e SUPABASE_JWT_SECRET=$SUPABASE_JWT_SECRET \
	tora
