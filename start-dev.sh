#!/bin/bash

# Tora Development Server Startup Script
# This script starts both the Rust API and SvelteKit frontend in development mode
# with hot reloading for both services.

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_info() {
	echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
	echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
	echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
	echo -e "${RED}[ERROR]${NC} $1"
}

# Function to cleanup background processes
cleanup() {
	# Optional first arg: exit code (default 0)
	local code=${1:-0}
	print_info "Shutting down development servers..."
	if [ ! -z "$API_PID" ]; then
		kill $API_PID 2>/dev/null || true
	fi
	if [ ! -z "$WEB_PID" ]; then
		kill $WEB_PID 2>/dev/null || true
	fi
	# Kill any remaining processes
	pkill -f "cargo-watch" 2>/dev/null || true
	pkill -f "vite dev" 2>/dev/null || true
	if [ "$code" -eq 0 ]; then
		print_success "Development servers stopped"
	else
		print_error "Development servers stopped due to an error (exit $code)"
	fi
	exit $code
}

# Set up signal handlers
trap 'cleanup 0' SIGINT SIGTERM

# Wait helper for backend readiness
wait_for_backend() {
	local url="$1" # e.g., http://127.0.0.1:8080/
	local timeout="${2:-60}"
	local start_ts
	start_ts=$(date +%s)

	print_info "Waiting for backend at $url (timeout ${timeout}s)..."

	# Prefer curl if available, otherwise try nc; as a last resort, simple TCP check via bash
	while true; do
		# If cargo-watch died, abort early
		if [ -n "$API_PID" ] && ! kill -0 "$API_PID" 2>/dev/null; then
			print_error "Backend process exited unexpectedly. See logs above."
			return 1
		fi

		if command -v curl >/dev/null 2>&1; then
			# Any HTTP response (even 404) means server is responding
			if curl -sS -o /dev/null "$url"; then
				print_success "Backend is responding at $url"
				return 0
			fi
		elif command -v nc >/dev/null 2>&1; then
			# Try a quick TCP connect to the host:port
			local host port
			host=$(echo "$url" | sed -E 's#^https?://([^/:]+).*$#\1#')
			port=$(echo "$url" | sed -E 's#^https?://[^/:]+:([0-9]+).*$#\1#')
			if [ -n "$host" ] && [ -n "$port" ] && nc -z -w 1 "$host" "$port" 2>/dev/null; then
				print_success "Backend TCP port open at $host:$port"
				return 0
			fi
		else
			# Fallback: bash TCP check (may not be available everywhere)
			local host port
			host=$(echo "$url" | sed -E 's#^https?://([^/:]+).*$#\1#')
			port=$(echo "$url" | sed -E 's#^https?://[^/:]+:([0-9]+).*$#\1#')
			if bash -c "</dev/tcp/$host/$port" 2>/dev/null; then
				print_success "Backend TCP port open at $host:$port"
				return 0
			fi
		fi

		local now elapsed
		now=$(date +%s)
		elapsed=$((now - start_ts))
		if [ "$elapsed" -ge "$timeout" ]; then
			print_error "Timed out after ${timeout}s waiting for backend at $url"
			return 1
		fi
		sleep 1
	done
}

# Check if required tools are installed
print_info "Checking required tools..."

if ! command -v cargo &>/dev/null; then
	print_error "Cargo not found. Please install Rust."
	exit 1
fi

if ! command -v cargo-watch &>/dev/null; then
	print_error "cargo-watch not found. Please install with: cargo install cargo-watch"
	exit 1
fi

if ! command -v node &>/dev/null; then
	print_error "Node.js not found. Please install Node.js."
	exit 1
fi

# Check if pnpm is available, fallback to npm
if command -v pnpm &>/dev/null; then
	NPM_CMD="pnpm"
	print_info "Using pnpm for package management"
else
	NPM_CMD="npm"
	print_info "Using npm for package management"
fi

print_success "All required tools are available"

# Check if dependencies are installed
print_info "Checking dependencies..."

if [ ! -d "web/node_modules" ]; then
	print_warning "Web dependencies not found. Installing..."
	cd web
	$NPM_CMD install
	cd ..
	print_success "Web dependencies installed"
fi

print_success "Dependencies check complete"

# Environment setup
print_info "Setting up environment..."

# Check for environment variables (optional for dev mode)
if [ -z "$SUPABASE_URL" ] || [ -z "$SUPABASE_API_KEY" ] || [ -z "$SUPABASE_JWT_SECRET" ]; then
	print_warning "Some Supabase environment variables are not set."
	print_warning "The API will run in development mode without database connection."
	print_warning "Set the following environment variables if you need database access:"
	print_warning "  - SUPABASE_URL"
	print_warning "  - SUPABASE_API_KEY"
	print_warning "  - SUPABASE_JWT_SECRET"
fi

# Export environment variables for the API
# export RUST_LOG=api::handlers::metric=debug
export RUST_LOG=error
export RUST_BACKTRACE=1
export RUST_ENV=dev
export PUBLIC_API_BASE_URL=http://localhost:8080
export FRONTEND_URL=http://localhost:5173

print_success "Environment configured"

# Start the API server with hot reloading
print_info "Starting Rust API server on port 8080..."
cd api
cargo-watch -x 'run' &
API_PID=$!
cd ..

# Determine API URL to wait on
API_PORT="${PORT:-8080}"
API_HOST="127.0.0.1"
API_WAIT_PATH="${API_HEALTH_PATH:-/health}"
API_URL="http://${API_HOST}:${API_PORT}${API_WAIT_PATH}"

# Wait for the API to become responsive before starting the frontend
if ! wait_for_backend "$API_URL" "${API_STARTUP_TIMEOUT:-90}"; then
	print_error "Backend failed to start. Not launching frontend."
	cleanup 1
fi

# Start the web server with hot reloading
print_info "Starting SvelteKit web server on port 5173..."
cd web
$NPM_CMD run dev &
WEB_PID=$!
cd ..

# Give the web server a moment to start
sleep 3

print_success "Development servers started successfully!"
print_info "Services running:"
print_info "  - API (Rust/Axum): http://localhost:8080"
print_info "  - Web (SvelteKit): http://localhost:5173"
print_info ""
print_info "Both servers support hot reloading:"
print_info "  - API: Changes to Rust files will trigger recompilation"
print_info "  - Web: Changes to Svelte/TypeScript files will trigger hot module replacement"
print_info ""
print_info "Press Ctrl+C to stop both servers"

# Wait for background processes
wait $API_PID $WEB_PID
