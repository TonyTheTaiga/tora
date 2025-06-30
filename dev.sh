#!/bin/bash

source .env

# Development script for Tora Rust + Svelte stack

echo "🚀 Starting Tora Development Environment"
echo "========================================"

# Function to cleanup background processes
cleanup() {
	echo "🛑 Shutting down development servers..."
	kill $SVELTE_PID 2>/dev/null
	kill $RUST_PID 2>/dev/null
	exit 0
}

# Set up signal handlers
trap cleanup SIGINT SIGTERM

# Start SvelteKit dev server
echo "📦 Starting SvelteKit development server..."
cd web-new
pnpm run dev --port 5173 --host 0.0.0.0 &
SVELTE_PID=$!
cd ..

# Wait a moment for SvelteKit to start
sleep 2

# Start Rust API server
echo "🦀 Starting Rust API server..."
cd api
cargo run &
RUST_PID=$!
cd ..

echo ""
echo "✅ Development servers started!"
echo "📱 Frontend: http://localhost:5173"
echo "🔧 Backend:  http://localhost:8080"
echo "🌐 Full app: http://localhost:8080 (after building frontend)"
echo ""
echo "💡 Tips:"
echo "   - Frontend changes will hot-reload automatically"
echo "   - Backend changes require restart (Ctrl+C then ./dev.sh)"
echo "   - Run 'pnpm run build' in web-new/ to test production build"
echo ""
echo "Press Ctrl+C to stop all servers"

# Wait for background processes
wait
