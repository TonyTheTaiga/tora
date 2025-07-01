#!/bin/bash

source .env

# Function to cleanup background processes
cleanup() {
	echo "🛑 Shutting down development servers..."
	kill $SVELTE_PID 2>/dev/null
	exit 0
}

# Set up signal handlers
trap cleanup SIGINT SIGTERM

# Start Rust API server
echo "🦀 Starting Rust API server..."
cd api
cargo run &
RUST_PID=$!
cd ..

echo ""
echo "✅ Development servers started!"
echo "🌐 http://localhost:8080"
echo ""

wait
