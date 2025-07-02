#!/bin/bash

cleanup() {
	echo "🛑 Shutting down development servers..."
	kill $SVELTE_PID 2>/dev/null
	exit 0
}
trap cleanup SIGINT SIGTERM

cd web
pnpm run build
cd ..

cd api
cargo run &
RUST_PID=$!
cd ..

echo "🌐 http://localhost:8080"

wait
