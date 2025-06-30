#!/bin/bash

# Production build and serve script for Tora

echo "🏗️  Building Tora for Production"
echo "================================"

# Build SvelteKit app
echo "📦 Building SvelteKit application..."
cd web-new
pnpm run build:production
if [ $? -ne 0 ]; then
	echo "❌ SvelteKit build failed!"
	exit 1
fi
cd ..

echo "✅ SvelteKit build completed successfully!"

# Build and run Rust server
echo "🦀 Building and starting Rust server..."
cd api
cargo build --release
if [ $? -ne 0 ]; then
	echo "❌ Rust build failed!"
	exit 1
fi

echo "✅ Rust build completed successfully!"
echo ""
echo "🚀 Starting production server..."
echo "🌐 Available at: http://localhost:8080"
echo "🔧 API endpoints: http://localhost:8080/api/*"
echo ""
echo "Press Ctrl+C to stop the server"

# Run the production server
./target/release/api
