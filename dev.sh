#!/bin/bash

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to cleanup processes
cleanup() {
	echo -e "\n${RED}🛑 Shutting down development servers...${NC}"
	if [ ! -z "$RUST_PID" ]; then
		kill $RUST_PID 2>/dev/null
		echo -e "${YELLOW}⚡ Rust API server stopped${NC}"
	fi
	if [ ! -z "$VITE_PID" ]; then
		kill $VITE_PID 2>/dev/null
		echo -e "${YELLOW}⚡ Vite dev server stopped${NC}"
	fi
	exit 0
}

# Trap cleanup function on script exit
trap cleanup SIGINT SIGTERM

echo -e "${BLUE}🚀 Starting Tora development with hot reloading...${NC}"

# Start Rust API server
echo -e "${GREEN}🦀 Starting Rust API server on http://localhost:8080${NC}"
cd api
STATIC_FILES_PATH="NULL" cargo watch -x run &
RUST_PID=$!
cd ..

# Wait a moment for the API server to start
sleep 2

# Start Vite dev server
echo -e "${GREEN}⚡ Starting Vite dev server on http://localhost:5173${NC}"
echo -e "${YELLOW}🔥 Hot reloading enabled - changes will update instantly!${NC}"
echo -e "${BLUE}🌐 Open http://localhost:5173 in your browser${NC}"
echo -e "${BLUE}📡 API requests will be proxied to http://localhost:8080${NC}"
echo ""

cd web
pnpm run dev &
VITE_PID=$!
cd ..

# Wait for both processes
wait $RUST_PID $VITE_PID
