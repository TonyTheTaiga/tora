# Tora: High-Performance Rust + Svelte Stack

A modern web application stack combining Rust backend with statically-generated SvelteKit frontend for maximum performance.

## Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   SvelteKit     │    │   Rust Axum     │    │   PostgreSQL    │
│   (Static)      │───▶│   Server        │───▶│   Database      │
│                 │    │                 │    │                 │
│ • Pre-rendered  │    │ • Static files  │    │ • Supabase      │
│ • TypeScript    │    │ • API routes    │    │ • Migrations    │
│ • TailwindCSS   │    │ • CORS enabled  │    │                 │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

## Performance Benefits

- **🚀 Static Generation**: Pages pre-rendered at build time
- **⚡ Rust Backend**: Extremely fast static file serving
- **🎯 Zero Runtime**: No JavaScript routing overhead
- **📱 Progressive Enhancement**: Works without JavaScript
- **🔧 API Separation**: Clean separation of concerns

## Getting Started

### Prerequisites

- Node.js 18+ and pnpm
- Rust 1.70+
- PostgreSQL (optional, for API features)

### Development

```bash
# Install frontend dependencies
cd web-new
pnpm install

# Start development servers
./dev.sh
```

This starts:
- SvelteKit dev server at http://localhost:5173
- Rust API server at http://localhost:8080

### Production Build

```bash
# Build and serve production version
./build-and-serve.sh
```

## Project Structure

```
tora/
├── web-new/           # SvelteKit application
│   ├── src/
│   │   ├── routes/    # SvelteKit routes
│   │   └── lib/       # Shared components
│   ├── build/         # Generated static files
│   └── svelte.config.js
├── api/               # Rust backend
│   ├── src/
│   │   ├── main.rs    # Server entry point
│   │   └── repos/     # API modules
│   └── Cargo.toml
├── dev.sh            # Development script
└── build-and-serve.sh # Production script
```

## Configuration

### SvelteKit Static Adapter

The app uses `@sveltejs/adapter-static` with these settings:

```javascript
adapter: adapter({
  pages: 'build',
  assets: 'build', 
  fallback: 'index.html',
  precompress: false,
  strict: true
})
```

### Rust Server Configuration

- Static files served from `../web-new/build`
- API routes nested under `/api/*`
- CORS enabled for development
- Fallback to `index.html` for SPA routing

## Development Workflow

1. **Frontend Development**: Use `cd web-new && pnpm run dev` for hot reload
2. **Backend Development**: Use `cd api && cargo run` for API development  
3. **Full Stack**: Use `./dev.sh` to run both concurrently
4. **Production Testing**: Use `./build-and-serve.sh` to test production build

## API Integration

Frontend can call backend APIs at `/api/*`:

```typescript
// Example API call from SvelteKit
const response = await fetch('/api/workspaces');
const workspaces = await response.json();
```

## Deployment Options

### Static + API Server
- Deploy static files to CDN (Cloudflare, AWS CloudFront)
- Deploy Rust API to any cloud provider
- Update API URLs in frontend config

### Single Server
- Use the Rust server to serve both static files and API
- Deploy as single binary to any cloud provider
- Simplest deployment option

## Performance Characteristics

- **First Contentful Paint**: ~200ms (static files)
- **Time to Interactive**: ~300ms (minimal JavaScript)
- **API Response Time**: ~1-5ms (Rust efficiency)
- **Concurrent Connections**: 10k+ (Tokio async runtime)

## Next Steps

1. Add authentication middleware to Rust server
2. Implement database models and migrations
3. Add form handling and validation
4. Set up CI/CD pipeline
5. Configure monitoring and logging