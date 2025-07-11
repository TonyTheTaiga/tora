# Tora: ML Experiment Management Tool

**Tora** is a modern, lightweight ML experiment tracking platform designed for speed and simplicity. Built for ML engineers and data scientists who want to track experiments without complexity.

## 🚀 Quick Start

Track your first experiment in 2 lines of code:

```python
import tora
tora.setup(name="my-experiment", workspace_id="ws-123")
tora.tlog("accuracy", 0.95, step=100)
```

## 📋 Table of Contents

- [Architecture Overview](#architecture-overview)
- [Getting Started](#getting-started)
- [Development Setup](#development-setup)
- [Project Structure](#project-structure)
- [API Documentation](#api-documentation)
- [Deployment](#deployment)
- [Contributing](#contributing)
- [Troubleshooting](#troubleshooting)

## 🏗️ Architecture Overview

Tora consists of three main components:

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Python SDK    │───▶│   Rust API      │───▶│   PostgreSQL    │
│   (Client)      │    │   (Backend)     │    │   (Database)    │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                                │
                                ▼
                       ┌─────────────────┐
                       │  SvelteKit Web  │
                       │   (Frontend)    │
                       └─────────────────┘
```

### Components

- **Python SDK** (`/python`): Client library for experiment tracking
- **Rust API** (`/api`): High-performance backend API server
- **SvelteKit Web** (`/web`): Modern web dashboard for visualization
- **PostgreSQL**: Database for storing experiments and metrics

## 🚀 Getting Started

### Prerequisites

- **Python 3.11+** (for SDK development)
- **Rust 1.88+** (for API development)
- **Node.js 18+** and **pnpm** (for web development)
- **PostgreSQL** (for database)
- **Docker & Docker Compose** (for containerized development)

### Installation Options

#### Option 1: Using the Python SDK (End Users)

```bash
pip install tora
```

#### Option 2: Development Setup (Contributors)

```bash
git clone https://github.com/taigaishida/tora.git
cd tora
./start-dev.sh  # Starts all services in development mode
```

### Basic Usage

```python
import tora

# Method 1: Global API (simplest)
tora.setup(
    name="my-experiment",
    workspace_id="your-workspace-id",
    hyperparams={"lr": 0.001, "batch_size": 32}
)

# Log metrics anywhere in your code
tora.tlog("accuracy", 0.95, step=100)
tora.tlog("loss", 0.05, step=100)

# Method 2: Client API (more control)
with tora.Tora.create_experiment("my-experiment", workspace_id="ws-123") as client:
    client.log("metric", 1.0, step=1)
    # Automatically flushes on exit
```

## 🛠️ Development Setup

### Quick Start with Docker

```bash
# Clone the repository
git clone https://github.com/taigaishida/tora.git
cd tora

# Start all services
./start-dev.sh
```

This will start:
- **API Server**: http://localhost:8080
- **Web Dashboard**: http://localhost:5173
- **Hot reloading** for all components

### Manual Setup

#### 1. Database Setup

```bash
# Set up PostgreSQL (using Supabase or local instance)
export DATABASE_URL="postgresql://user:password@localhost:5432/tora"
```

#### 2. API Server (Rust)

```bash
cd api
cargo install cargo-watch  # For hot reloading
export RUST_LOG=debug
export DATABASE_URL="your-database-url"
cargo run
# Or with hot reloading: cargo-watch -x 'run'
```

#### 3. Web Dashboard (SvelteKit)

```bash
cd web
pnpm install
export PUBLIC_API_BASE_URL=http://localhost:8080
pnpm run dev
```

#### 4. Python SDK

```bash
cd python
pip install -e ".[dev]"  # Install in development mode
pytest  # Run tests
```

### Environment Variables

Create a `.env` file in the root directory:

```bash
# Database
DATABASE_URL=postgresql://user:password@localhost:5432/tora

# Supabase (if using)
SUPABASE_URL=your-supabase-url
SUPABASE_API_KEY=your-supabase-api-key
SUPABASE_JWT_SECRET=your-jwt-secret

# API Configuration
RUST_LOG=debug
RUST_ENV=dev
PUBLIC_API_BASE_URL=http://localhost:8080
FRONTEND_URL=http://localhost:5173

# Python SDK
TORA_API_KEY=your-api-key
TORA_BASE_URL=http://localhost:8080
```

## 📁 Project Structure

```
tora/
├── api/                    # Rust API Server
│   ├── src/
│   │   ├── handlers/       # API route handlers
│   │   ├── middleware/     # Authentication & CORS
│   │   ├── types.rs        # Data structures
│   │   └── main.rs         # Server entry point
│   ├── Cargo.toml          # Rust dependencies
│   ├── Dockerfile          # API container
│   └── schema.sql          # Database schema
│
├── python/                 # Python SDK
│   ├── tora/
│   │   ├── _client.py      # Main Tora client
│   │   ├── _wrapper.py     # Global API functions
│   │   ├── _exceptions.py  # Custom exceptions
│   │   ├── _types.py       # Type definitions
│   │   └── __init__.py     # Public API exports
│   ├── tests/              # Test suite
│   ├── examples/           # Usage examples
│   ├── pyproject.toml      # Python project config
│   └── README.md           # SDK documentation
│
├── web/                    # SvelteKit Frontend
│   ├── src/
│   │   ├── routes/         # Page routes & API proxies
│   │   ├── lib/            # Shared components & utilities
│   │   └── app.html        # HTML template
│   ├── package.json        # Node.js dependencies
│   ├── svelte.config.js    # SvelteKit configuration
│   └── Dockerfile          # Web container
│
├── docs/                   # Documentation
├── supabase/              # Database migrations & config
├── docker-compose.yaml    # Multi-service orchestration
├── start-dev.sh          # Development startup script
└── README.md             # This file
```

### Key Files

- **`api/src/main.rs`**: API server entry point
- **`python/tora/_client.py`**: Core Python client implementation
- **`web/src/routes/api/[...proxyPath]/+server.ts`**: API proxy for web dashboard
- **`start-dev.sh`**: Automated development environment setup

## 🔌 API Documentation

### Core Endpoints

#### Experiments
- `POST /api/experiments` - Create experiment
- `GET /api/experiments` - List experiments
- `GET /api/experiments/{id}` - Get experiment details
- `PUT /api/experiments/{id}` - Update experiment
- `DELETE /api/experiments/{id}` - Delete experiment

#### Metrics
- `POST /api/experiments/{id}/metrics/batch` - Batch log metrics
- `GET /api/experiments/{id}/metrics` - Get experiment metrics

#### Workspaces
- `POST /api/workspaces` - Create workspace
- `GET /api/workspaces` - List workspaces
- `GET /api/workspaces/{id}` - Get workspace details

#### Authentication
- `POST /api/keys` - Generate API key
- `GET /api/keys` - List API keys
- `DELETE /api/keys/{id}` - Revoke API key

### Python SDK API

#### Main Client

```python
# Create experiment
client = tora.Tora.create_experiment(
    name="experiment-name",
    workspace_id="ws-123",
    description="Optional description",
    hyperparams={"lr": 0.001, "batch_size": 32},
    tags=["pytorch", "cnn"],
    max_buffer_len=100  # Buffer size for metrics
)

# Load existing experiment
client = tora.Tora.load_experiment(experiment_id="exp-123")

# Log metrics
client.log("accuracy", 0.95, step=100, metadata={"epoch": 10})

# Cleanup
client.shutdown()
```

#### Global API

```python
# Setup global experiment
experiment_id = tora.setup(
    name="global-experiment",
    workspace_id="ws-123",
    hyperparams={"lr": 0.001}
)

# Log from anywhere
tora.tlog("loss", 0.05, step=50)

# Utilities
tora.flush()           # Force send buffered metrics
tora.is_initialized()  # Check if setup was called
tora.get_experiment_id()  # Get current experiment ID
tora.shutdown()        # Cleanup
```

## 🚢 Deployment

### Docker Compose (Recommended)

```bash
# Production deployment
docker-compose up -d

# With custom environment
cp .env.example .env
# Edit .env with your configuration
docker-compose --env-file .env up -d
```

### Individual Services

#### API Server
```bash
cd api
docker build -t tora-api .
docker run -p 8080:8080 \
  -e DATABASE_URL="your-db-url" \
  -e RUST_ENV=production \
  tora-api
```

#### Web Dashboard
```bash
cd web
docker build -t tora-web .
docker run -p 3000:3000 \
  -e PUBLIC_API_BASE_URL="https://your-api-domain.com" \
  tora-web
```

### Cloud Deployment

The project includes configuration for:
- **Google Cloud Run** (`cloudbuild.yaml`)
- **Docker containers** for any cloud provider
- **Supabase** for managed PostgreSQL

## 🧪 Testing

### Python SDK
```bash
cd python
pytest                    # Run all tests
pytest tests/test_client.py  # Run specific test file
pytest --cov=tora        # Run with coverage
```

### API Server
```bash
cd api
cargo test               # Run Rust tests
cargo test -- --nocapture  # With output
```

### Web Dashboard
```bash
cd web
pnpm test               # Run frontend tests (if configured)
pnpm run check          # Type checking
```

## 🤝 Contributing

### Development Workflow

1. **Fork & Clone**
   ```bash
   git clone https://github.com/your-username/tora.git
   cd tora
   ```

2. **Setup Development Environment**
   ```bash
   ./start-dev.sh
   ```

3. **Make Changes**
   - Follow existing code style
   - Add tests for new features
   - Update documentation

4. **Test Your Changes**
   ```bash
   # Test Python SDK
   cd python && pytest

   # Test API
   cd api && cargo test

   # Test Web
   cd web && pnpm run check
   ```

5. **Submit Pull Request**
   - Clear description of changes
   - Reference any related issues
   - Ensure CI passes

### Code Style

- **Python**: Uses Ruff for linting and formatting (configured in `pyproject.toml`)
- **Rust**: Uses `rustfmt` and `clippy`
- **TypeScript/Svelte**: Uses Prettier

### Pre-commit Hooks

```bash
cd python
pip install pre-commit
pre-commit install
```

## 🔧 Troubleshooting

### Common Issues

#### "Connection refused" when starting services

**Problem**: API server can't connect to database
```
Error: Connection refused (os error 61)
```

**Solution**:
1. Ensure PostgreSQL is running
2. Check `DATABASE_URL` environment variable
3. Verify database credentials and network access

#### Python SDK import errors

**Problem**: `ModuleNotFoundError: No module named 'tora'`

**Solution**:
```bash
cd python
pip install -e .  # Install in development mode
```

#### Web dashboard shows "API Error"

**Problem**: Frontend can't reach API server

**Solution**:
1. Check API server is running on port 8080
2. Verify `PUBLIC_API_BASE_URL` environment variable
3. Check CORS configuration in API server

#### Metrics not appearing in dashboard

**Problem**: Logged metrics don't show up

**Solution**:
1. Call `client.shutdown()` or `tora.flush()` to send buffered metrics
2. Check API server logs for errors
3. Verify experiment ID is correct

### Development Tips

#### Hot Reloading Setup
```bash
# API with hot reload
cd api && cargo install cargo-watch
cargo-watch -x 'run'

# Web with hot reload
cd web && pnpm run dev

# Python SDK development
cd python && pip install -e ".[dev]"
```

#### Database Reset
```bash
# Reset database schema
psql $DATABASE_URL -f api/schema.sql
```

#### Debugging API Requests
```bash
# Enable debug logging
export RUST_LOG=debug

# View API logs
docker-compose logs -f api
```

### Performance Optimization

#### Python SDK
- Increase `max_buffer_len` for high-frequency logging
- Use context managers for automatic cleanup
- Batch metrics when possible

#### API Server
- Adjust database connection pool size in `main.rs`
- Monitor memory usage with high metric volume
- Use database indexes for query optimization

## 📚 Additional Resources

### Documentation
- [Python SDK Examples](python/examples/) - Complete usage examples
- [API Schema](api/schema.sql) - Database structure
- [Features Overview](docs/features.md) - Detailed feature documentation

### Community
- [GitHub Issues](https://github.com/taigaishida/tora/issues) - Bug reports and feature requests
- [Discussions](https://github.com/taigaishida/tora/discussions) - Community support

### Related Projects
- [MLflow](https://mlflow.org/) - Comprehensive ML lifecycle management
- [Weights & Biases](https://wandb.ai/) - Experiment tracking and visualization
- [Neptune](https://neptune.ai/) - ML metadata management

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](python/LICENSE) file for details.

## 🙏 Acknowledgments

- Built with [Axum](https://github.com/tokio-rs/axum) for the Rust API
- [SvelteKit](https://kit.svelte.dev/) for the modern web interface
- [Supabase](https://supabase.com/) for database and authentication
- [Chart.js](https://www.chartjs.org/) for experiment visualization

---

**Happy Experimenting! 🚀**

For questions or support, please [open an issue](https://github.com/taigaishida/tora/issues) or start a [discussion](https://github.com/taigaishida/tora/discussions).
