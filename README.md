# Tora: ML Experiment Management Tool

**Tora** is a modern, lightweight ML experiment tracking platform designed for speed and simplicity. Built for ML engineers and data scientists who want to track experiments without complexity.

## 🏗️ Architecture Overview

Tora consists of four main components:

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Python SDK    │───▶│   Rust API      │───▶│   PostgreSQL    │
│   (Client)      │    │   (Backend)     │    │ (via Supabase)  │
└─────────────────┘    └─────────────────┘    └─────────────────┘
        ▲                       ▲                      ▲
        │                       │                      │
        │                ┌──────┴──────┐               │
        │                │  SvelteKit  │               │
        │                │    Web      │               │
        │                └─────────────┘               │
        │                       ▲                      │
        │                       │                      │
        └───────────────────────┴──────────────────────┘
                         iOS (SwiftUI)
```

## 🏗️ Components

### Python SDK (`/python`)
The Python client library provides a simple interface for ML experiment tracking. It offers both a global API for quick setup and a client API for more control.

**Key Features:**
- **Buffered logging**: Efficient batched metric logging for better performance
- **Type safety**: Full type hints and validation
- **Framework agnostic**: Works with PyTorch, TensorFlow, scikit-learn, etc.
- **Context managers**: Automatic cleanup and flushing

**Core Files:**
- `tora/_client.py`: Main Tora client class with experiment management
- `tora/_wrapper.py`: Global convenience API (`setup()`, `tmetric()`, `tresult()`, etc.)
- `tora/_http.py`: Minimal HTTP client built on Python stdlib (`http.client`)
- `tora/_exceptions.py`: Custom exception classes for error handling
- `examples/`: Complete usage examples for different ML frameworks

### Rust API (`/api`)
High-performance backend API server built with Axum framework. Handles experiment creation, metric storage, and data retrieval.

**Key Features:**
- **High performance**: Async Rust with efficient database connections
- **Authentication**: Supabase-based bearer auth and API keys (via `x-api-key`)
- **CORS support**: Configured for web dashboard integration
- **Graceful shutdown**: Proper cleanup on termination signals

**Core Files:**
- `src/main.rs`: Server entry point with database connection pooling
- `src/handlers/`: API route handlers for experiments, metrics, workspaces
- `src/middleware/`: Authentication and CORS middleware
- `schema.sql`: PostgreSQL database schema

### SvelteKit Web (`/web`)
Modern web dashboard for visualizing experiments and managing workspaces. Built with SvelteKit and TailwindCSS.

**Key Features:**
- **Real-time visualization**: Charts and graphs for experiment metrics
- **Experiment comparison**: Side-by-side comparison of different runs
- **Workspace management**: Organize experiments into logical groups
- **API proxy**: Seamless integration with Rust API backend

**Core Files:**
- `src/routes/`: Page routes and API proxy endpoints
- `src/lib/components/`: Reusable Svelte components
- `src/routes/api/[...proxyPath]/+server.ts`: API proxy to Rust backend

### iOS App (`/ios`)
Native SwiftUI app for mobile access to workspaces, experiments, and live metrics.

- **Frameworks**: SwiftUI, Combine
- **Auth**: Same bearer-token flow as the web app (via Supabase)
- **Formatting/Linting**: `swift-format`, `swiftlint`
- **Config**: Base API URL in `ios/tora/Services/Config.swift`

### Supabase/PostgreSQL
Database layer for storing experiments, metrics, and user data. Can use either Supabase (managed) or self-hosted PostgreSQL.

**Key Features:**
- **Scalable storage**: Efficient storage of time-series metrics data
- **User management**: Authentication and workspace permissions
- **Real-time subscriptions**: Live updates for web dashboard
- **Backup and recovery**: Managed database features with Supabase

**Schema (tables):**
- `experiment`: Experiment metadata, hyperparameters, tags
- `log`: Time-series metric/result data with steps and metadata
- `workspace`, `user_workspaces`, `workspace_role`, `workspace_invitations`
- `api_keys`: Per-user API keys (only hashes stored)

See `up.sql` and `supabase/migrations/*` for SQL.

## 🔌 API Documentation

### Auth
- `POST /api/signup` — Create user (email + password)
- `GET /api/signup/confirm?token_hash=...&confirm_type=email` — Confirm signup
- `POST /api/login` — Login, returns access/refresh tokens
- `POST /api/refresh` — Refresh access token

Use either:
- Bearer auth: `Authorization: Bearer <access_token>`
- API key: `x-api-key: <api_key>`

### Workspaces
- `GET /api/workspaces` — List workspaces
- `POST /api/workspaces` — Create workspace
- `GET /api/workspaces/{id}` — Get workspace
- `DELETE /api/workspaces/{id}` — Delete workspace (owner)
- `POST /api/workspaces/{id}/leave` — Leave workspace
- `GET /api/workspaces/{id}/members` — List members
- `GET /api/workspaces/{id}/experiments` — List experiments in workspace

### Experiments
- `GET /api/experiments` — List experiments (optional `?workspace=<id>`)
- `POST /api/experiments` — Create experiment
- `GET /api/experiments/{id}` — Get experiment
- `PUT /api/experiments/{id}` — Update experiment
- `DELETE /api/experiments/{id}` — Delete experiment
- `POST /api/experiments/batch` — Batch fetch experiments by IDs

### Logs (metrics/results)
- `GET /api/experiments/{id}/logs` — Get logs
- `POST /api/experiments/{id}/logs` — Create single log
- `POST /api/experiments/{id}/logs/batch` — Batch create logs
- `GET /api/experiments/{id}/logs/csv` — Export logs as CSV

### API Keys
- `GET /api/api-keys` — List API keys
- `POST /api/api-keys` — Create API key
- `DELETE /api/api-keys/{id}` — Revoke API key

### Invitations
- `POST /api/workspace-invitations` — Invite user to workspace
- `GET /api/workspace-invitations` — List invitations for current user
- `PUT /api/workspaces/any/invitations?invitationId=...&action=accept|decline` — Respond to invitation

### Python SDK API

#### Main Client

```python
import tora

# Create experiment
client = tora.Tora.create_experiment(
    name="experiment-name",
    workspace_id="ws-123",
    description="Optional description",
    hyperparams={"lr": 0.001, "batch_size": 32},
    tags=["pytorch", "cnn"],
    max_buffer_len=100,  # Buffer size for metrics
)

# Load existing experiment
client = tora.Tora.load_experiment(experiment_id="exp-123")

# Log metrics/results
client.metric("accuracy", 0.95, step_or_epoch=100)
client.result("final_accuracy", 0.95)

# Flush and cleanup
client.flush()
client.shutdown()
```

#### Global API

```python
import tora

# Setup global experiment
experiment_id = tora.setup(
    name="global-experiment",
    workspace_id="ws-123",
    hyperparams={"lr": 0.001},
    max_buffer_len=25,
)

# Log from anywhere
tora.tmetric("loss", 0.05, step=50)
tora.tresult("best_acc", 0.97)

# Utilities
tora.flush()                 # Force send buffered metrics
tora.is_initialized()        # Check if setup was called
tora.get_experiment_id()     # Get current experiment ID
tora.get_experiment_url()    # Get current experiment URL
tora.shutdown()              # Cleanup
```

## 🧰 Local Development

- Tools: Rust (cargo), `cargo-watch`, Node 18+ with `pnpm` (or `npm`), direnv, optional Supabase CLI
- Quick start: `./start-dev.sh` runs API on `:8080` and Web on `:5173` with hot reload
- Env: see `.envrc` for handy local defaults

Key environment variables:
- API: `DATABASE_URL`, `SUPABASE_URL`, `SUPABASE_API_KEY`, `SUPABASE_JWT_SECRET`, `FRONTEND_URL`
- Web: `PUBLIC_API_BASE_URL` (e.g., `http://localhost:8080`)
- Python SDK: `TORA_BASE_URL` (e.g., `http://localhost:8080/api`), `TORA_API_KEY` (optional)
- iOS: edit `ios/tora/Services/Config.swift` `baseURL`

Docker compose (API + Web): `docker-compose up --build`

## 🧪 Dev Tooling

- Python: `ruff`, `mypy`, `pytest` (configured in `python/pyproject.toml`)
- Rust: `rustfmt` (via pre-commit), `tracing`
- Web: SvelteKit 2, Svelte 5, Vite 6, Tailwind 4, Prettier
- iOS: `swift-format`, `swiftlint`
- Pre-commit hooks configured in `.pre-commit-config.yaml`

## 🚀 Deploy

- Cloud Run via `cloudbuild.yaml` (multi-stage Docker builds)
- Secrets: `DATABASE_URL`, `SUPABASE_URL`, `SUPABASE_KEY`, `SUPABASE_JWT_SECRET`
- Frontend env: `PUBLIC_API_BASE_URL` should point at the deployed API base URL
