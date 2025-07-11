# Tora: ML Experiment Management Tool

**Tora** is a modern, lightweight ML experiment tracking platform designed for speed and simplicity. Built for ML engineers and data scientists who want to track experiments without complexity.

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
- `tora/_wrapper.py`: Global API functions (`setup()`, `tlog()`, etc.)
- `tora/_exceptions.py`: Custom exception classes for error handling
- `examples/`: Complete usage examples for different ML frameworks

### Rust API (`/api`)
High-performance backend API server built with Axum framework. Handles experiment creation, metric storage, and data retrieval.

**Key Features:**
- **High performance**: Async Rust with efficient database connections
- **Authentication**: API key-based authentication with middleware
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

### Supabase/PostgreSQL
Database layer for storing experiments, metrics, and user data. Can use either Supabase (managed) or self-hosted PostgreSQL.

**Key Features:**
- **Scalable storage**: Efficient storage of time-series metrics data
- **User management**: Authentication and workspace permissions
- **Real-time subscriptions**: Live updates for web dashboard
- **Backup and recovery**: Managed database features with Supabase

**Schema:**
- **experiments**: Experiment metadata, hyperparameters, tags
- **metrics**: Time-series metric data with steps and metadata
- **workspaces**: Organization and permission management
- **users**: User accounts and API keys

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
