# Tora: ML Experiment Management Tool

A lightweight ML experiment tracking platform designed for speed and simplicity

## Architecture Overview

Tora consists of the following components and data flows:

```mermaid
flowchart LR
  %% Clients
  subgraph Clients
    PY[Python SDK]
    WEB[Web App (SvelteKit)]
    IOS[iOS App (SwiftUI)]
  end

  %% Backend
  subgraph Backend
    API[[Rust API (Axum)]]
    WORKER[(Outbox Worker)]
  end

  %% Infrastructure
  subgraph Infrastructure
    DB[(PostgreSQL\nvia Supabase)]
    AUTH[(Supabase Auth)]
    VALKEY[(Valkey / Redis)]
  end

  %% Request/response (HTTP)
  PY -->|HTTP / JSON| API
  WEB -->|HTTP / JSON| API
  IOS -->|HTTP / JSON| API

  %% Live updates (WebSocket via API)
  WEB -.->|WebSocket: stream metrics| API
  IOS -.->|WebSocket: stream metrics| API

  %% Backend integrations
  API -->|SQLx| DB
  API -->|JWT verify| AUTH
  API -->|Pub/Sub + WS tokens| VALKEY

  %% Outbox pattern for reliable streaming
  API -->|write outbox| DB
  WORKER -->|read outbox| DB
  WORKER -->|publish events| VALKEY
```
