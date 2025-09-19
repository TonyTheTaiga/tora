# Tora: ML Experiment Management Tool

A lightweight ML experiment tracking platform designed for speed and simplicity

## Architecture Overview

Tora consists of the following components and data flows:

```mermaid
flowchart LR
  subgraph Clients
    PY[Python SDK]
    WEB[Web App - SvelteKit]
    IOS[iOS App - SwiftUI]
  end

  subgraph Backend
    API[[Rust API - Axum]]
    WORKER[(Outbox Worker)]
  end

  subgraph Infra
    DB[(PostgreSQL<br/>via Supabase)]
    AUTH[(Supabase Auth)]
    VALKEY[(Valkey/Redis)]
  end

  PY -->|HTTP+JSON| API
  WEB -->|HTTP+JSON| API
  IOS -->|HTTP+JSON| API

  WEB -.->|WebSocket| API
  IOS -.->|WebSocket| API

  API -->|SQLx| DB
  API -->|JWT verify| AUTH
  API -->|PubSub + WS tokens| VALKEY

  API -->|write outbox| DB
  WORKER -->|read outbox| DB
  WORKER -->|publish events| VALKEY
```
