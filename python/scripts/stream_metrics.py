"""
Continuously stream metrics to the Tora backend to exercise live updates
and WebSocket-driven views.

Usage examples:

  - Stream for 60 seconds at ~10 Hz (uses env TORA_BASE_URL, TORA_API_KEY):
      python scripts/stream_metrics.py --duration 60 --rate 10

  - Create a workspace first, then stream into a new experiment:
      python scripts/stream_metrics.py \
        --workspace-name "WS Test" \
        --duration 120 \
        --rate 5

  - Override API URL and key explicitly and stream with a small buffer:
      python scripts/stream_metrics.py \
        --server-url https://your-host/api \
        --api-key YOUR_API_KEY \
        --duration 30 \
        --rate 20 \
        --buffer 1

Environment variables:
  - TORA_BASE_URL: Default API base URL (e.g., https://host/api)
  - TORA_API_KEY:  API key for authenticated requests

Notes:
  - Creates a single experiment and emits a pair of metrics (train_loss, val_loss)
    at a configured rate. Use Ctrl-C to stop at any time. If --duration is 0 or
    negative, the script runs until interrupted.
"""

from __future__ import annotations

import argparse
import logging
import math
import os
import random
import string
import sys
import time
from dataclasses import dataclass
from typing import Iterable

from tora._client import Tora, create_workspace
from tora._exceptions import (
    ToraAPIError,
    ToraAuthenticationError,
    ToraConfigurationError,
    ToraError,
    ToraNetworkError,
    ToraValidationError,
)


def _rand_suffix(n: int = 6) -> str:
    return "".join(random.choices(string.ascii_lowercase + string.digits, k=n))


@dataclass
class StreamSpec:
    workspace_id: str | None
    name: str
    max_buffer_len: int
    api_key: str | None
    server_url: str | None
    rate_hz: float
    duration_sec: float | None


def _make_tora(spec: StreamSpec) -> Tora:
    return Tora.create_experiment(
        name=spec.name,
        workspace_id=spec.workspace_id,
        description="Streaming metrics load test",
        hyperparams={"lr": 1e-3, "batch_size": 32},
        tags=["ws", "stream", "load-test"],
        max_buffer_len=spec.max_buffer_len,
        api_key=spec.api_key,
        server_url=spec.server_url,
    )


def stream_metrics(spec: StreamSpec, verbose: bool = False) -> int:
    try:
        tora = _make_tora(spec)
    except (ToraValidationError, ToraConfigurationError, ToraAuthenticationError, ToraAPIError, ToraNetworkError) as e:
        logging.error("Failed to create experiment: %s", e)
        return 2

    # Always show where metrics are going so it is easy to observe in the UI
    print(f"Experiment created: {tora.experiment_id} ({tora.url})")
    if verbose:
        logging.info("Streaming to experiment %s (%s)", tora.experiment_id, tora.url)

    # Parameters to shape a realistic training curve
    train_start = 1.8
    train_asym = 0.04
    step = 0

    # Convert rate to a target interval; guard against nonsensical values
    rate_hz = max(0.1, float(spec.rate_hz))
    interval = 1.0 / rate_hz

    start_time = time.time()
    next_tick = start_time
    last_flush = start_time

    try:
        while True:
            now = time.time()
            if spec.duration_sec is not None and spec.duration_sec > 0 and (now - start_time) >= spec.duration_sec:
                break

            # Sleep until the next tick to approximate the target rate
            if now < next_tick:
                time.sleep(min(0.05, next_tick - now))
                continue

            step += 1

            # Exponential decay towards asymptote + mild noise/spikes
            decay = 5.0 / max(1, int(rate_hz * 60))  # decay horizon ~60 seconds worth of steps
            base_train = train_asym + (train_start - train_asym) * math.exp(-decay * step)
            noise = random.gauss(0.0, 0.012)
            spike = random.uniform(0.02, 0.06) if random.random() < 0.01 else 0.0
            train_loss = max(0.0, base_train + noise + spike)

            # Validation = train + shrinking gap + mild overfit tail
            gap_initial = 0.25
            gap_decay = gap_initial / 1000.0
            gap_linear = max(0.0, gap_initial - gap_decay * step)
            overfit_phase = 1.0 / (1.0 + math.exp(-(step - 600) / 60))
            gap_overfit = 0.12 * overfit_phase
            val_noise = random.gauss(0.0, 0.015)
            val_loss = max(0.0, train_loss + gap_linear + gap_overfit + val_noise)

            # Emit both metrics. Buffering/flush handled by client.
            tora.metric("train_loss", round(train_loss, 6), step_or_epoch=step)
            tora.metric("val_loss", round(val_loss, 6), step_or_epoch=step)

            # Opportunistically flush at least once per second if buffer not empty
            if spec.max_buffer_len > 1 and (now - last_flush) >= 1.0 and tora.buffer_size > 0:
                try:
                    tora.flush()
                except Exception:  # noqa: BLE001
                    logging.exception("Flush failed (continuing)")
                last_flush = now

            next_tick += interval

    except KeyboardInterrupt:
        logging.info("Interrupted by user; shutting down...")
    except Exception:  # noqa: BLE001
        logging.exception("Unexpected error while streaming metrics")
        return 1
    finally:
        try:
            tora.flush()
            tora.shutdown()
        except Exception:  # noqa: BLE001
            logging.exception("Error during shutdown")

    return 0


def parse_args(argv: Iterable[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Continuously stream metrics for WebSocket testing")
    p.add_argument("--workspace-id", type=str, default=None, help="Workspace ID to use (optional)")
    p.add_argument("--workspace-name", type=str, default=None, help="Create a workspace with this name, if provided")
    p.add_argument("--name", type=str, default=None, help="Experiment name (default: auto-generated)")
    p.add_argument("--duration", type=float, default=60.0, help="Duration to run in seconds (0 or <0 = infinite)")
    p.add_argument("--rate", type=float, default=10.0, help="Metrics emission rate in Hz (pairs per second)")
    p.add_argument("--buffer", type=int, default=1, help="Client max buffer length for batch logging")
    p.add_argument("--server-url", type=str, default=None, help="Override Tora server base URL (otherwise env TORA_BASE_URL)")
    p.add_argument("--api-key", type=str, default=None, help="Override API key (otherwise env TORA_API_KEY)")
    p.add_argument("--verbose", action="store_true", help="Enable verbose logging")
    return p.parse_args(list(argv))


def main(argv: Iterable[str] | None = None) -> int:
    ns = parse_args(argv or sys.argv[1:])

    logging.basicConfig(
        level=logging.INFO if ns.verbose else logging.WARNING,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    api_key = ns.api_key or os.getenv("TORA_API_KEY")
    server_url = ns.server_url or os.getenv("TORA_BASE_URL")

    workspace_id = ns.workspace_id

    if ns.workspace_name and not workspace_id:
        try:
            ws = create_workspace(
                ns.workspace_name, description="WS load test workspace", api_key=api_key, server_url=server_url
            )
            workspace_id = ws.get("id")
            print(f"Created workspace: {workspace_id} ({ws.get('name')})")
        except (ToraError, ToraNetworkError) as e:
            logging.error("Failed to create workspace: %s", e)
            return 2

    name = ns.name or f"ws-stream-{_rand_suffix(6)}"

    spec = StreamSpec(
        workspace_id=workspace_id,
        name=name,
        max_buffer_len=max(1, int(ns.buffer)),
        api_key=api_key,
        server_url=server_url,
        rate_hz=float(ns.rate),
        duration_sec=(None if ns.duration is None else float(ns.duration)),
    )

    start = time.time()
    rc = stream_metrics(spec, verbose=ns.verbose)
    elapsed = time.time() - start
    print(f"Done. Experiment='{name}', elapsed={elapsed:.2f}s, rc={rc}")
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
