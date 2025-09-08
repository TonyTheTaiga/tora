"""
Create an experiment, log some metrics, then "resume" by loading
the same experiment and continuing to log more metrics.

Usage examples:

  - Default run (uses env TORA_BASE_URL, TORA_API_KEY):
      python scripts/test_resume.py

  - Specify steps and buffer size explicitly:
      python scripts/test_resume.py --initial-steps 8 --resume-steps 12 --buffer 5

  - Create a workspace first and place the experiment there:
      python scripts/test_resume.py --workspace-name "Resume Test WS"

  - Override API URL and key explicitly:
      python scripts/test_resume.py \
        --server-url https://your-host/api \
        --api-key YOUR_API_KEY

Environment variables:
  - TORA_BASE_URL: Default API base URL (e.g., https://host/api)
  - TORA_API_KEY:  API key for authenticated requests

Notes:
  - Demonstrates use of Tora.create_experiment(...) followed by Tora.load_experiment(...)
    to continue logging to the same experiment.
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
class ResumeSpec:
    workspace_id: str | None
    name: str
    max_buffer_len: int
    api_key: str | None
    server_url: str | None
    initial_steps: int
    resume_steps: int


def _make_experiment(spec: ResumeSpec) -> Tora:
    return Tora.create_experiment(
        name=spec.name,
        workspace_id=spec.workspace_id,
        description="Resume test: create -> shutdown -> load -> continue",
        hyperparams={"lr": 1e-3, "batch_size": 32},
        tags=["resume", "demo"],
        max_buffer_len=spec.max_buffer_len,
        api_key=spec.api_key,
        server_url=spec.server_url,
    )


def _log_steps(tora: Tora, start_step: int, count: int) -> tuple[float, float]:
    """Log `count` steps of train/val losses, starting after `start_step`.

    Returns the last (train_loss, val_loss) logged for optional result logging.
    """
    if count <= 0:
        return (0.0, 0.0)

    train_start = 1.8
    train_asym = 0.04
    decay = 5.0 / max(1, count)
    gap_initial = 0.25
    gap_decay = gap_initial / max(1, int(0.8 * count))
    overfit_amp = 0.12
    overfit_start = 0.6 * count
    overfit_width = max(1.0, 0.15 * count)

    last_train = 0.0
    last_val = 0.0

    for i in range(1, count + 1):
        step = start_step + i
        base_train = train_asym + (train_start - train_asym) * math.exp(-decay * (start_step + i))
        noise = random.gauss(0.0, 0.012)
        spike = random.uniform(0.02, 0.05) if random.random() < 0.01 else 0.0
        train_loss = max(0.0, base_train + noise + spike)

        gap_linear = max(0.0, gap_initial - gap_decay * (start_step + i))
        overfit_phase = 1.0 / (1.0 + math.exp(-((start_step + i) - overfit_start) / overfit_width))
        gap_overfit = overfit_amp * overfit_phase
        val_noise = random.gauss(0.0, 0.015)
        val_loss = max(0.0, train_loss + gap_linear + gap_overfit + val_noise)

        tora.metric("train_loss", round(train_loss, 6), step_or_epoch=step)
        tora.metric("val_loss", round(val_loss, 6), step_or_epoch=step)

        last_train, last_val = train_loss, val_loss

    return (last_train, last_val)


def parse_args(argv: Iterable[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Test resume by loading an existing experiment and continuing to log")
    p.add_argument("--workspace-id", type=str, default=None, help="Workspace ID to use (optional)")
    p.add_argument("--workspace-name", type=str, default=None, help="Create a workspace with this name, if provided")
    p.add_argument("--name", type=str, default=None, help="Experiment name (default: auto-generated)")
    p.add_argument("--initial-steps", type=int, default=10, help="Steps to log before shutdown (phase 1)")
    p.add_argument("--resume-steps", type=int, default=10, help="Steps to log after load (phase 2)")
    p.add_argument("--buffer", type=int, default=10, help="Client max buffer length for batch logging")
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
                ns.workspace_name, description="Resume test workspace", api_key=api_key, server_url=server_url
            )
            workspace_id = ws.get("id")
            print(f"Created workspace: {workspace_id} ({ws.get('name')})")
        except (ToraError, ToraNetworkError) as e:
            logging.error("Failed to create workspace: %s", e)
            return 2

    name = ns.name or f"resume-test-{_rand_suffix(6)}"

    spec = ResumeSpec(
        workspace_id=workspace_id,
        name=name,
        max_buffer_len=max(1, int(ns.buffer)),
        api_key=api_key,
        server_url=server_url,
        initial_steps=max(0, int(ns.initial_steps)),
        resume_steps=max(0, int(ns.resume_steps)),
    )

    start_ts = time.time()

    # Phase 1: create and log some steps
    try:
        tora = _make_experiment(spec)
    except (ToraValidationError, ToraConfigurationError, ToraAuthenticationError, ToraAPIError, ToraNetworkError) as e:
        logging.error("Failed to create experiment: %s", e)
        return 2

    print(f"Experiment created: {tora.experiment_id} ({tora.url})")
    if ns.verbose:
        logging.info("Phase 1: logging %d steps", spec.initial_steps)

    try:
        _log_steps(tora, start_step=0, count=spec.initial_steps)
        tora.flush()
    except Exception:  # noqa: BLE001
        logging.exception("Error while logging phase 1 metrics (continuing to shutdown)")
    finally:
        try:
            tora.shutdown()
        except Exception:  # noqa: BLE001
            logging.exception("Error during phase 1 shutdown")

    # Phase 2: load and continue logging
    try:
        loaded = Tora.load_experiment(
            tora.experiment_id,
            max_buffer_len=spec.max_buffer_len,
            api_key=spec.api_key,
            server_url=spec.server_url,
        )
    except (ToraValidationError, ToraConfigurationError, ToraAuthenticationError, ToraAPIError, ToraNetworkError) as e:
        logging.error("Failed to load experiment %s: %s", tora.experiment_id, e)
        return 2

    print(f"Experiment loaded:  {loaded.experiment_id} ({loaded.url})")
    if ns.verbose:
        logging.info("Phase 2: logging %d steps (resume)", spec.resume_steps)

    try:
        _log_steps(loaded, start_step=spec.initial_steps, count=spec.resume_steps)
        # Optional: log a couple of results after resuming
        loaded.result("resume_marker", 1)
        loaded.result("total_steps", spec.initial_steps + spec.resume_steps)
        loaded.flush()
    except Exception:  # noqa: BLE001
        logging.exception("Error while logging phase 2 metrics (continuing to shutdown)")
    finally:
        try:
            loaded.shutdown()
        except Exception:  # noqa: BLE001
            logging.exception("Error during phase 2 shutdown")

    elapsed = time.time() - start_ts
    print(
        "Done. Experiment='%s' initial=%d resume=%d elapsed=%.2fs"
        % (name, spec.initial_steps, spec.resume_steps, elapsed)
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
