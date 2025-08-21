"""
Saturate the Tora backend with experiments and metrics.

Usage examples:

  - Basic (uses env `TORA_BASE_URL` and `TORA_API_KEY`):
      python scripts/saturate_backend.py --metrics 200

  - Create a workspace first, then populate it:
      python scripts/saturate_backend.py \
        --workspace-name "Load Test" \
        --metrics 100

  - Override API URL and key explicitly:
      python scripts/saturate_backend.py \
        --server-url https://your-host/api \
        --api-key YOUR_API_KEY \
        --metrics 50

Environment variables:
  - TORA_BASE_URL: Default API base URL (e.g., https://host/api)
  - TORA_API_KEY:  API key for authenticated requests

Notes:
  - Uses the local Tora client to create one experiment, log train/val losses via `Tora.metric`, and final results via `Tora.result`.
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
from typing import Any, Iterable

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


def _random_hparams() -> dict[str, int | float | str]:
    return {
        "lr": round(10 ** random.uniform(-4, -1), 5),
        "batch_size": random.choice([16, 32, 64, 128]),
        "optimizer": random.choice(["sgd", "adam", "adamw"]),
    }


def _random_tags() -> list[str]:
    return [random.choice(["baseline", "ablation", "debug", "prod", "nightly"]) for _ in range(2)]


@dataclass
class ExperimentSpec:
    workspace_id: str | None
    name_prefix: str
    metrics_per_experiment: int
    max_buffer_len: int
    api_key: str | None
    server_url: str | None


def run_one_experiment(spec: ExperimentSpec, index: int, verbose: bool = False) -> dict[str, Any]:
    name = f"loadtest-{spec.name_prefix}-{index:04d}-{_rand_suffix()}"
    try:
        tora = Tora.create_experiment(
            name=name,
            workspace_id=spec.workspace_id,
            description="Load test experiment",
            hyperparams=_random_hparams(),
            tags=_random_tags(),
            max_buffer_len=spec.max_buffer_len,
            api_key=spec.api_key,
            server_url=spec.server_url,
        )

        if verbose:
            logging.info("Created experiment %s (%s)", tora.experiment_id, tora.url)

        # Emit intra-training metrics with a DNN-like pattern
        # Train loss decays exponentially towards an asymptote with small noise/spikes
        # Validation loss follows train loss with an early shrinking gap and a mild overfitting tail
        best_val_loss = float("inf")
        final_train_loss: float | None = None
        final_val_loss: float | None = None

        steps = max(1, spec.metrics_per_experiment)
        train_start = 2.0
        train_asym = 0.05
        decay = 5.0 / steps
        gap_initial = 0.3
        gap_decay = gap_initial / (0.6 * steps)
        overfit_amp = 0.15
        overfit_start = 0.6 * steps
        overfit_width = max(1.0, 0.1 * steps)

        for step in range(1, steps + 1):
            base_train = train_asym + (train_start - train_asym) * math.exp(-decay * step)
            noise = random.gauss(0.0, 0.015)
            spike = random.uniform(0.03, 0.08) if random.random() < 0.02 else 0.0
            train_loss = max(0.0, base_train + noise + spike)

            gap_linear = max(0.0, gap_initial - gap_decay * step)
            overfit_phase = 1.0 / (1.0 + math.exp(-(step - overfit_start) / overfit_width))
            gap_overfit = overfit_amp * overfit_phase
            val_noise = random.gauss(0.0, 0.02)
            val_loss = max(0.0, train_loss + gap_linear + gap_overfit + val_noise)

            final_train_loss = train_loss
            final_val_loss = val_loss
            best_val_loss = min(best_val_loss, val_loss)

            tora.metric("train_loss", round(train_loss, 6), step_or_epoch=step)
            tora.metric("val_loss", round(val_loss, 6), step_or_epoch=step)

        # Log final results
        if final_train_loss is not None:
            tora.result("final_train_loss", round(final_train_loss, 6))
        if final_val_loss is not None:
            tora.result("final_val_loss", round(final_val_loss, 6))
        if best_val_loss != float("inf"):
            tora.result("best_val_loss", round(best_val_loss, 6))

        # Mock classification metrics derived from final validation loss
        if final_val_loss is not None:
            # Map loss to a quality score in [0, 1]
            # Lower loss -> higher quality; add slight noise
            quality = 1.0 - (final_val_loss / (final_val_loss + 1.0))
            quality = max(0.0, min(1.0, quality + random.gauss(0.0, 0.01)))

            # Sample precision/recall around quality with small, correlated noise
            recall = max(0.0, min(1.0, quality + random.gauss(0.0, 0.02)))
            precision = max(0.0, min(1.0, quality + random.gauss(0.005, 0.02)))
            f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0

            tora.result("precision", round(precision, 6))
            tora.result("recall", round(recall, 6))
            tora.result("f1", round(f1, 6))

        tora.flush()
        tora.shutdown()
        return {"ok": True, "experiment_id": tora.experiment_id, "name": name}

    except (ToraValidationError, ToraConfigurationError, ToraAuthenticationError, ToraAPIError, ToraNetworkError) as e:
        logging.error("Experiment %s failed: %s", name, e)
        return {"ok": False, "error": str(e), "name": name}
    except Exception as e:  # noqa: BLE001
        logging.exception("Unexpected error in experiment %s", name)
        return {"ok": False, "error": f"Unexpected: {e}", "name": name}


def parse_args(argv: Iterable[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Saturate backend with Tora experiments and metrics")
    p.add_argument("--workspace-id", type=str, default=None, help="Workspace ID to use (optional)")
    p.add_argument("--workspace-name", type=str, default=None, help="Create a workspace with this name, if provided")
    p.add_argument("--metrics", type=int, default=200, help="Number of training steps (metrics) to log")
    p.add_argument("--buffer", type=int, default=50, help="Client max buffer length for batch logging")
    p.add_argument(
        "--server-url", type=str, default=None, help="Override Tora server base URL (otherwise env TORA_BASE_URL)"
    )
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

    # No dry-run: immediately execute against API

    # Create workspace if requested
    workspace_id = ns.workspace_id
    if ns.workspace_name and not workspace_id:
        try:
            ws = create_workspace(
                ns.workspace_name, description="Load test workspace", api_key=api_key, server_url=server_url
            )
            workspace_id = ws.get("id")
            print(f"Created workspace: {workspace_id} ({ws.get('name')})")
        except (ToraError, ToraNetworkError) as e:
            logging.error("Failed to create workspace: %s", e)
            return 2

    # Prepare spec
    spec = ExperimentSpec(
        workspace_id=workspace_id,
        name_prefix=_rand_suffix(4),
        metrics_per_experiment=ns.metrics,
        max_buffer_len=max(1, int(ns.buffer)),
        api_key=api_key,
        server_url=server_url,
    )

    start = time.time()
    res = run_one_experiment(spec, 0, verbose=ns.verbose)
    ok = 1 if res.get("ok") else 0
    fail = 0 if ok == 1 else 1

    elapsed = time.time() - start
    print(f"Done. Experiments ok={ok}, failed={fail}, time={elapsed:.2f}s")
    return 0 if fail == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
