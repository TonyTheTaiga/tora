from typing import Optional, List

from .client import Tora


_CLIENT = None


def _get_client() -> Tora:
    if _CLIENT is None:
        raise ValueError("Tora client not initialized")
    return _CLIENT


def setup(
    name: str,
    workspace_id: Optional[str] = None,
    description: Optional[str] = None,
    hyperparams: Optional[dict] = None,
    tags: Optional[List[str]] = None,
    api_key: Optional[str] = None,
):
    global _CLIENT
    _CLIENT = Tora.create_experiment(
        name,
        workspace_id,
        description,
        hyperparams,
        tags,
        api_key=api_key,
        max_buffer_len=1,
    )
    print("http://localhost:5173/experiments/" + _CLIENT._experiment_id)


def tlog(name, value, step, metadata=None):
    _get_client().log(name, value, step, metadata)
