import os
from typing import Any, Dict, List, Mapping, Optional, TypedDict, Union

import httpx

TORA_BASE_URL: str = os.getenv("TORA_BASE_URL", "http://localhost:5173/api")
TORA_API_KEY: Optional[str] = os.getenv("TORA_API_KEY", None)
HPValue = Union[str, int, float]


class ToraHPEntry(TypedDict):
    key: str
    value: HPValue


ToraHPFormat = List[ToraHPEntry]
HyperparamsDict = Dict[str, HPValue]


def to_tora_hp(hp: Mapping[str, HPValue]) -> ToraHPFormat:
    """
    Convert a mapping of hyperparameters to Tora’s list-of-dicts format.
    """
    return [{"key": k, "value": v} for k, v in hp.items()]


def from_tora_hp(tora_hp: ToraHPFormat) -> HyperparamsDict:
    """
    Convert Tora’s list-of-dicts back into a plain hyperparameter dict.
    """
    return {entry["key"]: entry["value"] for entry in tora_hp}


class Tora:
    """
    A client for creating and logging to Tora experiments.
    """

    _experiment_id: str
    _workspace_id: Optional[str]
    _description: Optional[str]
    _hyperparams: Optional[HyperparamsDict]
    tags: Optional[List[str]]
    _max_buffer_len: int
    _buffer: List[Dict[str, Any]]
    _http_client: httpx.Client
    _api_key: Optional[str]

    def __init__(
        self,
        experiment_id: str,
        workspace_id: Optional[str] = None,
        description: Optional[str] = None,
        hyperparams: Optional[HyperparamsDict] = None,
        tags: Optional[List[str]] = None,
        max_buffer_len: int = 25,
        api_key: Optional[str] = None,
        server_url: str = TORA_BASE_URL,
    ):
        self._experiment_id = experiment_id
        self._workspace_id = workspace_id
        self._description = description
        self._hyperparams = hyperparams
        self.tags = tags
        self._max_buffer_len = max_buffer_len
        self._buffer = []
        self._api_key = api_key or TORA_API_KEY

        headers = {"Content-Type": "application/json"}
        if self._api_key:
            headers["x-api-key"] = self._api_key

        self._http_client = httpx.Client(base_url=server_url, headers=headers)

    @staticmethod
    def _get_api_key(api_key: Optional[str]) -> Optional[str]:
        """Helper to resolve API key from param or environment."""
        key = api_key or TORA_API_KEY
        if key is None:
            print("Warning: Tora API key not provided. Operating in anonymous mode.")
        return key

    @staticmethod
    def create_workspace(
        name: str,
        api_key: str,
        description: str = "",
        server_url: str = TORA_BASE_URL,
    ) -> Dict[str, Any]:
        """
        Creates a new Tora workspace. Requires an API key.

        Args:
            name: The name for the new workspace.
            description: An optional description for the workspace.
            server_url: The base URL of the Tora server.

        Returns:
            The full JSON response for the newly created workspace.

        Raises:
            httpx.HTTPStatusError: If the API request fails.
        """
        resolved_api_key = Tora._get_api_key(api_key)
        headers = {
            "x-api-key": resolved_api_key,
            "Content-Type": "application/json",
        }
        with httpx.Client(base_url=server_url, headers=headers) as client:
            req = client.post(
                "/workspaces", json={"name": name, "description": description}
            )
            req.raise_for_status()
            return req.json()

    @classmethod
    def create_experiment(
        cls,
        name: str,
        workspace_id: Optional[str] = None,
        description: Optional[str] = None,
        hyperparams: Optional[HyperparamsDict] = None,
        tags: Optional[List[str]] = None,
        max_buffer_len: int = 25,
        api_key: Optional[str] = None,
        server_url: str = TORA_BASE_URL,
    ) -> "Tora":
        """
        Creates a new experiment and returns a Tora instance to interact with it.
        An API key is required to create an experiment in a specific workspace.
        """
        resolved_api_key = Tora._get_api_key(api_key)

        data: Dict[str, Any] = {"name": name}
        if workspace_id:
            data["workspaceId"] = workspace_id
        if description:
            data["description"] = description
        if hyperparams:
            data["hyperparams"] = to_tora_hp(hyperparams)
        if tags:
            data["tags"] = tags

        url_path = (
            f"/workspaces/{workspace_id}/experiments"
            if workspace_id
            else "/experiments"
        )

        headers = {"Content-Type": "application/json"}
        if resolved_api_key:
            headers["x-api-key"] = resolved_api_key

        with httpx.Client(base_url=server_url, headers=headers) as client:
            req = client.post(url_path, json=data)
            req.raise_for_status()
            response_data = req.json()
            exp_id = response_data.get("experiment", {}).get("id") or response_data.get(
                "id"
            )

        return cls(
            experiment_id=exp_id,
            workspace_id=workspace_id,
            description=description,
            hyperparams=hyperparams,
            tags=tags,
            server_url=server_url,
            max_buffer_len=max_buffer_len,
            api_key=resolved_api_key,
        )

    @classmethod
    def load_experiment(
        cls,
        experiment_id: str,
        max_buffer_len: int = 25,
        api_key: Optional[str] = None,  # Key might be needed for private experiments
        server_url: str = TORA_BASE_URL,
    ) -> "Tora":
        """
        Loads an existing experiment and returns a Tora instance to interact with it.
        """
        resolved_api_key = Tora._get_api_key(api_key)
        headers = {}
        if resolved_api_key:
            headers["x-api-key"] = resolved_api_key

        with httpx.Client(base_url=server_url, headers=headers) as client:
            req = client.get(f"/experiments/{experiment_id}")
            req.raise_for_status()
            data = req.json()

        hyperparams = (
            from_tora_hp(data["hyperparams"]) if data.get("hyperparams") else None
        )

        return cls(
            experiment_id=data["id"],
            workspace_id=data.get("workspace_id"),
            description=data.get("description"),
            hyperparams=hyperparams,
            tags=data.get("tags"),
            max_buffer_len=max_buffer_len,
            api_key=resolved_api_key,
            server_url=server_url,
        )

    def log(
        self,
        name: str,
        value: Any,
        step: Optional[int] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Logs a metric. Metrics are buffered and sent in batches.
        """
        log_entry = {"name": name, "value": value}
        if step is not None:
            log_entry["step"] = step
        if metadata is not None:
            log_entry["metadata"] = metadata

        self._buffer.append(log_entry)

        if len(self._buffer) >= self._max_buffer_len:
            self._write_logs()

    def _write_logs(self) -> None:
        if not self._buffer:
            return

        try:
            req = self._http_client.post(
                f"/experiments/{self._experiment_id}/metrics/batch",
                json=self._buffer,
                timeout=120,
            )
            req.raise_for_status()
            self._buffer = []
        except httpx.HTTPStatusError as e:
            # Provide a more detailed error message
            print(
                f"Failed to write Tora logs. Status: {e.response.status_code}. "
                f"Response: {e.response.text}"
            )
        except Exception as e:
            # Catch other potential exceptions like timeouts
            print(f"An unexpected error occurred while writing Tora logs: {e}")

    def shutdown(self) -> None:
        """
        Ensures all buffered logs are sent before the program exits.
        """
        if self._buffer:
            print(f"Tora shutting down. Sending {len(self._buffer)} remaining logs...")
            self._write_logs()
        self._http_client.close()

    @property
    def max_buffer_len(self) -> int:
        return self._max_buffer_len

    @max_buffer_len.setter
    def max_buffer_len(self, value: int):
        self._max_buffer_len = int(value)

    def __enter__(self):
        """Enter context manager."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit context manager, ensuring logs are flushed."""
        self.shutdown()
