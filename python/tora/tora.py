import httpx
import os


TORA_BASE_URL = "http://localhost:5173/api"
TORA_DEV_KEY = "tosk_e5a828992e556642ba0d3edb097ca131a677188ef36d39dd"
TORA_API_KEY = os.getenv("TORA_API_KEY", TORA_DEV_KEY)


def hp_to_tora_format(
    hp: dict[str, str | int | float],
) -> list[dict[str, str | int | float]]:
    return [{"key": key, "value": value} for key, value in hp.items()]


def hp_from_tora_format(
    hp_list: list[dict[str, str | int | float]],
) -> dict[str, str | int | float]:
    """Convert hyperparameters from the API format to a dictionary."""
    return {single["key"]: single["value"] for single in hp_list}


def create_workspace(
    name: str, description: str = "", api_key: str = TORA_API_KEY
) -> str:
    http_client = httpx.Client(
        base_url=TORA_BASE_URL,
        headers={
            "x-api-key": api_key,
            "Content-Type": "application/json",
        },
    )
    req = http_client.post(
        "/workspaces", json={"name": name, "description": description}
    )
    req.raise_for_status()
    return req.json()


class Tora:
    def __init__(
        self,
        experiment_id: str,
        workspace_id: str,
        description: str | None = None,
        hyperparams: dict[str, str | int | float] | None = None,
        tags: list[str] | None = None,
        server_url: str = TORA_BASE_URL,
        max_buffer_len: int = 25,
        api_key: str = TORA_API_KEY,
    ):
        self._experiment_id = experiment_id
        self._workspace_id = workspace_id
        self.description = description
        self.hyperparams = hyperparams
        self.tags = tags
        self._max_buffer_len = max_buffer_len
        self._buffer = []
        self._http_client = httpx.Client(
            base_url=server_url,
            headers={
                "x-api-key": api_key,
                "Content-Type": "application/json",
            },
        )

    @property
    def max_buffer_len(self) -> int:
        return self._max_buffer_len

    @max_buffer_len.setter
    def max_buffer_len(self, value: int):
        self._max_buffer_len = int(value)

    @classmethod
    def create_experiment(
        cls,
        workspace_id: str,
        name: str,
        description: str | None = None,
        hyperparams: dict[str, str | int | float] | None = None,
        tags: list[str] | None = None,
        server_url: str = TORA_BASE_URL,
        max_buffer_len: int = 25,
        api_key: str = TORA_API_KEY,
    ):
        data = {}
        data["name"] = name
        data["workspaceId"] = workspace_id

        if description:
            data["description"] = description

        if hyperparams:
            data["hyperparams"] = hp_to_tora_format(hyperparams)

        if tags:
            data["tags"] = tags

        req = httpx.post(
            server_url + "/experiments",
            json=data,
            headers={
                "Content-Type": "application/json",
                "x-api-key": api_key,
            },
        )
        try:
            req.raise_for_status()
        except Exception as e:
            raise e

        return cls(
            experiment_id=req.json()["experiment"]["id"],
            workspace_id=workspace_id,
            description=description,
            hyperparams=hyperparams,
            tags=tags,
            server_url=server_url,
            max_buffer_len=max_buffer_len,
            api_key=api_key,
        )

    @classmethod
    def load_experiment(
        cls,
        experiment_id: str,
        server_url: str = "http://localhost:5173/api",
        max_buffer_len: int = 25,
        api_key: str = TORA_API_KEY,
    ):
        req = httpx.get(
            url=server_url + f"/experiments/{experiment_id}",
            headers={"x-api-key": api_key},
        )
        req.raise_for_status()
        data = req.json()
        experiment_id = data["id"]
        workspace_id = data["workspace_id"]
        description = data["description"]
        hyperparams = hp_from_tora_format(data["hyperparams"])
        tags = data["tags"]

        return cls(
            experiment_id,
            workspace_id,
            description,
            hyperparams,
            tags,
            server_url,
            max_buffer_len=max_buffer_len,
            api_key=api_key,
        )

    def log(self, name, value, step: int | None = None, metadata: dict | None = None):
        self._buffer.append(
            {"name": name, "value": value, "step": step, "metadata": metadata}
        )
        if len(self._buffer) >= self._max_buffer_len:
            self._write_logs()

    def _write_logs(self):
        req = self._http_client.post(
            f"/experiments/{self._experiment_id}/metrics/batch",
            json=self._buffer,
            timeout=120,
        )
        try:
            req.raise_for_status()
        except Exception as e:
            print(e, req.json())
        self._buffer = []

    def shutdown(self):
        if self._buffer:
            self._write_logs()
