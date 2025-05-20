import httpx


TORA_BASE_URL = "http://localhost:5173/api"


def hp_to_tora_format(
    hp: dict[str, str | int | float],
) -> list[dict[str, str | int | float]]:
    return [{"key": key, "value": value} for key, value in hp.items()]


def hp_from_tora_format(
    hp_list: list[dict[str, str | int | float]],
) -> dict[str, str | int | float]:
    return {k: v for single in hp_list for k, v in single.items()}


class Tora:
    def __init__(
        self,
        experiment_id: str,
        description: str | None = None,
        hyperparams: dict[str, str | int | float] | None = None,
        tags: list[str] | None = None,
        server_url: str = TORA_BASE_URL,
        max_buffer_len: int = 25,
    ):
        self._experiment_id = experiment_id
        self.description = description
        self.hyperparams = hyperparams
        self.tags = tags
        self._max_buffer_len = max_buffer_len
        self._buffer = []
        self._http_client = httpx.Client(
            base_url=server_url,
            headers={
                "x-api-key": "tosk_f1477fb04daa14c007a2fa5159306df98d8891bb9eb37e05",
                "Content-Type": "application/json",
            },
        )

    @classmethod
    def create_experiment(
        cls,
        name: str,
        description: str | None = None,
        hyperparams: dict[str, str | int | float] | None = None,
        tags: list[str] | None = None,
        server_url: str = TORA_BASE_URL,
    ):
        data = {}

        data["name"] = name
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
                "x-api-key": "tosk_e5a828992e556642ba0d3edb097ca131a677188ef36d39dd",
            },
        )
        try:
            req.raise_for_status()
        except Exception as e:
            raise e

        return cls(
            req.json()["experiment"]["id"],
            description,
            hyperparams,
            tags,
            server_url=server_url,
        )

    @classmethod
    def load_experiment(
        cls,
        experiment_id: str,
        server_url: str = "http://localhost:5173/api",
    ):
        req = httpx.get(url=server_url + f"/experiments/{experiment_id}")
        req.raise_for_status()
        data = req.json()
        experiment_id = data["id"]
        description = data["description"]
        hyperparams = hp_from_tora_format(data["hyperparams"])
        tags = data["tags"]

        return cls(experiment_id, description, hyperparams, tags, server_url)

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
