# What is Tora?

**Tora** is an experiment tracker built for speed and simplicity.

Track metrics, hyperparameters, and experiment metadata with just 2 lines of code.

---

## Global Functions

- setup: Initialize a global experiment session.
- create_workspace: Create a new workspace.
- tlog: Log a single metric.
- tresult: Log a result value.
- flush: Send buffered metrics immediately.
- shutdown: Flush remaining metrics and close the client.
- is_initialized: Check whether the global client is initialized.
- get_experiment_id: Return the current experiment ID.
- get_experiment_url: Return the current experiment URL.

```python
def setup(
    name: str,
    workspace_id: str | None = None,
    description: str | None = None,
    hyperparams: dict | None = None,
    tags: list[str] | None = None,
    api_key: str | None = None,
    server_url: str | None = None,
    max_buffer_len: int = 1,
) -> str:
    """Initialize the global experiment session.

    Args:
        name: Experiment name.
        workspace_id: Target workspace ID.
        description: Experiment description.
        hyperparams: Hyperparameter dictionary.
        tags: List of experiment tags.
        api_key: Authentication key.
        server_url: Custom server URL.
        max_buffer_len: Buffer size. Defaults to 1.

    Returns:
        Experiment ID.

    Notes:
        Prints the experiment URL to the console on creation.
    """

def create_workspace(
    name: str,
    description: str | None = None,
    api_key: str | None = None,
    server_url: str | None = None,
) -> dict:
    """Create a new workspace.

    Args:
        name: Workspace name (max 255 chars).
        description: Workspace description (max 1000 chars).
        api_key: Authentication key.
        server_url: Custom server URL.

    Returns:
        dict: Workspace data including id, name, and description.
    """

def tlog(
    name: str,
    value: float | int,
    step: int | None = None,
    metadata: dict | None = None,
) -> None:
    """Log a metric using the global experiment session.

    Args:
        name: Metric name.
        value: Metric value (int or float).
        step: Optional step number.
        metadata: Optional metadata dictionary (max 10KB).

    Raises:
        ToraError: If called before :func:`setup`.
    """

def tresult(
    name: str,
    value: float | int,
) -> None:
    """Log a result using the global experiment session.

    Args:
        name: Result name.
        value: Result value (int or float).
    """

def flush() -> None:
    """Send all buffered metrics immediately."""

def shutdown() -> None:
    """Flush remaining metrics and close the global client."""

def is_initialized() -> bool:
    """Return whether the global client is initialized."""

def get_experiment_id() -> str | None:
    """Return the current experiment ID, if available."""

def get_experiment_url() -> str | None:
    """Return the current experiment URL, if available."""
```

## Classes

### Tora

```python
class Tora:
    @classmethod
    def create_experiment(
        cls,
        name: str,
        workspace_id: str | None = None,
        description: str | None = None,
        hyperparams: dict | None = None,
        tags: list[str] | None = None,
        max_buffer_len: int = 25,
        api_key: str | None = None,
        server_url: str | None = None,
    ) -> "Tora":
        """Create and return a new Tora client.

        Args:
            name: Experiment name.
            workspace_id: Target workspace ID.
            description: Experiment description.
            hyperparams: Hyperparameter dictionary.
            tags: List of experiment tags.
            max_buffer_len: Buffer size. Defaults to 25.
            api_key: Authentication key.
            server_url: Custom server URL.

        Returns:
            Tora: A client instance for the created experiment.
        """

    @classmethod
    def load_experiment(
        cls,
        experiment_id: str,
        max_buffer_len: int = 25,
        api_key: str | None = None,
        server_url: str | None = None,
    ) -> "Tora":
        """Load an existing experiment and return a Tora client.

        Args:
            experiment_id: ID of an existing experiment.
            max_buffer_len: Buffer size. Defaults to 25.
            api_key: Authentication key.
            server_url: Custom server URL.

        Returns:
            Tora: A client instance for the existing experiment.
        """

    def log(
        self,
        name: str,
        value: float | int,
        step: int | None = None,
        metadata: dict | None = None,
    ) -> None:
        """Log a single metric with buffering.

        Args:
            name: Metric name.
            value: Metric value (int or float).
            step: Optional step number.
            metadata: Optional metadata dictionary (max 10KB).
        """

    def result(
        self,
        name: str,
        value: float | int,
    ) -> None:
        """Log a result value with buffering.

        Args:
            name: Result name.
            value: Result value (int or float).
        """

    def log_metrics(
        self,
        metrics: dict[str, float | int],
        step: int | None = None,
    ) -> None:
        """Log multiple metrics at once.

        Args:
            metrics: Mapping of metric names to values.
            step: Optional step number applied to all metrics.
        """

    def flush(self) -> None:
        """Send all buffered metrics immediately."""

    def shutdown(self) -> None:
        """Flush all metrics and close the client."""

    @property
    def experiment_id(self) -> str:
        """Experiment ID.

        Returns:
            str: Identifier for this experiment.
        """

    @property
    def url(self) -> str:
        """Experiment URL.

        Returns:
            str: URL for viewing this experiment.
        """

    @property
    def max_buffer_len(self) -> int:
        """Maximum buffer length.

        Returns:
            int: Current maximum buffer size.
        """

    @property
    def buffer_size(self) -> int:
        """Current buffer size.

        Returns:
            int: Number of buffered metrics.
        """

    @property
    def is_closed(self) -> bool:
        """Whether the client is closed.

        Returns:
            bool: True if the client is closed.
        """
```

#### Context Manager

```python
with Tora.create_experiment("name") as client:
    client.log("metric", 1.0)
```

---

## Configuration

### Authentication

For functions that accept an `api_key` parameter, you can either:

- Set the `TORA_API_KEY` environment variable (recommended)
- Pass the API key directly as a function parameter
