# Tora Project Features

## Python Client (tora.py)

The Tora Python client (`tora.py`) provides a convenient way to interact with the Tora server for experiment tracking and management directly from your Python scripts. It allows you to create experiments, log metrics, and manage your machine learning runs seamlessly.

### Key Features:

*   **Experiment Creation (`Tora.create_experiment()`)**:
    *   Allows users to programmatically create new experiments in Tora.
    *   Parameters:
        *   `name` (str): The name of the experiment.
        *   `description` (str, optional): A description for the experiment.
        *   `hyperparams` (dict, optional): A dictionary of hyperparameters to be associated with the experiment. These are typically converted using `hp_to_tora_format`.
        *   `tags` (list[str], optional): A list of tags to categorize the experiment.
        *   `workspace_id` (str, optional): The ID of the workspace to create the experiment in.
        *   `server_url` (str, optional): The base URL of the Tora server. Can also be configured via the `TORA_BASE_URL` environment variable.

*   **Loading Existing Experiments (`Tora.load_experiment()`)**:
    *   Enables fetching and interacting with an experiment that already exists in Tora.
    *   Parameters:
        *   `experiment_id` (str): The unique identifier of the experiment to load.
        *   `server_url` (str, optional): The base URL of the Tora server. Can also be configured via the `TORA_BASE_URL` environment variable.

*   **Logging Metrics (`tora.log()`)**:
    *   Used to log metrics (e.g., loss, accuracy) for an active experiment.
    *   Parameters:
        *   `name` (str): The name of the metric (e.g., "accuracy", "loss").
        *   `value` (float or int): The value of the metric.
        *   `step` (int, optional): The step number (e.g., epoch, batch number) for the metric.
        *   `metadata` (dict, optional): Additional metadata to store with the metric.
    *   Metrics are buffered locally (buffer size controlled by `max_buffer_len`) before being sent to the server. The `tora.shutdown()` function should be called at the end of your script to ensure all buffered logs are sent.

*   **Configuration**:
    *   The Tora server URL can be configured using the `TORA_BASE_URL` environment variable.
    *   API key authentication is used to secure communication with the Tora server. The client handles API key management, typically read from an environment variable or a configuration file.
    *   Note: Currently, the Python client uses hardcoded API keys. For production or security-sensitive environments, it is strongly recommended to modify the client to read these keys from environment variables or a secure configuration file.

*   **Helper Functions**:
    *   `hp_to_tora_format(hyperparams: dict) -> list`: Converts a dictionary of hyperparameters into the list format expected by the Tora API for creation.
    *   `hp_from_tora_format(hyperparams_list: list) -> dict`: Converts a list of hyperparameters from the Tora API format back into a dictionary.

## Web Application (Svelte Frontend & API)

The Tora Web Application provides a user-friendly interface and a robust API for managing and visualizing machine learning experiments. Built with Svelte for the frontend and a corresponding backend API, it complements the Python client by offering a centralized platform for experiment oversight.

### Key Features:

*   **User Authentication**: Securely manages access to experiment data, ensuring that users can only view and manage experiments they are authorized for.
*   **Experiment Dashboard**: Offers a comprehensive view of all experiments, allowing users to quickly see the status and key details of their ongoing and completed runs.
*   **Experiment Management**:
    *   Provides UI and API capabilities to create, update, and delete experiments.
    *   Relevant API Endpoints:
        *   `POST /api/experiments`: Create a new experiment.
        *   `GET /api/experiments`: Retrieve a list of experiments.
        *   `GET /api/experiments/{id}`: Retrieve a specific experiment by its ID.
        *   `PUT /api/experiments/{id}`: Update an existing experiment.
        *   `DELETE /api/experiments/{id}`: Delete an experiment.
*   **Metric Logging and Visualization**:
    *   Supports batch logging of metrics through the API for efficiency.
        *   API Endpoint: `POST /api/experiments/{id}/metrics/batch`.
    *   Logged metrics are visualized in the UI, providing charts and graphs to track experiment performance over time.
*   **Experiment Comparison**:
    *   Allows users to compare the metrics, hyperparameters, and outcomes of different experiments side-by-side.
    *   Includes a system for setting a "reference experiment" to benchmark against.
    *   Corresponding API route structure: `web/src/routes/api/experiments/[slug]/ref/`.
*   **Workspace Management**:
    *   Enables the organization of experiments into logical groups called "workspaces" for better project management and collaboration.
    *   API Endpoints:
        *   `POST /api/workspaces`: Create a new workspace.
        *   `GET /api/workspaces`: Retrieve a list of workspaces.
        *   (Update/delete operations are typically available for specific workspace IDs, e.g., `PUT /api/workspaces/{id}`, `DELETE /api/workspaces/{id}`).
    *   Integrated with the Python client via the `workspace_id` parameter in `Tora.create_experiment()`.
*   **API Key Management**:
    *   Provides a secure way to generate, manage, and revoke API keys used for authenticating the Python client and other API interactions.
    *   API endpoint structure: `web/src/routes/api/keys/`.
*   **AI-Powered Experiment Analysis**:
    *   Offers advanced analysis capabilities by leveraging AI to provide insights or suggestions based on the logged experiment data. This can help in identifying trends, potential issues, or areas for optimization.
    *   API endpoint structure: `web/src/routes/api/ai/analysis/`.
