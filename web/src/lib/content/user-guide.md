# What is Tora?

**Tora** is an experiment tracker built for speed and simplicity. Designed specifically for ML engineers and data scientists who want to track experiments without the complexity.

Track metrics, hyperparameters, and experiment metadata with just 2 lines of code.

---

## Core APIs

### Global API Functions

#### **`setup()`** - Initialize Global Experiment

Creates a global experiment session for simple logging workflows.

**Parameters:**

- **`name`** _(string)_ - Experiment name
- **`workspace_id`** _(string, optional)_ - Target workspace ID
- **`description`** _(string, optional)_ - Experiment description
- **`hyperparams`** _(dict, optional)_ - Hyperparameter dictionary
- **`tags`** _(list, optional)_ - Experiment tags
- **`api_key`** _(string, optional)_ - Authentication key
- **`server_url`** _(string, optional)_ - Custom server URL
- **`max_buffer_len`** _(int, optional)_ - Buffer size (default: 1)

**Returns:** Experiment ID string

Creates an experiment with immediate logging and prints the experiment URL to console.

#### **`tlog()`** - Log Metrics

Simple logging function that uses the global experiment created by `setup()`.

**Parameters:**

- **`name`** _(string)_ - Metric name
- **`value`** _(int|float)_ - Metric value
- **`step`** _(int, optional)_ - Step number
- **`metadata`** _(dict, optional)_ - Additional metadata (max 10KB)

**Note:** Must call `setup()` before using `tlog()`.

#### **`flush()`** - Flush Buffered Metrics

Forces immediate sending of all buffered metrics.

#### **`shutdown()`** - Cleanup Global Client

Shuts down the global client and flushes all remaining metrics.

#### **`is_initialized()`** - Check Initialization Status

**Returns:** `True` if global client is initialized, `False` otherwise

#### **`get_experiment_id()`** - Get Current Experiment ID

**Returns:** Experiment ID string or `None` if not initialized

#### **`get_experiment_url()`** - Get Experiment Web URL

**Returns:** Web URL string or `None` if not initialized

### Client Class API

#### **`Tora.create_experiment()`** - Create New Experiment

Creates a new experiment and returns a client instance.

**Parameters:**

- **`name`** _(string)_ - Experiment name
- **`workspace_id`** _(string, optional)_ - Target workspace ID
- **`description`** _(string, optional)_ - Experiment description
- **`hyperparams`** _(dict, optional)_ - Hyperparameter dictionary
- **`tags`** _(list, optional)_ - Experiment tags
- **`max_buffer_len`** _(int, optional)_ - Buffer size (default: 25)
- **`api_key`** _(string, optional)_ - Authentication key
- **`server_url`** _(string, optional)_ - Custom server URL

**Returns:** `Tora` client instance

#### **`Tora.load_experiment()`** - Load Existing Experiment

Loads an existing experiment and returns a client instance.

**Parameters:**

- **`experiment_id`** _(string)_ - ID of existing experiment
- **`max_buffer_len`** _(int, optional)_ - Buffer size (default: 25)
- **`api_key`** _(string, optional)_ - Authentication key
- **`server_url`** _(string, optional)_ - Custom server URL

**Returns:** `Tora` client instance

#### **`client.log()`** - Log Metrics

Log a metric value with automatic buffering.

**Parameters:**

- **`name`** _(string)_ - Metric name
- **`value`** _(int|float)_ - Metric value
- **`step`** _(int, optional)_ - Step number
- **`metadata`** _(dict, optional)_ - Additional metadata (max 10KB)

Metrics are buffered and sent in batches when buffer reaches `max_buffer_len`.

#### **`client.flush()`** - Flush Client Buffer

Forces immediate sending of all buffered metrics for this client.

#### **`client.shutdown()`** - Shutdown Client

Flushes remaining metrics and closes the client connection.

### Workspace Management

#### **`create_workspace()`** - Create New Workspace

Creates a new workspace for organizing experiments.

**Parameters:**

- **`name`** _(string)_ - Workspace name (max 255 chars)
- **`description`** _(string, optional)_ - Workspace description (max 1000 chars)
- **`api_key`** _(string, optional)_ - Authentication key (required)
- **`server_url`** _(string, optional)_ - Custom server URL

**Returns:** Workspace data dictionary with `id`, `name`, `description`, etc.

---

## Configuration

### Environment Variables

- **`TORA_API_KEY`** - API key for authentication
- **`TORA_BASE_URL`** - Custom server URL (default: https://tora-1030250455947.us-central1.run.app/api)

### Authentication

Tora operates in anonymous mode by default. For workspace features and collaboration, provide an API key via environment variable or function parameter.

### Buffering

- **Global API**: Default buffer size is 1 (immediate sending)
- **Client API**: Default buffer size is 25 (batched sending)
- Metrics are automatically flushed when buffer is full or during shutdown
- Call `flush()` to force immediate sending

---

## Exception Classes

- **`ToraError`** - Base exception class
- **`ToraValidationError`** - Input validation errors
- **`ToraNetworkError`** - Network-related errors
- **`ToraAPIError`** - API response errors
- **`ToraAuthenticationError`** - Authentication errors
- **`ToraConfigurationError`** - Configuration errors
- **`ToraExperimentError`** - Experiment-related errors
- **`ToraMetricError`** - Metric logging errors
- **`ToraWorkspaceError`** - Workspace-related errors
- **`ToraTimeoutError`** - Request timeout errors
