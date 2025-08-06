# What is Tora?

**Tora** is an experiment tracker built for speed and simplicity.

Track metrics, hyperparameters, and experiment metadata with just 2 lines of code.

---

## Core APIs

### Global API Functions

**NAME**
Global API - Simple experiment tracking functions

**SYNOPSIS**
`python
       import tora
       tora.setup(name, workspace_id=None, ...)
       tora.tlog(name, value, step=None, metadata=None)
       `

**DESCRIPTION**
Global API functions provide a simple interface for single experiment workflows.
Uses a global client instance for immediate logging with minimal setup.

**FUNCTIONS**

**setup**(name, workspace_id=None, description=None, hyperparams=None, tags=None, api_key=None, server_url=None, max_buffer_len=1)
Initialize global experiment session.

       name            Experiment name (required)
       workspace_id    Target workspace ID
       description     Experiment description
       hyperparams     Hyperparameter dictionary
       tags            List of experiment tags
       api_key         Authentication key
       server_url      Custom server URL
       max_buffer_len  Buffer size (default: 1)

       Returns experiment ID string. Prints experiment URL to console.

**tlog**(name, value, step=None, metadata=None)
Log metrics using global experiment.

       name      Metric name (required)
       value     Metric value - int or float (required)
       step      Step number
       metadata  Additional metadata dict (max 10KB)

       Note: Must call setup() first.

**flush**()
Force immediate sending of all buffered metrics.

**shutdown**()
Cleanup global client and flush remaining metrics.

**is_initialized**()
Check if global client is initialized.
Returns bool.

**get_experiment_id**()
Get current experiment ID.
Returns experiment ID string or None.

### Tora Client Class

**NAME**
Tora - Main client for experiment tracking

**SYNOPSIS**
`python
       client = Tora.create_experiment(name, workspace_id=None, ...)
       client = Tora.load_experiment(experiment_id, ...)
       `

**DESCRIPTION**
The Tora class provides methods for logging metrics, managing experiments,
and supports both buffered and immediate metric logging.

**CLASS METHODS**

**Tora.create_experiment**(name, workspace_id=None, description=None, hyperparams=None, tags=None, max_buffer_len=25, api_key=None, server_url=None)
Create new experiment and return Tora client instance.

       name            Experiment name (required)
       workspace_id    Target workspace ID
       description     Experiment description
       hyperparams     Hyperparameter dictionary
       tags            List of experiment tags
       max_buffer_len  Buffer size (default: 25)
       api_key         Authentication key
       server_url      Custom server URL

**Tora.load_experiment**(experiment_id, max_buffer_len=25, api_key=None, server_url=None)
Load existing experiment and return Tora client instance.

       experiment_id   ID of existing experiment (required)
       max_buffer_len  Buffer size (default: 25)
       api_key         Authentication key
       server_url      Custom server URL

**INSTANCE METHODS**

**log**(name, value, step=None, metadata=None)
Log single metric with automatic buffering.

       name      Metric name (required)
       value     Metric value - int or float (required)
       step      Step number
       metadata  Additional metadata dict (max 10KB)

**log_metrics**(metrics, step=None)
Log multiple metrics at once.

       metrics   Dictionary of metric names to values (required)
       step      Step number for all metrics

**flush**()
Force immediate sending of all buffered metrics.

**shutdown**()
Flush all metrics and close client. Required for cleanup.

**PROPERTIES**

**experiment_id**
Get experiment ID string (read-only)

**url**
Get experiment URL string (read-only)

**max_buffer_len**
Get/set maximum buffer length (read-write)

**buffer_size**
Get current number of buffered metrics (read-only)

**is_closed**
Check if client is closed - returns bool (read-only)

**CONTEXT MANAGER**
Supports 'with' statement for automatic cleanup:

       ```python
       with Tora.create_experiment("name") as client:
           client.log("metric", 1.0)
       ```

### Workspace Management

**NAME**
create_workspace - Create new workspace for organizing experiments

**SYNOPSIS**
`python
       import tora
       workspace = tora.create_workspace(name, description=None, api_key=None, server_url=None)
       `

**DESCRIPTION**
Creates a new workspace for organizing experiments. Requires authentication.

**FUNCTION**

**create_workspace**(name, description=None, api_key=None, server_url=None)
Create new workspace.

       name         Workspace name - max 255 chars (required)
       description  Workspace description - max 1000 chars
       api_key      Authentication key (required)
       server_url   Custom server URL

       Returns workspace data dictionary with id, name, description, etc.

---

## Configuration

**ENVIRONMENT VARIABLES**

**AUTHENTICATION**
Tora operates in anonymous mode by default, allowing you to track experiments without any authentication. Authentication is only required if you want to:

- Associate experiments with your user account
- Create and manage workspaces
- Access workspace-specific features and collaboration tools

For functions that accept an `api_key` parameter, you can either:

- Set the `TORA_API_KEY` environment variable (recommended)
- Pass the API key directly as a function parameter

When using the environment variable, simply set it in your shell:

```bash
export TORA_API_KEY=your_api_key_here
```

Or in Python:

```python
import os
os.environ['TORA_API_KEY'] = 'your_api_key_here'
```

---

## Exception Classes

**NAME**
Tora Exceptions - Error handling classes

**DESCRIPTION**
Exception hierarchy for error handling and debugging.

**EXCEPTIONS**

**ToraError**
Base exception class for all Tora errors

**ToraValidationError**
Input validation errors

**ToraNetworkError**
Network-related errors

**ToraAPIError**
API response errors

**ToraAuthenticationError**
Authentication errors

**ToraConfigurationError**
Configuration errors

**ToraExperimentError**
Experiment-related errors

**ToraMetricError**
Metric logging errors

**ToraWorkspaceError**
Workspace-related errors

**ToraTimeoutError**
Request timeout errors
