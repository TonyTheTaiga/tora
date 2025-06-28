# Why Choose Tora?

- **Zero Configuration** - Works out of the box with sensible defaults
- **Instant Visualization** - Automatic URLs for web-based experiment tracking
- **No Ops** - Nothing to deploy, just start tracking.
- **TODO** - Add more features

---

## What is Tora?

**Tora** is an experiment tracker built for speed and simplicity. Designed specifically for ML engineers and data scientists who want to track experiments without the complexity.

Track metrics, hyperparameters, and experiment metadata with just 2 lines of code.

---

## Core APIs

### **`setup()`** - Initialize Experiment

Creates a global experiment session for simple logging workflows.

**Essential Parameters:**

- **`name`** _(string)_ - Experiment name
- **`hyperparams`** _(dict, optional)_ - Hyperparameter dictionary
- **`tags`** _(list, optional)_ - Experiment tags

**Advanced Parameters:**

- **`workspace_id`** _(string, optional)_ - Target workspace ID
- **`description`** _(string, optional)_ - Experiment description
- **`api_key`** _(string, optional)_ - Authentication key

Creates an experiment with immediate logging and prints the experiment URL to console.

### **`tlog()`** - Log Metrics

Simple logging function that uses the global experiment created by `setup()`.

**Parameters:**

- **`name`** _(string)_ - Metric name
- **`value`** _(string|float|int)_ - Metric value
- **`step`** _(int)_ - Step number (required)
- **`metadata`** _(dict, optional)_ - Additional metadata

**Note:** Must call `setup()` before using `tlog()`.

---

## Configuration

### Environment Variables

- **`TORA_API_KEY`** - API key for authentication
- **`TORA_BASE_URL`** - Custom server URL

### Authentication

Tora operates in anonymous mode by default. For workspace features and collaboration, provide an API key via environment variable or function parameter.
