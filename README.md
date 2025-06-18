# Tora: ML Experiment Management Tool

## Introduction

Tora is a tool designed to help you manage and track your machine learning experiments. It consists of two main components:

* **Python Client/Library**: Allows you to log experiment data (parameters, metrics, artifacts, etc.) directly from your Python scripts.
* **Svelte-based Web Application**: Provides a user-friendly interface to visualize, compare, and manage your logged experiments.

The primary benefit of Tora is to provide a streamlined way to keep track of your ML experimentation workflow, making it easier to compare results, reproduce experiments, and manage your projects.

## Capabilities

Tora provides a comprehensive ML experiment management platform with the following capabilities:

### 🐍 Python Client Library (`tora.py`)

**Core Experiment Management:**
* **Simple Integration**: Add experiment tracking with just 3 lines of code to any Python ML script
* **Framework Agnostic**: Works seamlessly with PyTorch, TensorFlow, scikit-learn, and any Python ML framework
* **Experiment Creation**: Create new experiments with `create_experiment()` including name, description, and metadata
* **Hyperparameter Tracking**: Automatically serialize and log model configurations, training parameters, and settings
* **Metric Logging**: Log any numeric metric with `log()` including support for step numbers and custom metadata
* **Buffered Logging**: Efficient batch metric submission with configurable buffer sizes (default: 25 metrics)
* **Graceful Shutdown**: Ensure all buffered data is sent with `shutdown()` for reliable data persistence
* **Error Handling**: Built-in resilience with safe failure modes and network error recovery
* **Lightweight**: Minimal dependencies (just `httpx`) for fast installation and minimal overhead

**Advanced Features:**
* **Experiment References**: Link experiments together to track model iterations and hyperparameter sweeps
* **Flexible Metadata**: Attach custom JSON metadata to experiments for rich context
* **Real-time Streaming**: Live metric updates during model training
* **Configurable Endpoints**: Override server URLs for custom deployments

### 🌐 Web Application (Modern Svelte Frontend)

**Experiment Management Interface:**
* **Interactive Dashboard**: Modern, responsive interface with dark/light theme support
* **Experiment CRUD**: Create, read, update, delete experiments with intuitive forms
* **Advanced Filtering**: Search and filter experiments by name, tags, date, performance metrics
* **Experiment Details**: Comprehensive view of hyperparameters, metrics, and metadata
* **Bulk Operations**: Multi-select for batch experiment management

**Data Visualization & Analysis:**
* **Interactive Charts**: Chart.js-powered visualizations with zoom, pan, and multi-metric plotting
* **Real-time Updates**: Live metric streaming during active training runs
* **Metric Comparison**: Side-by-side comparison of multiple experiments with synchronized axes
* **Performance Tables**: Detailed tabular view of all logged metrics with sorting and filtering
* **Chart Customization**: Toggle between different chart types and metric combinations
* **Export Capabilities**: Download charts and data for external analysis

**Collaboration & Organization:**
* **Multi-user Workspaces**: Shared environments for team collaboration with role-based access
* **User Authentication**: Secure login/signup with Supabase Auth integration
* **Experiment Sharing**: Public/private experiment visibility controls
* **Team Management**: Invite collaborators and manage workspace permissions
* **Activity Tracking**: Audit logs of experiment modifications and access

**AI-Powered Intelligence:**
* **Hyperparameter Optimization**: AI-driven recommendations for parameter tuning
* **Performance Analysis**: Automated insights into experiment performance patterns
* **Anomaly Detection**: Identify unusual training behaviors and potential issues
* **Trend Analysis**: Understand performance evolution across experiment iterations

### 🏗️ Technical Infrastructure

**Backend & Database:**
* **Supabase Integration**: PostgreSQL database with real-time subscriptions
* **RESTful API**: Comprehensive API endpoints for all experiment operations
* **Authentication & Authorization**: Secure JWT-based auth with row-level security
* **Real-time Sync**: Live updates across all connected clients
* **Scalable Architecture**: Cloud-ready deployment with Google Cloud Platform support

**API Capabilities:**
* **Experiment Endpoints**: Full CRUD operations for experiments and metadata
* **Metrics Management**: Batch metric ingestion with efficient storage
* **User Management**: Account creation, authentication, and workspace management
* **Reference System**: Link and track experiment relationships
* **AI Analysis**: Integration endpoints for intelligent recommendations

**Deployment & Operations:**
* **Cloud Deployment**: Ready-to-deploy configurations for GCP Cloud Run (secrets loaded via environment variables)
* **Docker Support**: Containerized deployment options
* **Environment Management**: Comprehensive configuration via environment variables
* **Migration System**: Database schema versioning with Supabase migrations
* **Monitoring**: Built-in logging and error tracking capabilities

### 📊 Example Use Cases Supported

**Computer Vision:**
* Image classification with ResNet architectures (Imagenette dataset example)
* Transfer learning experiments with pre-trained models
* Custom CNN architectures for specialized vision tasks

**Natural Language Processing:**
* Text classification with transformer models
* Topic modeling with LDA and advanced techniques
* Document embedding and similarity analysis

**Traditional Machine Learning:**
* MNIST digit classification with various algorithms
* Logistic regression with hyperparameter optimization
* Recommendation systems with collaborative filtering

**Deep Learning Research:**
* Custom model architectures and ablation studies
* Hyperparameter sweeps with automated tracking
* Multi-metric optimization experiments

**Production ML Workflows:**
* Model performance monitoring across versions
* A/B testing of different model configurations
* Reproducible experiment pipelines with version control

### 🎯 Key Differentiators

* **Zero Learning Curve**: Integrate into existing workflows without changing code structure
* **Enterprise-Ready**: Scalable architecture with team collaboration features
* **AI-Enhanced**: Intelligent insights beyond basic metric tracking
* **Open Source**: Transparent, customizable, and community-driven development
* **Modern Stack**: Built with cutting-edge technologies (Svelte, TypeScript, Supabase)
* **Real-time Everything**: Live updates, streaming metrics, and instant collaboration

## Project Structure

The Tora project is organized into two main components:

### 1. Python Logger (`python/`)

* **Role**: This is a client library designed to be integrated into your machine learning training scripts.
* **Key File**: The core logic resides in `python/tora/tora.py`, which contains the `Tora` class.
* **Functionality**: It's responsible for sending experiment data, such as hyperparameters, metrics (loss, accuracy, etc.), and tags, to the backend of the web application.
* **Example Scripts**: The `python/scripts/` directory provides usage examples. For instance, `imagenette_training.py` demonstrates how to use the `Tora` library in a typical training scenario.

### 2. Web Application (`web/`)

* **Role**: This is a web interface for viewing, managing, and analyzing your logged experiments.
* **Technology**: The frontend is built using Svelte.
* **Backend**: The application includes a backend that handles API requests. These are used for managing experiment data, user authentication, and other related functionalities. API routes are typically defined within the Svelte application structure (e.g., under `web/src/routes/api/`).
* **Frontend**: The Svelte-based frontend provides the user interface for creating, viewing, updating, and deleting experiments, as well as visualizing their metrics.
* **Database**: Experiment data, user information, and other persistent data are stored in a database. (The project uses Supabase, as indicated by the `web/supabase/` directory).

## Setup and Installation

This section guides you through setting up both the Python logger and the Svelte web application.

### 1. Python Logger (`python/`)

The Python logger is used to send experiment data from your training scripts to the Tora web application.

**Prerequisites:**

* Python (3.8 or newer recommended)
* `uv` (Python package installer, can be installed via `pip install uv`)

**Setup Instructions:**

1. **Navigate to the Python directory:**

    ```bash
    cd python
    ```

2. **Create and activate a virtual environment (recommended):**

    ```bash
    python -m venv .venv
    source .venv/bin/activate  # On Windows: .venv\Scripts\activate
    ```

3. **Install the Tora client library:**
    The core dependency for the client is `httpx`. You can install the client and its dependencies using `uv`:

    ```bash
    uv pip install .
    ```

    This will install the `tora-client` package defined in `pyproject.toml`.

4. **Install dependencies for example scripts (optional):**
    The example scripts in `python/scripts/` require additional libraries such as `torch`, `torchvision`, `scikit-learn`, and `numpy`. To run these examples, install the dependencies:

    ```bash
    uv pip install torch torchvision scikit-learn numpy
    ```

**Configuration:**

* The Python client is pre-configured to send data to `http://localhost:5173/api`. Ensure the Tora web application is running and accessible at this address.
* The client now reads its API key from the environment variable `TORA_API_KEY`. If this variable is not set, it falls back to a built-in development key. Configure the web application backend to accept the key you provide via this environment variable.

### 2. Web Application (`web/`)

The web application provides the interface to view and manage your experiments. It's built with Svelte and uses `pnpm` for package management.

**Prerequisites:**

* Node.js (LTS version recommended)
* `pnpm` (can be installed via `npm install -g pnpm`)

**Setup Instructions:**

1. **Navigate to the web application directory:**

    ```bash
    cd web
    ```

2. **Install dependencies:**

    ```bash
    pnpm install
    ```

3. **Environment Variables:**
    The web application relies on Supabase for its backend and database. You will need to set up a Supabase project (either locally or on Supabase Cloud) and configure the necessary environment variables.
    Create a `.env` file in the `web/` directory by copying from a `.env.example` (if one was provided, otherwise create it manually).
    The essential variables are:
    * `VITE_SUPABASE_URL`: The URL of your Supabase project.
    * `VITE_SUPABASE_ANON_KEY`: The anonymous key for your Supabase project.

    You can find these in your Supabase project settings. For local Supabase development, these are typically provided when you run `supabase start`.

    Additionally, ensure the API service is configured to:
    * Run on port `5173` to match the Python client's default `TORA_BASE_URL`.
    * Accept the hardcoded API keys used by the Python client (see Python Logger configuration). This usually involves setting up API authentication middleware or an API gateway that checks the `x-api-key` header.

4. **Run the development server:**

    ```bash
    pnpm run dev
    ```

    This will typically start the application on `http://localhost:5173`.

**Notes on Supabase Integration:**

* The `web/supabase/config.toml` file contains configurations for local Supabase development. You can use the Supabase CLI to manage your local instance (e.g., `supabase start`, `supabase stop`, `supabase db reset`).
* Migrations for the database schema are located in `web/supabase/migrations/`. When setting up a new Supabase project, you'll need to apply these migrations.

## Usage / Quick Start

This section provides a quick guide to get Tora up and running with an example project.

**1. Prerequisites:**

* Ensure you have followed the **"Setup and Installation"** guide for both the Python Logger and the Web Application.
* The **Web Application must be running**. If you followed the setup, navigate to the `web/` directory and run:

    ```bash
    pnpm run dev
    ```

    This should make the web application accessible at `http://localhost:5173`.

**2. Run an Example ML Script:**

This step will execute a sample machine learning training script that uses the Tora Python client to log experiment data. We'll use the `imagenette_training.py` script as an example.

1. **Navigate to the Python directory:**

    ```bash
    cd python  # If you are not already there
    ```

2. **Activate your Python virtual environment:**

    ```bash
    source .venv/bin/activate  # On Windows: .venv\Scripts\activate
    ```

    *(If you didn't create one during setup, please refer back to the Python Logger setup instructions).*

3. **Run the example script:**

    ```bash
    python scripts/imagenette_training.py
    ```

    This script will:
    * Download the Imagenette dataset (a small subset of ImageNet).
    * Train a ResNet34 model on this dataset for a few epochs.
    * Use the `Tora` client to automatically create an experiment and log its name (e.g., "Imagenette_ResNet34"), description, hyperparameters (like learning rate, batch size), and metrics (like training/validation loss and accuracy) to your Tora web application.

**3. View Results in the Web Application:**

Once the script has started running (or completed), you can view the logged experiment in the Tora web interface.

1. **Open your web browser** and navigate to `http://localhost:5173` (or the address where your Svelte application is running).
2. **Log in** if your application requires authentication.
3. You should see a new experiment, likely named "Imagenette_ResNet34", listed on the main dashboard or experiments page.
4. **Click on this experiment** to view its detailed information, including:
    * The description and tags.
    * The hyperparameters used for the training run.
    * Graphs and tables of the metrics logged over time (e.g., loss, accuracy).

This quick start should give you a basic understanding of how to use Tora to track your machine learning experiments. You can adapt the example scripts or integrate the `Tora` client into your own projects following a similar pattern.

## Configuration

This section details key configuration options for both the Python client and the web application.

### 1. Python Client (`Tora` library - `python/tora/tora.py`)

The Python client offers several ways to customize its behavior:

* **API Server URL:**
  * **Default:** The client is hardcoded to communicate with `http://localhost:5173/api` via the `TORA_BASE_URL` constant in `python/tora/tora.py`.
  * **Customization:** You can override this default when initializing the `Tora` object or creating an experiment by passing the `server_url` parameter. This is useful if your Tora web application is hosted on a different domain or port.

        ```python
        from tora import Tora

        # When creating a new experiment
        tora_client = Tora.create_experiment(
            name="My Custom Server Experiment",
            server_url="https://my-tora-instance.com/api"
        )

        # Or when loading an existing experiment
        # tora_client = Tora(
        #     experiment_id="your_experiment_id",
        #     server_url="https://my-tora-instance.com/api"
        # )
        ```

* **API Key:**
  * The Python client reads its API key from the `TORA_API_KEY` environment variable. If not set, a development key is used by default.
  * **Server-Side:** Configure the web application's backend to validate the key provided in the `x-api-key` header. Avoid committing real keys to version control.

* **Log Buffer Size:**
  * The `Tora` class constructor accepts a `max_buffer_len` parameter, which defaults to `25`.
  * **Purpose:** Metrics logged via `tora_client.log()` are initially stored in a buffer. They are sent to the server in a batch when the number of buffered items reaches `max_buffer_len` or when `tora_client.shutdown()` is called. Adjust this value based on your logging frequency and network considerations.

        ```python
        tora_client = Tora(experiment_id="your_experiment_id", max_buffer_len=50)
        ```

### 2. Web Application (`web/`)

Configuration for the Svelte-based web application is primarily managed through environment variables and server settings.

* **Environment Variables (`web/.env`):**
  * As mentioned in the "Setup and Installation" section, you need to create a `.env` file in the `web/` directory.
  * **Essential Supabase Variables:**
    * `VITE_SUPABASE_URL`: The URL of your Supabase project.
    * `VITE_SUPABASE_ANON_KEY`: The anonymous public key for your Supabase project.
  * **Other Variables:** Review `web/supabase/config.toml` and any cloud deployment configurations (e.g., `Dockerfile`, `cloudbuild.yaml`) for other potential environment variables, especially if you plan to use features like:
    * Email services (e.g., for password resets, invitations) might require SMTP server details or API keys (e.g., `SENDGRID_API_KEY` if using SendGrid with Supabase).
    * Third-party OAuth providers (e.g., Google, GitHub) will require client IDs and secrets.
    * Specific Supabase features like `OPENAI_API_KEY` for Supabase AI.
  * Always refer to the Supabase documentation and your project's specific needs for a complete list of required environment variables.

* **API Key Configuration (Server-side):**
  * The API endpoints within the SvelteKit application (likely under `web/src/routes/api/`) must be configured to validate the `x-api-key` header sent by the Python client.
  * This involves implementing logic in your server-side code (e.g., in SvelteKit `hooks.server.ts` or directly in your API route handlers) to check if the received API key matches the expected valid key(s).
  * **This is critical for securing your API endpoints.** Ensure that the keys used by the Python client are securely stored and accessed by the server for validation.

* **Port Configuration:**
  * The SvelteKit development server (started with `pnpm run dev`) defaults to port `5173`.
  * **Changing the Port:**
    * You can specify a different port using the `--port` flag:

            ```bash
            pnpm run dev -- --port 3000
            ```

    * Alternatively, you can modify the `vite.config.ts` file in the `web/` directory to change the default server port. See the Vite documentation for details.
  * If you change the port, ensure the Python client's `server_url` is updated accordingly.
