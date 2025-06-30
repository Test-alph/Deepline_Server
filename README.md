# Deepline

# MCP EDA & ML Monitoring Agent

This repository contains an **MCP (Model Context Protocol) server** for end-to-end data science and MLOps workflows, covering:

* **Phase 1: EDA Agent** – Data loading, inspection, visualization, profiling
* **Phase 2: Data‑Quality Gate & Auto‑Fix Loop** – Data-quality scoring and automated cleanup
* **Phase 3: Drift & Performance Monitoring** – Data drift detection, regression performance, classification metrics

With this setup, you can rapidly iterate on datasets, enforce quality gates, detect drift, and monitor model performance, all via a standardized LLM‑callable API.

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [Prerequisites](#prerequisites)
3. [Installation & Environment Setup](#installation--environment-setup)
4. [Directory Structure](#directory-structure)
5. [FastMCP Server Architecture](#fastmcp-server-architecture)
6. [Available Tools](#available-tools)

   1. [Phase 1: EDA Tools](#phase-1-eda-tools)
   2. [Phase 2: Quality Gate Tools](#phase-2-quality-gate-tools)
   3. [Phase 3: Drift & Performance Tools](#phase-3-drift--performance-tools)
7. [Running the Server](#running-the-server)
8. [Claude Desktop Integration](#claude-desktop-integration)
9. [Troubleshooting & Common Issues](#troubleshooting--common-issues)
10. [Sample Usage](#sample-usage)
11. [Testing](#testing)
12. [Roadmap & Next Steps](#roadmap--next-steps)
13. [Development & Contributing](#development--contributing)
14. [License](#license)

---

## Project Overview

We follow the **Liquid Intelligence** wire‑frame, aiming to build an **Agent Framework** (Layer 3) that:

* Exposes Python functions as LLM‑callable tools via MCP
* Supports interactive EDA and clean data pipelines
* Automates quality checks and self‑healing loops
* Monitors data drift and model performance
* Lays the foundation for training, deployment, and continuous feedback

By the end of Phase 3, the server supports:

1. **Interactive EDA** (upload a CSV → inspect & plot)
2. **Data‑quality gating** (score + retry cleanup)
3. **Drift detection** (report drift count/share)
4. **Regression metrics** (RMSE, MAE, R²)
5. **Classification metrics** (accuracy, precision, recall, F1-weighted)

## Prerequisites

* **Python ≥ 3.12** (pyenv recommended for version management)
* **uv** for environment & dependency management
* **MCP CLI** (provided by `mcp[cli]`)
* **Claude Desktop** (for MCP integration)
* **Git** (optional, for version control)

## Installation & Environment Setup

1. **Clone the repo**:

   ```bash
   git clone <repo-url> mcp-server
   cd mcp-server
   ```

2. **Create & activate a local virtualenv**:

   ```bash
   uv venv --python 3.12   # creates .venv/
   source .venv/bin/activate
   ```

3. **Install runtime dependencies**:

   ```bash
   uv pip install \
     "mcp[cli]" pandas numpy polars pyarrow \
     matplotlib seaborn plotly scipy scikit-learn \
     ydata-profiling missingno \
     xgboost ipykernel mlflow \
     evidently python-dateutil pyod
   ```

4. **Lock dependencies** for reproducibility:

   ```bash
   uv pip compile pyproject.toml \
     --output-file requirements.lock \
     --generate-hashes
   ```

5. *(Optional)* **Install dev tools**:

   ```bash
   uv pip install .[dev]   # black, ruff, mypy, pytest
   ```

6. **Verify environment**:

   ```bash
   which python && python -V
   pip list | grep evidently
   python -c "import missingno, evidently, dateutil, pyod; print('All dependencies available')"
   ```

## Dependencies & Libraries

### Core Framework
| Library | Version | Purpose |
|---------|---------|---------|
| **mcp[cli]** | Latest | Model Context Protocol framework for tool registration and server management |
| **pydantic** | Latest | Data validation and settings management for structured outputs |
| **asyncio** | Built-in | Asynchronous programming for concurrent tool execution |

### Data Processing & Analysis
| Library | Version | Purpose |
|---------|---------|---------|
| **pandas** | Latest | Primary data manipulation and analysis library |
| **numpy** | Latest | Numerical computing and array operations |
| **polars** | Latest | Fast DataFrame library for large datasets |
| **pyarrow** | Latest | Columnar data format for efficient data storage |

### Data Science & Machine Learning
| Library | Version | Purpose |
|---------|---------|---------|
| **scikit-learn** | Latest | Machine learning algorithms for classification, regression, and clustering |
| **scipy** | Latest | Scientific computing and statistical functions |
| **xgboost** | Latest | Gradient boosting framework for ML models |
| **pyod** | Latest | Outlier detection algorithms (Isolation Forest, Local Outlier Factor) |

### Data Quality & Monitoring
| Library | Version | Purpose |
|---------|---------|---------|
| **evidently** | 0.7+ | Data quality, drift detection, and model performance monitoring |
| **ydata-profiling** | Latest | Automated data profiling and quality assessment |
| **missingno** | Latest | Missing data visualization and analysis |

### Visualization
| Library | Version | Purpose |
|---------|---------|---------|
| **matplotlib** | Latest | Core plotting library for static visualizations |
| **seaborn** | Latest | Statistical data visualization built on matplotlib |
| **plotly** | Latest | Interactive plotting for web-based visualizations |

### Development & Environment
| Library | Version | Purpose |
|---------|---------|---------|
| **uv** | Latest | Fast Python package installer and resolver |
| **python-dateutil** | Latest | Date parsing and manipulation utilities |
| **ipykernel** | Latest | Jupyter kernel for interactive development |
| **mlflow** | Latest | Machine learning lifecycle management |

### Development Tools (Optional)
| Library | Version | Purpose |
|---------|---------|---------|
| **black** | Latest | Code formatting and style enforcement |
| **ruff** | Latest | Fast Python linter and formatter |
| **mypy** | Latest | Static type checking for Python |
| **pytest** | Latest | Testing framework for unit and integration tests |

### Tool-Specific Dependencies

#### EDA Tools (`load_data`, `basic_info`, `list_datasets`)
- **pandas**: CSV/Excel/JSON file loading and data manipulation
- **numpy**: Data type detection and numerical operations

#### Visualization Tools (`create_visualization`, `missing_data_analysis`)
- **matplotlib**: Core plotting functionality
- **seaborn**: Statistical visualizations
- **missingno**: Missing data matrix visualization

#### Schema Inference (`infer_schema`)
- **pandas**: Data type detection and column analysis
- **re**: Regular expressions for pattern matching (email, phone, URL, date)

#### Outlier Detection (`detect_outliers`)
- **pyod**: Isolation Forest and Local Outlier Factor algorithms
- **scikit-learn**: Statistical outlier detection methods
- **numpy**: Statistical calculations (IQR, quantiles)

#### Data Quality & Drift (`data_quality_report`, `drift_analysis`)
- **evidently**: Data quality scoring and drift detection
- **pandas**: Data preprocessing and transformation

#### Model Performance (`model_performance_report`)
- **evidently**: Regression performance metrics
- **scikit-learn**: Classification metrics (accuracy, precision, recall, F1)
- **numpy**: Statistical calculations

### System Requirements
- **Python**: ≥ 3.12 (pyenv recommended)
- **Memory**: Minimum 4GB RAM (8GB+ recommended for large datasets)
- **Storage**: 1GB+ free space for reports and temporary files
- **OS**: macOS, Linux, Windows (with WSL recommended)

### Optional Enhancements
- **Docker**: Containerization for deployment
- **PostgreSQL/MySQL**: Database integration for persistent storage
- **Redis**: Caching for improved performance
- **Slack/Email**: Alerting for drift detection and quality issues

## Directory Structure

```text
mcp-server/
├── .venv/                  # Local virtual environment
├── pyproject.toml         # Project config & dependencies
├── requirements.lock      # Hash-pinned lockfile
├── server.py              # FastMCP server implementation
├── launch_server.py       # Environment-aware launcher script
├── test_evidently_tools.py # Comprehensive test harness
├── debug_test.py          # Debug summary structure for Evidently
├── reports/               # Generated HTML dashboards
└── README.md              # ← you are here
```

## FastMCP Server Architecture

* **`server.py`** uses **FastMCP** for concise tool registration:

  ```python
  from mcp.server.fastmcp import FastMCP
  mcp = FastMCP("EDA Server")

  @mcp.tool()
  async def load_data(file_path: str, name: str) -> str:
      # load CSV/Excel/JSON → in-memory store
      ...

  # … other tools …

  if __name__ == "__main__":
      mcp.run()
  ```
* **In-memory store** guarded by an `asyncio.Lock`
* **HTML reports** saved under `reports/`, surfaced via `file://` URIs

## Available Tools

### Phase 1 – EDA Tools

| Tool                        | Description                                                      |
| --------------------------- | ---------------------------------------------------------------- |
| **`load_data`**             | Load CSV, Excel, or JSON into shared memory                      |
| **`basic_info`**            | Show shape, columns, dtypes, and first five rows                 |
| **`missing_data_analysis`** | Compute missing-value stats + render missingno matrix            |
| **`create_visualization`**  | Render histogram, boxplot, scatter, correlation, or missing plot |
| **`statistical_summary`**   | Generate `describe()` and correlation matrix                     |
| **`list_datasets`**         | List dataset names with row/column counts                        |
| **`infer_schema`**          | Infer column types, nullability, ranges, and patterns            |
| **`detect_outliers`**       | Detect outliers using IQR, Isolation Forest, or Local Outlier Factor |

### Phase 2 – Quality Gate Tools

| Tool                      | Description                                                     |
| ------------------------- | --------------------------------------------------------------- |
| **`data_quality_report`** | Run Evidently `DataSummaryPreset` → returns `score` + HTML      |
| **`fix_and_retry`**       | Auto-impute, dedup, drop constant cols; loop until `score ≥ 80` |

### Phase 3 – Drift & Performance Tools

| Tool                           | Description                                                                       |
| ------------------------------ | --------------------------------------------------------------------------------- |
| **`drift_analysis`**           | Run Evidently `DataDriftPreset`; extract `DriftedColumnsCount`                    |
| **`model_performance_report`** | Regression via Evidently `RegressionPreset`; classification via `sklearn.metrics` |

## Running the Server

### Method 1: Direct Python Execution
```bash
cd mcp-server
source .venv/bin/activate
python server.py
# Server runs in stdio mode, ready for MCP connections
```

### Method 2: Using MCP CLI
```bash
cd mcp-server
source .venv/bin/activate
mcp run server.py
# Server runs in stdio mode, ready for MCP connections
```

### Method 3: Using uv (Recommended for Claude Desktop)
```bash
cd mcp-server
uv run server.py
# Server runs in stdio mode, ready for MCP connections
```

## Claude Desktop Integration

### Prerequisites
- **Claude Desktop** installed and running
- **uv** available in your PATH (`which uv` should return a path)
- **Python 3.12+** with all dependencies installed

### Setup Instructions

1. **Verify uv is available**:
   ```bash
   which uv
   # Should return a path like: /usr/local/bin/uv or /opt/homebrew/bin/uv
   ```

2. **Configure Claude Desktop**:
   
   **macOS/Linux**:
   ```bash
   # Open the Claude Desktop config file
   code ~/Library/Application\ Support/Claude/claude_desktop_config.json
   ```
   
   **Windows (PowerShell)**:
   ```powershell
   code $env:AppData\Claude\claude_desktop_config.json
   ```

3. **Add MCP server configuration**:
   
   Replace the contents with:
   ```json
   {
     "mcpServers": {
       "liquid-intel": {
         "command": "uv",
         "args": [
           "--directory",
           "/path/to/your/mcp-server",
           "run",
           "server.py"
         ]
       }
     }
   }
   ```
   
   **Important**: Replace `/path/to/your/mcp-server` with the absolute path to your `mcp-server` directory.

4. **Save and restart Claude Desktop**:
   - Save the config file
   - Completely quit Claude Desktop
   - Restart Claude Desktop

5. **Verify connection**:
   - In Claude Desktop, you should see the "liquid-intel" MCP server available
   - The tools (`load_data`, `basic_info`, `data_quality_report`, etc.) should appear in the UI
   - You can now call these tools directly in Claude conversations

### Alternative Configuration Methods

If the `uv` method doesn't work, you can also configure Claude Desktop to use:

**Method A: Direct Python with environment variables**
```json
{
  "mcpServers": {
    "liquid-intel": {
      "command": "python",
      "args": ["launch_server.py"],
      "cwd": "/path/to/your/mcp-server",
      "env": {
        "PYTHONPATH": "/path/to/your/python/site-packages"
      }
    }
  }
}
```

**Method B: Using the launcher script**
```json
{
  "mcpServers": {
    "liquid-intel": {
      "command": "python",
      "args": ["launch_server.py"],
      "cwd": "/path/to/your/mcp-server"
    }
  }
}
```

**Note**: If using Method B with the launcher script, you'll need to update `launch_server.py` with your actual paths:
- Replace `/path/to/your/python/site-packages` with your actual Python site-packages directory
- Replace `/path/to/your/mcp-server` with your actual mcp-server directory path

## Troubleshooting & Common Issues

### Issue 1: "No module named 'missingno'" or "No module named 'dateutil'"

**Symptoms**: Server fails to start with import errors

**Solution**:
```bash
# Install missing dependencies
pip install python-dateutil missingno ydata-profiling evidently

# Or use the launcher script which sets PYTHONPATH
python launch_server.py
```

### Issue 2: Python Environment Mismatch

**Symptoms**: Dependencies installed but server still can't find them

**Root Cause**: MCP CLI using different Python environment than where dependencies are installed

**Solutions**:

1. **Use explicit PYTHONPATH**:
   ```bash
   PYTHONPATH=/path/to/your/python/site-packages mcp run server.py
   ```

2. **Use the launcher script**:
   ```bash
   python launch_server.py
   ```

3. **Use uv (recommended)**:
   ```bash
   uv run server.py
   ```

### Issue 3: Claude Desktop Can't Connect

**Symptoms**: Claude Desktop shows "Server disconnected" or tools don't appear

**Solutions**:

1. **Check Claude Desktop logs**:
   - Open Claude Desktop
   - Go to Help → Toggle Developer Tools
   - Check Console for error messages

2. **Verify server is running**:
   ```bash
   ps aux | grep "mcp run server.py"
   # or
   ps aux | grep "python.*server.py"
   ```

3. **Test server manually**:
   ```bash
   cd mcp-server
   python server.py
   # Should start without errors
   ```

4. **Check config file syntax**:
   ```bash
   cat ~/Library/Application\ Support/Claude/claude_desktop_config.json | python -m json.tool
   ```

### Issue 4: "uv not found" or "uv command not available"

**Symptoms**: Claude Desktop can't find the `uv` command

**Solutions**:

1. **Install uv**:
   ```bash
   # macOS
   brew install uv
   
   # Linux
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```

2. **Use full path in config**:
   ```json
   {
     "mcpServers": {
       "liquid-intel": {
         "command": "/full/path/to/uv",
         "args": [
           "--directory",
           "/path/to/your/mcp-server",
           "run",
           "server.py"
         ]
       }
     }
   }
   ```

### Issue 5: Evidently Version Compatibility

**Symptoms**: Evidently API errors or missing methods

**Solution**: The server is configured for Evidently 0.7+ with fallback mechanisms:
- Classification metrics use sklearn fallback
- Drift metrics extracted from Evidently's flat metrics array
- Regression metrics use Evidently's RegressionPreset

### Issue 6: Server Starts But Tools Don't Work

**Symptoms**: Server connects but tool calls fail

**Solutions**:

1. **Check tool availability**:
   ```bash
   # Test a simple tool
   python -c "
   import asyncio
   from server import mcp
   print('Available tools:', [tool.name for tool in mcp.tools])
   "
   ```

2. **Verify data files exist**:
   ```bash
   ls -la *.csv *.xlsx *.json
   ```

3. **Check reports directory**:
   ```bash
   mkdir -p reports
   ```

### Debug Commands

**Check Python environment**:
```bash
which python
python --version
pip list | grep -E "(evidently|missingno|dateutil)"
```

**Test server startup**:
```bash
cd mcp-server
python -c "import server; print('Server imports successfully')"
```

**Check MCP CLI**:
```bash
mcp --help
mcp run --help
```

**Verify Claude Desktop config**:
```bash
cat ~/Library/Application\ Support/Claude/claude_desktop_config.json
```

## Sample Usage

### Basic EDA Workflow

1. **Load a dataset**:
   ```python
   # Load iris dataset
   await load_data("iris.csv", "iris")
   ```

2. **Inspect basic info**:
   ```python
   # Get dataset overview
   await basic_info("iris")
   ```

3. **Infer schema**:
   ```python
   # Get detailed column information
   schema = await infer_schema("iris")
   print(f"Dataset: {schema.dataset_name}")
   print(f"Shape: {schema.total_rows} rows × {schema.total_columns} columns")
   
   for col in schema.columns:
       print(f"- {col.name}: {col.type} (nullable: {col.nullable})")
       if col.type == "number":
           print(f"  Range: {col.min_value} to {col.max_value}")
       elif col.type == "string" and col.pattern:
           print(f"  Pattern: {col.pattern}")
   ```

4. **Analyze missing data**:
   ```python
   # Generate missing data visualization
   await missing_data_analysis("iris")
   ```

5. **Create visualizations**:
   ```python
   # Histogram
   await create_visualization("iris", "histogram", x="sepal_length")
   
   # Scatter plot
   await create_visualization("iris", "scatter", x="sepal_length", y="petal_length")
   ```

6. **Detect outliers**:
   ```python
   # IQR method
   outliers_iqr = await detect_outliers("iris", method="iqr", factor=1.5)
   print(f"Outlier counts: {outliers_iqr.result.counts}")
   print(f"Image: {outliers_iqr.image_uri}")
   
   # Isolation Forest method
   outliers_if = await detect_outliers("iris", method="isolation_forest", contamination=0.05)
   print(f"Outlier counts: {outliers_if.result.counts}")
   
   # Local Outlier Factor method
   outliers_lof = await detect_outliers("iris", method="lof", contamination=0.05)
   print(f"Outlier counts: {outliers_lof.result.counts}")
   ```

### Schema Inference Example

The `infer_schema` tool provides comprehensive column analysis:

```python
schema = await infer_schema("my_dataset")

# Example output structure:
{
    "dataset_name": "my_dataset",
    "total_rows": 1000,
    "total_columns": 5,
    "columns": [
        {
            "name": "user_id",
            "type": "number",
            "nullable": False,
            "min_value": 1.0,
            "max_value": 1000.0,
            "unique_count": 1000,
            "sample_values": [1, 2, 3, 4, 5]
        },
        {
            "name": "email",
            "type": "string",
            "nullable": False,
            "pattern": "email",
            "max_length": 50,
            "unique_count": 1000,
            "sample_values": ["user1@example.com", "user2@example.com"]
        },
        {
            "name": "join_date",
            "type": "datetime",
            "nullable": False,
            "unique_count": 365,
            "sample_values": ["2023-01-01", "2023-01-02"]
        }
    ]
}
```

**Supported patterns**: email, url, phone, date
**Data types**: number, string, datetime
**Metadata**: nullability, ranges, unique counts, sample values

### Outlier Detection Example

The `detect_outliers` tool provides comprehensive outlier analysis:

```python
# Detect outliers using different methods
outliers = await detect_outliers("my_dataset", method="iqr", factor=1.5)

# Example output structure:
{
    "result": {
        "outliers": {
            "age": [15, 23, 45, 67],
            "salary": [120000, 150000],
            "height": [210, 140]
        },
        "counts": {
            "age": 4,
            "salary": 2,
            "height": 2
        },
        "total_rows": 1000
    },
    "image_uri": "file:///path/to/reports/outliers_my_dataset_iqr.png"
}
```

**Supported methods**:
- **IQR**: Interquartile range method (factor=1.5 is standard)
- **isolation_forest**: Model-based detection using Isolation Forest
- **lof**: Model-based detection using Local Outlier Factor

**Parameters**:
- `factor`: IQR multiplier (default: 1.5)
- `contamination`: Expected outlier fraction for model-based methods (default: 0.05)
- `sample_size`: Maximum rows for plotting (default: 10,000)

**Output**: Structured outlier indices, counts, and visualization image

## Testing

* **`test_evidently_tools.py`**: end-to-end test harness calling all tools programmatically.
* **`debug_test.py`**: prints actual summary keys for drift/regression debugging.
* **Run tests**:

  ```bash
  pytest           # runs unit tests if any
  python test_evidently_tools.py
  ```

## Roadmap & Next Steps

* **Phase 3 cont'd**: add `train_model` (sklearn/XGBoost → MLflow) & `predict` tool.
* **Phase 4**: HTTP/SSE exposure (`mcp run server.py --port 8000`), Docker + CI, ArgoCD deployment.
* **Phase 5**: implement `drift_watcher`, Slack/email alerts when thresholds breach.

Contributions and improvements are welcome—see [Contributing](#development--contributing) below.

## Development & Contributing

1. Fork & clone the repo
2. Create a feature branch
3. Follow the existing style and add tests
4. Submit a PR for review

All code is linted with `ruff` + `black`, type-checked with `mypy`.

## License

This project is released under the **MIT License**. See [LICENSE](LICENSE) for details.

---