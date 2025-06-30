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
7. [Configuration Management](#configuration-management)
8. [Interactive Checkpoints](#interactive-checkpoints)
9. [Running the Server](#running-the-server)
10. [Claude Desktop Integration](#claude-desktop-integration)
11. [Testing](#testing)
12. [Troubleshooting & Common Issues](#troubleshooting--common-issues)
13. [Sample Usage](#sample-usage)
14. [EDA Server Enhancement Report](#eda-server-enhancement-report)
15. [Roadmap & Next Steps](#roadmap--next-steps)
16. [Development & Contributing](#development--contributing)
17. [License](#license)

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
6. **Target Analysis** (automated, human-in-the-loop target column analysis)

## Prerequisites

* **Python ≥ 3.12** (pyenv recommended for version management)
  * **Tested with**: Python 3.12.3 (pyenv)
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
     "mcp[cli]==1.10.1" "pandas==2.3.0" "numpy==2.1.3" "pyarrow==20.0.0" \
     "matplotlib==3.10.0" "seaborn==0.13.2" "plotly==5.24.1" "scipy==1.15.3" "scikit-learn==1.7.0" \
     "ydata-profiling==4.16.1" "missingno==0.5.2" \
     "evidently==0.7.9" "python-dateutil==2.9.0.post0" "pyod==2.0.5" \
     "pydantic==2.11.7" "pyyaml==6.0.1" "pytest==7.4.3"
   ```

4. **Lock dependencies** for reproducibility:

   ```bash
   uv pip compile pyproject.toml \
     --output-file requirements.lock \
     --generate-hashes
   ```

   **Alternative**: Use exact versions for guaranteed reproducibility:
   ```bash
   pip install -r requirements-exact.txt
   ```

5. *(Optional)* **Install dev tools**:

   ```bash
   uv pip install .[dev]   # black, ruff, mypy, pytest
   ```

6. **Verify environment**:

   ```bash
   which python && python -V
   pip list | grep -E "(evidently|missingno|dateutil|pyod|pandas|numpy|pyyaml|pytest)"
   python -c "import missingno, evidently, dateutil, pyod, pandas, numpy, yaml, pytest; print('All core dependencies available')"
   ```

## Dependencies & Libraries

### Core Framework
| Library | Version | Purpose |
|---------|---------|---------|
| **mcp[cli]** | 1.10.1 | Model Context Protocol framework for tool registration and server management |
| **pydantic** | 2.11.7 | Data validation and settings management for structured outputs |
| **asyncio** | Built-in | Asynchronous programming for concurrent tool execution |

### Data Processing & Analysis
| Library | Version | Purpose |
|---------|---------|---------|
| **pandas** | 2.3.0 | Primary data manipulation and analysis library |
| **numpy** | 2.1.3 | Numerical computing and array operations |
| **pyarrow** | 20.0.0 | Columnar data format for efficient data storage |

### Data Science & Machine Learning
| Library | Version | Purpose |
|---------|---------|---------|
| **scikit-learn** | 1.7.0 | Machine learning algorithms for classification, regression, and clustering |
| **scipy** | 1.15.3 | Scientific computing and statistical functions |
| **pyod** | 2.0.5 | Outlier detection algorithms (Isolation Forest, Local Outlier Factor) |

### Data Quality & Monitoring
| Library | Version | Purpose |
|---------|---------|---------|
| **evidently** | 0.7.9 | Data quality, drift detection, and model performance monitoring |
| **ydata-profiling** | 4.16.1 | Automated data profiling and quality assessment |
| **missingno** | 0.5.2 | Missing data visualization and analysis |

### Visualization
| Library | Version | Purpose |
|---------|---------|---------|
| **matplotlib** | 3.10.0 | Core plotting library for static visualizations |
| **seaborn** | 0.13.2 | Statistical data visualization built on matplotlib |
| **plotly** | 5.24.1 | Interactive plotting for web-based visualizations |

### Configuration & Testing
| Library | Version | Purpose |
|---------|---------|---------|
| **pyyaml** | 6.0.1 | YAML configuration file parsing and validation |
| **pytest** | 7.4.3 | Testing framework for unit and integration tests |

### Development & Environment
| Library | Version | Purpose |
|---------|---------|---------|
| **uv** | Latest | Fast Python package installer and resolver |
| **python-dateutil** | 2.9.0.post0 | Date parsing and manipulation utilities |

### Optional Dependencies (Not Currently Installed)
| Library | Version | Purpose |
|---------|---------|---------|
| **polars** | Latest | Fast DataFrame library for large datasets |
| **xgboost** | Latest | Gradient boosting framework for ML models |
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
- **re**: Regular expressions for pattern matching (email, phone, URL, date, UUID)

#### Outlier Detection (`detect_outliers`)
- **pyod**: Isolation Forest and Local Outlier Factor algorithms
- **scikit-learn**: Statistical outlier detection methods
- **numpy**: Statistical calculations (IQR, quantiles, Mahalanobis distance)

#### Feature Transformation (`feature_transformation`)
- **scipy**: Box-Cox transformation and statistical functions
- **scikit-learn**: VIF calculation and supervised discretization
- **pandas**: Quantile binning and cardinality reduction

#### Data Quality & Drift (`data_quality_report`, `drift_analysis`)
- **evidently**: Data quality scoring and drift detection
- **pandas**: Data preprocessing and transformation

#### Configuration Management (`config.py`, `config.yaml`)
- **pyyaml**: YAML configuration file parsing
- **pydantic**: Configuration validation and type safety

#### Testing (`test_eda.py`)
- **pytest**: Test framework and fixtures
- **pandas**: Test data generation and manipulation
- **numpy**: Statistical test data creation

## Directory Structure

```text
mcp-server/
├── .venv/                  # Local virtual environment
├── pyproject.toml         # Project config & dependencies
├── requirements.lock      # Hash-pinned lockfile
├── requirements-exact.txt # Exact pinned versions for reproducibility
├── server.py              # FastMCP server implementation
├── config.py              # Configuration management and validation
├── config.yaml            # Centralized configuration file
├── utils.py               # Utility functions for common operations
├── checkpoints.py         # Interactive checkpoint system
├── test_eda.py            # Comprehensive unit test suite
├── launch_server.py       # Environment-aware launcher script
├── reports/               # Generated HTML dashboards and visualizations
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
* **Configuration management** via `config.yaml` and `config.py`
* **Utility functions** in `utils.py` for common operations
* **Interactive checkpoints** via `checkpoints.py` for human approval

## Available Tools

### Phase 1 – EDA Tools

| Tool                        | Description                                                      |
| --------------------------- | ---------------------------------------------------------------- |
| **`load_data`**             | Load CSV, Excel, or JSON into shared memory with memory reporting |
| **`basic_info`**            | Show shape, columns, dtypes, and first five rows                 |
| **`missing_data_analysis`** | Compute missing-value stats + render missingno matrix with clustering, thresholded dropping, and imputation strategies |
| **`create_visualization`**  | Render histogram, boxplot, scatter, correlation (with labels), or missing plot |
| **`statistical_summary`**   | Generate `describe()` and correlation matrix with large dataset sampling |
| **`list_datasets`**         | List dataset names with row/column counts                        |
| **`infer_schema`**          | Infer column types, nullability, ranges, patterns (email, phone, URL, date, UUID), and ID heuristics |
| **`detect_outliers`**       | Detect outliers using IQR, Isolation Forest, Local Outlier Factor, and Mahalanobis distance |
| **`feature_transformation`** | Apply Box-Cox, log, quantile binning, cardinality reduction, VIF analysis, and supervised discretization |
| **`target_analysis`**       | Automated, human-in-the-loop target column analysis (classification/regression, priors, skew, top features, and visualizations) |

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

## Configuration Management

The EDA server uses a centralized configuration system for easy tuning and maintenance:

### Configuration File (`config.yaml`)

```yaml
# Missing Data Analysis
missing_data:
  column_drop_threshold: 0.50  # 50% - drop columns with >50% missing data
  row_drop_threshold: 0.50     # 50% - drop rows with >50% missing values
  systematic_correlation_threshold: 0.70  # High correlation for systematic missingness

# Outlier Detection
outlier_detection:
  iqr_factor: 1.5              # Standard IQR multiplier
  contamination_default: 0.05  # 5% expected outliers for model-based methods
  mahalanobis_confidence: 0.975  # Chi-square confidence level

# Feature Transformation
feature_transformation:
  rare_category_threshold: 0.005  # 0.5% threshold for rare categories
  vif_severe_threshold: 10.0      # VIF > 10 indicates severe multicollinearity
  boxcox_epsilon: 1e-6            # Small epsilon for Box-Cox shifting

# Performance
performance:
  memory_warning_threshold: 1000  # MB - warn if dataset > 1GB
  correlation_sample_size: 10000  # Sample size for correlation matrices
```

### Configuration Usage

```python
from config import config

# Access configuration values
drop_threshold = config.missing_data.column_drop_threshold
iqr_factor = config.outlier_detection.iqr_factor
```

### Configuration Validation

The configuration system uses Pydantic for type safety and validation:

```python
# Invalid configuration will raise validation errors
config.MissingDataConfig(column_drop_threshold=1.5)  # Raises error: > 1.0
```

## Interactive Checkpoints

The EDA server implements a human-in-the-loop approval system for critical operations:

### Checkpoint Types

1. **Missing Data Analysis**: Approve missing data patterns and recommended actions
2. **Outlier Detection**: Review outlier patterns and detection methods
3. **Feature Transformation**: Approve transformations before modeling
4. **Schema Inference**: Review inferred schema and data quality

### Checkpoint Configuration

```yaml
checkpoints:
  require_approval: true          # Whether to require human approval
  approval_timeout: 300           # Seconds to wait for approval
  auto_approve_small_changes: true  # Auto-approve minor transformations
```

### Checkpoint Usage

```python
from checkpoints import request_transformation_approval

# Request approval for feature transformations
approved = await request_transformation_approval(
    dataset_name="my_data",
    original_shape=(1000, 5),
    transformed_shape=(1000, 8),
    new_columns=["feature_1_log", "feature_2_boxcox"],
    transformations_applied=["log", "boxcox"]
)

if not approved:
    # Handle rejection
    return "Transformation rejected by user"
```

### Disabling Checkpoints

For automated processing, checkpoints can be disabled:

```python
from checkpoints import disable_checkpoints, enable_checkpoints

disable_checkpoints()  # For automated workflows
enable_checkpoints()   # For interactive workflows
```

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

## Testing

The EDA server includes comprehensive unit tests covering edge cases and core functionality:

### Running Tests

```bash
# Run all tests
pytest test_eda.py -v

# Run specific test categories
pytest test_eda.py::TestEDAServer -v
pytest test_eda.py::TestConfiguration -v
pytest test_eda.py::TestUtils -v

# Run tests with coverage
pytest test_eda.py --cov=server --cov=utils --cov=config -v
```

### Test Coverage

The test suite covers:

- **Edge Cases**: All-missing data, constant columns, single values, large datasets
- **Core Functionality**: Data loading, missing analysis, outlier detection, schema inference
- **Configuration**: Validation, defaults, error handling
- **Utilities**: Quantile binning, VIF calculation, memory formatting
- **Concurrent Access**: Multiple simultaneous operations
- **Pattern Detection**: Email, phone, date, UUID patterns

### Test Data

Tests use synthetic datasets including:
- Normal datasets with missing values
- All-missing datasets
- Constant value datasets
- Large datasets for performance testing
- Pattern-rich datasets for schema inference

## Troubleshooting & Common Issues

### Issue 1: "No module named 'missingno'" or "No module named 'dateutil'"

**Symptoms**: Server fails to start with import errors

**Solution**:
```bash
# Install missing dependencies
pip install python-dateutil missingno ydata-profiling evidently pyyaml pytest

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

### Issue 5: Configuration Errors

**Symptoms**: Server fails to start with configuration validation errors

**Solutions**:

1. **Check configuration file**:
   ```bash
   python -c "import config; print('Configuration loaded successfully')"
   ```

2. **Validate configuration manually**:
   ```bash
   python -c "from config import load_config; cfg = load_config(); print('Config valid')"
   ```

3. **Use default configuration**:
   ```bash
   # Remove config.yaml to use defaults
   mv config.yaml config.yaml.backup
   python server.py
   ```

### Issue 6: Test Failures

**Symptoms**: Unit tests fail with import or assertion errors

**Solutions**:

1. **Install test dependencies**:
   ```bash
   pip install pytest pytest-asyncio
   ```

2. **Run tests in isolation**:
   ```bash
   pytest test_eda.py::TestEDAServer::test_load_data -v
   ```

3. **Check test environment**:
   ```bash
   python -c "import server, config, utils, checkpoints; print('All modules available')"
   ```

### Debug Commands

**Check Python environment**:
```bash
which python
python --version
pip list | grep -E "(evidently|missingno|dateutil|pyod|pandas|numpy|pyyaml|pytest)"
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

**Test configuration**:
```bash
python -c "from config import get_config; cfg = get_config(); print('Config loaded:', cfg.missing_data.column_drop_threshold)"
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

7. **Apply feature transformations**:
   ```python
   # Apply multiple transformations
   transformations = await feature_transformation(
       "iris", 
       transformations=["boxcox", "log", "binning", "cardinality"],
       target_col="target"
   )
   print(f"Original shape: {transformations.original_shape}")
   print(f"Transformed shape: {transformations.transformed_shape}")
   print(f"New columns: {transformations.steps[-1].statistics['new_columns']}")
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

**Supported patterns**: email, url, phone, date, uuid, credit_card, postal_code, ip_address
**Data types**: number, string, datetime
**Metadata**: nullability, ranges, unique counts, sample values, precision/scale

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
        "total_rows": 1000,
        "steps": [
            {
                "step_name": "Data Distribution Analysis",
                "statistics": {
                    "age": {"mean": 35.2, "std": 12.1, "skewness": 0.8}
                }
            },
            {
                "step_name": "IQR Outlier Detection",
                "statistics": {
                    "age": {"q1": 26, "q3": 44, "iqr": 18, "outlier_count": 4}
                }
            }
        ],
        "method_used": "iqr"
    },
    "image_uri": "file:///path/to/reports/outliers_my_dataset_iqr.png",
    "human_checkpoint": "=== OUTLIER DETECTION CHECKPOINT ===\n..."
}
```

**Supported methods**:
- **IQR**: Interquartile range method (factor=1.5 is standard)
- **isolation_forest**: Model-based anomaly detection
- **lof**: Model-based detection using Local Outlier Factor

**Parameters**:
- `factor`: IQR multiplier (default: 1.5)
- `contamination`: Expected outlier fraction for model-based methods (default: 0.05)
- `sample_size`: Maximum rows for plotting (default: 10,000)

**Output**: Structured outlier indices, counts, step-by-step analysis, and visualization image

### Feature Transformation Example

The `feature_transformation` tool provides comprehensive feature engineering:

```python
# Apply multiple transformations
result = await feature_transformation(
    "my_dataset",
    transformations=["boxcox", "log", "binning", "cardinality"],
    target_col="target"
)

# Example output structure:
{
    "dataset_name": "my_dataset",
    "original_shape": (1000, 5),
    "transformed_shape": (1000, 9),
    "steps": [
        {
            "step_name": "Box-Cox Transformation",
            "statistics": {
                "age": {
                    "original_skewness": 1.2,
                    "transformed_skewness": 0.1,
                    "lambda_parameter": 0.3,
                    "shift_applied": 0.0
                }
            }
        },
        {
            "step_name": "Variance Inflation Factor (VIF) Analysis",
            "statistics": {
                "vif_scores": {"age": 1.2, "income": 8.5},
                "high_vif_columns": ["income"]
            }
        }
    ],
    "image_uri": "file:///path/to/reports/feature_transformation_my_dataset.png",
    "human_checkpoint": "=== FEATURE TRANSFORMATION CHECKPOINT ===\n..."
}
```

**Supported transformations**:
- **boxcox**: Box-Cox transformation for skewed data
- **log**: Logarithmic transformation with automatic shifting
- **quantile binning**: Discretize numeric variables
- **cardinality reduction**: Group rare categories (0.5% threshold)
- **VIF analysis**: Detect multicollinearity
- **supervised discretization**: Target-guided binning for classification

## EDA Server Enhancement Report

This document summarizes each module and logic step implemented in the EDA server, with concise reasoning and the key mathematical formulas used.

---

### 1. Configuration Centralization

**Reasoning:** Centralize all tunable thresholds and parameters for easy maintenance and validation.
**Math/Config:** e.g. `column_drop_threshold = 0.50` → drop if `missing_pct > 0.50`.

**Implementation:**
- **`config.yaml`**: Centralized configuration file with all thresholds
- **`config.py`**: Type-safe configuration loader with Pydantic validation
- **Benefits**: Easy tuning, validation, and maintenance of all parameters

### 2. `load_data`

**Reasoning:** Report dataset dimensions and memory footprint to anticipate performance issues.
**Math:** `memory_mb = df.memory_usage(deep=True).sum() / 1024**2`.

**Features:**
- Memory usage reporting with human-readable formatting
- Support for CSV, Excel, and JSON formats
- Error handling for missing files and unsupported formats

### 3. `basic_info`

**Reasoning:** Provide immediate context on shape, dtypes, and sample rows for rapid dataset familiarization.

**Output:**
- Dataset shape and dimensions
- Column data types
- First five rows for quick inspection

### 4. `missing_data_analysis`

**Reasoning:** Quantify, visualize, and diagnose missingness patterns before imputation.
**Math:**

* Missing count per column: `m_c = Σ 1_{isnull}`
* Missing percentage: `p_c = (m_c / N) × 100`
* Drop threshold: `p_c > 0.50`
* Little's test χ²: `χ² = Σ (O−E)²/E` for MCAR detection.

**Steps:**
1. **Missing Data Clustering**: Analyze patterns across columns
2. **Thresholded Column Dropping**: Identify columns with >50% missing data
3. **Missingness Mechanism Analysis**: MCAR/MAR/MNAR detection via Little's test
4. **Row-wise Missing Analysis**: Identify problematic records
5. **Imputation Strategy Analysis**: Median vs mean based on skewness, mode for categoricals
6. **Data Quality Impact Assessment**: Overall quality scoring

### 5. `create_visualization`

**Reasoning:** Guard plot types by dtype and annotate with correlation values for interpretability.
**Math:** Heatmap label: `corr_{ij} = Cov(X_i,X_j)/(σ_i σ_j)`.

**Supported plots:**
- **Histogram**: For numeric columns with density estimation
- **Boxplot**: With outlier highlighting
- **Scatter**: For numeric pairs with correlation analysis
- **Correlation**: Heatmap with coefficient labels
- **Missing**: Missing data pattern matrix

### 6. `statistical_summary`

**Reasoning:** Summarize descriptive stats and sample large datasets to control O(N²) correlation cost.

**Features:**
- Automatic sampling for datasets >10,000 rows
- Comprehensive descriptive statistics
- Correlation matrix with performance optimization

### 7. `list_datasets`

**Reasoning:** Enumerate loaded datasets and shapes to manage workspace state.

**Output:**
- Dataset names and shapes
- Memory usage information
- Quick workspace overview

### 8. `infer_schema`

**Reasoning:** Auto-detect types, ranges, patterns, and IDs to generate a formal schema contract.
**Math/Heuristics:**

* Uniqueness ratio: `u = unique_count / N`
* ID if `u ≥ 0.90` or matches `UUID` regex.
* Datetime parse success: `success_rate ≥ 0.80`.

**Detection Features:**
- **Type Inference**: number, string, datetime with precision/scale
- **Pattern Detection**: email, url, phone, date, uuid, credit_card, postal_code, ip_address
- **ID Detection**: Uniqueness ratio + naming patterns
- **Datetime Inference**: Automatic parsing with success rate validation
- **YAML Contract**: Formal schema specification

### 9. `detect_outliers`

**Reasoning:** Combine univariate (IQR), model-based (IF/LOF), and multivariate (Mahalanobis) detection for robust anomaly identification.
**Math:**

* IQR bounds: `LB = Q1 − 1.5·IQR`, `UB = Q3 + 1.5·IQR`
* Mahalanobis: `d² = (x−μ)^T Σ^{−1}(x−μ)`, outlier if `d² > χ²_{p,0.975}`.

**Methods:**
1. **IQR Method**: Standard statistical outlier detection
2. **Isolation Forest**: Model-based anomaly detection
3. **Local Outlier Factor**: Density-based outlier detection
4. **Mahalanobis Distance**: Multivariate outlier detection

**Visualizations:**
- Histogram + KDE with outlier highlighting
- Boxplot with outlier points
- Distribution analysis with skewness/kurtosis

### 10. `feature_transformation`

**Reasoning:** Normalize skew, discretize, and reduce cardinality to improve model readiness.
**Math:**

* Box–Cox: `x' = (x^λ−1)/λ` or `ln(x)` if `λ=0` (shift x>0)
* Log: `x' = ln(x + shift)` where `shift = max(0, 1−min(x))`
* Quantile binning: `bins = qcut(x, q=n_bins)`
* Rare grouping: category if `freq < 0.005`
* VIF: `VIF_i = 1/(1−R_i²)`.

**Transformations:**
1. **Box-Cox Transformation**: Normalize skewed data with automatic shifting
2. **Log Transformation**: Handle right-skewed data with shift calculation
3. **Quantile Binning**: Discretize numeric variables
4. **Cardinality Reduction**: Group rare categories (0.5% threshold)
5. **VIF Analysis**: Detect multicollinearity
6. **Supervised Discretization**: Target-guided binning for classification

### 11. Configuration Loader & `utils.py`

**Reasoning:** Abstract repeated logic (e.g. skew, VIF, sampling) into reusable functions for DRY code and testability.

**Key Utilities:**
- `calculate_skewness()` / `calculate_kurtosis()` with edge case handling
- `get_numeric_columns()` / `get_categorical_columns()` with exclusions
- `calculate_missing_stats()` for comprehensive missing data analysis
- `detect_id_columns()` / `infer_datetime_columns()` for schema inference
- `create_quantile_bins()` with fallback strategies
- `calculate_vif_scores()` for multicollinearity detection
- `sample_dataframe()` for large dataset handling
- `format_memory_usage()` for human-readable memory reporting

### 12. Interactive Checkpoints

**Reasoning:** Require explicit human approval for destructive or model-sensitive steps, with configurable timeouts.

**Features:**
- **CheckpointManager**: Centralized approval management
- **Specialized Functions**: Missing data, outlier detection, transformation, schema approval
- **Auto-approval**: Small changes automatically approved
- **Timeout Handling**: Configurable approval timeouts
- **Message Formatting**: Standardized checkpoint messages
- **Enable/Disable**: Toggle checkpoints for automated vs interactive processing

### 13. Unit Testing

**Reasoning:** Ensure correctness across edge cases (empty, constant, large datasets) and prevent regressions via automated tests.

**Test Coverage:**
- **Edge Cases**: All-missing data, constant columns, single values, large datasets
- **Core Functionality**: Data loading, missing analysis, outlier detection, schema inference
- **Configuration**: Validation, defaults, error handling
- **Utilities**: Quantile binning, VIF calculation, memory formatting
- **Concurrent Access**: Multiple simultaneous operations
- **Pattern Detection**: Email, phone, date, UUID patterns

---

*All formulas and thresholds align with **Introduction to Data Mining** best practices, ensuring transparent, auditable, and mathematically precise EDA workflows.*

## Roadmap & Next Steps

* **Phase 3 cont'd**: add `train_model` (sklearn/XGBoost → MLflow) & `predict` tool.
* **Phase 4**: HTTP/SSE exposure (`mcp run server.py --port 8000`), Docker + CI, ArgoCD deployment.
* **Phase 5**: implement `drift_watcher`, Slack/email alerts when thresholds breach.

**Immediate Enhancements:**
- **Performance Profiling**: Add profiling for very wide/tall datasets
- **Asynchronous Processing**: Implement chunked processing for large datasets
- **Advanced Visualizations**: Add more sophisticated plotting options
- **Model Integration**: Add model training and prediction capabilities
- **Alerting System**: Implement drift and quality alerts

Contributions and improvements are welcome—see [Contributing](#development--contributing) below.

## Development & Contributing

1. Fork & clone the repo
2. Create a feature branch
3. Follow the existing style and add tests
4. Submit a PR for review

All code is linted with `ruff` + `black`, type-checked with `mypy`.

### Development Setup

```bash
# Install development dependencies
uv pip install .[dev]

# Run linting
ruff check .
black .

# Run type checking
mypy server.py config.py utils.py checkpoints.py

# Run tests
pytest test_eda.py -v

# Run tests with coverage
pytest test_eda.py --cov=server --cov=utils --cov=config --cov-report=html
```

### Code Style

- **Type Hints**: Use type hints for all function parameters and return values
- **Docstrings**: Include detailed docstrings with examples
- **Error Handling**: Use proper exception handling with informative messages
- **Testing**: Write tests for all new functionality and edge cases
- **Configuration**: Add new parameters to `config.yaml` with validation

## License

This project is released under the **MIT License**. See [LICENSE](LICENSE) for details.

---