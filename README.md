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
8. [Sample Usage](#sample-usage)
9. [Testing](#testing)
10. [Roadmap & Next Steps](#roadmap--next-steps)
11. [Development & Contributing](#development--contributing)
12. [License](#license)

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

* **Python ≥ 3.12**
* **uv** for environment & dependency management
* **MCP CLI** (provided by `mcp[cli]`)
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
     evidently
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
   ```

## Directory Structure

```text
mcp-server/
├── .venv/                  # Local virtual environment
├── pyproject.toml         # Project config & dependencies
├── requirements.lock      # Hash-pinned lockfile
├── server.py              # FastMCP server implementation
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

```bash
cd mcp-server
source .venv/bin/activate
mcp dev server.py
# ➜ 🟢 EDA Server | ready – listening on stdio
```

## Sample Usage

**Helper** (in a second terminal):

```bash
call() { printf '%s\n' "$1" | mcp call stdio | jq .; }
```

1. **Load a dataset**:

   ```bash
   call '{"jsonrpc":"2.0","id":1,"method":"callTool","params":{  
     "name":"load_data","arguments":{"file_path":"iris.csv","name":"iris"}
   }}'
   ```

2. **Inspect basic info**:

   ```bash
   call '{"jsonrpc":"2.0","id":2,"method":"callTool","params":{  
     "name":"basic_info","arguments":{"name":"iris"}
   }}'
   ```

3. **Generate data-quality report**:

   ```bash
   call '{"jsonrpc":"2.0","id":3,"method":"callTool","params":{  
     "name":"data_quality_report","arguments":{"name":"iris"}
   }}'
   ```

4. **Detect drift** (baseline vs current):

   ```bash
   # assume iris_a & iris_b loaded
   call '{"jsonrpc":"2.0","id":4,"method":"callTool","params":{  
     "name":"drift_analysis","arguments":{"baseline":"iris_a","current":"iris_b"}
   }}'
   ```

5. **Regression performance**:

   ```bash
   python - <<'PY'
   import json, random, subprocess
   y_true=[random.uniform(0,10) for _ in range(100)]
   y_pred=[v+random.gauss(0,1) for v in y_true]
   payload={"jsonrpc":"2.0","id":5,"method":"callTool",
     "params":{"name":"model_performance_report","arguments":{  
       "y_true":y_true,"y_pred":y_pred,"model_type":"regression"}}}
   subprocess.run(["mcp","call","stdio"], input=(json.dumps(payload)+"\n").encode())
   PY
   ```

6. **Classification performance** (sklearn fallback):

   ```bash
   # similar to above but with random ints and model_type=classification
   ```

All HTML reports land under `reports/` (e.g. `reports/dq_iris.html`, `reports/drift_iris_a_vs_iris_b.html`).

## Testing

* **`test_evidently_tools.py`**: end-to-end test harness calling all tools programmatically.
* **`debug_test.py`**: prints actual summary keys for drift/regression debugging.
* **Run tests**:

  ```bash
  pytest           # runs unit tests if any
  python test_evidently_tools.py
  ```

## Roadmap & Next Steps

* **Phase 3 cont’d**: add `train_model` (sklearn/XGBoost → MLflow) & `predict` tool.
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

*Generated using the detailed change logs from the Liquid Intelligence project.*
