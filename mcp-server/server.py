#!/usr/bin/env python3
"""
FastMCP server for Data-Science / EDA.

Tools:
• load_data
• basic_info
• missing_data_analysis
• create_visualization
• statistical_summary
• list_datasets
"""

from typing import Dict
from pathlib import Path
import io, base64, tempfile, asyncio, json

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import missingno as msno

from mcp.server import FastMCP
from mcp.types import TextContent, ImageContent

from evidently import (
    DataDefinition, Dataset, Report,
    BinaryClassification, MulticlassClassification, Regression
)
from evidently.presets import (
    DataSummaryPreset, DataDriftPreset, RegressionPreset, ClassificationPreset,
)
from pydantic import BaseModel

from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    mean_squared_error,
    mean_absolute_error,
    r2_score,
)

# Initialize FastMCP server
mcp = FastMCP("data-science-eda")

# Data store and lock for thread safety
data_store: Dict[str, pd.DataFrame] = {}
store_lock = asyncio.Lock()

# --- Evidently helpers ------------------------------------------------------
REPORT_DIR = Path("reports")
REPORT_DIR.mkdir(exist_ok=True)

def _ds(df: pd.DataFrame) -> Dataset:
    return Dataset.from_pandas(df, DataDefinition())   # auto-mapping works

def _ds_classification(df: pd.DataFrame) -> Dataset:
    # For classification, explicitly define the task and columns
    return Dataset.from_pandas(
        df,
        DataDefinition(
            classification=[MulticlassClassification(target="target", prediction="prediction")]
        ),
    )

def _ds_regression(df: pd.DataFrame) -> Dataset:
    # For regression, explicitly define the task and columns
    return Dataset.from_pandas(
        df,
        DataDefinition(
            regression=[Regression(target="target", prediction="prediction")]
        ),
    )

# ─────────────────────────── Resources / Helpers ──────────────────────────
def _fig_to_base64_png() -> str:
    buf = io.BytesIO()
    plt.savefig(buf, format="png", bbox_inches="tight")
    buf.seek(0)
    return base64.b64encode(buf.read()).decode()


# ────────────────────────────────── Tools ──────────────────────────────────
@mcp.tool()
async def load_data(file_path: str, name: str) -> str:
    """
    Load CSV/Excel/JSON into the shared store.
    Returns a confirmation string.
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(path)

    if path.suffix == ".csv":
        df = pd.read_csv(path)
    elif path.suffix in {".xlsx", ".xls"}:
        df = pd.read_excel(path)
    elif path.suffix == ".json":
        df = pd.read_json(path)
    else:
        raise ValueError(f"Unsupported format: {path.suffix}")

    async with store_lock:
        data_store[name] = df

    return f"Loaded '{name}' with {len(df)} rows × {len(df.columns)} columns."


@mcp.tool()
async def basic_info(name: str) -> str:
    """
    Return shape, columns, dtypes and head() for a dataset.
    """
    async with store_lock:
        df = data_store.get(name)
    if df is None:
        raise KeyError(f"Dataset '{name}' not found.")

    info = (
        f"Dataset: {name}\n"
        f"Shape: {df.shape}\n\n"
        f"Dtypes:\n{df.dtypes.to_string()}\n\n"
        f"Head:\n{df.head().to_string(index=False)}"
    )
    return info


@mcp.tool()
async def missing_data_analysis(name: str) -> list:
    """
    Show missing-value stats and a matrix plot.
    Returns [TextContent, ImageContent].
    """
    async with store_lock:
        df = data_store.get(name)
    if df is None:
        raise KeyError(f"Dataset '{name}' not found.")

    missing_stats = df.isnull().sum()
    missing_pct = (missing_stats / len(df) * 100).round(2)

    plt.figure(figsize=(12, 8))
    msno.matrix(df)
    plt.title(f"Missing Data – {name}")
    img_b64 = _fig_to_base64_png()
    plt.close()

    analysis = (
        f"Missing values by column:\n{missing_stats.to_string()}\n\n"
        f"Percent missing:\n{missing_pct.to_string()}\n"
        f"Total missing: {missing_stats.sum()}"
    )
    return [
        TextContent(type="text", text=analysis),
        ImageContent(type="image", data=img_b64, mimeType="image/png"),
    ]


@mcp.tool()
async def create_visualization(
    name: str,
    kind: str,
    x: str | None = None,
    y: str | None = None,
) -> list:
    """
    kind = histogram | boxplot | scatter | correlation | missing
    """
    async with store_lock:
        df = data_store.get(name)
    if df is None:
        raise KeyError(f"Dataset '{name}' not found.")

    plt.figure(figsize=(10, 6))
    if kind == "histogram" and x:
        plt.hist(df[x].dropna(), bins=30, alpha=0.7)
        plt.title(f"Histogram – {x}")
    elif kind == "boxplot" and x:
        plt.boxplot(df[x].dropna())
        plt.title(f"Boxplot – {x}")
    elif kind == "scatter" and x and y:
        plt.scatter(df[x], df[y], alpha=0.6)
        plt.title(f"Scatter – {x} vs {y}")
    elif kind == "correlation":
        corr = df.select_dtypes(np.number).corr()
        plt.imshow(corr, cmap="coolwarm", aspect="auto")
        plt.colorbar()
        plt.title("Correlation matrix")
    elif kind == "missing":
        msno.matrix(df)
        plt.title(f"Missing pattern – {name}")
    else:
        raise ValueError("Invalid parameters for visualization.")

    img_b64 = _fig_to_base64_png()
    plt.close()
    return [
        TextContent(type="text", text=f"{kind.capitalize()} plot created."),
        ImageContent(type="image", data=img_b64, mimeType="image/png"),
    ]


@mcp.tool()
async def statistical_summary(name: str) -> str:
    """
    Return describe() and numeric correlation for a dataset.
    """
    async with store_lock:
        df = data_store.get(name)
    if df is None:
        raise KeyError(f"Dataset '{name}' not found.")

    num = df.select_dtypes(np.number)
    if num.empty:
        return "No numeric columns."

    return (
        f"Descriptive statistics:\n{num.describe().to_string()}\n\n"
        f"Correlation matrix:\n{num.corr().to_string()}"
    )


@mcp.tool()
async def list_datasets() -> str:
    """List names and shapes of all datasets in memory."""
    async with store_lock:
        if not data_store:
            return "No datasets loaded."
        lines = [
            f"- {name}: {df.shape[0]} rows × {df.shape[1]} columns"
            for name, df in data_store.items()
        ]
    return "Datasets:\n" + "\n".join(lines)


class DataQualityReportResult(BaseModel):
    html_uri: str

@mcp.tool(structured_output=True)
async def data_quality_report(
    name: str,
    title: str | None = None,
) -> DataQualityReportResult:
    """
    Run Evidently DataSummaryPreset on a single dataset.
    Returns {"html_uri": "file://..."}.
    """
    async with store_lock:
        df = data_store.get(name)
    if df is None:
        raise KeyError(f"Dataset '{name}' not found.")

    metadata = {"title": title} if title else None
    rpt = Report([DataSummaryPreset()], metadata=metadata)
    snapshot = await asyncio.to_thread(rpt.run, _ds(df))

    html = REPORT_DIR / f"dq_{name}.html"
    snapshot.save_html(html)

    # DataSummaryPreset may not have 'quality_score', so just return the HTML URI
    return DataQualityReportResult(html_uri=f"file://{html.resolve()}")


class DriftAnalysisResult(BaseModel):
    drift_count: float
    drift_share: float
    html_uri: str

@mcp.tool(structured_output=True)
async def drift_analysis(
    baseline: str,
    current: str,
) -> DriftAnalysisResult:
    async with store_lock:
        ref = data_store.get(baseline)
        cur = data_store.get(current)
    if ref is None or cur is None:
        raise KeyError("Datasets not found")

    rpt = Report([DataDriftPreset()])
    snap = await asyncio.to_thread(rpt.run, _ds(cur), _ds(ref))

    html = REPORT_DIR / f"drift_{baseline}_vs_{current}.html"
    snap.save_html(html)

    summary = json.loads(snap.json)
    # Find the DriftedColumnsCount metric
    drift_metric = next(
        (m["value"] for m in summary["metrics"]
         if m["metric_id"].startswith("DriftedColumnsCount")),
        {"count": 0.0, "share": 0.0},
    )

    return DriftAnalysisResult(
        drift_count=drift_metric["count"],
        drift_share=drift_metric["share"],
        html_uri=f"file://{html.resolve()}",
    )


class ModelPerformanceReportResult(BaseModel):
    metrics: dict
    html_uri: str

@mcp.tool(structured_output=True)
async def model_performance_report(
    y_true: list[float | int],
    y_pred: list[float | int],
    model_type: str = "classification",
) -> ModelPerformanceReportResult:

    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred length mismatch.")

    # First try Evidently for regression only
    if model_type.startswith("reg"):
        df = pd.DataFrame({"target": y_true, "prediction": y_pred})
        rpt = Report([RegressionPreset()])
        snap = await asyncio.to_thread(rpt.run, _ds(df))
        html = REPORT_DIR / f"perf_regression_{len(df)}.html"
        snap.save_html(html)

        summary = json.loads(snap.json)
        # Extract common regression metrics: RMSE, MAE, R2
        metrics = {}
        for m in summary["metrics"]:
            mid = m["metric_id"]
            val = m["value"]
            if mid.startswith("RMSE"):
                metrics["rmse"] = val
            elif mid.startswith("MeanError"):
                metrics["mae"] = val["mean"]
            elif mid.startswith("R2"):
                metrics["r2"] = val

    else:
        # Classification → fallback to sklearn metrics
        acc = accuracy_score(y_true, y_pred)
        prec, rec, f1, _ = precision_recall_fscore_support(
            y_true, y_pred, average="weighted", zero_division=0
        )
        metrics = {
            "accuracy": acc,
            "precision": prec,
            "recall": rec,
            "f1_weighted": f1,
        }
        # No Evidently HTML for classification fallback
        html = REPORT_DIR / f"perf_classification_{len(y_true)}.html"
        with open(html, "w") as f:
            f.write(
                "<html><body>"
                f"<h1>Classification Metrics</h1>"
                f"<ul>"
                f"<li>Accuracy: {acc:.3f}</li>"
                f"<li>Precision: {prec:.3f}</li>"
                f"<li>Recall: {rec:.3f}</li>"
                f"<li>F1 (weighted): {f1:.3f}</li>"
                "</ul>"
                "</body></html>"
            )

    return ModelPerformanceReportResult(
        metrics=metrics,
        html_uri=f"file://{html.resolve()}",
    )


# ─── DEBUG DUMP FOR SUMMARY KEYS ─────────────────────────────────────────

class DebugSummaryResult(BaseModel):
    html_uri: str
    keys: list[str]
    summary: dict

@mcp.tool(structured_output=True)
async def debug_drift_summary(
    baseline: str,
    current: str,
) -> DebugSummaryResult:
    """
    TEMPORARY: Run DataDrift and return summary keys.
    """
    async with store_lock:
        ref = data_store.get(baseline)
        cur = data_store.get(current)
    if ref is None or cur is None:
        raise KeyError("datasets not found")

    rpt = Report([DataDriftPreset()])
    snap = await asyncio.to_thread(rpt.run, _ds(cur), _ds(ref))

    html = REPORT_DIR / "drift_debug.html"
    snap.save_html(html)

    summary = json.loads(snap.json())
    return DebugSummaryResult(
        html_uri=f"file://{html.resolve()}",
        keys=list(summary.keys()),
        summary=summary,
    )

@mcp.tool(structured_output=True)
async def debug_perf_summary(
    y_true: list[int | float],
    y_pred: list[int | float],
    model_type: str = "classification",
) -> DebugSummaryResult:
    """
    TEMPORARY: Run PerformancePreset and return summary keys.
    """
    if len(y_true) != len(y_pred):
        raise ValueError("length mismatch")

    df = pd.DataFrame({"target": y_true, "prediction": y_pred})
    preset = ClassificationPreset() if model_type.startswith("class") else RegressionPreset()
    rpt = Report([preset])
    snap = await asyncio.to_thread(rpt.run, _ds(df))

    html = REPORT_DIR / "perf_debug.html"
    snap.save_html(html)

    summary = json.loads(snap.json())
    return DebugSummaryResult(
        html_uri=f"file://{html.resolve()}",
        keys=list(summary.keys()),
        summary=summary,
    )


# ──────────────────────────────── Entrypoint ───────────────────────────────
if __name__ == "__main__":
    # stdio is default; FastMCP also supports SSE ("http") if you add --port
    mcp.run() 