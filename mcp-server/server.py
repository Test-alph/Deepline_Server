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
• infer_schema
• detect_outliers
"""

from typing import Dict, Literal
from pathlib import Path
import io, base64, tempfile, asyncio, json, re

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import missingno as msno
import seaborn as sns

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

# Try to import pyod for outlier detection
try:
    from pyod.models.iforest import IsolationForest
    from pyod.models.lof import LOF
    PYOD_AVAILABLE = True
except ImportError:
    PYOD_AVAILABLE = False
    print("Warning: pyod not available. Model-based outlier detection will not work.")

# Initialize FastMCP server
mcp = FastMCP("data-science-eda")

# Data store and lock for thread safety
data_store: Dict[str, pd.DataFrame] = {}
store_lock = asyncio.Lock()

# --- Schema inference models -------------------------------------------------
class SchemaColumn(BaseModel):
    name: str
    type: str  # "number", "datetime", "string"
    nullable: bool
    min_value: float | None = None
    max_value: float | None = None
    pattern: str | None = None
    max_length: int | None = None
    unique_count: int | None = None
    sample_values: list | None = None

class SchemaResult(BaseModel):
    dataset_name: str
    columns: list[SchemaColumn]
    total_rows: int
    total_columns: int

# --- Outlier detection models -------------------------------------------------
class OutlierPayload(BaseModel):
    outliers: dict[str, list[int]]
    counts: dict[str, int]
    total_rows: int

class OutlierResult(BaseModel):
    result: OutlierPayload
    image_uri: str

# --- Target analysis models -------------------------------------------------
class TargetAnalysisResult(BaseModel):
    result: dict
    plots: dict[str, str]

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
        if x not in df.columns:
            raise KeyError(f"Column '{x}' not found in dataset.")
        if not pd.api.types.is_numeric_dtype(df[x]):
            raise ValueError(f"Column '{x}' must be numeric for histogram.")
        plt.hist(df[x].dropna(), bins=30, alpha=0.7)
        plt.title(f"Histogram – {x}")
        plt.xlabel(x)
        plt.ylabel("Frequency")
        
    elif kind == "boxplot":
        if not x:
            raise ValueError("Boxplot requires 'x' parameter.")
        if x not in df.columns:
            raise KeyError(f"Column '{x}' not found in dataset.")
        
        if y and y in df.columns:
            # Boxplot with categorical x and numeric y
            if not pd.api.types.is_numeric_dtype(df[y]):
                raise ValueError(f"Column '{y}' must be numeric for boxplot.")
            sns.boxplot(data=df, x=x, y=y)
            plt.title(f"Boxplot – {y} by {x}")
        else:
            # Simple boxplot of single column
            if not pd.api.types.is_numeric_dtype(df[x]):
                raise ValueError(f"Column '{x}' must be numeric for single-column boxplot.")
            plt.boxplot(df[x].dropna())
            plt.title(f"Boxplot – {x}")
            plt.ylabel(x)
            
    elif kind == "scatter" and x and y:
        if x not in df.columns or y not in df.columns:
            raise KeyError(f"Columns '{x}' and/or '{y}' not found in dataset.")
        if not pd.api.types.is_numeric_dtype(df[x]) or not pd.api.types.is_numeric_dtype(df[y]):
            raise ValueError(f"Both columns '{x}' and '{y}' must be numeric for scatter plot.")
        plt.scatter(df[x], df[y], alpha=0.6)
        plt.title(f"Scatter – {x} vs {y}")
        plt.xlabel(x)
        plt.ylabel(y)
        
    elif kind == "correlation":
        numeric_df = df.select_dtypes(include=[np.number])
        if numeric_df.empty:
            raise ValueError("No numeric columns found for correlation matrix.")
        corr = numeric_df.corr()
        plt.imshow(corr, cmap="coolwarm", aspect="auto")
        plt.colorbar()
        plt.title("Correlation Matrix")
        plt.xticks(range(len(corr.columns)), corr.columns, rotation=45)
        plt.yticks(range(len(corr.columns)), corr.columns)
        
    elif kind == "missing":
        msno.matrix(df)
        plt.title(f"Missing Data Pattern – {name}")
        
    else:
        raise ValueError(f"Invalid visualization type '{kind}' or missing required parameters.")

    plt.tight_layout()
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


@mcp.tool(structured_output=True)
async def infer_schema(name: str) -> SchemaResult:
    """
    Infer schema for a dataset including column types, nullability, ranges, and patterns.
    Returns structured schema information.
    """
    async with store_lock:
        df = data_store.get(name)
    if df is None:
        raise KeyError(f"Dataset '{name}' not found.")

    columns = []
    
    for col_name in df.columns:
        col_data = df[col_name]
        
        # Basic info
        nullable = col_data.isnull().any()
        unique_count = col_data.nunique()
        
        # Sample values (first 5 non-null)
        sample_values = col_data.dropna().head(5).tolist()
        
        # Type inference
        dtype = col_data.dtype
        
        if pd.api.types.is_numeric_dtype(dtype):
            col_type = "number"
            min_value = float(col_data.min()) if not col_data.empty else None
            max_value = float(col_data.max()) if not col_data.empty else None
            pattern = None
            max_length = None
            
        elif pd.api.types.is_datetime64_any_dtype(dtype):
            col_type = "datetime"
            min_value = None
            max_value = None
            pattern = None
            max_length = None
            
        else:
            col_type = "string"
            min_value = None
            max_value = None
            
            # String pattern analysis
            str_data = col_data.astype(str)
            max_length = int(str_data.str.len().max()) if not str_data.empty else None
            
            # Simple pattern detection
            pattern = None
            if not str_data.empty:
                # Check for email pattern
                email_pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
                email_matches = str_data.str.match(email_pattern, na=False)
                if email_matches.sum() > len(str_data) * 0.8:  # 80% match rate
                    pattern = "email"
                # Check for URL pattern
                elif str_data.str.contains(r'^https?://', na=False).sum() > len(str_data) * 0.8:
                    pattern = "url"
                # Check for phone pattern
                elif str_data.str.match(r'^[\d\s\-\(\)\+]+$', na=False).sum() > len(str_data) * 0.8:
                    pattern = "phone"
                # Check for date pattern
                elif str_data.str.match(r'^\d{4}-\d{2}-\d{2}$', na=False).sum() > len(str_data) * 0.8:
                    pattern = "date"
        
        column = SchemaColumn(
            name=col_name,
            type=col_type,
            nullable=nullable,
            min_value=min_value,
            max_value=max_value,
            pattern=pattern,
            max_length=max_length,
            unique_count=unique_count,
            sample_values=sample_values
        )
        columns.append(column)
    
    return SchemaResult(
        dataset_name=name,
        columns=columns,
        total_rows=len(df),
        total_columns=len(df.columns)
    )


@mcp.tool(structured_output=True)
async def detect_outliers(
    name: str,
    method: Literal["iqr", "isolation_forest", "lof"] = "iqr",
    factor: float = 1.5,
    contamination: float = 0.05,
    sample_size: int = 10_000
) -> OutlierResult:
    """
    Detect outliers in numeric columns using IQR, Isolation Forest, or Local Outlier Factor.
    Returns structured outlier information and visualization.
    """
    async with store_lock:
        df = data_store.get(name)
    if df is None:
        raise KeyError(f"Dataset '{name}' not found.")

    # Step 1: Identify numeric features
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if not num_cols:
        raise ValueError("No numeric columns to analyze")

    outliers = {}
    total_rows = len(df)

    # Step 2: Full-data outlier detection
    if method == "iqr":
        # IQR method - column by column
        for col in num_cols:
            Q1, Q3 = df[col].quantile([0.25, 0.75])
            IQR = Q3 - Q1
            lower = Q1 - factor * IQR
            upper = Q3 + factor * IQR
            outlier_indices = df[(df[col] < lower) | (df[col] > upper)].index.tolist()
            outliers[col] = outlier_indices

    elif method in ["isolation_forest", "lof"]:
        if not PYOD_AVAILABLE:
            raise ValueError(f"Model-based outlier detection requires pyod. Install with: pip install pyod")
        
        # Model-based methods - row-level detection
        try:
            if method == "isolation_forest":
                model = IsolationForest(contamination=contamination, random_state=42)
            else:  # lof
                model = LOF(n_neighbors=20, contamination=contamination)
            
            # Fit and predict on numeric columns
            model.fit(df[num_cols])
            predictions = model.predict(df[num_cols])
            
            # Get outlier indices (-1 = outlier)
            outlier_indices = np.where(predictions == -1)[0].tolist()
            
            # Attribute row-level flags to each column
            for col in num_cols:
                outliers[col] = outlier_indices
                
        except Exception as e:
            print(f"Warning: Model-based outlier detection failed: {e}")
            # Return empty outlier sets
            for col in num_cols:
                outliers[col] = []

    # Step 3: Aggregate counts
    counts = {col: len(outliers[col]) for col in num_cols}

    # Step 4: Prepare data for plotting
    plot_df = df if total_rows <= sample_size else df.sample(sample_size, random_state=42)

    # Step 5: Build multi-subplot figure
    n_cols = len(num_cols)
    n_rows = (n_cols + 1) // 2  # 2 columns per row, round up
    
    fig, axes = plt.subplots(n_rows, 2, figsize=(12, 4 * n_rows))
    if n_rows == 1:
        axes = axes.reshape(1, -1)
    
    # Flatten axes for easier iteration
    axes_flat = axes.flatten()
    
    for i, col in enumerate(num_cols):
        ax = axes_flat[i]
        
        # Create boxplot without outliers (we'll add them manually)
        box_data = plot_df[col].dropna()
        bp = ax.boxplot(box_data, vert=False, showfliers=False)
        
        # Overlay real outliers on top
        if outliers[col]:
            outlier_values = df.loc[outliers[col], col].tolist()
            ax.scatter(outlier_values, [1] * len(outlier_values), 
                      color="red", alpha=0.6, s=20, label=f"Outliers ({len(outlier_values)})")
        
        ax.set_title(f"{col} - {method.upper()}")
        ax.set_xlabel(col)
        
        # Add legend if there are outliers
        if outliers[col]:
            ax.legend()
    
    # Hide unused subplots
    for i in range(len(num_cols), len(axes_flat)):
        axes_flat[i].set_visible(False)
    
    plt.tight_layout()
    
    # Save to PNG
    img_path = REPORT_DIR / f"outliers_{name}_{method}.png"
    plt.savefig(img_path, dpi=150, bbox_inches="tight")
    plt.close()

    return OutlierResult(
        result=OutlierPayload(
            outliers=outliers,
            counts=counts,
            total_rows=total_rows
        ),
        image_uri=f"file://{img_path.resolve()}"
    )


@mcp.tool(structured_output=True)
async def target_analysis(name: str, target_col: str) -> TargetAnalysisResult:
    """
    Analyze target column for classification vs regression, compute priors/skewness,
    and find top correlated features. Returns structured analysis with visualizations.
    """
    async with store_lock:
        df = data_store.get(name)
    if df is None:
        raise KeyError(f"Dataset '{name}' not found.")
    
    if target_col not in df.columns:
        raise KeyError(f"Target column '{target_col}' not found in dataset '{name}'.")
    
    # Step 1: Load & Inspect Target Column
    series = df[target_col]
    n_unique = series.nunique()
    is_numeric = pd.api.types.is_numeric_dtype(series)
    
    # Branch Logic: Classification vs Regression
    is_classification = not is_numeric or n_unique <= 20
    
    result = {}
    plots = {}
    
    if is_classification:
        # Step 2: Classification Flow
        
        # 2.1 Compute Priors
        counts = series.value_counts()
        pcts = (counts / len(series) * 100).round(2)
        
        result.update({
            "task_type": "classification",
            "n_classes": n_unique,
            "counts": counts.to_dict(),
            "percentages": pcts.to_dict(),
            "imbalance_ratio": counts.max() / counts.min() if len(counts) > 1 else 1.0
        })
        
        # 2.2 Visualize Distribution (Bar chart)
        plt.figure(figsize=(10, 6))
        counts.plot(kind='bar')
        plt.title(f"Target Distribution - {target_col}")
        plt.xlabel("Class")
        plt.ylabel("Count")
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        bar_path = REPORT_DIR / f"target_bar_{name}_{target_col}.png"
        plt.savefig(bar_path, dpi=150, bbox_inches="tight")
        plt.close()
        plots["bar_uri"] = f"file://{bar_path.resolve()}"
        
        # 2.3 Human Checkpoint Message
        print(f"\n=== CLASSIFICATION ANALYSIS CHECKPOINT ===")
        print(f"Target column: {target_col}")
        print(f"Classes: {list(counts.index)}")
        print(f"Counts: {counts.to_dict()}")
        print(f"Imbalance ratio: {result['imbalance_ratio']:.2f}")
        print(f"Please verify the labels and imbalance ratio before proceeding.")
        print(f"Bar chart saved to: {bar_path}")
        print(f"==========================================\n")
        
    else:
        # Step 3: Regression Flow
        
        # 3.1 Univariate Summary
        skew = series.skew()
        kurt = series.kurtosis()
        
        result.update({
            "task_type": "regression",
            "skewness": round(skew, 3),
            "kurtosis": round(kurt, 3),
            "mean": round(series.mean(), 3),
            "std": round(series.std(), 3),
            "min": round(series.min(), 3),
            "max": round(series.max(), 3),
            "needs_transform": abs(skew) > 2
        })
        
        # 3.2 Visualize Distribution (Histogram + Q-Q plot)
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Histogram
        ax1.hist(series.dropna(), bins=30, alpha=0.7, edgecolor='black')
        ax1.set_title(f"Target Distribution - {target_col}")
        ax1.set_xlabel(target_col)
        ax1.set_ylabel("Frequency")
        ax1.axvline(series.mean(), color='red', linestyle='--', label=f'Mean: {series.mean():.2f}')
        ax1.legend()
        
        # Q-Q plot
        from scipy import stats
        stats.probplot(series.dropna(), dist="norm", plot=ax2)
        ax2.set_title("Q-Q Plot vs Normal Distribution")
        
        plt.tight_layout()
        
        hist_path = REPORT_DIR / f"target_hist_{name}_{target_col}.png"
        plt.savefig(hist_path, dpi=150, bbox_inches="tight")
        plt.close()
        plots["hist_uri"] = f"file://{hist_path.resolve()}"
        
        # 3.3 Human Checkpoint Message
        print(f"\n=== REGRESSION ANALYSIS CHECKPOINT ===")
        print(f"Target column: {target_col}")
        print(f"Skewness: {skew:.3f} {'(needs transform)' if abs(skew) > 2 else '(OK)'}")
        print(f"Kurtosis: {kurt:.3f}")
        print(f"Mean: {series.mean():.3f}, Std: {series.std():.3f}")
        print(f"Range: {series.min():.3f} to {series.max():.3f}")
        print(f"Please verify the distribution and consider transformation if skew > 2.")
        print(f"Histogram and Q-Q plot saved to: {hist_path}")
        print(f"=====================================\n")
    
    # Step 4: Feature-Target Correlation
    # Get numeric features (excluding target)
    numeric_features = df.select_dtypes(include=[np.number]).columns.tolist()
    if target_col in numeric_features:
        numeric_features.remove(target_col)
    
    if numeric_features:
        correlations = []
        
        for feature in numeric_features:
            if is_classification:
                # For classification, use point-biserial correlation (special case of Pearson)
                if n_unique == 2:
                    # Binary classification
                    corr = df[feature].corr(series.astype(int))
                else:
                    # Multi-class: use correlation with numeric encoding
                    corr = df[feature].corr(pd.Series(pd.Categorical(series).codes, index=series.index))
            else:
                # For regression, use Pearson correlation
                corr = df[feature].corr(series)
            
            if not pd.isna(corr):
                correlations.append({
                    "feature": feature,
                    "rho": round(corr, 3),
                    "abs_rho": abs(corr)
                })
        
        # Sort by absolute correlation and take top 10
        correlations.sort(key=lambda x: x["abs_rho"], reverse=True)
        top_features = correlations[:10]
        
        result["top_features"] = top_features
        
        # 4.3 Visual Confirmation (Scatter plots for top 3)
        if len(top_features) >= 3:
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))
            
            for i, feat_info in enumerate(top_features[:3]):
                feature = feat_info["feature"]
                rho = feat_info["rho"]
                x = df[feature]
                if is_classification:
                    # Use numeric codes for target
                    y = pd.Series(pd.Categorical(series).codes, index=series.index)
                else:
                    y = series
                axes[i].scatter(x, y, alpha=0.6, s=20)
                axes[i].set_xlabel(feature)
                axes[i].set_ylabel(target_col)
                axes[i].set_title(f"{feature} vs {target_col}\nρ = {rho:.3f}")
                # Add trend line only if both x and y are numeric
                if pd.api.types.is_numeric_dtype(x) and pd.api.types.is_numeric_dtype(y):
                    try:
                        z = np.polyfit(x.dropna(), y.dropna(), 1)
                        p = np.poly1d(z)
                        axes[i].plot(x, p(x), "r--", alpha=0.8)
                    except Exception:
                        pass
            
            plt.tight_layout()
            
            scatter_path = REPORT_DIR / f"target_scatter_{name}_{target_col}.png"
            plt.savefig(scatter_path, dpi=150, bbox_inches="tight")
            plt.close()
            plots["scatter_uri"] = f"file://{scatter_path.resolve()}"
        
        # 4.4 Human Checkpoint Message
        print(f"\n=== CORRELATION ANALYSIS CHECKPOINT ===")
        print(f"Top 10 features by absolute correlation:")
        for i, feat in enumerate(top_features[:10], 1):
            print(f"{i:2d}. {feat['feature']:15s} | ρ = {feat['rho']:6.3f}")
        print(f"Please verify these make sense (no ID leakage, no spurious spikes).")
        if len(top_features) >= 3:
            print(f"Scatter plots for top 3 saved to: {scatter_path}")
        print(f"========================================\n")
    
    return TargetAnalysisResult(result=result, plots=plots)


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

    summary = json.loads(snap.json())
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
        snap = await asyncio.to_thread(rpt.run, _ds_regression(df))
        html = REPORT_DIR / f"perf_regression_{len(df)}.html"
        snap.save_html(html)

        summary = json.loads(snap.json())
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