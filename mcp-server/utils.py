"""
Utility functions for EDA Server.
Common operations used across multiple tools.
"""

import pandas as pd
import numpy as np
from typing import List, Tuple, Dict, Any, Optional
from scipy.stats import skew, kurtosis
import matplotlib.pyplot as plt
from config import config

def calculate_skewness(data: pd.Series) -> float:
    """
    Calculate skewness with proper handling of edge cases.
    
    Args:
        data: Input data series
        
    Returns:
        Skewness value
    """
    if len(data) < 3:
        return 0.0
    return float(skew(data.dropna()))

def calculate_kurtosis(data: pd.Series) -> float:
    """
    Calculate kurtosis with proper handling of edge cases.
    
    Args:
        data: Input data series
        
    Returns:
        Kurtosis value
    """
    if len(data) < 4:
        return 0.0
    return float(kurtosis(data.dropna()))

def get_numeric_columns(df: pd.DataFrame, exclude_cols: Optional[List[str]] = None) -> List[str]:
    """
    Get numeric columns from dataframe with optional exclusions.
    
    Args:
        df: Input dataframe
        exclude_cols: Columns to exclude
        
    Returns:
        List of numeric column names
    """
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if exclude_cols:
        numeric_cols = [col for col in numeric_cols if col not in exclude_cols]
    return numeric_cols

def get_categorical_columns(df: pd.DataFrame) -> List[str]:
    """
    Get categorical columns from dataframe.
    
    Args:
        df: Input dataframe
        
    Returns:
        List of categorical column names
    """
    return df.select_dtypes(include=['object', 'category']).columns.tolist()

def calculate_missing_stats(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Calculate comprehensive missing data statistics.
    
    Args:
        df: Input dataframe
        
    Returns:
        Dictionary with missing data statistics
    """
    total_rows = len(df)
    total_columns = len(df.columns)
    
    missing_counts = df.isnull().sum()
    missing_percentages = (missing_counts / total_rows * 100).round(3)
    total_missing = missing_counts.sum()
    overall_missing_rate = (total_missing / (total_rows * total_columns) * 100).round(3)
    
    return {
        "total_missing_values": int(total_missing),
        "overall_missing_rate_percent": overall_missing_rate,
        "columns_with_missing": int((missing_counts > 0).sum()),
        "missing_by_column": missing_counts.to_dict(),
        "missing_percentages": missing_percentages.to_dict(),
        "total_rows": total_rows,
        "total_columns": total_columns
    }

def detect_id_columns(df: pd.DataFrame) -> List[str]:
    """
    Detect ID columns based on uniqueness and naming patterns.
    
    Args:
        df: Input dataframe
        
    Returns:
        List of detected ID column names
    """
    id_columns = []
    threshold = config.schema_inference.id_uniqueness_threshold
    
    for col in df.columns:
        unique_count = df[col].nunique()
        total_count = len(df[col])
        uniqueness_ratio = unique_count / total_count if total_count > 0 else 0
        
        if uniqueness_ratio >= threshold:
            # Check naming patterns
            col_lower = col.lower()
            id_patterns = ['id', 'id_', '_id', 'key', 'pk', 'primary_key', 'uuid']
            
            if any(pattern in col_lower for pattern in id_patterns):
                id_columns.append(col)
            elif uniqueness_ratio >= 0.95:  # Very high uniqueness
                id_columns.append(col)
    
    return id_columns

def infer_datetime_columns(df: pd.DataFrame) -> List[str]:
    """
    Infer datetime columns from string data.
    
    Args:
        df: Input dataframe
        
    Returns:
        List of inferred datetime column names
    """
    datetime_columns = []
    success_rate_threshold = config.schema_inference.datetime_success_rate
    
    for col in df.columns:
        if df[col].dtype == 'object':  # String columns
            unique_ratio = df[col].nunique() / len(df[col])
            if unique_ratio > 0.5:  # High uniqueness suggests potential datetime
                try:
                    sample_dates = df[col].dropna().head(100)
                    if len(sample_dates) > 0:
                        parsed_dates = pd.to_datetime(sample_dates, errors='coerce')
                        success_rate = parsed_dates.notna().sum() / len(sample_dates)
                        if success_rate > success_rate_threshold:
                            datetime_columns.append(col)
                except:
                    continue
    
    return datetime_columns

def create_quantile_bins(data: pd.Series, n_bins: int = 5) -> pd.Series:
    """
    Create quantile-based bins for numeric data.
    
    Args:
        data: Input data series
        n_bins: Number of bins
        
    Returns:
        Binned data series
    """
    if len(data) < n_bins:
        return pd.Series(index=data.index, dtype='object')
    
    try:
        bins = pd.qcut(data, q=n_bins, labels=[f"Q{i+1}" for i in range(n_bins)], duplicates='drop')
        return bins
    except ValueError:
        # Fallback to regular cut if quantiles fail
        try:
            bins = pd.cut(data, bins=n_bins, labels=[f"B{i+1}" for i in range(n_bins)], duplicates='drop')
            return bins
        except ValueError:
            return pd.Series(index=data.index, dtype='object')

def calculate_vif_scores(df: pd.DataFrame, numeric_cols: List[str]) -> Dict[str, float]:
    """
    Calculate Variance Inflation Factor for numeric columns.
    
    Args:
        df: Input dataframe
        numeric_cols: List of numeric columns to analyze
        
    Returns:
        Dictionary mapping column names to VIF scores
    """
    from sklearn.linear_model import LinearRegression
    
    vif_scores = {}
    numeric_data = df[numeric_cols].dropna()
    
    if len(numeric_data) <= len(numeric_cols):
        return vif_scores
    
    for col in numeric_cols:
        if col in numeric_data.columns:
            X = numeric_data.drop(columns=[col])
            y = numeric_data[col]
            
            if len(X.columns) > 0:
                try:
                    model = LinearRegression()
                    model.fit(X, y)
                    r_squared = model.score(X, y)
                    
                    if r_squared < 1:  # Avoid division by zero
                        vif = 1 / (1 - r_squared)
                        vif_scores[col] = round(vif, 3)
                    else:
                        vif_scores[col] = float('inf')
                except:
                    vif_scores[col] = float('inf')
    
    return vif_scores

def sample_dataframe(df: pd.DataFrame, max_rows: int = 10000) -> pd.DataFrame:
    """
    Sample dataframe if it exceeds size limits.
    
    Args:
        df: Input dataframe
        max_rows: Maximum number of rows
        
    Returns:
        Sampled dataframe
    """
    if len(df) <= max_rows:
        return df
    
    return df.sample(n=max_rows, random_state=42)

def format_memory_usage(memory_bytes: int) -> str:
    """
    Format memory usage in human-readable format.
    
    Args:
        memory_bytes: Memory usage in bytes
        
    Returns:
        Formatted memory string
    """
    if memory_bytes < 1024:
        return f"{memory_bytes} B"
    elif memory_bytes < 1024 * 1024:
        return f"{memory_bytes / 1024:.2f} KB"
    elif memory_bytes < 1024 * 1024 * 1024:
        return f"{memory_bytes / (1024 * 1024):.2f} MB"
    else:
        return f"{memory_bytes / (1024 * 1024 * 1024):.2f} GB"

def create_checkpoint_message(title: str, dataset_name: str, summary: Dict[str, Any]) -> str:
    """
    Create standardized checkpoint message.
    
    Args:
        title: Checkpoint title
        dataset_name: Name of the dataset
        summary: Summary statistics
        
    Returns:
        Formatted checkpoint message
    """
    message = f"""
=== {title.upper()} CHECKPOINT ===
Dataset: {dataset_name}
"""
    
    for key, value in summary.items():
        if isinstance(value, (int, float)):
            message += f"{key.replace('_', ' ').title()}: {value}\n"
        elif isinstance(value, list):
            message += f"{key.replace('_', ' ').title()}: {len(value)} items\n"
        else:
            message += f"{key.replace('_', ' ').title()}: {value}\n"
    
    message += "\nPlease review the results and approve before proceeding.\n"
    message += "=" * 50
    
    return message

def safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float:
    """
    Safely divide two numbers, returning default if denominator is zero.
    
    Args:
        numerator: Numerator
        denominator: Denominator
        default: Default value if division fails
        
    Returns:
        Division result or default value
    """
    try:
        return numerator / denominator if denominator != 0 else default
    except:
        return default 