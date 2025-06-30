"""
Unit tests for EDA Server.
Tests edge cases and core functionality.
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import tempfile
import asyncio
from unittest.mock import patch, MagicMock

# Import the modules to test
import server
import config
import utils

class TestEDAServer:
    """Test suite for EDA Server functionality."""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample datasets for testing."""
        # Normal dataset
        normal_df = pd.DataFrame({
            'id': range(100),
            'numeric': np.random.normal(0, 1, 100),
            'categorical': np.random.choice(['A', 'B', 'C'], 100),
            'missing_numeric': [np.nan if i % 10 == 0 else i for i in range(100)],
            'missing_categorical': [np.nan if i % 15 == 0 else f'Cat{i}' for i in range(100)]
        })
        
        # Edge case datasets
        all_missing_df = pd.DataFrame({
            'col1': [np.nan] * 50,
            'col2': [np.nan] * 50
        })
        
        constant_df = pd.DataFrame({
            'constant': [1] * 100,
            'mixed': [1 if i % 2 == 0 else 2 for i in range(100)]
        })
        
        large_df = pd.DataFrame({
            'id': range(50000),
            'value': np.random.normal(0, 1, 50000)
        })
        
        return {
            'normal': normal_df,
            'all_missing': all_missing_df,
            'constant': constant_df,
            'large': large_df
        }
    
    @pytest.fixture
    def temp_csv(self, sample_data):
        """Create temporary CSV file for testing."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            sample_data['normal'].to_csv(f.name, index=False)
            yield f.name
        Path(f.name).unlink()  # Clean up
    
    def test_load_data(self, temp_csv):
        """Test data loading functionality."""
        # Test successful load
        result = asyncio.run(server.load_data(temp_csv, "test_dataset"))
        assert "Loaded 'test_dataset'" in result
        assert "Memory usage:" in result
        
        # Test missing file
        with pytest.raises(FileNotFoundError):
            asyncio.run(server.load_data("nonexistent.csv", "test"))
        
        # Test unsupported format
        with tempfile.NamedTemporaryFile(suffix='.txt', delete=False) as f:
            f.write(b"test data")
            f.flush()
            with pytest.raises(ValueError):
                asyncio.run(server.load_data(f.name, "test"))
            Path(f.name).unlink()
    
    def test_basic_info(self, sample_data):
        """Test basic info functionality."""
        # Load data first
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            sample_data['normal'].to_csv(f.name, index=False)
            asyncio.run(server.load_data(f.name, "test"))
            Path(f.name).unlink()
        
        result = asyncio.run(server.basic_info("test"))
        assert "Dataset: test" in result
        assert "Shape: (100, 5)" in result
    
    def test_missing_data_analysis_edge_cases(self, sample_data):
        """Test missing data analysis with edge cases."""
        # Test all missing data
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            sample_data['all_missing'].to_csv(f.name, index=False)
            asyncio.run(server.load_data(f.name, "all_missing"))
            Path(f.name).unlink()
        
        result = asyncio.run(server.missing_data_analysis("all_missing"))
        assert result.total_rows == 50
        assert result.missing_summary['overall_missing_rate_percent'] == 100.0
    
    def test_outlier_detection_methods(self, sample_data):
        """Test different outlier detection methods."""
        # Add some outliers to normal data
        df_with_outliers = sample_data['normal'].copy()
        df_with_outliers.loc[0, 'numeric'] = 100  # Add outlier
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            df_with_outliers.to_csv(f.name, index=False)
            asyncio.run(server.load_data(f.name, "outliers"))
            Path(f.name).unlink()
        
        # Test IQR method
        result_iqr = asyncio.run(server.detect_outliers("outliers", method="iqr"))
        assert result_iqr.result.method_used == "iqr"
        
        # Test model-based methods if pyod is available
        if hasattr(server, 'PYOD_AVAILABLE') and server.PYOD_AVAILABLE:
            result_if = asyncio.run(server.detect_outliers("outliers", method="isolation_forest"))
            assert result_if.result.method_used == "isolation_forest"
    
    def test_schema_inference_patterns(self, sample_data):
        """Test schema inference with various patterns."""
        # Create dataset with patterns
        pattern_df = pd.DataFrame({
            'email': ['test@example.com'] * 50,
            'phone': ['123-456-7890'] * 50,
            'date': ['2023-01-01'] * 50,
            'uuid': ['550e8400-e29b-41d4-a716-446655440000'] * 50,
            'normal': ['text'] * 50
        })
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            pattern_df.to_csv(f.name, index=False)
            asyncio.run(server.load_data(f.name, "patterns"))
            Path(f.name).unlink()
        
        result = asyncio.run(server.infer_schema("patterns"))
        
        # Check pattern detection
        email_col = next((col for col in result.columns if col.name == 'email'), None)
        assert email_col is not None
        assert email_col.pattern == "email"
    
    def test_feature_transformation_edge_cases(self, sample_data):
        """Test feature transformation with edge cases."""
        # Test with constant column
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            sample_data['constant'].to_csv(f.name, index=False)
            asyncio.run(server.load_data(f.name, "constant"))
            Path(f.name).unlink()
        
        result = asyncio.run(server.feature_transformation("constant", ["boxcox", "binning"]))
        assert result.original_shape == (100, 2)
    
    def test_large_dataset_handling(self, sample_data):
        """Test handling of large datasets."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            sample_data['large'].to_csv(f.name, index=False)
            asyncio.run(server.load_data(f.name, "large"))
            Path(f.name).unlink()
        
        # Test statistical summary with sampling
        result = asyncio.run(server.statistical_summary("large"))
        assert "sample rows" in result  # Should indicate sampling
    
    def test_config_loading(self):
        """Test configuration loading and validation."""
        # Test default config
        cfg = config.get_config()
        assert cfg.missing_data.column_drop_threshold == 0.50
        assert cfg.outlier_detection.iqr_factor == 1.5
        
        # Test config validation
        with pytest.raises(Exception):
            config.EDAConfig(
                missing_data=config.MissingDataConfig(column_drop_threshold=1.5)  # Invalid value
            )
    
    def test_utility_functions(self, sample_data):
        """Test utility functions."""
        df = sample_data['normal']
        
        # Test numeric column detection
        numeric_cols = utils.get_numeric_columns(df)
        assert 'numeric' in numeric_cols
        assert 'id' in numeric_cols
        
        # Test categorical column detection
        cat_cols = utils.get_categorical_columns(df)
        assert 'categorical' in cat_cols
        
        # Test missing stats calculation
        missing_stats = utils.calculate_missing_stats(df)
        assert missing_stats['total_rows'] == 100
        assert missing_stats['total_columns'] == 5
        
        # Test ID column detection
        id_cols = utils.detect_id_columns(df)
        assert 'id' in id_cols
        
        # Test memory formatting
        memory_str = utils.format_memory_usage(1024 * 1024)  # 1MB
        assert "MB" in memory_str
    
    def test_edge_case_handling(self):
        """Test various edge cases."""
        # Test empty dataframe
        empty_df = pd.DataFrame()
        missing_stats = utils.calculate_missing_stats(empty_df)
        assert missing_stats['total_rows'] == 0
        
        # Test single value dataframe
        single_df = pd.DataFrame({'col': [1]})
        skewness = utils.calculate_skewness(single_df['col'])
        assert skewness == 0.0  # Should handle edge case
        
        # Test all NaN dataframe
        nan_df = pd.DataFrame({'col': [np.nan] * 10})
        skewness = utils.calculate_skewness(nan_df['col'])
        assert skewness == 0.0
    
    @pytest.mark.asyncio
    async def test_concurrent_access(self, sample_data):
        """Test concurrent access to data store."""
        # Load multiple datasets concurrently
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            sample_data['normal'].to_csv(f.name, index=False)
            
            # Load same dataset multiple times concurrently
            tasks = [
                server.load_data(f.name, f"test_{i}")
                for i in range(5)
            ]
            results = await asyncio.gather(*tasks)
            
            Path(f.name).unlink()
        
        # All should succeed
        assert all("Loaded" in result for result in results)
        
        # Test listing datasets
        list_result = await server.list_datasets()
        assert "test_0" in list_result
        assert "test_4" in list_result

class TestConfiguration:
    """Test configuration management."""
    
    def test_config_validation(self):
        """Test configuration validation."""
        # Valid config
        valid_config = config.EDAConfig(
            missing_data=config.MissingDataConfig(),
            outlier_detection=config.OutlierDetectionConfig(),
            schema_inference=config.SchemaInferenceConfig(),
            feature_transformation=config.FeatureTransformationConfig(),
            visualization=config.VisualizationConfig(),
            performance=config.PerformanceConfig(),
            checkpoints=config.CheckpointsConfig()
        )
        assert valid_config is not None
        
        # Invalid config should raise error
        with pytest.raises(Exception):
            config.MissingDataConfig(column_drop_threshold=1.5)  # > 1.0
    
    def test_config_defaults(self):
        """Test configuration defaults."""
        cfg = config.get_config()
        
        # Check some key defaults
        assert cfg.missing_data.column_drop_threshold == 0.50
        assert cfg.outlier_detection.iqr_factor == 1.5
        assert cfg.feature_transformation.rare_category_threshold == 0.005

class TestUtils:
    """Test utility functions."""
    
    def test_quantile_binning(self):
        """Test quantile binning utility."""
        data = pd.Series(range(100))
        bins = utils.create_quantile_bins(data, n_bins=5)
        assert len(bins.unique()) == 5
        
        # Test with insufficient data
        small_data = pd.Series([1, 2])
        bins = utils.create_quantile_bins(small_data, n_bins=5)
        assert len(bins) == 2
        
        # Test with all same values
        constant_data = pd.Series([1] * 10)
        bins = utils.create_quantile_bins(constant_data, n_bins=5)
        assert len(bins) == 10
    
    def test_vif_calculation(self):
        """Test VIF calculation utility."""
        # Create correlated data
        np.random.seed(42)
        x1 = np.random.normal(0, 1, 100)
        x2 = x1 + np.random.normal(0, 0.1, 100)  # Highly correlated
        x3 = np.random.normal(0, 1, 100)  # Independent
        
        df = pd.DataFrame({'x1': x1, 'x2': x2, 'x3': x3})
        vif_scores = utils.calculate_vif_scores(df, ['x1', 'x2', 'x3'])
        
        # x2 should have high VIF due to correlation with x1
        assert vif_scores['x2'] > 5.0
        # x3 should have low VIF
        assert vif_scores['x3'] < 5.0

if __name__ == "__main__":
    pytest.main([__file__, "-v"]) 