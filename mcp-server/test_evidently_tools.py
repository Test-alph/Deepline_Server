#!/usr/bin/env python3
"""
Test script for the MCP server with Evidently tools.
This script tests all the new tools we added.
"""

import asyncio
import pandas as pd
import numpy as np
from pathlib import Path

# Import the server module to test the tools directly
import server

async def test_basic_tools():
    """Test the basic EDA tools."""
    print("🔍 Testing Basic EDA Tools...")
    
    # Test 1: Load data
    print("\n1. Testing load_data...")
    result = await server.load_data("iris.csv", "iris_base")
    print(f"✅ {result}")
    
    # Test 2: Basic info
    print("\n2. Testing basic_info...")
    result = await server.basic_info("iris_base")
    print(f"✅ Basic info retrieved (length: {len(result)} chars)")
    
    # Test 3: Missing data analysis
    print("\n3. Testing missing_data_analysis...")
    result = await server.missing_data_analysis("iris_base")
    print(f"✅ Missing data analysis completed (returned {len(result)} items)")
    
    # Test 4: Statistical summary
    print("\n4. Testing statistical_summary...")
    result = await server.statistical_summary("iris_base")
    print(f"✅ Statistical summary completed (length: {len(result)} chars)")
    
    # Test 5: List datasets
    print("\n5. Testing list_datasets...")
    result = await server.list_datasets()
    print(f"✅ {result}")

async def test_evidently_tools():
    """Test the new Evidently-based tools."""
    print("\n🔬 Testing Evidently Tools...")
    
    # Load a second dataset for drift testing
    print("\n1. Loading second dataset for drift testing...")
    await server.load_data("iris.csv", "iris_current")
    
    # Test 1: Data quality report
    print("\n2. Testing data_quality_report...")
    try:
        result = await server.data_quality_report("iris_base", "Iris Data Quality Report")
        print(f"✅ Data quality report: {result}")
        print(f"   HTML saved to: {result.html_uri}")
    except Exception as e:
        print(f"❌ Data quality report failed: {e}")
    
    # Test 2: Drift analysis
    print("\n3. Testing drift_analysis...")
    try:
        result = await server.drift_analysis("iris_base", "iris_current")
        print(f"✅ Drift analysis: {result}")
        print(f"   Drift detected: {result.drift_detected}")
        print(f"   HTML saved to: {result.html_uri}")
    except Exception as e:
        print(f"❌ Drift analysis failed: {e}")
    
    # Test 3: Model performance report (classification)
    print("\n4. Testing model_performance_report (classification)...")
    try:
        # Generate dummy classification data
        n = 150
        y_true = [np.random.randint(0, 3) for _ in range(n)]
        y_pred = [np.random.randint(0, 3) for _ in range(n)]
        
        result = await server.model_performance_report(
            y_true=y_true,
            y_pred=y_pred,
            labels=[0, 1, 2],
            model_type="classification"
        )
        print(f"✅ Model performance report: {result}")
        print(f"   Metrics: {result.metrics}")
        print(f"   HTML saved to: {result.html_uri}")
    except Exception as e:
        print(f"❌ Model performance report failed: {e}")
    
    # Test 4: Model performance report (binary classification)
    print("\n5. Testing model_performance_report (binary classification)...")
    try:
        # Generate dummy binary classification data
        n = 150
        y_true = [np.random.randint(0, 2) for _ in range(n)]
        y_pred = [np.random.randint(0, 2) for _ in range(n)]
        
        result = await server.model_performance_report(
            y_true=y_true,
            y_pred=y_pred,
            model_type="classification"
        )
        print(f"✅ Binary classification report: {result}")
        print(f"   Metrics: {result.metrics}")
        print(f"   HTML saved to: {result.html_uri}")
    except Exception as e:
        print(f"❌ Binary classification report failed: {e}")
    
    # Test 5: Model performance report (regression)
    print("\n6. Testing model_performance_report (regression)...")
    try:
        # Generate dummy regression data
        n = 150
        y_true = [np.random.normal(5.5, 1.0) for _ in range(n)]
        y_pred = [y + np.random.normal(0, 0.2) for y in y_true]
        
        result = await server.model_performance_report(
            y_true=y_true,
            y_pred=y_pred,
            model_type="regression"
        )
        print(f"✅ Model performance report: {result}")
        print(f"   Metrics: {result.metrics}")
        print(f"   HTML saved to: {result.html_uri}")
    except Exception as e:
        print(f"❌ Model performance report failed: {e}")

async def main():
    """Run all tests."""
    print("🚀 Starting MCP Server Evidently Tools Test")
    print("=" * 50)
    
    try:
        await test_basic_tools()
        await test_evidently_tools()
        
        print("\n" + "=" * 50)
        print("✅ All tests completed!")
        
        # Check if reports directory was created
        reports_dir = Path("reports")
        if reports_dir.exists():
            print(f"📁 Reports directory: {reports_dir.absolute()}")
            html_files = list(reports_dir.glob("*.html"))
            print(f"📄 Generated {len(html_files)} HTML reports:")
            for html_file in html_files:
                print(f"   - {html_file.name}")
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(main()) 