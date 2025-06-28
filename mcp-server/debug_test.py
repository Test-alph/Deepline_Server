#!/usr/bin/env python3
"""
Debug script to test the debug tools and surface summary keys.
"""

import asyncio
import json
import random
import numpy as np

# Import the server module to test the tools directly
import server

async def test_debug_tools():
    """Test the debug tools to surface summary keys."""
    print("🔍 Testing Debug Tools to Surface Summary Keys")
    print("=" * 60)
    
    # Test 1: Load data
    print("\n1. Loading datasets...")
    await server.load_data("iris.csv", "iris_a")
    await server.load_data("iris.csv", "iris_b")
    print("✅ Datasets loaded")
    
    # Test 2: Debug drift summary
    print("\n2. Testing debug_drift_summary...")
    try:
        result = await server.debug_drift_summary("iris_a", "iris_b")
        print(f"✅ Debug drift summary completed")
        print(f"   HTML saved to: {result.html_uri}")
        print(f"   Top-level keys: {result.keys}")
        print(f"   Summary excerpt: {json.dumps(dict(list(result.summary.items())[:3]), indent=2)}")
    except Exception as e:
        print(f"❌ Debug drift summary failed: {e}")
        import traceback
        traceback.print_exc()
    
    # Test 3: Debug performance summary (classification) - with proper config
    print("\n3. Testing debug_perf_summary (classification)...")
    try:
        # Generate dummy classification data
        n = 100
        y_true = [random.randint(0, 2) for _ in range(n)]
        y_pred = [random.randint(0, 2) for _ in range(n)]
        
        # Create DataFrame with proper column names
        df = server.pd.DataFrame({"target": y_true, "prediction": y_pred})
        
        # Use proper classification configuration
        if len(set(y_true) | set(y_pred)) > 2:
            ds = server.Dataset.from_pandas(
                df,
                server.DataDefinition(
                    classification=[server.MulticlassClassification(target="target", prediction="prediction", labels=[0, 1, 2])]
                )
            )
        else:
            ds = server.Dataset.from_pandas(
                df,
                server.DataDefinition(
                    classification=[server.BinaryClassification(target="target", prediction="prediction")]
                )
            )
        
        preset = server.ClassificationPreset()
        rpt = server.Report([preset])
        snap = await asyncio.to_thread(rpt.run, ds)
        
        html = server.REPORT_DIR / "perf_debug_class.html"
        snap.save_html(html)
        
        summary = json.loads(snap.json())
        print(f"✅ Debug classification summary completed")
        print(f"   HTML saved to: file://{html.resolve()}")
        print(f"   Top-level keys: {list(summary.keys())}")
        print(f"   Summary excerpt: {json.dumps(dict(list(summary.items())[:3]), indent=2)}")
    except Exception as e:
        print(f"❌ Debug classification summary failed: {e}")
        import traceback
        traceback.print_exc()
    
    # Test 4: Debug performance summary (regression)
    print("\n4. Testing debug_perf_summary (regression)...")
    try:
        # Generate dummy regression data
        n = 100
        y_true = [np.random.normal(5.5, 1.0) for _ in range(n)]
        y_pred = [y + np.random.normal(0, 0.2) for y in y_true]
        
        # Create DataFrame with proper column names
        df = server.pd.DataFrame({"target": y_true, "prediction": y_pred})
        
        # Use proper regression configuration
        ds = server.Dataset.from_pandas(
            df,
            server.DataDefinition(
                regression=[server.Regression(target="target", prediction="prediction")]
            )
        )
        
        preset = server.RegressionPreset()
        rpt = server.Report([preset])
        snap = await asyncio.to_thread(rpt.run, ds)
        
        html = server.REPORT_DIR / "perf_debug_reg.html"
        snap.save_html(html)
        
        summary = json.loads(snap.json())
        print(f"✅ Debug regression summary completed")
        print(f"   HTML saved to: file://{html.resolve()}")
        print(f"   Top-level keys: {list(summary.keys())}")
        print(f"   Summary excerpt: {json.dumps(dict(list(summary.items())[:3]), indent=2)}")
    except Exception as e:
        print(f"❌ Debug regression summary failed: {e}")
        import traceback
        traceback.print_exc()

async def main():
    """Run all debug tests."""
    print("🚀 Starting Debug Tools Test")
    print("=" * 60)
    
    try:
        await test_debug_tools()
        
        print("\n" + "=" * 60)
        print("✅ Debug tests completed!")
        
    except Exception as e:
        print(f"\n❌ Debug test failed with error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(main()) 