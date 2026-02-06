# Test Dashboard JSON Serialization
# This script verifies that dashboard data can be properly serialized to JSON

import json
import pandas as pd
import numpy as np
from multi_agent_analyst import generate_dashboard_data
from data_loader import load_and_process_data

print("Testing Dashboard JSON Serialization...")
print("=" * 60)

try:
    # Check if demo file exists
    if not pd.io.common.file_exists('dashboard_demo.csv'):
        print("Creating demo dataset...")
        demo_data = pd.DataFrame({
            'Product': ['Laptop', 'Phone', 'Tablet'] * 20,
            'Sales': np.random.randint(100, 1000, 60),
            'Revenue': np.random.uniform(1000, 50000, 60),
            'Category': ['Electronics', 'Mobile', 'Computers'] * 20
        })
        demo_data.to_csv('dashboard_demo.csv', index=False)
        print("[OK] Demo dataset created")
    
    # Load and process data
    print("\nLoading data...")
    df, quality_report = load_and_process_data('dashboard_demo.csv')
    print(f"[OK] Data loaded: {len(df)} rows, {len(df.columns)} columns")
    
    # Generate dashboard data
    print("\nGenerating dashboard data...")
    dashboard_data = generate_dashboard_data(df, quality_report)
    print(f"[OK] Dashboard generated with {len(dashboard_data['charts'])} charts")
    
    # Test JSON serialization
    print("\nTesting JSON serialization...")
    json_string = json.dumps(dashboard_data, indent=2)
    print("[OK] JSON serialization successful!")
    
    # Verify it can be parsed back
    parsed_data = json.loads(json_string)
    print("[OK] JSON parsing successful!")
    
    # Show summary
    print("\n" + "=" * 60)
    print("Dashboard Summary:")
    print(f"   - Metrics Cards: {len(dashboard_data['metrics'])}")
    print(f"   - Charts: {len(dashboard_data['charts'])}")
    print(f"   - Data Quality Viz: {'Yes' if dashboard_data['data_quality_viz'] else 'No'}")
    
    print("\nCharts Generated:")
    for i, chart in enumerate(dashboard_data['charts'], 1):
        print(f"   {i}. {chart['title']} ({chart['type']})")
    
    print("\n" + "=" * 60)
    print("[SUCCESS] All tests passed! Dashboard is ready to use.")
    print("\nStart your server with:")
    print("   uvicorn backend:app --reload")
    print("\nThen upload 'dashboard_demo.csv' to test!")
    
except Exception as e:
    print(f"\n[ERROR] {str(e)}")
    import traceback
    traceback.print_exc()
    print("\nFix the error above and try again.")
