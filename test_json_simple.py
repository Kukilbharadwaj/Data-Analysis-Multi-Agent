# Simple JSON Test - No Dependencies
import json
import pandas as pd
import numpy as np

print("Testing JSON Serialization Fix...")
print("=" * 60)

# Create test dataframe
df = pd.DataFrame({
    'Product': ['Laptop', 'Phone', 'Tablet'] * 10,
    'Sales': np.random.randint(100, 1000, 30),
    'Revenue': np.random.uniform(1000, 50000, 30),
    'Category': ['A', 'B', 'C'] * 10
})

# Mock quality report
quality_report = {
    'data_quality': {
        'duplicates_removed': 5,
        'missing_values': {},
        'outliers': {}
    },
    'file_info': {
        'encoding': 'UTF-8'
    }
}

print("Test data created")
print(f"Rows: {len(df)}, Columns: {len(df.columns)}")

# Import the function
from multi_agent_analyst import generate_dashboard_data

print("\nGenerating dashboard...")
dashboard_data = generate_dashboard_data(df, quality_report)

print(f"Charts generated: {len(dashboard_data['charts'])}")
print(f"Metrics generated: {len(dashboard_data['metrics'])}")

# Test JSON serialization
print("\nTesting JSON serialization...")
try:
    json_str = json.dumps(dashboard_data, indent=2)
    print("[OK] JSON serialization successful!")
    
    # Parse back
    parsed = json.loads(json_str)
    print("[OK] JSON parsing successful!")
    
    # Show chart types
    print("\nCharts:")
    for chart in dashboard_data['charts']:
        print(f"  - {chart['title']} ({chart['type']})")
    
    print("\n" + "=" * 60)
    print("[SUCCESS] All JSON tests passed!")
    print("\nThe dashboard will now work without JSON errors.")
    
except Exception as e:
    print(f"[ERROR] {e}")
    import traceback
    traceback.print_exc()
