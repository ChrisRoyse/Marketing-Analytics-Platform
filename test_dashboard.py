#!/usr/bin/env python3
"""
Test Dashboard Components - Verification script for dashboard functionality.
Tests all components of the dashboard system to verify they're working correctly.
"""

import os
import sys
import json
from pathlib import Path
import importlib.util

def check_import(module_name):
    """Check if module can be imported and load it."""
    try:
        spec = importlib.util.find_spec(module_name)
        if spec is None:
            print(f"❌ {module_name} module not found")
            return False
        else:
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            print(f"✅ {module_name} module imported successfully")
            return True
    except ImportError:
        print(f"❌ {module_name} import failed")
        return False

def check_file_existence(file_path, description):
    """Check if file exists."""
    if os.path.exists(file_path):
        print(f"✅ {description} found at {file_path}")
        return True
    else:
        print(f"❌ {description} missing: {file_path}")
        return False

def check_directory_existence(dir_path, description):
    """Check if directory exists, create if missing."""
    if os.path.exists(dir_path) and os.path.isdir(dir_path):
        print(f"✅ {description} directory exists: {dir_path}")
        return True
    else:
        try:
            os.makedirs(dir_path, exist_ok=True)
            print(f"✅ {description} directory created: {dir_path}")
            return True
        except Exception as e:
            print(f"❌ {description} directory creation failed: {e}")
            return False

def load_json_file(file_path, description):
    """Try to load and validate JSON file."""
    if not os.path.exists(file_path):
        print(f"❌ {description} file missing: {file_path}")
        return False
    
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
        print(f"✅ {description} loaded successfully")
        return True
    except json.JSONDecodeError:
        print(f"❌ {description} is not valid JSON")
        return False
    except Exception as e:
        print(f"❌ {description} loading failed: {e}")
        return False

def check_dashboard_files():
    """Check if all required dashboard files exist."""
    success = True
    
    # Check main dashboard modules
    files_to_check = [
        ("enhanced_dashboard.py", "Enhanced Dashboard module"),
        ("executive_dashboard.py", "Executive Dashboard module"),
        ("operational_dashboard.py", "Operational Dashboard module"),
        ("advanced_visualization.py", "Advanced Visualization module"),
        ("run_dashboard.sh", "Dashboard shell script")
    ]
    
    for file_name, description in files_to_check:
        if not check_file_existence(file_name, description):
            success = False
    
    return success

def test_imports():
    """Test importing dashboard modules."""
    success = True
    
    # Required libraries
    libraries = ["dash", "plotly", "pandas", "numpy", "neo4j"]
    
    # Check plotly version compatibility
    try:
        import plotly
        plotly_version = plotly.__version__
        required_version = "5.18.0"
        if plotly_version != required_version:
            print(f"⚠️ Plotly version mismatch: found {plotly_version}, required {required_version}")
            print("This may cause compatibility issues with hover templates and labels")
        else:
            print(f"✅ Plotly version {plotly_version} matches required version")
    except:
        print("❌ Could not check Plotly version")
    
    print("\n=== Testing Required Libraries ===")
    for lib in libraries:
        if not check_import(lib):
            success = False
    
    # Dashboard components
    print("\n=== Testing Dashboard Components ===")
    dashboard_modules = [
        "enhanced_dashboard",
        "executive_dashboard", 
        "operational_dashboard",
        "advanced_visualization"
    ]
    
    for module in dashboard_modules:
        if not check_import(module):
            success = False
    
    return success

def test_visualization_generation():
    """Test visualization generation functionality."""
    print("\n=== Testing Visualization Generation ===")
    try:
        # Import the visualization module
        from advanced_visualization import AdvancedVisualization
        
        # Create the visualization object
        viz = AdvancedVisualization()
        
        # Check if the dashboard_data directory exists
        check_directory_existence("dashboard_data", "Dashboard data")
        
        # Create sample data for testing
        test_data = {
            "strategic_kpis": {
                "growth": {
                    "monthly_data": [
                        {"month": "2025-01", "monthly_revenue": 10000, "monthly_customers": 100, "monthly_purchases": 150},
                        {"month": "2025-02", "monthly_revenue": 12000, "monthly_customers": 120, "monthly_purchases": 180}
                    ],
                    "revenue_cagr": 12.5,
                    "customer_cagr": 8.5
                }
            }
        }
        
        # Create a test visualization
        print("Testing visualization creation...")
        test_chart = viz.create_interactive_revenue_chart(test_data["strategic_kpis"]["growth"])
        
        if test_chart:
            print("✅ Test visualization created successfully")
            return True
        else:
            print("❌ Test visualization creation failed")
            return False
            
    except Exception as e:
        print(f"❌ Visualization testing failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def run_dashboard_tests():
    """Run all dashboard tests."""
    print("=== Dashboard Verification Tests ===\n")
    
    tests = {
        "File existence": check_dashboard_files(),
        "Import tests": test_imports(),
        "Visualization tests": test_visualization_generation()
    }
    
    # Print summary
    print("\n=== Test Summary ===")
    all_pass = True
    for test_name, result in tests.items():
        status = "PASS" if result else "FAIL"
        if not result:
            all_pass = False
        print(f"{test_name}: {status}")
    
    if all_pass:
        print("\n✅ All dashboard tests passed!")
        print("You can now run ./run_dashboard.sh to start the dashboard")
        return 0
    else:
        print("\n❌ Some dashboard tests failed. Please fix the issues before running the dashboard.")
        return 1

if __name__ == "__main__":
    sys.exit(run_dashboard_tests())