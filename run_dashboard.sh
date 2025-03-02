#!/bin/bash

# Run Dashboard - Consolidated script to generate full dashboard data and visualizations
# This implements section 5 from the development plan

echo "===========================================" 
echo "   Marketing Ontology Platform Dashboard   "
echo "===========================================" 

# Check for virtual environment and activate it
if [ -d "dashboard_venv" ]; then
    echo "Using dashboard virtual environment..."
    source dashboard_venv/bin/activate
    PYTHON="python"
elif [ -d "marketing_venv" ]; then
    echo "Using marketing virtual environment..."
    source marketing_venv/bin/activate
    PYTHON="python"
else
    echo "No virtual environment found, using system Python..."
    PYTHON="/usr/bin/python3"
    
    # Check if we have the required dependencies
    if ! $PYTHON -c "import plotly" 2>/dev/null; then
        echo "Error: Plotly not installed. Please run:"
        echo "python -m venv dashboard_venv && source dashboard_venv/bin/activate && pip install -r requirements.txt"
        exit 1
    fi
fi

# Test dashboard components
echo "Testing dashboard components..."
$PYTHON test_dashboard.py
if [ $? -ne 0 ]; then
    echo "Warning: Dashboard tests failed. Continuing anyway..."
fi

# Create required directories
mkdir -p dashboard_data

# Generate Executive Dashboard data (section 5.1)
echo ""
echo "Generating Executive Dashboard data..."
$PYTHON executive_dashboard.py

# Generate Operational Dashboard data (section 5.2)
echo ""
echo "Generating Operational Dashboard data..."
$PYTHON operational_dashboard.py

# Generate Advanced Visualizations (section 5.3)
echo ""
echo "Generating Advanced Visualizations..."
$PYTHON advanced_visualization.py

# Start the Enhanced Dashboard Application
echo ""
echo "Starting Enhanced Dashboard..."
echo "Dashboard will be available at: http://localhost:8050"
$PYTHON enhanced_dashboard.py

echo ""
echo "Dashboard generation complete!"