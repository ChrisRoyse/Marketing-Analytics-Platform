# Marketing Ontology Platform Dashboard

## Overview

The Marketing Ontology Platform Dashboard is a comprehensive visualization and analytics interface that provides strategic insights, operational metrics, and advanced visualizations for marketing analytics. The dashboard implements section 5 of the development plan, providing:

1. **Executive Dashboard**: Strategic KPIs and benchmarking
2. **Operational Dashboard**: Marketing performance and channel analysis
3. **Advanced Visualizations**: Interactive charts with drill-down capabilities
4. **Customer Analysis**: Detailed customer journey and behavioral analysis
5. **Predictive Insights**: Future-looking metrics and ML model results

## Components

The dashboard system consists of several key components:

- **Enhanced Dashboard** (`enhanced_dashboard.py`): The main application that integrates all dashboard components
- **Executive Dashboard** (`executive_dashboard.py`): Strategic KPIs and benchmarking metrics
- **Operational Dashboard** (`operational_dashboard.py`): Campaign, channel, and customer service metrics
- **Advanced Visualization** (`advanced_visualization.py`): Interactive visualization components
- **Script Coordinator** (`run_dashboard.sh`): Shell script to run all components

## Dashboard Data Flow

1. Executive and Operational dashboard components connect to Neo4j
2. Dashboard services query the marketing ontology graph database
3. Metrics, KPIs, and raw data are processed and saved as JSON
4. Advanced Visualization component creates interactive charts
5. Enhanced Dashboard provides a web interface with all components

## How to Use

### One-Step Launch

For a complete dashboard deployment, use the provided shell script:

```bash
./run_dashboard.sh
```

This will:
1. Generate Executive Dashboard data
2. Generate Operational Dashboard data 
3. Create Advanced Visualizations
4. Start the Enhanced Dashboard application

### Individual Component Usage

You can also run each component separately:

```bash
# Generate Executive Dashboard data
python3 executive_dashboard.py

# Generate Operational Dashboard data
python3 operational_dashboard.py

# Create advanced visualizations
python3 advanced_visualization.py

# Start the dashboard web application
python3 enhanced_dashboard.py
```

### Accessing the Dashboard

Once started, the dashboard web application is available at:

- **URL**: http://localhost:8050
- **Port**: Configurable via PORT environment variable

## Key Features

### Executive Dashboard

- Revenue metrics with growth indicators
- Customer acquisition and retention metrics
- Industry benchmarking with competitive analysis
- Growth trends with compound annual growth rates

### Operational Dashboard

- Campaign performance metrics and ROI analysis
- Channel attribution modeling and comparison
- Customer service and support ticket analytics
- Marketing ROI and efficiency metrics

### Advanced Features

- Interactive drill-down capabilities
- Dynamic filtering and custom views
- Mobile-optimized visualizations
- Real-time updates and historical comparisons

## Technical Details

- **Framework**: Dash with Plotly
- **Database**: Neo4j graph database
- **Styling**: Bootstrap with responsive design
- **Data Format**: JSON for storage and exchange
- **Visualization**: Interactive Plotly charts

## Troubleshooting

### Connection Issues
If the dashboard fails to connect to Neo4j:
- Ensure Neo4j is running on the configured URI
- Check username and password in database settings
- Verify firewall settings if connecting to a remote database

### Visualization Issues
If visualizations don't appear:
- Ensure all dashboard data files exist in dashboard_data directory
- Check browser console for JavaScript errors
- Try clearing browser cache or using incognito mode

### Performance Issues
For slow dashboard performance:
- Increase server resources
- Optimize Neo4j queries
- Enable dashboard caching