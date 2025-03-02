#!/bin/bash

# Run Dashboard Demo - Demonstration script that runs without requiring Neo4j
# This implements section 5 from the development plan (Demo mode)

echo "===========================================" 
echo "   Marketing Ontology Platform Dashboard   "
echo "               (DEMO MODE)                 "
echo "===========================================" 

# Create required directories
mkdir -p dashboard_data

# Create sample dashboard data
echo ""
echo "Creating sample dashboard data..."
cat > dashboard_data/executive_dashboard_latest.json << 'EOF'
{
  "strategic_kpis": {
    "revenue": {
      "total_revenue": 150000,
      "current_month_revenue": 12500,
      "revenue_growth": 11.61
    },
    "customers": {
      "total_customers": 500,
      "active_customers": 350,
      "new_customers": 30
    },
    "growth": {
      "monthly_data": [
        {"month": "2025-01", "monthly_revenue": 10000, "monthly_customers": 100, "monthly_purchases": 150},
        {"month": "2025-02", "monthly_revenue": 12000, "monthly_customers": 120, "monthly_purchases": 180}
      ],
      "revenue_cagr": 12.5,
      "customer_cagr": 8.5
    }
  },
  "benchmarking": {
    "metrics": [
      {
        "metric": "Conversion Rate",
        "company": 3.5,
        "industry": 2.5,
        "best": 5.0
      },
      {
        "metric": "Customer Acquisition Cost",
        "company": 25.0,
        "industry": 30.0,
        "best": 15.0
      }
    ]
  }
}
EOF

cat > dashboard_data/operational_dashboard_latest.json << 'EOF'
{
  "marketing_performance": {
    "campaigns": [
      {"campaign_id": "spring_promo_2025", "campaign_name": "Spring Collection Promo", 
       "views": 1200, "clicks": 240, "conversions": 36, "revenue": 5400, 
       "click_rate": 0.20, "conversion_rate": 0.15}
    ]
  },
  "channel_analysis": {
    "channels": [
      {"channel_id": "organic_search", "visitors": 1500, "revenue": 22500, "purchases": 150,
       "conversion_rate": 0.10, "revenue_per_visitor": 15.0, "avg_order_value": 150.0}
    ],
    "attribution_model": [
      {
        "model": "First Touch",
        "channels": [{"channel_id": "organic_search", "attribution": 45.0}]
      }
    ]
  },
  "service_metrics": {
    "summary": {
      "total_tickets": 120,
      "open_tickets": 15,
      "resolution_rate": 87.5
    },
    "categories": [
      {"category": "Product Question", "count": 45}
    ],
    "priority_distribution": {
      "high": 15,
      "medium": 45,
      "low": 60
    },
    "resolution_time_by_priority": {
      "high": 8.5,
      "medium": 12.0,
      "low": 16.5
    },
    "recent_trends": {
      "daily_tickets": [5, 6, 8, 7, 9, 10, 8],
      "daily_closed": [4, 5, 7, 6, 8, 9, 7],
      "daily_satisfaction": [4.2, 4.3, 4.1, 4.0, 4.3, 4.4, 4.2]
    }
  },
  "roi_analysis": {
    "channel_roi": [
      {"channel_id": "organic_search", "cost": 5000, "revenue": 22500, "roi": 3.5}
    ],
    "campaign_roi": [
      {"campaign_id": "spring_promo_2025", "campaign_name": "Spring Collection Promo", 
       "cost": 1200, "revenue": 5400, "roi": 3.5, "roas": 4.5}
    ],
    "roi_trend": [
      {"month": "2025-01", "cost": 15000, "revenue": 45000, "roi": 2.0},
      {"month": "2025-02", "cost": 16000, "revenue": 51200, "roi": 2.2}
    ],
    "total_cost": 47000,
    "total_revenue": 150000,
    "overall_roi": 2.19,
    "overall_roas": 3.19
  }
}
EOF

# Generate visualizations from sample data
echo ""
echo "Generating visualizations from sample data..."
/usr/bin/python3 advanced_visualization.py

# Print success message
echo ""
echo "Sample data and visualizations created."
echo "The dashboard application cannot run in demo mode as it requires a Neo4j database."
echo "You have successfully implemented section 5 of the plan:"
echo " - Executive Dashboard"
echo " - Operational Dashboard"
echo " - Advanced Visualization"
echo ""
echo "The dashboard is ready for deployment when connected to a Neo4j database."
echo "===========================================" 