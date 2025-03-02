#!/usr/bin/env python3
"""
Enhanced Dashboard - Advanced visualization and analytics interface for the marketing ontology.
This dashboard implements Phase 5 enhancements with Executive, Operational, and Advanced visualizations.
"""

import os
import json
import dash
import pandas as pd
import numpy as np
import networkx as nx
from dash import dcc, html, callback, Input, Output, State
import dash_bootstrap_components as dbc
import plotly
import plotly.express as px
import plotly.graph_objects as go

# Check Plotly version compatibility
REQUIRED_PLOTLY_VERSION = "5.18.0"
if plotly.__version__ != REQUIRED_PLOTLY_VERSION:
    print(f"⚠️ Warning: This application was designed for Plotly {REQUIRED_PLOTLY_VERSION}")
    print(f"Current version: {plotly.__version__}")
    print("Some visualizations may not display correctly due to API changes")
else:
    print(f"✅ Using Plotly {plotly.__version__}")
from datetime import datetime, timedelta
from pathlib import Path
from flask import Flask
from neo4j import GraphDatabase
from dynamic_customer_analyzer import DynamicCustomerAnalyzer
from predictive_models import PredictiveModels
from enhanced_personalization import EnhancedPersonalization

# Initialize services
enhancer = EnhancedPersonalization()
analyzer = DynamicCustomerAnalyzer()
predictor = PredictiveModels()

# Initialize the Dash app with Bootstrap styling
server = Flask(__name__)
app = dash.Dash(
    __name__, 
    server=server,
    external_stylesheets=[dbc.themes.FLATLY],
    suppress_callback_exceptions=True,
    meta_tags=[
        {"name": "viewport", "content": "width=device-width, initial-scale=1.0"}
    ]
)

# Ensure directories exist
Path("customer_insights").mkdir(exist_ok=True)
Path("dashboard_data").mkdir(exist_ok=True)

class DashboardService:
    """Service for retrieving dashboard data from Neo4j"""
    
    def __init__(self, uri=None, username=None, password=None, database=None):
        """Initialize the DashboardService class with Neo4j connection details."""
        self.uri = uri or os.getenv('NEO4J_URI', 'bolt://localhost:7687')
        self.username = username or os.getenv('NEO4J_USERNAME', 'neo4j')
        self.password = password or os.getenv('NEO4J_PASSWORD', '#1Moneymaker')
        self.database = database or os.getenv('NEO4J_DATABASE', "marketing")
        self.driver = None
        
        # Log connection parameters (without password)
        print(f"Neo4j connection: {self.uri}, user: {self.username}, database: {self.database}")
        
    def connect(self):
        """Connect to the Neo4j database."""
        try:
            print(f"Attempting to connect to Neo4j at {self.uri}...")
            self.driver = GraphDatabase.driver(
                self.uri, 
                auth=(self.username, self.password)
            )
            # Test the connection
            with self.driver.session(database=self.database) as session:
                result = session.run("RETURN 1 AS test")
                record = result.single()
                if record and record["test"] == 1:
                    print("✅ Successfully connected to Neo4j database")
                    return True
                else:
                    print("❌ Neo4j connection test failed - record not found")
                    return False
        except Exception as e:
            print(f"❌ Failed to connect to Neo4j: {e}")
            print("Using sample data instead of live database data")
            return False
            
    def close(self):
        """Close the Neo4j connection."""
        if self.driver:
            self.driver.close()
            
    def run_query(self, query, parameters=None):
        """Run a Cypher query and return the results."""
        if not self.driver:
            if not self.connect():
                return None
                
        try:
            with self.driver.session(database=self.database) as session:
                result = session.run(query, parameters or {})
                return [record.data() for record in result]
        except Exception as e:
            print(f"Error running query: {e}")
            return None
    
    def get_revenue_metrics(self):
        """Get revenue metrics for executive dashboard."""
        query = """
        // Calculate total revenue
        MATCH (c:Customer)-[p:PURCHASES]->(pr:Product)
        WHERE p.amount IS NOT NULL
        
        WITH sum(p.amount) as total_revenue,
             count(DISTINCT c) as total_customers,
             count(p) as total_purchases
        
        // Calculate current month revenue
        MATCH (c:Customer)-[p:PURCHASES]->(pr:Product)
        WHERE p.amount IS NOT NULL AND 
              date(p.timestamp) >= date() - duration('P30D')
        WITH total_revenue, total_customers, total_purchases,
             sum(p.amount) as current_month_revenue,
             count(DISTINCT c) as current_month_customers,
             count(p) as current_month_purchases
             
        // Calculate previous month revenue
        MATCH (c:Customer)-[p:PURCHASES]->(pr:Product)
        WHERE p.amount IS NOT NULL AND 
              date(p.timestamp) >= date() - duration('P60D') AND
              date(p.timestamp) < date() - duration('P30D')
        
        RETURN total_revenue,
               total_customers,
               total_purchases,
               current_month_revenue,
               current_month_customers,
               current_month_purchases,
               sum(p.amount) as previous_month_revenue,
               count(DISTINCT c) as previous_month_customers,
               count(p) as previous_month_purchases
        """
        
        results = self.run_query(query)
        
        if not results:
            # Return sample data if query fails
            return {
                "total_revenue": 150000,
                "total_customers": 500,
                "total_purchases": 1200,
                "current_month_revenue": 12500,
                "current_month_customers": 120,
                "current_month_purchases": 180,
                "previous_month_revenue": 11200,
                "previous_month_customers": 105,
                "previous_month_purchases": 165,
                "revenue_growth": 11.61,
                "customer_growth": 14.29,
                "purchases_growth": 9.09,
                "avg_order_value": 69.44,
                "avg_revenue_per_customer": 104.17
            }
        
        # Extract data
        data = results[0]
        
        # Calculate derived metrics
        revenue_growth = ((data.get("current_month_revenue", 0) / max(1, data.get("previous_month_revenue", 1))) - 1) * 100
        customer_growth = ((data.get("current_month_customers", 0) / max(1, data.get("previous_month_customers", 1))) - 1) * 100
        purchases_growth = ((data.get("current_month_purchases", 0) / max(1, data.get("previous_month_purchases", 1))) - 1) * 100
        avg_order_value = data.get("current_month_revenue", 0) / max(1, data.get("current_month_purchases", 1))
        avg_revenue_per_customer = data.get("current_month_revenue", 0) / max(1, data.get("current_month_customers", 1))
        
        # Combine all metrics
        metrics = {
            "total_revenue": data.get("total_revenue", 0),
            "total_customers": data.get("total_customers", 0),
            "total_purchases": data.get("total_purchases", 0),
            "current_month_revenue": data.get("current_month_revenue", 0),
            "current_month_customers": data.get("current_month_customers", 0),
            "current_month_purchases": data.get("current_month_purchases", 0),
            "previous_month_revenue": data.get("previous_month_revenue", 0),
            "previous_month_customers": data.get("previous_month_customers", 0),
            "previous_month_purchases": data.get("previous_month_purchases", 0),
            "revenue_growth": revenue_growth,
            "customer_growth": customer_growth,
            "purchases_growth": purchases_growth,
            "avg_order_value": avg_order_value,
            "avg_revenue_per_customer": avg_revenue_per_customer
        }
        
        return metrics
    
    def get_customer_metrics(self):
        """Get customer metrics for executive dashboard."""
        query = """
        // Count total and active customers
        MATCH (c:Customer)
        WITH count(c) as total_customers
        
        MATCH (c:Customer)
        WHERE EXISTS {
            MATCH (c)-[p]->(n)
            WHERE p.timestamp IS NOT NULL AND
                  date(p.timestamp) >= date() - duration('P90D')
        }
        WITH total_customers, count(c) as active_customers
        
        // Churn metrics
        MATCH (c:Customer)-[:CHURNED_AT]->(:FunnelStage)
        WITH total_customers, active_customers, count(c) as churned_customers
        
        // New customers in last 30 days
        MATCH (c:Customer)
        WHERE EXISTS {
            MATCH (c)-[p:VIEWS|CLICKS_ON]->(:Advertisement)
            WHERE p.timestamp IS NOT NULL AND
                  date(p.timestamp) >= date() - duration('P30D')
        }
        AND NOT EXISTS {
            MATCH (c)-[p]->(n)
            WHERE p.timestamp IS NOT NULL AND
                  date(p.timestamp) < date() - duration('P30D')
        }
        
        RETURN total_customers,
               active_customers,
               churned_customers,
               count(c) as new_customers
        """
        
        results = self.run_query(query)
        
        if not results:
            # Return sample data if query fails
            return {
                "total_customers": 500,
                "active_customers": 350,
                "churned_customers": 45,
                "new_customers": 30,
                "churn_rate": 9.0,
                "activation_rate": 70.0,
                "new_customer_rate": 6.0
            }
        
        # Extract data
        data = results[0]
        
        # Calculate derived metrics
        total = data.get("total_customers", 100)
        churn_rate = (data.get("churned_customers", 0) / max(1, total)) * 100
        activation_rate = (data.get("active_customers", 0) / max(1, total)) * 100
        new_customer_rate = (data.get("new_customers", 0) / max(1, total)) * 100
        
        # Combine all metrics
        metrics = {
            "total_customers": total,
            "active_customers": data.get("active_customers", 0),
            "churned_customers": data.get("churned_customers", 0),
            "new_customers": data.get("new_customers", 0),
            "churn_rate": churn_rate,
            "activation_rate": activation_rate,
            "new_customer_rate": new_customer_rate
        }
        
        return metrics
    
    def get_marketing_performance(self):
        """Get marketing performance metrics for operational dashboard."""
        query = """
        // Email campaign performance
        MATCH (e:Email)<-[v:VIEWS]-(c:Customer)
        OPTIONAL MATCH (e)<-[cl:CLICKS_ON]-(c)
        WITH e.id as campaign_id, e.name as campaign_name, count(DISTINCT v) as views, count(DISTINCT cl) as clicks
        
        // Purchases attributed to email
        OPTIONAL MATCH (e:Email {id: campaign_id})<-[cl:CLICKS_ON]-(c:Customer)-[p:PURCHASES]->(pr:Product)
        WHERE p.timestamp IS NOT NULL AND
              cl.timestamp IS NOT NULL AND
              p.timestamp > cl.timestamp AND
              duration.inSeconds(datetime(cl.timestamp), datetime(p.timestamp)).seconds < 86400 // 24 hours
        
        WITH campaign_id, campaign_name, views, clicks, count(DISTINCT p) as conversions, sum(p.amount) as revenue
        
        RETURN campaign_id, 
               campaign_name, 
               views, 
               clicks, 
               conversions,
               revenue,
               CASE WHEN views > 0 THEN toFloat(clicks) / views ELSE 0 END as click_rate,
               CASE WHEN clicks > 0 THEN toFloat(conversions) / clicks ELSE 0 END as conversion_rate
        ORDER BY revenue DESC
        LIMIT 10
        """
        
        results = self.run_query(query)
        
        if not results:
            # Return sample data if query fails
            sample_campaigns = [
                {"campaign_id": "spring_promo_2025", "campaign_name": "Spring Collection Promo", 
                 "views": 1200, "clicks": 240, "conversions": 36, "revenue": 5400, 
                 "click_rate": 0.20, "conversion_rate": 0.15},
                {"campaign_id": "welcome_new_customers", "campaign_name": "New Customer Welcome", 
                 "views": 500, "clicks": 150, "conversions": 45, "revenue": 4050, 
                 "click_rate": 0.30, "conversion_rate": 0.30},
                {"campaign_id": "abandoned_cart_recovery", "campaign_name": "Cart Recovery", 
                 "views": 300, "clicks": 75, "conversions": 25, "revenue": 3750, 
                 "click_rate": 0.25, "conversion_rate": 0.33},
                {"campaign_id": "loyalty_program", "campaign_name": "Loyalty Program Announcement", 
                 "views": 800, "clicks": 160, "conversions": 24, "revenue": 2880, 
                 "click_rate": 0.20, "conversion_rate": 0.15},
                {"campaign_id": "summer_sale", "campaign_name": "Summer Sale Preview", 
                 "views": 950, "clicks": 190, "conversions": 19, "revenue": 1900, 
                 "click_rate": 0.20, "conversion_rate": 0.10}
            ]
            return sample_campaigns
        
        # Format and return the results
        return results
    
    def get_channel_analysis(self):
        """Get channel analysis data for operational dashboard."""
        query = """
        // Channel performance metrics
        MATCH (ch:Channel)<-[r:COMES_FROM]-(c:Customer)
        
        // Get purchases by channel
        OPTIONAL MATCH (c)-[p:PURCHASES]->(pr:Product)
        WHERE p.amount IS NOT NULL
        
        WITH ch.id as channel_id, count(DISTINCT c) as visitors, sum(p.amount) as revenue, count(p) as purchases
        
        RETURN channel_id, 
               visitors, 
               revenue, 
               purchases,
               CASE WHEN visitors > 0 THEN toFloat(purchases) / visitors ELSE 0 END as conversion_rate,
               CASE WHEN visitors > 0 THEN toFloat(revenue) / visitors ELSE 0 END as revenue_per_visitor,
               CASE WHEN purchases > 0 THEN toFloat(revenue) / purchases ELSE 0 END as avg_order_value
        ORDER BY revenue DESC
        """
        
        results = self.run_query(query)
        
        if not results:
            # Return sample data if query fails
            sample_channels = [
                {"channel_id": "organic_search", "visitors": 1500, "revenue": 22500, "purchases": 150,
                 "conversion_rate": 0.10, "revenue_per_visitor": 15.0, "avg_order_value": 150.0},
                {"channel_id": "paid_search", "visitors": 800, "revenue": 16000, "purchases": 80,
                 "conversion_rate": 0.10, "revenue_per_visitor": 20.0, "avg_order_value": 200.0},
                {"channel_id": "email", "visitors": 600, "revenue": 12000, "purchases": 100,
                 "conversion_rate": 0.17, "revenue_per_visitor": 20.0, "avg_order_value": 120.0},
                {"channel_id": "social_media", "visitors": 1200, "revenue": 9600, "purchases": 80,
                 "conversion_rate": 0.07, "revenue_per_visitor": 8.0, "avg_order_value": 120.0},
                {"channel_id": "direct", "visitors": 400, "revenue": 8000, "purchases": 40,
                 "conversion_rate": 0.10, "revenue_per_visitor": 20.0, "avg_order_value": 200.0},
                {"channel_id": "referral", "visitors": 300, "revenue": 7500, "purchases": 30,
                 "conversion_rate": 0.10, "revenue_per_visitor": 25.0, "avg_order_value": 250.0}
            ]
            return sample_channels
        
        # Format and return the results
        return results
    
    def get_service_metrics(self):
        """Get customer service metrics for operational dashboard."""
        query = """
        // Customer service metrics
        MATCH (t:Ticket)<-[:CREATES]-(c:Customer)
        WITH t.created_at as created_at, 
             t.resolved_at as resolved_at,
             t.status as status,
             t.priority as priority,
             t.satisfaction_score as satisfaction
        
        RETURN 
            count(t) as total_tickets,
            sum(CASE WHEN status = 'open' THEN 1 ELSE 0 END) as open_tickets,
            sum(CASE WHEN status = 'closed' THEN 1 ELSE 0 END) as closed_tickets,
            avg(CASE WHEN resolved_at IS NOT NULL AND created_at IS NOT NULL
                THEN duration.inSeconds(datetime(created_at), datetime(resolved_at)).seconds / 3600.0
                ELSE NULL END) as avg_resolution_hours,
            avg(satisfaction) as avg_satisfaction
        """
        
        results = self.run_query(query)
        
        if not results or not results[0].get("total_tickets"):
            # Return sample data if query fails or no data
            return {
                "total_tickets": 120,
                "open_tickets": 15,
                "closed_tickets": 105,
                "avg_resolution_hours": 12.5,
                "avg_satisfaction": 4.2,
                "resolution_rate": 87.5,
                "categories": [
                    {"category": "Product Question", "count": 45},
                    {"category": "Order Status", "count": 30},
                    {"category": "Return/Refund", "count": 25},
                    {"category": "Technical Issue", "count": 15},
                    {"category": "Other", "count": 5}
                ]
            }
        
        # Extract data
        data = results[0]
        
        # Add sample ticket categories (typically this would be a separate query)
        ticket_categories = [
            {"category": "Product Question", "count": int(data.get("total_tickets", 100) * 0.4)},
            {"category": "Order Status", "count": int(data.get("total_tickets", 100) * 0.25)},
            {"category": "Return/Refund", "count": int(data.get("total_tickets", 100) * 0.2)},
            {"category": "Technical Issue", "count": int(data.get("total_tickets", 100) * 0.1)},
            {"category": "Other", "count": int(data.get("total_tickets", 100) * 0.05)}
        ]
        
        # Calculate resolution rate
        total = data.get("total_tickets", 0)
        closed = data.get("closed_tickets", 0)
        resolution_rate = (closed / max(1, total)) * 100
        
        # Combine metrics
        metrics = {
            "total_tickets": total,
            "open_tickets": data.get("open_tickets", 0),
            "closed_tickets": closed,
            "avg_resolution_hours": data.get("avg_resolution_hours", 0),
            "avg_satisfaction": data.get("avg_satisfaction", 0),
            "resolution_rate": resolution_rate,
            "categories": ticket_categories
        }
        
        return metrics
    
    def get_growth_metrics(self):
        """Get growth metrics over time for the executive dashboard."""
        query = """
        // Monthly revenue and customer growth for the past 12 months
        UNWIND range(0, 11) as month_offset
        WITH date() - duration('P' + month_offset + 'M') as period_start,
             date() - duration('P' + (month_offset-1) + 'M') as period_end
        
        OPTIONAL MATCH (c:Customer)-[p:PURCHASES]->(pr:Product)
        WHERE p.amount IS NOT NULL AND
              date(p.timestamp) >= period_start AND
              date(p.timestamp) < period_end
        
        WITH period_start, period_end,
             sum(p.amount) as monthly_revenue,
             count(DISTINCT c) as monthly_customers,
             count(p) as monthly_purchases
             
        RETURN substring(toString(period_start), 0, 7) as month,
               monthly_revenue,
               monthly_customers,
               monthly_purchases
        ORDER BY month
        """
        
        results = self.run_query(query)
        
        if not results:
            # Return sample data if query fails
            months = []
            now = datetime.now()
            for i in range(12):
                month_date = now - timedelta(days=30 * i)
                months.append(month_date.strftime("%Y-%m"))
            
            months.reverse()  # Most recent last
            
            # Create sample growth data
            base_revenue = 8000
            base_customers = 80
            base_purchases = 100
            growth_factor = 1.05
            
            sample_data = []
            for i, month in enumerate(months):
                revenue = base_revenue * (growth_factor ** i)
                customers = base_customers * (growth_factor ** i)
                purchases = base_purchases * (growth_factor ** i)
                
                # Add some random variation
                revenue *= (0.95 + 0.1 * random.random())
                customers *= (0.95 + 0.1 * random.random())
                purchases *= (0.95 + 0.1 * random.random())
                
                sample_data.append({
                    "month": month,
                    "monthly_revenue": revenue,
                    "monthly_customers": int(customers),
                    "monthly_purchases": int(purchases)
                })
            
            return sample_data
        
        # Format and return results
        return results
    
    def get_benchmarking_data(self):
        """Get benchmarking data for executive dashboard."""
        # In a real implementation, this would query external APIs or internal reference data
        # For this example, we'll generate representative benchmark data
        
        # Compare current metrics to industry averages and best-in-class
        metrics = [
            "Conversion Rate", 
            "Customer Acquisition Cost", 
            "Customer Lifetime Value",
            "Average Order Value",
            "Cart Abandonment Rate",
            "Email Open Rate",
            "Email Click Rate"
        ]
        
        # Query for actual metrics (simplified version)
        query = """
        // Calculate key metrics for the company
        MATCH (c:Customer)
        OPTIONAL MATCH (c)-[p:PURCHASES]->(pr:Product)
        WHERE p.amount IS NOT NULL
        
        WITH count(DISTINCT c) as total_customers,
             count(p) as total_purchases,
             sum(p.amount) as total_revenue,
             sum(CASE WHEN EXISTS((c)-[:ABANDONS]->(:Cart)) THEN 1 ELSE 0 END) as abandoners
        
        // Email metrics
        MATCH (e:Email)<-[v:VIEWS]-(cu:Customer)
        OPTIONAL MATCH (e)<-[cl:CLICKS_ON]-(cu)
        
        WITH total_customers, total_purchases, total_revenue, abandoners,
             count(DISTINCT v) as email_views,
             count(DISTINCT cl) as email_clicks
        
        RETURN 
            CASE WHEN total_customers > 0 
                 THEN toFloat(total_purchases) / total_customers
                 ELSE 0
            END as conversion_rate,
            CASE WHEN total_purchases > 0 
                 THEN toFloat(total_revenue) / total_purchases
                 ELSE 0
            END as aov,
            CASE WHEN total_customers > 0
                 THEN toFloat(abandoners) / total_customers
                 ELSE 0
            END as cart_abandonment_rate,
            CASE WHEN email_views > 0
                 THEN toFloat(email_clicks) / email_views
                 ELSE 0
            END as email_click_rate
        """
        
        results = self.run_query(query)
        
        company_values = {}
        if results:
            data = results[0]
            company_values = {
                "Conversion Rate": data.get("conversion_rate", 0.10) * 100,
                "Average Order Value": data.get("aov", 100),
                "Cart Abandonment Rate": data.get("cart_abandonment_rate", 0.70) * 100,
                "Email Click Rate": data.get("email_click_rate", 0.05) * 100,
                "Customer Acquisition Cost": 25,  # Sample value
                "Customer Lifetime Value": 120,   # Sample value
                "Email Open Rate": 25,            # Sample value
            }
        else:
            # Sample company data
            company_values = {
                "Conversion Rate": 3.5,
                "Customer Acquisition Cost": 25,
                "Customer Lifetime Value": 120,
                "Average Order Value": 85,
                "Cart Abandonment Rate": 68,
                "Email Open Rate": 22,
                "Email Click Rate": 2.8
            }
        
        # Industry average benchmarks
        industry_values = {
            "Conversion Rate": 2.5,
            "Customer Acquisition Cost": 30,
            "Customer Lifetime Value": 100,
            "Average Order Value": 75,
            "Cart Abandonment Rate": 70,
            "Email Open Rate": 20,
            "Email Click Rate": 2.5
        }
        
        # Best-in-class benchmarks
        best_values = {
            "Conversion Rate": 5.0,
            "Customer Acquisition Cost": 15,
            "Customer Lifetime Value": 200,
            "Average Order Value": 110,
            "Cart Abandonment Rate": 55,
            "Email Open Rate": 35,
            "Email Click Rate": 4.0
        }
        
        # Format the benchmark data for visualization
        benchmark_data = []
        for metric in metrics:
            benchmark_data.append({
                "metric": metric,
                "company": company_values.get(metric, 0),
                "industry": industry_values.get(metric, 0),
                "best": best_values.get(metric, 0)
            })
        
        return benchmark_data

# Create dashboard service instance
dashboard_service = DashboardService()

# App layout
app.layout = dbc.Container([
    dbc.Row([
        dbc.Col(
            html.Div([
                html.H1("Marketing Ontology Platform", className="display-4 text-center my-4"),
                html.H4("Advanced Analytics & Insights Dashboard", className="text-center text-muted mb-4")
            ]),
            width=12
        )
    ]),
    
    # Main navigation
    dbc.Row([
        dbc.Col(
            dbc.Tabs(
                [
                    dbc.Tab(label="Executive KPIs", tab_id="tab-executive", 
                           labelClassName="font-weight-bold"),
                    dbc.Tab(label="Operational Metrics", tab_id="tab-operational",
                           labelClassName="font-weight-bold"),
                    dbc.Tab(label="Customer Analysis", tab_id="tab-customer",
                           labelClassName="font-weight-bold"),
                    dbc.Tab(label="Predictive Insights", tab_id="tab-predictive",
                           labelClassName="font-weight-bold"),
                    dbc.Tab(label="Context & Personalization", tab_id="tab-context",
                           labelClassName="font-weight-bold"),
                ],
                id="main-tabs",
                active_tab="tab-executive",
                className="mb-3"
            ),
            width=12
        )
    ]),
    
    # Tab content container
    html.Div(id="tab-content"),
    
    # Hidden stores for data
    dcc.Store(id="revenue-metrics-store"),
    dcc.Store(id="customer-metrics-store"),
    dcc.Store(id="growth-metrics-store"),
    dcc.Store(id="benchmark-data-store"),
    dcc.Store(id="marketing-performance-store"),
    dcc.Store(id="channel-analysis-store"),
    dcc.Store(id="service-metrics-store"),
    dcc.Store(id="selected-customer-data"),
    
    # Footer
    dbc.Row([
        dbc.Col(
            html.Footer([
                html.P("Marketing Behavior Pattern Ontology - Enhanced Dashboard", className="text-center mb-0"),
                html.P("© 2025 Your Company", className="text-center small text-muted")
            ], className="py-3 mt-5"),
            width=12
        )
    ])
], fluid=True)

# Callback to switch between main tabs
@app.callback(
    Output("tab-content", "children"),
    [Input("main-tabs", "active_tab")]
)
def render_tab_content(active_tab):
    """Render content based on active tab."""
    if active_tab == "tab-executive":
        return executive_dashboard_layout()
    elif active_tab == "tab-operational":
        return operational_dashboard_layout()
    elif active_tab == "tab-customer":
        return customer_analysis_layout()
    elif active_tab == "tab-predictive":
        return predictive_insights_layout()
    elif active_tab == "tab-context":
        return context_personalization_layout()
    return html.P("No content available")

# ==== Executive Dashboard Layout ====
def executive_dashboard_layout():
    """Create the executive dashboard layout."""
    return html.Div([
        # First row: Key Performance Indicators
        dbc.Row([
            dbc.Col(
                dbc.Card(
                    dbc.CardBody([
                        html.H5("Revenue Metrics", className="card-title"),
                        html.Div(id="revenue-kpi-cards", className="d-flex flex-wrap")
                    ]),
                    className="mb-4"
                ),
                width=12
            )
        ]),
        
        # Second row: Customer and Growth Metrics
        dbc.Row([
            dbc.Col(
                dbc.Card(
                    dbc.CardBody([
                        html.H5("Customer Metrics", className="card-title"),
                        html.Div(id="customer-kpi-cards", className="d-flex flex-wrap")
                    ]),
                    className="mb-4"
                ),
                width=6
            ),
            dbc.Col(
                dbc.Card(
                    dbc.CardBody([
                        html.H5("Revenue Growth", className="card-title"),
                        dcc.Graph(id="revenue-growth-chart", style={"height": "300px"})
                    ]),
                    className="mb-4"
                ),
                width=6
            )
        ]),
        
        # Third row: Benchmarking 
        dbc.Row([
            dbc.Col(
                dbc.Card(
                    dbc.CardBody([
                        html.H5("Industry Benchmarking", className="card-title"),
                        dcc.Graph(id="benchmark-radar-chart", style={"height": "400px"})
                    ]),
                    className="mb-4"
                ),
                width=6
            ),
            dbc.Col(
                dbc.Card(
                    dbc.CardBody([
                        html.H5("Benchmark Comparison", className="card-title"),
                        dcc.Graph(id="benchmark-comparison-chart", style={"height": "400px"})
                    ]),
                    className="mb-4"
                ),
                width=6
            )
        ]),
        
        # Data loading triggers
        html.Div([
            dcc.Interval(
                id="executive-data-interval",
                interval=5*1000,  # in milliseconds - refresh every 5 seconds initially
                n_intervals=0,
                max_intervals=1  # Only fire once for initial load
            )
        ])
    ])

# ==== Operational Dashboard Layout ====
def operational_dashboard_layout():
    """Create the operational dashboard layout."""
    return html.Div([
        # First row: Marketing Performance
        dbc.Row([
            dbc.Col(
                dbc.Card(
                    dbc.CardBody([
                        html.H5("Campaign Performance", className="card-title"),
                        dcc.Graph(id="campaign-performance-chart", style={"height": "350px"})
                    ]),
                    className="mb-4"
                ),
                width=8
            ),
            dbc.Col(
                dbc.Card(
                    dbc.CardBody([
                        html.H5("Email Campaign Metrics", className="card-title"),
                        html.Div(id="campaign-metrics-table")
                    ]),
                    className="mb-4"
                ),
                width=4
            )
        ]),
        
        # Second row: Channel Analysis
        dbc.Row([
            dbc.Col(
                dbc.Card(
                    dbc.CardBody([
                        html.H5("Channel Performance", className="card-title"),
                        dcc.Graph(id="channel-performance-chart", style={"height": "350px"})
                    ]),
                    className="mb-4"
                ),
                width=6
            ),
            dbc.Col(
                dbc.Card(
                    dbc.CardBody([
                        html.H5("Attribution Model", className="card-title"),
                        dcc.Graph(id="attribution-model-chart", style={"height": "350px"})
                    ]),
                    className="mb-4"
                ),
                width=6
            )
        ]),
        
        # Third row: Customer Service
        dbc.Row([
            dbc.Col(
                dbc.Card(
                    dbc.CardBody([
                        html.H5("Customer Service Metrics", className="card-title"),
                        html.Div(id="service-metric-cards", className="d-flex flex-wrap")
                    ]),
                    className="mb-4"
                ),
                width=6
            ),
            dbc.Col(
                dbc.Card(
                    dbc.CardBody([
                        html.H5("Support Ticket Categories", className="card-title"),
                        dcc.Graph(id="ticket-categories-chart", style={"height": "300px"})
                    ]),
                    className="mb-4"
                ),
                width=6
            )
        ]),
        
        # Data loading triggers
        html.Div([
            dcc.Interval(
                id="operational-data-interval",
                interval=5*1000,  # in milliseconds
                n_intervals=0,
                max_intervals=1  # Only fire once
            )
        ])
    ])

# ==== Customer Analysis Layout ====
def customer_analysis_layout():
    """Create the customer analysis layout."""
    return html.Div([
        # Customer selection
        dbc.Row([
            dbc.Col(
                dbc.Card(
                    dbc.CardBody([
                        html.H5("Customer Analysis", className="card-title"),
                        html.P("Enter a customer ID to analyze their journey and predictions", className="card-text"),
                        dbc.InputGroup([
                            dbc.Input(id="customer-id-input", placeholder="Enter Customer ID...", type="text"),
                            dbc.Button("Analyze", id="analyze-button", color="primary")
                        ]),
                        html.Div(id="customer-lookup-status", className="mt-2")
                    ]),
                    className="mb-4"
                ),
                width=12
            )
        ]),
        
        # Customer tabs - only visible when a customer is selected
        html.Div(
            dbc.Tabs([
                dbc.Tab(
                    dbc.Card(dbc.CardBody([
                        html.Div(id="customer-profile-content")
                    ])),
                    label="Profile", tab_id="tab-profile"
                ),
                dbc.Tab(
                    dbc.Card(dbc.CardBody([
                        html.Div(id="customer-journey-content")
                    ])),
                    label="Journey", tab_id="tab-journey"
                ),
                dbc.Tab(
                    dbc.Card(dbc.CardBody([
                        html.Div(id="customer-funnel-content")
                    ])),
                    label="Funnel", tab_id="tab-funnel"
                ),
                dbc.Tab(
                    dbc.Card(dbc.CardBody([
                        html.Div(id="customer-predictions-content")
                    ])),
                    label="Predictions", tab_id="tab-predictions"
                ),
                dbc.Tab(
                    dbc.Card(dbc.CardBody([
                        html.Div(id="customer-recommendations-content")
                    ])),
                    label="Recommendations", tab_id="tab-recommendations"
                ),
                dbc.Tab(
                    dbc.Card(dbc.CardBody([
                        html.Div(id="customer-feedback-content")
                    ])),
                    label="Feedback", tab_id="tab-feedback"
                )
            ], id="customer-tabs", active_tab="tab-profile"),
            id="customer-tabs-container", 
            style={"display": "none"}
        )
    ])

# ==== Predictive Insights Layout ====
def predictive_insights_layout():
    """Create the predictive insights layout."""
    return html.Div([
        # First row: Model training and batch processing
        dbc.Row([
            dbc.Col(
                dbc.Card(
                    dbc.CardBody([
                        html.H5("Predictive Models", className="card-title"),
                        html.P("Train models and run batch predictions", className="card-text"),
                        dbc.Button("Train All Models", id="train-models-button", color="primary", className="me-2"),
                        dbc.Button("Process All Customers", id="batch-process-button", color="secondary"),
                        html.Div(id="model-training-status", className="mt-3")
                    ]),
                    className="mb-4"
                ),
                width=12
            )
        ]),
        
        # Second row: Model performance
        dbc.Row([
            dbc.Col(
                dbc.Card(
                    dbc.CardBody([
                        html.H5("Churn Model Performance", className="card-title"),
                        dcc.Graph(id="churn-model-performance", style={"height": "300px"})
                    ]),
                    className="mb-4"
                ),
                width=6
            ),
            dbc.Col(
                dbc.Card(
                    dbc.CardBody([
                        html.H5("CLV Model Performance", className="card-title"),
                        dcc.Graph(id="clv-model-performance", style={"height": "300px"})
                    ]),
                    className="mb-4"
                ),
                width=6
            )
        ]),
        
        # Third row: Feature importance
        dbc.Row([
            dbc.Col(
                dbc.Card(
                    dbc.CardBody([
                        html.H5("Feature Importance", className="card-title"),
                        dcc.Graph(id="feature-importance-chart", style={"height": "400px"})
                    ]),
                    className="mb-4"
                ),
                width=6
            ),
            dbc.Col(
                dbc.Card(
                    dbc.CardBody([
                        html.H5("Anomaly Detection", className="card-title"),
                        dcc.Graph(id="anomaly-detection-chart", style={"height": "400px"})
                    ]),
                    className="mb-4"
                ),
                width=6
            )
        ]),
        
        # Data stores
        dcc.Store(id="model-metadata-store")
    ])

# ==== Context & Personalization Layout ====
def context_personalization_layout():
    """Create the context and personalization layout."""
    return html.Div([
        # First row: Customer selection & context generation
        dbc.Row([
            dbc.Col(
                dbc.Card(
                    dbc.CardBody([
                        html.H5("Context-Aware Personalization", className="card-title"),
                        html.P("Generate context-aware recommendations for a customer", className="card-text"),
                        dbc.InputGroup([
                            dbc.Input(id="context-customer-id", placeholder="Enter Customer ID...", type="text"),
                            dbc.Button("Generate", id="generate-context-button", color="primary")
                        ]),
                        html.Div(id="context-generation-status", className="mt-2")
                    ]),
                    className="mb-4"
                ),
                width=12
            )
        ]),
        
        # Second row: Context visualization
        dbc.Row([
            dbc.Col(
                dbc.Card(
                    dbc.CardBody([
                        html.H5("Customer Context", className="card-title"),
                        html.Div(id="context-visualization")
                    ]),
                    className="mb-4"
                ),
                width=12
            )
        ]),
        
        # Third row: Context-aware recommendations
        dbc.Row([
            dbc.Col(
                dbc.Card(
                    dbc.CardBody([
                        html.H5("Context-Aware Recommendations", className="card-title"),
                        html.Div(id="context-recommendations")
                    ]),
                    className="mb-4"
                ),
                width=12
            )
        ]),
        
        # Fourth row: Feedback analysis
        dbc.Row([
            dbc.Col(
                dbc.Card(
                    dbc.CardBody([
                        html.H5("Customer Feedback Analysis", className="card-title"),
                        html.Div(id="feedback-analysis")
                    ]),
                    className="mb-4"
                ),
                width=12
            )
        ]),
        
        # Data stores
        dcc.Store(id="context-data-store"),
        dcc.Store(id="recommendations-store"),
        dcc.Store(id="feedback-data-store")
    ])

# ==== Executive Dashboard Callbacks ====
@app.callback(
    [
        Output("revenue-metrics-store", "data"),
        Output("customer-metrics-store", "data"),
        Output("growth-metrics-store", "data"),
        Output("benchmark-data-store", "data")
    ],
    [Input("executive-data-interval", "n_intervals")]
)
def load_executive_data(n_intervals):
    """Load all data needed for the executive dashboard."""
    revenue_metrics = dashboard_service.get_revenue_metrics()
    customer_metrics = dashboard_service.get_customer_metrics()
    growth_metrics = dashboard_service.get_growth_metrics()
    benchmark_data = dashboard_service.get_benchmarking_data()
    
    return revenue_metrics, customer_metrics, growth_metrics, benchmark_data

@app.callback(
    Output("revenue-kpi-cards", "children"),
    [Input("revenue-metrics-store", "data")]
)
def update_revenue_kpi_cards(data):
    """Update revenue KPI cards with the latest data."""
    if not data:
        return html.P("No revenue data available", className="text-muted")
    
    # Format the data for display
    current_revenue = data.get("current_month_revenue", 0)
    revenue_growth = data.get("revenue_growth", 0)
    avg_order = data.get("avg_order_value", 0)
    avg_revenue_per_customer = data.get("avg_revenue_per_customer", 0)
    
    # Create KPI cards
    cards = [
        dbc.Card(
            dbc.CardBody([
                html.H6("Monthly Revenue", className="card-subtitle text-muted"),
                html.H3(f"${current_revenue:,.2f}", className="mt-2"),
                html.P([
                    html.Span(f"{revenue_growth:+.1f}% vs prev month", 
                             className=f"text-{'success' if revenue_growth >= 0 else 'danger'}")
                ], className="mt-2 mb-0")
            ]),
            className="m-2", style={"width": "12rem"}
        ),
        dbc.Card(
            dbc.CardBody([
                html.H6("Avg Order Value", className="card-subtitle text-muted"),
                html.H3(f"${avg_order:,.2f}", className="mt-2"),
                html.P("Per purchase", className="text-muted mt-2 mb-0")
            ]),
            className="m-2", style={"width": "12rem"}
        ),
        dbc.Card(
            dbc.CardBody([
                html.H6("Revenue per Customer", className="card-subtitle text-muted"),
                html.H3(f"${avg_revenue_per_customer:,.2f}", className="mt-2"),
                html.P("Monthly average", className="text-muted mt-2 mb-0")
            ]),
            className="m-2", style={"width": "12rem"}
        ),
        dbc.Card(
            dbc.CardBody([
                html.H6("Total Revenue", className="card-subtitle text-muted"),
                html.H3(f"${data.get('total_revenue', 0):,.2f}", className="mt-2"),
                html.P(f"Lifetime", className="text-muted mt-2 mb-0")
            ]),
            className="m-2", style={"width": "12rem"}
        )
    ]
    
    return cards

@app.callback(
    Output("customer-kpi-cards", "children"),
    [Input("customer-metrics-store", "data")]
)
def update_customer_kpi_cards(data):
    """Update customer KPI cards with the latest data."""
    if not data:
        return html.P("No customer data available", className="text-muted")
    
    # Format the data for display
    total_customers = data.get("total_customers", 0)
    active_customers = data.get("active_customers", 0)
    new_customers = data.get("new_customers", 0)
    churn_rate = data.get("churn_rate", 0)
    
    # Create KPI cards
    cards = [
        dbc.Card(
            dbc.CardBody([
                html.H6("Active Customers", className="card-subtitle text-muted"),
                html.H3(f"{active_customers:,}", className="mt-2"),
                html.P([
                    html.Span(f"{data.get('activation_rate', 0):.1f}% of total", 
                              className="text-muted")
                ], className="mt-2 mb-0")
            ]),
            className="m-2", style={"width": "10rem"}
        ),
        dbc.Card(
            dbc.CardBody([
                html.H6("New Customers", className="card-subtitle text-muted"),
                html.H3(f"{new_customers:,}", className="mt-2"),
                html.P("Last 30 days", className="text-muted mt-2 mb-0")
            ]),
            className="m-2", style={"width": "10rem"}
        ),
        dbc.Card(
            dbc.CardBody([
                html.H6("Churn Rate", className="card-subtitle text-muted"),
                html.H3(f"{churn_rate:.1f}%", className="mt-2"),
                html.P([
                    html.Span(f"{data.get('churned_customers', 0):,} customers", 
                              className="text-muted")
                ], className="mt-2 mb-0")
            ]),
            className="m-2", style={"width": "10rem"}
        )
    ]
    
    return cards

@app.callback(
    Output("revenue-growth-chart", "figure"),
    [Input("growth-metrics-store", "data")]
)
def update_revenue_growth_chart(data):
    """Update revenue growth chart with the latest data."""
    if not data:
        return blank_figure("No growth data available")
    
    # Create dataframe from the metrics
    df = pd.DataFrame(data)
    
    # Create figure with secondary y-axis
    fig = go.Figure()
    
    # Add revenue bars
    fig.add_trace(go.Bar(
        x=df["month"],
        y=df["monthly_revenue"],
        name="Revenue",
        marker_color="#3498db"
    ))
    
    # Add customer line on secondary axis
    fig.add_trace(go.Scatter(
        x=df["month"],
        y=df["monthly_customers"],
        name="Customers",
        marker_color="#e74c3c",
        mode="lines+markers",
        yaxis="y2"
    ))
    
    # Set layout with two y-axes
    fig.update_layout(
        title="Revenue and Customer Growth",
        xaxis_title="Month",
        yaxis_title="Revenue ($)",
        yaxis2=dict(
            title="Customers",
            overlaying="y",
            side="right"
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        height=300,
        margin=dict(l=40, r=40, t=40, b=40),
        template="plotly_white"
    )
    
    return fig

@app.callback(
    Output("benchmark-radar-chart", "figure"),
    [Input("benchmark-data-store", "data")]
)
def update_benchmark_radar_chart(data):
    """Update benchmark radar chart with the latest data."""
    if not data:
        return blank_figure("No benchmark data available")
    
    # Create dataframe from the benchmark data
    metrics = [item["metric"] for item in data]
    
    # For radar chart, we need to normalize the values to make them comparable
    # Create a function for simple min-max normalization
    def normalize_values(values, min_val, max_val):
        return [(v - min_val) / (max_val - min_val) if max_val > min_val else 0.5 for v in values]
    
    # Extract and normalize values
    # For metrics where lower is better (like CAC), we invert the normalization
    normalized_company = []
    normalized_industry = []
    normalized_best = []
    
    for item in data:
        metric = item["metric"]
        company_val = item["company"]
        industry_val = item["industry"]
        best_val = item["best"]
        
        # Determine if lower values are better for this metric
        lower_is_better = metric in ["Customer Acquisition Cost", "Cart Abandonment Rate"]
        
        # Get min and max
        if lower_is_better:
            min_val = min(company_val, industry_val, best_val)
            max_val = max(company_val, industry_val, best_val)
            # Invert normalization for metrics where lower is better
            normalized_company.append(1 - ((company_val - min_val) / (max_val - min_val)) if max_val > min_val else 0.5)
            normalized_industry.append(1 - ((industry_val - min_val) / (max_val - min_val)) if max_val > min_val else 0.5)
            normalized_best.append(1 - ((best_val - min_val) / (max_val - min_val)) if max_val > min_val else 0.5)
        else:
            min_val = min(company_val, industry_val, best_val)
            max_val = max(company_val, industry_val, best_val)
            normalized_company.append((company_val - min_val) / (max_val - min_val) if max_val > min_val else 0.5)
            normalized_industry.append((industry_val - min_val) / (max_val - min_val) if max_val > min_val else 0.5)
            normalized_best.append((best_val - min_val) / (max_val - min_val) if max_val > min_val else 0.5)
    
    # Create radar chart
    fig = go.Figure()
    
    # Add radar chart traces
    fig.add_trace(go.Scatterpolar(
        r=normalized_company,
        theta=metrics,
        fill='toself',
        name='Your Company',
        line_color='#3498db'
    ))
    
    fig.add_trace(go.Scatterpolar(
        r=normalized_industry,
        theta=metrics,
        fill='toself',
        name='Industry Average',
        line_color='#95a5a6'
    ))
    
    fig.add_trace(go.Scatterpolar(
        r=normalized_best,
        theta=metrics,
        fill='toself',
        name='Best in Class',
        line_color='#2ecc71'
    ))
    
    # Update layout
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 1]
            )
        ),
        showlegend=True,
        height=400,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        margin=dict(l=60, r=60, t=40, b=40)
    )
    
    return fig

@app.callback(
    Output("benchmark-comparison-chart", "figure"),
    [Input("benchmark-data-store", "data")]
)
def update_benchmark_comparison_chart(data):
    """Update benchmark comparison chart with the latest data."""
    if not data:
        return blank_figure("No benchmark data available")
    
    # Create dataframe for grouped bar chart
    df = pd.DataFrame(data)
    
    # Focus on just a few key metrics to avoid cluttering the chart
    key_metrics = ["Conversion Rate", "Average Order Value", "Customer Lifetime Value", "Email Open Rate"]
    df_filtered = df[df["metric"].isin(key_metrics)]
    
    # Create grouped bar chart
    fig = go.Figure()
    
    # Add traces for each category
    for category in ["company", "industry", "best"]:
        name_map = {"company": "Your Company", "industry": "Industry Average", "best": "Best in Class"}
        color_map = {"company": "#3498db", "industry": "#95a5a6", "best": "#2ecc71"}
        
        fig.add_trace(go.Bar(
            x=df_filtered["metric"],
            y=df_filtered[category],
            name=name_map[category],
            marker_color=color_map[category]
        ))
    
    # Update layout
    fig.update_layout(
        title="Key Metric Comparison",
        xaxis_title="Metric",
        yaxis_title="Value",
        barmode="group",
        height=400,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        template="plotly_white",
        margin=dict(l=40, r=40, t=40, b=40)
    )
    
    return fig

# ==== Operational Dashboard Callbacks ====
@app.callback(
    [
        Output("marketing-performance-store", "data"),
        Output("channel-analysis-store", "data"),
        Output("service-metrics-store", "data")
    ],
    [Input("operational-data-interval", "n_intervals")]
)
def load_operational_data(n_intervals):
    """Load all data needed for the operational dashboard."""
    marketing_performance = dashboard_service.get_marketing_performance()
    channel_analysis = dashboard_service.get_channel_analysis()
    service_metrics = dashboard_service.get_service_metrics()
    
    return marketing_performance, channel_analysis, service_metrics

@app.callback(
    Output("campaign-performance-chart", "figure"),
    [Input("marketing-performance-store", "data")]
)
def update_campaign_performance_chart(data):
    """Update campaign performance chart with the latest data."""
    if not data:
        return blank_figure("No campaign performance data available")
    
    # Create dataframe
    df = pd.DataFrame(data)
    if len(df) > 5:
        df = df.head(5)  # Limit to top 5 for readability
    
    # Create figure with secondary y-axis
    fig = go.Figure()
    
    # Add revenue bars
    fig.add_trace(go.Bar(
        x=df["campaign_name"],
        y=df["revenue"],
        name="Revenue",
        marker_color="#3498db",
        text=df["revenue"].apply(lambda x: f"${x:,.2f}"),
        textposition="auto"
    ))
    
    # Add conversion rate line on secondary axis
    fig.add_trace(go.Scatter(
        x=df["campaign_name"],
        y=df["conversion_rate"] * 100,  # Convert to percentage
        name="Conversion Rate",
        marker_color="#e74c3c",
        mode="lines+markers",
        yaxis="y2",
        text=df["conversion_rate"].apply(lambda x: f"{x*100:.1f}%"),
        textposition="top center"
    ))
    
    # Update layout
    fig.update_layout(
        title="Top Campaign Performance",
        xaxis_title="Campaign",
        yaxis_title="Revenue ($)",
        yaxis2=dict(
            title="Conversion Rate (%)",
            overlaying="y",
            side="right"
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        height=350,
        template="plotly_white",
        margin=dict(l=40, r=40, t=40, b=40)
    )
    
    return fig

@app.callback(
    Output("campaign-metrics-table", "children"),
    [Input("marketing-performance-store", "data")]
)
def update_campaign_metrics_table(data):
    """Update campaign metrics table with the latest data."""
    if not data:
        return html.P("No campaign metrics available", className="text-muted")
    
    # Create table rows
    rows = []
    for i, campaign in enumerate(data):
        if i >= 5:  # Limit to top 5 for readability
            break
            
        rows.append(
            html.Tr([
                html.Td(campaign.get("campaign_name", "")),
                html.Td(f"{campaign.get('click_rate', 0)*100:.1f}%"),
                html.Td(f"{campaign.get('conversion_rate', 0)*100:.1f}%")
            ])
        )
    
    # Create table
    table = dbc.Table(
        [
            html.Thead(
                html.Tr([
                    html.Th("Campaign"),
                    html.Th("CTR"),
                    html.Th("CVR")
                ])
            ),
            html.Tbody(rows)
        ],
        bordered=True,
        striped=True,
        size="sm",
        hover=True
    )
    
    return table

@app.callback(
    Output("channel-performance-chart", "figure"),
    [Input("channel-analysis-store", "data")]
)
def update_channel_performance_chart(data):
    """Update channel performance chart with the latest data."""
    if not data:
        return blank_figure("No channel performance data available")
    
    # Create dataframe
    df = pd.DataFrame(data)
    
    # Create bubble chart
    fig = go.Figure()
    
    # Add bubble trace
    fig.add_trace(go.Scatter(
        x=df["conversion_rate"] * 100,  # Convert to percentage
        y=df["revenue_per_visitor"],
        size=df["visitors"],
        text=df["channel_id"],
        mode="markers",
        marker=dict(
            sizemode="area",
            sizeref=max(df["visitors"]) / 1000,
            color=df["avg_order_value"],
            colorscale="Viridis",
            showscale=True,
            colorbar=dict(title="Avg Order Value")
        ),
        hovertemplate="<b>%{text}</b><br>Conversion: %{x:.1f}%<br>Revenue/Visitor: $%{y:.2f}<br>Visitors: %{marker.size:,}<br>AOV: $%{marker.color:.2f}"
    ))
    
    # Update layout
    fig.update_layout(
        title="Channel Performance Comparison",
        xaxis_title="Conversion Rate (%)",
        yaxis_title="Revenue per Visitor ($)",
        height=350,
        template="plotly_white",
        margin=dict(l=40, r=40, t=40, b=40)
    )
    
    return fig

@app.callback(
    Output("attribution-model-chart", "figure"),
    [Input("channel-analysis-store", "data")]
)
def update_attribution_model_chart(data):
    """Update attribution model chart with the latest data."""
    if not data:
        return blank_figure("No attribution data available")
    
    # Create a sample attribution model (in a real implementation this would be a separate query)
    # First touch, last touch, linear, and time decay attribution models
    
    # Extract channels from data
    channels = [item["channel_id"] for item in data]
    
    # Generate sample attribution percentages
    attribution_models = {
        "First Touch": [],
        "Last Touch": [],
        "Linear": [],
        "Time Decay": []
    }
    
    for channel in data:
        # Calculate sample attribution based on conversion rate and visitors
        channel_strength = channel["conversion_rate"] * channel["visitors"]
        total_strength = sum([c["conversion_rate"] * c["visitors"] for c in data])
        base_attribution = channel_strength / max(total_strength, 0.001)
        
        # Adjust for different models
        attribution_models["First Touch"].append(base_attribution * (1 + np.random.uniform(-0.3, 0.3)))
        attribution_models["Last Touch"].append(base_attribution * (1 + np.random.uniform(-0.3, 0.3)))
        attribution_models["Linear"].append(base_attribution)
        attribution_models["Time Decay"].append(base_attribution * (1 + np.random.uniform(-0.2, 0.2)))
    
    # Normalize to ensure they sum to 1
    for model in attribution_models:
        total = sum(attribution_models[model])
        attribution_models[model] = [v / total for v in attribution_models[model]]
    
    # Create stacked bar chart
    fig = go.Figure()
    
    # Add bars for each channel
    for i, channel in enumerate(channels):
        for model in attribution_models:
            fig.add_trace(go.Bar(
                x=[model],
                y=[attribution_models[model][i] * 100],  # Convert to percentage
                name=channel,
                text=f"{attribution_models[model][i] * 100:.1f}%",
                textposition="inside",
                hovertemplate=f"{channel}: %{{y:.1f}}%"
            ))
    
    # Update layout
    fig.update_layout(
        title="Attribution Model Comparison",
        xaxis_title="Attribution Model",
        yaxis_title="Attribution (%)",
        barmode="stack",
        height=350,
        template="plotly_white",
        margin=dict(l=40, r=40, t=40, b=40)
    )
    
    return fig

@app.callback(
    Output("service-metric-cards", "children"),
    [Input("service-metrics-store", "data")]
)
def update_service_metric_cards(data):
    """Update service metric cards with the latest data."""
    if not data:
        return html.P("No service metrics available", className="text-muted")
    
    # Format the data for display
    total_tickets = data.get("total_tickets", 0)
    open_tickets = data.get("open_tickets", 0)
    resolution_rate = data.get("resolution_rate", 0)
    avg_resolution_hours = data.get("avg_resolution_hours", 0)
    avg_satisfaction = data.get("avg_satisfaction", 0)
    
    # Create KPI cards
    cards = [
        dbc.Card(
            dbc.CardBody([
                html.H6("Resolution Rate", className="card-subtitle text-muted"),
                html.H3(f"{resolution_rate:.1f}%", className="mt-2"),
                html.P([
                    html.Span(f"{data.get('closed_tickets', 0)} of {total_tickets}", 
                              className="text-muted")
                ], className="mt-2 mb-0")
            ]),
            className="m-2", style={"width": "10rem"}
        ),
        dbc.Card(
            dbc.CardBody([
                html.H6("Avg Resolution Time", className="card-subtitle text-muted"),
                html.H3(f"{avg_resolution_hours:.1f}h", className="mt-2"),
                html.P(f"{open_tickets} open tickets", className="text-muted mt-2 mb-0")
            ]),
            className="m-2", style={"width": "10rem"}
        ),
        dbc.Card(
            dbc.CardBody([
                html.H6("Satisfaction Score", className="card-subtitle text-muted"),
                html.H3(f"{avg_satisfaction:.1f}/5", className="mt-2"),
                html.P("Customer rating", className="text-muted mt-2 mb-0")
            ]),
            className="m-2", style={"width": "10rem"}
        )
    ]
    
    return cards

@app.callback(
    Output("ticket-categories-chart", "figure"),
    [Input("service-metrics-store", "data")]
)
def update_ticket_categories_chart(data):
    """Update ticket categories chart with the latest data."""
    if not data or "categories" not in data:
        return blank_figure("No ticket category data available")
    
    # Extract categories and counts
    categories = [item["category"] for item in data["categories"]]
    counts = [item["count"] for item in data["categories"]]
    
    # Create pie chart
    fig = go.Figure(data=[go.Pie(
        labels=categories,
        values=counts,
        hole=0.4,
        textinfo="label+percent",
        insidetextorientation="radial"
    )])
    
    # Update layout
    fig.update_layout(
        title="Support Ticket Categories",
        height=300,
        template="plotly_white",
        margin=dict(l=20, r=20, t=40, b=20)
    )
    
    return fig

# ==== Helper Functions ====
def blank_figure(message="No data available"):
    """Create a blank figure with a message."""
    fig = go.Figure()
    fig.add_annotation(
        text=message,
        xref="paper",
        yref="paper",
        x=0.5,
        y=0.5,
        showarrow=False,
        font=dict(size=16, color="#95a5a6")
    )
    fig.update_layout(
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        paper_bgcolor="rgba(0,0,0,0)"
    )
    return fig

# ==== Run the server ====
def main():
    """Run the Dash app."""
    port = int(os.environ.get("PORT", 8050))
    app.run_server(debug=True, host="0.0.0.0", port=port)

if __name__ == "__main__":
    print("Starting Enhanced Marketing Dashboard...")
    main()