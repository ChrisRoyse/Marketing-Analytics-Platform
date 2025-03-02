#!/usr/bin/env python3
"""
Predictive Dashboard - Enhanced web interface for the marketing ontology.
This dashboard extends the Phase 3 interface with predictive analytics features.
"""

import os
import json
import dash
import base64
import pandas as pd
import numpy as np
import networkx as nx
from dash import dcc, html, callback, Input, Output, State
import dash_bootstrap_components as dbc
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
from pathlib import Path
from flask import Flask
from dynamic_customer_analyzer import DynamicCustomerAnalyzer
from predictive_models import PredictiveModels

# Initialize the Dash app with Bootstrap styling
server = Flask(__name__)
app = dash.Dash(
    __name__, 
    server=server,
    external_stylesheets=[dbc.themes.FLATLY],
    suppress_callback_exceptions=True
)

# Initialize the analyzer and predictor
analyzer = DynamicCustomerAnalyzer()
predictor = PredictiveModels()

# Ensure customer insights directory exists
Path("customer_insights").mkdir(exist_ok=True)

# App layout
app.layout = dbc.Container([
    dbc.Row(
        dbc.Col(
            html.H1("Marketing Ontology - Predictive Customer Analytics", className="text-center my-4"),
            width=12
        )
    ),
    
    dbc.Row([
        dbc.Col(
            dbc.Card(
                dbc.CardBody([
                    html.H4("Customer Lookup", className="card-title"),
                    html.P("Enter a customer ID to analyze their journey and predictions", className="card-text"),
                    dbc.Input(id="customer-id-input", placeholder="Enter Customer ID...", type="text"),
                    dbc.Button("Analyze", id="analyze-button", color="primary", className="mt-3"),
                    html.Div(id="loading-output", className="mt-3"),
                ]),
                className="mb-4"
            ),
            width=4
        ),
        
        dbc.Col(
            dbc.Card(
                dbc.CardBody([
                    html.H4("Model Actions", className="card-title"),
                    html.P("Run predictive models and batch processing", className="card-text"),
                    dbc.Button("Train All Models", id="train-models-button", color="success", className="mt-2 me-2"),
                    dbc.Button("Process All Customers", id="batch-button", color="secondary", className="mt-2"),
                    dcc.Store(id="batch-results"),
                    html.Div(id="batch-output", className="mt-3"),
                ]),
                className="mb-4"
            ),
            width=4
        ),
        
        dbc.Col(
            dbc.Card(
                dbc.CardBody([
                    html.H4("Prediction Summary", className="card-title"),
                    html.Div(id="prediction-stats"),
                ]),
                className="mb-4"
            ),
            width=4
        ),
    ]),
    
    dbc.Tabs([
        dbc.Tab(
            dbc.Card(dbc.CardBody([
                html.H4("Customer Profile", className="card-title"),
                html.Div(id="customer-profile"),
            ])),
            label="Profile", tab_id="tab-profile"
        ),
        dbc.Tab(
            dbc.Card(dbc.CardBody([
                html.H4("Journey Visualization", className="card-title"),
                html.Div(id="journey-visualization"),
            ])),
            label="Journey", tab_id="tab-journey"
        ),
        dbc.Tab(
            dbc.Card(dbc.CardBody([
                html.H4("Funnel Analysis", className="card-title"),
                html.Div(id="funnel-analysis"),
            ])),
            label="Funnel", tab_id="tab-funnel"
        ),
        dbc.Tab(
            dbc.Card(dbc.CardBody([
                html.H4("Predictive Insights", className="card-title"),
                html.Div(id="predictive-insights"),
            ])),
            label="Predictions", tab_id="tab-predictions"
        ),
        dbc.Tab(
            dbc.Card(dbc.CardBody([
                html.H4("Recommendations", className="card-title"),
                html.Div(id="recommendations"),
            ])),
            label="Recommendations", tab_id="tab-recommendations"
        ),
        dbc.Tab(
            dbc.Card(dbc.CardBody([
                html.H4("Churn Risk Analysis", className="card-title"),
                html.Div(id="churn-analysis"),
            ])),
            label="Churn Risk", tab_id="tab-churn"
        ),
        dbc.Tab(
            dbc.Card(dbc.CardBody([
                html.H4("Anomaly Detection", className="card-title"),
                html.Div(id="anomaly-detection"),
            ])),
            label="Anomalies", tab_id="tab-anomalies"
        ),
        dbc.Tab(
            dbc.Card(dbc.CardBody([
                html.H4("Raw Data", className="card-title"),
                dcc.Dropdown(id="raw-data-selector", className="mb-3"),
                html.Div(id="raw-data-output"),
            ])),
            label="Raw Data", tab_id="tab-raw"
        ),
        dbc.Tab(
            dbc.Card(dbc.CardBody([
                html.H4("Model Performance", className="card-title"),
                html.Div(id="model-performance"),
            ])),
            label="Models", tab_id="tab-models"
        ),
    ], id="customer-tabs", active_tab="tab-profile"),
    
    # Store the current customer data
    dcc.Store(id="customer-data"),
    dcc.Store(id="prediction-data"),
    dcc.Store(id="model-metadata"),
    
    # Footer
    dbc.Row(
        dbc.Col(
            html.Footer([
                html.P("Marketing Behavior Pattern Ontology - Predictive Dashboard", className="mb-1"),
                html.P("© 2025 Your Company", className="small text-muted")
            ], className="text-center my-4"),
            width=12
        )
    )
], fluid=True)

@app.callback(
    [
        Output("loading-output", "children"),
        Output("customer-data", "data"),
        Output("prediction-data", "data")
    ],
    [Input("analyze-button", "n_clicks")],
    [State("customer-id-input", "value")],
    prevent_initial_call=True
)
def analyze_customer(n_clicks, customer_id):
    """Analyze a specific customer when the button is clicked."""
    if not n_clicks or not customer_id:
        return "Please enter a customer ID", None, None
    
    if not analyzer.connect():
        return "Failed to connect to database. Check configuration.", None, None
    
    if not analyzer.validate_customer_id(customer_id):
        return f"Customer ID '{customer_id}' not found in database", None, None
    
    try:
        # Generate the customer report
        report = analyzer.create_customer_report(customer_id)
        
        if "error" in report:
            return f"Error: {report['error']}", None, None
        
        # Get predictive insights
        if not predictor.connect():
            return "Analysis complete but failed to get predictions.", report, None
        
        insights = predictor.predict_customer_insights(customer_id)
        predictor.close()
        
        if not insights:
            return "Analysis complete but no predictions available.", report, None
        
        return "Analysis with predictions complete! View the tabs below for details.", report, insights
    
    except Exception as e:
        return f"Error during analysis: {str(e)}", None, None
    
    finally:
        analyzer.close()

@app.callback(
    Output("batch-output", "children"),
    [Input("batch-button", "n_clicks")],
    prevent_initial_call=True
)
def batch_process(n_clicks):
    """Process all customers in the database."""
    if not n_clicks:
        return ""
    
    try:
        if analyzer.run(customer_id=None):
            reports_path = Path("customer_insights")
            report_count = len(list(reports_path.glob("customer_*_report.json")))
            
            # Also update all customer predictions
            predictor.connect()
            customer_query = """
            MATCH (c:Customer)
            RETURN c.customer_id as customer_id
            """
            customers = predictor.run_query(customer_query)
            
            prediction_count = 0
            if customers:
                for customer in customers:
                    customer_id = customer["customer_id"]
                    insights = predictor.predict_customer_insights(customer_id)
                    if insights:
                        prediction_count += 1
            
            predictor.close()
            
            return html.Div([
                html.P(f"Successfully processed {report_count} customers", className="text-success"),
                html.P(f"Generated predictions for {prediction_count} customers", className="text-success"),
                html.P(f"Reports saved to {reports_path}/")
            ])
        else:
            return html.P("Batch processing failed. Check logs for details.", className="text-danger")
    
    except Exception as e:
        return html.P(f"Error during batch processing: {str(e)}", className="text-danger")

@app.callback(
    Output("model-metadata", "data"),
    [Input("train-models-button", "n_clicks")],
    prevent_initial_call=True
)
def train_models(n_clicks):
    """Train all prediction models."""
    if not n_clicks:
        return None
    
    try:
        # Train models and get results
        results = predictor.run_phase4_modeling()
        
        # Get model metadata from Neo4j
        model_query = """
        MATCH (m:PredictiveModel)
        RETURN m.name as model_name, 
               m.metrics as metrics,
               m.feature_importances as feature_importances,
               m.last_updated as last_updated
        """
        predictor.connect()
        model_data = predictor.run_query(model_query)
        predictor.close()
        
        if not model_data:
            return {"status": "error", "message": "No model metadata found"}
        
        # Process model data
        formatted_data = {}
        for model in model_data:
            model_name = model.get("model_name")
            metrics_json = model.get("metrics", "{}")
            feature_importances_json = model.get("feature_importances", "{}")
            
            try:
                metrics = json.loads(metrics_json)
                feature_importances = json.loads(feature_importances_json)
                
                formatted_data[model_name] = {
                    "metrics": metrics,
                    "feature_importances": feature_importances,
                    "last_updated": model.get("last_updated", "")
                }
            except json.JSONDecodeError:
                continue
        
        formatted_data["training_results"] = results
        return formatted_data
    
    except Exception as e:
        return {"status": "error", "message": str(e)}

@app.callback(
    Output("prediction-stats", "children"),
    [
        Input("customer-data", "data"),
        Input("prediction-data", "data")
    ]
)
def update_prediction_stats(customer_data, prediction_data):
    """Update prediction summary stats."""
    if not customer_data:
        return html.P("No customer selected", className="text-muted")
    
    if not prediction_data:
        return html.P("No prediction data available", className="text-muted")
    
    # Extract key metrics
    customer_id = customer_data.get("customer_id", "Unknown")
    churn_probability = prediction_data.get("churn_probability", 0)
    churn_risk_level = prediction_data.get("churn_risk_level", "Unknown")
    predicted_clv = prediction_data.get("predicted_lifetime_value", 0)
    days_until_purchase = prediction_data.get("days_until_next_purchase", 0)
    
    # Create stats cards
    stats = [
        dbc.ListGroupItem([
            html.Span("Churn Risk: ", className="fw-bold"),
            html.Span(f"{churn_probability:.1%}", 
                      className={
                          "text-danger": churn_probability > 0.7,
                          "text-warning": 0.3 < churn_probability <= 0.7,
                          "text-success": churn_probability <= 0.3
                      })
        ]),
        dbc.ListGroupItem([
            html.Span("Risk Level: ", className="fw-bold"),
            html.Span(churn_risk_level, 
                      className={
                          "High": "text-danger",
                          "Medium": "text-warning",
                          "Low": "text-success"
                      }.get(churn_risk_level, ""))
        ]),
        dbc.ListGroupItem([
            html.Span("Predicted CLV: ", className="fw-bold"),
            html.Span(f"${predicted_clv:.2f}")
        ]),
        dbc.ListGroupItem([
            html.Span("Next Purchase: ", className="fw-bold"),
            html.Span(f"~{days_until_purchase} days" if days_until_purchase else "Unknown")
        ])
    ]
    
    return dbc.ListGroup(stats)

@app.callback(
    Output("customer-profile", "children"),
    [Input("customer-data", "data")]
)
def update_customer_profile(data):
    """Update customer profile tab with customer data."""
    if not data:
        return html.P("No customer selected", className="text-muted")
    
    profile = data.get("profile", {})
    
    # Extract profile information
    basic_info = profile.get("basic_info", {})
    segments = profile.get("segments", [])
    personas = profile.get("personas", [])
    devices = profile.get("devices", [])
    browsers = profile.get("browsers", [])
    locations = profile.get("locations", [])
    
    # Create profile cards
    profile_sections = []
    
    # Basic info card
    basic_info_items = []
    for key, value in basic_info.items():
        if key == "id":
            continue
        basic_info_items.append(html.Li(f"{key}: {value}"))
    
    profile_sections.append(
        dbc.Card(
            dbc.CardBody([
                html.H5("Basic Information", className="card-title"),
                html.Ul(basic_info_items)
            ]),
            className="mb-3"
        )
    )
    
    # Segments and Personas
    segment_items = [html.Li(segment.get("id", "Unknown")) for segment in segments]
    persona_items = [html.Li(persona.get("id", "Unknown")) for persona in personas]
    
    profile_sections.append(
        dbc.Row([
            dbc.Col(
                dbc.Card(
                    dbc.CardBody([
                        html.H5("Customer Segments", className="card-title"),
                        html.Ul(segment_items if segment_items else html.Li("No segments"))
                    ])
                ),
                width=6
            ),
            dbc.Col(
                dbc.Card(
                    dbc.CardBody([
                        html.H5("Customer Personas", className="card-title"),
                        html.Ul(persona_items if persona_items else html.Li("No personas"))
                    ])
                ),
                width=6
            )
        ]),
        className="mb-3"
    )
    
    # Technical profile
    device_items = [html.Li(device.get("id", "Unknown")) for device in devices]
    browser_items = [html.Li(browser.get("id", "Unknown")) for browser in browsers]
    location_items = [html.Li(location.get("id", "Unknown")) for location in locations]
    
    profile_sections.append(
        dbc.Row([
            dbc.Col(
                dbc.Card(
                    dbc.CardBody([
                        html.H5("Devices", className="card-title"),
                        html.Ul(device_items if device_items else html.Li("No devices"))
                    ])
                ),
                width=4
            ),
            dbc.Col(
                dbc.Card(
                    dbc.CardBody([
                        html.H5("Browsers", className="card-title"),
                        html.Ul(browser_items if browser_items else html.Li("No browsers"))
                    ])
                ),
                width=4
            ),
            dbc.Col(
                dbc.Card(
                    dbc.CardBody([
                        html.H5("Locations", className="card-title"),
                        html.Ul(location_items if location_items else html.Li("No locations"))
                    ])
                ),
                width=4
            )
        ]),
        className="mb-3"
    )
    
    return html.Div(profile_sections)

@app.callback(
    Output("journey-visualization", "children"),
    [Input("customer-data", "data")]
)
def update_journey_visualization(data):
    """Update journey visualization tab with customer journey data."""
    if not data:
        return html.P("No customer selected", className="text-muted")
    
    journey = data.get("journey", {})
    timeline = journey.get("timeline", [])
    
    if not timeline:
        return html.P("No journey data available for this customer", className="text-muted")
    
    # Create journey timeline visualization
    # First, prepare data for plotting
    df = pd.DataFrame(timeline)
    if df.empty:
        return html.P("No journey events to display", className="text-muted")
    
    # Ensure timestamp is a datetime
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.sort_values('timestamp')
    
    # Create timeline figure
    fig = go.Figure()
    
    # Define event type colors
    event_colors = {
        'VIEWS': '#3498db',
        'CLICKS_ON': '#2ecc71',
        'VISITS': '#9b59b6',
        'ADDS_TO_CART': '#e74c3c',
        'PURCHASES': '#f1c40f',
        'COMMENTS_ON': '#1abc9c',
        'REFERS': '#34495e'
    }
    
    # Create scatter plot for timeline
    for event_type in df['event_type'].unique():
        event_df = df[df['event_type'] == event_type]
        
        fig.add_trace(go.Scatter(
            x=event_df['timestamp'],
            y=[event_type] * len(event_df),
            mode='markers+text',
            marker=dict(
                size=12,
                color=event_colors.get(event_type, '#95a5a6'),
                symbol='circle'
            ),
            text=event_df['description'],
            hoverinfo='text',
            hovertext=[
                f"Action: {row['description']}<br>"
                f"Time: {row['timestamp']}<br>"
                f"Target: {row['target_type']} ({row['target_id']})" 
                for _, row in event_df.iterrows()
            ],
            name=event_type
        ))
    
    # Layout
    fig.update_layout(
        title="Customer Journey Timeline",
        xaxis_title="Time",
        yaxis_title="Interaction Type",
        height=500,
        hovermode='closest',
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    
    # Add visualization graph
    journey_sections = [
        dcc.Graph(figure=fig),
        html.H5("Journey Events", className="mt-4"),
        generate_journey_table(timeline)
    ]
    
    return html.Div(journey_sections)

def generate_journey_table(timeline):
    """Generate a table from journey timeline data."""
    if not timeline:
        return html.P("No journey events", className="text-muted")
    
    # Create table rows
    rows = []
    for event in sorted(timeline, key=lambda x: x.get('timestamp', ''), reverse=True):
        rows.append(
            html.Tr([
                html.Td(event.get('timestamp', '')),
                html.Td(event.get('event_type', '')),
                html.Td(event.get('description', '')),
                html.Td(f"{event.get('target_type', '')} ({event.get('target_id', '')})")
            ])
        )
    
    # Create table
    table = dbc.Table(
        [
            html.Thead(
                html.Tr([
                    html.Th("Timestamp"),
                    html.Th("Event Type"),
                    html.Th("Description"),
                    html.Th("Target")
                ])
            ),
            html.Tbody(rows)
        ],
        bordered=True,
        hover=True,
        responsive=True,
        striped=True
    )
    
    return table

@app.callback(
    Output("funnel-analysis", "children"),
    [Input("customer-data", "data")]
)
def update_funnel_analysis(data):
    """Update funnel analysis tab with customer funnel data."""
    if not data:
        return html.P("No customer selected", className="text-muted")
    
    journey = data.get("journey", {})
    funnel_status = journey.get("funnel_status", {})
    
    if not funnel_status:
        return html.P("No funnel data available for this customer", className="text-muted")
    
    # Extract funnel data
    current_stage = funnel_status.get("current_stage", "Unknown")
    completed_stages = funnel_status.get("completed_stages", [])
    all_stages = funnel_status.get("all_stages", [])
    has_churned = funnel_status.get("has_churned", False)
    churn_details = funnel_status.get("churn_details", {})
    
    # Create funnel visualization
    funnel_values = []
    for stage in all_stages:
        if stage in completed_stages:
            funnel_values.append(1)
        else:
            funnel_values.append(0)
    
    fig = go.Figure(go.Funnel(
        y=all_stages,
        x=funnel_values,
        textposition="inside",
        textinfo="value+percent initial",
        marker={"color": [
            "#27ae60" if stage in completed_stages else "#ecf0f1" 
            for stage in all_stages
        ]}
    ))
    
    fig.update_layout(
        title="Customer Funnel Progress",
        height=400
    )
    
    # Create funnel status sections
    funnel_sections = [
        dbc.Card(
            dbc.CardBody([
                html.H5("Funnel Status"),
                html.P([
                    html.Span("Current Stage: ", className="fw-bold"),
                    html.Span(current_stage)
                ]),
                html.P([
                    html.Span("Churned: ", className="fw-bold"),
                    html.Span(
                        "Yes" if has_churned else "No",
                        className="text-danger" if has_churned else "text-success"
                    )
                ])
            ]),
            className="mb-3"
        ),
        dcc.Graph(figure=fig)
    ]
    
    # Add churn details if available
    if has_churned and churn_details:
        churn_time = churn_details.get("churn_time", "")
        churn_reason = churn_details.get("churn_reason", "Unknown")
        previous_stage = churn_details.get("previous_stage", "Unknown")
        
        funnel_sections.append(
            dbc.Card(
                dbc.CardBody([
                    html.H5("Churn Details"),
                    html.P([
                        html.Span("Churn Time: ", className="fw-bold"),
                        html.Span(churn_time)
                    ]),
                    html.P([
                        html.Span("Churn Reason: ", className="fw-bold"),
                        html.Span(churn_reason)
                    ]),
                    html.P([
                        html.Span("Previous Stage: ", className="fw-bold"),
                        html.Span(previous_stage)
                    ])
                ]),
                className="mt-3 border-danger"
            )
        )
    
    return html.Div(funnel_sections)

@app.callback(
    Output("predictive-insights", "children"),
    [
        Input("customer-data", "data"),
        Input("prediction-data", "data")
    ]
)
def update_predictive_insights(customer_data, prediction_data):
    """Update predictive insights tab with customer predictions."""
    if not customer_data:
        return html.P("No customer selected", className="text-muted")
    
    if not prediction_data:
        return html.P("No prediction data available for this customer", className="text-muted")
    
    # Extract prediction data
    customer_id = customer_data.get("customer_id", "Unknown")
    current_clv = prediction_data.get("current_lifetime_value", 0)
    predicted_clv = prediction_data.get("predicted_lifetime_value", 0)
    clv_growth = prediction_data.get("lifetime_value_growth", 0)
    churn_probability = prediction_data.get("churn_probability", 0)
    churn_risk_level = prediction_data.get("churn_risk_level", "Unknown")
    days_until_next_purchase = prediction_data.get("days_until_next_purchase", 0)
    predicted_next_purchase_date = prediction_data.get("predicted_next_purchase_date", "")
    
    if predicted_next_purchase_date:
        try:
            predicted_date = datetime.fromisoformat(predicted_next_purchase_date).strftime("%B %d, %Y")
        except:
            predicted_date = "Unknown"
    else:
        predicted_date = "Unknown"
    
    # Create visualizations and cards
    
    # 1. Churn probability gauge
    churn_gauge = go.Figure(go.Indicator(
        mode="gauge+number",
        value=churn_probability * 100,
        title={"text": "Churn Probability"},
        gauge={
            "axis": {"range": [0, 100], "tickwidth": 1, "tickcolor": "darkblue"},
            "bar": {"color": "darkblue"},
            "bgcolor": "white",
            "borderwidth": 2,
            "bordercolor": "gray",
            "steps": [
                {"range": [0, 30], "color": "green"},
                {"range": [30, 70], "color": "yellow"},
                {"range": [70, 100], "color": "red"}
            ],
            "threshold": {
                "line": {"color": "red", "width": 4},
                "thickness": 0.75,
                "value": churn_probability * 100
            }
        }
    ))
    
    churn_gauge.update_layout(
        height=250, 
        margin=dict(l=20, r=20, t=50, b=20)
    )
    
    # 2. CLV prediction bar chart
    clv_fig = go.Figure()
    
    clv_fig.add_trace(go.Bar(
        x=["Current CLV", "Predicted CLV"],
        y=[current_clv, predicted_clv],
        marker_color=["#3498db", "#2ecc71"],
        text=[f"${current_clv:.2f}", f"${predicted_clv:.2f}"],
        textposition="auto",
    ))
    
    clv_fig.update_layout(
        title="Customer Lifetime Value Projection",
        height=300,
        margin=dict(l=20, r=20, t=50, b=20)
    )
    
    # 3. Next purchase countdown
    days_color = "success" if days_until_next_purchase < 7 else "warning" if days_until_next_purchase < 30 else "secondary"
    
    # Create insight cards
    insight_sections = [
        dbc.Row([
            dbc.Col(
                dbc.Card(
                    dbc.CardBody([
                        html.H5("Churn Risk Assessment", className="card-title"),
                        dcc.Graph(figure=churn_gauge),
                        html.P([
                            html.Span("Risk Level: ", className="fw-bold"),
                            html.Span(churn_risk_level, className={"High": "text-danger", "Medium": "text-warning", "Low": "text-success"}.get(churn_risk_level, ""))
                        ], className="mt-2")
                    ])
                ),
                width=6
            ),
            dbc.Col(
                dbc.Card(
                    dbc.CardBody([
                        html.H5("Next Purchase Prediction", className="card-title"),
                        html.Div([
                            html.H1(days_until_next_purchase, className=f"text-{days_color} text-center display-1 mt-3"),
                            html.P("days until predicted next purchase", className="text-center")
                        ]),
                        html.P([
                            html.Span("Estimated date: ", className="fw-bold"),
                            html.Span(predicted_date)
                        ], className="mt-3")
                    ])
                ),
                width=6
            ),
        ], className="mb-3"),
        
        dbc.Card(
            dbc.CardBody([
                html.H5("Customer Lifetime Value Prediction"),
                dcc.Graph(figure=clv_fig),
                html.P([
                    html.Span("Projected CLV Growth: ", className="fw-bold"),
                    html.Span(
                        f"${clv_growth:.2f} ({(clv_growth/current_clv*100):.1f}%)" if current_clv > 0 else f"${clv_growth:.2f}",
                        className="text-success" if clv_growth > 0 else "text-danger"
                    )
                ], className="mt-2")
            ]),
            className="mb-3"
        ),
        
        dbc.Card(
            dbc.CardBody([
                html.H5("Prediction Details"),
                html.P("These predictions are generated using machine learning models trained on your customer data. The models analyze behavioral patterns, purchase history, and engagement levels to forecast future customer actions."),
                html.P(f"Last updated: {datetime.fromisoformat(prediction_data.get('timestamp', datetime.now().isoformat())).strftime('%Y-%m-%d %H:%M')}")
            ])
        )
    ]
    
    return html.Div(insight_sections)

@app.callback(
    Output("recommendations", "children"),
    [
        Input("customer-data", "data"),
        Input("prediction-data", "data")
    ]
)
def update_recommendations(customer_data, prediction_data):
    """Update recommendations tab with customer recommendations."""
    if not customer_data:
        return html.P("No customer selected", className="text-muted")
    
    insights = customer_data.get("insights", {})
    product_recommendations = insights.get("product_recommendations", [])
    similar_customers = insights.get("similar_customers", [])
    actions = customer_data.get("actions", [])
    
    # Add prediction-based recommendations
    if prediction_data:
        churn_probability = prediction_data.get("churn_probability", 0)
        days_until_next_purchase = prediction_data.get("days_until_next_purchase", 0)
        clv_growth = prediction_data.get("lifetime_value_growth", 0)
        
        # Create additional recommendations based on predictions
        prediction_actions = []
        
        if churn_probability > 0.7:
            prediction_actions.append({
                "action_type": "High Churn Risk",
                "priority": "High",
                "description": "Immediate retention campaign",
                "details": "Customer has a high probability of churning. Send a personalized retention offer with a significant discount or exclusive benefit."
            })
        elif churn_probability > 0.3:
            prediction_actions.append({
                "action_type": "Medium Churn Risk",
                "priority": "Medium",
                "description": "Proactive engagement",
                "details": "Customer shows moderate churn risk. Increase engagement through targeted content and offers related to their interests."
            })
        
        if days_until_next_purchase <= 7:
            prediction_actions.append({
                "action_type": "Purchase Opportunity",
                "priority": "High",
                "description": "Prepare for upcoming purchase",
                "details": f"Customer is likely to make a purchase in the next {days_until_next_purchase} days. Ensure inventory is available and consider personalized recommendations."
            })
        
        if clv_growth < 0:
            prediction_actions.append({
                "action_type": "Declining Value",
                "priority": "Medium",
                "description": "Value recovery strategy",
                "details": "Customer's predicted lifetime value is decreasing. Implement a win-back strategy with loyalty incentives and personalized messaging."
            })
        elif clv_growth > 100:
            prediction_actions.append({
                "action_type": "Growth Opportunity",
                "priority": "Medium",
                "description": "Nurture high-potential customer",
                "details": "Customer shows significant growth potential. Consider premium offerings and VIP treatment to maximize lifetime value."
            })
        
        # Add prediction-based actions to regular actions
        actions = prediction_actions + (actions or [])
    
    recommendation_sections = []
    
    # Next best actions
    action_items = []
    for action in actions or []:
        priority = action.get("priority", "Medium")
        priority_class = {
            "High": "danger",
            "Medium": "warning",
            "Low": "info"
        }.get(priority, "secondary")
        
        action_items.append(
            dbc.Card(
                dbc.CardBody([
                    html.Div([
                        html.H5(action.get("description", ""), className="card-title"),
                        dbc.Badge(priority, color=priority_class, className="ms-2")
                    ], className="d-flex justify-content-between align-items-center"),
                    html.P(action.get("action_type", ""), className="card-subtitle text-muted"),
                    html.P(action.get("details", ""), className="mt-2")
                ]),
                className="mb-2"
            )
        )
    
    if action_items:
        recommendation_sections.append(
            html.Div([
                html.H5("Next Best Actions"),
                html.Div(action_items)
            ], className="mb-4")
        )
    else:
        recommendation_sections.append(
            html.P("No action recommendations available", className="text-muted mb-4")
        )
    
    # Product recommendations
    if product_recommendations:
        product_rows = []
        for product in product_recommendations:
            product_id = product.get("product_id", "Unknown")
            purchase_count = product.get("purchase_count", 0)
            previously_viewed = product.get("previously_viewed", False)
            
            product_rows.append(
                html.Tr([
                    html.Td(product_id),
                    html.Td(purchase_count),
                    html.Td("Yes" if previously_viewed else "No")
                ])
            )
        
        product_table = dbc.Table(
            [
                html.Thead(
                    html.Tr([
                        html.Th("Product ID"),
                        html.Th("Popularity"),
                        html.Th("Previously Viewed")
                    ])
                ),
                html.Tbody(product_rows)
            ],
            bordered=True,
            hover=True,
            striped=True,
            responsive=True
        )
        
        recommendation_sections.append(
            html.Div([
                html.H5("Product Recommendations"),
                product_table
            ], className="mb-4")
        )
    else:
        recommendation_sections.append(
            html.P("No product recommendations available", className="text-muted mb-4")
        )
    
    # Similar customers
    if similar_customers:
        similar_rows = []
        for customer in similar_customers:
            customer_id = customer.get("customer_id", "Unknown")
            name = customer.get("name", "Unknown")
            similarity_score = customer.get("similarity_score", 0)
            
            similar_rows.append(
                html.Tr([
                    html.Td(customer_id),
                    html.Td(name),
                    html.Td(f"{similarity_score:.2f}")
                ])
            )
        
        similar_table = dbc.Table(
            [
                html.Thead(
                    html.Tr([
                        html.Th("Customer ID"),
                        html.Th("Name"),
                        html.Th("Similarity Score")
                    ])
                ),
                html.Tbody(similar_rows)
            ],
            bordered=True,
            hover=True,
            striped=True,
            responsive=True
        )
        
        recommendation_sections.append(
            html.Div([
                html.H5("Similar Customers"),
                similar_table
            ])
        )
    else:
        recommendation_sections.append(
            html.P("No similar customers found", className="text-muted")
        )
    
    return html.Div(recommendation_sections)

@app.callback(
    Output("churn-analysis", "children"),
    [
        Input("customer-data", "data"),
        Input("prediction-data", "data")
    ]
)
def update_churn_analysis(customer_data, prediction_data):
    """Update churn analysis tab with customer churn risk data."""
    if not customer_data:
        return html.P("No customer selected", className="text-muted")
    
    # Get traditional churn data
    insights = customer_data.get("insights", {})
    churn_risk = insights.get("churn_risk", {})
    
    # Get predictive churn data
    predicted_churn_probability = None
    predicted_churn_level = None
    
    if prediction_data:
        predicted_churn_probability = prediction_data.get("churn_probability", None)
        predicted_churn_level = prediction_data.get("churn_risk_level", None)
    
    if not churn_risk and predicted_churn_probability is None:
        return html.P("No churn risk data available for this customer", className="text-muted")
    
    # Extract churn risk data
    overall_risk = churn_risk.get("overall_risk", predicted_churn_level or "Unknown")
    factors = churn_risk.get("factors", {})
    
    # Create risk indicator
    risk_color = {
        "High": "danger",
        "Medium": "warning",
        "Low": "success",
        "Very Low": "info",
        "Unknown": "secondary"
    }.get(overall_risk, "secondary")
    
    # If we have predictive data, create a gauge chart
    if predicted_churn_probability is not None:
        gauge_fig = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=predicted_churn_probability * 100,
            number={"suffix": "%", "font": {"size": 24}},
            title={"text": "Churn Probability (ML Model)", "font": {"size": 20}},
            delta={"reference": 30, "increasing": {"color": "red"}, "decreasing": {"color": "green"}},
            gauge={
                "axis": {"range": [0, 100], "tickwidth": 1},
                "bar": {"color": "darkblue"},
                "bgcolor": "white",
                "borderwidth": 2,
                "bordercolor": "gray",
                "steps": [
                    {"range": [0, 30], "color": "green"},
                    {"range": [30, 70], "color": "yellow"},
                    {"range": [70, 100], "color": "red"}
                ],
                "threshold": {
                    "line": {"color": "red", "width": 4},
                    "thickness": 0.75,
                    "value": predicted_churn_probability * 100
                }
            }
        ))
        
        gauge_fig.update_layout(
            height=300,
            margin=dict(l=20, r=20, t=100, b=20)
        )
        
        risk_indicator = dbc.Card(
            dbc.CardBody([
                dcc.Graph(figure=gauge_fig)
            ]),
            className="mb-4"
        )
    else:
        # Traditional risk indicator
        risk_indicator = dbc.Card(
            dbc.CardBody([
                html.H5("Overall Churn Risk"),
                dbc.Progress(
                    value={"High": 75, "Medium": 50, "Low": 25, "Very Low": 10, "Unknown": 0}.get(overall_risk, 0),
                    color=risk_color,
                    className="mb-3"
                ),
                html.H3(
                    overall_risk,
                    className=f"text-center text-{risk_color}"
                )
            ]),
            className="mb-4"
        )
    
    # Create risk factor cards
    factor_cards = []
    for factor_name, factor_data in factors.items():
        level = factor_data.get("level", "Unknown")
        factor_color = {
            "High": "danger",
            "Medium": "warning",
            "Low": "success",
            "Very Low": "info",
            "Unknown": "secondary"
        }.get(level, "secondary")
        
        # Get specific details based on factor type
        details = []
        if factor_name == "inactivity":
            days = factor_data.get("days_inactive", 0)
            details.append(html.P(f"Days inactive: {days}"))
        elif factor_name == "purchase_history":
            count = factor_data.get("purchase_count", 0)
            details.append(html.P(f"Purchases: {count}"))
        elif factor_name == "cart_abandonment":
            has_abandoned = factor_data.get("has_abandoned", False)
            details.append(html.P(f"Has abandoned cart: {'Yes' if has_abandoned else 'No'}"))
        elif factor_name == "engagement":
            interactions = factor_data.get("email_interactions", 0)
            details.append(html.P(f"Email interactions: {interactions}"))
        
        factor_cards.append(
            dbc.Col(
                dbc.Card(
                    dbc.CardBody([
                        html.H5(factor_name.title(), className="card-title"),
                        html.Div([
                            html.H4(
                                level,
                                className=f"text-{factor_color}"
                            ),
                            *details
                        ])
                    ]),
                    className="h-100"
                ),
                width=6,
                className="mb-3"
            )
        )
    
    # Create recommendations based on risk factors
    recommendations = []
    high_risk_factors = [
        name for name, data in factors.items() 
        if data.get("level") == "High"
    ]
    
    if "inactivity" in high_risk_factors:
        recommendations.append(
            html.Li("Send a re-engagement email with special offer")
        )
    if "purchase_history" in high_risk_factors:
        recommendations.append(
            html.Li("Offer a first-purchase discount")
        )
    if "cart_abandonment" in factors and factors["cart_abandonment"].get("has_abandoned"):
        recommendations.append(
            html.Li("Send cart recovery email with limited-time discount")
        )
    if "engagement" in high_risk_factors:
        recommendations.append(
            html.Li("Improve email subject lines and content relevance")
        )
    
    if predicted_churn_probability is not None:
        if predicted_churn_probability > 0.7:
            recommendations.append(
                html.Li("Immediate retention offer with significant discount")
            )
        elif predicted_churn_probability > 0.3:
            recommendations.append(
                html.Li("Proactive outreach with personalized content")
            )
    
    recommendations_card = dbc.Card(
        dbc.CardBody([
            html.H5("Recommendations to Reduce Churn Risk"),
            html.Ul(recommendations if recommendations else html.Li("No specific recommendations"))
        ]),
        className="mt-3"
    )
    
    # Create the whole section
    churn_sections = [
        risk_indicator,
    ]
    
    # Add factor cards if available
    if factor_cards:
        churn_sections.extend([
            html.H5("Risk Factors"),
            dbc.Row(factor_cards),
        ])
    
    # Add model attribution if we have predictions
    if predicted_churn_probability is not None:
        churn_sections.append(
            dbc.Alert(
                [
                    html.H5("Machine Learning Prediction", className="alert-heading"),
                    html.P("This churn probability is calculated using a machine learning model trained on historical customer behavior patterns."),
                    html.P("The model analyzes factors such as activity levels, purchase frequency, engagement metrics, and previous churn patterns to estimate the likelihood of customer churn."),
                ],
                color="info",
                className="mt-3 mb-3"
            )
        )
    
    # Add recommendations
    churn_sections.append(recommendations_card)
    
    return html.Div(churn_sections)

@app.callback(
    Output("anomaly-detection", "children"),
    [Input("customer-data", "data")]
)
def update_anomaly_detection(customer_data):
    """Update anomaly detection tab with customer anomalies."""
    if not customer_data:
        return html.P("No customer selected", className="text-muted")
    
    customer_id = customer_data.get("customer_id", "Unknown")
    
    # Query Neo4j for anomalies
    anomaly_query = """
    MATCH (c:Customer {customer_id: $customer_id})-[:HAS_ANOMALY]->(a:Anomaly)
    RETURN a.detected_at as detected_at,
           a.activity_count as activity_count,
           a.purchase_amount as purchase_amount,
           a.activity_z_score as activity_z_score,
           a.purchase_z_score as purchase_z_score,
           a.anomaly_types as anomaly_types
    """
    
    predictor.connect()
    anomalies = predictor.run_query(anomaly_query, {"customer_id": customer_id})
    predictor.close()
    
    if not anomalies:
        # Check for segment anomalies (comparison to segment)
        segment_query = """
        MATCH (c:Customer {customer_id: $customer_id})-[:BELONGS_TO]->(s:Segment)
        
        // Count recent interactions (last 30 days)
        OPTIONAL MATCH (c)-[r]->()
        WHERE r.timestamp IS NOT NULL AND 
              duration.inDays(datetime(r.timestamp), datetime()).days <= 30
        WITH c, s, count(r) as recent_activity_count
        
        // Get purchase amounts
        OPTIONAL MATCH (c)-[p:PURCHASES]->()
        WHERE p.amount IS NOT NULL AND
              duration.inDays(datetime(p.timestamp), datetime()).days <= 30
        WITH c, s, recent_activity_count, sum(p.amount) as recent_purchase_amount
        
        // Get segment averages
        MATCH (other:Customer)-[:BELONGS_TO]->(s)
        OPTIONAL MATCH (other)-[r]->()
        WHERE r.timestamp IS NOT NULL AND 
              duration.inDays(datetime(r.timestamp), datetime()).days <= 30
        WITH c, s, recent_activity_count, recent_purchase_amount, avg(count(r)) as avg_segment_activity
        
        OPTIONAL MATCH (other:Customer)-[:BELONGS_TO]->(s)
        OPTIONAL MATCH (other)-[p:PURCHASES]->()
        WHERE p.amount IS NOT NULL AND
              duration.inDays(datetime(p.timestamp), datetime()).days <= 30
        WITH c, s, recent_activity_count, recent_purchase_amount, avg_segment_activity, 
             avg(sum(p.amount)) as avg_segment_purchases
        
        RETURN 
            s.id as segment_id,
            recent_activity_count,
            avg_segment_activity,
            recent_purchase_amount,
            avg_segment_purchases
        """
        
        predictor.connect()
        segment_comparison = predictor.run_query(segment_query, {"customer_id": customer_id})
        predictor.close()
        
        if not segment_comparison:
            return html.P("No anomaly data available for this customer", className="text-muted")
        
        # Create comparison visualization
        segment_sections = []
        
        for segment in segment_comparison:
            segment_id = segment.get("segment_id", "Unknown")
            activity_count = segment.get("recent_activity_count", 0)
            avg_segment_activity = segment.get("avg_segment_activity", 0)
            purchase_amount = segment.get("recent_purchase_amount", 0)
            avg_segment_purchases = segment.get("avg_segment_purchases", 0)
            
            # Calculate percentage differences
            activity_diff = (activity_count - avg_segment_activity) / max(1, avg_segment_activity) * 100
            purchase_diff = (purchase_amount - avg_segment_purchases) / max(1, avg_segment_purchases) * 100
            
            # Create comparison cards
            segment_sections.append(
                dbc.Card(
                    dbc.CardBody([
                        html.H5(f"Comparison to Segment: {segment_id}", className="card-title"),
                        
                        html.Div([
                            html.Div([
                                html.H6("Activity Level", className="text-center"),
                                html.Div([
                                    html.Span(f"{activity_count:.0f}", className="h3 me-2"),
                                    html.Small(f"vs. segment avg: {avg_segment_activity:.1f}")
                                ], className="text-center"),
                                html.Div([
                                    html.Span(
                                        f"{activity_diff:+.1f}%", 
                                        className=f"text-{'success' if activity_diff > 0 else 'danger'}"
                                    )
                                ], className="text-center")
                            ], className="col-6"),
                            
                            html.Div([
                                html.H6("Purchase Amount", className="text-center"),
                                html.Div([
                                    html.Span(f"${purchase_amount:.2f}", className="h3 me-2"),
                                    html.Small(f"vs. segment avg: ${avg_segment_purchases:.2f}")
                                ], className="text-center"),
                                html.Div([
                                    html.Span(
                                        f"{purchase_diff:+.1f}%", 
                                        className=f"text-{'success' if purchase_diff > 0 else 'danger'}"
                                    )
                                ], className="text-center")
                            ], className="col-6")
                        ], className="row")
                    ]),
                    className="mb-4"
                )
            )
            
            # Add interpretation
            anomaly_found = abs(activity_diff) > 50 or abs(purchase_diff) > 50
            
            if anomaly_found:
                interpretations = []
                
                if activity_diff < -50:
                    interpretations.append(html.Li("Significantly lower activity level than segment average"))
                elif activity_diff > 50:
                    interpretations.append(html.Li("Significantly higher activity level than segment average"))
                
                if purchase_diff < -50:
                    interpretations.append(html.Li("Significantly lower purchase amount than segment average"))
                elif purchase_diff > 50:
                    interpretations.append(html.Li("Significantly higher purchase amount than segment average"))
                
                segment_sections.append(
                    dbc.Alert(
                        [
                            html.H6("Potential Anomalies Detected", className="alert-heading"),
                            html.Ul(interpretations)
                        ],
                        color="warning" if abs(activity_diff) > 100 or abs(purchase_diff) > 100 else "info",
                        className="mb-4"
                    )
                )
            else:
                segment_sections.append(
                    dbc.Alert(
                        "Customer behavior is within normal range for their segment",
                        color="success",
                        className="mb-4"
                    )
                )
        
        return html.Div([
            html.H5("Segment Comparison Analysis"),
            html.P("No specific anomalies detected by the anomaly detection system, but here's how this customer compares to their segments:"),
            html.Div(segment_sections)
        ])
    
    # Process anomalies
    anomaly_cards = []
    
    for anomaly in anomalies:
        detected_at = anomaly.get("detected_at", "")
        if detected_at:
            try:
                detected_at = datetime.fromisoformat(detected_at).strftime("%Y-%m-%d %H:%M")
            except:
                pass
        
        activity_count = anomaly.get("activity_count", 0)
        purchase_amount = anomaly.get("purchase_amount", 0)
        activity_z_score = anomaly.get("activity_z_score", 0)
        purchase_z_score = anomaly.get("purchase_z_score", 0)
        anomaly_types = anomaly.get("anomaly_types", [])
        
        # Determine anomaly type and color
        anomaly_descriptions = []
        anomaly_color = "info"
        
        for anomaly_type in anomaly_types:
            if anomaly_type == "HIGH_ACTIVITY":
                anomaly_descriptions.append("Unusually high activity level")
                anomaly_color = "warning"
            elif anomaly_type == "LOW_ACTIVITY":
                anomaly_descriptions.append("Unusually low activity level")
                anomaly_color = "warning"
            elif anomaly_type == "HIGH_SPENDING":
                anomaly_descriptions.append("Unusually high purchase amount")
                anomaly_color = "danger"
            elif anomaly_type == "LOW_SPENDING":
                anomaly_descriptions.append("Unusually low purchase amount")
                anomaly_color = "danger"
        
        # Create anomaly card
        anomaly_cards.append(
            dbc.Card(
                dbc.CardBody([
                    html.H5("Behavior Anomaly Detected", className="card-title"),
                    html.P(f"Detected at: {detected_at}", className="card-subtitle mb-2 text-muted"),
                    
                    html.Ul([html.Li(desc) for desc in anomaly_descriptions], className="mb-3"),
                    
                    dbc.Row([
                        dbc.Col([
                            html.H6("Activity Metrics", className="text-center"),
                            html.P(f"Activity count: {activity_count}", className="text-center"),
                            html.P(f"Z-score: {activity_z_score:.2f}", className="text-center")
                        ], width=6),
                        dbc.Col([
                            html.H6("Purchase Metrics", className="text-center"),
                            html.P(f"Purchase amount: ${purchase_amount:.2f}", className="text-center"),
                            html.P(f"Z-score: {purchase_z_score:.2f}", className="text-center")
                        ], width=6)
                    ])
                ]),
                color=anomaly_color,
                outline=True,
                className="mb-4"
            )
        )
    
    # Explanation of anomaly detection
    explanation = dbc.Card(
        dbc.CardBody([
            html.H5("About Anomaly Detection", className="card-title"),
            html.P("Anomalies are detected by comparing customer behavior to their segment peers using statistical methods:"),
            html.Ul([
                html.Li("Z-scores measure how many standard deviations a value is from the segment mean"),
                html.Li("Values beyond ±3.0 standard deviations are considered anomalous"),
                html.Li("Anomalies may indicate unusual engagement patterns or potential fraud"),
                html.Li("Both activity levels and purchase amounts are monitored")
            ])
        ]),
        className="mt-3"
    )
    
    return html.Div([
        html.H5("Anomaly Detection Results"),
        html.Div(anomaly_cards),
        explanation
    ])

@app.callback(
    Output("model-performance", "children"),
    [Input("model-metadata", "data")]
)
def update_model_performance(model_data):
    """Update model performance tab with model metrics and feature importances."""
    if not model_data:
        # Get model metadata from Neo4j if not already loaded
        model_query = """
        MATCH (m:PredictiveModel)
        RETURN m.name as model_name, 
               m.metrics as metrics,
               m.feature_importances as feature_importances,
               m.last_updated as last_updated
        """
        predictor.connect()
        query_result = predictor.run_query(model_query)
        predictor.close()
        
        if not query_result:
            return html.Div([
                html.P("No model data available. Train models first by clicking the 'Train All Models' button.", className="text-muted"),
                dbc.Button("Train Models", id="train-models-button", color="primary", className="mt-2")
            ])
        
        # Process model data
        model_data = {}
        for model in query_result:
            model_name = model.get("model_name")
            metrics_json = model.get("metrics", "{}")
            feature_importances_json = model.get("feature_importances", "{}")
            
            try:
                metrics = json.loads(metrics_json)
                feature_importances = json.loads(feature_importances_json)
                
                model_data[model_name] = {
                    "metrics": metrics,
                    "feature_importances": feature_importances,
                    "last_updated": model.get("last_updated", "")
                }
            except json.JSONDecodeError:
                continue
    
    # Create model cards
    model_cards = []
    for model_name, model_details in model_data.items():
        if model_name == "training_results":
            continue
            
        metrics = model_details.get("metrics", {})
        feature_importances = model_details.get("feature_importances", {})
        last_updated = model_details.get("last_updated", "")
        
        if last_updated:
            try:
                last_updated = datetime.fromisoformat(last_updated).strftime("%Y-%m-%d %H:%M")
            except:
                pass
        
        # Create metrics table
        metrics_rows = []
        for metric, value in metrics.items():
            metrics_rows.append(
                html.Tr([
                    html.Td(metric),
                    html.Td(f"{value:.4f}" if isinstance(value, float) else str(value))
                ])
            )
        
        metrics_table = dbc.Table(
            [
                html.Thead(
                    html.Tr([
                        html.Th("Metric"),
                        html.Th("Value")
                    ])
                ),
                html.Tbody(metrics_rows)
            ],
            bordered=True,
            hover=True,
            striped=True,
            responsive=True,
            size="sm"
        )
        
        # Create feature importance visualization
        if feature_importances:
            # Sort by importance
            features = sorted(feature_importances.items(), key=lambda x: x[1], reverse=True)
            features = features[:10]  # Top 10 features
            
            feature_fig = go.Figure(go.Bar(
                x=[f[1] for f in features],
                y=[f[0] for f in features],
                orientation='h',
                marker_color='royalblue'
            ))
            
            feature_fig.update_layout(
                title="Top Feature Importances",
                xaxis_title="Importance",
                yaxis_title="Feature",
                height=400,
                margin=dict(l=20, r=20, t=50, b=20)
            )
            
            feature_importance_viz = dcc.Graph(figure=feature_fig)
        else:
            feature_importance_viz = html.P("No feature importance data available", className="text-muted")
        
        # Create model card
        model_cards.append(
            dbc.Card(
                dbc.CardBody([
                    html.H5(f"Model: {model_name}", className="card-title"),
                    html.P(f"Last updated: {last_updated}", className="card-subtitle mb-3 text-muted"),
                    
                    dbc.Tabs([
                        dbc.Tab(
                            html.Div([
                                html.H6("Performance Metrics", className="mt-3"),
                                metrics_table
                            ]),
                            label="Metrics", 
                            tab_id=f"metrics-{model_name}"
                        ),
                        dbc.Tab(
                            html.Div([
                                html.H6("Feature Importance", className="mt-3"),
                                feature_importance_viz
                            ]),
                            label="Features", 
                            tab_id=f"features-{model_name}"
                        )
                    ])
                ]),
                className="mb-4"
            )
        )
    
    # If no models, show message
    if not model_cards:
        return html.Div([
            html.P("No model data available. Train models first by clicking the 'Train All Models' button.", className="text-muted"),
            dbc.Button("Train Models", id="train-models-button", color="primary", className="mt-2")
        ])
    
    return html.Div([
        html.P("These models are trained on your customer data and used to generate predictions and insights.", className="mb-3"),
        html.Div(model_cards)
    ])

@app.callback(
    [
        Output("raw-data-selector", "options"),
        Output("raw-data-selector", "value")
    ],
    [
        Input("customer-data", "data"),
        Input("prediction-data", "data")
    ]
)
def update_raw_data_options(customer_data, prediction_data):
    """Update raw data selector options based on available data."""
    if not customer_data:
        return [], None
    
    options = [
        {"label": "Full Customer Report", "value": "full_report"},
        {"label": "Customer Profile", "value": "profile"},
        {"label": "Journey Timeline", "value": "timeline"},
        {"label": "Funnel Status", "value": "funnel_status"},
        {"label": "Similar Customers", "value": "similar_customers"},
        {"label": "Product Recommendations", "value": "product_recommendations"},
        {"label": "Churn Risk Assessment", "value": "churn_risk"},
        {"label": "Next Best Actions", "value": "actions"}
    ]
    
    if prediction_data:
        options.append({"label": "Predictive Insights", "value": "prediction_data"})
    
    return options, "full_report"

@app.callback(
    Output("raw-data-output", "children"),
    [
        Input("raw-data-selector", "value"),
        Input("customer-data", "data"),
        Input("prediction-data", "data")
    ]
)
def update_raw_data(selected_data, customer_data, prediction_data):
    """Update raw data display based on selected data option."""
    if not selected_data or not customer_data:
        return html.P("No data to display", className="text-muted")
    
    if selected_data == "full_report":
        data_to_display = customer_data
    elif selected_data == "profile":
        data_to_display = customer_data.get("profile", {})
    elif selected_data == "timeline":
        data_to_display = customer_data.get("journey", {}).get("timeline", [])
    elif selected_data == "funnel_status":
        data_to_display = customer_data.get("journey", {}).get("funnel_status", {})
    elif selected_data == "similar_customers":
        data_to_display = customer_data.get("insights", {}).get("similar_customers", [])
    elif selected_data == "product_recommendations":
        data_to_display = customer_data.get("insights", {}).get("product_recommendations", [])
    elif selected_data == "churn_risk":
        data_to_display = customer_data.get("insights", {}).get("churn_risk", {})
    elif selected_data == "actions":
        data_to_display = customer_data.get("actions", [])
    elif selected_data == "prediction_data" and prediction_data:
        data_to_display = prediction_data
    else:
        data_to_display = {"message": "Invalid selection"}
    
    formatted_data = json.dumps(data_to_display, indent=2)
    
    return html.Div([
        dbc.Button(
            "Copy to Clipboard", 
            id="copy-button", 
            color="secondary", 
            size="sm",
            className="mb-2"
        ),
        dbc.Collapse(
            dbc.Alert("Copied to clipboard!", color="success"),
            id="copy-alert",
            is_open=False
        ),
        dcc.Textarea(
            value=formatted_data,
            id="raw-data-text",
            style={"width": "100%", "height": "400px", "fontFamily": "monospace"},
            readOnly=True
        )
    ])

@app.callback(
    Output("copy-alert", "is_open"),
    [Input("copy-button", "n_clicks")],
    [State("raw-data-text", "value")],
    prevent_initial_call=True
)
def copy_to_clipboard(n_clicks, text):
    """Show alert when data is copied to clipboard."""
    if n_clicks:
        # Note: In a real app, this would use a clientside callback
        # to access the clipboard API. Since this is server-side,
        # we just simulate it with a notification.
        return True
    return False

def main():
    """Run the Dash app."""
    port = int(os.environ.get("PORT", 8050))
    app.run_server(debug=True, host="0.0.0.0", port=port)

if __name__ == "__main__":
    print("Starting Predictive Customer Dashboard...")
    main()