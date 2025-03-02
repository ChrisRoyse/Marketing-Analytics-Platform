#!/usr/bin/env python3
"""
Customer Journey Dashboard - Web interface for the marketing ontology.
This provides a user-friendly interface to interact with the marketing ontology
and visualize customer journeys with minimal input.
"""

import os
import json
import dash
import base64
import pandas as pd
import networkx as nx
from dash import dcc, html, callback, Input, Output, State
import dash_bootstrap_components as dbc
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
from flask import Flask
from dynamic_customer_analyzer import DynamicCustomerAnalyzer

# Initialize the Dash app with Bootstrap styling
server = Flask(__name__)
app = dash.Dash(
    __name__, 
    server=server,
    external_stylesheets=[dbc.themes.FLATLY],
    suppress_callback_exceptions=True
)

# Initialize the analyzer
analyzer = DynamicCustomerAnalyzer()

# Ensure customer insights directory exists
Path("customer_insights").mkdir(exist_ok=True)

# App layout
app.layout = dbc.Container([
    dbc.Row(
        dbc.Col(
            html.H1("Marketing Ontology - Customer Journey Dashboard", className="text-center my-4"),
            width=12
        )
    ),
    
    dbc.Row([
        dbc.Col(
            dbc.Card(
                dbc.CardBody([
                    html.H4("Customer Lookup", className="card-title"),
                    html.P("Enter a customer ID to analyze their journey", className="card-text"),
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
                    html.H4("Batch Processing", className="card-title"),
                    html.P("Process multiple customers automatically", className="card-text"),
                    dbc.Button("Process All Customers", id="batch-button", color="secondary", className="mt-3"),
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
                    html.H4("Quick Stats", className="card-title"),
                    html.Div(id="quick-stats"),
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
                html.H4("Raw Data", className="card-title"),
                dcc.Dropdown(id="raw-data-selector", className="mb-3"),
                html.Div(id="raw-data-output"),
            ])),
            label="Raw Data", tab_id="tab-raw"
        ),
    ], id="customer-tabs", active_tab="tab-profile"),
    
    # Store the current customer data
    dcc.Store(id="customer-data"),
    
    # Footer
    dbc.Row(
        dbc.Col(
            html.Footer([
                html.P("Marketing Behavior Pattern Ontology - Customer Dashboard", className="mb-1"),
                html.P("© 2025 Your Company", className="small text-muted")
            ], className="text-center my-4"),
            width=12
        )
    )
], fluid=True)

@app.callback(
    [
        Output("loading-output", "children"),
        Output("customer-data", "data")
    ],
    [Input("analyze-button", "n_clicks")],
    [State("customer-id-input", "value")],
    prevent_initial_call=True
)
def analyze_customer(n_clicks, customer_id):
    """Analyze a specific customer when the button is clicked."""
    if not n_clicks or not customer_id:
        return "Please enter a customer ID", None
    
    if not analyzer.connect():
        return "Failed to connect to database. Check configuration.", None
    
    if not analyzer.validate_customer_id(customer_id):
        return f"Customer ID '{customer_id}' not found in database", None
    
    try:
        # Generate the customer report
        report = analyzer.create_customer_report(customer_id)
        
        if "error" in report:
            return f"Error: {report['error']}", None
        
        return "Analysis complete! View the tabs below for details.", report
    
    except Exception as e:
        return f"Error during analysis: {str(e)}", None
    
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
            
            return html.Div([
                html.P(f"Successfully processed {report_count} customers", className="text-success"),
                html.P(f"Reports saved to {reports_path}/")
            ])
        else:
            return html.P("Batch processing failed. Check logs for details.", className="text-danger")
    
    except Exception as e:
        return html.P(f"Error during batch processing: {str(e)}", className="text-danger")

@app.callback(
    Output("quick-stats", "children"),
    [Input("customer-data", "data")]
)
def update_quick_stats(data):
    """Update quick stats based on the current customer data."""
    if not data:
        return html.P("No customer selected", className="text-muted")
    
    profile = data.get("profile", {})
    journey = data.get("journey", {})
    insights = data.get("insights", {})
    
    # Extract key metrics
    customer_id = data.get("customer_id", "Unknown")
    name = profile.get("basic_info", {}).get("name", "Unknown")
    current_stage = journey.get("funnel_status", {}).get("current_stage", "Unknown")
    has_churned = journey.get("funnel_status", {}).get("has_churned", False)
    churn_risk = insights.get("churn_risk", {}).get("overall_risk", "Unknown")
    
    # Create stats cards
    stats = [
        dbc.ListGroupItem([
            html.Span("ID: ", className="fw-bold"),
            html.Span(customer_id)
        ]),
        dbc.ListGroupItem([
            html.Span("Name: ", className="fw-bold"),
            html.Span(name)
        ]),
        dbc.ListGroupItem([
            html.Span("Funnel Stage: ", className="fw-bold"),
            html.Span(current_stage)
        ]),
        dbc.ListGroupItem([
            html.Span("Churned: ", className="fw-bold"),
            html.Span("Yes" if has_churned else "No", 
                      className="text-danger" if has_churned else "text-success")
        ]),
        dbc.ListGroupItem([
            html.Span("Churn Risk: ", className="fw-bold"),
            html.Span(churn_risk, 
                      className={
                          "High": "text-danger",
                          "Medium": "text-warning",
                          "Low": "text-success"
                      }.get(churn_risk, ""))
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
    Output("recommendations", "children"),
    [Input("customer-data", "data")]
)
def update_recommendations(data):
    """Update recommendations tab with customer recommendations."""
    if not data:
        return html.P("No customer selected", className="text-muted")
    
    insights = data.get("insights", {})
    product_recommendations = insights.get("product_recommendations", [])
    similar_customers = insights.get("similar_customers", [])
    actions = data.get("actions", [])
    
    recommendation_sections = []
    
    # Next best actions
    action_items = []
    for action in actions:
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
    [Input("customer-data", "data")]
)
def update_churn_analysis(data):
    """Update churn analysis tab with customer churn risk data."""
    if not data:
        return html.P("No customer selected", className="text-muted")
    
    insights = data.get("insights", {})
    churn_risk = insights.get("churn_risk", {})
    
    if not churn_risk:
        return html.P("No churn risk data available for this customer", className="text-muted")
    
    # Extract churn risk data
    overall_risk = churn_risk.get("overall_risk", "Unknown")
    factors = churn_risk.get("factors", {})
    
    # Create risk indicator
    risk_color = {
        "High": "danger",
        "Medium": "warning",
        "Low": "success",
        "Very Low": "info",
        "Unknown": "secondary"
    }.get(overall_risk, "secondary")
    
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
    
    recommendations_card = dbc.Card(
        dbc.CardBody([
            html.H5("Recommendations to Reduce Churn Risk"),
            html.Ul(recommendations if recommendations else html.Li("No specific recommendations"))
        ]),
        className="mt-3"
    )
    
    return html.Div([
        risk_indicator,
        html.H5("Risk Factors"),
        dbc.Row(factor_cards),
        recommendations_card
    ])

@app.callback(
    [
        Output("raw-data-selector", "options"),
        Output("raw-data-selector", "value")
    ],
    [Input("customer-data", "data")]
)
def update_raw_data_options(data):
    """Update raw data selector options based on available data."""
    if not data:
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
    
    return options, "full_report"

@app.callback(
    Output("raw-data-output", "children"),
    [
        Input("raw-data-selector", "value"),
        Input("customer-data", "data")
    ]
)
def update_raw_data(selected_data, customer_data):
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
    print("Starting Customer Journey Dashboard...")
    main()