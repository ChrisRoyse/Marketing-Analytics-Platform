#!/usr/bin/env python3
import sys
from pathlib import Path

def enhance_dashboard():
    source_dashboard = Path("predictive_dashboard.py")
    target_dashboard = Path("enhanced_dashboard.py")
    
    if not source_dashboard.exists():
        print("Source dashboard file not found")
        return False
    
    # Read the existing dashboard
    with open(source_dashboard, "r") as f:
        dashboard_content = f.read()
    
    # Add new imports
    new_imports = '''
import re
from enhanced_personalization import EnhancedPersonalization

# Initialize the enhanced personalization engine
enhancer = EnhancedPersonalization()
'''
    
    # Find the import section end
    import_end = dashboard_content.find("# Initialize the")
    if import_end == -1:
        import_end = dashboard_content.find("# App layout")
    
    # Insert new imports
    modified_content = dashboard_content[:import_end] + new_imports + dashboard_content[import_end:]
    
    # Add new tab for context-aware recommendations
    context_tab = '''
        dbc.Tab(
            dbc.Card(dbc.CardBody([
                html.H4("Context-Aware Recommendations", className="card-title"),
                html.Div(id="context-recommendations"),
            ])),
            label="Context Recs", tab_id="tab-context"
        ),
        dbc.Tab(
            dbc.Card(dbc.CardBody([
                html.H4("Customer Feedback Analysis", className="card-title"),
                html.Div(id="feedback-analysis"),
            ])),
            label="Feedback", tab_id="tab-feedback"
        ),
'''
    
    # Find the tab section and insert the new tab
    tabs_start = modified_content.find("dbc.Tabs([")
    if tabs_start != -1:
        tab_location = modified_content.find('tab_id="tab-recommendations"')
        if tab_location != -1:
            close_bracket = modified_content.find(")", tab_location)
            insert_pos = close_bracket + 1
            modified_content = modified_content[:insert_pos] + "," + context_tab + modified_content[insert_pos:]
    
    # Add new callbacks for the context-aware recommendations
    context_callback = '''
@app.callback(
    Output("context-recommendations", "children"),
    [
        Input("customer-data", "data"),
        Input("analyze-button", "n_clicks")
    ],
    [State("customer-id-input", "value")],
    prevent_initial_call=True
)
def update_context_recommendations(customer_data, n_clicks, customer_id):
    # Update context-aware recommendations tab
    if not customer_data or not customer_id:
        return html.P("No customer selected", className="text-muted")
    
    try:
        # Connect to Neo4j
        if not enhancer.connect():
            return html.P("Failed to connect to database", className="text-danger")
        
        # Get context data
        context = enhancer.get_context_data(customer_id)
        
        # Generate recommendations
        recommendations = enhancer.generate_context_aware_recommendations(customer_id)
        
        enhancer.close()
        
        if not recommendations or not context:
            return html.P("Could not generate context-aware recommendations", className="text-warning")
        
        # Create context information section
        time_context = context.get("time_context", {})
        weather_context = context.get("weather_context", {})
        event_context = context.get("event_context", {})
        
        context_cards = []
        
        # Time context
        time_info = [
            html.Li(f"Time of day: {time_context.get('time_of_day', 'Unknown')}"),
            html.Li(f"Season: {time_context.get('season', 'Unknown')}"),
            html.Li(f"{'Weekend' if time_context.get('is_weekend') else 'Weekday'}")
        ]
        
        context_cards.append(
            dbc.Col(
                dbc.Card(
                    dbc.CardBody([
                        html.H5("Time Context", className="card-title"),
                        html.Ul(time_info)
                    ])
                ),
                width=4
            )
        )
        
        # Weather context
        if weather_context:
            weather_info = [
                html.Li(f"Conditions: {weather_context.get('conditions', 'Unknown')}"),
                html.Li(f"Temperature: {weather_context.get('temperature', {}).get('fahrenheit', 0)}°F / {weather_context.get('temperature', {}).get('celsius', 0)}°C"),
                html.Li(f"Precipitation: {'Yes' if weather_context.get('is_precipitation') else 'No'}")
            ]
            
            context_cards.append(
                dbc.Col(
                    dbc.Card(
                        dbc.CardBody([
                            html.H5("Weather Context", className="card-title"),
                            html.Ul(weather_info)
                        ])
                    ),
                    width=4
                )
            )
        
        # Event context
        if event_context and event_context.get("has_events"):
            events = event_context.get("events", [])
            event_items = [html.Li(event.get("name", "")) for event in events]
            
            context_cards.append(
                dbc.Col(
                    dbc.Card(
                        dbc.CardBody([
                            html.H5("Local Events", className="card-title"),
                            html.Ul(event_items)
                        ])
                    ),
                    width=4
                )
            )
        
        # Create recommendation cards
        recommendation_cards = []
        
        for i, rec in enumerate(recommendations[:6]):  # Show top 6 recommendations
            product_id = rec.get("product_id", "")
            name = rec.get("name", "Unknown Product")
            category = rec.get("category", "")
            price = rec.get("price", 0)
            explanations = rec.get("explanation", [])
            
            card = dbc.Card(
                dbc.CardBody([
                    html.H5(name, className="card-title"),
                    html.H6(f"${price:.2f}", className="card-subtitle text-success mb-2"),
                    html.P(f"Category: {category}", className="card-text"),
                    html.P("Why we recommend this:", className="text-muted small mt-2 mb-1"),
                    html.Ul([html.Li(exp) for exp in explanations], className="small ps-3")
                ]),
                className="mb-3 h-100"
            )
            
            recommendation_cards.append(
                dbc.Col(card, width=4, className="mb-3")
            )
        
        sections = [
            html.H5("Current Context"),
            dbc.Row(context_cards, className="mb-4"),
            html.H5("Context-Aware Recommendations"),
            dbc.Row(recommendation_cards),
            html.Hr(),
            html.P("These recommendations are personalized based on your current context, " +
                   "purchase history, and browsing behavior.", className="mt-3 small text-muted")
        ]
        
        return html.Div(sections)
    
    except Exception as e:
        return html.P(f"Error generating context recommendations: {str(e)}", className="text-danger")

@app.callback(
    Output("feedback-analysis", "children"),
    [
        Input("customer-data", "data"),
        Input("analyze-button", "n_clicks")
    ],
    [State("customer-id-input", "value")],
    prevent_initial_call=True
)
def update_feedback_analysis(customer_data, n_clicks, customer_id):
    # Update feedback analysis tab
    if not customer_data or not customer_id:
        return html.P("No customer selected", className="text-muted")
    
    try:
        # Connect to Neo4j
        if not enhancer.connect():
            return html.P("Failed to connect to database", className="text-danger")
        
        # Get NLP insights
        insights_query = """
        MATCH (c:Customer {customer_id: $customer_id})-[:HAS_INSIGHT]->(i:NLPInsight)
        OPTIONAL MATCH (i)-[r:HAS_TOPIC]->(t:Topic)
        RETURN i.sentiment_score as sentiment_score,
               i.predominant_sentiment as predominant_sentiment,
               i.positive_count as positive_count,
               i.negative_count as negative_count,
               i.keywords as keywords,
               i.feedback_count as feedback_count,
               i.average_rating as average_rating,
               collect({id: t.id, words: t.top_words, relevance: r.relevance}) as topics
        """
        
        insights_result = enhancer.run_query(insights_query, {"customer_id": customer_id})
        
        # If no insights, analyze feedback
        if not insights_result:
            # Analyze customer feedback
            insights = enhancer.analyze_customer_feedback(customer_id)
            if insights:
                # Query again
                insights_result = enhancer.run_query(insights_query, {"customer_id": customer_id})
            else:
                enhancer.close()
                return html.P("No feedback data available for analysis", className="text-warning")
        
        enhancer.close()
        
        if not insights_result:
            return html.P("Failed to retrieve NLP insights", className="text-warning")
        
        insights = insights_result[0]
        
        # Extract data
        sentiment_score = insights.get("sentiment_score", 0.5)
        predominant_sentiment = insights.get("predominant_sentiment", "NEUTRAL")
        positive_count = insights.get("positive_count", 0)
        negative_count = insights.get("negative_count", 0)
        keywords = insights.get("keywords", [])
        feedback_count = insights.get("feedback_count", 0)
        average_rating = insights.get("average_rating", 0)
        topics = [t for t in insights.get("topics", []) if t.get("id")]
        
        # Create sentiment gauge
        sentiment_color = "success" if sentiment_score > 0.7 else "warning" if sentiment_score > 0.4 else "danger"
        
        sentiment_gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=sentiment_score * 100,
            title={"text": "Sentiment Score"},
            gauge={
                "axis": {"range": [0, 100], "tickwidth": 1},
                "bar": {"color": {
                    "success": "#2ecc71",
                    "warning": "#f39c12",
                    "danger": "#e74c3c"
                }.get(sentiment_color, "#95a5a6")},
                "steps": [
                    {"range": [0, 40], "color": "#ffcccc"},
                    {"range": [40, 70], "color": "#ffffcc"},
                    {"range": [70, 100], "color": "#ccffcc"}
                ]
            }
        ))
        
        sentiment_gauge.update_layout(
            height=250,
            margin=dict(l=20, r=20, t=50, b=20)
        )
        
        # Create bar chart for sentiment breakdown
        sentiment_breakdown = go.Figure()
        sentiment_breakdown.add_trace(go.Bar(
            x=["Positive", "Negative"],
            y=[positive_count, negative_count],
            marker_color=["#2ecc71", "#e74c3c"]
        ))
        
        sentiment_breakdown.update_layout(
            title="Feedback Sentiment Breakdown",
            height=250,
            margin=dict(l=20, r=20, t=50, b=20)
        )
        
        # Create keyword and topic sections
        sections = [
            dbc.Row([
                dbc.Col(
                    dbc.Card(
                        dbc.CardBody([
                            html.H5("Overall Sentiment"),
                            dcc.Graph(figure=sentiment_gauge),
                            html.P([
                                html.Span("Predominant sentiment: ", className="fw-bold"),
                                html.Span(predominant_sentiment, className={
                                    "POSITIVE": "text-success",
                                    "NEGATIVE": "text-danger",
                                    "NEUTRAL": "text-muted"
                                }.get(predominant_sentiment, ""))
                            ], className="mt-3")
                        ])
                    ),
                    width=6
                ),
                dbc.Col(
                    dbc.Card(
                        dbc.CardBody([
                            html.H5("Feedback Analysis"),
                            dcc.Graph(figure=sentiment_breakdown),
                            html.P([
                                html.Span("Total feedback: ", className="fw-bold"),
                                html.Span(f"{feedback_count} items")
                            ], className="mt-3"),
                            html.P([
                                html.Span("Average rating: ", className="fw-bold"),
                                html.Span(f"{average_rating:.1f} / 5.0")
                            ])
                        ])
                    ),
                    width=6
                )
            ], className="mb-4"),
            
            dbc.Row([
                dbc.Col(
                    dbc.Card(
                        dbc.CardBody([
                            html.H5("Key Topics"),
                            html.Div([
                                dbc.Badge(
                                    topic.get("words", ["Unknown"])[0:3],
                                    color="primary",
                                    className="me-2 mb-2"
                                ) for topic in topics
                            ]) if topics else html.P("No topics identified")
                        ])
                    ),
                    width=6
                ),
                dbc.Col(
                    dbc.Card(
                        dbc.CardBody([
                            html.H5("Common Keywords"),
                            html.Div([
                                dbc.Badge(
                                    keyword,
                                    color="secondary",
                                    className="me-2 mb-2"
                                ) for keyword in keywords
                            ]) if keywords else html.P("No keywords identified")
                        ])
                    ),
                    width=6
                )
            ])
        ]
        
        return html.Div(sections)
    
    except Exception as e:
        return html.P(f"Error generating feedback analysis: {str(e)}", className="text-danger")
'''

    # Find where to add the new callbacks
    end_of_callbacks = modified_content.rfind("if __name__ ==")
    if end_of_callbacks != -1:
        # Insert before the main check
        modified_content = modified_content[:end_of_callbacks] + context_callback + "\n" + modified_content[end_of_callbacks:]
    
    # Update the dashboard title
    modified_content = modified_content.replace(
        "Marketing Ontology - Predictive Customer Analytics",
        "Marketing Ontology - Enhanced Personalized Analytics"
    )
    
    # Write to the new file
    with open(target_dashboard, "w") as f:
        f.write(modified_content)
    
    print(f"Created enhanced dashboard: {target_dashboard}")
    return True

if __name__ == "__main__":
    sys.exit(0 if enhance_dashboard() else 1)