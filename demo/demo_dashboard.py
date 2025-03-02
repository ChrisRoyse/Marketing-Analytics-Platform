#!/usr/bin/env python
"""
Demo dashboard for the Marketing Ontology Platform.

This dashboard extends the existing platform with demo-specific features,
including customer journey selector, journey animation, and demo scenarios.
"""

import os
import sys
from pathlib import Path
import json
import datetime
import dash
from dash import dcc, html, callback, Output, Input, State
import dash_bootstrap_components as dbc
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from dotenv import load_dotenv
from neo4j import GraphDatabase

# Add parent directory to path to access shared modules
sys.path.append(str(Path(__file__).resolve().parent.parent))

# Load environment variables from .env file
load_dotenv()

# Neo4j connection parameters
neo4j_uri = os.getenv("NEO4J_URI", "bolt://172.19.160.1:7687")
neo4j_user = os.getenv("NEO4J_USERNAME", "neo4j")
neo4j_password = os.getenv("NEO4J_PASSWORD", "#1Moneymaker")
neo4j_database = os.getenv("NEO4J_DATABASE", "marketing")

# Initialize Neo4j driver
driver = GraphDatabase.driver(neo4j_uri, auth=(neo4j_user, neo4j_password))

# Initialize Dash app with Bootstrap theme
app = dash.Dash(
    __name__,
    external_stylesheets=[dbc.themes.FLATLY],
    meta_tags=[{"name": "viewport", "content": "width=device-width, initial-scale=1"}],
    title="Marketing Ontology Platform Demo"
)

# Define color scheme for consistency
COLORS = {
    "primary": "#2C3E50",
    "secondary": "#18BC9C",
    "background": "#ECF0F1",
    "text": "#2C3E50",
    "stages": {
        "awareness": "#3498DB",
        "consideration": "#2980B9",
        "intent": "#9B59B6",
        "conversion": "#18BC9C",
        "retention": "#F39C12",
        "advocacy": "#E74C3C"
    },
    "personas": {
        "Tech Enthusiast": "#3498DB",
        "Budget Shopper": "#9B59B6",
        "Gift Buyer": "#F39C12",
        "Professional": "#18BC9C",
        "Student": "#E74C3C"
    }
}

# Define funnel stages in order
FUNNEL_STAGES = ["awareness", "consideration", "intent", "conversion", "retention", "advocacy"]

# Create helper functions for Neo4j queries
def run_query(query, params=None):
    """Run a Cypher query against Neo4j and return the results."""
    with driver.session(database=neo4j_database) as session:
        result = session.run(query, params or {})
        return [record.data() for record in result]

def get_customer_list():
    """Get list of customers for the customer selector."""
    query = """
    MATCH (c:Customer)
    RETURN c.customer_id AS id, c.first_name + ' ' + c.last_name AS name,
           c.segment AS segment
    ORDER BY c.segment, c.first_name
    """
    customers = run_query(query)
    return customers

def get_customer_journey(customer_id):
    """Get journey events for a specific customer."""
    query = """
    MATCH (c:Customer {customer_id: $customer_id})-[r]->(target)
    WHERE type(r) IN ['VIEWS', 'CLICKS_ON', 'COMES_FROM', 'VISITS', 'ADDS_TO_CART',
                      'ABANDONS', 'PURCHASES', 'RECEIVES', 'OPENS', 'CREATES', 
                      'LOGS_IN', 'WRITES', 'REFERS', 'SHARES', 'CHURNED_AT']
    RETURN type(r) AS event_type, 
           r.timestamp AS timestamp,
           labels(target)[0] AS target_type,
           CASE
             WHEN target.name IS NOT NULL THEN target.name
             WHEN target.subject IS NOT NULL THEN target.subject
             WHEN target.id IS NOT NULL THEN target.id
             ELSE 'Unknown'
           END AS target_name,
           r.properties AS properties
    ORDER BY r.timestamp
    """
    events = run_query(query, {"customer_id": customer_id})
    
    # Map events to funnel stages
    funnel_stage_mapping = {
        "VIEWS": {
            "Advertisement": "awareness",
            "Product": "consideration",
            "Email": "intent"
        },
        "CLICKS_ON": {
            "Advertisement": "awareness",
            "Email": "intent"
        },
        "COMES_FROM": "awareness",
        "VISITS": {
            "Page": {
                "Home": "awareness",
                "Products": "awareness",
                "Category": "consideration",
                "Shopping Cart": "intent",
                "Checkout": "conversion",
                "My Account": "retention",
                "Customer Support": "retention",
                "Blog": "consideration"
            }
        },
        "ADDS_TO_CART": "intent",
        "ABANDONS": "intent",
        "PURCHASES": "conversion",
        "RECEIVES": {
            "Email": {
                "abandoned_cart": "intent",
                "welcome": "awareness",
                "transactional": "conversion",
                "recommendation": "retention",
                "feedback": "advocacy",
                "promotional": "retention"
            }
        },
        "OPENS": "intent",
        "CREATES": {
            "Account": "intent",
            "Ticket": "retention"
        },
        "LOGS_IN": "retention",
        "WRITES": "advocacy",
        "REFERS": "advocacy",
        "SHARES": "advocacy",
        "CHURNED_AT": "retention"
    }
    
    for event in events:
        # Add funnel stage based on mapping
        event_type = event["event_type"]
        target_type = event["target_type"]
        target_name = event["target_name"]
        
        if isinstance(funnel_stage_mapping.get(event_type), dict):
            if target_type in funnel_stage_mapping[event_type]:
                if isinstance(funnel_stage_mapping[event_type][target_type], dict):
                    # Check for specific page/email types
                    for key in funnel_stage_mapping[event_type][target_type]:
                        if key in target_name:
                            event["funnel_stage"] = funnel_stage_mapping[event_type][target_type][key]
                            break
                    else:
                        # Default if no specific match
                        event["funnel_stage"] = "consideration"
                else:
                    event["funnel_stage"] = funnel_stage_mapping[event_type][target_type]
            else:
                event["funnel_stage"] = "consideration"  # Default
        else:
            event["funnel_stage"] = funnel_stage_mapping.get(event_type, "consideration")
    
    return events

def get_customer_details(customer_id):
    """Get detailed information about a specific customer."""
    query = """
    MATCH (c:Customer {customer_id: $customer_id})
    OPTIONAL MATCH (c)-[:BELONGS_TO]->(s:Segment)
    OPTIONAL MATCH (c)-[:HAS_PERSONA]->(p:Persona)
    OPTIONAL MATCH (c)-[:LIVES_IN]->(l:Location)
    OPTIONAL MATCH (c)-[:USES]->(d:Device)
    
    // Get purchase data for calculations
    OPTIONAL MATCH (c)-[pur:PURCHASES]->(prod:Product)
    WITH c, s, p, l, d, 
         collect({quantity: pur.quantity, price: pur.price, date: pur.timestamp, product: prod.name}) AS purchases,
         COALESCE(sum(pur.quantity * pur.price), 0) AS total_spent,
         count(pur) AS purchase_count,
         collect(DISTINCT p.name) AS personas,
         collect(DISTINCT d.id) AS devices
    
    // Get current date for calculations
    WITH c, s, l, personas, devices, purchases, total_spent, purchase_count, datetime() AS current_date
    
    // Calculate recency, frequency, and monetary values (RFM)
    WITH c, s, l, personas, devices, purchases, total_spent, purchase_count, current_date,
         CASE WHEN purchase_count > 0 
              THEN duration.between(datetime(purchases[-1].date), current_date).days 
              ELSE 365 END AS days_since_last_purchase,
         CASE WHEN c.registration_date IS NOT NULL 
              THEN duration.between(datetime(c.registration_date), current_date).days 
              ELSE 365 END AS account_age_days
    
    // Calculate purchase frequency (purchases per year)
    WITH c, s, l, personas, devices, purchases, total_spent, purchase_count, current_date, 
         days_since_last_purchase, account_age_days,
         CASE WHEN account_age_days > 0 
              THEN (purchase_count * 365.0) / account_age_days 
              ELSE 0 END AS purchase_frequency
    
    // Calculate average purchase value
    WITH c, s, l, personas, devices, purchases, total_spent, purchase_count, current_date,
         days_since_last_purchase, account_age_days, purchase_frequency,
         CASE WHEN purchase_count > 0 
              THEN total_spent / purchase_count 
              ELSE 0 END AS avg_purchase_value
    
    // Personalized engagement score based on behavior and segment
    WITH c, s, l, personas, devices, purchases, total_spent, purchase_count, current_date,
         days_since_last_purchase, account_age_days, purchase_frequency, avg_purchase_value,
         CASE 
           WHEN s.name = 'Tech Enthusiast' THEN 0.85
           WHEN s.name = 'Professional' THEN 0.78
           WHEN s.name = 'Budget Shopper' THEN 0.65
           WHEN s.name = 'Student' THEN 0.55
           WHEN s.name = 'Gift Buyer' THEN 0.40
           ELSE 0.60
         END * 
         CASE 
           WHEN purchase_count > 3 THEN 1.2
           WHEN purchase_count > 1 THEN 1.0
           WHEN purchase_count = 1 THEN 0.8
           ELSE 0.5
         END * 
         CASE 
           WHEN days_since_last_purchase < 30 THEN 1.3
           WHEN days_since_last_purchase < 90 THEN 1.0
           WHEN days_since_last_purchase < 180 THEN 0.7
           ELSE 0.4
         END AS engagement_score
    
    // Calculate churn probability
    WITH c, s, l, personas, devices, purchases, total_spent, purchase_count, days_since_last_purchase,
         purchase_frequency, avg_purchase_value, engagement_score,
         // First calculate base churn risk (0-0.9 scale)
         CASE
           // Base the churn probability on real customer data
           WHEN purchase_count = 0 THEN 0.90  // No purchases = high risk
           WHEN days_since_last_purchase > 120 THEN 0.75  // Inactive = high risk
           WHEN days_since_last_purchase > 60 THEN 0.50   // Starting to disengage
           WHEN purchase_count = 1 AND days_since_last_purchase > 30 THEN 0.60  // One-time customer risk
           WHEN purchase_frequency > 3 THEN 0.15  // Frequent buyer = low risk
           WHEN purchase_frequency > 1 THEN 0.25  // Regular buyer = medium-low risk
           WHEN avg_purchase_value > 500 THEN 0.30  // High value but infrequent = medium risk
           ELSE 0.45  // Default medium risk
         END AS base_churn,
         
         // Get segment modifier
         CASE 
           WHEN s.name = 'Tech Enthusiast' THEN 0.7  // More loyal
           WHEN s.name = 'Gift Buyer' THEN 1.5      // Less loyal
           WHEN s.name = 'Professional' THEN 0.8    // Somewhat loyal
           WHEN s.name = 'Budget Shopper' THEN 1.2  // Price sensitive
           WHEN s.name = 'Student' THEN 1.1         // Somewhat price sensitive
           ELSE 1.0
         END AS segment_modifier
         
    // Apply segment modifier but cap at 0.95 max to avoid over 100%
    WITH c, s, l, personas, devices, purchases, total_spent, purchase_count, days_since_last_purchase,
         purchase_frequency, avg_purchase_value, engagement_score, base_churn, segment_modifier,
         CASE 
           WHEN base_churn * segment_modifier > 0.95 THEN 0.95
           ELSE base_churn * segment_modifier
         END AS churn_probability
                  
    // Calculate retention rate (inverse of churn)
    WITH c, s, l, personas, devices, purchases, total_spent, purchase_count, days_since_last_purchase,
         purchase_frequency, avg_purchase_value, engagement_score, churn_probability,
         (1 - churn_probability) AS retention_rate
    
    // Calculate customer lifetime value - with reasonable multipliers
    WITH c, s, l, personas, devices, purchases, total_spent, purchase_count, days_since_last_purchase,
         purchase_frequency, avg_purchase_value, engagement_score, churn_probability, retention_rate,
         CASE 
           // For customers with purchase history - base prediction on actual spending
           WHEN total_spent > 0 THEN
             // Apply a reasonable multiplier based on retention likelihood
             CASE
               WHEN retention_rate > 0.8 THEN total_spent * 3.0  // High retention - 3x current value
               WHEN retention_rate > 0.6 THEN total_spent * 2.0  // Good retention - 2x current value
               ELSE total_spent * 1.3  // Lower retention - only slight increase expected
             END
           // For customers with no purchases yet
           ELSE
             // Estimate based on segment average purchase values
             CASE
               WHEN s.name = 'Tech Enthusiast' THEN 1200
               WHEN s.name = 'Professional' THEN 900
               WHEN s.name = 'Budget Shopper' THEN 400
               WHEN s.name = 'Gift Buyer' THEN 350
               WHEN s.name = 'Student' THEN 300
               ELSE 500
             END
         END AS predicted_lifetime_value
    
    // Predict next purchase date in days
    WITH c, s, l, personas, devices, purchases, total_spent, purchase_count, days_since_last_purchase,
         purchase_frequency, avg_purchase_value, engagement_score, churn_probability, 
         retention_rate, predicted_lifetime_value,
         CASE 
           WHEN purchase_frequency > 0 THEN
             // Base calculation on actual purchase frequency with realistic timelines
             (365.0 / purchase_frequency) * 
             // Apply modifiers based on recency and segment
             CASE 
               WHEN days_since_last_purchase < 30 THEN 0.85  // Recent purchasers likely to buy again sooner
               WHEN days_since_last_purchase > 90 THEN 1.25  // Dormant customers take longer
               ELSE 1.0
             END *
             CASE
               WHEN s.name = 'Tech Enthusiast' THEN 0.9
               WHEN s.name = 'Gift Buyer' THEN 1.5
               WHEN s.name = 'Professional' THEN 1.0
               WHEN s.name = 'Budget Shopper' THEN 1.3
               WHEN s.name = 'Student' THEN 1.2
               ELSE 1.0
             END *
             // Add subtle randomization (0.9-1.1)
             (0.9 + (rand() * 0.2))
           WHEN purchase_count = 0 THEN
             // For customers with no purchases yet - more realistic timelines
             CASE
               WHEN s.name = 'Tech Enthusiast' THEN 45 + rand() * 30  // 45-75 days
               WHEN s.name = 'Gift Buyer' THEN 120 + rand() * 60      // 120-180 days
               WHEN s.name = 'Professional' THEN 60 + rand() * 45     // 60-105 days
               WHEN s.name = 'Budget Shopper' THEN 90 + rand() * 60   // 90-150 days
               WHEN s.name = 'Student' THEN 75 + rand() * 45          // 75-120 days
               ELSE 90 + rand() * 60                                  // 90-150 days
             END
           ELSE 90 + rand() * 90  // Default 90-180 days
         END AS days_to_next_purchase
    
    RETURN c.customer_id AS id,
           c.first_name AS first_name,
           c.last_name AS last_name,
           c.email AS email,
           c.phone AS phone,
           c.age AS age,
           c.gender AS gender,
           c.registration_date AS registration_date,
           s.name AS segment,
           personas,
           devices,
           l.city AS city,
           l.country AS country,
           total_spent AS lifetime_value,
           predicted_lifetime_value AS predicted_value,
           churn_probability AS churn_rate,
           days_to_next_purchase AS next_purchase_days,
           days_since_last_purchase AS days_since_purchase,
           purchase_count AS num_purchases,
           engagement_score AS engagement
    """
    results = run_query(query, {"customer_id": customer_id})
    if results:
        return results[0]
    return None

def calculate_funnel_metrics(customer_ids=None):
    """Calculate funnel metrics for all customers or specific customers."""
    where_clause = ""
    if customer_ids:
        if isinstance(customer_ids, list):
            where_clause = "WHERE c.customer_id IN $customer_ids"
        else:
            where_clause = "WHERE c.customer_id = $customer_ids"
    
    query = f"""
    MATCH (c:Customer) {where_clause}
    
    // Awareness count - customers who viewed ads or visited site
    OPTIONAL MATCH (c)-[:VIEWS]->(:Advertisement)
    WITH c, count(c) > 0 AS had_awareness
    
    // Consideration count - customers who viewed products
    OPTIONAL MATCH (c)-[:VIEWS]->(:Product)
    WITH c, had_awareness, count(c) > 0 AS had_consideration
    
    // Intent count - customers who added to cart
    OPTIONAL MATCH (c)-[:ADDS_TO_CART]->(:Product)
    WITH c, had_awareness, had_consideration, count(c) > 0 AS had_intent
    
    // Conversion count - customers who purchased
    OPTIONAL MATCH (c)-[:PURCHASES]->(:Product)
    WITH c, had_awareness, had_consideration, had_intent, count(c) > 0 AS had_conversion
    
    // Retention count - customers who logged in or visited after purchase
    OPTIONAL MATCH (c)-[:LOGS_IN|VISITS]->()
    WHERE exists((c)-[:PURCHASES]->())
    WITH c, had_awareness, had_consideration, had_intent, had_conversion, count(c) > 0 AS had_retention
    
    // Advocacy count - customers who referred or wrote reviews
    OPTIONAL MATCH (c)-[:REFERS|WRITES|SHARES]->()
    WITH c, had_awareness, had_consideration, had_intent, had_conversion, had_retention, count(c) > 0 AS had_advocacy
    
    // Count totals for each stage
    RETURN 
        sum(CASE WHEN had_awareness THEN 1 ELSE 0 END) AS awareness_count,
        sum(CASE WHEN had_consideration THEN 1 ELSE 0 END) AS consideration_count,
        sum(CASE WHEN had_intent THEN 1 ELSE 0 END) AS intent_count,
        sum(CASE WHEN had_conversion THEN 1 ELSE 0 END) AS conversion_count,
        sum(CASE WHEN had_retention THEN 1 ELSE 0 END) AS retention_count,
        sum(CASE WHEN had_advocacy THEN 1 ELSE 0 END) AS advocacy_count,
        count(c) AS total_customers
    """
    
    results = run_query(query, {"customer_ids": customer_ids})
    if results:
        return results[0]
    return None

def get_journey_metrics(customer_id):
    """Get metrics for a specific customer journey."""
    query = """
    // Product views
    MATCH (c:Customer {customer_id: $customer_id})-[r:VIEWS]->(p:Product)
    WITH c, count(r) AS product_views
    
    // Ad views
    OPTIONAL MATCH (c)-[r2:VIEWS]->(a:Advertisement)
    WITH c, product_views, count(r2) AS ad_views
    
    // Cart additions
    OPTIONAL MATCH (c)-[r3:ADDS_TO_CART]->(p2:Product)
    WITH c, product_views, ad_views, count(r3) AS cart_additions
    
    // Cart abandons
    OPTIONAL MATCH (c)-[r4:ABANDONS]->()
    WITH c, product_views, ad_views, cart_additions, count(r4) AS cart_abandons
    
    // Purchases
    OPTIONAL MATCH (c)-[r5:PURCHASES]->(p3:Product)
    WITH c, product_views, ad_views, cart_additions, cart_abandons, 
         count(r5) AS purchase_count,
         sum(r5.quantity * r5.price) AS total_spent
    
    // Email metrics
    OPTIONAL MATCH (c)-[r6:RECEIVES]->(e:Email)
    WITH c, product_views, ad_views, cart_additions, cart_abandons,
         purchase_count, total_spent, count(r6) AS emails_received
         
    OPTIONAL MATCH (c)-[r7:OPENS]->(e2:Email)
    WITH c, product_views, ad_views, cart_additions, cart_abandons,
         purchase_count, total_spent, emails_received, count(r7) AS emails_opened
    
    // Support tickets
    OPTIONAL MATCH (c)-[r8:CREATES]->(t:Ticket)
    WITH c, product_views, ad_views, cart_additions, cart_abandons,
         purchase_count, total_spent, emails_received, emails_opened,
         count(r8) AS support_tickets
    
    // Referrals and reviews
    OPTIONAL MATCH (c)-[r9:REFERS]->()
    WITH c, product_views, ad_views, cart_additions, cart_abandons,
         purchase_count, total_spent, emails_received, emails_opened,
         support_tickets, count(r9) AS referrals
         
    OPTIONAL MATCH (c)-[r10:WRITES]->(r:Review)
    WITH c, product_views, ad_views, cart_additions, cart_abandons,
         purchase_count, total_spent, emails_received, emails_opened,
         support_tickets, referrals, count(r10) AS reviews
         
    // Social shares
    OPTIONAL MATCH (c)-[r11:SHARES]->()
    
    RETURN product_views, ad_views, cart_additions, cart_abandons,
           purchase_count, COALESCE(total_spent, 0) AS total_spent,
           emails_received, emails_opened,
           support_tickets, referrals, reviews,
           count(r11) AS social_shares
    """
    
    results = run_query(query, {"customer_id": customer_id})
    if results:
        return results[0]
    return None

def get_persona_comparison():
    """Get comparative metrics for all persona groups."""
    query = """
    MATCH (c:Customer)-[:BELONGS_TO]->(s:Segment)
    
    // Purchase metrics by segment
    OPTIONAL MATCH (c)-[p:PURCHASES]->()
    WITH s.name AS segment, 
         count(DISTINCT c) AS customer_count,
         count(p) AS purchase_count,
         sum(p.quantity * p.price) AS total_revenue,
         collect(c.customer_id) AS customer_ids
    
    // Calculate average time between purchases
    WITH segment, customer_count, purchase_count, total_revenue, customer_ids,
         CASE 
           WHEN purchase_count > 0 AND customer_count > 0 
           THEN round(purchase_count / customer_count * 365 / 30) 
           ELSE 0 
         END AS avg_days_between_purchases
    
    // Calculate churn rate by segment
    WITH segment, customer_count, purchase_count, total_revenue, customer_ids,
         avg_days_between_purchases,
         CASE 
           WHEN segment = 'Tech Enthusiast' THEN 0.11
           WHEN segment = 'Budget Shopper' THEN 0.32
           WHEN segment = 'Gift Buyer' THEN 0.48
           WHEN segment = 'Professional' THEN 0.21
           WHEN segment = 'Student' THEN 0.39
           ELSE 0.30
         END AS churn_rate
    
    // Calculate average customer lifespan in years based on churn rate
    WITH segment, customer_count, purchase_count, total_revenue, customer_ids,
         avg_days_between_purchases, churn_rate,
         CASE
           WHEN churn_rate > 0 THEN 1.0 / churn_rate
           ELSE 3.0
         END AS avg_lifespan_years
    
    // Calculate anomaly score (0-1) for each segment
    WITH segment, customer_count, purchase_count, total_revenue, customer_ids,
         avg_days_between_purchases, churn_rate, avg_lifespan_years,
         CASE
           WHEN segment = 'Tech Enthusiast' THEN 0.12
           WHEN segment = 'Budget Shopper' THEN 0.37
           WHEN segment = 'Gift Buyer' THEN 0.08
           WHEN segment = 'Professional' THEN 0.25
           WHEN segment = 'Student' THEN 0.44
           ELSE 0.2
         END AS anomaly_score
    
    // Calculate aggregate metrics
    RETURN segment,
           customer_count,
           purchase_count,
           total_revenue,
           CASE WHEN customer_count > 0 THEN toFloat(purchase_count) / customer_count ELSE 0 END AS purchases_per_customer,
           CASE WHEN purchase_count > 0 THEN total_revenue / purchase_count ELSE 0 END AS avg_order_value,
           CASE WHEN customer_count > 0 THEN total_revenue / customer_count ELSE 0 END AS revenue_per_customer,
           avg_days_between_purchases,
           churn_rate,
           avg_lifespan_years,
           anomaly_score
    ORDER BY segment
    """
    
    results = run_query(query)
    return results

def get_channel_effectiveness():
    """Analyze the effectiveness of different marketing channels."""
    query = """
    // Match channel entry events
    MATCH (c:Customer)-[r:COMES_FROM]->(ch:Channel)
    WITH ch.name AS channel, count(DISTINCT c) AS visitor_count
    
    // Match customers who converted from each channel
    OPTIONAL MATCH (c2:Customer)-[:COMES_FROM]->(ch2:Channel)
    WHERE ch2.name = channel
    AND exists((c2)-[:PURCHASES]->())
    WITH channel, visitor_count, count(DISTINCT c2) AS converter_count
    
    // Match revenue from each channel
    OPTIONAL MATCH (c3:Customer)-[:COMES_FROM]->(ch3:Channel)
    WHERE ch3.name = channel
    OPTIONAL MATCH (c3)-[p:PURCHASES]->()
    WITH channel, visitor_count, converter_count,
         sum(p.quantity * p.price) AS revenue
    
    // Calculate conversion rates and revenue metrics
    RETURN channel,
           visitor_count,
           converter_count,
           COALESCE(revenue, 0) AS revenue,
           CASE WHEN visitor_count > 0 
                THEN toFloat(converter_count) / visitor_count * 100
                ELSE 0 
           END AS conversion_rate,
           CASE WHEN converter_count > 0 
                THEN COALESCE(revenue, 0) / converter_count
                ELSE 0 
           END AS revenue_per_converter
    ORDER BY revenue DESC
    """
    
    results = run_query(query)
    return results

def get_product_performance():
    """Get metrics on product performance."""
    query = """
    MATCH (p:Product)<-[v:VIEWS]-(:Customer)
    WITH p, count(v) AS view_count
    
    OPTIONAL MATCH (p)<-[a:ADDS_TO_CART]-(:Customer)
    WITH p, view_count, count(a) AS cart_add_count
    
    OPTIONAL MATCH (p)<-[pur:PURCHASES]-(:Customer)
    
    RETURN p.id AS product_id,
           p.name AS product_name,
           p.category AS category,
           p.price AS price,
           view_count,
           cart_add_count,
           count(pur) AS purchase_count,
           sum(COALESCE(pur.quantity, 0)) AS units_sold,
           sum(COALESCE(pur.quantity * p.price, 0)) AS revenue,
           CASE WHEN view_count > 0 
                THEN toFloat(cart_add_count) / view_count * 100
                ELSE 0 
           END AS view_to_cart_rate,
           CASE WHEN cart_add_count > 0 
                THEN toFloat(count(pur)) / cart_add_count * 100
                ELSE 0 
           END AS cart_to_purchase_rate
    ORDER BY revenue DESC
    """
    
    results = run_query(query)
    return results

def get_customer_path_analysis(segment=None):
    """Analyze common paths through the customer journey."""
    segment_filter = ""
    if segment:
        segment_filter = "WHERE s.name = $segment"
        
    query = f"""
    // Find all customers in the segment
    MATCH (c:Customer)-[:BELONGS_TO]->(s:Segment)
    {segment_filter}
    
    // Get journey events in order
    MATCH (c)-[r]->(target)
    WHERE type(r) IN ['VIEWS', 'CLICKS_ON', 'VISITS', 'ADDS_TO_CART', 'PURCHASES']
    
    // Create path strings based on event types and targets
    WITH c, r, target,
         CASE 
           WHEN type(r) = 'VIEWS' AND labels(target)[0] = 'Advertisement' 
                THEN 'View Ad'
           WHEN type(r) = 'CLICKS_ON' AND labels(target)[0] = 'Advertisement' 
                THEN 'Click Ad'
           WHEN type(r) = 'VISITS' AND target.name = 'Home' 
                THEN 'Visit Homepage'
           WHEN type(r) = 'VISITS' AND target.name CONTAINS 'Category' 
                THEN 'Browse Category'
           WHEN type(r) = 'VIEWS' AND labels(target)[0] = 'Product' 
                THEN 'View Product'
           WHEN type(r) = 'ADDS_TO_CART' 
                THEN 'Add to Cart'
           WHEN type(r) = 'VISITS' AND target.name = 'Shopping Cart' 
                THEN 'View Cart'
           WHEN type(r) = 'VISITS' AND target.name = 'Checkout' 
                THEN 'Checkout'
           WHEN type(r) = 'PURCHASES' 
                THEN 'Purchase'
           ELSE type(r)
         END AS event_node,
         r.timestamp AS timestamp
    
    // Order events by timestamp for each customer
    WITH c, event_node, timestamp
    ORDER BY c.customer_id, timestamp
    
    // Collect events into paths
    WITH c.customer_id AS customer_id, collect(event_node) AS events
    
    // Get the first 4 events in each path
    WITH customer_id, 
         CASE WHEN size(events) >= 1 THEN events[0] ELSE null END AS event1,
         CASE WHEN size(events) >= 2 THEN events[1] ELSE null END AS event2,
         CASE WHEN size(events) >= 3 THEN events[2] ELSE null END AS event3,
         CASE WHEN size(events) >= 4 THEN events[3] ELSE null END AS event4
    
    // Filter out incomplete paths first
    WHERE event1 IS NOT NULL AND event2 IS NOT NULL AND 
          event3 IS NOT NULL AND event4 IS NOT NULL
          
    // Count occurrences of each path
    WITH event1 + ' → ' + event2 + ' → ' + event3 + ' → ' + event4 AS path, count(*) AS count
    RETURN path, count
    ORDER BY count DESC
    LIMIT 10
    """
    
    results = run_query(query, {"segment": segment})
    return results

# Define layout components
def create_header():
    """Create the dashboard header."""
    return dbc.Navbar(
        dbc.Container([
            html.A(
                dbc.Row([
                    dbc.Col(html.Img(src="/assets/logo.png", height="40px"), width="auto"),
                    dbc.Col(dbc.NavbarBrand("Marketing Ontology Platform Demo", className="ms-2")),
                ], align="center", className="g-0"),
                href="/",
                style={"textDecoration": "none"},
            ),
            dbc.NavbarToggler(id="navbar-toggler"),
            dbc.Collapse(
                dbc.Nav([
                    dbc.NavItem(dbc.NavLink("Customer Journeys", href="#")),
                    dbc.NavItem(dbc.NavLink("Executive Overview", href="#")),
                    dbc.NavItem(dbc.NavLink("Predictive Analytics", href="#")),
                    dbc.DropdownMenu(
                        [
                            dbc.DropdownMenuItem("Customer Journey Analysis", id="scenario-journey"),
                            dbc.DropdownMenuItem("Personalization Demo", id="scenario-personalization"),
                            dbc.DropdownMenuItem("Predictive Analytics", id="scenario-predictive"),
                            dbc.DropdownMenuItem("Marketing Optimization", id="scenario-optimization"),
                        ],
                        label="Demo Scenarios",
                        nav=True,
                    ),
                ], className="ms-auto", navbar=True),
                id="navbar-collapse",
                navbar=True,
            ),
        ]),
        color="primary",
        dark=True,
    )

def create_customer_selector(customers):
    """Create the customer selector component."""
    # Create flat list of customer options
    options = []
    for customer in sorted(customers, key=lambda x: x.get('name', '')):
        options.append({
            "label": f"{customer.get('name', 'Unknown')} ({customer['id']})",
            "value": customer["id"]
        })
    
    return dbc.Card([
        dbc.CardHeader("Select Customer to Explore"),
        dbc.CardBody([
            dcc.Dropdown(
                id="customer-selector",
                options=options,
                placeholder="Select a customer to view their journey...",
                value=customers[0]["id"] if customers else None,
                className="mb-3"
            ),
            html.Div(id="customer-info")
        ])
    ])

def create_funnel_visualization():
    """Create the marketing funnel visualization component."""
    return dbc.Card([
        dbc.CardHeader("Marketing Funnel"),
        dbc.CardBody([
            dcc.Graph(
                id="funnel-chart",
                config={"displayModeBar": False}
            )
        ])
    ])

def create_journey_timeline():
    """Create the customer journey timeline component."""
    return dbc.Card([
        dbc.CardHeader("Customer Journey Timeline"),
        dbc.CardBody([
            dbc.Row([
                dbc.Col([
                    dbc.Button("▶ Play Journey", id="play-journey", color="success", className="me-2"),
                    dbc.Button("⏹ Stop", id="stop-journey", color="danger", disabled=True),
                ], width="auto"),
                dbc.Col([
                    dcc.Slider(
                        id="journey-speed-slider",
                        min=0.5,
                        max=5,
                        step=0.5,
                        value=2,
                        marks={i: f"{i}x" for i in range(1, 6)},
                        tooltip={"placement": "bottom", "always_visible": False}
                    ),
                ], width=True),
                dbc.Col([
                    html.Div("Playback Speed", className="text-muted small")
                ], width="auto"),
            ], className="mb-3 align-items-center"),
            dcc.Graph(
                id="journey-timeline",
                config={"displayModeBar": False}
            ),
            dcc.Interval(
                id="journey-interval",
                interval=1000,  # 1 second by default
                n_intervals=0,
                disabled=True
            ),
            html.Div(id="journey-event-details", className="mt-3")
        ])
    ])

def create_journey_metrics():
    """Create the journey metrics component."""
    return dbc.Card([
        dbc.CardHeader("Journey Metrics"),
        dbc.CardBody([
            html.Div(id="journey-metrics-content"),
        ])
    ])

def create_path_analysis():
    """Create the path analysis component."""
    return dbc.Card([
        dbc.CardHeader("Common Journey Paths"),
        dbc.CardBody([
            dcc.Dropdown(
                id="path-segment-selector",
                options=[
                    {"label": "All Segments", "value": "all"},
                    {"label": "Tech Enthusiast", "value": "Tech Enthusiast"},
                    {"label": "Budget Shopper", "value": "Budget Shopper"},
                    {"label": "Gift Buyer", "value": "Gift Buyer"},
                    {"label": "Professional", "value": "Professional"},
                    {"label": "Student", "value": "Student"}
                ],
                value="all",
                className="mb-3"
            ),
            dcc.Graph(
                id="path-analysis-chart",
                config={"displayModeBar": False}
            )
        ])
    ])

def create_persona_comparison():
    """Create the persona comparison component."""
    return dbc.Card([
        dbc.CardHeader("Persona Comparison"),
        dbc.CardBody([
            dbc.Tabs([
                dbc.Tab(
                    dcc.Graph(id="persona-conversion-chart"),
                    label="Conversion Rates"
                ),
                dbc.Tab(
                    dcc.Graph(id="persona-revenue-chart"),
                    label="Revenue Metrics"
                )
            ])
        ])
    ])

def create_channel_analysis():
    """Create the channel analysis component."""
    return dbc.Card([
        dbc.CardHeader("Channel Effectiveness"),
        dbc.CardBody([
            dcc.Graph(id="channel-analysis-chart")
        ])
    ])

def create_product_performance():
    """Create the product performance component."""
    return dbc.Card([
        dbc.CardHeader("Product Performance"),
        dbc.CardBody([
            dbc.Tabs([
                dbc.Tab(
                    dcc.Graph(id="product-revenue-chart"),
                    label="Revenue"
                ),
                dbc.Tab(
                    dcc.Graph(id="product-conversion-chart"),
                    label="Conversion Rate"
                )
            ])
        ])
    ])

def create_personalization_demo():
    """Create the personalization demo component."""
    return dbc.Card([
        dbc.CardHeader("AI-Powered Customer Intelligence"),
        dbc.CardBody([
            html.Div(id="personalization-content"),
            dbc.Row([
                # Left Column: Customer Propensity Models
                dbc.Col([
                    html.H5("Propensity Models", className="mt-2 mb-3"),
                    dbc.Card([
                        dbc.CardHeader("Purchase Propensity"),
                        dbc.CardBody([
                            html.Div([
                                html.Span("Product Category Affinities", className="fw-bold"),
                                dbc.Progress(value=78, color="success", className="mb-1", style={"height": "8px"}),
                                html.Small("Computers (78%)", className="text-muted d-block"),
                                dbc.Progress(value=65, color="info", className="mb-1", style={"height": "8px"}),
                                html.Small("Audio (65%)", className="text-muted d-block"),
                                dbc.Progress(value=42, color="primary", className="mb-1", style={"height": "8px"}),
                                html.Small("Accessories (42%)", className="text-muted d-block"),
                                dbc.Progress(value=28, color="secondary", className="mb-1", style={"height": "8px"}),
                                html.Small("Wearables (28%)", className="text-muted d-block"),
                            ], className="mb-3"),
                            
                            html.Div([
                                html.Span("Price Sensitivity Model", className="fw-bold"),
                                dbc.Progress(value=38, color="success", className="mb-1", style={"height": "8px"}),
                                html.Small("Low Sensitivity: 38% - Focuses on features over price", className="text-muted d-block"),
                            ], className="mb-3"),
                            
                            html.Div([
                                html.Span("Discount Threshold Model", className="fw-bold"),
                                html.Div([
                                    html.Span("15%", className="badge bg-primary me-1"),
                                    html.Small("Minimum discount to motivate purchase", className="text-muted"),
                                ]),
                            ]),
                        ])
                    ], className="mb-3"),
                    
                    dbc.Card([
                        dbc.CardHeader("Churn Prevention"),
                        dbc.CardBody([
                            html.Div([
                                html.Span("Churn Risk Factors", className="fw-bold"),
                                dbc.Progress(value=25, color="danger", className="mb-1", style={"height": "8px"}),
                                html.Small("Engagement Decline (25%)", className="text-muted d-block"),
                                dbc.Progress(value=15, color="warning", className="mb-1", style={"height": "8px"}),
                                html.Small("Price Complaints (15%)", className="text-muted d-block"),
                                dbc.Progress(value=5, color="info", className="mb-1", style={"height": "8px"}),
                                html.Small("Competitor Interaction (5%)", className="text-muted d-block"),
                            ], className="mb-3"),
                            
                            html.Div([
                                html.Span("Retention Recommendation", className="fw-bold"),
                                html.Div([
                                    html.Span("Custom Bundle Offer", className="badge bg-success me-1"),
                                    html.Small("Premium Laptop + Wireless Earbuds", className="text-muted d-block"),
                                    html.Small("Effectiveness: 68%", className="text-muted d-block"),
                                ]),
                            ]),
                        ])
                    ]),
                ], md=6),
                
                # Right Column: Behavioral Analysis & NLP
                dbc.Col([
                    html.H5("Customer Journey Analytics", className="mt-2 mb-3"),
                    dbc.Card([
                        dbc.CardHeader("Behavioral Pattern Analysis"),
                        dbc.CardBody([
                            html.Div([
                                html.Span("Device Usage Pattern", className="fw-bold"),
                                html.Div([
                                    html.Small("65% Mobile → 35% Desktop", className="d-block"),
                                    html.Small("Primarily shops on mobile evenings 8-10pm", className="text-muted d-block"),
                                    html.Small("Research intensifies before purchase (3-4 product views)", className="text-muted d-block"),
                                ], className="mb-3"),
                                
                                html.Span("Purchase Cycle Analysis", className="fw-bold"),
                                html.Div([
                                    html.Small("Avg. time between purchases: 68 days", className="d-block"),
                                    html.Small("Upcoming purchase probability: 78% (next 14 days)", className="text-muted d-block")
                                ], className="mb-3"),
                                
                                html.Span("Feature Importance (ML Model)", className="fw-bold"),
                                dbc.Progress(value=35, color="danger", className="mb-1", style={"height": "8px"}),
                                dbc.Progress(value=28, color="warning", className="mb-1", style={"height": "8px"}),
                                dbc.Progress(value=22, color="info", className="mb-1", style={"height": "8px"}),
                                dbc.Progress(value=15, color="success", className="mb-1", style={"height": "8px"}),
                                html.Small([
                                    html.Span("Time since last visit ", className="text-danger"),
                                    html.Span("Price point ", className="text-warning"),
                                    html.Span("Previous category ", className="text-info"),
                                    html.Span("Email engagement", className="text-success")
                                ], className="text-muted d-block"),
                            ]),
                        ])
                    ], className="mb-3"),
                    
                    dbc.Card([
                        dbc.CardHeader("Sentiment & Interest Analysis"),
                        dbc.CardBody([
                            html.Div([
                                html.Span("Natural Language Processing Insights", className="fw-bold"),
                                html.Div([
                                    html.Small("Based on 6 customer interactions and feedback", className="text-muted d-block"),
                                    dbc.Progress(value=88, color="success", className="mb-1", style={"height": "8px"}),
                                    html.Small("Positive brand sentiment (88%)", className="text-muted d-block"),
                                    dbc.Progress(value=75, color="primary", className="mb-1", style={"height": "8px"}),
                                    html.Small("Interest in new premium laptop launch (75%)", className="text-muted d-block"),
                                    dbc.Progress(value=62, color="info", className="mb-1", style={"height": "8px"}),
                                    html.Small("Satisfaction with shipping speed (62%)", className="text-muted d-block"),
                                ]),
                            ]),
                        ])
                    ]),
                ], md=6),
            ], className="mt-3"),
        ])
    ])

# Main layout
app.layout = html.Div([
    create_header(),
    dbc.Container([
        dbc.Row([
            dbc.Col([
                html.H3("Marketing Ontology Platform Demo", className="mt-4"),
                html.P("Explore 15 simulated customer journeys through the marketing funnel of our mock business 'TechGear'"),
            ]),
        ]),
        
        # Store components for internal data
        dcc.Store(id="customer-journey-data"),
        dcc.Store(id="current-journey-index", data=0),
        
        dbc.Row([
            # Left column - Customer selection and info
            dbc.Col([
                create_customer_selector(get_customer_list()),
                html.Div(id="journey-metrics", className="mt-4"),
            ], md=4),
            
            # Right column - Customer journey visualization
            dbc.Col([
                create_funnel_visualization(),
            ], md=8),
        ], className="mt-4"),
        
        dbc.Row([
            dbc.Col([
                create_journey_timeline(),
            ], width=12),
        ], className="mt-4"),
        
        # Executive overview section
        html.Hr(className="mt-5"),
        html.H3("Executive Overview", className="mt-4"),
        
        dbc.Row([
            dbc.Col([
                create_path_analysis(),
            ], md=6),
            dbc.Col([
                create_persona_comparison(),
            ], md=6),
        ], className="mt-4"),
        
        dbc.Row([
            dbc.Col([
                create_channel_analysis(),
            ], md=6),
            dbc.Col([
                create_product_performance(),
            ], md=6),
        ], className="mt-4"),
        
        # Personalization and predictive demo section
        html.Hr(className="mt-5"),
        html.H3("Personalization & Predictive Analytics", className="mt-4"),
        
        dbc.Row([
            dbc.Col([
                create_personalization_demo(),
            ], width=12),
        ], className="mt-4"),
        
        html.Div(style={"height": "50px"}),  # Bottom spacing
        
    ], fluid=True),
])

# Callbacks
@callback(
    Output("customer-info", "children"),
    Input("customer-selector", "value")
)
def update_customer_info(customer_id):
    """Update the customer information panel when a customer is selected."""
    if not customer_id:
        return html.Div("Select a customer to view their details")
    
    # Get customer details
    customer = get_customer_details(customer_id)
    if not customer:
        return html.Div("Customer not found")
    
    # Format the date if it exists
    if customer.get("registration_date"):
        try:
            registration_date = datetime.datetime.fromisoformat(customer["registration_date"]).strftime("%b %d, %Y")
        except (TypeError, ValueError):
            registration_date = "Not available"
    else:
        registration_date = "Not available"
    
    # Format predictive metrics
    lifetime_value = customer.get("lifetime_value", 0)
    predicted_value = customer.get("predicted_value", 0)
    churn_rate = customer.get("churn_rate", 0)
    next_purchase_days = customer.get("next_purchase_days", 0)
    
    # Create customer info display
    return dbc.Card([
        dbc.CardBody([
            html.H4(f"{customer['first_name']} {customer['last_name']}", className="mb-3"),
            
            # Standard info
            dbc.Row([
                dbc.Col([html.Strong("Customer ID:"), html.Span(f" {customer['id']}")], width=12, className="mb-2"),
                dbc.Col([html.Strong("Email:"), html.Span(f" {customer['email']}")], width=12, className="mb-2"),
                dbc.Col([html.Strong("Phone:"), html.Span(f" {customer['phone']}")], width=12, className="mb-2"),
                dbc.Col([html.Strong("Age:"), html.Span(f" {customer['age']}")], width=6, className="mb-2"),
                dbc.Col([html.Strong("Gender:"), html.Span(f" {customer['gender']}")], width=6, className="mb-2"),
                dbc.Col([html.Strong("Location:"), html.Span(f" {customer['city']}, {customer['country']}")], width=12, className="mb-2"),
                dbc.Col([html.Strong("Registration:"), html.Span(f" {registration_date}")], width=12, className="mb-2"),
                dbc.Col([
                    html.Strong("Segment: "),
                    html.Span(
                        customer['segment'],
                        style={
                            "background-color": COLORS["personas"].get(customer['segment'], "#3498DB"),
                            "color": "white",
                            "padding": "3px 8px",
                            "border-radius": "12px",
                            "font-size": "0.8rem"
                        }
                    )
                ], width=12, className="mb-2"),
                dbc.Col([
                    html.Strong("Personas: "),
                    html.Div([
                        html.Span(
                            persona,
                            style={
                                "background-color": "#7F8C8D",
                                "color": "white",
                                "padding": "2px 6px",
                                "border-radius": "10px",
                                "font-size": "0.75rem",
                                "margin-right": "5px"
                            }
                        ) for persona in customer['personas']
                    ], style={"display": "inline-block"})
                ], width=12, className="mb-2"),
                dbc.Col([
                    html.Strong("Devices: "),
                    html.Div([
                        html.Span(
                            device.replace("_", " ").title(),
                            style={
                                "background-color": "#BDC3C7",
                                "color": "#2C3E50",
                                "padding": "2px 6px",
                                "border-radius": "10px",
                                "font-size": "0.75rem",
                                "margin-right": "5px"
                            }
                        ) for device in customer['devices']
                    ], style={"display": "inline-block"})
                ], width=12, className="mb-2"),
            ]),
            
            # Advanced predictive metrics section
            html.Hr(),
            html.H5("Predictive Analytics", className="mt-3 mb-3"),
            
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.H3(f"${lifetime_value:.2f}", className="text-center text-primary mb-0"),
                            html.P("Current Value", className="text-center text-muted small mb-0"),
                            html.Small(f"Based on {customer.get('num_purchases', 0)} purchases", 
                                      className="d-block text-center text-muted")
                        ])
                    ], className="h-100 border-primary")
                ], md=6, className="mb-2"),
                
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.H3(f"${predicted_value:.2f}", className="text-center text-success mb-0"),
                            html.P("Predicted Lifetime Value", className="text-center text-muted small mb-0"),
                            html.Small(f"Engagement: {customer.get('engagement', 0):.2f}", 
                                      className="d-block text-center text-muted")
                        ])
                    ], className="h-100 border-success")
                ], md=6, className="mb-2"),
                
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.H3(f"{churn_rate*100:.1f}%", 
                                   className=f"text-center text-{'danger' if churn_rate > 0.5 else 'warning' if churn_rate > 0.3 else 'success'} mb-0"),
                            html.P("Churn Probability", className="text-center text-muted small mb-0"),
                            html.Small(f"Last purchase: {customer.get('days_since_purchase', 'N/A')} days ago", 
                                      className="d-block text-center text-muted")
                        ])
                    ], className="h-100")
                ], md=6, className="mb-2"),
                
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.H3(f"{int(next_purchase_days)} days", className="text-center text-info mb-0"),
                            html.P("Next Purchase ETA", className="text-center text-muted small mb-0"),
                            html.Small(f"Based on {customer.get('segment')} behavior patterns", 
                                      className="d-block text-center text-muted")
                        ])
                    ], className="h-100")
                ], md=6, className="mb-2"),
            ])
        ])
    ])

@callback(
    [Output("customer-journey-data", "data"),
     Output("journey-metrics", "children")],
    Input("customer-selector", "value")
)
def load_customer_journey(customer_id):
    """Load the journey data for the selected customer."""
    if not customer_id:
        return None, None
    
    # Get customer journey data
    journey = get_customer_journey(customer_id)
    
    # Get journey metrics
    metrics = get_journey_metrics(customer_id)
    
    # Create metrics card
    metrics_card = create_journey_metrics()
    
    if metrics:
        metrics_content = [
            dbc.Row([
                dbc.Col([
                    html.Div([
                        html.H3(metrics["product_views"], className="mb-0"),
                        html.Div("Product Views", className="text-muted small")
                    ], className="border rounded p-2 text-center mb-2")
                ], md=6),
                dbc.Col([
                    html.Div([
                        html.H3(metrics["cart_additions"], className="mb-0"),
                        html.Div("Cart Additions", className="text-muted small")
                    ], className="border rounded p-2 text-center mb-2")
                ], md=6),
                dbc.Col([
                    html.Div([
                        html.H3(metrics["purchase_count"], className="mb-0"),
                        html.Div("Purchases", className="text-muted small")
                    ], className="border rounded p-2 text-center mb-2")
                ], md=6),
                dbc.Col([
                    html.Div([
                        html.H3(f"${metrics['total_spent']:.2f}", className="mb-0"),
                        html.Div("Total Spent", className="text-muted small")
                    ], className="border rounded p-2 text-center mb-2")
                ], md=6),
                dbc.Col([
                    html.Div([
                        html.H3(metrics["emails_opened"], className="mb-0"),
                        html.Div(f"Emails Opened ({metrics['emails_received']} received)", className="text-muted small")
                    ], className="border rounded p-2 text-center mb-2")
                ], md=6),
                dbc.Col([
                    html.Div([
                        html.H3(metrics["reviews"] + metrics["referrals"], className="mb-0"),
                        html.Div(f"Reviews + Referrals", className="text-muted small")
                    ], className="border rounded p-2 text-center mb-2")
                ], md=6),
            ])
        ]
    else:
        metrics_content = [html.Div("No metrics available")]
    
    metrics_card.children[1].children = metrics_content
    
    return journey, metrics_card

@callback(
    Output("funnel-chart", "figure"),
    [Input("customer-selector", "value"),
     Input("customer-journey-data", "data")]
)
def update_funnel_chart(customer_id, journey_data):
    """Update the funnel visualization for the selected customer."""
    if not customer_id:
        # Show overall funnel for all customers
        funnel_data = calculate_funnel_metrics()
    else:
        # Show funnel for selected customer
        funnel_data = calculate_funnel_metrics(customer_id)
    
    if not funnel_data:
        # Fallback empty funnel
        funnel_data = {
            "awareness_count": 0,
            "consideration_count": 0,
            "intent_count": 0,
            "conversion_count": 0,
            "retention_count": 0,
            "advocacy_count": 0,
            "total_customers": 0
        }
    
    # Create funnel data
    funnel_values = [
        funnel_data["awareness_count"],
        funnel_data["consideration_count"],
        funnel_data["intent_count"],
        funnel_data["conversion_count"],
        funnel_data["retention_count"],
        funnel_data["advocacy_count"]
    ]
    
    # Add percentages to stage names
    percentages = []
    for i, value in enumerate(funnel_values):
        if i == 0 or funnel_values[i-1] == 0:
            pct = 100.0 if i == 0 else 0.0
        else:
            pct = (value / funnel_values[i-1]) * 100.0
        percentages.append(f"{pct:.1f}%")
    
    stage_names = [
        f"Awareness ({percentages[0]})",
        f"Consideration ({percentages[1]})",
        f"Intent ({percentages[2]})",
        f"Conversion ({percentages[3]})",
        f"Retention ({percentages[4]})",
        f"Advocacy ({percentages[5]})"
    ]
    
    # Create colors list from stage colors
    colors = [COLORS["stages"][stage] for stage in FUNNEL_STAGES]
    
    # Create the funnel chart
    fig = go.Figure(go.Funnel(
        y=stage_names,
        x=funnel_values,
        textinfo="value+percent initial",
        marker={"color": colors},
        connector={"line": {"color": "gray", "width": 1}}
    ))
    
    fig.update_layout(
        title={
            "text": "Marketing Funnel Progression" + (f" for {customer_id}" if customer_id else ""),
            "x": 0.5,
            "xanchor": "center"
        },
        margin={"t": 80, "l": 150, "r": 10, "b": 10},
        height=400
    )
    
    return fig

@callback(
    Output("journey-timeline", "figure"),
    [Input("customer-journey-data", "data"),
     Input("current-journey-index", "data")]
)
def update_journey_timeline(journey_data, current_index):
    """Update the journey timeline visualization."""
    if not journey_data:
        # Return empty figure if no journey data
        fig = go.Figure()
        fig.update_layout(
            title="No journey data available",
            height=400,
            xaxis={"title": "Timeline"},
            yaxis={"title": "Event"}
        )
        return fig
    
    # Create a time-ordered sequence of events
    events = sorted(journey_data, key=lambda x: x["timestamp"])
    
    # Define colors for each stage
    stage_colors = COLORS["stages"]
    
    # Create horizontal timeline
    fig = go.Figure()
    
    # Add events to timeline
    for i, event in enumerate(events):
        is_active = i <= current_index if current_index is not None else True
        opacity = 1.0 if is_active else 0.3
        
        # Convert timestamp to datetime
        timestamp = datetime.datetime.fromisoformat(event["timestamp"])
        
        # Generate event description
        event_type = event["event_type"]
        target_type = event["target_type"]
        target_name = event["target_name"]
        stage = event["funnel_stage"]
        
        description = f"{event_type} {target_type}"
        if target_name and target_name != "Unknown":
            description += f": {target_name}"
        
        # Add event to timeline
        fig.add_trace(go.Scatter(
            x=[timestamp],
            y=[0],
            mode="markers+text",
            marker={
                "size": 20,
                "color": stage_colors.get(stage, "#3498DB"),
                "line": {"width": 2, "color": "white"},
                "opacity": opacity
            },
            text=[event_type[0]],  # First letter of event type
            textposition="middle center",
            textfont={"color": "white", "size": 10},
            hoverinfo="text",
            hovertext=f"{timestamp.strftime('%b %d, %H:%M')}<br>{description}<br>Stage: {stage.title()}",
            name=event_type
        ))
    
    # Add connecting line for all events
    if events:
        timestamps = [datetime.datetime.fromisoformat(event["timestamp"]) for event in events]
        fig.add_trace(go.Scatter(
            x=timestamps,
            y=[0] * len(timestamps),
            mode="lines",
            line={"color": "gray", "width": 1},
            hoverinfo="skip",
            showlegend=False
        ))
    
    # Add highlighting for current event
    if events and current_index is not None and 0 <= current_index < len(events):
        current_event = events[current_index]
        current_timestamp = datetime.datetime.fromisoformat(current_event["timestamp"])
        fig.add_trace(go.Scatter(
            x=[current_timestamp],
            y=[0],
            mode="markers",
            marker={
                "size": 30,
                "color": "rgba(0,0,0,0)",
                "line": {"width": 3, "color": "yellow"}
            },
            hoverinfo="skip",
            showlegend=False
        ))
    
    # Add tick marks for each week
    if events:
        first_date = datetime.datetime.fromisoformat(events[0]["timestamp"])
        last_date = datetime.datetime.fromisoformat(events[-1]["timestamp"])
        
        # Add 10% padding on both sides
        date_range = (last_date - first_date).total_seconds()
        padding = datetime.timedelta(seconds=date_range * 0.1)
        
        first_date -= padding
        last_date += padding
        
        # Generate ticks every week
        current_date = first_date - datetime.timedelta(days=first_date.weekday())  # Start on a Monday
        ticks = []
        while current_date <= last_date:
            ticks.append(current_date)
            current_date += datetime.timedelta(days=7)
    
    # Update layout
    fig.update_layout(
        title="Customer Journey Timeline",
        height=300,
        xaxis={
            "title": "Timeline",
            "showgrid": True,
            "zeroline": False,
            "showline": True,
            "linecolor": "gray",
            "showticklabels": True,
            "tickformat": "%b %d"
        },
        yaxis={
            "title": "",
            "showgrid": False,
            "zeroline": False,
            "showline": False,
            "showticklabels": False,
            "range": [-0.5, 0.5]
        },
        legend={
            "orientation": "h",
            "yanchor": "bottom",
            "y": 1.02,
            "xanchor": "right",
            "x": 1
        },
        margin={"t": 50, "l": 20, "r": 20, "b": 50},
        hovermode="closest"
    )
    
    # Add annotation for funnel stages
    for stage, color in stage_colors.items():
        fig.add_trace(go.Scatter(
            x=[],
            y=[],
            mode="markers",
            marker={"size": 10, "color": color},
            name=stage.title(),
            showlegend=True
        ))
    
    return fig

@callback(
    Output("journey-event-details", "children"),
    [Input("journey-timeline", "clickData"),
     Input("customer-journey-data", "data")]
)
def update_event_details(click_data, journey_data):
    """Update the event details when a point is clicked on the timeline."""
    if not click_data or not journey_data:
        return html.Div("Click an event on the timeline to see details")
    
    point_index = click_data["points"][0]["pointIndex"]
    trace_index = click_data["points"][0]["curveNumber"]
    
    # Skip if clicking on the connecting line (usually trace index 15)
    if trace_index >= len(journey_data):
        return html.Div("Click an event marker to see details")
    
    # Get the clicked event
    events = sorted(journey_data, key=lambda x: x["timestamp"])
    if point_index >= len(events):
        return html.Div("Event not found")
    
    event = events[point_index]
    
    # Format timestamp
    timestamp = datetime.datetime.fromisoformat(event["timestamp"])
    formatted_time = timestamp.strftime("%B %d, %Y at %I:%M %p")
    
    # Create event details card
    return dbc.Card([
        dbc.CardHeader(f"Event: {event['event_type']} {event['target_type']}"),
        dbc.CardBody([
            dbc.Row([
                dbc.Col([
                    html.Div([
                        html.Strong("Event Type: "),
                        html.Span(event["event_type"])
                    ], className="mb-2"),
                    html.Div([
                        html.Strong("Target Type: "),
                        html.Span(event["target_type"])
                    ], className="mb-2"),
                    html.Div([
                        html.Strong("Target Name: "),
                        html.Span(event["target_name"])
                    ], className="mb-2"),
                    html.Div([
                        html.Strong("Funnel Stage: "),
                        html.Span(event["funnel_stage"].title())
                    ], className="mb-2"),
                ], md=6),
                dbc.Col([
                    html.Div([
                        html.Strong("Timestamp: "),
                        html.Span(formatted_time)
                    ], className="mb-2"),
                    html.Div([
                        html.Strong("Properties: "),
                        html.Pre(
                            json.dumps(event.get("properties", {}), indent=2), 
                            style={"background-color": "#f8f9fa", "padding": "10px", "border-radius": "5px", "max-height": "100px", "overflow-y": "auto", "font-size": "0.8rem"}
                        ) if event.get("properties") else html.Span("None")
                    ], className="mb-2"),
                ], md=6),
            ])
        ])
    ])

@callback(
    [Output("journey-interval", "disabled"),
     Output("journey-interval", "interval"),
     Output("play-journey", "disabled"),
     Output("stop-journey", "disabled")],
    [Input("play-journey", "n_clicks"),
     Input("stop-journey", "n_clicks"),
     Input("journey-speed-slider", "value")],
    prevent_initial_call=True
)
def control_journey_playback(play_clicks, stop_clicks, speed):
    """Control the journey playback when play/stop buttons are clicked."""
    ctx = dash.callback_context
    if not ctx.triggered:
        return True, 1000, False, True
    
    trigger_id = ctx.triggered[0]["prop_id"].split(".")[0]
    
    if trigger_id == "play-journey":
        # Calculate interval based on speed (500ms to 2000ms)
        interval = int(1000 / speed)
        return False, interval, True, False
    
    elif trigger_id == "stop-journey":
        return True, 1000, False, True
    
    elif trigger_id == "journey-speed-slider":
        # Update the speed but maintain current playback state
        interval = int(1000 / speed)
        is_playing = not dash.callback_context.states["journey-interval.disabled"]
        return not is_playing, interval, is_playing, not is_playing
    
    # Default fallback
    return True, 1000, False, True

@callback(
    Output("current-journey-index", "data"),
    [Input("journey-interval", "n_intervals"),
     Input("customer-journey-data", "data"),
     Input("journey-timeline", "clickData")],
    [State("current-journey-index", "data"),
     State("journey-interval", "disabled")],
    prevent_initial_call=True
)
def update_journey_playback(n_intervals, journey_data, click_data, current_index, interval_disabled):
    """Update the current journey event during playback."""
    ctx = dash.callback_context
    if not ctx.triggered or not journey_data:
        return 0
    
    trigger_id = ctx.triggered[0]["prop_id"].split(".")[0]
    
    if trigger_id == "journey-timeline" and click_data:
        # User clicked a point on the timeline
        point_index = click_data["points"][0]["pointIndex"]
        trace_index = click_data["points"][0]["curveNumber"]
        
        # Only update if clicking an event marker, not the connecting line
        if trace_index < len(journey_data):
            return point_index
    
    elif trigger_id == "journey-interval" and not interval_disabled:
        # Playback interval triggered
        if current_index is None:
            current_index = 0
        else:
            current_index += 1
        
        # Reset to beginning if reached the end
        if current_index >= len(journey_data):
            current_index = 0
        
        return current_index
    
    elif trigger_id == "customer-journey-data":
        # New customer selected, reset index
        return 0
    
    # Return existing index as fallback
    return current_index if current_index is not None else 0

@callback(
    Output("path-analysis-chart", "figure"),
    Input("path-segment-selector", "value")
)
def update_path_analysis(segment):
    """Update the path analysis chart based on selected segment."""
    if segment == "all":
        segment = None
    
    # Get path analysis data
    paths = get_customer_path_analysis(segment)
    
    if not paths:
        # Return empty figure if no data
        fig = go.Figure()
        fig.update_layout(
            title="No path data available",
            height=400
        )
        return fig
    
    # Sort paths by count
    paths = sorted(paths, key=lambda x: x["count"], reverse=True)
    
    # Prepare data for horizontal bar chart
    fig = go.Figure(go.Bar(
        y=[p["path"] for p in paths],
        x=[p["count"] for p in paths],
        orientation="h",
        marker_color=COLORS["secondary"]
    ))
    
    fig.update_layout(
        title=f"Most Common Customer Journeys{' for ' + segment if segment else ''}",
        height=500,
        xaxis_title="Number of Customers",
        yaxis_title="Journey Path",
        margin={"l": 20, "r": 20, "t": 50, "b": 20}
    )
    
    return fig

@callback(
    [Output("persona-conversion-chart", "figure"),
     Output("persona-revenue-chart", "figure")],
    Input("persona-conversion-chart", "id")  # Just to trigger on load
)
def update_persona_comparison(dummy):
    """Update the persona comparison charts."""
    # Get persona comparison data
    personas = get_persona_comparison()
    
    if not personas:
        # Return empty figures if no data
        empty_fig = go.Figure()
        empty_fig.update_layout(
            title="No persona data available",
            height=400
        )
        return empty_fig, empty_fig
    
    # Create color map for personas
    colors = [COLORS["personas"].get(p["segment"], "#3498DB") for p in personas]
    
    # Conversion rate chart with anomaly overlay
    conversion_fig = go.Figure()
    
    # Add purchases per customer bars
    conversion_fig.add_trace(go.Bar(
        x=[p["segment"] for p in personas],
        y=[p["purchases_per_customer"] for p in personas],
        marker_color=colors,
        text=[f"{p['purchases_per_customer']:.2f}" for p in personas],
        textposition="auto",
        name="Purchases per Customer"
    ))
    
    # Add churn rate line
    conversion_fig.add_trace(go.Scatter(
        x=[p["segment"] for p in personas],
        y=[p["churn_rate"] * 2 for p in personas],  # Scale to make visible on same chart
        name="Churn Rate",
        mode="lines+markers",
        line={"color": "#E74C3C", "width": 2, "dash": "dot"},
        marker={"size": 8, "symbol": "diamond"},
        yaxis="y2",
        text=[f"Churn: {p['churn_rate']*100:.1f}%<br>Avg Lifespan: {p['avg_lifespan_years']:.1f} years" for p in personas],
        hoverinfo="text+name"
    ))
    
    # Add anomaly scores as bubble markers
    conversion_fig.add_trace(go.Scatter(
        x=[p["segment"] for p in personas],
        y=[p["anomaly_score"] * 2 for p in personas],  # Scale to make visible
        mode="markers",
        marker=dict(
            size=[p["anomaly_score"] * 40 + 10 for p in personas],
            color=["rgba(255,0,0,0.5)" if p["anomaly_score"] > 0.3 else "rgba(255,165,0,0.3)" for p in personas],
            line=dict(color="rgba(156, 165, 196, 0.8)", width=1)
        ),
        name="Anomaly Score",
        text=[f"Anomaly Score: {p['anomaly_score']:.2f}" for p in personas],
        hoverinfo="text+name"
    ))
    
    conversion_fig.update_layout(
        title="Customer Behavior Metrics by Persona",
        height=400,
        yaxis={"title": "Avg. Purchases per Customer"},
        yaxis2={"title": "Churn Rate", "overlaying": "y", "side": "right", "range": [0, 2]},
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "xanchor": "right", "x": 1},
        margin={"l": 20, "r": 20, "t": 50, "b": 20}
    )
    
    # Revenue chart - with additional metrics
    revenue_fig = go.Figure()
    
    # Add average order value bars
    revenue_fig.add_trace(go.Bar(
        x=[p["segment"] for p in personas],
        y=[p["avg_order_value"] for p in personas],
        name="Avg. Order Value",
        marker_color=colors,
        text=[f"${p['avg_order_value']:.2f}" for p in personas],
        textposition="auto"
    ))
    
    # Add revenue per customer line
    revenue_fig.add_trace(go.Scatter(
        x=[p["segment"] for p in personas],
        y=[p["revenue_per_customer"] for p in personas],
        name="Revenue per Customer",
        mode="lines+markers",
        line={"color": "#E74C3C", "width": 3},
        marker={"size": 10},
        yaxis="y2",
        text=[f"${p['revenue_per_customer']:.2f}" for p in personas],
        hoverinfo="text+name"
    ))
    
    # Add customer lifetime value estimate (revenue per customer * lifespan years)
    revenue_fig.add_trace(go.Scatter(
        x=[p["segment"] for p in personas],
        y=[p["revenue_per_customer"] * p["avg_lifespan_years"] for p in personas],
        name="Est. Lifetime Value",
        mode="markers",
        marker={
            "symbol": "star",
            "size": 14,
            "color": "#2ECC71",
            "line": {"width": 1, "color": "white"}
        },
        yaxis="y2",
        text=[f"LTV: ${p['revenue_per_customer'] * p['avg_lifespan_years']:.2f}<br>({p['avg_lifespan_years']:.1f} years)" for p in personas],
        hoverinfo="text+name"
    ))
    
    revenue_fig.update_layout(
        title="Revenue & Lifetime Value by Persona",
        height=400,
        yaxis={"title": "Avg. Order Value ($)"},
        yaxis2={"title": "Customer Value ($)", "overlaying": "y", "side": "right"},
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "xanchor": "right", "x": 1},
        margin={"l": 20, "r": 20, "t": 50, "b": 20}
    )
    
    return conversion_fig, revenue_fig

@callback(
    Output("channel-analysis-chart", "figure"),
    Input("channel-analysis-chart", "id")  # Just to trigger on load
)
def update_channel_analysis(dummy):
    """Update the channel analysis chart."""
    # Get channel effectiveness data
    channels = get_channel_effectiveness()
    
    if not channels:
        # Return empty figure if no data
        fig = go.Figure()
        fig.update_layout(
            title="No channel data available",
            height=400
        )
        return fig
    
    # Sort channels by revenue
    channels = sorted(channels, key=lambda x: x["revenue"], reverse=True)
    
    # Create dual-axis chart
    fig = go.Figure()
    
    # Add revenue bars
    fig.add_trace(go.Bar(
        x=[c["channel"].replace("_", " ").title() for c in channels],
        y=[c["revenue"] for c in channels],
        name="Revenue",
        marker_color=COLORS["secondary"],
        text=[f"${c['revenue']:.2f}" for c in channels],
        textposition="auto"
    ))
    
    # Add conversion rate line
    fig.add_trace(go.Scatter(
        x=[c["channel"].replace("_", " ").title() for c in channels],
        y=[c["conversion_rate"] for c in channels],
        name="Conversion Rate",
        mode="lines+markers",
        line={"color": "#E74C3C", "width": 3},
        marker={"size": 10},
        yaxis="y2",
        text=[f"{c['conversion_rate']:.1f}%" for c in channels],
        hoverinfo="text+name"
    ))
    
    fig.update_layout(
        title="Channel Performance",
        height=400,
        xaxis_title="Channel",
        yaxis={"title": "Revenue ($)"},
        yaxis2={
            "title": "Conversion Rate (%)",
            "overlaying": "y",
            "side": "right",
            "range": [0, max([c["conversion_rate"] for c in channels]) * 1.2 if channels else 100]
        },
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "xanchor": "right", "x": 1},
        margin={"l": 20, "r": 20, "t": 50, "b": 20}
    )
    
    return fig

@callback(
    [Output("product-revenue-chart", "figure"),
     Output("product-conversion-chart", "figure")],
    Input("product-revenue-chart", "id")  # Just to trigger on load
)
def update_product_performance(dummy):
    """Update the product performance charts."""
    # Get product performance data
    products = get_product_performance()
    
    if not products:
        # Return empty figures if no data
        empty_fig = go.Figure()
        empty_fig.update_layout(
            title="No product data available",
            height=400
        )
        return empty_fig, empty_fig
    
    # Sort products by revenue for first chart
    revenue_products = sorted(products, key=lambda x: x["revenue"], reverse=True)[:10]  # Top 10
    
    # Sort products by cart to purchase rate for second chart
    conversion_products = sorted(products, key=lambda x: x["cart_to_purchase_rate"], reverse=True)[:10]  # Top 10
    
    # Revenue chart
    revenue_fig = go.Figure(go.Bar(
        x=[p["product_name"] for p in revenue_products],
        y=[p["revenue"] for p in revenue_products],
        marker_color=COLORS["secondary"],
        text=[f"${p['revenue']:.2f}" for p in revenue_products],
        textposition="auto",
        hovertemplate="%{x}<br>Revenue: $%{y:.2f}<br>Units Sold: %{customdata}<extra></extra>",
        customdata=[[p["units_sold"]] for p in revenue_products]
    ))
    
    revenue_fig.update_layout(
        title="Top Products by Revenue",
        height=400,
        xaxis={"title": "Product"},
        yaxis={"title": "Revenue ($)"},
        margin={"l": 20, "r": 20, "t": 50, "b": 120}
    )
    
    # Make x-axis labels at an angle to fit better
    revenue_fig.update_xaxes(tickangle=45)
    
    # Conversion rate chart (cart to purchase)
    conversion_fig = go.Figure(go.Bar(
        x=[p["product_name"] for p in conversion_products],
        y=[p["cart_to_purchase_rate"] for p in conversion_products],
        marker_color="#E74C3C",
        text=[f"{p['cart_to_purchase_rate']:.1f}%" for p in conversion_products],
        textposition="auto",
        hovertemplate="%{x}<br>Cart to Purchase: %{y:.1f}%<br>Views to Cart: %{customdata}%<extra></extra>",
        customdata=[[f"{p['view_to_cart_rate']:.1f}"] for p in conversion_products]
    ))
    
    conversion_fig.update_layout(
        title="Top Products by Cart-to-Purchase Rate",
        height=400,
        xaxis={"title": "Product"},
        yaxis={"title": "Cart to Purchase Rate (%)"},
        margin={"l": 20, "r": 20, "t": 50, "b": 120}
    )
    
    # Make x-axis labels at an angle to fit better
    conversion_fig.update_xaxes(tickangle=45)
    
    return revenue_fig, conversion_fig

@callback(
    Output("personalization-content", "children"),
    Input("customer-selector", "value")
)
def update_personalization_demo(customer_id):
    """Update the personalization demo for the selected customer."""
    if not customer_id:
        return html.Div("Select a customer to view personalization recommendations")
    
    # Get customer details
    customer = get_customer_details(customer_id)
    if not customer:
        return html.Div("Customer not found")
    
    # Get customer's purchase history
    query = """
    MATCH (c:Customer {customer_id: $customer_id})-[p:PURCHASES]->(prod:Product)
    RETURN prod.id AS product_id, prod.name AS product_name, 
           prod.category AS category, prod.price AS price,
           p.timestamp AS purchase_date
    ORDER BY p.timestamp DESC
    """
    
    purchases = run_query(query, {"customer_id": customer_id})
    
    # Get product view history
    query = """
    MATCH (c:Customer {customer_id: $customer_id})-[v:VIEWS]->(prod:Product)
    WHERE NOT exists((c)-[:PURCHASES]->(prod))
    RETURN prod.id AS product_id, prod.name AS product_name, 
           prod.category AS category, prod.price AS price,
           v.timestamp AS view_date
    ORDER BY v.timestamp DESC
    LIMIT 5
    """
    
    views = run_query(query, {"customer_id": customer_id})
    
    # Get abandoned cart items
    query = """
    MATCH (c:Customer {customer_id: $customer_id})-[a:ADDS_TO_CART]->(prod:Product)
    WHERE NOT exists((c)-[:PURCHASES]->(prod))
    RETURN prod.id AS product_id, prod.name AS product_name, 
           prod.category AS category, prod.price AS price,
           a.timestamp AS cart_date
    ORDER BY a.timestamp DESC
    LIMIT 3
    """
    
    abandoned = run_query(query, {"customer_id": customer_id})
    
    # Get product recommendations based on purchase history and segment
    query = """
    MATCH (c:Customer {customer_id: $customer_id})-[:BELONGS_TO]->(s:Segment)
    MATCH (other:Customer)-[:BELONGS_TO]->(s)
    WHERE other.customer_id <> c.customer_id
    
    // Products purchased by similar customers
    MATCH (other)-[:PURCHASES]->(rec_prod:Product)
    WHERE NOT exists((c)-[:PURCHASES]->(rec_prod))
    
    WITH rec_prod, count(distinct other) AS frequency
    ORDER BY frequency DESC
    LIMIT 5
    
    RETURN rec_prod.id AS product_id, rec_prod.name AS product_name,
           rec_prod.category AS category, rec_prod.price AS price,
           frequency
    """
    
    recommendations = run_query(query, {"customer_id": customer_id})
    
    # Build the personalization demo UI
    return html.Div([
        dbc.Row([
            dbc.Col([
                html.H4(f"Personalized Experience for {customer['first_name']} {customer['last_name']}"),
                html.P([
                    "Segment: ",
                    html.Span(
                        customer['segment'],
                        style={
                            "background-color": COLORS["personas"].get(customer['segment'], "#3498DB"),
                            "color": "white",
                            "padding": "3px 8px",
                            "border-radius": "12px",
                            "font-size": "0.8rem"
                        }
                    ),
                    " | Personas: ",
                    html.Span(", ".join(customer['personas']))
                ]),
            ]),
        ]),
        
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader("Recommended Products"),
                    dbc.CardBody([
                        html.P("Based on purchase history, browsing behavior, and similar customers"),
                        dbc.ListGroup([
                            dbc.ListGroupItem([
                                dbc.Row([
                                    dbc.Col([
                                        html.Div(rec["product_name"], style={"font-weight": "bold"}),
                                        html.Small(f"Category: {rec['category']} | ${rec['price']:.2f}")
                                    ], md=9),
                                    dbc.Col([
                                        dbc.Button("Add to Cart", color="success", size="sm", className="w-100")
                                    ], md=3)
                                ])
                            ]) for rec in recommendations
                        ]) if recommendations else html.Div("No recommendations available")
                    ])
                ]),
            ], md=6),
            
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader("Abandoned Cart Reminder"),
                    dbc.CardBody([
                        html.P("Items you were interested in but didn't purchase"),
                        dbc.ListGroup([
                            dbc.ListGroupItem([
                                dbc.Row([
                                    dbc.Col([
                                        html.Div(item["product_name"], style={"font-weight": "bold"}),
                                        html.Small(f"Category: {item['category']} | ${item['price']:.2f}")
                                    ], md=9),
                                    dbc.Col([
                                        dbc.Button("Buy Now", color="warning", size="sm", className="w-100")
                                    ], md=3)
                                ])
                            ]) for item in abandoned
                        ]) if abandoned else html.Div("No abandoned items")
                    ])
                ]),
            ], md=6),
        ], className="mt-3"),
        
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader("Personalized Marketing Messages"),
                    dbc.CardBody([
                        dbc.ListGroup([
                            dbc.ListGroupItem([
                                html.Div("🎯 Limited Time Offer", style={"font-weight": "bold"}),
                                html.P([
                                    f"Based on your interest in {purchases[0]['category'] if purchases else 'technology'}, ",
                                    f"we're offering a special 15% discount on selected items just for you."
                                ]) if purchases else html.P("Discover our latest promotions customized for your preferences.")
                            ], className="mb-2"),
                            dbc.ListGroupItem([
                                html.Div("📅 Upcoming Events", style={"font-weight": "bold"}),
                                html.P([
                                    f"Join us for our {customer['segment']} showcase event next week. ",
                                    f"We'll be demonstrating products that match your interests!"
                                ])
                            ], className="mb-2"),
                            dbc.ListGroupItem([
                                html.Div("🔔 Restock Alert", style={"font-weight": "bold"}),
                                html.P([
                                    f"Good news! The item you viewed recently, ",
                                    html.Strong(views[0]["product_name"] if views else "Premium Laptop"),
                                    ", is back in stock and available for purchase."
                                ]) if views else html.P("Items matching your preferences are now back in stock.")
                            ], className="mb-2"),
                        ])
                    ])
                ]),
            ], md=12),
        ], className="mt-3"),
        
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader("Next Best Actions"),
                    dbc.CardBody([
                        dbc.Row([
                            dbc.Col([
                                dbc.Button([
                                    html.I(className="fas fa-tags me-2"),
                                    "View Special Offers"
                                ], color="primary", className="w-100 mb-2")
                            ], md=4),
                            dbc.Col([
                                dbc.Button([
                                    html.I(className="fas fa-heart me-2"),
                                    "Create Wishlist"
                                ], color="secondary", className="w-100 mb-2")
                            ], md=4),
                            dbc.Col([
                                dbc.Button([
                                    html.I(className="fas fa-bell me-2"),
                                    "Set Price Alert"
                                ], color="info", className="w-100 mb-2")
                            ], md=4),
                        ]),
                    ])
                ]),
            ], md=12),
        ], className="mt-3"),
    ], className="p-3")

# Run the server
if __name__ == "__main__":
    try:
        # Check if demo_data directory exists and customers.json is present
        data_file = os.path.join("/home/cabdru/marketing/demo/demo_data", "customers.json")
        if not os.path.exists(data_file):
            print(f"Warning: Demo data file not found at {data_file}")
            print("Please run generate_demo_data.py first")
            
        # Start the server
        app.run_server(debug=True, port=8051)
    
    finally:
        # Close Neo4j connection
        if driver:
            driver.close()