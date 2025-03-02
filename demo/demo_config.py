#!/usr/bin/env python
"""
Configuration settings for the Marketing Ontology Platform Demo.

This module defines configuration settings and constants used across
the demo components, ensuring consistency in data formatting and
visualization styling.
"""

import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Neo4j connection parameters
NEO4J_URI = os.getenv("NEO4J_URI", "bolt://172.19.160.1:7687")
NEO4J_USER = os.getenv("NEO4J_USERNAME", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "#1Moneymaker")
NEO4J_DATABASE = os.getenv("NEO4J_DATABASE", "marketing")

# Dashboard configuration
DASHBOARD_PORT = int(os.getenv("DASHBOARD_PORT", "8050"))
DASHBOARD_DEBUG = os.getenv("DASHBOARD_DEBUG", "True").lower() in ("true", "1", "t")
DASHBOARD_TITLE = "Marketing Ontology Platform Demo"

# Demo business configuration
BUSINESS_NAME = "TechGear"
BUSINESS_DESCRIPTION = "E-commerce retailer selling consumer electronics and accessories"
BUSINESS_LOGO_PATH = "/assets/logo.png"  # Path relative to the assets directory

# Directory paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEMO_DIR = os.path.join(BASE_DIR, "demo")
DATA_DIR = os.path.join(DEMO_DIR, "demo_data")
ASSETS_DIR = os.path.join(DEMO_DIR, "demo_assets")

# Funnel stages
FUNNEL_STAGES = ["awareness", "consideration", "intent", "conversion", "retention", "advocacy"]

# Persona groups (segments)
PERSONA_GROUPS = [
    "Tech Enthusiast",
    "Budget Shopper",
    "Gift Buyer",
    "Professional",
    "Student"
]

# Color scheme for visualization consistency
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

# Demo scenarios for guided walkthroughs
DEMO_SCENARIOS = {
    "executive_overview": {
        "title": "Executive Overview",
        "description": "High-level business metrics and KPIs across customer segments",
        "visualizations": ["funnel_overview", "persona_comparison", "channel_effectiveness"],
        "customer_ids": []  # No specific customer focus
    },
    "customer_journey": {
        "title": "Customer Journey Analysis",
        "description": "Detailed analysis of individual customer journeys",
        "visualizations": ["journey_timeline", "event_details", "path_analysis"],
        "customer_ids": ["CUST001", "CUST004", "CUST007", "CUST010", "CUST013"]  # One from each segment
    },
    "personalization": {
        "title": "Personalization Demo",
        "description": "Context-aware personalization examples",
        "visualizations": ["personalization_recommendations", "abandoned_cart", "next_best_action"],
        "customer_ids": ["CUST002", "CUST005", "CUST009"]  # Selected engaged customers
    },
    "predictive_analytics": {
        "title": "Predictive Analytics",
        "description": "Predictive models for customer behavior",
        "visualizations": ["churn_prediction", "clv_forecast", "next_purchase"],
        "customer_ids": ["CUST003", "CUST006", "CUST011", "CUST015"]  # Mix of likely to churn and high CLV
    },
    "marketing_optimization": {
        "title": "Marketing Optimization",
        "description": "Channel effectiveness and budget allocation insights",
        "visualizations": ["channel_roi", "campaign_attribution", "budget_allocation"],
        "customer_ids": []  # No specific customer focus
    }
}

# Event type to funnel stage mapping
EVENT_TO_STAGE_MAPPING = {
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

# Define event types and their display names
EVENT_TYPES = {
    "VIEWS": "Views",
    "CLICKS_ON": "Clicks On",
    "COMES_FROM": "Comes From",
    "VISITS": "Visits",
    "ADDS_TO_CART": "Adds to Cart",
    "ABANDONS": "Abandons",
    "PURCHASES": "Purchases",
    "RECEIVES": "Receives",
    "OPENS": "Opens",
    "CREATES": "Creates",
    "LOGS_IN": "Logs In",
    "WRITES": "Writes",
    "REFERS": "Refers",
    "SHARES": "Shares",
    "CHURNED_AT": "Churned At"
}

# Journey path events for path analysis
JOURNEY_PATH_EVENTS = [
    "View Ad",
    "Click Ad",
    "Visit Homepage",
    "Browse Category",
    "View Product",
    "Add to Cart",
    "View Cart",
    "Checkout",
    "Purchase"
]

# Initialize any required directories
def init_directories():
    """Create required directories if they don't exist."""
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(ASSETS_DIR, exist_ok=True)

# Call initialization function
init_directories()