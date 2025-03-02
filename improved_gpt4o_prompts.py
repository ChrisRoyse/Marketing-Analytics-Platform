#!/usr/bin/env python3
"""
Improved GPT-4o prompting system using the SPARC framework for marketing ontology.
This helps the model better distinguish between node properties and relationships.
"""

import json
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('gpt4o_prompts.log')
    ]
)

def generate_sparc_prompt(customer_data):
    """
    Generate a prompt using the SPARC framework:
    - Situation: Context about the marketing ontology task
    - Problem: Distinguishing between properties and entities
    - Analysis: Request to analyze the customer data 
    - Response: Expected format of the response
    - Confirmation: Verification steps to ensure proper modeling
    """
    prompt = """
    # Situation
    You're building a behavior pattern ontology for marketing in Neo4j. This graph database will track detailed customer journeys through marketing funnels to identify patterns and churn points.

    # Problem
    Currently, many customer attributes are incorrectly being stored as properties of Customer nodes, when they should be separate nodes with relationships to the Customer. We need to distinguish between:
    1. True properties that belong directly on the Customer node
    2. Entities that should be their own nodes with relationships to Customer nodes

    # Analysis
    Analyze this customer data record:
    ```
    {customer_data}
    ```

    For each attribute, determine whether it should be:
    
    A. A PROPERTY of the Customer node (inherent attributes like name, ID, lifetime value, timestamps)
    B. A SEPARATE NODE with a relationship to the Customer (journey elements like devices, browsers, emails, locations, stages)
    
    The following must ALWAYS be modeled as separate nodes with relationships:
    - Device type (desktop/mobile)
    - Browser
    - Email (both as entity and communication channel)  
    - Geographic location
    - Current behavior stage
    - Operating system
    - Persona
    - Satisfaction score
    - Segment
    - Newsletter subscription status
    - Preferred channel
    - Churn status (modeled with a CHURNED_AT relationship to a funnel stage)

    # Response
    Return a structured JSON with:

    1. "customer_properties": An object with attributes that should remain as properties on the Customer node
    2. "related_entities": An array of objects representing separate nodes with relationships to the Customer, each with:
       - "type": Entity node type (Device, Browser, Email, Location, etc.)
       - "id": A unique identifier for this entity
       - "relationship": The type of relationship (USES, PREFERS, LOCATED_IN, etc.)
       - "properties": Any properties that belong to this entity node
       - "direction": "outgoing" (Customer→Entity) or "incoming" (Entity→Customer)

    # Confirmation
    Verify that:
    1. Only inherent customer attributes are listed as properties
    2. All journey-related elements are modeled as separate nodes
    3. Each relationship has a meaningful type that describes the connection
    4. Timestamps are preserved as relationship properties when relevant
    5. Any funnel stage data is properly modeled for tracking customer journey progression
    """
    
    # Format the prompt with the customer data
    formatted_prompt = prompt.format(customer_data=json.dumps(customer_data, indent=2))
    return formatted_prompt

def analyze_mock_gpt4o_response(customer_data):
    """
    Simulate GPT-4o response for testing purposes.
    In a real implementation, this would call the OpenAI API.
    
    This function demonstrates what a well-formed response would look like.
    """
    # Extract key fields from customer data
    customer_id = customer_data.get('customer_id', 'UNKNOWN')
    name = customer_data.get('name', '')
    email = customer_data.get('email', '')
    device = customer_data.get('device', 'unknown')
    browser = customer_data.get('browser', 'unknown')
    location = customer_data.get('location', '')
    segment = customer_data.get('segment', '')
    persona = customer_data.get('persona', '')
    behavior_stage = customer_data.get('current_behavior_stage', '')
    operating_system = customer_data.get('operating_system', '')
    
    # Simulate GPT-4o thinking about properties vs. entities
    mock_response = {
        "customer_properties": {
            "name": name,
            "lifetime_value": customer_data.get('lifetime_value', 0),
            "last_seen": customer_data.get('last_seen', ''),
            "is_churned": customer_data.get('is_churned', False)
        },
        "related_entities": [
            {
                "type": "Device",
                "id": device,
                "relationship": "USES",
                "properties": {
                    "timestamp": datetime.now().isoformat(),
                    "is_primary": True
                },
                "direction": "outgoing"
            },
            {
                "type": "Browser",
                "id": browser,
                "relationship": "ACCESSES_WITH",
                "properties": {
                    "timestamp": datetime.now().isoformat(),
                    "version": customer_data.get('browser_version', 'unknown')
                },
                "direction": "outgoing"
            },
            {
                "type": "Email",
                "id": email,
                "relationship": "HAS_EMAIL",
                "properties": {
                    "timestamp": datetime.now().isoformat(),
                    "is_primary": True,
                    "verified": True
                },
                "direction": "outgoing"
            },
            {
                "type": "Location",
                "id": location,
                "relationship": "LOCATED_IN",
                "properties": {
                    "timestamp": datetime.now().isoformat()
                },
                "direction": "outgoing"
            },
            {
                "type": "OperatingSystem",
                "id": operating_system,
                "relationship": "USES",
                "properties": {
                    "timestamp": datetime.now().isoformat()
                },
                "direction": "outgoing"
            },
            {
                "type": "Segment",
                "id": segment,
                "relationship": "BELONGS_TO",
                "properties": {
                    "timestamp": datetime.now().isoformat()
                },
                "direction": "outgoing"
            },
            {
                "type": "Persona",
                "id": persona,
                "relationship": "HAS_PERSONA",
                "properties": {
                    "timestamp": datetime.now().isoformat()
                },
                "direction": "outgoing"
            },
            {
                "type": "BehaviorStage",
                "id": behavior_stage,
                "relationship": "AT_STAGE",
                "properties": {
                    "timestamp": datetime.now().isoformat()
                },
                "direction": "outgoing"
            }
        ]
    }
    
    # Add newsletter subscription if present
    if customer_data.get('subscribed_to_newsletter'):
        mock_response["related_entities"].append({
            "type": "Newsletter",
            "id": "company_newsletter",
            "relationship": "SUBSCRIBED_TO",
            "properties": {
                "timestamp": datetime.now().isoformat(),
                "opt_in_source": customer_data.get('newsletter_source', 'website')
            },
            "direction": "outgoing"
        })
    
    # Add preferred channel if present
    if customer_data.get('preferred_channel'):
        mock_response["related_entities"].append({
            "type": "Channel",
            "id": customer_data.get('preferred_channel'),
            "relationship": "PREFERS",
            "properties": {
                "timestamp": datetime.now().isoformat()
            },
            "direction": "outgoing"
        })
    
    # Add satisfaction score if present
    if customer_data.get('satisfaction_score'):
        mock_response["related_entities"].append({
            "type": "SatisfactionScore",
            "id": f"score_{customer_data.get('satisfaction_score')}",
            "relationship": "RATED",
            "properties": {
                "timestamp": datetime.now().isoformat(),
                "value": customer_data.get('satisfaction_score'),
                "source": customer_data.get('satisfaction_source', 'survey')
            },
            "direction": "outgoing"
        })
    
    return mock_response

def process_customer_with_gpt4o(customer_data):
    """
    Process customer data with GPT-4o to determine properties vs. entities.
    
    In a production implementation, this would:
    1. Generate the SPARC prompt
    2. Call OpenAI API with the prompt
    3. Parse and validate the response
    4. Return the structured data for Neo4j import
    
    For simulation, we'll use our mock response.
    """
    try:
        # 1. Generate the prompt
        prompt = generate_sparc_prompt(customer_data)
        logging.info(f"Generated SPARC prompt for customer {customer_data.get('customer_id')}")
        
        # 2. In production: Call OpenAI API
        # For simulation, use mock response
        logging.info(f"Simulating GPT-4o analysis for customer {customer_data.get('customer_id')}")
        gpt4o_response = analyze_mock_gpt4o_response(customer_data)
        
        # 3. Validate the response
        if not validate_gpt4o_response(gpt4o_response):
            logging.warning(f"Invalid GPT-4o response for customer {customer_data.get('customer_id')}")
            return None
        
        logging.info(f"Successfully processed customer {customer_data.get('customer_id')} with GPT-4o")
        return gpt4o_response
        
    except Exception as e:
        logging.error(f"Error processing customer with GPT-4o: {e}")
        return None

def validate_gpt4o_response(response):
    """Validate that the GPT-4o response has the expected structure."""
    if not isinstance(response, dict):
        return False
        
    if "customer_properties" not in response or not isinstance(response["customer_properties"], dict):
        return False
        
    if "related_entities" not in response or not isinstance(response["related_entities"], list):
        return False
        
    for entity in response["related_entities"]:
        if not all(k in entity for k in ["type", "id", "relationship", "properties"]):
            return False
    
    return True

if __name__ == "__main__":
    # Example usage
    test_customer = {
        "customer_id": "CUST001",
        "name": "John Doe",
        "email": "john.doe@example.com",
        "device": "mobile",
        "browser": "Chrome",
        "location": "New York",
        "lifetime_value": 1250.75,
        "last_seen": "2024-10-26T14:30:00",
        "segment": "High-Value",
        "persona": "Tech Enthusiast",
        "current_behavior_stage": "Consideration",
        "operating_system": "iOS",
        "is_churned": False,
        "subscribed_to_newsletter": True,
        "newsletter_source": "checkout",
        "preferred_channel": "email",
        "satisfaction_score": 4.5,
        "satisfaction_source": "post-purchase survey"
    }
    
    result = process_customer_with_gpt4o(test_customer)
    if result:
        print(json.dumps(result, indent=2))