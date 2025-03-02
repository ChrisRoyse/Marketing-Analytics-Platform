#!/usr/bin/env python3
"""
Dynamic Customer Analyzer for Phase 3 of the marketing ontology project.
This module enables real-time customer journey analysis and visualization
with only a customer ID as input.
"""

import os
import json
import argparse
import logging
import datetime
import time
from pathlib import Path
from neo4j import GraphDatabase
from collections import defaultdict

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('dynamic_analyzer.log')
    ]
)

class DynamicCustomerAnalyzer:
    """
    Dynamic analysis tool that requires only a customer ID to generate
    comprehensive insights and visualizations about their marketing journey.
    """
    
    def __init__(self, uri=None, username=None, password=None, database=None):
        """Initialize the DynamicCustomerAnalyzer with Neo4j connection details."""
        self.uri = uri or os.getenv('NEO4J_URI', 'bolt://localhost:7687')
        self.username = username or os.getenv('NEO4J_USERNAME', 'neo4j')
        self.password = password or os.getenv('NEO4J_PASSWORD', 'neo4j')
        self.database = database or os.getenv('NEO4J_DATABASE', 'neo4j')
        self.driver = None
        self.output_dir = "customer_insights"
        
    def connect(self):
        """Connect to the Neo4j database."""
        try:
            self.driver = GraphDatabase.driver(
                self.uri, 
                auth=(self.username, self.password)
            )
            # Test the connection
            with self.driver.session(database=self.database) as session:
                result = session.run("RETURN 1 AS test")
                record = result.single()
                if record and record["test"] == 1:
                    logging.info("Successfully connected to Neo4j database")
                    return True
                else:
                    logging.error("Failed to verify Neo4j connection")
                    return False
        except Exception as e:
            logging.error(f"Failed to connect to Neo4j: {e}")
            return False
            
    def close(self):
        """Close the Neo4j connection."""
        if self.driver:
            self.driver.close()
            logging.info("Neo4j connection closed")
            
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
            logging.error(f"Error running query: {e}")
            return None

    def validate_customer_id(self, customer_id):
        """Verify if the customer ID exists in the database."""
        query = """
        MATCH (c:Customer {customer_id: $customer_id})
        RETURN c.customer_id AS id
        """
        result = self.run_query(query, {"customer_id": customer_id})
        return bool(result and len(result) > 0)

    def get_customer_profile(self, customer_id):
        """Get comprehensive customer profile information."""
        # Primary customer information
        customer_query = """
        MATCH (c:Customer {customer_id: $customer_id})
        RETURN c
        """
        
        # Related entities and relationships
        entities_query = """
        MATCH (c:Customer {customer_id: $customer_id})-[r]-(n)
        WHERE NOT n:FunnelStage
        RETURN labels(n)[0] as entity_type, 
               n.id as entity_id, 
               type(r) as relationship_type, 
               properties(n) as entity_properties,
               properties(r) as relationship_properties
        """
        
        # Funnel stage information
        funnel_query = """
        MATCH (c:Customer {customer_id: $customer_id})-[r]-(stage:FunnelStage)
        RETURN stage.name as stage_name, 
               type(r) as relationship_type,
               properties(r) as relationship_properties
        """
        
        # Execute all queries
        customer_data = self.run_query(customer_query, {"customer_id": customer_id})
        entity_data = self.run_query(entities_query, {"customer_id": customer_id})
        funnel_data = self.run_query(funnel_query, {"customer_id": customer_id})
        
        if not customer_data:
            logging.error(f"No data found for customer ID: {customer_id}")
            return None
        
        # Organize the data into a structured profile
        profile = {
            "customer_id": customer_id,
            "basic_info": customer_data[0]["c"] if customer_data else {},
            "devices": [],
            "browsers": [],
            "locations": [],
            "channels": [],
            "segments": [],
            "personas": [],
            "interactions": [],
            "email_activity": [],
            "funnel_status": funnel_data if funnel_data else []
        }
        
        # Process entity data
        entity_counts = defaultdict(int)
        
        for entity in entity_data or []:
            entity_type = entity.get("entity_type", "")
            entity_counts[entity_type] += 1
            
            # Add to appropriate category
            if entity_type == "Device":
                profile["devices"].append({
                    "id": entity.get("entity_id"),
                    "relationship": entity.get("relationship_type"),
                    "properties": entity.get("entity_properties", {}),
                    "rel_properties": entity.get("relationship_properties", {})
                })
            elif entity_type == "Browser":
                profile["browsers"].append({
                    "id": entity.get("entity_id"),
                    "relationship": entity.get("relationship_type"),
                    "properties": entity.get("entity_properties", {}),
                    "rel_properties": entity.get("relationship_properties", {})
                })
            elif entity_type == "Location":
                profile["locations"].append({
                    "id": entity.get("entity_id"),
                    "relationship": entity.get("relationship_type"),
                    "properties": entity.get("entity_properties", {}),
                    "rel_properties": entity.get("relationship_properties", {})
                })
            elif entity_type == "Channel":
                profile["channels"].append({
                    "id": entity.get("entity_id"),
                    "relationship": entity.get("relationship_type"),
                    "properties": entity.get("entity_properties", {}),
                    "rel_properties": entity.get("relationship_properties", {})
                })
            elif entity_type == "Segment":
                profile["segments"].append({
                    "id": entity.get("entity_id"),
                    "relationship": entity.get("relationship_type"),
                    "properties": entity.get("entity_properties", {}),
                    "rel_properties": entity.get("relationship_properties", {})
                })
            elif entity_type == "Persona":
                profile["personas"].append({
                    "id": entity.get("entity_id"),
                    "relationship": entity.get("relationship_type"),
                    "properties": entity.get("entity_properties", {}),
                    "rel_properties": entity.get("relationship_properties", {})
                })
            elif entity_type == "Email":
                profile["email_activity"].append({
                    "id": entity.get("entity_id"),
                    "relationship": entity.get("relationship_type"),
                    "properties": entity.get("entity_properties", {}),
                    "rel_properties": entity.get("relationship_properties", {})
                })
            elif entity_type in ["Product", "Advertisement", "Page", "Cart", "Content"]:
                profile["interactions"].append({
                    "type": entity_type,
                    "id": entity.get("entity_id"),
                    "relationship": entity.get("relationship_type"),
                    "properties": entity.get("entity_properties", {}),
                    "rel_properties": entity.get("relationship_properties", {})
                })
        
        # Sort interactions by timestamp if available
        profile["interactions"].sort(
            key=lambda x: x.get("rel_properties", {}).get("timestamp", ""),
            reverse=True
        )
        
        # Add entity counts summary
        profile["entity_summary"] = dict(entity_counts)
        
        return profile

    def get_journey_timeline(self, customer_id):
        """Get a chronological timeline of customer journey events."""
        journey_query = """
        MATCH (c:Customer {customer_id: $customer_id})-[r]->(n)
        WHERE r.timestamp IS NOT NULL
        RETURN r.timestamp as timestamp,
               type(r) as event_type,
               r.action as action,
               labels(n)[0] as target_type,
               n.id as target_id,
               properties(r) as properties
        ORDER BY r.timestamp
        """
        
        journey_data = self.run_query(journey_query, {"customer_id": customer_id})
        
        if not journey_data:
            logging.warning(f"No journey data found for customer {customer_id}")
            return []
        
        # Format into a timeline
        timeline = []
        for event in journey_data:
            formatted_event = {
                "timestamp": event.get("timestamp"),
                "event_type": event.get("event_type"),
                "action": event.get("action"),
                "target_type": event.get("target_type"),
                "target_id": event.get("target_id"),
                "details": event.get("properties", {}),
                "description": self._generate_event_description(event)
            }
            timeline.append(formatted_event)
        
        return timeline
    
    def _generate_event_description(self, event):
        """Generate a human-readable description of a journey event."""
        action = event.get("action", "interacted with")
        target_type = event.get("target_type", "item")
        target_id = event.get("target_id", "")
        
        # Map actions to more readable descriptions
        action_map = {
            "viewed_ad": "viewed advertisement",
            "clicked_ad": "clicked on advertisement",
            "visited_website": "visited webpage",
            "viewed_product": "viewed product",
            "added_to_cart": "added product to cart",
            "abandoned_cart": "abandoned shopping cart",
            "purchased": "purchased product",
            "wrote_review": "wrote review for",
            "referred_friend": "referred a friend",
            "viewed_email": "opened email",
            "clicked_email": "clicked link in email",
            "subscribed_to_newsletter": "subscribed to newsletter",
            "unsubscribed_from_newsletter": "unsubscribed from newsletter"
        }
        
        readable_action = action_map.get(action, action)
        return f"{readable_action} {target_id}"

    def get_conversion_funnel_status(self, customer_id):
        """Analyze customer's current position in the conversion funnel."""
        funnel_query = """
        MATCH (c:Customer {customer_id: $customer_id})
        
        // Check awareness stage
        OPTIONAL MATCH (c)-[awareness:VIEWS|CLICKS_ON]->(awareness_node:Advertisement)
        WITH c, count(awareness) > 0 as reached_awareness
        
        // Check consideration stage
        OPTIONAL MATCH (c)-[consideration:VISITS|VIEWS]->(consideration_node)
        WHERE consideration_node:Page OR consideration_node:Product
        WITH c, reached_awareness, count(consideration) > 0 as reached_consideration
        
        // Check intent stage
        OPTIONAL MATCH (c)-[intent:ADDS_TO_CART]->(intent_node:Product)
        WITH c, reached_awareness, reached_consideration, count(intent) > 0 as reached_intent
        
        // Check conversion stage
        OPTIONAL MATCH (c)-[conversion:PURCHASES]->(conversion_node:Product)
        WITH c, reached_awareness, reached_consideration, reached_intent, count(conversion) > 0 as reached_conversion
        
        // Check retention stage
        OPTIONAL MATCH (c)-[retention:INTERACTS_WITH]->(retention_node)
        WHERE retention_node:Content AND retention_node.id CONTAINS 'post_purchase'
        WITH c, reached_awareness, reached_consideration, reached_intent, reached_conversion, count(retention) > 0 as reached_retention
        
        // Check advocacy stage
        OPTIONAL MATCH (c)-[advocacy:REFERS|COMMENTS_ON]->(advocacy_node)
        
        RETURN 
            CASE WHEN reached_awareness THEN 'Awareness' ELSE NULL END as awareness,
            CASE WHEN reached_consideration THEN 'Consideration' ELSE NULL END as consideration,
            CASE WHEN reached_intent THEN 'Intent' ELSE NULL END as intent,
            CASE WHEN reached_conversion THEN 'Conversion' ELSE NULL END as conversion,
            CASE WHEN reached_retention THEN 'Retention' ELSE NULL END as retention,
            CASE WHEN count(advocacy) > 0 THEN 'Advocacy' ELSE NULL END as advocacy
        """
        
        funnel_data = self.run_query(funnel_query, {"customer_id": customer_id})
        
        if not funnel_data or len(funnel_data) == 0:
            logging.warning(f"No funnel data found for customer {customer_id}")
            return {"current_stage": "Unknown", "completed_stages": [], "all_stages": []}
        
        # Process funnel stages
        stages = ["Awareness", "Consideration", "Intent", "Conversion", "Retention", "Advocacy"]
        completed_stages = []
        current_stage = "Unknown"
        
        data = funnel_data[0]
        for stage in stages:
            stage_lower = stage.lower()
            if data.get(stage_lower):
                completed_stages.append(stage)
                current_stage = stage
        
        # Check if churned
        churn_query = """
        MATCH (c:Customer {customer_id: $customer_id})-[r:CHURNED_AT]->(:FunnelStage)
        RETURN r.timestamp as churn_time, r.reason as churn_reason, r.previous_stage as previous_stage
        """
        
        churn_data = self.run_query(churn_query, {"customer_id": customer_id})
        
        result = {
            "current_stage": current_stage,
            "completed_stages": completed_stages,
            "all_stages": stages,
            "has_churned": bool(churn_data and len(churn_data) > 0)
        }
        
        if result["has_churned"]:
            result["churn_details"] = churn_data[0]
        
        return result

    def get_similar_customers(self, customer_id, limit=5):
        """Find similar customers based on common segments, personas, and behavior."""
        similar_query = """
        MATCH (c:Customer {customer_id: $customer_id})
        
        // Find segments this customer belongs to
        OPTIONAL MATCH (c)-[:BELONGS_TO]->(segment:Segment)
        WITH c, collect(segment) as segments
        
        // Find personas this customer has
        OPTIONAL MATCH (c)-[:HAS_PERSONA]->(persona:Persona)
        WITH c, segments, collect(persona) as personas
        
        // Find other customers with similar segments or personas
        MATCH (other:Customer)
        WHERE other.customer_id <> $customer_id
        
        OPTIONAL MATCH (other)-[:BELONGS_TO]->(other_segment:Segment)
        WHERE other_segment IN segments
        
        OPTIONAL MATCH (other)-[:HAS_PERSONA]->(other_persona:Persona)
        WHERE other_persona IN personas
        
        WITH other, 
             count(DISTINCT other_segment) as segment_matches,
             count(DISTINCT other_persona) as persona_matches,
             size(segments) as total_segments,
             size(personas) as total_personas
        
        // Calculate similarity score (higher is better)
        WITH other, 
             CASE 
                WHEN total_segments > 0 THEN toFloat(segment_matches) / total_segments 
                ELSE 0 
             END + 
             CASE 
                WHEN total_personas > 0 THEN toFloat(persona_matches) / total_personas
                ELSE 0
             END as similarity_score
        
        WHERE similarity_score > 0
        RETURN other.customer_id as customer_id, 
               other.name as name,
               similarity_score
        ORDER BY similarity_score DESC
        LIMIT $limit
        """
        
        similar_data = self.run_query(similar_query, {"customer_id": customer_id, "limit": limit})
        
        if not similar_data:
            logging.warning(f"No similar customers found for customer {customer_id}")
            return []
        
        return similar_data

    def get_product_recommendations(self, customer_id, limit=5):
        """Generate personalized product recommendations for this customer."""
        recommendations_query = """
        MATCH (c:Customer {customer_id: $customer_id})
        
        // Find products this customer has viewed but not purchased
        OPTIONAL MATCH (c)-[:VIEWS]->(viewed:Product)
        OPTIONAL MATCH (c)-[:PURCHASES]->(purchased:Product)
        WITH c, collect(viewed) as viewed_products, collect(purchased) as purchased_products
        
        // Find segments this customer belongs to
        OPTIONAL MATCH (c)-[:BELONGS_TO]->(segment:Segment)
        WITH c, viewed_products, purchased_products, collect(segment) as segments
        
        // Find products purchased by customers in the same segments
        MATCH (other:Customer)-[:BELONGS_TO]->(segment:Segment)
        WHERE segment IN segments AND other.customer_id <> $customer_id
        
        MATCH (other)-[:PURCHASES]->(product:Product)
        WHERE NOT product IN purchased_products
        
        WITH product, count(DISTINCT other) as purchase_count
        
        RETURN product.id as product_id,
               purchase_count,
               CASE
                 WHEN product IN viewed_products THEN true
                 ELSE false
               END as previously_viewed
        ORDER BY previously_viewed DESC, purchase_count DESC
        LIMIT $limit
        """
        
        recommendations = self.run_query(
            recommendations_query, 
            {"customer_id": customer_id, "limit": limit}
        )
        
        if not recommendations:
            # Fallback to popular products if no personalized recommendations
            popular_query = """
            MATCH (c:Customer)-[:PURCHASES]->(p:Product)
            WITH p, count(c) as purchase_count
            RETURN p.id as product_id, purchase_count, false as previously_viewed
            ORDER BY purchase_count DESC
            LIMIT $limit
            """
            
            recommendations = self.run_query(popular_query, {"limit": limit})
        
        if not recommendations:
            logging.warning(f"No product recommendations available for customer {customer_id}")
            return []
        
        return recommendations

    def get_churn_risk_assessment(self, customer_id):
        """Assess the customer's risk of churning."""
        risk_query = """
        MATCH (c:Customer {customer_id: $customer_id})
        
        // Check last activity timestamp
        OPTIONAL MATCH (c)-[r]->(n)
        WHERE r.timestamp IS NOT NULL
        WITH c, max(r.timestamp) as last_activity
        
        // Check if cart was abandoned
        OPTIONAL MATCH (c)-[:ABANDONS]->(:Cart)
        WITH c, last_activity, count(*) > 0 as abandoned_cart
        
        // Check purchase history
        OPTIONAL MATCH (c)-[:PURCHASES]->(:Product)
        WITH c, last_activity, abandoned_cart, count(*) as purchase_count
        
        // Check email engagement
        OPTIONAL MATCH (c)-[email_interaction:VIEWS|CLICKS_ON]->(:Email)
        WITH c, last_activity, abandoned_cart, purchase_count,
             count(email_interaction) as email_interactions
        
        // Get current date
        WITH c, last_activity, abandoned_cart, purchase_count, email_interactions,
             datetime() as current_date
        
        // Calculate days since last activity
        WITH c, last_activity, abandoned_cart, purchase_count, email_interactions,
             CASE 
                WHEN last_activity IS NOT NULL 
                THEN duration.inDays(datetime(last_activity), current_date).days 
                ELSE 999
             END as days_inactive
        
        // Calculate risk factors
        RETURN 
            days_inactive,
            purchase_count,
            abandoned_cart,
            email_interactions,
            CASE
                WHEN days_inactive > 60 THEN 'High'
                WHEN days_inactive > 30 THEN 'Medium'
                WHEN days_inactive > 14 THEN 'Low'
                ELSE 'Very Low'
            END as inactivity_risk,
            CASE
                WHEN purchase_count = 0 THEN 'High'
                WHEN purchase_count = 1 THEN 'Medium'
                WHEN purchase_count < 5 THEN 'Low'
                ELSE 'Very Low'
            END as purchase_risk,
            CASE
                WHEN abandoned_cart THEN 'Medium'
                ELSE 'Low'
            END as abandonment_risk,
            CASE
                WHEN email_interactions = 0 THEN 'High'
                WHEN email_interactions < 3 THEN 'Medium'
                ELSE 'Low'
            END as engagement_risk
        """
        
        risk_data = self.run_query(risk_query, {"customer_id": customer_id})
        
        if not risk_data or len(risk_data) == 0:
            logging.warning(f"No risk assessment data available for customer {customer_id}")
            return {
                "overall_risk": "Unknown",
                "factors": {
                    "inactivity": "Unknown",
                    "purchase_history": "Unknown",
                    "cart_abandonment": "Unknown",
                    "engagement": "Unknown"
                }
            }
        
        data = risk_data[0]
        
        # Calculate overall risk level
        risk_levels = {
            "High": 3,
            "Medium": 2,
            "Low": 1,
            "Very Low": 0
        }
        
        risk_factors = [
            data.get("inactivity_risk", "Low"),
            data.get("purchase_risk", "Low"),
            data.get("abandonment_risk", "Low"),
            data.get("engagement_risk", "Low")
        ]
        
        # Convert to numeric values
        risk_values = [risk_levels.get(level, 0) for level in risk_factors]
        
        # Calculate average risk
        avg_risk = sum(risk_values) / len(risk_values)
        
        # Convert back to category
        overall_risk = "Low"
        if avg_risk >= 2.5:
            overall_risk = "High"
        elif avg_risk >= 1.5:
            overall_risk = "Medium"
        
        return {
            "overall_risk": overall_risk,
            "factors": {
                "inactivity": {
                    "level": data.get("inactivity_risk"),
                    "days_inactive": data.get("days_inactive", 0)
                },
                "purchase_history": {
                    "level": data.get("purchase_risk"),
                    "purchase_count": data.get("purchase_count", 0)
                },
                "cart_abandonment": {
                    "level": data.get("abandonment_risk"),
                    "has_abandoned": data.get("abandoned_cart", False)
                },
                "engagement": {
                    "level": data.get("engagement_risk"),
                    "email_interactions": data.get("email_interactions", 0)
                }
            }
        }

    def get_next_best_actions(self, customer_id):
        """Recommend next best actions for customer engagement."""
        # First get funnel status and risk assessment
        funnel_status = self.get_conversion_funnel_status(customer_id)
        risk_assessment = self.get_churn_risk_assessment(customer_id)
        
        current_stage = funnel_status.get("current_stage", "Unknown")
        has_churned = funnel_status.get("has_churned", False)
        overall_risk = risk_assessment.get("overall_risk", "Low")
        
        # Define next best actions based on funnel stage and churn risk
        next_actions = []
        
        if has_churned:
            next_actions.append({
                "action_type": "Re-engagement",
                "priority": "High",
                "description": "Send a personalized win-back offer",
                "details": "This customer has churned and requires a targeted re-engagement campaign with a compelling offer to return"
            })
            
        elif overall_risk == "High":
            next_actions.append({
                "action_type": "Churn Prevention",
                "priority": "High",
                "description": "Immediate engagement with special offer",
                "details": "This customer is at high risk of churning and should receive an immediate retention offer"
            })
            
        elif current_stage == "Awareness":
            next_actions.append({
                "action_type": "Consideration Boost",
                "priority": "Medium",
                "description": "Send targeted product recommendations",
                "details": "Customer is aware but hasn't shown consideration behavior. Send personalized product recommendations based on their profile"
            })
            
        elif current_stage == "Consideration":
            next_actions.append({
                "action_type": "Intent Generation",
                "priority": "Medium",
                "description": "Limited-time offer for viewed products",
                "details": "Customer has viewed products but not added to cart. Send limited-time offer for products they've viewed"
            })
            
        elif current_stage == "Intent":
            next_actions.append({
                "action_type": "Abandoned Cart",
                "priority": "High",
                "description": "Cart recovery email with discount",
                "details": "Customer has items in cart but hasn't purchased. Send cart recovery email with a small discount to incentivize completion"
            })
            
        elif current_stage == "Conversion":
            next_actions.append({
                "action_type": "Cross-sell",
                "priority": "Medium",
                "description": "Recommend complementary products",
                "details": "Customer has purchased but hasn't engaged with post-purchase content. Send recommendations for complementary products"
            })
            
        elif current_stage == "Retention":
            next_actions.append({
                "action_type": "Loyalty Building",
                "priority": "Medium",
                "description": "Invite to loyalty program",
                "details": "Customer has engaged post-purchase. Invite them to join your loyalty program for special benefits"
            })
            
        elif current_stage == "Advocacy":
            next_actions.append({
                "action_type": "Referral Program",
                "priority": "Low",
                "description": "Invite to refer friends",
                "details": "Customer is an advocate. Invite them to participate in your referral program with rewards for successful referrals"
            })
            
        # Add one more action based on risk assessment
        if overall_risk == "Medium":
            next_actions.append({
                "action_type": "Engagement Boost",
                "priority": "Medium",
                "description": "Send personalized content",
                "details": "Customer shows moderate churn risk. Send personalized content based on their interests to boost engagement"
            })
                
        return next_actions

    def create_customer_report(self, customer_id):
        """Create a comprehensive customer report with all available insights."""
        # Validate customer exists
        if not self.validate_customer_id(customer_id):
            logging.error(f"Customer ID {customer_id} not found in database")
            return {
                "error": "Customer not found",
                "customer_id": customer_id,
                "timestamp": datetime.datetime.now().isoformat()
            }
        
        # Start timing the report generation
        start_time = time.time()
        
        # Collect all customer data
        profile = self.get_customer_profile(customer_id)
        timeline = self.get_journey_timeline(customer_id)
        funnel_status = self.get_conversion_funnel_status(customer_id)
        similar_customers = self.get_similar_customers(customer_id)
        product_recommendations = self.get_product_recommendations(customer_id)
        churn_risk = self.get_churn_risk_assessment(customer_id)
        next_actions = self.get_next_best_actions(customer_id)
        
        # Calculate execution time
        execution_time = time.time() - start_time
        
        # Build the comprehensive report
        report = {
            "report_type": "Customer 360° Analysis",
            "customer_id": customer_id,
            "generated_at": datetime.datetime.now().isoformat(),
            "execution_time_seconds": execution_time,
            "profile": profile,
            "journey": {
                "timeline": timeline,
                "funnel_status": funnel_status
            },
            "insights": {
                "similar_customers": similar_customers,
                "product_recommendations": product_recommendations,
                "churn_risk": churn_risk
            },
            "actions": next_actions
        }
        
        # Ensure output directory exists
        Path(self.output_dir).mkdir(exist_ok=True)
        
        # Save the report to a JSON file
        report_file = os.path.join(self.output_dir, f"customer_{customer_id}_report.json")
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        logging.info(f"Comprehensive report for customer {customer_id} generated in {execution_time:.2f} seconds")
        return report

    def run(self, customer_id=None):
        """Run the dynamic analyzer for a specific customer or all customers."""
        try:
            # Connect to Neo4j
            if not self.connect():
                return False
            
            if customer_id:
                # Process a single customer
                report = self.create_customer_report(customer_id)
                if "error" in report:
                    logging.error(f"Failed to generate report for customer {customer_id}: {report['error']}")
                    return False
                return True
            else:
                # Process all customers (limit to 10 for testing)
                customers_query = """
                MATCH (c:Customer)
                RETURN c.customer_id AS customer_id
                LIMIT 10
                """
                
                customers = self.run_query(customers_query)
                if not customers:
                    logging.error("No customers found in database")
                    return False
                
                success_count = 0
                for customer in customers:
                    report = self.create_customer_report(customer["customer_id"])
                    if "error" not in report:
                        success_count += 1
                
                logging.info(f"Successfully processed {success_count}/{len(customers)} customers")
                return success_count > 0
                
        except Exception as e:
            logging.error(f"Error running dynamic analyzer: {e}")
            return False
            
        finally:
            # Close the Neo4j connection
            self.close()

def main():
    """Main function to run the dynamic customer analyzer from command line."""
    parser = argparse.ArgumentParser(description="Dynamic Customer Analysis Tool")
    parser.add_argument("--customer-id", help="Specific customer ID to analyze")
    parser.add_argument("--uri", help="Neo4j URI", default=os.getenv("NEO4J_URI"))
    parser.add_argument("--username", help="Neo4j username", default=os.getenv("NEO4J_USERNAME"))
    parser.add_argument("--password", help="Neo4j password", default=os.getenv("NEO4J_PASSWORD"))
    parser.add_argument("--database", help="Neo4j database", default=os.getenv("NEO4J_DATABASE"))
    args = parser.parse_args()
    
    analyzer = DynamicCustomerAnalyzer(
        uri=args.uri,
        username=args.username,
        password=args.password,
        database=args.database
    )
    
    if analyzer.run(args.customer_id):
        print(f"Analysis completed successfully!")
        print(f"Results saved to '{analyzer.output_dir}/' directory")
        return 0
    else:
        print("Analysis failed. Check the logs for details.")
        return 1

if __name__ == "__main__":
    print("Starting Dynamic Customer Analyzer...")
    main()