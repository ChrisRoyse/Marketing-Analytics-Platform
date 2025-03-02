#!/usr/bin/env python3
"""
Marketing Analytics module for Phase 2 of the marketing ontology project.
This module provides advanced analytics capabilities on top of the Neo4j graph database.
It includes journey analysis, funnel optimization, churn prediction, and recommendations.
"""

import os
import json
import logging
from datetime import datetime
from neo4j import GraphDatabase

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('marketing_analytics.log')
    ]
)

class MarketingAnalytics:
    """Class for performing advanced marketing analytics on Neo4j graph data."""
    
    def __init__(self, uri=None, username=None, password=None, database=None):
        """Initialize the MarketingAnalytics class with Neo4j connection details."""
        self.uri = uri or os.getenv('NEO4J_URI', 'bolt://localhost:7687')
        self.username = username or os.getenv('NEO4J_USERNAME', 'neo4j')
        self.password = password or os.getenv('NEO4J_PASSWORD', 'neo4j')
        self.database = database or os.getenv('NEO4J_DATABASE', 'neo4j')
        self.driver = None
        
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
            
    def get_database_statistics(self):
        """Get basic statistics about the database."""
        # Count nodes by label
        node_query = """
        CALL db.labels() YIELD label
        CALL {
            WITH label
            MATCH (n:`' + label + '`)
            RETURN count(n) AS count
        }
        RETURN label, count
        ORDER BY count DESC
        """
        
        # Count relationships by type
        rel_query = """
        CALL db.relationshipTypes() YIELD relationshipType
        CALL {
            WITH relationshipType
            MATCH ()-[r:`' + relationshipType + '`]->()
            RETURN count(r) AS count
        }
        RETURN relationshipType, count
        ORDER BY count DESC
        """
        
        nodes = self.run_query(node_query)
        relationships = self.run_query(rel_query)
        
        return {
            "node_counts": {record["label"]: record["count"] for record in nodes} if nodes else {},
            "relationship_counts": {record["relationshipType"]: record["count"] for record in relationships} if relationships else {}
        }
        
    def analyze_customer_journey(self, customer_id):
        """Analyze the journey of a specific customer."""
        # Get customer details
        customer_query = """
        MATCH (c:Customer {customer_id: $customer_id})
        RETURN c
        """
        
        # Get customer journey timeline
        journey_query = """
        MATCH (c:Customer {customer_id: $customer_id})-[r]->(n)
        WHERE r.timestamp IS NOT NULL
        RETURN type(r) as interaction_type,
               r.action as action,
               r.timestamp as timestamp,
               labels(n)[0] as node_type,
               n.id as node_id
        ORDER BY r.timestamp
        """
        
        # Get customer device and channel preferences
        preferences_query = """
        MATCH (c:Customer {customer_id: $customer_id})
        OPTIONAL MATCH (c)-[:USES]->(d:Device)
        OPTIONAL MATCH (c)-[:ACCESSES_WITH]->(b:Browser)
        OPTIONAL MATCH (c)-[:PREFERS]->(ch:Channel)
        OPTIONAL MATCH (c)-[:LOCATED_IN]->(l:Location)
        RETURN c.customer_id as customer_id,
               collect(DISTINCT d.id) as devices,
               collect(DISTINCT b.id) as browsers,
               collect(DISTINCT ch.id) as preferred_channels,
               collect(DISTINCT l.id) as locations
        """
        
        # Get customer segments and personas
        segmentation_query = """
        MATCH (c:Customer {customer_id: $customer_id})
        OPTIONAL MATCH (c)-[:BELONGS_TO]->(s:Segment)
        OPTIONAL MATCH (c)-[:HAS_PERSONA]->(p:Persona)
        OPTIONAL MATCH (c)-[:AT_STAGE]->(bs:BehaviorStage)
        RETURN c.customer_id as customer_id,
               collect(DISTINCT s.id) as segments,
               collect(DISTINCT p.id) as personas,
               collect(DISTINCT bs.id) as behavior_stages
        """
        
        # Execute all queries with the customer_id parameter
        params = {"customer_id": customer_id}
        customer = self.run_query(customer_query, params)
        journey = self.run_query(journey_query, params)
        preferences = self.run_query(preferences_query, params)
        segmentation = self.run_query(segmentation_query, params)
        
        # Combine all data into a single analysis result
        result = {
            "customer_details": customer[0]["c"] if customer and len(customer) > 0 else None,
            "journey_timeline": journey if journey else [],
            "preferences": preferences[0] if preferences and len(preferences) > 0 else {},
            "segmentation": segmentation[0] if segmentation and len(segmentation) > 0 else {},
            "analysis": {
                "journey_length": len(journey) if journey else 0,
                "first_interaction": journey[0]["timestamp"] if journey and len(journey) > 0 else None,
                "last_interaction": journey[-1]["timestamp"] if journey and len(journey) > 0 else None,
                "has_churned": self._has_customer_churned(customer_id)
            }
        }
        
        return result
        
    def _has_customer_churned(self, customer_id):
        """Check if a customer has churned."""
        churn_query = """
        MATCH (c:Customer {customer_id: $customer_id})-[r:CHURNED_AT]->(:FunnelStage)
        RETURN r.timestamp as churn_time, r.reason as churn_reason
        """
        
        churn_data = self.run_query(churn_query, {"customer_id": customer_id})
        return churn_data[0] if churn_data and len(churn_data) > 0 else None
        
    def analyze_funnel_performance(self):
        """Analyze the performance of the marketing funnel."""
        # Get funnel stage counts
        funnel_query = """
        MATCH (c:Customer)
        OPTIONAL MATCH (c)-[awareness:VIEWS|CLICKS_ON]->(awareness_node)
        WHERE awareness_node:Advertisement
        WITH c, count(awareness) > 0 as reached_awareness
        
        OPTIONAL MATCH (c)-[consideration:VISITS|VIEWS|ADDS_TO_CART]->(consideration_node)
        WHERE consideration_node:Page OR consideration_node:Product
        WITH c, reached_awareness, count(consideration) > 0 as reached_consideration
        
        OPTIONAL MATCH (c)-[conversion:PURCHASES]->(conversion_node)
        WHERE conversion_node:Product
        WITH c, reached_awareness, reached_consideration, count(conversion) > 0 as reached_conversion
        
        OPTIONAL MATCH (c)-[retention:INTERACTS_WITH]->(retention_node)
        WHERE retention_node:Content AND retention_node.id CONTAINS 'post_purchase'
        WITH c, reached_awareness, reached_consideration, reached_conversion, count(retention) > 0 as reached_retention
        
        OPTIONAL MATCH (c)-[advocacy:REFERS|COMMENTS_ON]->(advocacy_node)
        WITH c, reached_awareness, reached_consideration, reached_conversion, reached_retention, count(advocacy) > 0 as reached_advocacy
        
        RETURN 
            count(c) as total_customers,
            sum(CASE WHEN reached_awareness THEN 1 ELSE 0 END) as awareness_count,
            sum(CASE WHEN reached_consideration THEN 1 ELSE 0 END) as consideration_count,
            sum(CASE WHEN reached_conversion THEN 1 ELSE 0 END) as conversion_count,
            sum(CASE WHEN reached_retention THEN 1 ELSE 0 END) as retention_count,
            sum(CASE WHEN reached_advocacy THEN 1 ELSE 0 END) as advocacy_count
        """
        
        # Get conversion rates by segment
        segment_conversion_query = """
        MATCH (c:Customer)-[:BELONGS_TO]->(s:Segment)
        OPTIONAL MATCH (c)-[:PURCHASES]->(:Product)
        WITH s.id as segment, count(c) as total, sum(CASE WHEN EXISTS((c)-[:PURCHASES]->()) THEN 1 ELSE 0 END) as converted
        RETURN segment, total, converted,
               CASE WHEN total > 0 THEN toFloat(converted) / total ELSE 0 END as conversion_rate
        ORDER BY conversion_rate DESC
        """
        
        # Get churn points analysis
        churn_query = """
        MATCH (c:Customer)-[r:CHURNED_AT]->(stage:FunnelStage)
        RETURN stage.name as funnel_stage, count(c) as churn_count
        ORDER BY churn_count DESC
        """
        
        # Execute all queries
        funnel_data = self.run_query(funnel_query)
        segment_conversion = self.run_query(segment_conversion_query)
        churn_data = self.run_query(churn_query)
        
        # Calculate funnel conversion rates
        funnel_metrics = funnel_data[0] if funnel_data and len(funnel_data) > 0 else {}
        conversion_metrics = {}
        
        if funnel_metrics:
            total = funnel_metrics.get("total_customers", 0)
            if total > 0:
                conversion_metrics = {
                    "awareness_rate": funnel_metrics.get("awareness_count", 0) / total,
                    "consideration_rate": funnel_metrics.get("consideration_count", 0) / funnel_metrics.get("awareness_count", 1),
                    "conversion_rate": funnel_metrics.get("conversion_count", 0) / funnel_metrics.get("consideration_count", 1),
                    "retention_rate": funnel_metrics.get("retention_count", 0) / funnel_metrics.get("conversion_count", 1),
                    "advocacy_rate": funnel_metrics.get("advocacy_count", 0) / funnel_metrics.get("retention_count", 1),
                    "overall_conversion": funnel_metrics.get("conversion_count", 0) / total
                }
        
        # Combine all analysis data
        result = {
            "funnel_stage_counts": funnel_metrics,
            "conversion_rates": conversion_metrics,
            "segment_conversion": segment_conversion if segment_conversion else [],
            "churn_analysis": churn_data if churn_data else []
        }
        
        return result
        
    def identify_bottlenecks(self):
        """Identify bottlenecks in the customer journey."""
        # Find stages with the highest drop-off rates
        bottleneck_query = """
        MATCH (c:Customer)
        OPTIONAL MATCH (c)-[awareness:VIEWS|CLICKS_ON]->(awareness_node)
        WHERE awareness_node:Advertisement
        WITH c, count(awareness) > 0 as reached_awareness
        
        OPTIONAL MATCH (c)-[consideration:VISITS|VIEWS|ADDS_TO_CART]->(consideration_node)
        WHERE consideration_node:Page OR consideration_node:Product
        WITH c, reached_awareness, count(consideration) > 0 as reached_consideration
        
        OPTIONAL MATCH (c)-[conversion:PURCHASES]->(conversion_node)
        WHERE conversion_node:Product
        WITH c, reached_awareness, reached_consideration, count(conversion) > 0 as reached_conversion
        
        OPTIONAL MATCH (c)-[retention:INTERACTS_WITH]->(retention_node)
        WHERE retention_node:Content AND retention_node.id CONTAINS 'post_purchase'
        WITH c, reached_awareness, reached_consideration, reached_conversion, count(retention) > 0 as reached_retention
        
        OPTIONAL MATCH (c)-[advocacy:REFERS|COMMENTS_ON]->(advocacy_node)
        WITH c, reached_awareness, reached_consideration, reached_conversion, reached_retention, count(advocacy) > 0 as reached_advocacy
        
        RETURN 
            count(c) as total_customers,
            sum(CASE WHEN reached_awareness THEN 1 ELSE 0 END) as awareness_count,
            sum(CASE WHEN reached_consideration THEN 1 ELSE 0 END) as consideration_count,
            sum(CASE WHEN reached_conversion THEN 1 ELSE 0 END) as conversion_count,
            sum(CASE WHEN reached_retention THEN 1 ELSE 0 END) as retention_count,
            sum(CASE WHEN reached_advocacy THEN 1 ELSE 0 END) as advocacy_count
        """
        
        # Find product pages with high view-to-purchase drop-off
        product_bottleneck_query = """
        MATCH (p:Product)<-[:VIEWS]-(c:Customer)
        WITH p, count(c) as view_count
        OPTIONAL MATCH (p)<-[:PURCHASES]-(c2:Customer)
        WITH p, view_count, count(c2) as purchase_count
        WHERE view_count > 5
        RETURN p.id as product, 
               view_count, 
               purchase_count,
               CASE WHEN view_count > 0 THEN toFloat(purchase_count) / view_count ELSE 0 END as conversion_rate
        ORDER BY conversion_rate ASC
        LIMIT 10
        """
        
        # Find device and browser combinations with high churn rates
        device_bottleneck_query = """
        MATCH (c:Customer)-[:USES]->(d:Device)
        MATCH (c)-[:ACCESSES_WITH]->(b:Browser)
        WITH d.id as device, b.id as browser, collect(c) as customers
        WITH device, browser, customers, size(customers) as total_count
        WITH device, browser, customers, total_count,
             size([c in customers WHERE EXISTS((c)-[:CHURNED_AT]->())]) as churn_count
        WHERE total_count > 5
        RETURN device, browser, total_count, churn_count,
               CASE WHEN total_count > 0 THEN toFloat(churn_count) / total_count ELSE 0 END as churn_rate
        ORDER BY churn_rate DESC
        LIMIT 10
        """
        
        # Execute all queries
        funnel_data = self.run_query(bottleneck_query)
        product_bottlenecks = self.run_query(product_bottleneck_query)
        device_bottlenecks = self.run_query(device_bottleneck_query)
        
        # Calculate drop-off rates between funnel stages
        funnel_metrics = funnel_data[0] if funnel_data and len(funnel_data) > 0 else {}
        dropoff_rates = {}
        
        if funnel_metrics:
            # Calculate drop-off between each consecutive stage
            awareness_count = funnel_metrics.get("awareness_count", 0)
            consideration_count = funnel_metrics.get("consideration_count", 0)
            conversion_count = funnel_metrics.get("conversion_count", 0)
            retention_count = funnel_metrics.get("retention_count", 0)
            advocacy_count = funnel_metrics.get("advocacy_count", 0)
            
            dropoff_rates = {
                "awareness_to_consideration": 1 - (consideration_count / awareness_count if awareness_count > 0 else 0),
                "consideration_to_conversion": 1 - (conversion_count / consideration_count if consideration_count > 0 else 0),
                "conversion_to_retention": 1 - (retention_count / conversion_count if conversion_count > 0 else 0),
                "retention_to_advocacy": 1 - (advocacy_count / retention_count if retention_count > 0 else 0)
            }
            
            # Identify the largest bottleneck
            largest_bottleneck = max(dropoff_rates.items(), key=lambda x: x[1])
            dropoff_rates["largest_bottleneck"] = {
                "stage": largest_bottleneck[0],
                "dropoff_rate": largest_bottleneck[1]
            }
        
        # Combine all bottleneck data
        result = {
            "funnel_dropoff_rates": dropoff_rates,
            "product_bottlenecks": product_bottlenecks if product_bottlenecks else [],
            "device_browser_bottlenecks": device_bottlenecks if device_bottlenecks else []
        }
        
        return result
        
    def recommend_journey_improvements(self):
        """Generate recommendations for improving customer journeys."""
        # Get bottleneck analysis to base recommendations on
        bottlenecks = self.identify_bottlenecks()
        
        # Prepare recommendations based on bottleneck analysis
        recommendations = {
            "funnel_optimizations": [],
            "product_optimizations": [],
            "device_optimizations": [],
            "general_recommendations": []
        }
        
        # Add funnel optimization recommendations
        dropoff_rates = bottlenecks.get("funnel_dropoff_rates", {})
        if dropoff_rates:
            largest_bottleneck = dropoff_rates.get("largest_bottleneck", {})
            if largest_bottleneck:
                stage = largest_bottleneck.get("stage")
                rate = largest_bottleneck.get("dropoff_rate", 0)
                
                if stage == "awareness_to_consideration" and rate > 0.7:
                    recommendations["funnel_optimizations"].append({
                        "priority": "high",
                        "focus_area": "Awareness to Consideration",
                        "issue": f"High dropoff rate ({rate:.2%}) from awareness to consideration",
                        "recommendation": "Improve ad targeting and landing page relevance. Ensure ads set clear expectations."
                    })
                elif stage == "consideration_to_conversion" and rate > 0.8:
                    recommendations["funnel_optimizations"].append({
                        "priority": "high",
                        "focus_area": "Consideration to Conversion",
                        "issue": f"High dropoff rate ({rate:.2%}) from consideration to conversion",
                        "recommendation": "Optimize product pages, simplify checkout, add social proof, and consider exit-intent offers."
                    })
                elif stage == "conversion_to_retention" and rate > 0.6:
                    recommendations["funnel_optimizations"].append({
                        "priority": "medium",
                        "focus_area": "Conversion to Retention",
                        "issue": f"High dropoff rate ({rate:.2%}) from conversion to retention",
                        "recommendation": "Improve post-purchase communication, implement customer onboarding, and solicit feedback."
                    })
                elif stage == "retention_to_advocacy" and rate > 0.8:
                    recommendations["funnel_optimizations"].append({
                        "priority": "medium",
                        "focus_area": "Retention to Advocacy",
                        "issue": f"High dropoff rate ({rate:.2%}) from retention to advocacy",
                        "recommendation": "Create referral programs, encourage reviews, and reward loyalty."
                    })
        
        # Add product optimization recommendations
        product_bottlenecks = bottlenecks.get("product_bottlenecks", [])
        for idx, product in enumerate(product_bottlenecks):
            if idx >= 3:  # Limit to top 3 products
                break
                
            conversion_rate = product.get("conversion_rate", 0)
            view_count = product.get("view_count", 0)
            
            if conversion_rate < 0.1 and view_count > 20:
                recommendations["product_optimizations"].append({
                    "priority": "high",
                    "product_id": product.get("product"),
                    "issue": f"Low conversion rate ({conversion_rate:.2%}) despite high views ({view_count})",
                    "recommendation": "Review pricing, improve product description, add better images/videos, and highlight reviews."
                })
        
        # Add device optimization recommendations
        device_bottlenecks = bottlenecks.get("device_browser_bottlenecks", [])
        for idx, device_combo in enumerate(device_bottlenecks):
            if idx >= 3:  # Limit to top 3 device/browser combinations
                break
                
            device = device_combo.get("device")
            browser = device_combo.get("browser")
            churn_rate = device_combo.get("churn_rate", 0)
            
            if churn_rate > 0.4:
                recommendations["device_optimizations"].append({
                    "priority": "high",
                    "device_browser": f"{device} / {browser}",
                    "issue": f"High churn rate ({churn_rate:.2%}) for users on {device} with {browser}",
                    "recommendation": f"Test and optimize user experience specifically for {device}/{browser} combination."
                })
        
        # Add general recommendations
        recommendations["general_recommendations"] = [
            {
                "priority": "medium",
                "focus_area": "Personalization",
                "recommendation": "Implement personalized recommendations based on customer browsing and purchase history."
            },
            {
                "priority": "medium",
                "focus_area": "Email Marketing",
                "recommendation": "Optimize email campaigns with segmentation and personalized content."
            },
            {
                "priority": "medium",
                "focus_area": "Customer Feedback",
                "recommendation": "Implement a systematic approach to collect and act on customer feedback."
            }
        ]
        
        return recommendations
        
    def generate_analysis_report(self):
        """Generate a comprehensive marketing analysis report."""
        # Collect all analytics data
        stats = self.get_database_statistics()
        funnel_performance = self.analyze_funnel_performance()
        bottlenecks = self.identify_bottlenecks()
        recommendations = self.recommend_journey_improvements()
        
        # Create the comprehensive report
        report = {
            "report_name": "Marketing Ontology Analysis",
            "generated_at": datetime.now().isoformat(),
            "database_statistics": stats,
            "funnel_performance": funnel_performance,
            "journey_bottlenecks": bottlenecks,
            "recommendations": recommendations,
            "summary": {
                "key_findings": [],
                "priority_actions": []
            }
        }
        
        # Extract key findings based on the analysis
        # Find the stages with highest drop-off
        dropoff_rates = bottlenecks.get("funnel_dropoff_rates", {})
        largest_bottleneck = dropoff_rates.get("largest_bottleneck", {})
        if largest_bottleneck:
            report["summary"]["key_findings"].append(
                f"The largest funnel bottleneck is at the {largest_bottleneck.get('stage', '').replace('_', ' to ')} " +
                f"stage with a drop-off rate of {largest_bottleneck.get('dropoff_rate', 0):.2%}"
            )
        
        # Overall conversion rate
        conversion_rates = funnel_performance.get("conversion_rates", {})
        overall_conversion = conversion_rates.get("overall_conversion", 0)
        report["summary"]["key_findings"].append(
            f"Overall conversion rate from awareness to purchase is {overall_conversion:.2%}"
        )
        
        # Top segment by conversion rate
        segment_conversion = funnel_performance.get("segment_conversion", [])
        if segment_conversion and len(segment_conversion) > 0:
            top_segment = segment_conversion[0]
            report["summary"]["key_findings"].append(
                f"The {top_segment.get('segment')} segment has the highest conversion rate at {top_segment.get('conversion_rate', 0):.2%}"
            )
        
        # Extract priority actions from recommendations
        for area, recs in recommendations.items():
            if area == "general_recommendations":
                continue
                
            for rec in recs:
                if rec.get("priority") == "high":
                    report["summary"]["priority_actions"].append(
                        f"{rec.get('focus_area', '')}: {rec.get('recommendation', '')}"
                    )
        
        # Return the complete report
        return report
        
    def run_phase2_analytics(self, output_file="marketing_analysis_report.json"):
        """Run the Phase 2 analytics process and save the report to a file."""
        try:
            # Connect to Neo4j
            if not self.connect():
                return False
                
            # Generate the analysis report
            report = self.generate_analysis_report()
            
            # Save the report to a file
            with open(output_file, 'w') as f:
                json.dump(report, f, indent=2)
                
            logging.info(f"Analysis report generated and saved to {output_file}")
            return True
            
        except Exception as e:
            logging.error(f"Error running Phase 2 analytics: {e}")
            return False
        finally:
            # Close the Neo4j connection
            self.close()

if __name__ == "__main__":
    print("Starting Phase 2 Marketing Analytics...")
    analytics = MarketingAnalytics()
    if analytics.run_phase2_analytics():
        print("Phase 2 analytics completed successfully!")
        print("Results saved to marketing_analysis_report.json")
    else:
        print("Phase 2 analytics failed. Check the logs for details.")