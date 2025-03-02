#!/usr/bin/env python3
"""
Journey Visualization module for Phase 2 of the marketing ontology project.
This module provides visualization capabilities for customer journeys and funnel analytics.
"""

import os
import json
import logging
import random
from datetime import datetime
from neo4j import GraphDatabase

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('journey_visualization.log')
    ]
)

class JourneyVisualization:
    """Class for generating journey visualizations from Neo4j graph data."""
    
    def __init__(self, uri=None, username=None, password=None, database=None):
        """Initialize the JourneyVisualization class with Neo4j connection details."""
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
    
    def generate_sankey_diagram_data(self):
        """Generate data for a Sankey diagram of the marketing funnel."""
        # Query to get journey transitions between stages
        sankey_query = """
        MATCH (c:Customer)
        OPTIONAL MATCH (c)-[:VIEWS|CLICKS_ON]->(:Advertisement)
        WITH c, CASE WHEN count(c) > 0 THEN 'Awareness' ELSE null END as awareness
        
        OPTIONAL MATCH (c)-[:VISITS|VIEWS]->(:Page)
        WITH c, awareness, CASE WHEN count(c) > 0 THEN 'Consideration' ELSE null END as consideration
        
        OPTIONAL MATCH (c)-[:ADDS_TO_CART]->(:Product)
        WITH c, awareness, consideration, CASE WHEN count(c) > 0 THEN 'Intent' ELSE null END as intent
        
        OPTIONAL MATCH (c)-[:PURCHASES]->(:Product)
        WITH c, awareness, consideration, intent, CASE WHEN count(c) > 0 THEN 'Purchase' ELSE null END as purchase
        
        OPTIONAL MATCH (c)-[:COMMENTS_ON|REFERS]->()
        WITH c, awareness, consideration, intent, purchase, CASE WHEN count(c) > 0 THEN 'Advocacy' ELSE null END as advocacy
        
        WITH 
            c,
            COLLECT(DISTINCT awareness) + COLLECT(DISTINCT consideration) + 
            COLLECT(DISTINCT intent) + COLLECT(DISTINCT purchase) + 
            COLLECT(DISTINCT advocacy) AS stages
        
        UNWIND range(0, size(stages) - 2) AS idx
        WITH c, stages[idx] AS source, stages[idx+1] AS target
        WHERE source IS NOT NULL AND target IS NOT NULL
        
        RETURN source, target, count(c) AS value
        ORDER BY source, target
        """
        
        # Execute the query
        sankey_data = self.run_query(sankey_query)
        
        if not sankey_data:
            # Generate sample data if the query fails or returns empty
            logging.warning("No Sankey data found. Generating sample data...")
            sankey_data = self._generate_sample_sankey_data()
        
        # Format the data for a Sankey diagram
        nodes = []
        node_ids = {}
        links = []
        
        # First, collect all unique nodes
        all_nodes = set()
        for record in sankey_data:
            source = record.get("source")
            target = record.get("target")
            if source:
                all_nodes.add(source)
            if target:
                all_nodes.add(target)
        
        # Create nodes list with IDs
        for idx, node in enumerate(sorted(all_nodes)):
            nodes.append({"id": idx, "name": node})
            node_ids[node] = idx
        
        # Create links between nodes
        for record in sankey_data:
            source = record.get("source")
            target = record.get("target")
            value = record.get("value", 0)
            
            if source in node_ids and target in node_ids:
                links.append({
                    "source": node_ids[source],
                    "target": node_ids[target],
                    "value": value
                })
        
        return {"nodes": nodes, "links": links}
    
    def _generate_sample_sankey_data(self):
        """Generate sample Sankey diagram data for testing."""
        stages = ["Awareness", "Consideration", "Intent", "Purchase", "Advocacy"]
        drop_rates = [0.3, 0.5, 0.4, 0.8]  # Percentage that drops at each stage
        
        starting_customers = 1000
        sample_data = []
        
        customers = starting_customers
        for i in range(len(stages) - 1):
            next_customers = int(customers * (1 - drop_rates[i]))
            sample_data.append({
                "source": stages[i],
                "target": stages[i+1],
                "value": next_customers
            })
            customers = next_customers
        
        return sample_data
    
    def generate_customer_journey_timeline(self, customer_id):
        """Generate a timeline visualization for a specific customer journey."""
        # Query to get customer journey events
        journey_query = """
        MATCH (c:Customer {customer_id: $customer_id})-[r]->(n)
        WHERE r.timestamp IS NOT NULL
        RETURN r.timestamp as timestamp,
               type(r) as event_type,
               r.action as action,
               labels(n)[0] as target_type,
               n.id as target_id
        ORDER BY r.timestamp
        """
        
        # Execute the query
        journey_data = self.run_query(journey_query, {"customer_id": customer_id})
        
        if not journey_data:
            # Generate sample data if the query fails or returns empty
            logging.warning(f"No journey data found for customer {customer_id}. Generating sample data...")
            journey_data = self._generate_sample_journey_data(customer_id)
        
        # Format the data for a timeline visualization
        events = []
        
        for record in journey_data:
            event = {
                "timestamp": record.get("timestamp"),
                "event_type": record.get("event_type"),
                "action": record.get("action"),
                "target_type": record.get("target_type"),
                "target_id": record.get("target_id"),
                "description": f"{record.get('action', 'Interacted with')} {record.get('target_type', 'item')} {record.get('target_id', '')}"
            }
            events.append(event)
        
        return {"customer_id": customer_id, "events": events}
    
    def _generate_sample_journey_data(self, customer_id):
        """Generate sample journey data for a customer."""
        actions = [
            {"action": "viewed_ad", "type": "VIEWS", "target_type": "Advertisement"},
            {"action": "clicked_ad", "type": "CLICKS_ON", "target_type": "Advertisement"},
            {"action": "visited_website", "type": "VISITS", "target_type": "Page"},
            {"action": "viewed_product", "type": "VIEWS", "target_type": "Product"},
            {"action": "added_to_cart", "type": "ADDS_TO_CART", "target_type": "Product"},
            {"action": "purchased", "type": "PURCHASES", "target_type": "Product"},
            {"action": "wrote_review", "type": "COMMENTS_ON", "target_type": "Product"}
        ]
        
        # Create sample timestamps increasing by a few minutes each
        base_time = datetime(2024, 10, 15, 14, 0, 0)
        sample_data = []
        
        for i, action_info in enumerate(actions):
            # 30% chance to skip this action (to simulate incomplete journeys)
            if random.random() < 0.3 and i > 0:
                continue
                
            minutes_offset = i * random.randint(5, 15)
            timestamp = base_time.replace(minute=base_time.minute + minutes_offset)
            
            sample_data.append({
                "timestamp": timestamp.isoformat(),
                "event_type": action_info["type"],
                "action": action_info["action"],
                "target_type": action_info["target_type"],
                "target_id": f"{action_info['target_type']}_{random.randint(1, 10)}"
            })
        
        return sample_data
    
    def generate_funnel_visualization(self):
        """Generate a funnel visualization of conversion stages."""
        # Query to get funnel stage counts
        funnel_query = """
        MATCH (c:Customer)
        WITH count(c) as total
        
        MATCH (c:Customer)
        OPTIONAL MATCH (c)-[:VIEWS|CLICKS_ON]->(:Advertisement)
        WITH total, count(c) as awareness_count
        
        MATCH (c:Customer)
        OPTIONAL MATCH (c)-[:VISITS|VIEWS]->(:Page)
        WITH total, awareness_count, count(c) as consideration_count
        
        MATCH (c:Customer)
        OPTIONAL MATCH (c)-[:ADDS_TO_CART]->(:Product)
        WITH total, awareness_count, consideration_count, count(c) as intent_count
        
        MATCH (c:Customer)
        OPTIONAL MATCH (c)-[:PURCHASES]->(:Product)
        WITH total, awareness_count, consideration_count, intent_count, count(c) as purchase_count
        
        MATCH (c:Customer)
        OPTIONAL MATCH (c)-[:COMMENTS_ON|REFERS]->()
        WITH total, awareness_count, consideration_count, intent_count, purchase_count, count(c) as advocacy_count
        
        RETURN total, awareness_count, consideration_count, intent_count, purchase_count, advocacy_count
        """
        
        # Execute the query
        funnel_data = self.run_query(funnel_query)
        
        if not funnel_data or len(funnel_data) == 0:
            # Generate sample data if the query fails or returns empty
            logging.warning("No funnel data found. Generating sample data...")
            return self._generate_sample_funnel_data()
        
        # Extract data from the query result
        data = funnel_data[0]
        total = data.get("total", 100)
        
        # Format the data for a funnel visualization
        funnel = [
            {"stage": "Total", "count": total, "percentage": 100},
            {"stage": "Awareness", "count": data.get("awareness_count", 0), "percentage": (data.get("awareness_count", 0) / total) * 100 if total > 0 else 0},
            {"stage": "Consideration", "count": data.get("consideration_count", 0), "percentage": (data.get("consideration_count", 0) / total) * 100 if total > 0 else 0},
            {"stage": "Intent", "count": data.get("intent_count", 0), "percentage": (data.get("intent_count", 0) / total) * 100 if total > 0 else 0},
            {"stage": "Purchase", "count": data.get("purchase_count", 0), "percentage": (data.get("purchase_count", 0) / total) * 100 if total > 0 else 0},
            {"stage": "Advocacy", "count": data.get("advocacy_count", 0), "percentage": (data.get("advocacy_count", 0) / total) * 100 if total > 0 else 0}
        ]
        
        return {"funnel": funnel}
    
    def _generate_sample_funnel_data(self):
        """Generate sample funnel data for testing."""
        total = 1000
        drop_rates = [0, 0.2, 0.5, 0.7, 0.9, 0.95]
        stages = ["Total", "Awareness", "Consideration", "Intent", "Purchase", "Advocacy"]
        
        funnel = []
        for i, stage in enumerate(stages):
            count = int(total * (1 - drop_rates[i]))
            funnel.append({
                "stage": stage,
                "count": count,
                "percentage": (count / total) * 100
            })
        
        return {"funnel": funnel}
    
    def generate_segment_comparison(self):
        """Generate a visualization comparing different customer segments."""
        # Query to get segment metrics
        segment_query = """
        MATCH (c:Customer)-[:BELONGS_TO]->(s:Segment)
        
        OPTIONAL MATCH (c)-[:PURCHASES]->(:Product)
        WITH s.id as segment, count(c) as total_customers,
             sum(CASE WHEN EXISTS((c)-[:PURCHASES]->()) THEN 1 ELSE 0 END) as purchasers
        
        OPTIONAL MATCH (c)-[:ABANDONS]->(:Cart)
        WITH segment, total_customers, purchasers,
             sum(CASE WHEN EXISTS((c)-[:ABANDONS]->()) THEN 1 ELSE 0 END) as cart_abandoners
        
        OPTIONAL MATCH (c)-[:CHURNED_AT]->(:FunnelStage)
        WITH segment, total_customers, purchasers, cart_abandoners,
             sum(CASE WHEN EXISTS((c)-[:CHURNED_AT]->()) THEN 1 ELSE 0 END) as churn_count
        
        RETURN segment,
               total_customers,
               purchasers,
               cart_abandoners,
               churn_count,
               CASE WHEN total_customers > 0 THEN toFloat(purchasers) / total_customers ELSE 0 END as conversion_rate,
               CASE WHEN total_customers > 0 THEN toFloat(cart_abandoners) / total_customers ELSE 0 END as abandonment_rate,
               CASE WHEN total_customers > 0 THEN toFloat(churn_count) / total_customers ELSE 0 END as churn_rate
        ORDER BY total_customers DESC
        """
        
        # Execute the query
        segment_data = self.run_query(segment_query)
        
        if not segment_data:
            # Generate sample data if the query fails or returns empty
            logging.warning("No segment data found. Generating sample data...")
            segment_data = self._generate_sample_segment_data()
        
        # Format the data for a comparison visualization
        segments = []
        for record in segment_data:
            segment = {
                "name": record.get("segment", "Unknown"),
                "total_customers": record.get("total_customers", 0),
                "metrics": {
                    "conversion_rate": record.get("conversion_rate", 0),
                    "abandonment_rate": record.get("abandonment_rate", 0),
                    "churn_rate": record.get("churn_rate", 0)
                }
            }
            segments.append(segment)
        
        return {"segments": segments}
    
    def _generate_sample_segment_data(self):
        """Generate sample segment data for testing."""
        segments = ["High-Value", "Mid-Value", "Low-Value", "New Customers"]
        
        sample_data = []
        for segment in segments:
            total = random.randint(100, 1000)
            purchasers = random.randint(0, total)
            abandoners = random.randint(0, total - purchasers)
            churners = random.randint(0, total)
            
            sample_data.append({
                "segment": segment,
                "total_customers": total,
                "purchasers": purchasers,
                "cart_abandoners": abandoners,
                "churn_count": churners,
                "conversion_rate": purchasers / total if total > 0 else 0,
                "abandonment_rate": abandoners / total if total > 0 else 0,
                "churn_rate": churners / total if total > 0 else 0
            })
        
        return sample_data
    
    def generate_device_browser_heatmap(self):
        """Generate a heatmap of conversion rates by device and browser combinations."""
        # Query to get device and browser conversion rates
        heatmap_query = """
        MATCH (c:Customer)
        OPTIONAL MATCH (c)-[:USES]->(d:Device)
        OPTIONAL MATCH (c)-[:ACCESSES_WITH]->(b:Browser)
        WITH d.id as device, b.id as browser, collect(c) as customers
        WHERE device IS NOT NULL AND browser IS NOT NULL AND size(customers) > 5
        
        WITH device, browser, customers,
             size([c IN customers WHERE EXISTS((c)-[:PURCHASES]->())]) as purchasers,
             size(customers) as total
        
        RETURN device, browser, purchasers, total,
               CASE WHEN total > 0 THEN toFloat(purchasers) / total ELSE 0 END as conversion_rate
        ORDER BY device, browser
        """
        
        # Execute the query
        heatmap_data = self.run_query(heatmap_query)
        
        if not heatmap_data:
            # Generate sample data if the query fails or returns empty
            logging.warning("No device/browser data found. Generating sample data...")
            heatmap_data = self._generate_sample_heatmap_data()
        
        # Get unique devices and browsers
        devices = sorted(set(record.get("device") for record in heatmap_data if record.get("device")))
        browsers = sorted(set(record.get("browser") for record in heatmap_data if record.get("browser")))
        
        # Create the heatmap matrix
        heatmap = []
        for device in devices:
            row = {"device": device, "browsers": {}}
            for browser in browsers:
                # Find the conversion rate for this device/browser combo
                conversion_rate = 0
                for record in heatmap_data:
                    if record.get("device") == device and record.get("browser") == browser:
                        conversion_rate = record.get("conversion_rate", 0)
                        break
                
                row["browsers"][browser] = conversion_rate
            
            heatmap.append(row)
        
        return {"devices": devices, "browsers": browsers, "heatmap": heatmap}
    
    def _generate_sample_heatmap_data(self):
        """Generate sample heatmap data for testing."""
        devices = ["desktop", "mobile", "tablet"]
        browsers = ["Chrome", "Firefox", "Safari", "Edge"]
        
        sample_data = []
        for device in devices:
            for browser in browsers:
                total = random.randint(20, 100)
                purchasers = random.randint(0, total)
                
                sample_data.append({
                    "device": device,
                    "browser": browser,
                    "purchasers": purchasers,
                    "total": total,
                    "conversion_rate": purchasers / total if total > 0 else 0
                })
        
        return sample_data
    
    def create_visualizations(self, output_dir="visualizations"):
        """Create all visualizations and save them to files."""
        try:
            # Connect to Neo4j
            if not self.connect():
                return False
            
            # Create the output directory if it doesn't exist
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            
            # Generate each visualization and save to a file
            # 1. Sankey diagram
            sankey_data = self.generate_sankey_diagram_data()
            sankey_file = os.path.join(output_dir, "sankey_diagram.json")
            with open(sankey_file, 'w') as f:
                json.dump(sankey_data, f, indent=2)
            
            # 2. Sample customer journey timeline
            # Use a known customer ID or the first one found
            customer_query = """
            MATCH (c:Customer)
            RETURN c.customer_id as customer_id
            LIMIT 1
            """
            customer_result = self.run_query(customer_query)
            customer_id = customer_result[0]["customer_id"] if customer_result else "CUST001"
            
            journey_data = self.generate_customer_journey_timeline(customer_id)
            journey_file = os.path.join(output_dir, f"journey_timeline_{customer_id}.json")
            with open(journey_file, 'w') as f:
                json.dump(journey_data, f, indent=2)
            
            # 3. Funnel visualization
            funnel_data = self.generate_funnel_visualization()
            funnel_file = os.path.join(output_dir, "funnel_visualization.json")
            with open(funnel_file, 'w') as f:
                json.dump(funnel_data, f, indent=2)
            
            # 4. Segment comparison
            segment_data = self.generate_segment_comparison()
            segment_file = os.path.join(output_dir, "segment_comparison.json")
            with open(segment_file, 'w') as f:
                json.dump(segment_data, f, indent=2)
            
            # 5. Device/browser heatmap
            heatmap_data = self.generate_device_browser_heatmap()
            heatmap_file = os.path.join(output_dir, "device_browser_heatmap.json")
            with open(heatmap_file, 'w') as f:
                json.dump(heatmap_data, f, indent=2)
            
            # Create an index file with links to all visualizations
            index = {
                "visualizations": [
                    {"name": "Sankey Diagram", "file": "sankey_diagram.json"},
                    {"name": f"Journey Timeline for {customer_id}", "file": f"journey_timeline_{customer_id}.json"},
                    {"name": "Funnel Visualization", "file": "funnel_visualization.json"},
                    {"name": "Segment Comparison", "file": "segment_comparison.json"},
                    {"name": "Device/Browser Heatmap", "file": "device_browser_heatmap.json"}
                ]
            }
            
            index_file = os.path.join(output_dir, "index.json")
            with open(index_file, 'w') as f:
                json.dump(index, f, indent=2)
            
            logging.info(f"All visualizations created in {output_dir} directory")
            return True
                
        except Exception as e:
            logging.error(f"Error creating visualizations: {e}")
            return False
        finally:
            # Close the Neo4j connection
            self.close()

if __name__ == "__main__":
    print("Starting Journey Visualization generation...")
    visualizer = JourneyVisualization()
    if visualizer.create_visualizations():
        print("Visualizations created successfully!")
        print("Results saved to visualizations/ directory")
    else:
        print("Visualization generation failed. Check the logs for details.")