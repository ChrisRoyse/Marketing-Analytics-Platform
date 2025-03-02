#!/usr/bin/env python
"""
Load demo data into Neo4j for the Marketing Ontology Platform Demo.

This script connects to Neo4j, creates necessary constraints and indexes,
loads customer journey data, and verifies data integrity.
"""

import json
import os
import sys
from pathlib import Path
from dotenv import load_dotenv
from neo4j import GraphDatabase
import logging

# Add parent directory to path to access shared modules
sys.path.append(str(Path(__file__).resolve().parent.parent))

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("demo_data_load.log"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

class Neo4jDemoLoader:
    """Handles loading demo data into Neo4j database."""
    
    def __init__(self, uri=None, username=None, password=None, database="marketing"):
        """Initialize the Neo4j connection for loading demo data."""
        # Load environment variables from .env file
        load_dotenv()
        
        # Use provided credentials or fall back to environment variables
        self.uri = uri or os.getenv("NEO4J_URI", "bolt://172.19.160.1:7687")
        self.username = username or os.getenv("NEO4J_USERNAME", "neo4j")
        self.password = password or os.getenv("NEO4J_PASSWORD")
        self.database = database
        
        # Exit if password is not provided
        if not self.password:
            logger.error("Neo4j password is required. Set it in .env file or provide it as an argument.")
            sys.exit(1)
            
        # Create Neo4j driver
        self.driver = GraphDatabase.driver(self.uri, auth=(self.username, self.password))
        logger.info(f"Connected to Neo4j at {self.uri}")
        
    def close(self):
        """Close the Neo4j driver connection."""
        if self.driver:
            self.driver.close()
            logger.info("Neo4j connection closed")
    
    def setup_constraints_and_indexes(self):
        """Create necessary constraints and indexes for demo data."""
        with self.driver.session(database=self.database) as session:
            # Constraints for unique IDs
            constraints = [
                "CREATE CONSTRAINT customer_id IF NOT EXISTS FOR (c:Customer) REQUIRE c.customer_id IS UNIQUE",
                "CREATE CONSTRAINT product_id IF NOT EXISTS FOR (p:Product) REQUIRE p.id IS UNIQUE",
                "CREATE CONSTRAINT email_id IF NOT EXISTS FOR (e:Email) REQUIRE e.id IS UNIQUE",
                "CREATE CONSTRAINT ad_id IF NOT EXISTS FOR (a:Advertisement) REQUIRE a.id IS UNIQUE",
                "CREATE CONSTRAINT page_id IF NOT EXISTS FOR (p:Page) REQUIRE p.id IS UNIQUE",
                "CREATE CONSTRAINT device_id IF NOT EXISTS FOR (d:Device) REQUIRE d.id IS UNIQUE",
                "CREATE CONSTRAINT channel_id IF NOT EXISTS FOR (c:Channel) REQUIRE c.id IS UNIQUE",
                "CREATE CONSTRAINT stage_id IF NOT EXISTS FOR (s:FunnelStage) REQUIRE s.id IS UNIQUE",
                "CREATE CONSTRAINT segment_id IF NOT EXISTS FOR (s:Segment) REQUIRE s.id IS UNIQUE",
                "CREATE CONSTRAINT persona_id IF NOT EXISTS FOR (p:Persona) REQUIRE p.id IS UNIQUE",
                "CREATE CONSTRAINT location_id IF NOT EXISTS FOR (l:Location) REQUIRE l.city IS UNIQUE"
            ]
            
            # Indexes for frequent lookups
            indexes = [
                "CREATE INDEX customer_email_idx IF NOT EXISTS FOR (c:Customer) ON (c.email)",
                "CREATE INDEX product_category_idx IF NOT EXISTS FOR (p:Product) ON (p.category)",
                "CREATE INDEX customer_registration_idx IF NOT EXISTS FOR (c:Customer) ON (c.registration_date)",
                "CREATE INDEX customer_segment_idx IF NOT EXISTS FOR ()-[r:BELONGS_TO]->() ON (r.type)"
            ]
            
            # Execute all constraints
            for constraint in constraints:
                try:
                    session.run(constraint)
                    logger.info(f"Created constraint: {constraint}")
                except Exception as e:
                    logger.error(f"Error creating constraint: {constraint}\n{str(e)}")
            
            # Execute all indexes
            for index in indexes:
                try:
                    session.run(index)
                    logger.info(f"Created index: {index}")
                except Exception as e:
                    logger.error(f"Error creating index: {index}\n{str(e)}")
                    
            logger.info("Constraints and indexes setup complete")
            
    def clear_existing_data(self):
        """Clear all existing data from the database."""
        with self.driver.session(database=self.database) as session:
            clear_query = """
            MATCH (n)
            DETACH DELETE n
            """
            
            result = session.run(clear_query)
            logger.info(f"Cleared existing data: {result.consume().counters}")
    
    def load_data(self, data_file):
        """Load data from the Neo4j import format JSON file."""
        try:
            with open(data_file, 'r') as f:
                data = json.load(f)
            
            # Process nodes
            self._load_nodes(data["nodes"])
            
            # Process relationships
            self._load_relationships(data["relationships"])
            
            logger.info(f"Loaded data from {data_file}")
            return True
        
        except Exception as e:
            logger.error(f"Error loading data: {str(e)}")
            return False
    
    def _load_nodes(self, nodes):
        """Load nodes into Neo4j."""
        with self.driver.session(database=self.database) as session:
            # Group nodes by label for batch processing
            nodes_by_label = {}
            for node in nodes:
                label = node["labels"][0]
                if label not in nodes_by_label:
                    nodes_by_label[label] = []
                nodes_by_label[label].append(node)
            
            # Process each label group
            for label, label_nodes in nodes_by_label.items():
                # Process in batches of 100
                batch_size = 100
                for i in range(0, len(label_nodes), batch_size):
                    batch = label_nodes[i:i+batch_size]
                    
                    # Create parametrized query based on node type
                    if label == "Customer":
                        query = f"""
                        UNWIND $nodes AS node
                        MERGE (n:{label} {{customer_id: node.properties.customer_id}})
                        SET n = node.properties
                        """
                    elif label == "Location":
                        query = f"""
                        UNWIND $nodes AS node
                        MERGE (n:{label} {{city: node.properties.city}})
                        SET n = node.properties
                        """
                    else:
                        query = f"""
                        UNWIND $nodes AS node
                        MERGE (n:{label} {{id: node.properties.id}})
                        SET n = node.properties
                        """
                    
                    # Execute batch
                    try:
                        result = session.run(query, nodes=batch)
                        logger.info(f"Created {len(batch)} {label} nodes")
                    except Exception as e:
                        logger.error(f"Error creating {label} nodes: {str(e)}")
                        logger.error(f"First node in batch: {batch[0] if batch else 'None'}")
    
    def _load_relationships(self, relationships):
        """Load relationships into Neo4j."""
        with self.driver.session(database=self.database) as session:
            # Group relationships by type for batch processing
            relationships_by_type = {}
            for rel in relationships:
                rel_type = rel["type"]
                if rel_type not in relationships_by_type:
                    relationships_by_type[rel_type] = []
                relationships_by_type[rel_type].append(rel)
            
            # Process each relationship type
            for rel_type, rels in relationships_by_type.items():
                # Process in batches of 100
                batch_size = 100
                for i in range(0, len(rels), batch_size):
                    batch = rels[i:i+batch_size]
                    
                    # Create parametrized query with more specific matching
                    query = f"""
                    UNWIND $rels AS rel
                    MATCH (a), (b)
                    WHERE (a.customer_id = rel.startNode OR 
                          (a.id IS NOT NULL AND a.id = rel.startNode) OR 
                          (a:Location AND a.city = rel.startNode))
                    AND (b.id = rel.endNode OR 
                         (b:Location AND b.city = rel.endNode) OR 
                         (rel.endNode STARTS WITH 'LOC_' AND b.city = substring(rel.endNode, 4)) OR
                         (rel.endNode STARTS WITH 'SEG_' AND b.id = substring(rel.endNode, 4)) OR
                         (rel.endNode STARTS WITH 'PERS_' AND b.id = substring(rel.endNode, 5)) OR
                         (rel.endNode STARTS WITH 'DEV_' AND b.id = substring(rel.endNode, 4)) OR
                         (rel.endNode STARTS WITH 'Product_' AND b.id = substring(rel.endNode, 8)) OR
                         (rel.endNode STARTS WITH 'Email_' AND b.id = substring(rel.endNode, 6)) OR
                         (rel.endNode STARTS WITH 'Advertisement_' AND b.id = substring(rel.endNode, 14)) OR
                         (rel.endNode STARTS WITH 'Page_' AND b.id = substring(rel.endNode, 5)) OR
                         (rel.endNode STARTS WITH 'Channel_' AND b.id = substring(rel.endNode, 8)) OR
                         (rel.endNode STARTS WITH 'FunnelStage_' AND b.id = substring(rel.endNode, 12)))
                    CREATE (a)-[r:{rel_type}]->(b)
                    SET r = rel.properties
                    """
                    
                    try:
                        result = session.run(query, rels=batch)
                        logger.info(f"Created {len(batch)} {rel_type} relationships")
                    except Exception as e:
                        logger.error(f"Error creating {rel_type} relationships: {str(e)}")
                        
                        # Try individual relationships in case of failure
                        logger.info(f"Attempting to create {rel_type} relationships individually")
                        success_count = 0
                        
                        for rel in batch:
                            try:
                                # Simplify the matching for individual relationships
                                customer_match = "a.customer_id = $startNode" if rel["startNode"].startswith("CUST") else ""
                                id_match = "a.id = $startNode" if not rel["startNode"].startswith("CUST") else ""
                                where_clause_a = f"WHERE {customer_match if customer_match else id_match}"
                                
                                # Simplified matching for end node
                                if rel["endNode"].startswith("LOC_"):
                                    endNode = rel["endNode"][4:]  # Remove "LOC_" prefix
                                    where_clause_b = "AND b.city = $endNode"
                                elif rel["endNode"].startswith("SEG_"):
                                    endNode = rel["endNode"][4:]  # Remove "SEG_" prefix
                                    where_clause_b = "AND b.id = $endNode"
                                elif rel["endNode"].startswith("PERS_"):
                                    endNode = rel["endNode"][5:]  # Remove "PERS_" prefix
                                    where_clause_b = "AND b.id = $endNode"
                                elif rel["endNode"].startswith("DEV_"):
                                    endNode = rel["endNode"][4:]  # Remove "DEV_" prefix
                                    where_clause_b = "AND b.id = $endNode"
                                else:
                                    endNode = rel["endNode"]
                                    where_clause_b = "AND b.id = $endNode"
                                
                                individual_query = f"""
                                MATCH (a), (b)
                                {where_clause_a}
                                {where_clause_b}
                                CREATE (a)-[r:{rel_type}]->(b)
                                SET r = $properties
                                """
                                
                                session.run(
                                    individual_query, 
                                    startNode=rel["startNode"],
                                    endNode=endNode,
                                    properties=rel["properties"]
                                )
                                success_count += 1
                            except Exception as inner_e:
                                logger.error(f"Error on relationship {rel['id']}: {str(inner_e)}")
                        
                        logger.info(f"Successfully created {success_count} out of {len(batch)} {rel_type} relationships individually")
    
    def verify_data_integrity(self):
        """Verify that the data was loaded correctly."""
        with self.driver.session(database=self.database) as session:
            # Check counts for different node types
            node_counts = {}
            for label in ["Customer", "Product", "Advertisement", "Email", "Page", "Channel", "Device", "FunnelStage", "Segment", "Persona", "Location"]:
                query = f"MATCH (n:{label}) RETURN count(n) AS count"
                result = session.run(query)
                count = result.single()["count"]
                node_counts[label] = count
                logger.info(f"Found {count} {label} nodes")
            
            # Check counts for different relationship types
            rel_counts = {}
            for rel_type in ["VIEWS", "CLICKS_ON", "PURCHASES", "ABANDONS", "VISITS", "COMES_FROM", "ADDS_TO_CART", "RECEIVES", "OPENS", "BELONGS_TO", "HAS_PERSONA", "USES", "LIVES_IN"]:
                query = f"MATCH ()-[r:{rel_type}]->() RETURN count(r) AS count"
                result = session.run(query)
                count = result.single()["count"]
                rel_counts[rel_type] = count
                logger.info(f"Found {count} {rel_type} relationships")
            
            # Check for any orphaned nodes or relationships
            orphan_query = """
            MATCH (n)
            WHERE NOT (n)--()
            RETURN labels(n) AS label, count(*) AS count
            """
            result = session.run(orphan_query)
            orphans = {}
            for record in result:
                orphans[record["label"][0]] = record["count"]
                
            if orphans:
                for label, count in orphans.items():
                    logger.warning(f"Found {count} orphaned {label} nodes")
            else:
                logger.info("No orphaned nodes found")
                
            return {
                "node_counts": node_counts,
                "relationship_counts": rel_counts,
                "orphans": orphans
            }

def main():
    """Load generated demo data into Neo4j."""
    data_file = os.path.join("/home/cabdru/marketing/demo/demo_data", "neo4j_import.json")
    
    # Check if data file exists
    if not os.path.exists(data_file):
        logger.error(f"Data file not found: {data_file}")
        logger.info("Please run generate_demo_data.py first")
        return
    
    # Create Neo4j loader
    loader = Neo4jDemoLoader()
    
    try:
        # Clear existing data - automatically proceed in demo mode
        logger.info("Clearing existing database data...")
        loader.clear_existing_data()
        
        # Set up constraints and indexes
        loader.setup_constraints_and_indexes()
        
        # Load data
        logger.info(f"Loading data from {data_file}...")
        success = loader.load_data(data_file)
        
        if success:
            # Verify data integrity
            verification_results = loader.verify_data_integrity()
            
            # Check if we have customers and events
            if verification_results["node_counts"].get("Customer", 0) > 0 and \
               sum(count for rel_type, count in verification_results["relationship_counts"].items() if rel_type in ["VIEWS", "CLICKS_ON", "PURCHASES"]) > 0:
                logger.info("Demo data loaded successfully!")
            else:
                logger.warning("Data loaded but verification shows missing customers or events")
        else:
            logger.error("Failed to load demo data")
    
    finally:
        # Close connection
        loader.close()

if __name__ == "__main__":
    main()