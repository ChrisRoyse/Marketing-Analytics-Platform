
#!/usr/bin/env python3
import os
import json
import logging
from neo4j import GraphDatabase
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Load environment variables from .env file
load_dotenv('/mnt/c/code/marketing/.env')

def explore_neo4j_data():
    uri = os.getenv('NEO4J_URI', 'bolt://localhost:7687')
    username = os.getenv('NEO4J_USERNAME', 'neo4j')
    password = os.getenv('NEO4J_PASSWORD', 'neo4j')
    database = os.getenv('NEO4J_DATABASE', 'marketing')
    
    try:
        driver = GraphDatabase.driver(uri, auth=(username, password))
        with driver.session(database=database) as session:
            # Get node counts by label
            result = session.run("CALL db.labels() YIELD label")
            labels = [record["label"] for record in result]
            
            node_counts = {}
            for label in labels:
                result = session.run(
                    f"MATCH (n:`{label}`) RETURN count(n) AS count"
                )
                count = result.single()["count"]
                node_counts[label] = count
            
            # Get relationship counts by type
            result = session.run("CALL db.relationshipTypes() YIELD relationshipType")
            rel_types = [record["relationshipType"] for record in result]
            
            rel_counts = {}
            for rel_type in rel_types:
                result = session.run(
                    f"MATCH ()-[r:`{rel_type}`]->() RETURN count(r) AS count"
                )
                count = result.single()["count"]
                rel_counts[rel_type] = count
            
            # Sample some customer journeys
            result = session.run("MATCH (c:Customer)-[r]->(o) "
                                "WITH c, collect({rel: type(r), target: labels(o)[0], targetId: o.id, props: properties(r)}) AS rels "
                                "RETURN c.customer_id AS customer_id, rels "
                                "LIMIT 5")
            
            journeys = []
            for record in result:
                journeys.append({
                    "customer_id": record["customer_id"],
                    "relationships": record["rels"]
                })
            
            data = {
                "node_counts": node_counts,
                "relationship_counts": rel_counts,
                "sample_journeys": journeys
            }
            
            # Write to file
            with open('neo4j_data_summary.json', 'w') as f:
                json.dump(data, f, indent=2)
                
            logging.info(f"Data summary written to neo4j_data_summary.json")
            return True
                
    except Exception as e:
        logging.error(f"Error exploring Neo4j data: {e}")
        return False
    finally:
        if 'driver' in locals():
            driver.close()

if __name__ == "__main__":
    print("Exploring Neo4j database data...")
    if explore_neo4j_data():
        print("Data exploration completed successfully!")
    else:
        print("Data exploration failed. Check logs for details.")
