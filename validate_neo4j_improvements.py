#!/usr/bin/env python3
"""
Validation script for Neo4j data loading improvements.
Tests the improved node and relationship creation functionality.
"""

import os
import json
import datetime
import logging
from neo4j import GraphDatabase

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('validation.log')
    ]
)

# Test cases for Neo4j nodes
TEST_NODES = [
    # Valid node with complete data
    {
        "type": "Customer",
        "id": "TEST001",
        "properties": {
            "customer_id": "TEST001",
            "name": "Test Customer",
            "email": "test@example.com"
        }
    },
    # Node with empty ID that should be handled
    {
        "type": "Customer",
        "id": "",
        "properties": {
            "customer_id": "",
            "name": "Empty ID Customer"
        }
    },
    # Node with null properties that should be cleaned
    {
        "type": "Product",
        "id": "PROD001",
        "properties": {
            "name": "Test Product",
            "description": None,
            "price": 99.99,
            "category": "",
            "in_stock": True
        }
    }
]

# Test cases for Neo4j relationships
TEST_RELATIONSHIPS = [
    # Valid relationship
    {
        "from_type": "Customer",
        "from_id": "TEST001",
        "to_type": "Product",
        "to_id": "PROD001",
        "type": "PURCHASES",
        "properties": {
            "timestamp": datetime.datetime.now().isoformat(),
            "amount": 99.99
        }
    },
    # Relationship with missing timestamp that should be added
    {
        "from_type": "Customer",
        "from_id": "TEST001", 
        "to_type": "Segment",
        "to_id": "Test-Segment",
        "type": "BELONGS_TO",
        "properties": {
            "is_primary": True
        }
    },
    # Relationship with null/empty properties that should be cleaned
    {
        "from_type": "Customer",
        "from_id": "TEST001",
        "to_type": "Content",
        "to_id": "CONTENT001",
        "type": "INTERACTS_WITH",
        "properties": {
            "action": "view",
            "duration": None,
            "rating": "",
            "comment": None
        }
    },
    # Relationship with empty IDs that should be skipped
    {
        "from_type": "Customer", 
        "from_id": "TEST001",
        "to_type": "Email",
        "to_id": "",
        "type": "HAS_EMAIL",
        "properties": {
            "is_primary": True
        }
    }
]

def run_validation_tests(uri, username, password, database):
    """Run validation tests for the Neo4j data loading improvements."""
    try:
        # Connect to Neo4j
        driver = GraphDatabase.driver(uri, auth=(username, password))
        with driver.session(database=database) as session:
            # Test connection
            result = session.run("RETURN 1 AS test")
            if not result.single() or result.single()["test"] != 1:
                logging.error("Failed to connect to Neo4j")
                return False
            
            logging.info("Successfully connected to Neo4j")
            
            # Import improved functions
            from neo4j_data_loader import create_neo4j_node, create_neo4j_relationship, validate_triple, enrich_node_properties
            
            # Create a test tag to identify our test nodes/relationships
            test_tag = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
            
            # Test node creation
            logging.info("Testing node creation...")
            node_results = []
            for i, node in enumerate(TEST_NODES):
                try:
                    # Add test tag to properties
                    if "properties" not in node:
                        node["properties"] = {}
                    node["properties"]["test_tag"] = test_tag
                    
                    # Create node
                    result = session.execute_write(
                        create_neo4j_node,
                        node["type"],
                        node["id"],
                        node["properties"]
                    )
                    
                    # Store result
                    success = result is not None
                    node_results.append({
                        "test_case": i,
                        "node_type": node["type"],
                        "node_id": node["id"],
                        "success": success,
                        "result": dict(result) if result else None
                    })
                    
                    logging.info(f"Node test {i}: {'Success' if success else 'Failure'}")
                
                except Exception as e:
                    logging.error(f"Error in node test {i}: {e}")
                    node_results.append({
                        "test_case": i,
                        "node_type": node["type"],
                        "node_id": node["id"],
                        "success": False,
                        "error": str(e)
                    })
            
            # Test relationship creation
            logging.info("Testing relationship creation...")
            rel_results = []
            for i, rel in enumerate(TEST_RELATIONSHIPS):
                try:
                    # Add test tag to properties
                    if "properties" not in rel:
                        rel["properties"] = {}
                    rel["properties"]["test_tag"] = test_tag
                    
                    # Create relationship
                    result = session.execute_write(
                        create_neo4j_relationship,
                        rel["from_type"],
                        rel["from_id"],
                        rel["to_type"],
                        rel["to_id"],
                        rel["type"],
                        rel["properties"]
                    )
                    
                    # Store result
                    success = result is not None
                    rel_results.append({
                        "test_case": i,
                        "from_type": rel["from_type"],
                        "to_type": rel["to_type"],
                        "rel_type": rel["type"],
                        "success": success,
                        "result": dict(result) if result else None
                    })
                    
                    logging.info(f"Relationship test {i}: {'Success' if success else 'Failure'}")
                
                except Exception as e:
                    logging.error(f"Error in relationship test {i}: {e}")
                    rel_results.append({
                        "test_case": i,
                        "from_type": rel["from_type"],
                        "to_type": rel["to_type"],
                        "rel_type": rel["type"],
                        "success": False,
                        "error": str(e)
                    })
            
            # Verify created data
            verify_query = f"""
            MATCH (n)
            WHERE n.test_tag = $test_tag
            RETURN labels(n)[0] as label, count(n) as count
            """
            verify_nodes = session.run(verify_query, {"test_tag": test_tag})
            
            node_counts = {}
            for record in verify_nodes:
                node_counts[record["label"]] = record["count"]
            
            verify_rels_query = f"""
            MATCH ()-[r]->()
            WHERE r.test_tag = $test_tag
            RETURN type(r) as type, count(r) as count
            """
            verify_rels = session.run(verify_rels_query, {"test_tag": test_tag})
            
            rel_counts = {}
            for record in verify_rels:
                rel_counts[record["type"]] = record["count"]
            
            # Compile results
            validation_results = {
                "timestamp": datetime.datetime.now().isoformat(),
                "test_tag": test_tag,
                "node_tests": {
                    "total": len(TEST_NODES),
                    "successful": sum(1 for r in node_results if r["success"]),
                    "failed": sum(1 for r in node_results if not r["success"]),
                    "results": node_results
                },
                "relationship_tests": {
                    "total": len(TEST_RELATIONSHIPS),
                    "successful": sum(1 for r in rel_results if r["success"]),
                    "failed": sum(1 for r in rel_results if not r["success"]),
                    "results": rel_results
                },
                "verification": {
                    "node_counts": node_counts,
                    "relationship_counts": rel_counts
                }
            }
            
            # Save results to file
            with open('validation_results.json', 'w') as f:
                json.dump(validation_results, f, indent=2)
            
            logging.info("Validation tests completed")
            logging.info(f"Node tests: {validation_results['node_tests']['successful']}/{validation_results['node_tests']['total']} successful")
            logging.info(f"Relationship tests: {validation_results['relationship_tests']['successful']}/{validation_results['relationship_tests']['total']} successful")
            
            # Cleanup test data
            if os.getenv('KEEP_TEST_DATA', 'false').lower() != 'true':
                cleanup_query = f"""
                MATCH (n)
                WHERE n.test_tag = $test_tag
                DETACH DELETE n
                """
                session.run(cleanup_query, {"test_tag": test_tag})
                logging.info("Test data cleaned up")
            
            return validation_results
        
    except Exception as e:
        logging.error(f"Error in validation tests: {e}")
        return None
    finally:
        if 'driver' in locals():
            driver.close()

if __name__ == "__main__":
    # Neo4j connection parameters
    uri = os.getenv('NEO4J_URI', 'bolt://localhost:7687')
    username = os.getenv('NEO4J_USERNAME', 'neo4j')
    password = os.getenv('NEO4J_PASSWORD', '#1Moneymaker')
    database = os.getenv('NEO4J_DATABASE', 'marketing')
    
    results = run_validation_tests(uri, username, password, database)
    
    if results:
        node_success_rate = results['node_tests']['successful'] / results['node_tests']['total'] * 100
        rel_success_rate = results['relationship_tests']['successful'] / results['relationship_tests']['total'] * 100
        
        print("\nValidation Test Results:")
        print("------------------------")
        print(f"Node tests: {results['node_tests']['successful']}/{results['node_tests']['total']} successful ({node_success_rate:.1f}%)")
        print(f"Relationship tests: {results['relationship_tests']['successful']}/{results['relationship_tests']['total']} successful ({rel_success_rate:.1f}%)")
        print(f"Created nodes: {sum(results['verification']['node_counts'].values())}")
        print(f"Created relationships: {sum(results['verification']['relationship_counts'].values())}")
        print("\nDetailed results saved to validation_results.json")
    else:
        print("Validation tests failed. Check logs for details.")