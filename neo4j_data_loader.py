#!/usr/bin/env python3
"""
Neo4j data loader for marketing ontology.
This script loads the processed customer journey triples into Neo4j.
"""

import os
import json
import logging
import datetime
from typing import Dict, List, Optional, Any, Tuple
from neo4j import GraphDatabase

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('neo4j_loader.log')
    ]
)

def load_journey_triples(file_path):
    """Load journey triples from a JSON file."""
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
            return data.get('triples', [])
    except Exception as e:
        logging.error(f"Error loading journey triples: {e}")
        return []

def apply_neo4j_constraints(driver, database, constraints_file):
    """Apply Neo4j constraints and indexes from a Cypher file."""
    try:
        with open(constraints_file, 'r') as f:
            constraints = f.read().strip().split('\n')
        
        with driver.session(database=database) as session:
            for constraint in constraints:
                if constraint:
                    logging.info(f"Applying constraint: {constraint}")
                    session.run(constraint)
            
            logging.info("Applied all Neo4j constraints and indexes")
            return True
    except Exception as e:
        logging.error(f"Error applying Neo4j constraints: {e}")
        return False

def create_neo4j_node(tx, node_type, node_id, properties):
    """Create a node in Neo4j with proper data validation and handling."""
    # Handle empty or null node_id
    if not node_id or node_id == "null" or node_id == "undefined":
        logging.warning(f"Skipping node with empty/null ID of type {node_type}")
        return None
    
    # Handle special case for Customer nodes (using customer_id instead of id)
    id_field = "customer_id" if node_type == "Customer" else "id"
    
    # Sanitize properties: remove None/null/empty values
    clean_props = {k: v for k, v in properties.items() if v is not None and v != ""}
    
    # Create property string with proper typing
    properties_str = ', '.join([f"n.{k} = ${k}" for k in clean_props.keys()])
    
    # Build the query with proper error handling
    query = f"""
    MERGE (n:{node_type} {{{id_field}: $id}})
    {f"SET {properties_str}" if properties_str else ""}
    RETURN n
    """
    
    params = {"id": node_id, **clean_props}
    
    try:
        result = tx.run(query, params)
        return result.single()
    except Exception as e:
        logging.error(f"Error creating node {node_type} with ID {node_id}: {e}")
        return None

def create_neo4j_relationship(tx, from_type, from_id, to_type, to_id, rel_type, properties):
    """Create a relationship in Neo4j with proper data validation and handling."""
    # Skip if any required fields are missing
    if not from_id or not to_id or not rel_type:
        logging.warning(f"Skipping relationship {rel_type} due to missing IDs: {from_id} -> {to_id}")
        return None
    
    # Handle special case for Customer nodes (using customer_id instead of id)
    from_id_field = "customer_id" if from_type == "Customer" else "id"
    to_id_field = "customer_id" if to_type == "Customer" else "id"
    
    # Sanitize properties: remove None/null/empty values
    clean_props = {k: v for k, v in properties.items() if v is not None and v != ""}
    
    # Add timestamp if not present
    if "timestamp" not in clean_props:
        clean_props["timestamp"] = datetime.datetime.now().isoformat()
    
    # Create property strings with proper typing
    properties_str = ', '.join([f"r.{k} = ${k}" for k in clean_props.keys()])
    
    # Build query with existence checks and improved error handling
    query = f"""
    MATCH (from:{from_type} {{{from_id_field}: $from_id}})
    MATCH (to:{to_type} {{{to_id_field}: $to_id}})
    MERGE (from)-[r:{rel_type}]->(to)
    {f"ON CREATE SET {properties_str}" if properties_str else ""}
    {f"ON MATCH SET {properties_str}" if properties_str else ""}
    RETURN r
    """
    
    params = {"from_id": from_id, "to_id": to_id, **clean_props}
    
    try:
        result = tx.run(query, params)
        return result.single()
    except Exception as e:
        logging.error(f"Error creating relationship {from_type}({from_id})-[{rel_type}]->{to_type}({to_id}): {e}")
        return None

def validate_triple(triple: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
    """
    Validate a triple to ensure it has all necessary components.
    Returns (is_valid, error_message).
    """
    if not triple.get('subject'):
        return False, "Triple is missing subject"
    
    if not triple.get('relationship'):
        return False, "Triple is missing relationship"
    
    if not triple.get('object'):
        return False, "Triple is missing object"
    
    subject = triple.get('subject', {})
    if not subject.get('type'):
        return False, "Subject is missing type"
    
    object_ = triple.get('object', {})
    if not object_.get('type'):
        return False, "Object is missing type"
    
    relationship = triple.get('relationship', {})
    if not relationship.get('type'):
        return False, "Relationship is missing type"
    
    return True, None

def enrich_node_properties(node_data: Dict[str, Any], node_type: str) -> Dict[str, Any]:
    """
    Add missing properties to nodes based on their type.
    """
    properties = node_data.get('properties', {}).copy()
    
    # Add timestamp if not present
    if 'timestamp' not in properties:
        properties['timestamp'] = datetime.datetime.now().isoformat()
    
    # Add type-specific default properties
    if node_type == 'Customer':
        if 'lifetime_value' not in properties:
            properties['lifetime_value'] = 0
        if 'is_churned' not in properties:
            properties['is_churned'] = False
        if 'name' not in properties and 'customer_id' in properties:
            properties['name'] = f"Customer {properties['customer_id']}"
    
    # Add consistent label/name properties
    if node_type in ['Segment', 'Persona', 'BehaviorStage'] and 'name' not in properties and 'id' in node_data:
        properties['name'] = node_data['id']
    
    return properties

def load_triples_into_neo4j(driver, database, triples):
    """Load journey triples into Neo4j with improved validation and enrichment."""
    try:
        # Tracking dictionaries
        unique_subjects = {}  # Store unique subject nodes
        unique_objects = {}   # Store unique object nodes
        relationships = []    # Store all relationships
        validation_errors = 0 # Count validation errors
        
        # First pass: validate and collect all unique subjects and objects
        for i, triple in enumerate(triples):
            # Validate the triple structure
            is_valid, error = validate_triple(triple)
            if not is_valid:
                logging.warning(f"Skipping invalid triple at index {i}: {error}")
                validation_errors += 1
                continue
            
            subject = triple.get('subject', {})
            object_ = triple.get('object', {})
            relationship = triple.get('relationship', {})
            
            subject_id = subject.get('id')
            subject_type = subject.get('type')
            subject_props = subject.get('properties', {})
            
            object_id = object_.get('id')
            object_type = object_.get('type')
            object_props = object_.get('properties', {})
            
            # Store and enrich unique subjects
            if subject_id and subject_type:
                key = f"{subject_type}:{subject_id}"
                if key not in unique_subjects:
                    enriched_props = enrich_node_properties(subject, subject_type)
                    unique_subjects[key] = {
                        'type': subject_type,
                        'id': subject_id,
                        'properties': enriched_props
                    }
                elif subject_props:  # Merge properties from duplicate entries
                    for k, v in subject_props.items():
                        if v and v != "null" and v != "undefined":
                            unique_subjects[key]['properties'][k] = v
            
            # Store and enrich unique objects
            if object_id and object_type:
                key = f"{object_type}:{object_id}"
                if key not in unique_objects:
                    enriched_props = enrich_node_properties(object_, object_type)
                    unique_objects[key] = {
                        'type': object_type,
                        'id': object_id,
                        'properties': enriched_props
                    }
                elif object_props:  # Merge properties from duplicate entries
                    for k, v in object_props.items():
                        if v and v != "null" and v != "undefined":
                            unique_objects[key]['properties'][k] = v
            
            # Store relationship with enriched properties
            if relationship.get('type'):
                rel_props = relationship.get('properties', {}).copy()
                
                # Add timestamp if not present
                if 'timestamp' not in rel_props:
                    rel_props['timestamp'] = datetime.datetime.now().isoformat()
                
                relationships.append({
                    'from_type': subject_type,
                    'from_id': subject_id,
                    'to_type': object_type,
                    'to_id': object_id,
                    'type': relationship.get('type'),
                    'properties': rel_props
                })
        
        if validation_errors > 0:
            logging.warning(f"Found {validation_errors} invalid triples out of {len(triples)}")
        
        # Batch create all nodes first
        logging.info(f"Creating {len(unique_subjects)} subject nodes and {len(unique_objects)} object nodes")
        with driver.session(database=database) as session:
            # Track successful/failed creations
            node_successes = 0
            node_failures = 0
            rel_successes = 0
            rel_failures = 0
            
            # Create subject nodes in batches
            batch_size = 100
            subject_batches = [list(unique_subjects.values())[i:i+batch_size] 
                            for i in range(0, len(unique_subjects), batch_size)]
            
            for i, batch in enumerate(subject_batches):
                logging.info(f"Creating subject nodes batch {i+1}/{len(subject_batches)}")
                for subject in batch:
                    result = session.execute_write(
                        create_neo4j_node,
                        subject['type'],
                        subject['id'],
                        subject['properties']
                    )
                    if result:
                        node_successes += 1
                    else:
                        node_failures += 1
            
            # Create object nodes in batches
            object_batches = [list(unique_objects.values())[i:i+batch_size] 
                            for i in range(0, len(unique_objects), batch_size)]
            
            for i, batch in enumerate(object_batches):
                logging.info(f"Creating object nodes batch {i+1}/{len(object_batches)}")
                for object_ in batch:
                    result = session.execute_write(
                        create_neo4j_node,
                        object_['type'],
                        object_['id'],
                        object_['properties']
                    )
                    if result:
                        node_successes += 1
                    else:
                        node_failures += 1
            
            # Create relationships in batches
            logging.info(f"Creating {len(relationships)} relationships")
            rel_batches = [relationships[i:i+batch_size] 
                        for i in range(0, len(relationships), batch_size)]
            
            for i, batch in enumerate(rel_batches):
                logging.info(f"Creating relationship batch {i+1}/{len(rel_batches)}")
                for rel in batch:
                    result = session.execute_write(
                        create_neo4j_relationship,
                        rel['from_type'],
                        rel['from_id'],
                        rel['to_type'],
                        rel['to_id'],
                        rel['type'],
                        rel['properties']
                    )
                    if result:
                        rel_successes += 1
                    else:
                        rel_failures += 1
        
        # Generate completion summary
        total_nodes = len(unique_subjects) + len(unique_objects)
        total_rels = len(relationships)
        
        logging.info(f"Node creation: {node_successes}/{total_nodes} successful ({node_failures} failures)")
        logging.info(f"Relationship creation: {rel_successes}/{total_rels} successful ({rel_failures} failures)")
        logging.info("Successfully loaded all valid triples into Neo4j")
        
        return True
    
    except Exception as e:
        logging.error(f"Error loading triples into Neo4j: {e}")
        return False

def verify_neo4j_load(driver, database):
    """Verify that data was loaded successfully into Neo4j with detailed statistics."""
    try:
        with driver.session(database=database) as session:
            # Get all node labels
            labels_result = session.run("CALL db.labels()")
            labels = [record["label"] for record in labels_result]
            
            # Count nodes by label with detailed properties
            node_stats = {}
            for label in labels:
                # Get count
                count_result = session.run(
                    f"MATCH (n:`{label}`) RETURN count(n) as count"
                )
                count = count_result.single()["count"]
                
                # Get property statistics
                property_result = session.run(f"""
                MATCH (n:`{label}`)
                WITH keys(n) AS properties, count(n) as nodes
                UNWIND properties AS property
                RETURN property, count(property) AS count
                ORDER BY count DESC
                """)
                
                properties = {}
                for record in property_result:
                    properties[record["property"]] = record["count"]
                
                # Get a sample of nodes if there are any
                sample_nodes = []
                if count > 0:
                    sample_result = session.run(f"""
                    MATCH (n:`{label}`)
                    RETURN n LIMIT 3
                    """)
                    
                    for record in sample_result:
                        node = record["n"]
                        sample_nodes.append(dict(node))
                
                node_stats[label] = {
                    "count": count,
                    "properties": properties,
                    "samples": sample_nodes
                }
            
            # Get all relationship types
            rel_types_result = session.run("CALL db.relationshipTypes()")
            rel_types = [record["relationshipType"] for record in rel_types_result]
            
            # Count relationships by type with detailed properties
            rel_stats = {}
            for rel_type in rel_types:
                # Get count
                count_result = session.run(
                    f"MATCH ()-[r:`{rel_type}`]->() RETURN count(r) as count"
                )
                count = count_result.single()["count"]
                
                # Get property statistics
                property_result = session.run(f"""
                MATCH ()-[r:`{rel_type}`]->()
                WITH keys(r) AS properties, count(r) as rels
                UNWIND properties AS property
                RETURN property, count(property) AS count
                ORDER BY count DESC
                """)
                
                properties = {}
                for record in property_result:
                    properties[record["property"]] = record["count"]
                
                # Get the most common node combinations
                combo_result = session.run(f"""
                MATCH (a)-[r:`{rel_type}`]->(b)
                RETURN labels(a)[0] AS from_label, labels(b)[0] AS to_label, count(*) AS combo_count
                ORDER BY combo_count DESC
                LIMIT 5
                """)
                
                combinations = []
                for record in combo_result:
                    combinations.append({
                        "from_label": record["from_label"],
                        "to_label": record["to_label"],
                        "count": record["combo_count"]
                    })
                
                rel_stats[rel_type] = {
                    "count": count,
                    "properties": properties,
                    "combinations": combinations
                }
            
            # Get connection statistics to identify potential data quality issues
            orphan_nodes_query = """
            MATCH (n)
            WHERE NOT (n)--()
            RETURN labels(n)[0] as label, count(*) as orphan_count
            ORDER BY orphan_count DESC
            """
            
            orphan_result = session.run(orphan_nodes_query)
            orphan_stats = {}
            for record in orphan_result:
                orphan_stats[record["label"]] = record["orphan_count"]
            
            # Compile complete verification object
            verification = {
                "timestamp": datetime.datetime.now().isoformat(),
                "database": database,
                "node_stats": node_stats,
                "relationship_stats": rel_stats,
                "orphan_nodes": orphan_stats,
                "summary": {
                    "total_nodes": sum(stats["count"] for stats in node_stats.values()),
                    "total_relationships": sum(stats["count"] for stats in rel_stats.values()),
                    "total_orphans": sum(orphan_stats.values())
                }
            }
            
            # Write detailed verification to file
            with open('neo4j_verification_detailed.json', 'w') as f:
                json.dump(verification, f, indent=2)
            
            # Also create a simplified summary version
            summary = {
                "node_counts": {label: stats["count"] for label, stats in node_stats.items()},
                "relationship_counts": {rel: stats["count"] for rel, stats in rel_stats.items()}
            }
            
            with open('neo4j_verification.json', 'w') as f:
                json.dump(summary, f, indent=2)
            
            logging.info(f"Verification complete: {verification['summary']['total_nodes']} nodes and {verification['summary']['total_relationships']} relationships")
            logging.info(f"Found {verification['summary']['total_orphans']} orphaned nodes with no relationships")
            
            return verification
    
    except Exception as e:
        logging.error(f"Error verifying Neo4j data load: {e}")
        return None

def run_neo4j_loader():
    """Run the Neo4j data loader."""
    try:
        # Neo4j connection parameters
        uri = os.getenv('NEO4J_URI', 'bolt://localhost:7687')
        username = os.getenv('NEO4J_USERNAME', 'neo4j')
        password = os.getenv('NEO4J_PASSWORD', 'neo4j')
        database = os.getenv('NEO4J_DATABASE', 'neo4j')
        
        # File paths
        triples_file = 'enhanced_customer_journey_triples.json'
        constraints_file = 'neo4j_constraints.cypher'
        
        # Check if files exist
        if not os.path.exists(triples_file):
            logging.error(f"Triples file not found: {triples_file}")
            return False
        
        if not os.path.exists(constraints_file):
            logging.error(f"Constraints file not found: {constraints_file}")
            return False
        
        # Load triples from file
        triples = load_journey_triples(triples_file)
        if not triples:
            logging.error("No triples loaded. Aborting.")
            return False
        
        logging.info(f"Loaded {len(triples)} triples from file")
        
        # Connect to Neo4j
        driver = GraphDatabase.driver(uri, auth=(username, password))
        
        try:
            # Test connection
            with driver.session(database=database) as session:
                result = session.run("RETURN 1 AS test")
                record = result.single()
                if not record or record["test"] != 1:
                    logging.error("Failed to connect to Neo4j")
                    return False
            
            logging.info("Successfully connected to Neo4j")
            
            # Apply constraints and indexes
            if not apply_neo4j_constraints(driver, database, constraints_file):
                logging.error("Failed to apply Neo4j constraints")
                return False
            
            # Load triples into Neo4j
            if not load_triples_into_neo4j(driver, database, triples):
                logging.error("Failed to load triples into Neo4j")
                return False
            
            # Verify data load
            verification = verify_neo4j_load(driver, database)
            if not verification:
                logging.warning("Failed to verify Neo4j data load")
            
            logging.info("Neo4j data loading completed successfully")
            return True
        
        finally:
            driver.close()
    
    except Exception as e:
        logging.error(f"Error in Neo4j data loader: {e}")
        return False

if __name__ == "__main__":
    print("Starting Neo4j data loader for marketing ontology...")
    if run_neo4j_loader():
        print("Neo4j data loading completed successfully!")
        print("Verification results saved to neo4j_verification.json")
    else:
        print("Neo4j data loading failed. Check logs for details.")