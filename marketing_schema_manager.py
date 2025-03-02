#!/usr/bin/env python3
"""
Neo4j schema management utilities for the marketing ontology.
This script provides utilities for schema definition, validation, and visualization.
"""

import os
import json
import logging
import datetime
from typing import Dict, List, Optional, Any, Set
from neo4j import GraphDatabase

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('schema_manager.log')
    ]
)

class MarketingSchemaManager:
    """Manages the Neo4j schema for the marketing ontology."""
    
    def __init__(self, uri, username, password, database):
        """Initialize the schema manager with Neo4j connection details."""
        self.uri = uri
        self.username = username
        self.password = password
        self.database = database
        self.driver = None
    
    def connect(self):
        """Connect to Neo4j."""
        try:
            self.driver = GraphDatabase.driver(self.uri, auth=(self.username, self.password))
            with self.driver.session(database=self.database) as session:
                result = session.run("RETURN 1 AS test")
                record = result.single()
                if not record or record["test"] != 1:
                    logging.error("Failed to connect to Neo4j")
                    return False
                
                logging.info("Successfully connected to Neo4j")
                return True
        except Exception as e:
            logging.error(f"Error connecting to Neo4j: {e}")
            return False
    
    def close(self):
        """Close the Neo4j connection."""
        if self.driver:
            self.driver.close()
            self.driver = None
    
    def extract_schema(self) -> Dict[str, Any]:
        """
        Extract the current schema from the Neo4j database.
        Returns a dictionary containing node labels, relationship types, properties, etc.
        """
        if not self.driver:
            if not self.connect():
                return {}
        
        try:
            schema = {
                "node_labels": [],
                "relationship_types": [],
                "node_property_keys": {},
                "relationship_property_keys": {},
                "constraints": [],
                "indexes": [],
                "patterns": []
            }
            
            with self.driver.session(database=self.database) as session:
                # Get node labels
                labels_result = session.run("CALL db.labels()")
                schema["node_labels"] = [record["label"] for record in labels_result]
                
                # Get relationship types
                rel_types_result = session.run("CALL db.relationshipTypes()")
                schema["relationship_types"] = [record["relationshipType"] for record in rel_types_result]
                
                # Get property keys for each node label
                for label in schema["node_labels"]:
                    property_result = session.run(f"""
                    MATCH (n:`{label}`) 
                    UNWIND keys(n) AS key
                    RETURN DISTINCT key
                    ORDER BY key
                    """)
                    
                    schema["node_property_keys"][label] = [record["key"] for record in property_result]
                
                # Get property keys for each relationship type
                for rel_type in schema["relationship_types"]:
                    property_result = session.run(f"""
                    MATCH ()-[r:`{rel_type}`]->() 
                    UNWIND keys(r) AS key
                    RETURN DISTINCT key
                    ORDER BY key
                    """)
                    
                    schema["relationship_property_keys"][rel_type] = [record["key"] for record in property_result]
                
                # Get constraints
                if session.run("CALL dbms.components() YIELD versions").single()["versions"][0].startswith("5"):
                    # Neo4j 5.x
                    constraints_result = session.run("SHOW CONSTRAINTS")
                    for record in constraints_result:
                        constraint = {
                            "name": record.get("name", ""),
                            "type": record.get("type", ""),
                            "entity_type": record.get("entityType", ""),
                            "labelsOrTypes": record.get("labelsOrTypes", []),
                            "properties": record.get("properties", [])
                        }
                        schema["constraints"].append(constraint)
                else:
                    # Neo4j 4.x
                    constraints_result = session.run("CALL db.constraints()")
                    for record in constraints_result:
                        constraint = {
                            "name": record.get("name", ""),
                            "description": record.get("description", "")
                        }
                        schema["constraints"].append(constraint)
                
                # Get indexes
                if session.run("CALL dbms.components() YIELD versions").single()["versions"][0].startswith("5"):
                    # Neo4j 5.x
                    indexes_result = session.run("SHOW INDEXES")
                    for record in indexes_result:
                        index = {
                            "name": record.get("name", ""),
                            "type": record.get("type", ""),
                            "entity_type": record.get("entityType", ""),
                            "labelsOrTypes": record.get("labelsOrTypes", []),
                            "properties": record.get("properties", [])
                        }
                        schema["indexes"].append(index)
                else:
                    # Neo4j 4.x
                    indexes_result = session.run("CALL db.indexes()")
                    for record in indexes_result:
                        index = {
                            "name": record.get("name", ""),
                            "description": record.get("description", "")
                        }
                        schema["indexes"].append(index)
                
                # Get common relationship patterns
                patterns_result = session.run("""
                MATCH (a)-[r]->(b)
                WITH labels(a)[0] AS from_label, type(r) AS rel_type, labels(b)[0] AS to_label, count(*) AS frequency
                WHERE frequency > 10
                RETURN from_label, rel_type, to_label, frequency
                ORDER BY frequency DESC
                LIMIT 50
                """)
                
                for record in patterns_result:
                    pattern = {
                        "from_label": record["from_label"],
                        "relationship_type": record["rel_type"],
                        "to_label": record["to_label"],
                        "frequency": record["frequency"]
                    }
                    schema["patterns"].append(pattern)
            
            # Add metadata
            schema["metadata"] = {
                "extracted_at": datetime.datetime.now().isoformat(),
                "database": self.database,
                "node_count": len(schema["node_labels"]),
                "relationship_type_count": len(schema["relationship_types"])
            }
            
            return schema
        
        except Exception as e:
            logging.error(f"Error extracting schema: {e}")
            return {}
    
    def generate_schema_cypher(self, schema: Dict[str, Any]) -> str:
        """
        Generate Cypher statements to recreate the schema.
        Takes a schema dictionary (from extract_schema) and returns Cypher script.
        """
        cypher_statements = []
        
        # Add comments and header
        cypher_statements.append("// Marketing Ontology Schema")
        cypher_statements.append(f"// Generated: {datetime.datetime.now().isoformat()}")
        cypher_statements.append("// Node Labels: " + ", ".join(schema.get("node_labels", [])))
        cypher_statements.append("// Relationship Types: " + ", ".join(schema.get("relationship_types", [])))
        cypher_statements.append("")
        
        # Add constraint statements
        cypher_statements.append("// Constraints")
        for label in schema.get("node_labels", []):
            # Skip if no properties
            if label not in schema.get("node_property_keys", {}):
                continue
                
            properties = schema.get("node_property_keys", {}).get(label, [])
            id_field = "customer_id" if label == "Customer" else "id"
            
            if id_field in properties:
                cypher_statements.append(f"CREATE CONSTRAINT {label.lower()}_{id_field} IF NOT EXISTS")
                cypher_statements.append(f"FOR (n:{label}) REQUIRE n.{id_field} IS UNIQUE;")
        
        cypher_statements.append("")
        
        # Add index statements
        cypher_statements.append("// Indexes")
        for label in schema.get("node_labels", []):
            if label not in schema.get("node_property_keys", {}):
                continue
                
            properties = schema.get("node_property_keys", {}).get(label, [])
            for prop in properties:
                # Only index certain properties that might be queried frequently
                # Skip primary keys (already have constraint) and metadata fields
                if prop in ['id', 'customer_id', 'timestamp', 'created_at', 'updated_at']:
                    continue
                    
                # Only index searchable properties
                if prop in ['name', 'email', 'type', 'category', 'status']:
                    cypher_statements.append(f"CREATE INDEX {label.lower()}_{prop}_idx IF NOT EXISTS")
                    cypher_statements.append(f"FOR (n:{label}) ON (n.{prop});")
        
        cypher_statements.append("")
        
        # Add example queries based on patterns
        cypher_statements.append("// Example Queries")
        seen_patterns = set()
        for pattern in schema.get("patterns", []):
            from_label = pattern.get("from_label")
            rel_type = pattern.get("relationship_type")
            to_label = pattern.get("to_label")
            
            pattern_key = f"{from_label}-{rel_type}-{to_label}"
            if pattern_key in seen_patterns:
                continue
                
            seen_patterns.add(pattern_key)
            
            # Add example query
            cypher_statements.append(f"// Query {from_label}-[{rel_type}]->{to_label} pattern")
            cypher_statements.append(f"MATCH (a:{from_label})-[r:{rel_type}]->(b:{to_label})")
            cypher_statements.append("RETURN a, r, b LIMIT 10;")
            cypher_statements.append("")
        
        return "\n".join(cypher_statements)
    
    def save_schema(self, filepath="marketing_schema.json"):
        """Extract and save the schema to a JSON file."""
        schema = self.extract_schema()
        if not schema:
            logging.error("Failed to extract schema")
            return False
        
        try:
            with open(filepath, 'w') as f:
                json.dump(schema, f, indent=2)
            
            # Also generate and save Cypher statements
            cypher_path = filepath.replace('.json', '.cypher')
            cypher = self.generate_schema_cypher(schema)
            with open(cypher_path, 'w') as f:
                f.write(cypher)
            
            logging.info(f"Schema saved to {filepath} and {cypher_path}")
            return True
        
        except Exception as e:
            logging.error(f"Error saving schema: {e}")
            return False
    
    def validate_triples(self, triples_file: str) -> Dict[str, Any]:
        """
        Validate a triples JSON file against the schema.
        Returns validation results with errors and warnings.
        """
        try:
            # Load triples
            with open(triples_file, 'r') as f:
                data = json.load(f)
                triples = data.get('triples', [])
            
            if not triples:
                return {"success": False, "error": "No triples found in file"}
            
            # Extract schema
            schema = self.extract_schema()
            if not schema:
                return {"success": False, "error": "Failed to extract schema from database"}
            
            # Setup validation results
            results = {
                "success": True,
                "total_triples": len(triples),
                "valid_triples": 0,
                "errors": [],
                "warnings": [],
                "node_types": {},
                "relationship_types": {}
            }
            
            # Get valid node types and relationship types from schema
            valid_node_types = set(schema.get("node_labels", []))
            valid_rel_types = set(schema.get("relationship_types", []))
            
            # Validate each triple
            for i, triple in enumerate(triples):
                triple_errors = []
                triple_warnings = []
                
                # Extract and validate subject
                subject = triple.get('subject', {})
                subject_type = subject.get('type')
                subject_id = subject.get('id')
                subject_props = subject.get('properties', {})
                
                if not subject:
                    triple_errors.append(f"Missing 'subject' in triple {i}")
                
                if not subject_type:
                    triple_errors.append(f"Missing 'type' in subject of triple {i}")
                elif subject_type not in valid_node_types:
                    triple_warnings.append(f"Unknown subject node type '{subject_type}' in triple {i}")
                
                if not subject_id:
                    triple_errors.append(f"Missing 'id' in subject of triple {i}")
                
                # Track node types
                if subject_type:
                    if subject_type not in results["node_types"]:
                        results["node_types"][subject_type] = 1
                    else:
                        results["node_types"][subject_type] += 1
                
                # Extract and validate relationship
                relationship = triple.get('relationship', {})
                rel_type = relationship.get('type')
                rel_props = relationship.get('properties', {})
                
                if not relationship:
                    triple_errors.append(f"Missing 'relationship' in triple {i}")
                
                if not rel_type:
                    triple_errors.append(f"Missing 'type' in relationship of triple {i}")
                elif rel_type not in valid_rel_types:
                    triple_warnings.append(f"Unknown relationship type '{rel_type}' in triple {i}")
                
                # Track relationship types
                if rel_type:
                    if rel_type not in results["relationship_types"]:
                        results["relationship_types"][rel_type] = 1
                    else:
                        results["relationship_types"][rel_type] += 1
                
                # Extract and validate object
                object_ = triple.get('object', {})
                object_type = object_.get('type')
                object_id = object_.get('id')
                object_props = object_.get('properties', {})
                
                if not object_:
                    triple_errors.append(f"Missing 'object' in triple {i}")
                
                if not object_type:
                    triple_errors.append(f"Missing 'type' in object of triple {i}")
                elif object_type not in valid_node_types:
                    triple_warnings.append(f"Unknown object node type '{object_type}' in triple {i}")
                
                if not object_id:
                    triple_errors.append(f"Missing 'id' in object of triple {i}")
                
                # Track node types
                if object_type:
                    if object_type not in results["node_types"]:
                        results["node_types"][object_type] = 1
                    else:
                        results["node_types"][object_type] += 1
                
                # Validate that id fields match schema constraints
                if subject_type == "Customer" and "customer_id" not in subject_props:
                    triple_warnings.append(f"Customer node in triple {i} should have 'customer_id' property")
                
                # Check for timestamp in relationship
                if "timestamp" not in rel_props:
                    triple_warnings.append(f"Relationship in triple {i} is missing 'timestamp' property")
                
                # Add errors and warnings to results
                if triple_errors:
                    results["errors"].extend(triple_errors)
                
                if triple_warnings:
                    results["warnings"].extend(triple_warnings)
                
                # Count valid triples
                if not triple_errors:
                    results["valid_triples"] += 1
            
            # Overall success
            if results["errors"]:
                results["success"] = False
            
            # Add validation metadata
            results["metadata"] = {
                "validated_at": datetime.datetime.now().isoformat(),
                "file": triples_file,
                "database": self.database
            }
            
            return results
        
        except Exception as e:
            logging.error(f"Error validating triples: {e}")
            return {"success": False, "error": str(e)}
    
    def visualize_schema(self, output_file="schema_visualization.html"):
        """
        Generate a visual representation of the schema as an HTML file.
        Uses vis.js for an interactive graph visualization.
        """
        try:
            schema = self.extract_schema()
            if not schema:
                logging.error("Failed to extract schema")
                return False
            
            # Prepare nodes and edges for visualization
            nodes = []
            edges = []
            
            # Add nodes for each label
            for i, label in enumerate(schema.get("node_labels", [])):
                props = schema.get("node_property_keys", {}).get(label, [])
                props_str = "<br>".join(props)
                
                nodes.append({
                    "id": label,
                    "label": label,
                    "title": f"<strong>{label}</strong><br>{props_str}",
                    "group": "nodes"
                })
            
            # Add edges for each relationship pattern
            edge_keys = set()
            for pattern in schema.get("patterns", []):
                from_label = pattern.get("from_label")
                rel_type = pattern.get("relationship_type")
                to_label = pattern.get("to_label")
                frequency = pattern.get("frequency", 0)
                
                # Avoid duplicate edges
                edge_key = f"{from_label}-{rel_type}-{to_label}"
                if edge_key in edge_keys:
                    continue
                
                edge_keys.add(edge_key)
                
                # Get relationship properties
                props = schema.get("relationship_property_keys", {}).get(rel_type, [])
                props_str = "<br>".join(props)
                
                edges.append({
                    "from": from_label,
                    "to": to_label,
                    "label": rel_type,
                    "title": f"<strong>{rel_type}</strong><br>{props_str}<br>Count: {frequency}",
                    "arrows": "to"
                })
            
            # Create HTML template with vis.js
            html = """
            <!DOCTYPE html>
            <html>
            <head>
                <title>Marketing Ontology Schema</title>
                <script type="text/javascript" src="https://unpkg.com/vis-network/standalone/umd/vis-network.min.js"></script>
                <style type="text/css">
                    body {
                        font-family: Arial, sans-serif;
                        margin: 0;
                        padding: 0;
                    }
                    #header {
                        background-color: #333;
                        color: white;
                        padding: 10px 20px;
                    }
                    #container {
                        display: flex;
                        height: calc(100vh - 60px);
                    }
                    #network {
                        flex: 3;
                        height: 100%;
                    }
                    #details {
                        flex: 1;
                        padding: 20px;
                        overflow-y: auto;
                        border-left: 1px solid #ccc;
                    }
                    .stats {
                        margin-bottom: 20px;
                    }
                    table {
                        width: 100%;
                        border-collapse: collapse;
                    }
                    th, td {
                        padding: 8px;
                        text-align: left;
                        border-bottom: 1px solid #ddd;
                    }
                    th {
                        background-color: #f2f2f2;
                    }
                </style>
            </head>
            <body>
                <div id="header">
                    <h2>Marketing Ontology Schema Visualization</h2>
                </div>
                <div id="container">
                    <div id="network"></div>
                    <div id="details">
                        <div class="stats">
                            <h3>Schema Statistics</h3>
                            <p>Node Labels: {{node_count}}</p>
                            <p>Relationship Types: {{rel_count}}</p>
                            <p>Generated: {{timestamp}}</p>
                        </div>
                        
                        <div class="node-labels">
                            <h3>Node Labels</h3>
                            <table>
                                <tr>
                                    <th>Label</th>
                                    <th>Properties</th>
                                </tr>
                                {{node_rows}}
                            </table>
                        </div>
                        
                        <div class="rel-types">
                            <h3>Relationship Types</h3>
                            <table>
                                <tr>
                                    <th>Type</th>
                                    <th>Properties</th>
                                </tr>
                                {{rel_rows}}
                            </table>
                        </div>
                    </div>
                </div>

                <script type="text/javascript">
                    // Create nodes and edges
                    var nodes = new vis.DataSet({{nodes_json}});
                    var edges = new vis.DataSet({{edges_json}});

                    // Create network
                    var container = document.getElementById('network');
                    var data = {
                        nodes: nodes,
                        edges: edges
                    };
                    var options = {
                        nodes: {
                            shape: 'ellipse',
                            color: {
                                background: '#4CAF50',
                                border: '#388E3C',
                                highlight: {
                                    background: '#81C784',
                                    border: '#388E3C'
                                }
                            },
                            font: {
                                color: '#ffffff'
                            }
                        },
                        edges: {
                            color: {
                                color: '#999999',
                                highlight: '#666666'
                            },
                            font: {
                                size: 12,
                                color: '#343434',
                                background: '#ffffff'
                            }
                        },
                        physics: {
                            barnesHut: {
                                gravitationalConstant: -4000,
                                centralGravity: 0.3,
                                springLength: 150,
                                springConstant: 0.04
                            }
                        },
                        interaction: {
                            navigationButtons: true,
                            keyboard: true
                        }
                    };
                    var network = new vis.Network(container, data, options);
                </script>
            </body>
            </html>
            """
            
            # Generate node rows
            node_rows = ""
            for label in schema.get("node_labels", []):
                props = schema.get("node_property_keys", {}).get(label, [])
                props_str = ", ".join(props)
                node_rows += f"<tr><td>{label}</td><td>{props_str}</td></tr>"
            
            # Generate relationship rows
            rel_rows = ""
            for rel_type in schema.get("relationship_types", []):
                props = schema.get("relationship_property_keys", {}).get(rel_type, [])
                props_str = ", ".join(props)
                rel_rows += f"<tr><td>{rel_type}</td><td>{props_str}</td></tr>"
            
            # Replace placeholders
            html = html.replace("{{node_count}}", str(len(schema.get("node_labels", []))))
            html = html.replace("{{rel_count}}", str(len(schema.get("relationship_types", []))))
            html = html.replace("{{timestamp}}", datetime.datetime.now().isoformat())
            html = html.replace("{{node_rows}}", node_rows)
            html = html.replace("{{rel_rows}}", rel_rows)
            html = html.replace("{{nodes_json}}", json.dumps(nodes))
            html = html.replace("{{edges_json}}", json.dumps(edges))
            
            # Write to file
            with open(output_file, 'w') as f:
                f.write(html)
            
            logging.info(f"Schema visualization saved to {output_file}")
            return True
        
        except Exception as e:
            logging.error(f"Error visualizing schema: {e}")
            return False

def main():
    """Main function to demonstrate schema management capabilities."""
    # Neo4j connection parameters
    uri = os.getenv('NEO4J_URI', 'bolt://localhost:7687')
    username = os.getenv('NEO4J_USERNAME', 'neo4j')
    password = os.getenv('NEO4J_PASSWORD', '#1Moneymaker')
    database = os.getenv('NEO4J_DATABASE', 'marketing')
    
    # Create schema manager
    manager = MarketingSchemaManager(uri, username, password, database)
    
    try:
        # Connect to Neo4j
        if not manager.connect():
            logging.error("Failed to connect to Neo4j")
            return
        
        # Extract and save schema
        manager.save_schema("marketing_schema.json")
        
        # Generate schema visualization
        manager.visualize_schema("marketing_schema.html")
        
        # Validate triples files
        for triples_file in ["customer_journey_triples.json", "enhanced_customer_journey_triples.json"]:
            if os.path.exists(triples_file):
                results = manager.validate_triples(triples_file)
                logging.info(f"Validation of {triples_file}: {results['success']}")
                
                # Save validation results
                output_file = triples_file.replace('.json', '_validation.json')
                with open(output_file, 'w') as f:
                    json.dump(results, f, indent=2)
                
                logging.info(f"Validation results saved to {output_file}")
    
    finally:
        # Close Neo4j connection
        manager.close()

if __name__ == "__main__":
    main()