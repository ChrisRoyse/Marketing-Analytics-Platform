#!/usr/bin/env python3
#
# This script runs the Marketing Ontology Platform Demo
# It checks if Neo4j is running, if data is loaded, and starts the dashboard
#

import os
import sys
import subprocess

print("======================================")
print("  Marketing Ontology Platform Demo")
print("======================================")
print("")

# Check if .env file exists
if not os.path.exists(".env"):
    print("Creating default .env file...")
    with open(".env", "w") as f:
        f.write("""NEO4J_URI=bolt://172.19.160.1:7687
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=#1Moneymaker
NEO4J_DATABASE=marketing
DASHBOARD_PORT=8050
DASHBOARD_DEBUG=True
""")

# Check if Neo4j is running
print("Checking Neo4j connection...")
# Get connection details from .env file
neo4j_uri = None
neo4j_username = None
neo4j_password = None

with open(".env", "r") as f:
    for line in f:
        if line.startswith("NEO4J_URI="):
            neo4j_uri = line.strip().split("=", 1)[1]
        elif line.startswith("NEO4J_USERNAME="):
            neo4j_username = line.strip().split("=", 1)[1]
        elif line.startswith("NEO4J_PASSWORD="):
            neo4j_password = line.strip().split("=", 1)[1]

try:
    # Test Neo4j connection
    from neo4j import GraphDatabase
    driver = GraphDatabase.driver(neo4j_uri, auth=(neo4j_username, neo4j_password))
    with driver.session() as session:
        session.run("RETURN 1")
    driver.close()
    print("✓ Neo4j connection successful")
except Exception as e:
    print("✗ Cannot connect to Neo4j")
    print(f"Please ensure Neo4j is running at {neo4j_uri} with username '{neo4j_username}'")
    print("If Neo4j is running on a different host/port, edit the .env file")
    sys.exit(1)

# Check if demo data exists
if not os.path.exists("demo_data/customers.json"):
    print("Demo data not found at 'demo_data/customers.json'")
    print("Generating demo data...")
    try:
        subprocess.check_call(["python3", "generate_demo_data.py"])
        print("✓ Demo data generated")
    except subprocess.CalledProcessError:
        print("✗ Failed to generate demo data")
        sys.exit(1)

# Check if data is loaded in Neo4j
print("Checking if demo data is loaded in Neo4j...")
try:
    # Check if customers exist in database
    driver = GraphDatabase.driver(neo4j_uri, auth=(neo4j_username, neo4j_password))
    with driver.session() as session:
        result = session.run("MATCH (c:Customer) RETURN count(c) as count")
        count = result.single()["count"]
    driver.close()
    
    if count > 0:
        print(f"✓ Demo data found in Neo4j ({count} customers)")
    else:
        print("No demo data found in Neo4j database")
        print("Loading demo data...")
        try:
            subprocess.check_call(["python3", "load_demo_data.py"])
            print("✓ Demo data loaded")
        except subprocess.CalledProcessError:
            print("✗ Failed to load demo data")
            sys.exit(1)
except Exception as e:
    print(f"Error checking Neo4j data: {e}")
    sys.exit(1)

# Create assets directory if it doesn't exist
if not os.path.exists("demo_dashboard_assets"):
    os.makedirs("demo_dashboard_assets")

# Copy logo file if it doesn't exist
if not os.path.exists("demo_dashboard_assets/logo.png") and os.path.exists("demo_assets/logo.png"):
    try:
        import shutil
        shutil.copy("demo_assets/logo.png", "demo_dashboard_assets/logo.png")
    except Exception as e:
        print(f"Warning: Could not copy logo: {e}")

# Run the dashboard
print("")
print("Starting demo dashboard...")
print("======================================")
print("Access the dashboard at http://localhost:8050")
print("Press Ctrl+C to stop")
print("")

try:
    subprocess.check_call(["python3", "demo_dashboard.py"])
except KeyboardInterrupt:
    print("\nStopping dashboard...")
except subprocess.CalledProcessError:
    print("\n✗ Dashboard error")
    sys.exit(1)