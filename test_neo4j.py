#!/usr/bin/env python3
from neo4j import GraphDatabase
import os

# Check if we're using Windows localhost or need to find the Windows IP
try:
    # Load variables from .env file
    from dotenv import load_dotenv
    load_dotenv()
    
    uri = os.getenv('NEO4J_URI', 'bolt://localhost:7687')
    username = os.getenv('NEO4J_USERNAME', 'neo4j')
    password = os.getenv('NEO4J_PASSWORD', '#1Moneymaker')
    database = os.getenv('NEO4J_DATABASE', 'marketing')
    
    print(f"Connecting to Neo4j at {uri} with username {username}, database {database}...")
    driver = GraphDatabase.driver(uri, auth=(username, password))
    
    # Test the connection
    with driver.session(database=database) as session:
        result = session.run("RETURN 1 AS test")
        record = result.single()
        print(f"Neo4j connection test: {record['test']}")
    
    # Test APOC availability
    with driver.session(database=database) as session:
        result = session.run("CALL apoc.help('overview')")
        print(f"APOC is available: {result.peek() is not None}")
    
    # Test GDS availability
    with driver.session(database=database) as session:
        result = session.run("CALL gds.list()")
        print(f"Graph Data Science library is available: {result.peek() is not None}")
    
    print("Connected successfully!")
    driver.close()
except Exception as e:
    print(f"Error connecting to Neo4j: {e}")
    print("\nTroubleshooting Tips:")
    print("1. Check if the Neo4j server is running on Windows")
    print("2. Verify the connection details in .env match your Neo4j instance")
    print("3. Make sure port 7687 is open and allowed in your Windows firewall")
    print("4. Try using the Windows IP address instead of 'localhost'")
    print("5. For WSL users, try using host.docker.internal instead of localhost")
    
    # Let's print networking info for diagnostics
    print("\nDiagnostic Information:")
    import subprocess
    try:
        # Get WSL networking info
        print("\nWSL Network Configuration:")
        subprocess.run(["ip", "addr"], check=True)
        
        # Try to ping the Windows host
        print("\nTrying to ping Windows host:")
        subprocess.run(["ping", "-c", "3", "172.19.160.1"], check=False)
        
        # Check if Neo4j port is reachable (this won't work with standard tools in WSL)
        print("\nChecking if Neo4j port is reachable:")
        import socket
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(3)
        result = sock.connect_ex(('172.19.160.1', 7687))
        if result == 0:
            print("Port 7687 is open")
        else:
            print(f"Port 7687 is not reachable (result: {result})")
        sock.close()
    except Exception as diag_error:
        print(f"Error during diagnostics: {diag_error}")