from neo4j import GraphDatabase
import sys

# Try different connection strings
connection_strings = [
    "bolt://localhost:7687",
    "bolt://127.0.0.1:7687",
    "bolt://172.19.160.1:7687"
]

for uri in connection_strings:
    print(f"Trying to connect to: {uri}")
    try:
        driver = GraphDatabase.driver(uri, auth=("neo4j", "#1Moneymaker"))
        with driver.session() as session:
            result = session.run("RETURN 1 as num")
            print(f"Connected successfully to {uri}!")
            print(f"Result: {result.single()['num']}")
        driver.close()
        print("Connection test successful!")
        sys.exit(0)
    except Exception as e:
        print(f"Failed to connect to {uri}: {str(e)}")
        print()

print("All connection attempts failed")
sys.exit(1)