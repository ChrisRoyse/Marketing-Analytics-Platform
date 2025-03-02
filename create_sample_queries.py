#!/usr/bin/env python3
"""
Sample Neo4j queries for the marketing behavior pattern ontology.
These queries demonstrate how to analyze customer journeys, churn points,
and marketing funnel patterns.
"""

import json
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('sample_queries.log')
    ]
)

def generate_sample_queries():
    """Generate sample Neo4j queries for the marketing ontology."""
    queries = {
        "Basic Queries": [
            {
                "name": "Count nodes by label",
                "query": """
                CALL db.labels() YIELD label
                MATCH (n:`' + label + '`)
                RETURN label, count(n) AS count
                ORDER BY count DESC
                """
            },
            {
                "name": "Count relationships by type",
                "query": """
                CALL db.relationshipTypes() YIELD relationshipType
                MATCH ()-[r:`' + relationshipType + '`]->()
                RETURN relationshipType, count(r) AS count
                ORDER BY count DESC
                """
            },
            {
                "name": "Get a sample customer with all their attributes",
                "query": """
                MATCH (c:Customer)
                RETURN c
                LIMIT 1
                """
            }
        ],
        "Customer Journey Analysis": [
            {
                "name": "Complete customer journey visualization",
                "query": """
                MATCH (c:Customer {customer_id: $customer_id})-[r]->(n)
                RETURN c, r, n
                """
            },
            {
                "name": "Get customer devices and browsers",
                "query": """
                MATCH (c:Customer {customer_id: $customer_id})
                OPTIONAL MATCH (c)-[r1:USES]->(d:Device)
                OPTIONAL MATCH (c)-[r2:ACCESSES_WITH]->(b:Browser)
                RETURN c.customer_id, c.name, 
                       collect(DISTINCT {type: 'Device', id: d.id, since: r1.timestamp}) as devices,
                       collect(DISTINCT {type: 'Browser', id: b.id, since: r2.timestamp}) as browsers
                """
            },
            {
                "name": "Customer journey timeline",
                "query": """
                MATCH (c:Customer {customer_id: $customer_id})-[r]->(n)
                WHERE r.timestamp IS NOT NULL
                RETURN c.customer_id, c.name, 
                       type(r) as interaction,
                       n.id as target_id,
                       labels(n)[0] as target_type,
                       r.action as action,
                       r.timestamp as timestamp
                ORDER BY r.timestamp
                """
            }
        ],
        "Funnel and Churn Analysis": [
            {
                "name": "Identify customers who abandoned carts",
                "query": """
                MATCH (c:Customer)-[r:ABANDONS]->(cart:Cart)
                RETURN c.customer_id, c.name, cart.id, r.timestamp
                ORDER BY r.timestamp DESC
                """
            },
            {
                "name": "Find explicit churn points",
                "query": """
                MATCH (c:Customer)-[r:CHURNED_AT]->(stage:FunnelStage)
                RETURN c.customer_id, c.name, stage.name as funnel_stage,
                       r.timestamp, r.action, r.reason
                ORDER BY r.timestamp DESC
                """
            },
            {
                "name": "Analyze funnel progression",
                "query": """
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
                
                RETURN c.customer_id, c.name,
                       CASE WHEN reached_awareness THEN 1 ELSE 0 END as awareness,
                       CASE WHEN reached_consideration THEN 1 ELSE 0 END as consideration,
                       CASE WHEN reached_conversion THEN 1 ELSE 0 END as conversion,
                       CASE WHEN reached_retention THEN 1 ELSE 0 END as retention,
                       CASE WHEN reached_advocacy THEN 1 ELSE 0 END as advocacy,
                       CASE 
                         WHEN reached_advocacy THEN 'Advocacy'
                         WHEN reached_retention THEN 'Retention'
                         WHEN reached_conversion THEN 'Conversion'
                         WHEN reached_consideration THEN 'Consideration'
                         WHEN reached_awareness THEN 'Awareness'
                         ELSE 'Pre-awareness'
                       END as current_stage
                """
            }
        ],
        "Device and Channel Analysis": [
            {
                "name": "Device usage distribution",
                "query": """
                MATCH (d:Device)<-[r:USES]-(c:Customer)
                RETURN d.id as device_type, count(c) as customer_count
                ORDER BY customer_count DESC
                """
            },
            {
                "name": "Browser distribution",
                "query": """
                MATCH (b:Browser)<-[r:ACCESSES_WITH]-(c:Customer)
                RETURN b.id as browser, count(c) as customer_count
                ORDER BY customer_count DESC
                """
            },
            {
                "name": "Preferred channel distribution",
                "query": """
                MATCH (ch:Channel)<-[r:PREFERS]-(c:Customer)
                RETURN ch.id as channel, count(c) as customer_count
                ORDER BY customer_count DESC
                """
            },
            {
                "name": "Device and browser combinations",
                "query": """
                MATCH (c:Customer)-[:USES]->(d:Device)
                MATCH (c)-[:ACCESSES_WITH]->(b:Browser)
                RETURN d.id as device, b.id as browser, count(c) as customer_count
                ORDER BY customer_count DESC
                """
            }
        ],
        "Segment and Persona Analysis": [
            {
                "name": "Customer distribution by segment",
                "query": """
                MATCH (s:Segment)<-[r:BELONGS_TO]-(c:Customer)
                RETURN s.id as segment, count(c) as customer_count
                ORDER BY customer_count DESC
                """
            },
            {
                "name": "Customer distribution by persona",
                "query": """
                MATCH (p:Persona)<-[r:HAS_PERSONA]-(c:Customer)
                RETURN p.id as persona, count(c) as customer_count
                ORDER BY customer_count DESC
                """
            },
            {
                "name": "Segment and persona combinations",
                "query": """
                MATCH (c:Customer)-[:BELONGS_TO]->(s:Segment)
                MATCH (c)-[:HAS_PERSONA]->(p:Persona)
                RETURN s.id as segment, p.id as persona, count(c) as customer_count
                ORDER BY s.id, customer_count DESC
                """
            },
            {
                "name": "Device preferences by persona",
                "query": """
                MATCH (p:Persona)<-[:HAS_PERSONA]-(c:Customer)-[:USES]->(d:Device)
                RETURN p.id as persona, d.id as device, count(c) as customer_count
                ORDER BY persona, customer_count DESC
                """
            }
        ],
        "Purchase and Behavior Analysis": [
            {
                "name": "Most purchased products",
                "query": """
                MATCH (c:Customer)-[r:PURCHASES]->(p:Product)
                RETURN p.id as product, count(c) as purchase_count
                ORDER BY purchase_count DESC
                """
            },
            {
                "name": "Most viewed products that weren't purchased",
                "query": """
                MATCH (c:Customer)-[v:VIEWS]->(p:Product)
                WHERE NOT (c)-[:PURCHASES]->(p)
                RETURN p.id as product, count(c) as view_count
                ORDER BY view_count DESC
                """
            },
            {
                "name": "Average time from view to purchase",
                "query": """
                MATCH (c:Customer)-[v:VIEWS]->(p:Product)
                MATCH (c)-[pu:PURCHASES]->(p)
                WHERE v.timestamp < pu.timestamp
                WITH c, p, v.timestamp as view_time, pu.timestamp as purchase_time
                RETURN p.id as product,
                       avg(duration.between(datetime(view_time), datetime(purchase_time)).seconds) as avg_seconds_to_purchase
                ORDER BY avg_seconds_to_purchase
                """
            }
        ],
        "Email and Content Analysis": [
            {
                "name": "Most effective email campaigns",
                "query": """
                MATCH (c:Customer)-[v:VIEWS]->(e:Email)
                OPTIONAL MATCH (c)-[cl:CLICKS_ON]->(e)
                WITH e.id as email_campaign, count(v) as view_count, count(cl) as click_count
                RETURN email_campaign, view_count, click_count, 
                       CASE WHEN view_count > 0 THEN toFloat(click_count) / view_count ELSE 0 END as click_rate
                ORDER BY click_rate DESC
                """
            },
            {
                "name": "Most engaged content",
                "query": """
                MATCH (c:Customer)-[r]->(co:Content)
                RETURN co.id as content, type(r) as interaction_type, count(c) as interaction_count
                ORDER BY interaction_count DESC
                """
            }
        ],
        "Location-Based Analysis": [
            {
                "name": "Customer distribution by location",
                "query": """
                MATCH (l:Location)<-[r:LOCATED_IN]-(c:Customer)
                RETURN l.id as location, count(c) as customer_count
                ORDER BY customer_count DESC
                """
            },
            {
                "name": "Purchase behavior by location",
                "query": """
                MATCH (c:Customer)-[:LOCATED_IN]->(l:Location)
                OPTIONAL MATCH (c)-[p:PURCHASES]->(pr:Product)
                WITH l.id as location, count(DISTINCT c) as customer_count, count(p) as purchase_count
                RETURN location, customer_count, purchase_count,
                       CASE WHEN customer_count > 0 THEN toFloat(purchase_count) / customer_count ELSE 0 END as purchases_per_customer
                ORDER BY purchases_per_customer DESC
                """
            }
        ],
        "Advanced Graph Analysis": [
            {
                "name": "Customer similarity based on behavior",
                "query": """
                MATCH (c1:Customer)-[r1]->(n)
                MATCH (c2:Customer)-[r2]->(n)
                WHERE id(c1) < id(c2) AND type(r1) = type(r2)
                WITH c1, c2, count(DISTINCT n) as common_interactions
                WHERE common_interactions >= 3
                RETURN c1.customer_id, c1.name, c2.customer_id, c2.name, common_interactions
                ORDER BY common_interactions DESC
                LIMIT 20
                """
            },
            {
                "name": "Product recommendation",
                "query": """
                MATCH (c:Customer {customer_id: $customer_id})-[:PURCHASES]->(p:Product)
                MATCH (other:Customer)-[:PURCHASES]->(p)
                MATCH (other)-[:PURCHASES]->(rec:Product)
                WHERE NOT (c)-[:PURCHASES]->(rec)
                RETURN rec.id as recommended_product, count(DISTINCT other) as customer_count
                ORDER BY customer_count DESC
                LIMIT 5
                """
            },
            {
                "name": "Customer journey patterns",
                "query": """
                MATCH path = (c:Customer)-[r1]->(n1)-[r2]->(n2)
                WHERE type(r1) <> type(r2)
                WITH [type(r1), labels(n1)[0], type(r2), labels(n2)[0]] as pattern, count(*) as pattern_count
                RETURN pattern, pattern_count
                ORDER BY pattern_count DESC
                LIMIT 10
                """
            }
        ]
    }
    
    # Write the queries to a file
    with open('marketing_ontology_queries.json', 'w') as f:
        json.dump(queries, f, indent=2)
    
    # Generate a Cypher script file with comments
    with open('marketing_ontology_queries.cypher', 'w') as f:
        for category, query_list in queries.items():
            f.write(f"// ===== {category} =====\n\n")
            
            for query_obj in query_list:
                f.write(f"// {query_obj['name']}\n")
                f.write(f"{query_obj['query'].strip()}\n\n")
    
    logging.info("Generated sample Neo4j queries for marketing ontology")
    return queries

def generate_funnel_visualization_query():
    """Generate a special query for visualizing the marketing funnel in Neo4j Browser."""
    viz_query = """
    // Marketing Funnel Visualization
    // This query creates a visualization of the entire marketing funnel with all connections
    
    // First, get all nodes and relationships in the marketing funnel
    MATCH p=(c:Customer)-[r]->(n)
    WHERE n:Advertisement OR n:Page OR n:Product OR n:Email OR n:Content OR 
          n:Device OR n:Browser OR n:Location OR n:OperatingSystem OR
          n:Segment OR n:Persona OR n:BehaviorStage OR n:FunnelStage OR
          n:Channel OR n:Newsletter OR n:SatisfactionScore
    
    // Return paths limited to avoid browser performance issues
    RETURN p
    LIMIT 100
    """
    
    with open('funnel_visualization.cypher', 'w') as f:
        f.write(viz_query)
    
    logging.info("Generated funnel visualization query")

def generate_churn_analysis_dashboard():
    """Generate queries for a churn analysis dashboard."""
    churn_queries = [
        {
            "name": "Churn by Funnel Stage",
            "query": """
            MATCH (c:Customer)-[r:CHURNED_AT]->(stage:FunnelStage)
            RETURN stage.name as funnel_stage, count(c) as churn_count
            ORDER BY churn_count DESC
            """
        },
        {
            "name": "Churn by Segment",
            "query": """
            MATCH (c:Customer)-[:CHURNED_AT]->(:FunnelStage)
            MATCH (c)-[:BELONGS_TO]->(s:Segment)
            RETURN s.id as segment, count(c) as churn_count
            ORDER BY churn_count DESC
            """
        },
        {
            "name": "Churn by Device",
            "query": """
            MATCH (c:Customer)-[:CHURNED_AT]->(:FunnelStage)
            MATCH (c)-[:USES]->(d:Device)
            RETURN d.id as device, count(c) as churn_count
            ORDER BY churn_count DESC
            """
        },
        {
            "name": "Churn by Browser",
            "query": """
            MATCH (c:Customer)-[:CHURNED_AT]->(:FunnelStage)
            MATCH (c)-[:ACCESSES_WITH]->(b:Browser)
            RETURN b.id as browser, count(c) as churn_count
            ORDER BY churn_count DESC
            """
        },
        {
            "name": "Churn Reasons",
            "query": """
            MATCH (c:Customer)-[r:CHURNED_AT]->(:FunnelStage)
            RETURN r.reason as churn_reason, count(c) as churn_count
            ORDER BY churn_count DESC
            """
        },
        {
            "name": "Churn Over Time",
            "query": """
            MATCH (c:Customer)-[r:CHURNED_AT]->(:FunnelStage)
            WITH date(r.timestamp) as churn_date, count(c) as churn_count
            RETURN churn_date, churn_count
            ORDER BY churn_date
            """
        }
    ]
    
    with open('churn_analysis_dashboard.cypher', 'w') as f:
        for query in churn_queries:
            f.write(f"// {query['name']}\n")
            f.write(f"{query['query'].strip()}\n\n")
    
    logging.info("Generated churn analysis dashboard queries")

def create_sample_queries_file():
    """Create all sample query files."""
    try:
        # Generate general sample queries
        queries = generate_sample_queries()
        
        # Generate funnel visualization query
        generate_funnel_visualization_query()
        
        # Generate churn analysis dashboard
        generate_churn_analysis_dashboard()
        
        logging.info("Successfully created all sample query files")
        return True
    except Exception as e:
        logging.error(f"Error generating sample queries: {e}")
        return False

if __name__ == "__main__":
    print("Generating sample Neo4j queries for marketing ontology...")
    if create_sample_queries_file():
        print("Sample queries generated successfully!")
        print("Files created: marketing_ontology_queries.json, marketing_ontology_queries.cypher, funnel_visualization.cypher, churn_analysis_dashboard.cypher")
    else:
        print("Failed to generate sample queries. Check logs for details.")