// ===== Basic Queries =====

// Count nodes by label
CALL db.labels() YIELD label
                MATCH (n:`' + label + '`)
                RETURN label, count(n) AS count
                ORDER BY count DESC

// Count relationships by type
CALL db.relationshipTypes() YIELD relationshipType
                MATCH ()-[r:`' + relationshipType + '`]->()
                RETURN relationshipType, count(r) AS count
                ORDER BY count DESC

// Get a sample customer with all their attributes
MATCH (c:Customer)
                RETURN c
                LIMIT 1

// ===== Customer Journey Analysis =====

// Complete customer journey visualization
MATCH (c:Customer {customer_id: $customer_id})-[r]->(n)
                RETURN c, r, n

// Get customer devices and browsers
MATCH (c:Customer {customer_id: $customer_id})
                OPTIONAL MATCH (c)-[r1:USES]->(d:Device)
                OPTIONAL MATCH (c)-[r2:ACCESSES_WITH]->(b:Browser)
                RETURN c.customer_id, c.name, 
                       collect(DISTINCT {type: 'Device', id: d.id, since: r1.timestamp}) as devices,
                       collect(DISTINCT {type: 'Browser', id: b.id, since: r2.timestamp}) as browsers

// Customer journey timeline
MATCH (c:Customer {customer_id: $customer_id})-[r]->(n)
                WHERE r.timestamp IS NOT NULL
                RETURN c.customer_id, c.name, 
                       type(r) as interaction,
                       n.id as target_id,
                       labels(n)[0] as target_type,
                       r.action as action,
                       r.timestamp as timestamp
                ORDER BY r.timestamp

// ===== Funnel and Churn Analysis =====

// Identify customers who abandoned carts
MATCH (c:Customer)-[r:ABANDONS]->(cart:Cart)
                RETURN c.customer_id, c.name, cart.id, r.timestamp
                ORDER BY r.timestamp DESC

// Find explicit churn points
MATCH (c:Customer)-[r:CHURNED_AT]->(stage:FunnelStage)
                RETURN c.customer_id, c.name, stage.name as funnel_stage,
                       r.timestamp, r.action, r.reason
                ORDER BY r.timestamp DESC

// Analyze funnel progression
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

// ===== Device and Channel Analysis =====

// Device usage distribution
MATCH (d:Device)<-[r:USES]-(c:Customer)
                RETURN d.id as device_type, count(c) as customer_count
                ORDER BY customer_count DESC

// Browser distribution
MATCH (b:Browser)<-[r:ACCESSES_WITH]-(c:Customer)
                RETURN b.id as browser, count(c) as customer_count
                ORDER BY customer_count DESC

// Preferred channel distribution
MATCH (ch:Channel)<-[r:PREFERS]-(c:Customer)
                RETURN ch.id as channel, count(c) as customer_count
                ORDER BY customer_count DESC

// Device and browser combinations
MATCH (c:Customer)-[:USES]->(d:Device)
                MATCH (c)-[:ACCESSES_WITH]->(b:Browser)
                RETURN d.id as device, b.id as browser, count(c) as customer_count
                ORDER BY customer_count DESC

// ===== Segment and Persona Analysis =====

// Customer distribution by segment
MATCH (s:Segment)<-[r:BELONGS_TO]-(c:Customer)
                RETURN s.id as segment, count(c) as customer_count
                ORDER BY customer_count DESC

// Customer distribution by persona
MATCH (p:Persona)<-[r:HAS_PERSONA]-(c:Customer)
                RETURN p.id as persona, count(c) as customer_count
                ORDER BY customer_count DESC

// Segment and persona combinations
MATCH (c:Customer)-[:BELONGS_TO]->(s:Segment)
                MATCH (c)-[:HAS_PERSONA]->(p:Persona)
                RETURN s.id as segment, p.id as persona, count(c) as customer_count
                ORDER BY s.id, customer_count DESC

// Device preferences by persona
MATCH (p:Persona)<-[:HAS_PERSONA]-(c:Customer)-[:USES]->(d:Device)
                RETURN p.id as persona, d.id as device, count(c) as customer_count
                ORDER BY persona, customer_count DESC

// ===== Purchase and Behavior Analysis =====

// Most purchased products
MATCH (c:Customer)-[r:PURCHASES]->(p:Product)
                RETURN p.id as product, count(c) as purchase_count
                ORDER BY purchase_count DESC

// Most viewed products that weren't purchased
MATCH (c:Customer)-[v:VIEWS]->(p:Product)
                WHERE NOT (c)-[:PURCHASES]->(p)
                RETURN p.id as product, count(c) as view_count
                ORDER BY view_count DESC

// Average time from view to purchase
MATCH (c:Customer)-[v:VIEWS]->(p:Product)
                MATCH (c)-[pu:PURCHASES]->(p)
                WHERE v.timestamp < pu.timestamp
                WITH c, p, v.timestamp as view_time, pu.timestamp as purchase_time
                RETURN p.id as product,
                       avg(duration.between(datetime(view_time), datetime(purchase_time)).seconds) as avg_seconds_to_purchase
                ORDER BY avg_seconds_to_purchase

// ===== Email and Content Analysis =====

// Most effective email campaigns
MATCH (c:Customer)-[v:VIEWS]->(e:Email)
                OPTIONAL MATCH (c)-[cl:CLICKS_ON]->(e)
                WITH e.id as email_campaign, count(v) as view_count, count(cl) as click_count
                RETURN email_campaign, view_count, click_count, 
                       CASE WHEN view_count > 0 THEN toFloat(click_count) / view_count ELSE 0 END as click_rate
                ORDER BY click_rate DESC

// Most engaged content
MATCH (c:Customer)-[r]->(co:Content)
                RETURN co.id as content, type(r) as interaction_type, count(c) as interaction_count
                ORDER BY interaction_count DESC

// ===== Location-Based Analysis =====

// Customer distribution by location
MATCH (l:Location)<-[r:LOCATED_IN]-(c:Customer)
                RETURN l.id as location, count(c) as customer_count
                ORDER BY customer_count DESC

// Purchase behavior by location
MATCH (c:Customer)-[:LOCATED_IN]->(l:Location)
                OPTIONAL MATCH (c)-[p:PURCHASES]->(pr:Product)
                WITH l.id as location, count(DISTINCT c) as customer_count, count(p) as purchase_count
                RETURN location, customer_count, purchase_count,
                       CASE WHEN customer_count > 0 THEN toFloat(purchase_count) / customer_count ELSE 0 END as purchases_per_customer
                ORDER BY purchases_per_customer DESC

// ===== Advanced Graph Analysis =====

// Customer similarity based on behavior
MATCH (c1:Customer)-[r1]->(n)
                MATCH (c2:Customer)-[r2]->(n)
                WHERE id(c1) < id(c2) AND type(r1) = type(r2)
                WITH c1, c2, count(DISTINCT n) as common_interactions
                WHERE common_interactions >= 3
                RETURN c1.customer_id, c1.name, c2.customer_id, c2.name, common_interactions
                ORDER BY common_interactions DESC
                LIMIT 20

// Product recommendation
MATCH (c:Customer {customer_id: $customer_id})-[:PURCHASES]->(p:Product)
                MATCH (other:Customer)-[:PURCHASES]->(p)
                MATCH (other)-[:PURCHASES]->(rec:Product)
                WHERE NOT (c)-[:PURCHASES]->(rec)
                RETURN rec.id as recommended_product, count(DISTINCT other) as customer_count
                ORDER BY customer_count DESC
                LIMIT 5

// Customer journey patterns
MATCH path = (c:Customer)-[r1]->(n1)-[r2]->(n2)
                WHERE type(r1) <> type(r2)
                WITH [type(r1), labels(n1)[0], type(r2), labels(n2)[0]] as pattern, count(*) as pattern_count
                RETURN pattern, pattern_count
                ORDER BY pattern_count DESC
                LIMIT 10

