// Churn by Funnel Stage
MATCH (c:Customer)-[r:CHURNED_AT]->(stage:FunnelStage)
            RETURN stage.name as funnel_stage, count(c) as churn_count
            ORDER BY churn_count DESC

// Churn by Segment
MATCH (c:Customer)-[:CHURNED_AT]->(:FunnelStage)
            MATCH (c)-[:BELONGS_TO]->(s:Segment)
            RETURN s.id as segment, count(c) as churn_count
            ORDER BY churn_count DESC

// Churn by Device
MATCH (c:Customer)-[:CHURNED_AT]->(:FunnelStage)
            MATCH (c)-[:USES]->(d:Device)
            RETURN d.id as device, count(c) as churn_count
            ORDER BY churn_count DESC

// Churn by Browser
MATCH (c:Customer)-[:CHURNED_AT]->(:FunnelStage)
            MATCH (c)-[:ACCESSES_WITH]->(b:Browser)
            RETURN b.id as browser, count(c) as churn_count
            ORDER BY churn_count DESC

// Churn Reasons
MATCH (c:Customer)-[r:CHURNED_AT]->(:FunnelStage)
            RETURN r.reason as churn_reason, count(c) as churn_count
            ORDER BY churn_count DESC

// Churn Over Time
MATCH (c:Customer)-[r:CHURNED_AT]->(:FunnelStage)
            WITH date(r.timestamp) as churn_date, count(c) as churn_count
            RETURN churn_date, churn_count
            ORDER BY churn_date

