
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
    