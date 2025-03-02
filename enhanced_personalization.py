#!/usr/bin/env python3
"""
Enhanced Personalization module for Phase 5 of the marketing ontology project.
This module implements context-aware recommendations, NLP analysis, and reinforcement
learning to provide highly personalized customer experiences.
"""

import os
import json
import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from collections import defaultdict
import requests
import re
from pathlib import Path
from scipy.spatial.distance import cosine
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation, NMF
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from transformers import pipeline
from neo4j import GraphDatabase

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('enhanced_personalization.log')
    ]
)

class EnhancedPersonalization:
    """
    Class for implementing advanced personalization techniques including:
    - Context-aware recommendations (time, location, weather, events)
    - Natural Language Processing for customer feedback analysis
    - Reinforcement learning for recommendation optimization
    """
    
    def __init__(self, uri=None, username=None, password=None, database=None):
        """Initialize the EnhancedPersonalization class with Neo4j connection details."""
        self.uri = uri or os.getenv('NEO4J_URI', 'bolt://localhost:7687')
        self.username = username or os.getenv('NEO4J_USERNAME', 'neo4j')
        self.password = password or os.getenv('NEO4J_PASSWORD', '#1Moneymaker')
        self.database = "marketing"  # Explicitly set to your database name
        self.driver = None
        
        # Initialize NLP components
        self.sentiment_analyzer = None
        self.nlp_models = {}
        
        # Initialize reinforcement learning components
        self.recommendation_rewards = defaultdict(dict)
        self.exploration_rate = 0.2  # Initial exploration rate
        self.learning_rate = 0.1     # Learning rate for updating rewards
        self.discount_factor = 0.9   # Discount factor for future rewards
        
        # Context data cache
        self.weather_cache = {}
        self.events_cache = {}
        self.location_cache = {}
        
        # Create directories for outputs
        Path("nlp_insights").mkdir(exist_ok=True)
        Path("context_data").mkdir(exist_ok=True)
        Path("reinforcement_learning").mkdir(exist_ok=True)
    
    def connect(self):
        """Connect to the Neo4j database."""
        try:
            self.driver = GraphDatabase.driver(
                self.uri, 
                auth=(self.username, self.password)
            )
            # Test the connection
            with self.driver.session(database=self.database) as session:
                result = session.run("RETURN 1 AS test")
                record = result.single()
                if record and record["test"] == 1:
                    logging.info("Successfully connected to Neo4j database")
                    return True
                else:
                    logging.error("Failed to verify Neo4j connection")
                    return False
        except Exception as e:
            logging.error(f"Failed to connect to Neo4j: {e}")
            return False
    
    def close(self):
        """Close the Neo4j connection."""
        if self.driver:
            self.driver.close()
            logging.info("Neo4j connection closed")
    
    def run_query(self, query, parameters=None):
        """Run a Cypher query and return the results."""
        if not self.driver:
            if not self.connect():
                return None
        
        try:
            with self.driver.session(database=self.database) as session:
                result = session.run(query, parameters or {})
                return [record.data() for record in result]
        except Exception as e:
            logging.error(f"Error running query: {e}")
            return None
    
    def initialize_nlp_models(self):
        """Initialize NLP models for sentiment analysis and topic modeling."""
        try:
            # Initialize sentiment analysis pipeline
            self.sentiment_analyzer = pipeline(
                "sentiment-analysis", 
                model="distilbert-base-uncased-finetuned-sst-2-english",
                truncation=True
            )
            
            # Initialize TF-IDF vectorizer for topic modeling
            self.nlp_models["vectorizer"] = TfidfVectorizer(
                max_features=5000,
                stop_words='english',
                max_df=0.95,
                min_df=2
            )
            
            # Initialize topic modeling with Latent Dirichlet Allocation
            self.nlp_models["lda"] = LatentDirichletAllocation(
                n_components=5,
                random_state=42,
                learning_method='online'
            )
            
            # Initialize Non-negative Matrix Factorization for topic modeling
            self.nlp_models["nmf"] = NMF(
                n_components=5,
                random_state=42
            )
            
            logging.info("NLP models initialized successfully")
            return True
        except Exception as e:
            logging.error(f"Error initializing NLP models: {e}")
            return False
    
    def analyze_customer_feedback(self, customer_id=None):
        """
        Analyze customer feedback using NLP techniques.
        If customer_id is provided, analyze only that customer's feedback.
        Otherwise, analyze all customer feedback.
        """
        try:
            # Make sure NLP models are initialized
            if not self.sentiment_analyzer:
                self.initialize_nlp_models()
            
            # Query to get customer feedback
            if customer_id:
                query = """
                MATCH (c:Customer {customer_id: $customer_id})-[:PROVIDES]->(f:Feedback)
                RETURN c.customer_id as customer_id, f.text as feedback_text, 
                       f.timestamp as timestamp, f.source as source, f.rating as rating
                ORDER BY f.timestamp DESC
                """
                feedback_data = self.run_query(query, {"customer_id": customer_id})
            else:
                query = """
                MATCH (c:Customer)-[:PROVIDES]->(f:Feedback)
                RETURN c.customer_id as customer_id, f.text as feedback_text, 
                       f.timestamp as timestamp, f.source as source, f.rating as rating
                ORDER BY c.customer_id, f.timestamp DESC
                """
                feedback_data = self.run_query(query)
            
            if not feedback_data:
                # Check if we need to create sample feedback data
                self._create_sample_feedback_data()
                
                # Try query again
                if customer_id:
                    feedback_data = self.run_query(query, {"customer_id": customer_id})
                else:
                    feedback_data = self.run_query(query)
                
                if not feedback_data:
                    logging.warning("No customer feedback found for analysis")
                    return None
            
            # Process feedback with NLP
            feedback_insights = self._process_feedback_with_nlp(feedback_data)
            
            # Store insights in Neo4j
            self._store_feedback_insights(feedback_insights)
            
            logging.info(f"Analyzed {len(feedback_data)} feedback entries")
            return feedback_insights
            
        except Exception as e:
            logging.error(f"Error analyzing customer feedback: {e}")
            return None
    
    def _process_feedback_with_nlp(self, feedback_data):
        """Process customer feedback with NLP techniques."""
        # Group feedback by customer
        customer_feedback = defaultdict(list)
        all_feedback_texts = []
        
        for entry in feedback_data:
            customer_id = entry.get("customer_id")
            feedback_text = entry.get("feedback_text", "")
            
            if feedback_text and len(feedback_text.strip()) > 5:  # Ensure text is substantial
                customer_feedback[customer_id].append(entry)
                all_feedback_texts.append(feedback_text)
        
        # Perform sentiment analysis
        sentiments = []
        for text in all_feedback_texts:
            try:
                result = self.sentiment_analyzer(text)
                sentiments.append(result[0])
            except Exception as e:
                logging.warning(f"Error in sentiment analysis: {e}")
                sentiments.append({"label": "NEUTRAL", "score": 0.5})
        
        # Create feature matrix for topic modeling
        if len(all_feedback_texts) >= 5:  # Need minimum texts for meaningful topic modeling
            try:
                # Vectorize feedback text
                tfidf_matrix = self.nlp_models["vectorizer"].fit_transform(all_feedback_texts)
                
                # Extract topics using LDA
                lda_topics = self.nlp_models["lda"].fit_transform(tfidf_matrix)
                
                # Get feature names
                feature_names = self.nlp_models["vectorizer"].get_feature_names_out()
                
                # Extract top words for each topic
                topics = []
                for topic_idx, topic in enumerate(self.nlp_models["lda"].components_):
                    top_words_idx = topic.argsort()[:-11:-1]  # Top 10 words
                    top_words = [feature_names[i] for i in top_words_idx]
                    topics.append({
                        "id": f"topic_{topic_idx}",
                        "top_words": top_words,
                        "weight": float(np.mean(lda_topics[:, topic_idx]))
                    })
            except Exception as e:
                logging.warning(f"Error in topic modeling: {e}")
                topics = []
        else:
            topics = []
        
        # Combine insights for each customer
        customer_insights = {}
        for idx, (customer_id, entries) in enumerate(customer_feedback.items()):
            # Calculate average sentiment
            customer_sentiments = [
                sentiments[all_feedback_texts.index(entry["feedback_text"])]
                for entry in entries if entry.get("feedback_text") in all_feedback_texts
            ]
            
            # Skip if no valid sentiments
            if not customer_sentiments:
                continue
                
            avg_sentiment_score = np.mean([s["score"] for s in customer_sentiments])
            predominant_sentiment = max(
                set(s["label"] for s in customer_sentiments),
                key=[s["label"] for s in customer_sentiments].count
            )
            
            # Extract key phrases and themes
            feedback_texts = [entry["feedback_text"] for entry in entries]
            combined_text = " ".join(feedback_texts)
            
            # Simple keyword extraction based on frequency
            words = re.findall(r'\b[a-zA-Z]{3,}\b', combined_text.lower())
            word_freq = pd.Series(words).value_counts()
            common_words = word_freq[word_freq > 1].index.tolist()[:10]
            
            # Assign topics to customer
            customer_topics = []
            if topics and len(all_feedback_texts) >= 5:
                customer_indices = [all_feedback_texts.index(text) for text in feedback_texts if text in all_feedback_texts]
                for topic in topics:
                    if customer_indices:
                        topic_relevance = np.mean(lda_topics[customer_indices, int(topic["id"].split("_")[1])])
                        if topic_relevance > 0.1:  # Only include relevant topics
                            customer_topics.append({
                                "id": topic["id"],
                                "top_words": topic["top_words"],
                                "relevance": float(topic_relevance)
                            })
            
            # Store insights
            customer_insights[customer_id] = {
                "customer_id": customer_id,
                "sentiment": {
                    "predominant": predominant_sentiment,
                    "score": float(avg_sentiment_score),
                    "positive_count": sum(1 for s in customer_sentiments if s["label"] == "POSITIVE"),
                    "negative_count": sum(1 for s in customer_sentiments if s["label"] == "NEGATIVE")
                },
                "topics": customer_topics,
                "keywords": common_words,
                "feedback_count": len(entries),
                "latest_feedback": entries[0].get("timestamp", ""),
                "sources": list(set(entry.get("source", "") for entry in entries)),
                "average_rating": np.mean([entry.get("rating", 0) for entry in entries if entry.get("rating") is not None])
            }
        
        # Add global topic analysis for all customers
        global_insights = {
            "topics": topics,
            "total_feedback_count": len(all_feedback_texts),
            "average_sentiment_score": float(np.mean([s["score"] for s in sentiments])),
            "positive_feedback_percentage": sum(1 for s in sentiments if s["label"] == "POSITIVE") / len(sentiments) * 100 if sentiments else 0,
            "analyzed_at": datetime.now().isoformat()
        }
        
        return {
            "customer_insights": customer_insights,
            "global_insights": global_insights
        }
    
    def _store_feedback_insights(self, insights):
        """Store feedback insights in Neo4j."""
        if not insights:
            return False
        
        try:
            # Store customer-specific insights
            for customer_id, customer_insight in insights["customer_insights"].items():
                # Create insight node
                insight_query = """
                MATCH (c:Customer {customer_id: $customer_id})
                MERGE (i:NLPInsight {customer_id: $customer_id})
                SET i.timestamp = datetime(),
                    i.sentiment_score = $sentiment_score,
                    i.predominant_sentiment = $predominant_sentiment,
                    i.positive_count = $positive_count,
                    i.negative_count = $negative_count,
                    i.keywords = $keywords,
                    i.feedback_count = $feedback_count,
                    i.latest_feedback = $latest_feedback,
                    i.sources = $sources,
                    i.average_rating = $average_rating
                
                MERGE (c)-[:HAS_INSIGHT]->(i)
                
                RETURN i
                """
                
                self.run_query(insight_query, {
                    "customer_id": customer_id,
                    "sentiment_score": customer_insight["sentiment"]["score"],
                    "predominant_sentiment": customer_insight["sentiment"]["predominant"],
                    "positive_count": customer_insight["sentiment"]["positive_count"],
                    "negative_count": customer_insight["sentiment"]["negative_count"],
                    "keywords": customer_insight["keywords"],
                    "feedback_count": customer_insight["feedback_count"],
                    "latest_feedback": customer_insight["latest_feedback"],
                    "sources": customer_insight["sources"],
                    "average_rating": customer_insight["average_rating"]
                })
                
                # Store topics
                for topic in customer_insight["topics"]:
                    topic_query = """
                    MATCH (i:NLPInsight {customer_id: $customer_id})
                    MERGE (t:Topic {id: $topic_id})
                    SET t.top_words = $top_words,
                        t.last_updated = datetime()
                    
                    MERGE (i)-[:HAS_TOPIC {relevance: $relevance}]->(t)
                    """
                    
                    self.run_query(topic_query, {
                        "customer_id": customer_id,
                        "topic_id": topic["id"],
                        "top_words": topic["top_words"],
                        "relevance": topic["relevance"]
                    })
            
            # Store global insights
            global_query = """
            MERGE (g:GlobalNLPInsight {id: 'current'})
            SET g.timestamp = datetime(),
                g.total_feedback_count = $total_feedback_count,
                g.average_sentiment_score = $average_sentiment_score,
                g.positive_feedback_percentage = $positive_feedback_percentage
            """
            
            self.run_query(global_query, {
                "total_feedback_count": insights["global_insights"]["total_feedback_count"],
                "average_sentiment_score": insights["global_insights"]["average_sentiment_score"],
                "positive_feedback_percentage": insights["global_insights"]["positive_feedback_percentage"]
            })
            
            # Save a copy of insights to file
            insights_file = Path(f"nlp_insights/feedback_insights_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
            with open(insights_file, 'w') as f:
                json.dump(insights, f, indent=2)
            
            logging.info(f"Stored NLP insights for {len(insights['customer_insights'])} customers")
            return True
            
        except Exception as e:
            logging.error(f"Error storing feedback insights: {e}")
            return False
    
    def _create_sample_feedback_data(self, num_samples=50):
        """
        Create sample feedback data if no real data exists.
        This is just for demonstration purposes.
        """
        try:
            # First check if we already have some feedback data
            check_query = """
            MATCH (f:Feedback)
            RETURN count(f) as feedback_count
            """
            
            result = self.run_query(check_query)
            if result and result[0]["feedback_count"] > 0:
                logging.info(f"Found {result[0]['feedback_count']} existing feedback entries")
                return True
            
            # Get customer IDs
            customer_query = """
            MATCH (c:Customer)
            RETURN c.customer_id as customer_id
            LIMIT 20
            """
            
            customers = self.run_query(customer_query)
            if not customers:
                logging.warning("No customers found to create sample feedback")
                return False
            
            # Sample feedback templates
            positive_templates = [
                "I really love the {product}. It's exactly what I needed.",
                "Great experience with {product}. Will buy again!",
                "The customer service was excellent when I asked about {product}.",
                "Very satisfied with my recent purchase of {product}.",
                "This {product} exceeded my expectations. 5 stars!",
                "Fast shipping and the {product} works perfectly.",
                "I've recommended the {product} to all my friends.",
                "Best {product} I've ever bought, worth every penny.",
                "The quality of the {product} is outstanding.",
                "Your website made finding the right {product} so easy!"
            ]
            
            negative_templates = [
                "Disappointed with the {product}. Not what I expected.",
                "The {product} stopped working after a week.",
                "Customer service was unhelpful when I had issues with my {product}.",
                "The {product} is overpriced for the quality.",
                "Had to return the {product} because it was defective.",
                "Shipping took too long for my {product}.",
                "The {product} doesn't match the description on your website.",
                "I wouldn't recommend the {product} to anyone.",
                "Instructions for the {product} were confusing.",
                "The {product} broke easily. Poor quality."
            ]
            
            neutral_templates = [
                "The {product} is okay. Nothing special.",
                "Received the {product} as expected.",
                "The {product} works for my needs but has some limitations.",
                "Average quality {product}, fair for the price.",
                "The {product} has both pros and cons.",
                "Not sure yet about the {product}, still testing it.",
                "The {product} is different from what I'm used to.",
                "Might buy the {product} again, haven't decided.",
                "The {product} is adequate but could be improved.",
                "Reasonable value for money with the {product}."
            ]
            
            # Get product IDs
            product_query = """
            MATCH (p:Product)
            RETURN p.id as product_id
            LIMIT 30
            """
            
            products = self.run_query(product_query)
            product_ids = [p["product_id"] for p in products] if products else ["Product A", "Product B", "Product C"]
            
            # Create sample feedback
            feedback_count = 0
            for i in range(num_samples):
                customer = customers[i % len(customers)]
                product = product_ids[i % len(product_ids)]
                
                sentiment_type = np.random.choice(["positive", "negative", "neutral"], p=[0.6, 0.3, 0.1])
                if sentiment_type == "positive":
                    template = np.random.choice(positive_templates)
                    rating = np.random.randint(4, 6)  # 4-5 stars
                elif sentiment_type == "negative":
                    template = np.random.choice(negative_templates)
                    rating = np.random.randint(1, 3)  # 1-2 stars
                else:
                    template = np.random.choice(neutral_templates)
                    rating = 3  # 3 stars
                
                feedback_text = template.format(product=product)
                source = np.random.choice(["website", "email", "support_ticket", "social_media", "app"])
                
                # Generate a random timestamp within the last 90 days
                days_ago = np.random.randint(1, 90)
                timestamp = (datetime.now() - timedelta(days=days_ago)).isoformat()
                
                # Create feedback node
                feedback_query = """
                MATCH (c:Customer {customer_id: $customer_id})
                CREATE (f:Feedback {
                    id: $feedback_id,
                    text: $feedback_text,
                    timestamp: $timestamp,
                    source: $source,
                    rating: $rating
                })
                CREATE (c)-[:PROVIDES]->(f)
                RETURN f
                """
                
                self.run_query(feedback_query, {
                    "customer_id": customer["customer_id"],
                    "feedback_id": f"feedback_{customer['customer_id']}_{i}",
                    "feedback_text": feedback_text,
                    "timestamp": timestamp,
                    "source": source,
                    "rating": rating
                })
                
                feedback_count += 1
            
            logging.info(f"Created {feedback_count} sample feedback entries")
            return True
            
        except Exception as e:
            logging.error(f"Error creating sample feedback data: {e}")
            return False
    
    def get_context_data(self, customer_id):
        """
        Gather contextual data for a customer including:
        - Time of day
        - Location
        - Weather
        - Current events
        """
        try:
            # First get customer details
            query = """
            MATCH (c:Customer {customer_id: $customer_id})
            OPTIONAL MATCH (c)-[:LIVES_IN]->(l:Location)
            RETURN c.customer_id as customer_id,
                   c.timezone as timezone,
                   l.city as city,
                   l.state as state,
                   l.country as country,
                   l.postal_code as postal_code,
                   l.latitude as latitude,
                   l.longitude as longitude
            """
            
            result = self.run_query(query, {"customer_id": customer_id})
            if not result:
                logging.warning(f"No customer data found for ID: {customer_id}")
                return None
            
            customer_data = result[0]
            
            # Build context object
            context = {
                "customer_id": customer_id,
                "timestamp": datetime.now().isoformat(),
                "time_context": self._get_time_context(customer_data.get("timezone")),
                "location_context": None,
                "weather_context": None,
                "event_context": None
            }
            
            # Get location context if available
            if customer_data.get("city") or customer_data.get("postal_code"):
                context["location_context"] = self._get_location_context(customer_data)
                
                # Use location to get weather and events
                if context["location_context"]:
                    context["weather_context"] = self._get_weather_context(context["location_context"])
                    context["event_context"] = self._get_event_context(context["location_context"])
            
            # Store context in Neo4j
            self._store_context_data(customer_id, context)
            
            logging.info(f"Retrieved context data for customer {customer_id}")
            return context
            
        except Exception as e:
            logging.error(f"Error getting context data: {e}")
            return None
    
    def _get_time_context(self, timezone=None):
        """Get time-based context for recommendations."""
        now = datetime.now()
        
        # If no timezone, default to UTC
        if not timezone:
            timezone = "UTC"
        
        # Simple time context for now (could be enhanced with pytz for accurate timezones)
        hour = now.hour
        if 5 <= hour < 12:
            time_of_day = "morning"
        elif 12 <= hour < 17:
            time_of_day = "afternoon"
        elif 17 <= hour < 22:
            time_of_day = "evening"
        else:
            time_of_day = "night"
        
        # Determine day type
        weekday = now.weekday()
        is_weekend = weekday >= 5  # 5=Saturday, 6=Sunday
        
        # Determine season (Northern Hemisphere-centric, could be improved)
        month = now.month
        if 3 <= month <= 5:
            season = "spring"
        elif 6 <= month <= 8:
            season = "summer"
        elif 9 <= month <= 11:
            season = "fall"
        else:
            season = "winter"
        
        return {
            "timestamp": now.isoformat(),
            "hour": hour,
            "time_of_day": time_of_day,
            "day_of_week": weekday,
            "is_weekend": is_weekend,
            "month": month,
            "season": season,
            "timezone": timezone
        }
    
    def _get_location_context(self, customer_data):
        """Get location-based context for recommendations."""
        # Extract location data
        city = customer_data.get("city")
        state = customer_data.get("state")
        country = customer_data.get("country")
        postal_code = customer_data.get("postal_code")
        latitude = customer_data.get("latitude")
        longitude = customer_data.get("longitude")
        
        # Check if we have enough data
        if not city and not postal_code:
            return None
        
        # Create location context
        location_context = {
            "city": city,
            "state": state,
            "country": country or "Unknown",
            "postal_code": postal_code,
            "coordinates": {
                "latitude": float(latitude) if latitude else None,
                "longitude": float(longitude) if longitude else None
            }
        }
        
        # If we don't have coordinates but have city/postal, we could get them
        # from a geocoding service in a production environment
        
        return location_context
    
    def _get_weather_context(self, location_context):
        """
        Get weather data for customer location.
        
        Note: In a production environment, this would call a real weather API.
        Here we'll simulate weather data based on location and date.
        """
        if not location_context:
            return None
        
        city = location_context.get("city")
        postal_code = location_context.get("postal_code")
        
        cache_key = f"{city}_{postal_code}_{datetime.now().strftime('%Y-%m-%d')}"
        
        # Check cache first
        if cache_key in self.weather_cache:
            return self.weather_cache[cache_key]
        
        # In a real implementation, this would call a weather API
        # For demonstration, we'll generate simulated weather
        
        # Use month to determine season and likely weather
        now = datetime.now()
        month = now.month
        
        # Very simple weather simulation based on month
        # This would be replaced with actual API calls in production
        if 3 <= month <= 5:  # Spring
            conditions = np.random.choice(
                ["sunny", "partly cloudy", "light rain", "mild"],
                p=[0.4, 0.3, 0.2, 0.1]
            )
            temp_base = 65  # Fahrenheit
            temp_variance = 15
        elif 6 <= month <= 8:  # Summer
            conditions = np.random.choice(
                ["sunny", "hot", "thunderstorm", "humid"],
                p=[0.6, 0.2, 0.1, 0.1]
            )
            temp_base = 80
            temp_variance = 10
        elif 9 <= month <= 11:  # Fall
            conditions = np.random.choice(
                ["cloudy", "windy", "rainy", "cool"],
                p=[0.3, 0.3, 0.2, 0.2]
            )
            temp_base = 60
            temp_variance = 15
        else:  # Winter
            conditions = np.random.choice(
                ["snow", "freezing", "cloudy", "cold"],
                p=[0.3, 0.3, 0.2, 0.2]
            )
            temp_base = 35
            temp_variance = 20
        
        # Generate random temperature based on season
        temperature = temp_base + np.random.randint(-temp_variance, temp_variance)
        
        # Determine if precipitation is happening
        is_precipitation = conditions in ["light rain", "thunderstorm", "rainy", "snow"]
        
        # Create weather context
        weather_context = {
            "conditions": conditions,
            "temperature": {
                "fahrenheit": temperature,
                "celsius": round((temperature - 32) * 5/9, 1)
            },
            "is_precipitation": is_precipitation,
            "simulated": True,  # Flag to indicate this is not real weather data
            "timestamp": datetime.now().isoformat()
        }
        
        # Cache the result
        self.weather_cache[cache_key] = weather_context
        
        return weather_context
    
    def _get_event_context(self, location_context):
        """
        Get current events near customer location.
        
        Note: In a production environment, this would call a real events API.
        Here we'll simulate event data.
        """
        if not location_context:
            return None
        
        city = location_context.get("city")
        
        if not city:
            return None
        
        cache_key = f"{city}_{datetime.now().strftime('%Y-%m-%d')}"
        
        # Check cache first
        if cache_key in self.events_cache:
            return self.events_cache[cache_key]
        
        # In a real implementation, this would call an events API
        # For demonstration, we'll generate simulated events
        
        # List of possible event types
        event_types = [
            "concert", "festival", "sports", "conference", 
            "exhibition", "holiday", "sales", "local_event"
        ]
        
        # Randomly determine if there are events today
        has_events = np.random.choice([True, False], p=[0.7, 0.3])
        
        if has_events:
            # Generate 1-3 random events
            num_events = np.random.randint(1, 4)
            events = []
            
            for i in range(num_events):
                event_type = np.random.choice(event_types)
                
                # Generate event name based on type
                if event_type == "concert":
                    event_name = f"{np.random.choice(['Rock', 'Pop', 'Jazz', 'Classical'])} Concert"
                elif event_type == "festival":
                    event_name = f"{city} {np.random.choice(['Food', 'Film', 'Music', 'Art'])} Festival"
                elif event_type == "sports":
                    event_name = f"{np.random.choice(['Football', 'Basketball', 'Baseball', 'Soccer'])} Game"
                elif event_type == "holiday":
                    holidays = [
                        "New Year's", "Valentine's Day", "St. Patrick's Day", "Easter",
                        "Memorial Day", "Independence Day", "Labor Day", "Halloween",
                        "Thanksgiving", "Christmas"
                    ]
                    event_name = f"{np.random.choice(holidays)} Celebration"
                elif event_type == "sales":
                    event_name = f"{np.random.choice(['Summer', 'Winter', 'Holiday', 'Flash'])} Sale"
                else:
                    event_name = f"{city} {event_type.replace('_', ' ').title()}"
                
                events.append({
                    "name": event_name,
                    "type": event_type,
                    "location": city,
                    "simulated": True  # Flag to indicate this is not real event data
                })
        else:
            events = []
        
        # Create event context
        event_context = {
            "has_events": has_events,
            "events": events,
            "timestamp": datetime.now().isoformat(),
            "simulated": True
        }
        
        # Cache the result
        self.events_cache[cache_key] = event_context
        
        return event_context
    
    def _store_context_data(self, customer_id, context):
        """Store context data in Neo4j for future reference."""
        if not context:
            return False
        
        try:
            # Convert context to JSON strings for storage
            time_context_json = json.dumps(context.get("time_context", {}))
            location_context_json = json.dumps(context.get("location_context", {}))
            weather_context_json = json.dumps(context.get("weather_context", {}))
            event_context_json = json.dumps(context.get("event_context", {}))
            
            # Store in Neo4j
            query = """
            MATCH (c:Customer {customer_id: $customer_id})
            MERGE (ctx:Context {customer_id: $customer_id})
            SET ctx.timestamp = datetime(),
                ctx.time_context = $time_context,
                ctx.location_context = $location_context,
                ctx.weather_context = $weather_context,
                ctx.event_context = $event_context
            
            MERGE (c)-[:HAS_CONTEXT]->(ctx)
            RETURN ctx
            """
            
            self.run_query(query, {
                "customer_id": customer_id,
                "time_context": time_context_json,
                "location_context": location_context_json,
                "weather_context": weather_context_json,
                "event_context": event_context_json
            })
            
            # Also save to file for analysis
            context_file = Path(f"context_data/{customer_id}_context_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
            with open(context_file, 'w') as f:
                json.dump(context, f, indent=2)
            
            return True
            
        except Exception as e:
            logging.error(f"Error storing context data: {e}")
            return False
    
    def generate_context_aware_recommendations(self, customer_id):
        """
        Generate personalized, context-aware recommendations for a customer.
        """
        try:
            # First get customer preferences and purchase history
            customer_query = """
            MATCH (c:Customer {customer_id: $customer_id})
            
            // Get purchase history
            OPTIONAL MATCH (c)-[p:PURCHASES]->(product:Product)
            WITH c, collect({id: product.id, timestamp: p.timestamp, category: product.category}) as purchases
            
            // Get viewed products
            OPTIONAL MATCH (c)-[v:VIEWS]->(viewed:Product)
            WHERE NOT (c)-[:PURCHASES]->(viewed)
            WITH c, purchases, collect({id: viewed.id, timestamp: v.timestamp, category: viewed.category}) as viewed_products
            
            // Get customer segments
            OPTIONAL MATCH (c)-[:BELONGS_TO]->(s:Segment)
            WITH c, purchases, viewed_products, collect(s.id) as segments
            
            // Get NLP insights if available
            OPTIONAL MATCH (c)-[:HAS_INSIGHT]->(i:NLPInsight)
            WITH c, purchases, viewed_products, segments, i
            
            RETURN c.customer_id as customer_id,
                   purchases,
                   viewed_products,
                   segments,
                   i.predominant_sentiment as sentiment,
                   i.keywords as keywords
            """
            
            customer_result = self.run_query(customer_query, {"customer_id": customer_id})
            
            if not customer_result:
                logging.warning(f"No customer data found for ID: {customer_id}")
                return None
            
            customer_data = customer_result[0]
            
            # Get context data
            context = self.get_context_data(customer_id)
            
            if not context:
                logging.warning(f"No context data available for customer {customer_id}")
                # Continue with limited recommendations
            
            # Get product catalog
            product_query = """
            MATCH (p:Product)
            RETURN p.id as id, p.category as category, p.name as name, 
                   p.price as price, p.attributes as attributes
            """
            
            products = self.run_query(product_query)
            
            if not products:
                logging.warning("No products found in catalog")
                return None
            
            # Generate personalized recommendations
            recommendations = self._generate_recommendations(customer_data, products, context)
            
            # Store recommendations in Neo4j
            self._store_recommendations(customer_id, recommendations)
            
            logging.info(f"Generated {len(recommendations)} context-aware recommendations for customer {customer_id}")
            return recommendations
            
        except Exception as e:
            logging.error(f"Error generating context-aware recommendations: {e}")
            return None
    
    def _generate_recommendations(self, customer_data, products, context=None):
        """
        Core recommendation engine that combines:
        1. Product affinity (based on purchase history)
        2. Context relevance (time, location, weather, events)
        3. NLP insights (sentiment, preferences)
        4. Reinforcement learning feedback (if available)
        """
        customer_id = customer_data.get("customer_id")
        purchases = customer_data.get("purchases", [])
        viewed_products = customer_data.get("viewed_products", [])
        segments = customer_data.get("segments", [])
        sentiment = customer_data.get("sentiment")
        keywords = customer_data.get("keywords", [])
        
        # Create a product lookup for faster processing
        product_map = {p["id"]: p for p in products}
        
        # Track scores for each product
        product_scores = {}
        
        # 1. Base recommendation score from purchase history
        purchased_categories = [p.get("category") for p in purchases if p.get("category")]
        category_counts = pd.Series(purchased_categories).value_counts()
        
        # Calculate category preferences
        total_purchases = sum(category_counts)
        category_preferences = {}
        
        if total_purchases > 0:
            for category, count in category_counts.items():
                category_preferences[category] = count / total_purchases
        
        # Create a list of purchased product IDs
        purchased_ids = [p.get("id") for p in purchases]
        viewed_ids = [p.get("id") for p in viewed_products]
        
        # Score all products
        for product in products:
            product_id = product.get("id")
            
            # Skip already purchased products
            if product_id in purchased_ids:
                continue
            
            # Initialize score
            score = 0.0
            
            # Category affinity score
            category = product.get("category")
            if category in category_preferences:
                score += category_preferences[category] * 10  # Scale to 0-10 range
            
            # Viewed but not purchased bonus
            if product_id in viewed_ids:
                score += 5  # Significant boost for viewed products
            
            # Store base score
            product_scores[product_id] = score
        
        # 2. Apply contextual factors if available
        if context:
            time_context = context.get("time_context", {})
            weather_context = context.get("weather_context", {})
            event_context = context.get("event_context", {})
            
            # Time of day context
            time_of_day = time_context.get("time_of_day")
            if time_of_day:
                time_relevant_categories = {
                    "morning": ["breakfast", "coffee", "news", "vitamins"],
                    "afternoon": ["lunch", "productivity", "snacks", "hydration"],
                    "evening": ["dinner", "entertainment", "relaxation", "home"],
                    "night": ["sleep", "relaxation", "books", "self-care"]
                }.get(time_of_day, [])
                
                for product_id, product in product_map.items():
                    category = product.get("category", "").lower()
                    attributes = product.get("attributes", {})
                    
                    # Extract attributes if they're stored as a JSON string
                    if isinstance(attributes, str):
                        try:
                            attributes = json.loads(attributes)
                        except:
                            attributes = {}
                    
                    for relevant_category in time_relevant_categories:
                        if relevant_category in category or any(relevant_category in str(attr).lower() for attr in attributes.values()):
                            product_scores[product_id] = product_scores.get(product_id, 0) + 3
            
            # Weekend/weekday context
            is_weekend = time_context.get("is_weekend", False)
            weekend_relevant_categories = ["leisure", "outdoor", "entertainment", "hobby"]
            weekday_relevant_categories = ["work", "productivity", "convenience", "quick"]
            
            for product_id, product in product_map.items():
                category = product.get("category", "").lower()
                attributes = product.get("attributes", {})
                
                # Extract attributes if they're stored as a JSON string
                if isinstance(attributes, str):
                    try:
                        attributes = json.loads(attributes)
                    except:
                        attributes = {}
                
                relevant_categories = weekend_relevant_categories if is_weekend else weekday_relevant_categories
                for relevant_category in relevant_categories:
                    if relevant_category in category or any(relevant_category in str(attr).lower() for attr in attributes.values()):
                        product_scores[product_id] = product_scores.get(product_id, 0) + 2
            
            # Season context
            season = time_context.get("season")
            if season:
                season_relevant_categories = {
                    "spring": ["gardening", "cleaning", "light_clothing", "allergy"],
                    "summer": ["cooling", "outdoor", "beach", "suncare", "grilling"],
                    "fall": ["warm_clothing", "halloween", "thanksgiving", "hot_drinks"],
                    "winter": ["heating", "winter_clothing", "holiday", "snow", "immunity"]
                }.get(season, [])
                
                for product_id, product in product_map.items():
                    category = product.get("category", "").lower()
                    attributes = product.get("attributes", {})
                    
                    # Extract attributes if they're stored as a JSON string
                    if isinstance(attributes, str):
                        try:
                            attributes = json.loads(attributes)
                        except:
                            attributes = {}
                    
                    for relevant_category in season_relevant_categories:
                        if relevant_category in category or any(relevant_category in str(attr).lower() for attr in attributes.values()):
                            product_scores[product_id] = product_scores.get(product_id, 0) + 2.5
            
            # Weather context
            if weather_context:
                conditions = weather_context.get("conditions", "").lower()
                is_precipitation = weather_context.get("is_precipitation", False)
                temperature = weather_context.get("temperature", {}).get("fahrenheit", 70)
                
                # Weather-based recommendations
                weather_relevant_items = []
                
                if is_precipitation:
                    weather_relevant_items.extend(["umbrella", "raincoat", "boots"])
                
                if "snow" in conditions:
                    weather_relevant_items.extend(["snowboots", "winter_coat", "gloves", "heater"])
                
                if temperature > 80:
                    weather_relevant_items.extend(["sunscreen", "hat", "fan", "cooling", "water"])
                elif temperature < 40:
                    weather_relevant_items.extend(["jacket", "warm_clothing", "heater", "hot_drinks"])
                
                for product_id, product in product_map.items():
                    category = product.get("category", "").lower()
                    name = product.get("name", "").lower()
                    attributes = product.get("attributes", {})
                    
                    # Extract attributes if they're stored as a JSON string
                    if isinstance(attributes, str):
                        try:
                            attributes = json.loads(attributes)
                        except:
                            attributes = {}
                    
                    for item in weather_relevant_items:
                        if (item in category or item in name or 
                            any(item in str(attr).lower() for attr in attributes.values())):
                            product_scores[product_id] = product_scores.get(product_id, 0) + 4
            
            # Event context
            if event_context and event_context.get("has_events"):
                events = event_context.get("events", [])
                for event in events:
                    event_type = event.get("type", "").lower()
                    event_name = event.get("name", "").lower()
                    
                    event_relevant_items = []
                    
                    if "concert" in event_type or "festival" in event_type:
                        event_relevant_items.extend(["tickets", "outdoor_gear", "camera"])
                    
                    if "sports" in event_type:
                        event_relevant_items.extend(["team_merchandise", "sports_gear", "snacks"])
                    
                    if "holiday" in event_type:
                        if "christmas" in event_name:
                            event_relevant_items.extend(["gift", "decoration", "christmas"])
                        elif "halloween" in event_name:
                            event_relevant_items.extend(["costume", "candy", "halloween"])
                        elif "valentine" in event_name:
                            event_relevant_items.extend(["gift", "chocolate", "romantic"])
                    
                    if "sales" in event_type:
                        event_relevant_items.extend(["deals", "discount", "limited_time"])
                    
                    for product_id, product in product_map.items():
                        category = product.get("category", "").lower()
                        name = product.get("name", "").lower()
                        attributes = product.get("attributes", {})
                        
                        # Extract attributes if they're stored as a JSON string
                        if isinstance(attributes, str):
                            try:
                                attributes = json.loads(attributes)
                            except:
                                attributes = {}
                        
                        for item in event_relevant_items:
                            if (item in category or item in name or 
                                any(item in str(attr).lower() for attr in attributes.values())):
                                product_scores[product_id] = product_scores.get(product_id, 0) + 3
        
        # 3. Apply NLP insights if available
        if keywords:
            for product_id, product in product_map.items():
                category = product.get("category", "").lower()
                name = product.get("name", "").lower()
                attributes = product.get("attributes", {})
                
                # Extract attributes if they're stored as a JSON string
                if isinstance(attributes, str):
                    try:
                        attributes = json.loads(attributes)
                    except:
                        attributes = {}
                
                # Check for keyword matches
                for keyword in keywords:
                    keyword = keyword.lower()
                    if (keyword in category or keyword in name or 
                        any(keyword in str(attr).lower() for attr in attributes.values())):
                        product_scores[product_id] = product_scores.get(product_id, 0) + 3
        
        # 4. Apply reinforcement learning adjustments
        customer_rewards = self.recommendation_rewards.get(customer_id, {})
        for product_id, reward in customer_rewards.items():
            if product_id in product_scores:
                # Apply learned reward
                product_scores[product_id] += reward * 2
        
        # Exploration component for reinforcement learning
        if np.random.random() < self.exploration_rate:
            # Add random boost to some products for exploration
            for product_id in np.random.choice(list(product_scores.keys()), 
                                              size=min(5, len(product_scores)),
                                              replace=False):
                product_scores[product_id] += np.random.uniform(2, 5)
        
        # Sort products by final score
        sorted_products = sorted(
            [(product_id, score) for product_id, score in product_scores.items()],
            key=lambda x: x[1],
            reverse=True
        )
        
        # Generate final recommendations with explanations
        recommendations = []
        for product_id, score in sorted_products[:10]:  # Top 10 recommendations
            product = product_map.get(product_id, {})
            
            # Generate explanation for recommendation
            explanation = self._generate_recommendation_explanation(
                product, score, customer_data, context
            )
            
            recommendations.append({
                "product_id": product_id,
                "name": product.get("name", ""),
                "category": product.get("category", ""),
                "price": product.get("price", 0),
                "score": score,
                "explanation": explanation,
                "context_aware": context is not None
            })
        
        return recommendations
    
    def _generate_recommendation_explanation(self, product, score, customer_data, context):
        """Generate a natural language explanation for a recommendation."""
        explanations = []
        
        # Base on purchase history
        purchased_categories = [p.get("category") for p in customer_data.get("purchases", [])]
        if product.get("category") in purchased_categories:
            explanations.append(f"Based on your interest in {product.get('category')} products")
        
        # Based on browsing history
        viewed_ids = [p.get("id") for p in customer_data.get("viewed_products", [])]
        if product.get("id") in viewed_ids:
            explanations.append("You previously viewed this item")
        
        # Based on context if available
        if context:
            time_context = context.get("time_context", {})
            weather_context = context.get("weather_context", {})
            
            # Time context
            time_of_day = time_context.get("time_of_day")
            if time_of_day:
                time_phrases = {
                    "morning": "Perfect for your morning routine",
                    "afternoon": "Great for afternoon use",
                    "evening": "Ideal for your evening activities",
                    "night": "Recommended for nighttime"
                }
                if time_of_day in time_phrases:
                    explanations.append(time_phrases[time_of_day])
            
            # Weekend context
            is_weekend = time_context.get("is_weekend", False)
            if is_weekend:
                explanations.append("Great for weekend activities")
            
            # Season context
            season = time_context.get("season")
            if season:
                season_phrases = {
                    "spring": f"Perfect for spring weather",
                    "summer": f"Ideal for summer activities",
                    "fall": f"Great for fall season",
                    "winter": f"Recommended for winter use"
                }
                if season in season_phrases:
                    explanations.append(season_phrases[season])
            
            # Weather context
            if weather_context:
                conditions = weather_context.get("conditions", "").lower()
                is_precipitation = weather_context.get("is_precipitation", False)
                
                if is_precipitation:
                    explanations.append("Useful in today's rainy weather")
                
                if "snow" in conditions:
                    explanations.append("Recommended for today's snowy conditions")
                
                if "sunny" in conditions or "hot" in conditions:
                    explanations.append("Perfect for today's sunny weather")
        
        # If we have no explanations, add a generic one
        if not explanations:
            explanations.append("Recommended based on your preferences")
        
        # Add reinforcement learning context if applicable
        if product.get("id") in self.recommendation_rewards.get(customer_data.get("customer_id"), {}):
            explanations.append("Matches items you've shown interest in before")
        
        # Return the top 2 most relevant explanations
        return explanations[:2]
    
    def _store_recommendations(self, customer_id, recommendations):
        """Store recommendations in Neo4j."""
        if not recommendations:
            return False
        
        try:
            # First, delete old recommendations
            delete_query = """
            MATCH (c:Customer {customer_id: $customer_id})-[r:RECOMMENDED]->(p:Product)
            DELETE r
            """
            
            self.run_query(delete_query, {"customer_id": customer_id})
            
            # Store new recommendations
            for i, rec in enumerate(recommendations):
                store_query = """
                MATCH (c:Customer {customer_id: $customer_id})
                MATCH (p:Product {id: $product_id})
                CREATE (c)-[r:RECOMMENDED {
                    timestamp: datetime(),
                    rank: $rank,
                    score: $score,
                    explanation: $explanation,
                    context_aware: $context_aware
                }]->(p)
                RETURN r
                """
                
                self.run_query(store_query, {
                    "customer_id": customer_id,
                    "product_id": rec["product_id"],
                    "rank": i + 1,
                    "score": rec["score"],
                    "explanation": json.dumps(rec["explanation"]),
                    "context_aware": rec["context_aware"]
                })
            
            # Save a backup to file
            rec_file = Path(f"context_data/{customer_id}_recommendations_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
            with open(rec_file, 'w') as f:
                json.dump(recommendations, f, indent=2)
            
            return True
            
        except Exception as e:
            logging.error(f"Error storing recommendations: {e}")
            return False
    
    def record_customer_feedback(self, customer_id, product_id, action, score=None):
        """
        Record customer feedback on recommendations for reinforcement learning.
        
        Args:
            customer_id: The customer's ID
            product_id: The product's ID
            action: Action taken (view, click, purchase, ignore)
            score: Optional explicit score (1-5)
        """
        try:
            # Convert action to reward values for reinforcement learning
            action_rewards = {
                "view": 0.2,     # Customer viewed the product
                "click": 0.5,    # Customer clicked on the product
                "cart": 0.8,     # Customer added to cart
                "purchase": 1.0, # Customer purchased (strongest signal)
                "ignore": -0.1,  # Customer saw but ignored
                "dismiss": -0.3  # Customer explicitly dismissed recommendation
            }
            
            reward = action_rewards.get(action, 0)
            
            # If explicit score is provided, use that too
            if score is not None:
                normalized_score = max(0, min(1, (score - 1) / 4))  # Convert 1-5 to 0-1
                reward = (reward + normalized_score) / 2  # Combine signals
            
            # Store the feedback in Neo4j
            feedback_query = """
            MATCH (c:Customer {customer_id: $customer_id})
            MATCH (p:Product {id: $product_id})
            MERGE (c)-[f:FEEDBACK_ON]->(p)
            SET f.timestamp = datetime(),
                f.action = $action,
                f.score = $score,
                f.reward = $reward
            RETURN f
            """
            
            self.run_query(feedback_query, {
                "customer_id": customer_id,
                "product_id": product_id,
                "action": action,
                "score": score,
                "reward": reward
            })
            
            # Update reinforcement learning model
            self._update_recommendation_model(customer_id, product_id, reward)
            
            logging.info(f"Recorded feedback for customer {customer_id} on product {product_id}: {action}")
            return True
            
        except Exception as e:
            logging.error(f"Error recording customer feedback: {e}")
            return False
    
    def _update_recommendation_model(self, customer_id, product_id, reward):
        """Update the reinforcement learning model with new feedback."""
        if customer_id not in self.recommendation_rewards:
            self.recommendation_rewards[customer_id] = {}
        
        # Get current value
        current_value = self.recommendation_rewards[customer_id].get(product_id, 0)
        
        # Update with learning rate
        new_value = current_value + self.learning_rate * (reward - current_value)
        
        # Store updated value
        self.recommendation_rewards[customer_id][product_id] = new_value
        
        # Also update related products for collaborative filtering effect
        self._update_related_products(customer_id, product_id, reward)
        
        # Save the model periodically
        self._save_reinforcement_model()
    
    def _update_related_products(self, customer_id, product_id, reward):
        """Update rewards for related products for collaborative filtering."""
        # Get related products
        related_query = """
        MATCH (target:Product {id: $product_id})-[:RELATED_TO]->(related:Product)
        RETURN related.id as related_id
        
        UNION
        
        MATCH (target:Product {id: $product_id})<-[:RELATED_TO]-(related:Product)
        RETURN related.id as related_id
        
        UNION
        
        MATCH (target:Product {id: $product_id})-[:HAS_CATEGORY]->(c:Category)<-[:HAS_CATEGORY]-(related:Product)
        WHERE target <> related
        RETURN related.id as related_id
        """
        
        related_products = self.run_query(related_query, {"product_id": product_id})
        
        if not related_products:
            return
        
        # Apply smaller reward to related products
        smaller_reward = reward * self.discount_factor
        for related in related_products:
            related_id = related.get("related_id")
            if not related_id or related_id == product_id:
                continue
                
            current_value = self.recommendation_rewards[customer_id].get(related_id, 0)
            new_value = current_value + self.learning_rate * (smaller_reward - current_value)
            self.recommendation_rewards[customer_id][related_id] = new_value
    
    def _save_reinforcement_model(self):
        """Save the reinforcement learning model to disk."""
        try:
            # Convert to serializable format
            model_data = {
                "recommendation_rewards": dict(self.recommendation_rewards),
                "exploration_rate": self.exploration_rate,
                "learning_rate": self.learning_rate,
                "discount_factor": self.discount_factor,
                "timestamp": datetime.now().isoformat()
            }
            
            # Save to file
            model_file = Path("reinforcement_learning/recommendation_model.json")
            with open(model_file, 'w') as f:
                json.dump(model_data, f, indent=2)
            
            return True
        except Exception as e:
            logging.error(f"Error saving reinforcement learning model: {e}")
            return False
    
    def load_reinforcement_model(self):
        """Load the reinforcement learning model from disk."""
        try:
            model_file = Path("reinforcement_learning/recommendation_model.json")
            if not model_file.exists():
                logging.info("No existing reinforcement learning model found")
                return False
            
            with open(model_file, 'r') as f:
                model_data = json.load(f)
            
            # Load model parameters
            self.recommendation_rewards = defaultdict(dict, model_data.get("recommendation_rewards", {}))
            self.exploration_rate = model_data.get("exploration_rate", 0.2)
            self.learning_rate = model_data.get("learning_rate", 0.1)
            self.discount_factor = model_data.get("discount_factor", 0.9)
            
            logging.info("Loaded reinforcement learning model")
            return True
        except Exception as e:
            logging.error(f"Error loading reinforcement learning model: {e}")
            return False
    
    def decay_exploration_rate(self, min_rate=0.05, decay_factor=0.95):
        """Decay the exploration rate over time as the model learns."""
        self.exploration_rate = max(min_rate, self.exploration_rate * decay_factor)
        return self.exploration_rate
    
    def create_enhanced_context_schema(self):
        """
        Create Neo4j schema for enhanced personalization features.
        This will set up the necessary constraints and indexes.
        """
        try:
            # Create constraints and indexes
            constraints = [
                """
                CREATE CONSTRAINT IF NOT EXISTS FOR (f:Feedback) REQUIRE f.id IS UNIQUE
                """,
                """
                CREATE CONSTRAINT IF NOT EXISTS FOR (i:NLPInsight) REQUIRE i.customer_id IS UNIQUE
                """,
                """
                CREATE CONSTRAINT IF NOT EXISTS FOR (t:Topic) REQUIRE t.id IS UNIQUE
                """,
                """
                CREATE CONSTRAINT IF NOT EXISTS FOR (c:Context) REQUIRE c.customer_id IS UNIQUE
                """,
                """
                CREATE CONSTRAINT IF NOT EXISTS FOR (g:GlobalNLPInsight) REQUIRE g.id IS UNIQUE
                """
            ]
            
            indexes = [
                """
                CREATE INDEX IF NOT EXISTS FOR (f:Feedback) ON (f.timestamp)
                """,
                """
                CREATE INDEX IF NOT EXISTS FOR (f:Feedback) ON (f.source)
                """,
                """
                CREATE INDEX IF NOT EXISTS FOR (c:Context) ON (c.timestamp)
                """
            ]
            
            # Execute all constraints and indexes
            for query in constraints + indexes:
                self.run_query(query)
            
            logging.info("Created Neo4j schema for enhanced personalization")
            return True
            
        except Exception as e:
            logging.error(f"Error creating schema: {e}")
            return False
    
    def run_phase5_personalization(self, customer_id=None):
        """
        Run enhanced personalization for a customer or all customers.
        
        Args:
            customer_id: Optional, process just one customer or all if None
        """
        results = {}
        
        # Connect to Neo4j
        if not self.connect():
            return {"status": "error", "message": "Failed to connect to Neo4j database"}
        
        try:
            # Create schema if needed
            self.create_enhanced_context_schema()
            
            # Initialize NLP models
            self.initialize_nlp_models()
            
            # Load existing reinforcement learning model
            self.load_reinforcement_model()
            
            # If customer_id provided, process just that customer
            if customer_id:
                customers = [{"customer_id": customer_id}]
            else:
                # Get all customers
                query = """
                MATCH (c:Customer)
                RETURN c.customer_id as customer_id
                """
                customers = self.run_query(query)
            
            if not customers:
                return {"status": "error", "message": "No customers found"}
            
            # Track successful operations
            success_count = {"nlp": 0, "context": 0, "recommendations": 0}
            
            # Process each customer
            for customer in customers:
                customer_id = customer["customer_id"]
                logging.info(f"Processing customer {customer_id}")
                
                # 1. Analyze customer feedback with NLP
                nlp_result = self.analyze_customer_feedback(customer_id)
                if nlp_result:
                    success_count["nlp"] += 1
                
                # 2. Get context data
                context = self.get_context_data(customer_id)
                if context:
                    success_count["context"] += 1
                
                # 3. Generate context-aware recommendations
                recommendations = self.generate_context_aware_recommendations(customer_id)
                if recommendations:
                    success_count["recommendations"] += 1
                
                # 4. Decay exploration rate for reinforcement learning
                self.decay_exploration_rate()
            
            # Generate result summary
            results = {
                "status": "success",
                "customers_processed": len(customers),
                "nlp_analysis_success": success_count["nlp"],
                "context_data_success": success_count["context"],
                "recommendations_success": success_count["recommendations"],
                "exploration_rate": self.exploration_rate,
                "timestamp": datetime.now().isoformat()
            }
            
            # Save the state of our models
            self._save_reinforcement_model()
            
            logging.info(f"Phase 5 personalization completed for {len(customers)} customers")
            return results
            
        except Exception as e:
            logging.error(f"Error running Phase 5 personalization: {e}")
            return {"status": "error", "message": str(e)}
            
        finally:
            self.close()

if __name__ == "__main__":
    print("Starting Enhanced Personalization...")
    enhancer = EnhancedPersonalization()
    results = enhancer.run_phase5_personalization()
    print(f"Enhanced personalization completed with status: {results.get('status', 'unknown')}")
    print(f"Processed {results.get('customers_processed', 0)} customers")
