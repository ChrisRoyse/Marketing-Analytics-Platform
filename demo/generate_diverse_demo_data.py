#!/usr/bin/env python
"""
Generate diverse customer journey data for Marketing Ontology Platform Demo.

This script creates customer profiles with a balanced distribution of metrics:
- At least 4 out of 15 customers with high churn probability
- Varied customer lifetime values (current and predicted)
- Diverse next purchase day predictions
- A mix of purchase frequency and recency patterns
"""

import json
import random
import uuid
import datetime
import os
import sys
import argparse
from pathlib import Path

# Add parent directory to path to access shared modules
sys.path.append(str(Path(__file__).resolve().parent.parent))

class DiverseDemoDataGenerator:
    """Generates diverse synthetic customer journey data with varied metrics."""
    
    def __init__(self, config=None):
        """
        Initialize the data generator with configuration options.
        
        Args:
            config (dict): Configuration settings for data generation.
                           If None, default settings will be used.
        """
        self.config = config or self._get_default_config()
        self.output_dir = self.config["output_dir"]
        self.customers = []
        self.all_entities = {
            "customers": [],
            "products": self.config["products"],
            "advertisements": self.config["advertisements"],
            "emails": self.config["emails"],
            "pages": self.config["pages"],
            "locations": self.config["locations"],
            "devices": [{"id": d} for d in self.config["devices"]],
            "channels": [{"id": c} for c in self.config["channels"]],
            "funnel_stages": [{"id": s} for s in self.config["funnel_stages"]],
            "persona_groups": [{"id": p} for p in self.config["persona_groups"].keys()],
            "personas": [],
        }
        
        # Create persona entities
        for group, data in self.config["persona_groups"].items():
            for profile in data["profiles"]:
                self.all_entities["personas"].append({
                    "id": profile.replace(" ", "_").lower(),
                    "name": profile,
                    "group": group
                })
                
        # Ensure output directory exists
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Set start and end dates for data generation
        self.start_date = datetime.datetime.fromisoformat(self.config["start_date"])
        self.end_date = datetime.datetime.fromisoformat(self.config["end_date"])
        
    def _get_default_config(self):
        """Return default configuration for data generation."""
        default_config = {
            "output_dir": "/home/cabdru/marketing/demo/demo_data",
            "scenario_name": "diverse_metrics",
            "start_date": (datetime.datetime.now() - datetime.timedelta(days=365)).isoformat(),
            "end_date": datetime.datetime.now().isoformat(),
            "purchase_frequency_multiplier": 1.0,
            "price_sensitivity_multiplier": 1.0,
            "purchase_interval_min": 30,
            "purchase_interval_variance": 60,
            "customer_profiles": [
                # HIGH CHURN RISK CUSTOMERS (at least 4)
                {
                    "id": "high_churn_1",
                    "segment": "Gift Buyer",
                    "persona": "Seasonal Gifter",
                    "purchase_count": 1,
                    "days_since_last_purchase": 140,
                    "purchase_interval_days": 180,
                    "value_tier": "medium",
                    "will_churn": True
                },
                {
                    "id": "high_churn_2",
                    "segment": "Budget Shopper",
                    "persona": "Deal Seeker",
                    "purchase_count": 2,
                    "days_since_last_purchase": 80,
                    "purchase_interval_days": 120,
                    "value_tier": "low",
                    "will_churn": True
                },
                {
                    "id": "high_churn_3",
                    "segment": "Student",
                    "persona": "Budget Student",
                    "purchase_count": 1,
                    "days_since_last_purchase": 150,
                    "purchase_interval_days": 150,
                    "value_tier": "low",
                    "will_churn": True
                },
                {
                    "id": "high_churn_4",
                    "segment": "Student",
                    "persona": "International Student",
                    "purchase_count": 2,
                    "days_since_last_purchase": 90,
                    "purchase_interval_days": 150,
                    "value_tier": "medium",
                    "will_churn": True
                },
                {
                    "id": "high_churn_5",
                    "segment": "Gift Buyer",
                    "persona": "Special Occasion Buyer",
                    "purchase_count": 2,
                    "days_since_last_purchase": 120,
                    "purchase_interval_days": 180,
                    "value_tier": "medium",
                    "will_churn": True
                },
                
                # MEDIUM CHURN RISK CUSTOMERS
                {
                    "id": "medium_churn_1",
                    "segment": "Budget Shopper",
                    "persona": "Comparison Shopper",
                    "purchase_count": 2,
                    "days_since_last_purchase": 50,
                    "purchase_interval_days": 120,
                    "value_tier": "medium",
                    "will_churn": False
                },
                {
                    "id": "medium_churn_2",
                    "segment": "Professional",
                    "persona": "Remote Worker",
                    "purchase_count": 2,
                    "days_since_last_purchase": 40,
                    "purchase_interval_days": 90,
                    "value_tier": "high",
                    "will_churn": False
                },
                {
                    "id": "medium_churn_3",
                    "segment": "Tech Enthusiast",
                    "persona": "Feature Hunter",
                    "purchase_count": 1,
                    "days_since_last_purchase": 45,
                    "purchase_interval_days": 75,
                    "value_tier": "high",
                    "will_churn": False
                },
                {
                    "id": "medium_churn_4",
                    "segment": "Student",
                    "persona": "Tech Student",
                    "purchase_count": 2,
                    "days_since_last_purchase": 30,
                    "purchase_interval_days": 150,
                    "value_tier": "medium",
                    "will_churn": False
                },
                
                # LOW CHURN RISK CUSTOMERS
                {
                    "id": "low_churn_1",
                    "segment": "Tech Enthusiast",
                    "persona": "Early Adopter",
                    "purchase_count": 4,
                    "days_since_last_purchase": 20,
                    "purchase_interval_days": 75,
                    "value_tier": "high",
                    "will_churn": False
                },
                {
                    "id": "low_churn_2",
                    "segment": "Professional",
                    "persona": "Business User",
                    "purchase_count": 5,
                    "days_since_last_purchase": 15,
                    "purchase_interval_days": 90,
                    "value_tier": "high",
                    "will_churn": False
                },
                {
                    "id": "low_churn_3",
                    "segment": "Tech Enthusiast",
                    "persona": "Upgrade Cycler",
                    "purchase_count": 3,
                    "days_since_last_purchase": 25,
                    "purchase_interval_days": 75,
                    "value_tier": "medium",
                    "will_churn": False
                },
                {
                    "id": "low_churn_4",
                    "segment": "Professional",
                    "persona": "Executive Buyer",
                    "purchase_count": 3,
                    "days_since_last_purchase": 10,
                    "purchase_interval_days": 90,
                    "value_tier": "high",
                    "will_churn": False
                },
                {
                    "id": "low_churn_5",
                    "segment": "Budget Shopper",
                    "persona": "Bargain Hunter",
                    "purchase_count": 3,
                    "days_since_last_purchase": 15,
                    "purchase_interval_days": 120,
                    "value_tier": "low",
                    "will_churn": False
                },
                {
                    "id": "low_churn_6",
                    "segment": "Gift Buyer",
                    "persona": "Corporate Gifter",
                    "purchase_count": 2,
                    "days_since_last_purchase": 20,
                    "purchase_interval_days": 180,
                    "value_tier": "high",
                    "will_churn": False
                }
            ],
            "persona_groups": {
                "Tech Enthusiast": {
                    "profiles": ["Early Adopter", "Feature Hunter", "Upgrade Cycler"],
                    "avg_purchase_interval_days": 75,
                    "price_sensitivity": "low",
                    "research_depth": "high",
                    "churn_rate": 0.1,
                },
                "Budget Shopper": {
                    "profiles": ["Deal Seeker", "Comparison Shopper", "Bargain Hunter"],
                    "avg_purchase_interval_days": 120,
                    "price_sensitivity": "high",
                    "research_depth": "medium",
                    "churn_rate": 0.3,
                },
                "Gift Buyer": {
                    "profiles": ["Seasonal Gifter", "Special Occasion Buyer", "Corporate Gifter"],
                    "avg_purchase_interval_days": 180,
                    "price_sensitivity": "medium",
                    "research_depth": "low",
                    "churn_rate": 0.5,
                },
                "Professional": {
                    "profiles": ["Business User", "Remote Worker", "Executive Buyer"],
                    "avg_purchase_interval_days": 90,
                    "price_sensitivity": "low",
                    "research_depth": "high",
                    "churn_rate": 0.2,
                },
                "Student": {
                    "profiles": ["Budget Student", "Tech Student", "International Student"],
                    "avg_purchase_interval_days": 150,
                    "price_sensitivity": "high",
                    "research_depth": "medium",
                    "churn_rate": 0.4,
                },
            },
            "channels": ["facebook", "instagram", "google_search", "email", "direct", "referral"],
            "funnel_stages": ["awareness", "consideration", "intent", "conversion", "retention", "advocacy"],
            "devices": ["desktop_chrome", "desktop_firefox", "desktop_safari", "mobile_chrome", "mobile_safari", "tablet_chrome"],
            "locations": [
                {"city": "New York", "state": "NY", "country": "USA", "postal_code": "10001"},
                {"city": "San Francisco", "state": "CA", "country": "USA", "postal_code": "94105"},
                {"city": "Austin", "state": "TX", "country": "USA", "postal_code": "78701"},
                {"city": "Seattle", "state": "WA", "country": "USA", "postal_code": "98101"},
                {"city": "Chicago", "state": "IL", "country": "USA", "postal_code": "60601"},
                {"city": "London", "city_code": "LD", "country": "UK", "postal_code": "EC1A 1BB"},
                {"city": "Toronto", "province": "ON", "country": "Canada", "postal_code": "M5V 2A8"},
                {"city": "Sydney", "state": "NSW", "country": "Australia", "postal_code": "2000"},
            ],
            "products": [
                {"id": "PRD001", "name": "Premium Laptop", "category": "Computers", "price": 1299.99},
                {"id": "PRD002", "name": "Wireless Earbuds", "category": "Audio", "price": 129.99},
                {"id": "PRD003", "name": "Smartphone", "category": "Mobile", "price": 899.99},
                {"id": "PRD004", "name": "Smart Watch", "category": "Wearables", "price": 249.99},
                {"id": "PRD005", "name": "4K Monitor", "category": "Displays", "price": 349.99},
                {"id": "PRD006", "name": "Wireless Keyboard", "category": "Accessories", "price": 79.99},
                {"id": "PRD007", "name": "Wireless Mouse", "category": "Accessories", "price": 49.99},
                {"id": "PRD008", "name": "External SSD", "category": "Storage", "price": 159.99},
                {"id": "PRD009", "name": "Bluetooth Speaker", "category": "Audio", "price": 199.99},
                {"id": "PRD010", "name": "Tablet", "category": "Computers", "price": 499.99},
                {"id": "PRD011", "name": "Noise-Cancelling Headphones", "category": "Audio", "price": 299.99},
                {"id": "PRD012", "name": "Webcam", "category": "Accessories", "price": 89.99},
                {"id": "PRD013", "name": "Gaming Mouse", "category": "Gaming", "price": 69.99},
                {"id": "PRD014", "name": "Mechanical Keyboard", "category": "Gaming", "price": 149.99},
                {"id": "PRD015", "name": "WiFi Router", "category": "Networking", "price": 179.99},
            ],
            "advertisements": [
                {"id": "AD001", "name": "Summer Sale", "channel": "facebook", "campaign": "seasonal_promotions"},
                {"id": "AD002", "name": "Back to School", "channel": "instagram", "campaign": "seasonal_promotions"},
                {"id": "AD003", "name": "Tech Deals", "channel": "google_search", "campaign": "always_on"},
                {"id": "AD004", "name": "New Arrivals", "channel": "email", "campaign": "product_launches"},
                {"id": "AD005", "name": "Holiday Gifts", "channel": "facebook", "campaign": "seasonal_promotions"},
                {"id": "AD006", "name": "Upgrade Your Setup", "channel": "google_search", "campaign": "always_on"},
            ],
            "emails": [
                {"id": "EM001", "subject": "Welcome to TechGear", "type": "welcome"},
                {"id": "EM002", "subject": "Your cart is waiting", "type": "abandoned_cart"},
                {"id": "EM003", "subject": "Exclusive deals just for you", "type": "promotional"},
                {"id": "EM004", "subject": "Thank you for your purchase", "type": "transactional"},
                {"id": "EM005", "subject": "New products you might like", "type": "recommendation"},
                {"id": "EM006", "subject": "Rate your recent purchase", "type": "feedback"},
            ],
            "pages": [
                {"id": "PG001", "url": "/", "name": "Home"},
                {"id": "PG002", "url": "/products", "name": "Products"},
                {"id": "PG003", "url": "/category/computers", "name": "Computers Category"},
                {"id": "PG004", "url": "/category/audio", "name": "Audio Category"},
                {"id": "PG005", "url": "/category/accessories", "name": "Accessories Category"},
                {"id": "PG006", "url": "/cart", "name": "Shopping Cart"},
                {"id": "PG007", "url": "/checkout", "name": "Checkout"},
                {"id": "PG008", "url": "/account", "name": "My Account"},
                {"id": "PG009", "url": "/support", "name": "Customer Support"},
                {"id": "PG010", "url": "/blog", "name": "Blog"},
            ],
        }
        return default_config
    
    def generate_customer_base(self):
        """Generate diverse customer profiles with pre-configured characteristics."""
        customer_id = 1
        
        # Use the pre-configured customer profiles
        for profile in self.config["customer_profiles"]:
            # Create customer with basic info
            customer = {
                "customer_id": f"CUST{customer_id:03d}",
                "profile": {
                    "first_name": self._generate_first_name(),
                    "last_name": self._generate_last_name(),
                    "email": f"{self._generate_first_name().lower()}.{self._generate_last_name().lower()}@example.com",
                    "phone": f"555-{random.randint(100, 999)}-{random.randint(1000, 9999)}",
                    "age": random.randint(18, 65),
                    "gender": random.choice(["Male", "Female", "Non-binary"]),
                    "location": random.choice(self.config["locations"]),
                    "registration_date": self._generate_date(
                        start_date=self.start_date,
                        end_offset=0.3  # Register in first 30% of timeline
                    ),
                    "segment": profile["segment"],
                    "personas": [profile["persona"]]
                },
                "devices": random.sample(self.config["devices"], random.randint(1, 3)),
                "journey_events": [],
                # Store the profile configuration for journey generation
                "_config": {
                    "purchase_count": profile["purchase_count"],
                    "days_since_last_purchase": profile["days_since_last_purchase"],
                    "purchase_interval_days": profile["purchase_interval_days"],
                    "value_tier": profile["value_tier"],
                    "will_churn": profile["will_churn"]
                }
            }
            
            # Add to customer list
            self.customers.append(customer)
            
            # Add to entities
            self.all_entities["customers"].append({
                "customer_id": customer["customer_id"],
                "first_name": customer["profile"]["first_name"],
                "last_name": customer["profile"]["last_name"],
                "email": customer["profile"]["email"]
            })
            
            customer_id += 1
        
        print(f"Generated {len(self.customers)} customer profiles")
        return self.customers
    
    def generate_journey_events(self):
        """Generate journey events based on each customer's pre-configured characteristics."""
        for customer in self.customers:
            persona_group = customer["profile"]["segment"]
            persona_data = self.config["persona_groups"][persona_group]
            profile_config = customer["_config"]
            
            # Starting point for journey (registration date or slightly earlier for ad exposure)
            registration_date = datetime.datetime.fromisoformat(customer["profile"]["registration_date"])
            journey_start = registration_date - datetime.timedelta(days=random.randint(1, 14))
            if journey_start < self.start_date:
                journey_start = self.start_date
            
            # Will this customer convert at least once?
            will_convert = profile_config["purchase_count"] > 0
            
            # Will this customer churn after conversion?
            will_churn = profile_config["will_churn"]
            
            # Current timestamp for event generation
            current_time = journey_start
            
            # Channel first encountered
            primary_channel = random.choice(self.config["channels"])
            
            # Add awareness events
            current_time = self._add_awareness_events(customer, current_time, primary_channel)
            
            # Consideration to first purchase typically takes 2-4 weeks
            consideration_days = random.randint(14, 28)
            current_time = self._add_consideration_events(
                customer, 
                current_time, 
                days_spent=consideration_days,
                research_depth=persona_data["research_depth"]
            )
            
            # Intent events leading to first purchase (or not)
            intent_days = random.randint(3, 10)
            current_time = self._add_intent_events(
                customer, 
                current_time,
                days_spent=intent_days,
                will_convert=will_convert,
                price_sensitivity=persona_data["price_sensitivity"]
            )
            
            # Add first conversion and subsequent purchases
            if will_convert:
                # Calculate purchase timestamps backward from now
                purchase_timestamps = self._calculate_purchase_timestamps(
                    profile_config["purchase_count"],
                    profile_config["days_since_last_purchase"],
                    profile_config["purchase_interval_days"],
                    self.end_date
                )
                
                for purchase_time in purchase_timestamps:
                    # Skip if purchase time is before registration
                    if purchase_time < registration_date:
                        continue
                    
                    # Add consideration and intent events before purchase
                    # Only for purchases after the first one
                    if purchase_time != purchase_timestamps[0]:
                        pre_purchase_days = random.randint(3, 10)
                        temp_time = purchase_time - datetime.timedelta(days=pre_purchase_days)
                        
                        # Brief consideration events
                        temp_time = self._add_consideration_events(
                            customer,
                            temp_time,
                            days_spent=max(1, pre_purchase_days // 2),
                            research_depth=persona_data["research_depth"]
                        )
                        
                        # Brief intent events
                        temp_time = self._add_intent_events(
                            customer,
                            temp_time,
                            days_spent=max(1, pre_purchase_days // 2),
                            will_convert=True,
                            price_sensitivity=persona_data["price_sensitivity"]
                        )
                    
                    # Add the purchase with appropriate value tier
                    self._add_conversion_events(
                        customer, 
                        purchase_time,
                        value_tier=profile_config["value_tier"]
                    )
                
                # Add retention events after all purchases
                # Calculate retention period based on last purchase
                last_purchase_time = purchase_timestamps[-1]
                retention_days = (self.end_date - last_purchase_time).days
                
                if will_churn:
                    # For churning customers, cap retention period
                    retention_days = min(retention_days, random.randint(30, 120))
                
                if retention_days > 0:
                    current_time = self._add_retention_events(
                        customer, 
                        last_purchase_time,
                        days_spent=retention_days,
                        will_churn=will_churn
                    )
                    
                    # Add advocacy events if not churning and some time still available
                    if not will_churn and (self.end_date - current_time).days > 15:
                        self._add_advocacy_events(customer, current_time)
            
            # Sort events by timestamp
            customer["journey_events"].sort(key=lambda x: x["timestamp"])
            
            # Remove temporary config
            del customer["_config"]
            
            print(f"Generated {len(customer['journey_events'])} events for {customer['customer_id']}")
        
        return self.customers
    
    def _calculate_purchase_timestamps(self, purchase_count, days_since_last_purchase, base_interval, end_date):
        """Calculate purchase timestamps based on the customer profile parameters."""
        timestamps = []
        
        # Start from the end date and work backward
        current_time = end_date - datetime.timedelta(days=days_since_last_purchase)
        timestamps.append(current_time)
        
        # Add previous purchases
        for i in range(1, purchase_count):
            # Add variation to the interval
            interval_variance = random.randint(-10, 20)
            interval = base_interval + interval_variance
            
            # Move back in time
            current_time = current_time - datetime.timedelta(days=interval)
            timestamps.append(current_time)
        
        # Reverse the list to have chronological order
        timestamps.reverse()
        
        return timestamps
    
    def _generate_date(self, start_date, end_offset=1.0):
        """
        Generate a random date within a portion of the available date range.
        
        Args:
            start_date: The starting date
            end_offset: How far into the date range to go (0.0-1.0)
        
        Returns:
            A datetime string in ISO format
        """
        end_date = self.start_date + (self.end_date - self.start_date) * end_offset
        delta = end_date - start_date
        
        if delta.days <= 0:
            return start_date.isoformat()
            
        random_days = random.randint(0, delta.days)
        random_seconds = random.randint(0, 86399)  # 24 hours in seconds
        random_date = start_date + datetime.timedelta(days=random_days, seconds=random_seconds)
        return random_date.isoformat()
    
    def _add_awareness_events(self, customer, start_time, primary_channel):
        """Add awareness stage events to the customer journey."""
        current_time = start_time
        
        # Ad view event
        if primary_channel in ["facebook", "instagram", "google_search"]:
            ad = next((ad for ad in self.config["advertisements"] if ad["channel"] == primary_channel), 
                      random.choice(self.config["advertisements"]))
            self._add_event(customer, {
                "event_type": "VIEWS",
                "target_type": "Advertisement",
                "target_id": ad["id"],
                "timestamp": self._timestamp_str(current_time),
                "channel": primary_channel,
                "properties": {
                    "duration": random.randint(1, 10)
                }
            })
            
            # Maybe they click the ad
            if random.random() > 0.5:
                current_time = current_time + datetime.timedelta(seconds=random.randint(1, 30))
                self._add_event(customer, {
                    "event_type": "CLICKS_ON",
                    "target_type": "Advertisement",
                    "target_id": ad["id"],
                    "timestamp": self._timestamp_str(current_time),
                    "channel": primary_channel,
                    "properties": {}
                })
                
                # They arrive at the website
                current_time = current_time + datetime.timedelta(seconds=random.randint(3, 10))
                self._add_event(customer, {
                    "event_type": "COMES_FROM",
                    "target_type": "Channel",
                    "target_id": primary_channel,
                    "timestamp": self._timestamp_str(current_time),
                    "properties": {
                        "referrer": f"{primary_channel}_ad"
                    }
                })
        else:
            # Direct traffic or email or referral
            self._add_event(customer, {
                "event_type": "COMES_FROM",
                "target_type": "Channel",
                "target_id": primary_channel,
                "timestamp": self._timestamp_str(current_time),
                "properties": {
                    "referrer": primary_channel
                }
            })
        
        # Initial page visit
        current_time = current_time + datetime.timedelta(seconds=random.randint(1, 5))
        home_page = next(page for page in self.config["pages"] if page["name"] == "Home")
        self._add_event(customer, {
            "event_type": "VISITS",
            "target_type": "Page",
            "target_id": home_page["id"],
            "timestamp": self._timestamp_str(current_time),
            "properties": {
                "duration": random.randint(10, 120),
                "device": random.choice(customer["devices"])
            }
        })
        
        # Update current time by a few hours to a few days
        current_time = current_time + datetime.timedelta(hours=random.randint(1, 72))
        
        return current_time
    
    def _add_consideration_events(self, customer, start_time, days_spent, research_depth):
        """Add consideration stage events to the customer journey."""
        current_time = start_time
        end_time = start_time + datetime.timedelta(days=days_spent)
        
        # Number of products to view based on research depth
        if research_depth == "high":
            num_products = random.randint(5, 10)
        elif research_depth == "medium":
            num_products = random.randint(3, 6)
        else:  # low
            num_products = random.randint(1, 3)
        
        # Products this customer is interested in
        interested_products = random.sample(self.config["products"], min(num_products, len(self.config["products"])))
        
        # Generate product browsing events spread over the consideration period
        while current_time < end_time and interested_products:
            # Visit a category page
            current_time = current_time + datetime.timedelta(hours=random.randint(1, 24))
            if current_time >= end_time:
                break
                
            # Pick a category page related to products they're interested in
            categories = set(p["category"] for p in interested_products)
            category = random.choice(list(categories))
            category_page = next(
                (p for p in self.config["pages"] if p["name"] == f"{category} Category"), 
                next(p for p in self.config["pages"] if "Category" in p["name"])
            )
            
            self._add_event(customer, {
                "event_type": "VISITS",
                "target_type": "Page",
                "target_id": category_page["id"],
                "timestamp": self._timestamp_str(current_time),
                "properties": {
                    "duration": random.randint(20, 180),
                    "device": random.choice(customer["devices"])
                }
            })
            
            # View some products in this category
            category_products = [p for p in interested_products if p["category"] == category]
            for product in category_products:
                current_time = current_time + datetime.timedelta(minutes=random.randint(1, 10))
                if current_time >= end_time:
                    break
                    
                self._add_event(customer, {
                    "event_type": "VIEWS",
                    "target_type": "Product",
                    "target_id": product["id"],
                    "timestamp": self._timestamp_str(current_time),
                    "properties": {
                        "duration": random.randint(30, 300),
                        "device": random.choice(customer["devices"])
                    }
                })
                
                # Remove from interested products so we don't view it again
                if random.random() > 0.7:  # 30% chance to revisit
                    interested_products.remove(product)
            
            # Possibly visit the blog for research
            if research_depth in ["high", "medium"] and random.random() > 0.6:
                current_time = current_time + datetime.timedelta(minutes=random.randint(10, 60))
                if current_time >= end_time:
                    break
                    
                blog_page = next(page for page in self.config["pages"] if page["name"] == "Blog")
                self._add_event(customer, {
                    "event_type": "VISITS",
                    "target_type": "Page",
                    "target_id": blog_page["id"],
                    "timestamp": self._timestamp_str(current_time),
                    "properties": {
                        "duration": random.randint(60, 600),
                        "device": random.choice(customer["devices"])
                    }
                })
        
        return current_time
    
    def _add_intent_events(self, customer, start_time, days_spent, will_convert, price_sensitivity):
        """Add intent stage events to the customer journey."""
        current_time = start_time
        end_time = start_time + datetime.timedelta(days=days_spent)
        
        # Products they'll consider adding to cart
        cart_products = random.sample(self.config["products"], random.randint(1, 3))
        
        # Account creation (if they'll convert or randomly)
        if will_convert or random.random() > 0.7:
            account_page = next(page for page in self.config["pages"] if page["name"] == "My Account")
            current_time = current_time + datetime.timedelta(hours=random.randint(1, 12))
            
            # Visit account page
            self._add_event(customer, {
                "event_type": "VISITS",
                "target_type": "Page",
                "target_id": account_page["id"],
                "timestamp": self._timestamp_str(current_time),
                "properties": {
                    "duration": random.randint(60, 300),
                    "device": random.choice(customer["devices"])
                }
            })
            
            # Add account creation event
            current_time = current_time + datetime.timedelta(minutes=random.randint(2, 5))
            self._add_event(customer, {
                "event_type": "CREATES",
                "target_type": "Account",
                "target_id": "ACCOUNT",
                "timestamp": self._timestamp_str(current_time),
                "properties": {
                    "registration_source": "website",
                    "device": random.choice(customer["devices"])
                }
            })
        
        # Cart additions
        for product in cart_products:
            current_time = current_time + datetime.timedelta(hours=random.randint(1, 48))
            if current_time >= end_time:
                break
                
            # View the product again
            self._add_event(customer, {
                "event_type": "VIEWS",
                "target_type": "Product",
                "target_id": product["id"],
                "timestamp": self._timestamp_str(current_time),
                "properties": {
                    "duration": random.randint(30, 300),
                    "device": random.choice(customer["devices"])
                }
            })
            
            # Apply price sensitivity multiplier
            price_sensitivity_factor = 1.0
            if self.config.get("price_sensitivity_multiplier"):
                price_sensitivity_factor = self.config["price_sensitivity_multiplier"]
                
            # Add to cart if price sensitivity allows or randomly
            if (price_sensitivity == "low" or 
                (price_sensitivity == "medium" and product["price"] < 300 * price_sensitivity_factor) or
                (price_sensitivity == "high" and product["price"] < 150 * price_sensitivity_factor) or
                random.random() > 0.7):
                
                current_time = current_time + datetime.timedelta(minutes=random.randint(1, 10))
                if current_time >= end_time:
                    break
                    
                self._add_event(customer, {
                    "event_type": "ADDS_TO_CART",
                    "target_type": "Product",
                    "target_id": product["id"],
                    "timestamp": self._timestamp_str(current_time),
                    "properties": {
                        "quantity": random.randint(1, 2),
                        "device": random.choice(customer["devices"])
                    }
                })
                
                # Visit cart
                current_time = current_time + datetime.timedelta(minutes=random.randint(1, 5))
                if current_time >= end_time:
                    break
                    
                cart_page = next(page for page in self.config["pages"] if page["name"] == "Shopping Cart")
                self._add_event(customer, {
                    "event_type": "VISITS",
                    "target_type": "Page",
                    "target_id": cart_page["id"],
                    "timestamp": self._timestamp_str(current_time),
                    "properties": {
                        "duration": random.randint(30, 180),
                        "device": random.choice(customer["devices"])
                    }
                })
                
                # Abandon cart if not converting or randomly
                if not will_convert or random.random() > 0.7:
                    current_time = current_time + datetime.timedelta(minutes=random.randint(2, 10))
                    if current_time >= end_time:
                        break
                        
                    self._add_event(customer, {
                        "event_type": "ABANDONS",
                        "target_type": "Cart",
                        "target_id": "CART",
                        "timestamp": self._timestamp_str(current_time),
                        "properties": {
                            "value": product["price"] * random.randint(1, 2),
                            "items": 1,
                            "reason": random.choice([
                                "price_too_high", "shipping_cost", "just_browsing",
                                "found_elsewhere", "technical_issue", "undecided"
                            ])
                        }
                    })
                    
                    # They might receive an abandoned cart email
                    if random.random() > 0.6:
                        current_time = current_time + datetime.timedelta(hours=random.randint(1, 24))
                        if current_time >= end_time:
                            break
                            
                        abandon_email = next(email for email in self.config["emails"] if email["type"] == "abandoned_cart")
                        self._add_event(customer, {
                            "event_type": "RECEIVES",
                            "target_type": "Email",
                            "target_id": abandon_email["id"],
                            "timestamp": self._timestamp_str(current_time),
                            "properties": {
                                "subject": abandon_email["subject"]
                            }
                        })
                        
                        # They might open the email
                        if random.random() > 0.4:
                            current_time = current_time + datetime.timedelta(hours=random.randint(1, 48))
                            if current_time >= end_time:
                                break
                                
                            self._add_event(customer, {
                                "event_type": "OPENS",
                                "target_type": "Email",
                                "target_id": abandon_email["id"],
                                "timestamp": self._timestamp_str(current_time),
                                "properties": {
                                    "device": random.choice(customer["devices"])
                                }
                            })
                            
                            # They might click the email
                            if random.random() > 0.5:
                                current_time = current_time + datetime.timedelta(seconds=random.randint(10, 60))
                                if current_time >= end_time:
                                    break
                                    
                                self._add_event(customer, {
                                    "event_type": "CLICKS_ON",
                                    "target_type": "Email",
                                    "target_id": abandon_email["id"],
                                    "timestamp": self._timestamp_str(current_time),
                                    "properties": {
                                        "device": random.choice(customer["devices"])
                                    }
                                })
                                
                                # Return to cart
                                current_time = current_time + datetime.timedelta(seconds=random.randint(5, 20))
                                if current_time >= end_time:
                                    break
                                    
                                self._add_event(customer, {
                                    "event_type": "COMES_FROM",
                                    "target_type": "Channel",
                                    "target_id": "email",
                                    "timestamp": self._timestamp_str(current_time),
                                    "properties": {
                                        "referrer": "abandoned_cart_email"
                                    }
                                })
                                
                                current_time = current_time + datetime.timedelta(seconds=random.randint(1, 5))
                                if current_time >= end_time:
                                    break
                                    
                                self._add_event(customer, {
                                    "event_type": "VISITS",
                                    "target_type": "Page",
                                    "target_id": cart_page["id"],
                                    "timestamp": self._timestamp_str(current_time),
                                    "properties": {
                                        "duration": random.randint(30, 180),
                                        "device": random.choice(customer["devices"])
                                    }
                                })
                            
        return current_time
    
    def _add_conversion_events(self, customer, current_time, value_tier="medium"):
        """Add conversion stage events to the customer journey with value tier."""
        # Visit checkout page
        checkout_page = next(page for page in self.config["pages"] if page["name"] == "Checkout")
        self._add_event(customer, {
            "event_type": "VISITS",
            "target_type": "Page",
            "target_id": checkout_page["id"],
            "timestamp": self._timestamp_str(current_time),
            "properties": {
                "duration": random.randint(120, 600),
                "device": random.choice(customer["devices"])
            }
        })
        
        # Select products based on value tier
        if value_tier == "high":
            # High value: expensive products, possibly multiple
            potential_products = [p for p in self.config["products"] if p["price"] > 300]
            if not potential_products:
                potential_products = self.config["products"]
            purchased_products = random.sample(
                potential_products, 
                min(random.randint(1, 3), len(potential_products))
            )
            quantities = [random.randint(1, 2) for _ in purchased_products]
        elif value_tier == "medium":
            # Medium value: mid-range products
            potential_products = [p for p in self.config["products"] if 100 <= p["price"] <= 500]
            if not potential_products:
                potential_products = self.config["products"]
            purchased_products = random.sample(
                potential_products, 
                min(random.randint(1, 2), len(potential_products))
            )
            quantities = [1 for _ in purchased_products]
        else:
            # Low value: cheaper products
            potential_products = [p for p in self.config["products"] if p["price"] < 200]
            if not potential_products:
                potential_products = self.config["products"]
            purchased_products = random.sample(
                potential_products, 
                min(1, len(potential_products))
            )
            quantities = [1 for _ in purchased_products]
        
        # Make purchase
        current_time = current_time + datetime.timedelta(minutes=random.randint(5, 15))
        
        # Generate a common order ID for all products in this purchase
        order_id = f"ORD{uuid.uuid4().hex[:8].upper()}"
        
        total_value = 0
        for i, product in enumerate(purchased_products):
            quantity = quantities[i]
            total_value += product["price"] * quantity
            
            self._add_event(customer, {
                "event_type": "PURCHASES",
                "target_type": "Product",
                "target_id": product["id"],
                "timestamp": self._timestamp_str(current_time),
                "properties": {
                    "quantity": quantity,
                    "price": product["price"],
                    "order_id": order_id,
                    "payment_method": random.choice(["credit_card", "paypal", "apple_pay", "google_pay"]),
                    "device": random.choice(customer["devices"])
                }
            })
        
        # Receipt email
        current_time = current_time + datetime.timedelta(minutes=random.randint(1, 5))
        receipt_email = next(email for email in self.config["emails"] if email["type"] == "transactional")
        self._add_event(customer, {
            "event_type": "RECEIVES",
            "target_type": "Email",
            "target_id": receipt_email["id"],
            "timestamp": self._timestamp_str(current_time),
            "properties": {
                "subject": receipt_email["subject"],
                "order_value": total_value
            }
        })
        
        # Open receipt email
        current_time = current_time + datetime.timedelta(hours=random.randint(1, 12))
        self._add_event(customer, {
            "event_type": "OPENS",
            "target_type": "Email",
            "target_id": receipt_email["id"],
            "timestamp": self._timestamp_str(current_time),
            "properties": {
                "device": random.choice(customer["devices"])
            }
        })
        
        # Update current time by a few hours or days
        current_time = current_time + datetime.timedelta(hours=random.randint(4, 36))
        
        return current_time
    
    def _add_retention_events(self, customer, start_time, days_spent, will_churn):
        """Add retention stage events to the customer journey."""
        current_time = start_time
        end_time = start_time + datetime.timedelta(days=days_spent)
        
        # Number of retention interactions
        num_interactions = random.randint(1, 5) if not will_churn else random.randint(0, 2)
        
        # Space out the interactions
        if num_interactions > 0 and days_spent > 0:
            avg_interval = min(30, days_spent / num_interactions)
        else:
            avg_interval = 30  # Default if no meaningful calculation possible
            
        for _ in range(num_interactions):
            interval_days = max(7, int(avg_interval * (0.7 + random.random() * 0.6)))  # 70-130% of average
            current_time = current_time + datetime.timedelta(days=interval_days)
            if current_time >= end_time:
                break
                
            # Random retention event type
            event_type = random.choice([
                "recommendation_email", "product_visit", "support_ticket", 
                "account_login"
            ])
            
            if event_type == "recommendation_email":
                # Receive recommendation email
                rec_email = next(email for email in self.config["emails"] if email["type"] == "recommendation")
                self._add_event(customer, {
                    "event_type": "RECEIVES",
                    "target_type": "Email",
                    "target_id": rec_email["id"],
                    "timestamp": self._timestamp_str(current_time),
                    "properties": {
                        "subject": rec_email["subject"]
                    }
                })
                
                # Maybe open the email
                if random.random() > 0.4:
                    current_time = current_time + datetime.timedelta(hours=random.randint(1, 48))
                    if current_time >= end_time:
                        break
                        
                    self._add_event(customer, {
                        "event_type": "OPENS",
                        "target_type": "Email",
                        "target_id": rec_email["id"],
                        "timestamp": self._timestamp_str(current_time),
                        "properties": {
                            "device": random.choice(customer["devices"])
                        }
                    })
                    
                    # Maybe click the email
                    if random.random() > 0.3:
                        current_time = current_time + datetime.timedelta(minutes=random.randint(1, 10))
                        if current_time >= end_time:
                            break
                            
                        self._add_event(customer, {
                            "event_type": "CLICKS_ON",
                            "target_type": "Email",
                            "target_id": rec_email["id"],
                            "timestamp": self._timestamp_str(current_time),
                            "properties": {
                                "device": random.choice(customer["devices"])
                            }
                        })
                        
                        # Return to site
                        current_time = current_time + datetime.timedelta(seconds=random.randint(5, 20))
                        if current_time >= end_time:
                            break
                            
                        self._add_event(customer, {
                            "event_type": "COMES_FROM",
                            "target_type": "Channel",
                            "target_id": "email",
                            "timestamp": self._timestamp_str(current_time),
                            "properties": {
                                "referrer": "recommendation_email"
                            }
                        })
                        
                        # Visit a product
                        product = random.choice(self.config["products"])
                        current_time = current_time + datetime.timedelta(seconds=random.randint(10, 60))
                        if current_time >= end_time:
                            break
                            
                        self._add_event(customer, {
                            "event_type": "VIEWS",
                            "target_type": "Product",
                            "target_id": product["id"],
                            "timestamp": self._timestamp_str(current_time),
                            "properties": {
                                "duration": random.randint(30, 300),
                                "device": random.choice(customer["devices"])
                            }
                        })
                        
            elif event_type == "product_visit":
                # Directly visit a product page
                self._add_event(customer, {
                    "event_type": "COMES_FROM",
                    "target_type": "Channel",
                    "target_id": random.choice(self.config["channels"]),
                    "timestamp": self._timestamp_str(current_time),
                    "properties": {
                        "referrer": random.choice(["direct", "search", "social"])
                    }
                })
                
                product = random.choice(self.config["products"])
                current_time = current_time + datetime.timedelta(seconds=random.randint(5, 20))
                if current_time >= end_time:
                    break
                    
                self._add_event(customer, {
                    "event_type": "VIEWS",
                    "target_type": "Product",
                    "target_id": product["id"],
                    "timestamp": self._timestamp_str(current_time),
                    "properties": {
                        "duration": random.randint(30, 300),
                        "device": random.choice(customer["devices"])
                    }
                })
                
            elif event_type == "support_ticket":
                # Visit support page
                support_page = next(page for page in self.config["pages"] if page["name"] == "Customer Support")
                self._add_event(customer, {
                    "event_type": "VISITS",
                    "target_type": "Page",
                    "target_id": support_page["id"],
                    "timestamp": self._timestamp_str(current_time),
                    "properties": {
                        "duration": random.randint(60, 300),
                        "device": random.choice(customer["devices"])
                    }
                })
                
                # Create support ticket
                current_time = current_time + datetime.timedelta(minutes=random.randint(5, 15))
                if current_time >= end_time:
                    break
                    
                self._add_event(customer, {
                    "event_type": "CREATES",
                    "target_type": "Ticket",
                    "target_id": f"TICKET{uuid.uuid4().hex[:8].upper()}",
                    "timestamp": self._timestamp_str(current_time),
                    "properties": {
                        "category": random.choice([
                            "product_question", "order_status", "technical_issue",
                            "return_request", "general_inquiry"
                        ]),
                        "priority": random.choice(["low", "medium", "high"]),
                        "device": random.choice(customer["devices"])
                    }
                })
            
            elif event_type == "account_login":
                # Visit account page
                account_page = next(page for page in self.config["pages"] if page["name"] == "My Account")
                self._add_event(customer, {
                    "event_type": "VISITS",
                    "target_type": "Page",
                    "target_id": account_page["id"],
                    "timestamp": self._timestamp_str(current_time),
                    "properties": {
                        "duration": random.randint(60, 300),
                        "device": random.choice(customer["devices"])
                    }
                })
                
                # Login event
                current_time = current_time + datetime.timedelta(seconds=random.randint(10, 30))
                if current_time >= end_time:
                    break
                    
                self._add_event(customer, {
                    "event_type": "LOGS_IN",
                    "target_type": "Account",
                    "target_id": "ACCOUNT",
                    "timestamp": self._timestamp_str(current_time),
                    "properties": {
                        "device": random.choice(customer["devices"])
                    }
                })
        
        # Add churn event if applicable
        if will_churn:
            current_time = min(end_time, current_time + datetime.timedelta(days=random.randint(15, 45)))
            self._add_event(customer, {
                "event_type": "CHURNED_AT",
                "target_type": "FunnelStage",
                "target_id": "retention",
                "timestamp": self._timestamp_str(current_time),
                "properties": {
                    "reason": random.choice([
                        "found_competitor", "price_sensitivity", "no_longer_needed",
                        "bad_experience", "missing_features", "unknown"
                    ])
                }
            })
        
        return current_time
    
    def _add_advocacy_events(self, customer, start_time):
        """Add advocacy stage events to the customer journey."""
        current_time = start_time
        
        # Determine number of advocacy events
        num_advocacy = random.randint(0, 3)
        
        for _ in range(num_advocacy):
            # Space out advocacy events
            current_time = current_time + datetime.timedelta(days=random.randint(7, 30))
            if current_time > self.end_date:
                break
                
            # Choose advocacy event type
            event_type = random.choice(["review", "referral", "social_share"])
            
            if event_type == "review":
                # Get a purchased product from journey events
                purchased_products = [
                    e["target_id"] for e in customer["journey_events"] 
                    if e["event_type"] == "PURCHASES" and e["target_type"] == "Product"
                ]
                
                if purchased_products:
                    product_id = random.choice(purchased_products)
                    
                    # Receive review request email
                    feedback_email = next(email for email in self.config["emails"] if email["type"] == "feedback")
                    self._add_event(customer, {
                        "event_type": "RECEIVES",
                        "target_type": "Email",
                        "target_id": feedback_email["id"],
                        "timestamp": self._timestamp_str(current_time),
                        "properties": {
                            "subject": feedback_email["subject"],
                            "product_id": product_id
                        }
                    })
                    
                    # Open email
                    current_time = current_time + datetime.timedelta(hours=random.randint(1, 48))
                    if current_time > self.end_date:
                        break
                        
                    self._add_event(customer, {
                        "event_type": "OPENS",
                        "target_type": "Email",
                        "target_id": feedback_email["id"],
                        "timestamp": self._timestamp_str(current_time),
                        "properties": {
                            "device": random.choice(customer["devices"])
                        }
                    })
                    
                    # Click email
                    current_time = current_time + datetime.timedelta(minutes=random.randint(1, 10))
                    if current_time > self.end_date:
                        break
                        
                    self._add_event(customer, {
                        "event_type": "CLICKS_ON",
                        "target_type": "Email",
                        "target_id": feedback_email["id"],
                        "timestamp": self._timestamp_str(current_time),
                        "properties": {
                            "device": random.choice(customer["devices"])
                        }
                    })
                    
                    # Write review
                    current_time = current_time + datetime.timedelta(minutes=random.randint(5, 30))
                    if current_time > self.end_date:
                        break
                        
                    self._add_event(customer, {
                        "event_type": "WRITES",
                        "target_type": "Review",
                        "target_id": f"REVIEW{uuid.uuid4().hex[:8].upper()}",
                        "timestamp": self._timestamp_str(current_time),
                        "properties": {
                            "product_id": product_id,
                            "rating": random.randint(3, 5),  # Advocates generally leave positive reviews
                            "length": random.choice(["short", "medium", "detailed"]),
                            "device": random.choice(customer["devices"])
                        }
                    })
                    
            elif event_type == "referral":
                # Generate a referral
                if current_time > self.end_date:
                    break
                    
                self._add_event(customer, {
                    "event_type": "REFERS",
                    "target_type": "Customer",
                    "target_id": f"REF{uuid.uuid4().hex[:8].upper()}",  # Placeholder for a new customer
                    "timestamp": self._timestamp_str(current_time),
                    "properties": {
                        "method": random.choice(["email", "social", "link_share", "direct"]),
                        "incentive": random.choice([True, False])
                    }
                })
                
            elif event_type == "social_share":
                # Share on social media
                if current_time > self.end_date:
                    break
                    
                self._add_event(customer, {
                    "event_type": "SHARES",
                    "target_type": "Product",
                    "target_id": random.choice([p["id"] for p in self.config["products"]]),
                    "timestamp": self._timestamp_str(current_time),
                    "properties": {
                        "platform": random.choice(["facebook", "twitter", "instagram", "tiktok", "linkedin"]),
                        "share_type": random.choice(["product", "purchase", "review", "referral"]),
                        "device": random.choice(customer["devices"])
                    }
                })
                
        return current_time
    
    def _add_event(self, customer, event):
        """Add a journey event to the customer's journey."""
        customer["journey_events"].append(event)
    
    def _generate_first_name(self):
        """Generate a random first name."""
        first_names = [
            "James", "John", "Robert", "Michael", "William", "David", "Richard", "Joseph", "Thomas", "Charles",
            "Mary", "Patricia", "Jennifer", "Linda", "Elizabeth", "Barbara", "Susan", "Jessica", "Sarah", "Karen",
            "Christopher", "Daniel", "Matthew", "Anthony", "Mark", "Donald", "Steven", "Paul", "Andrew", "Joshua",
            "Michelle", "Amanda", "Kimberly", "Melissa", "Stephanie", "Nicole", "Angela", "Deborah", "Rachel", "Laura",
            "Olivia", "Emma", "Noah", "Liam", "Ava", "Sophia", "Isabella", "Mia", "Charlotte", "Amelia",
            "Miguel", "Maria", "Jose", "Sofia", "Luis", "Elena", "Alejandro", "Isabella", "Diego", "Julia",
            "Wei", "Li", "Hui", "Yan", "Ming", "Lin", "Yang", "Jie", "Yi", "Yong",
            "Aiden", "Harper", "Mason", "Evelyn", "Elijah", "Abigail", "Logan", "Emily", "Lucas", "Madison"
        ]
        return random.choice(first_names)
    
    def _generate_last_name(self):
        """Generate a random last name."""
        last_names = [
            "Smith", "Johnson", "Williams", "Jones", "Brown", "Davis", "Miller", "Wilson", "Moore", "Taylor",
            "Anderson", "Thomas", "Jackson", "White", "Harris", "Martin", "Thompson", "Garcia", "Martinez", "Robinson",
            "Clark", "Rodriguez", "Lewis", "Lee", "Walker", "Hall", "Allen", "Young", "Hernandez", "King",
            "Wright", "Lopez", "Hill", "Scott", "Green", "Adams", "Baker", "Gonzalez", "Nelson", "Carter",
            "Mitchell", "Perez", "Roberts", "Turner", "Phillips", "Campbell", "Parker", "Evans", "Edwards", "Collins",
            "Chen", "Wang", "Li", "Zhang", "Liu", "Singh", "Kumar", "Kim", "Nguyen", "Patel",
            "Muller", "Schmidt", "Schneider", "Fischer", "Weber", "Meyer", "Schulz", "Wagner", "Becker", "Hoffmann"
        ]
        return random.choice(last_names)
    
    def _timestamp_str(self, dt):
        """Convert datetime to ISO format string."""
        return dt.isoformat()
    
    def save_data(self):
        """Save all generated data to files with scenario information."""
        # Save customers data
        with open(os.path.join(self.output_dir, "customers.json"), "w") as f:
            json.dump(self.customers, f, indent=2)
            
        # Save all entities data
        with open(os.path.join(self.output_dir, "entities.json"), "w") as f:
            json.dump(self.all_entities, f, indent=2)
            
        # Save neo4j import format
        self._save_neo4j_format()
            
        # Save configuration information
        with open(os.path.join(self.output_dir, "scenario_info.json"), "w") as f:
            scenario_info = {
                "scenario_name": self.config.get("scenario_name", "diverse_metrics"),
                "generated_at": datetime.datetime.now().isoformat(),
                "date_range": {
                    "start": self.start_date.isoformat(),
                    "end": self.end_date.isoformat()
                },
                "customer_count": len(self.customers),
                "event_count": sum(len(c["journey_events"]) for c in self.customers),
                "configuration": {
                    k: v for k, v in self.config.items() 
                    if k not in ["products", "advertisements", "emails", "pages", "locations", "customer_profiles"]
                }
            }
            json.dump(scenario_info, f, indent=2)
            
        print(f"Data saved to {self.output_dir}")
        print(f"Scenario: {self.config.get('scenario_name', 'diverse_metrics')}")
        
    def _save_neo4j_format(self):
        """Save data in a format suitable for Neo4j import."""
        neo4j_data = {
            "nodes": [],
            "relationships": []
        }
        
        # Add all customers as nodes
        for customer in self.customers:
            neo4j_data["nodes"].append({
                "id": customer["customer_id"],
                "labels": ["Customer"],
                "properties": {
                    "customer_id": customer["customer_id"],
                    "first_name": customer["profile"]["first_name"],
                    "last_name": customer["profile"]["last_name"],
                    "email": customer["profile"]["email"],
                    "phone": customer["profile"]["phone"],
                    "age": customer["profile"]["age"],
                    "gender": customer["profile"]["gender"],
                    "registration_date": customer["profile"]["registration_date"]
                }
            })
            
            # Location relationship
            neo4j_data["relationships"].append({
                "id": f"{customer['customer_id']}_LIVES_IN_{random.randint(10000, 99999)}",
                "type": "LIVES_IN",
                "startNode": customer["customer_id"],
                "endNode": f"LOC_{customer['profile']['location']['city']}",
                "properties": {}
            })
            
            # Segment relationship
            neo4j_data["relationships"].append({
                "id": f"{customer['customer_id']}_BELONGS_TO_{random.randint(10000, 99999)}",
                "type": "BELONGS_TO",
                "startNode": customer["customer_id"],
                "endNode": f"SEG_{customer['profile']['segment']}",
                "properties": {}
            })
            
            # Persona relationship
            for persona in customer["profile"]["personas"]:
                neo4j_data["relationships"].append({
                    "id": f"{customer['customer_id']}_HAS_PERSONA_{random.randint(10000, 99999)}",
                    "type": "HAS_PERSONA",
                    "startNode": customer["customer_id"],
                    "endNode": f"PERS_{persona.replace(' ', '_').lower()}",
                    "properties": {}
                })
                
            # Device relationships
            for device in customer["devices"]:
                neo4j_data["relationships"].append({
                    "id": f"{customer['customer_id']}_USES_{random.randint(10000, 99999)}",
                    "type": "USES",
                    "startNode": customer["customer_id"],
                    "endNode": f"DEV_{device}",
                    "properties": {}
                })
                
            # All journey event relationships
            for event in customer["journey_events"]:
                rel_id = f"{customer['customer_id']}_{event['event_type']}_{random.randint(10000, 99999)}"
                neo4j_data["relationships"].append({
                    "id": rel_id,
                    "type": event["event_type"],
                    "startNode": customer["customer_id"],
                    "endNode": f"{event['target_type']}_{event['target_id']}",
                    "properties": {
                        "timestamp": event["timestamp"],
                        **event.get("properties", {})
                    }
                })
                
        # Add all reference entities as nodes
        
        # Products
        for product in self.config["products"]:
            neo4j_data["nodes"].append({
                "id": f"Product_{product['id']}",
                "labels": ["Product"],
                "properties": product
            })
            
        # Advertisements
        for ad in self.config["advertisements"]:
            neo4j_data["nodes"].append({
                "id": f"Advertisement_{ad['id']}",
                "labels": ["Advertisement"],
                "properties": ad
            })
            
        # Emails
        for email in self.config["emails"]:
            neo4j_data["nodes"].append({
                "id": f"Email_{email['id']}",
                "labels": ["Email"],
                "properties": email
            })
            
        # Pages
        for page in self.config["pages"]:
            neo4j_data["nodes"].append({
                "id": f"Page_{page['id']}",
                "labels": ["Page"],
                "properties": page
            })
            
        # Locations
        for location in self.config["locations"]:
            loc_id = f"LOC_{location['city']}"
            neo4j_data["nodes"].append({
                "id": loc_id,
                "labels": ["Location"],
                "properties": location
            })
            
        # Channels
        for channel in self.config["channels"]:
            neo4j_data["nodes"].append({
                "id": f"Channel_{channel}",
                "labels": ["Channel"],
                "properties": {
                    "id": channel,
                    "name": channel.replace("_", " ").title()
                }
            })
            
        # Devices
        for device in self.config["devices"]:
            neo4j_data["nodes"].append({
                "id": f"DEV_{device}",
                "labels": ["Device"],
                "properties": {
                    "id": device,
                    "name": device.replace("_", " ").title()
                }
            })
            
        # Funnel Stages
        for stage in self.config["funnel_stages"]:
            neo4j_data["nodes"].append({
                "id": f"FunnelStage_{stage}",
                "labels": ["FunnelStage"],
                "properties": {
                    "id": stage,
                    "name": stage.title()
                }
            })
            
        # Segments (Persona Groups)
        for segment in self.config["persona_groups"].keys():
            neo4j_data["nodes"].append({
                "id": f"SEG_{segment}",
                "labels": ["Segment"],
                "properties": {
                    "id": segment,
                    "name": segment
                }
            })
            
        # Personas
        for group, data in self.config["persona_groups"].items():
            for persona in data["profiles"]:
                persona_id = persona.replace(" ", "_").lower()
                neo4j_data["nodes"].append({
                    "id": f"PERS_{persona_id}",
                    "labels": ["Persona"],
                    "properties": {
                        "id": persona_id,
                        "name": persona,
                        "group": group
                    }
                })
                
        # Save Neo4j import format
        with open(os.path.join(self.output_dir, "neo4j_import.json"), "w") as f:
            json.dump(neo4j_data, f, indent=2)


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Generate diverse customer data for demos with varied metrics")
    
    parser.add_argument("--output", type=str, default="/home/cabdru/marketing/demo/demo_data",
                      help="Output directory for generated data")
    
    parser.add_argument("--start-date", type=str, 
                      default=(datetime.datetime.now() - datetime.timedelta(days=365)).strftime('%Y-%m-%d'),
                      help="Start date for data generation (YYYY-MM-DD)")
    
    parser.add_argument("--end-date", type=str,
                      default=datetime.datetime.now().strftime('%Y-%m-%d'),
                      help="End date for data generation (YYYY-MM-DD)")
    
    parser.add_argument("--high-churn-count", type=int, default=5,
                      help="Number of high churn risk customers (default: 5)")
    
    parser.add_argument("--save-config", action="store_true",
                      help="Save the configuration as a reusable scenario")
    
    return parser.parse_args()


def get_scenario_config(args=None):
    """Get configuration for the diverse metrics scenario."""
    
    # Get default configuration which includes all entity data and customer profiles
    default_config = DiverseDemoDataGenerator()._get_default_config()
    
    if args:
        # Override defaults with command line arguments
        config = default_config.copy()
        config["output_dir"] = args.output
        config["start_date"] = f"{args.start_date}T00:00:00"
        config["end_date"] = f"{args.end_date}T23:59:59"
        
        # Adjust the number of high churn customers if specified
        if args.high_churn_count != 5:
            high_churn_profiles = [p for p in config["customer_profiles"] if p["id"].startswith("high_churn_")]
            if args.high_churn_count < len(high_churn_profiles):
                # Reduce high churn customers
                config["customer_profiles"] = [p for p in config["customer_profiles"] 
                                             if not p["id"].startswith("high_churn_") or 
                                             int(p["id"].split("_")[-1]) <= args.high_churn_count]
            elif args.high_churn_count > len(high_churn_profiles):
                # Need to add more high churn customers
                for i in range(len(high_churn_profiles) + 1, args.high_churn_count + 1):
                    # Clone one of the existing profiles and modify it
                    new_profile = high_churn_profiles[0].copy()
                    new_profile["id"] = f"high_churn_{i}"
                    # Randomize some parameters to make it different
                    new_profile["days_since_last_purchase"] = random.randint(80, 150)
                    config["customer_profiles"].append(new_profile)
        
        return config
    
    return default_config


def main():
    """Generate and save diverse demo data based on command line arguments."""
    args = parse_arguments()
    
    # Get scenario configuration
    config = get_scenario_config(args)
    
    print(f"Generating diverse customer data with varied metrics")
    print(f"Date range: {args.start_date} to {args.end_date}")
    print(f"High churn customer count: {args.high_churn_count}")
    print(f"Output directory: {args.output}")
    print("\nGenerating data...")
    
    # Create generator with scenario config
    generator = DiverseDemoDataGenerator(config)
    
    # Generate data
    generator.generate_customer_base()
    generator.generate_journey_events()
    generator.save_data()
    
    print("\nDiverse demo data generation complete.")
    print(f"Generated {len(generator.customers)} customers with varied metrics")
    print(f"Total events: {sum(len(c['journey_events']) for c in generator.customers)}")
    
    # Save scenario config if requested
    if args.save_config:
        output_config = os.path.join(args.output, "scenario_diverse_metrics.json")
        with open(output_config, "w") as f:
            json.dump(config, f, indent=2)
        print(f"Saved scenario configuration to {output_config}")
    
    # Print command for loading data
    print("\nTo load this data into Neo4j, run:")
    print(f"python load_demo_data.py --data-dir {args.output}")


if __name__ == "__main__":
    main()