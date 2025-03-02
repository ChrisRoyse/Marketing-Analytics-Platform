#!/usr/bin/env python
"""
Generate synthetic customer journey data for the Marketing Ontology Platform Demo.

This script creates 15 detailed customer profiles across 5 persona groups,
along with their complete journey events from awareness to advocacy/churn.
"""

import json
import random
import uuid
import datetime
import os
import sys
from pathlib import Path

# Add parent directory to path to access shared modules
sys.path.append(str(Path(__file__).resolve().parent.parent))

# Constants for data generation
PERSONA_GROUPS = {
    "Tech Enthusiast": {
        "profiles": ["Early Adopter", "Feature Hunter", "Upgrade Cycler"],
        "avg_conversion_days": 5,
        "price_sensitivity": "low",
        "research_depth": "high",
        "churn_rate": 0.1,
    },
    "Budget Shopper": {
        "profiles": ["Deal Seeker", "Comparison Shopper", "Bargain Hunter"],
        "avg_conversion_days": 20,
        "price_sensitivity": "high",
        "research_depth": "medium",
        "churn_rate": 0.3,
    },
    "Gift Buyer": {
        "profiles": ["Seasonal Gifter", "Special Occasion Buyer", "Corporate Gifter"],
        "avg_conversion_days": 3,
        "price_sensitivity": "medium",
        "research_depth": "low",
        "churn_rate": 0.5,
    },
    "Professional": {
        "profiles": ["Business User", "Remote Worker", "Executive Buyer"],
        "avg_conversion_days": 10,
        "price_sensitivity": "low",
        "research_depth": "high",
        "churn_rate": 0.2,
    },
    "Student": {
        "profiles": ["Budget Student", "Tech Student", "International Student"],
        "avg_conversion_days": 15,
        "price_sensitivity": "high",
        "research_depth": "medium",
        "churn_rate": 0.4,
    },
}

CHANNELS = ["facebook", "instagram", "google_search", "email", "direct", "referral"]
FUNNEL_STAGES = ["awareness", "consideration", "intent", "conversion", "retention", "advocacy"]
DEVICES = ["desktop_chrome", "desktop_firefox", "desktop_safari", "mobile_chrome", "mobile_safari", "tablet_chrome"]
LOCATIONS = [
    {"city": "New York", "state": "NY", "country": "USA", "postal_code": "10001"},
    {"city": "San Francisco", "state": "CA", "country": "USA", "postal_code": "94105"},
    {"city": "Austin", "state": "TX", "country": "USA", "postal_code": "78701"},
    {"city": "Seattle", "state": "WA", "country": "USA", "postal_code": "98101"},
    {"city": "Chicago", "state": "IL", "country": "USA", "postal_code": "60601"},
    {"city": "London", "city_code": "LD", "country": "UK", "postal_code": "EC1A 1BB"},
    {"city": "Toronto", "province": "ON", "country": "Canada", "postal_code": "M5V 2A8"},
    {"city": "Sydney", "state": "NSW", "country": "Australia", "postal_code": "2000"},
]

PRODUCTS = [
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
]

ADVERTISEMENTS = [
    {"id": "AD001", "name": "Summer Sale", "channel": "facebook", "campaign": "seasonal_promotions"},
    {"id": "AD002", "name": "Back to School", "channel": "instagram", "campaign": "seasonal_promotions"},
    {"id": "AD003", "name": "Tech Deals", "channel": "google_search", "campaign": "always_on"},
    {"id": "AD004", "name": "New Arrivals", "channel": "email", "campaign": "product_launches"},
    {"id": "AD005", "name": "Holiday Gifts", "channel": "facebook", "campaign": "seasonal_promotions"},
    {"id": "AD006", "name": "Upgrade Your Setup", "channel": "google_search", "campaign": "always_on"},
]

EMAILS = [
    {"id": "EM001", "subject": "Welcome to TechGear", "type": "welcome"},
    {"id": "EM002", "subject": "Your cart is waiting", "type": "abandoned_cart"},
    {"id": "EM003", "subject": "Exclusive deals just for you", "type": "promotional"},
    {"id": "EM004", "subject": "Thank you for your purchase", "type": "transactional"},
    {"id": "EM005", "subject": "New products you might like", "type": "recommendation"},
    {"id": "EM006", "subject": "Rate your recent purchase", "type": "feedback"},
]

PAGES = [
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
]


class DemoDataGenerator:
    """Generates synthetic customer journey data for the Marketing Ontology Platform Demo."""
    
    def __init__(self, output_dir="/home/cabdru/marketing/demo/demo_data"):
        """Initialize the data generator with an output directory."""
        self.output_dir = output_dir
        self.customers = []
        self.all_entities = {
            "customers": [],
            "products": PRODUCTS,
            "advertisements": ADVERTISEMENTS,
            "emails": EMAILS,
            "pages": PAGES,
            "locations": LOCATIONS,
            "devices": [{"id": d} for d in DEVICES],
            "channels": [{"id": c} for c in CHANNELS],
            "funnel_stages": [{"id": s} for s in FUNNEL_STAGES],
            "persona_groups": [{"id": p} for p in PERSONA_GROUPS.keys()],
            "personas": [],
        }
        
        # Create persona entities
        for group, data in PERSONA_GROUPS.items():
            for profile in data["profiles"]:
                self.all_entities["personas"].append({
                    "id": profile.replace(" ", "_").lower(),
                    "name": profile,
                    "group": group
                })
                
        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)
    
    def generate_customer_base(self):
        """Generate 15 diverse customer profiles - 3 from each persona group."""
        customer_id = 1
        
        for persona_group, data in PERSONA_GROUPS.items():
            for i, persona in enumerate(data["profiles"]):
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
                        "location": random.choice(LOCATIONS),
                        "registration_date": self._generate_date(
                            start_date="2023-01-01T00:00:00",
                            end_date="2023-12-31T23:59:59"
                        ),
                        "segment": persona_group,
                        "personas": [persona]
                    },
                    "devices": random.sample(DEVICES, random.randint(1, 3)),
                    "journey_events": []
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
        """Generate realistic journey events for each customer."""
        for customer in self.customers:
            persona_group = customer["profile"]["segment"]
            persona_data = PERSONA_GROUPS[persona_group]
            
            # Determine journey parameters based on persona
            conversion_days = max(1, int(random.gauss(
                persona_data["avg_conversion_days"], 
                persona_data["avg_conversion_days"] / 3
            )))
            
            # Starting point for journey (registration date or earlier for ad exposure)
            registration_date = datetime.datetime.fromisoformat(customer["profile"]["registration_date"])
            journey_start = registration_date - datetime.timedelta(days=random.randint(1, 30))
            
            # Will this customer convert?
            will_convert = random.random() > 0.2  # 80% chance of conversion
            
            # Will this customer churn after conversion?
            will_churn = random.random() < persona_data["churn_rate"]
            
            # Current timestamp for event generation
            current_time = journey_start
            
            # Channel first encountered
            primary_channel = random.choice(CHANNELS)
            
            # Add awareness events
            current_time = self._add_awareness_events(customer, current_time, primary_channel)
            
            # Add consideration events
            current_time = self._add_consideration_events(
                customer, 
                current_time, 
                days_spent=max(1, int(conversion_days * 0.4)),
                research_depth=persona_data["research_depth"]
            )
            
            # Add intent events
            current_time = self._add_intent_events(
                customer, 
                current_time,
                days_spent=max(1, int(conversion_days * 0.3)),
                will_convert=will_convert,
                price_sensitivity=persona_data["price_sensitivity"]
            )
            
            # Add conversion events if applicable
            if will_convert:
                current_time = self._add_conversion_events(customer, current_time)
                
                # Add retention events
                current_time = self._add_retention_events(
                    customer, 
                    current_time,
                    days_spent=random.randint(30, 180),
                    will_churn=will_churn
                )
                
                # Add advocacy events if not churning
                if not will_churn:
                    self._add_advocacy_events(customer, current_time)
            
            # Sort events by timestamp
            customer["journey_events"].sort(key=lambda x: x["timestamp"])
            
            print(f"Generated {len(customer['journey_events'])} events for {customer['customer_id']}")
        
        return self.customers
    
    def _add_awareness_events(self, customer, start_time, primary_channel):
        """Add awareness stage events to the customer journey."""
        current_time = start_time
        
        # Ad view event
        if primary_channel in ["facebook", "instagram", "google_search"]:
            ad = next((ad for ad in ADVERTISEMENTS if ad["channel"] == primary_channel), random.choice(ADVERTISEMENTS))
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
        home_page = next(page for page in PAGES if page["name"] == "Home")
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
        interested_products = random.sample(PRODUCTS, min(num_products, len(PRODUCTS)))
        
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
                (p for p in PAGES if p["name"] == f"{category} Category"), 
                next(p for p in PAGES if "Category" in p["name"])
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
                    
                blog_page = next(page for page in PAGES if page["name"] == "Blog")
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
        cart_products = random.sample(PRODUCTS, random.randint(1, 3))
        
        # Account creation (if they'll convert or randomly)
        if will_convert or random.random() > 0.7:
            account_page = next(page for page in PAGES if page["name"] == "My Account")
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
            
            # Add to cart if price sensitivity allows or randomly
            if (price_sensitivity == "low" or 
                (price_sensitivity == "medium" and product["price"] < 300) or
                (price_sensitivity == "high" and product["price"] < 150) or
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
                    
                cart_page = next(page for page in PAGES if page["name"] == "Shopping Cart")
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
                            
                        abandon_email = next(email for email in EMAILS if email["type"] == "abandoned_cart")
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
    
    def _add_conversion_events(self, customer, start_time):
        """Add conversion stage events to the customer journey."""
        current_time = start_time
        
        # Visit checkout page
        checkout_page = next(page for page in PAGES if page["name"] == "Checkout")
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
        
        # Purchase 1-3 products
        purchased_products = random.sample(PRODUCTS, random.randint(1, 3))
        current_time = current_time + datetime.timedelta(minutes=random.randint(5, 15))
        
        total_value = 0
        for product in purchased_products:
            quantity = random.randint(1, 2)
            total_value += product["price"] * quantity
            
            self._add_event(customer, {
                "event_type": "PURCHASES",
                "target_type": "Product",
                "target_id": product["id"],
                "timestamp": self._timestamp_str(current_time),
                "properties": {
                    "quantity": quantity,
                    "price": product["price"],
                    "order_id": f"ORD{uuid.uuid4().hex[:8].upper()}",
                    "payment_method": random.choice(["credit_card", "paypal", "apple_pay", "google_pay"]),
                    "device": random.choice(customer["devices"])
                }
            })
        
        # Receipt email
        current_time = current_time + datetime.timedelta(minutes=random.randint(1, 5))
        receipt_email = next(email for email in EMAILS if email["type"] == "transactional")
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
        
        # Update current time by 1-3 days
        current_time = current_time + datetime.timedelta(days=random.randint(1, 3))
        
        return current_time
    
    def _add_retention_events(self, customer, start_time, days_spent, will_churn):
        """Add retention stage events to the customer journey."""
        current_time = start_time
        end_time = start_time + datetime.timedelta(days=days_spent)
        
        # Number of retention interactions
        num_interactions = random.randint(1, 5) if not will_churn else random.randint(0, 2)
        
        for _ in range(num_interactions):
            current_time = current_time + datetime.timedelta(days=random.randint(7, 30))
            if current_time >= end_time:
                break
                
            # Random retention event type
            event_type = random.choice([
                "recommendation_email", "product_visit", "support_ticket", 
                "repeat_purchase", "account_login"
            ])
            
            if event_type == "recommendation_email":
                # Receive recommendation email
                rec_email = next(email for email in EMAILS if email["type"] == "recommendation")
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
                        product = random.choice(PRODUCTS)
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
                    "target_id": random.choice(CHANNELS),
                    "timestamp": self._timestamp_str(current_time),
                    "properties": {
                        "referrer": random.choice(["direct", "search", "social"])
                    }
                })
                
                product = random.choice(PRODUCTS)
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
                support_page = next(page for page in PAGES if page["name"] == "Customer Support")
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
                
            elif event_type == "repeat_purchase":
                # Add to cart
                product = random.choice(PRODUCTS)
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
                current_time = current_time + datetime.timedelta(minutes=random.randint(1, 10))
                if current_time >= end_time:
                    break
                    
                cart_page = next(page for page in PAGES if page["name"] == "Shopping Cart")
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
                
                # Visit checkout
                current_time = current_time + datetime.timedelta(minutes=random.randint(2, 10))
                if current_time >= end_time:
                    break
                    
                checkout_page = next(page for page in PAGES if page["name"] == "Checkout")
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
                
                # Purchase
                current_time = current_time + datetime.timedelta(minutes=random.randint(5, 15))
                if current_time >= end_time:
                    break
                    
                self._add_event(customer, {
                    "event_type": "PURCHASES",
                    "target_type": "Product",
                    "target_id": product["id"],
                    "timestamp": self._timestamp_str(current_time),
                    "properties": {
                        "quantity": random.randint(1, 2),
                        "price": product["price"],
                        "order_id": f"ORD{uuid.uuid4().hex[:8].upper()}",
                        "payment_method": random.choice(["credit_card", "paypal", "apple_pay", "google_pay"]),
                        "device": random.choice(customer["devices"])
                    }
                })
                
            elif event_type == "account_login":
                # Visit account page
                account_page = next(page for page in PAGES if page["name"] == "My Account")
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
            current_time = current_time + datetime.timedelta(days=random.randint(30, 60))
            if current_time < end_time:
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
                    feedback_email = next(email for email in EMAILS if email["type"] == "feedback")
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
                self._add_event(customer, {
                    "event_type": "SHARES",
                    "target_type": "Product",
                    "target_id": random.choice([p["id"] for p in PRODUCTS]),
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
    
    def _generate_date(self, start_date, end_date):
        """Generate a random date between start_date and end_date."""
        start = datetime.datetime.fromisoformat(start_date)
        end = datetime.datetime.fromisoformat(end_date)
        delta = end - start
        random_days = random.randint(0, delta.days)
        random_seconds = random.randint(0, 86399)  # 24 hours in seconds
        random_date = start + datetime.timedelta(days=random_days, seconds=random_seconds)
        return random_date.isoformat()
    
    def _timestamp_str(self, dt):
        """Convert datetime to ISO format string."""
        return dt.isoformat()
    
    def save_data(self):
        """Save all generated data to files."""
        # Save customers data
        with open(os.path.join(self.output_dir, "customers.json"), "w") as f:
            json.dump(self.customers, f, indent=2)
            
        # Save all entities data
        with open(os.path.join(self.output_dir, "entities.json"), "w") as f:
            json.dump(self.all_entities, f, indent=2)
            
        # Save neo4j import format
        self._save_neo4j_format()
            
        print(f"Data saved to {self.output_dir}")
        
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
        for product in PRODUCTS:
            neo4j_data["nodes"].append({
                "id": f"Product_{product['id']}",
                "labels": ["Product"],
                "properties": product
            })
            
        # Advertisements
        for ad in ADVERTISEMENTS:
            neo4j_data["nodes"].append({
                "id": f"Advertisement_{ad['id']}",
                "labels": ["Advertisement"],
                "properties": ad
            })
            
        # Emails
        for email in EMAILS:
            neo4j_data["nodes"].append({
                "id": f"Email_{email['id']}",
                "labels": ["Email"],
                "properties": email
            })
            
        # Pages
        for page in PAGES:
            neo4j_data["nodes"].append({
                "id": f"Page_{page['id']}",
                "labels": ["Page"],
                "properties": page
            })
            
        # Locations
        for location in LOCATIONS:
            loc_id = f"LOC_{location['city']}"
            neo4j_data["nodes"].append({
                "id": loc_id,
                "labels": ["Location"],
                "properties": location
            })
            
        # Channels
        for channel in CHANNELS:
            neo4j_data["nodes"].append({
                "id": f"Channel_{channel}",
                "labels": ["Channel"],
                "properties": {
                    "id": channel,
                    "name": channel.replace("_", " ").title()
                }
            })
            
        # Devices
        for device in DEVICES:
            neo4j_data["nodes"].append({
                "id": f"DEV_{device}",
                "labels": ["Device"],
                "properties": {
                    "id": device,
                    "name": device.replace("_", " ").title()
                }
            })
            
        # Funnel Stages
        for stage in FUNNEL_STAGES:
            neo4j_data["nodes"].append({
                "id": f"FunnelStage_{stage}",
                "labels": ["FunnelStage"],
                "properties": {
                    "id": stage,
                    "name": stage.title()
                }
            })
            
        # Segments (Persona Groups)
        for segment in PERSONA_GROUPS.keys():
            neo4j_data["nodes"].append({
                "id": f"SEG_{segment}",
                "labels": ["Segment"],
                "properties": {
                    "id": segment,
                    "name": segment
                }
            })
            
        # Personas
        for group, data in PERSONA_GROUPS.items():
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


def main():
    """Generate and save demo data."""
    generator = DemoDataGenerator()
    generator.generate_customer_base()
    generator.generate_journey_events()
    generator.save_data()
    print("Demo data generation complete.")


if __name__ == "__main__":
    main()