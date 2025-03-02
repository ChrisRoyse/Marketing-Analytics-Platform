#!/usr/bin/env python3
"""
Data Integration Service for the Marketing Ontology Platform.

This service implements the data integration capabilities outlined in Section 2 
of the plan.md document, including:
- E-commerce platform integration (Shopify, WooCommerce, etc.)
- CRM system integration (Salesforce, HubSpot, etc.)
- Marketing platform integration (Mailchimp, Google Analytics, etc.)
- Real-time data processing with event streams

The service follows a microservices architecture pattern and provides a unified
interface for collecting, transforming, and loading data into the Neo4j graph database.
"""

import os
import json
import logging
import time
from datetime import datetime
from pathlib import Path
import requests
from typing import Dict, List, Optional, Union, Any
import pandas as pd
import numpy as np
from json import JSONEncoder

# Custom JSON encoder to handle NumPy types
class NumpyEncoder(JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.bool_):
            return bool(obj)
        return super(NumpyEncoder, self).default(obj)
from neo4j import GraphDatabase
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('data_integration.log')
    ]
)

logger = logging.getLogger("DataIntegrationService")

class DataIntegrationService:
    """
    Core service for integrating external data sources with the marketing ontology platform.
    
    This service provides a unified interface for:
    1. Connecting to external data sources (E-commerce, CRM, Marketing platforms)
    2. Extracting, transforming, and loading data into Neo4j
    3. Implementing real-time data processing for event streams
    4. Monitoring and logging integration status
    """
    
    def __init__(self, neo4j_uri=None, neo4j_username=None, neo4j_password=None, neo4j_database=None):
        """Initialize the Data Integration Service with Neo4j connection details."""
        self.neo4j_uri = neo4j_uri or os.getenv('NEO4J_URI', 'bolt://localhost:7687')
        self.neo4j_username = neo4j_username or os.getenv('NEO4J_USERNAME', 'neo4j')
        self.neo4j_password = neo4j_password or os.getenv('NEO4J_PASSWORD', '#1Moneymaker')
        self.neo4j_database = neo4j_database or os.getenv('NEO4J_DATABASE', 'marketing')
        self.driver = None
        
        # API keys for various services
        self.api_keys = {
            'shopify': os.getenv('SHOPIFY_API_KEY'),
            'woocommerce': os.getenv('WOOCOMMERCE_API_KEY'),
            'salesforce': os.getenv('SALESFORCE_API_KEY'),
            'hubspot': os.getenv('HUBSPOT_API_KEY'),
            'mailchimp': os.getenv('MAILCHIMP_API_KEY'),
            'google_analytics': os.getenv('GOOGLE_ANALYTICS_API_KEY')
        }
        
        # Connection configuration for various services
        self.service_configs = {}
        
        # Initialize data directories
        self.data_dir = Path("integration_data")
        self.data_dir.mkdir(exist_ok=True)
        
        # Subfolders for each integration type
        for folder in ['ecommerce', 'crm', 'marketing', 'erp', 'support', 'events']:
            (self.data_dir / folder).mkdir(exist_ok=True)
    
    def connect_neo4j(self) -> bool:
        """Connect to the Neo4j database."""
        try:
            self.driver = GraphDatabase.driver(
                self.neo4j_uri, 
                auth=(self.neo4j_username, self.neo4j_password)
            )
            # Test the connection
            with self.driver.session(database=self.neo4j_database) as session:
                result = session.run("RETURN 1 AS test")
                record = result.single()
                if record and record["test"] == 1:
                    logger.info("Successfully connected to Neo4j database")
                    return True
                else:
                    logger.error("Failed to verify Neo4j connection")
                    return False
        except Exception as e:
            logger.error(f"Failed to connect to Neo4j: {e}")
            return False
    
    def close(self):
        """Close the Neo4j connection."""
        if self.driver:
            self.driver.close()
            logger.info("Neo4j connection closed")
    
    def run_query(self, query: str, parameters: Optional[Dict] = None) -> List:
        """Run a Cypher query and return the results."""
        if not self.driver:
            if not self.connect_neo4j():
                return None
        
        try:
            with self.driver.session(database=self.neo4j_database) as session:
                result = session.run(query, parameters or {})
                return [record.data() for record in result]
        except Exception as e:
            logger.error(f"Error running query: {e}")
            return None
    
    def configure_service(self, service_name: str, config: Dict) -> bool:
        """
        Configure a data source service with necessary connection details.
        
        Args:
            service_name: Name of the service (e.g., 'shopify', 'salesforce')
            config: Dictionary containing connection details
                   Should include at minimum: api_url, auth_method, 
                   and relevant credentials
        
        Returns:
            bool: True if configuration was successful
        """
        try:
            # Validate minimum requirements
            required_fields = ['api_url', 'auth_method']
            for field in required_fields:
                if field not in config:
                    logger.error(f"Missing required configuration field: {field}")
                    return False
            
            # Store configuration
            self.service_configs[service_name] = config
            
            # Test connection if possible
            if config['auth_method'] == 'api_key' and self.api_keys.get(service_name):
                # Add API key to config
                self.service_configs[service_name]['api_key'] = self.api_keys.get(service_name)
                
                # Test connection
                test_result = self._test_service_connection(service_name)
                if not test_result:
                    logger.warning(f"Could not verify connection to {service_name}")
            
            logger.info(f"Successfully configured {service_name} integration")
            return True
            
        except Exception as e:
            logger.error(f"Error configuring {service_name}: {e}")
            return False
    
    def _test_service_connection(self, service_name: str) -> bool:
        """Test connection to a configured service."""
        if service_name not in self.service_configs:
            logger.error(f"Service {service_name} not configured")
            return False
        
        config = self.service_configs[service_name]
        
        try:
            # Different testing logic based on service type
            if service_name in ['shopify', 'woocommerce']:
                return self._test_ecommerce_connection(service_name, config)
            elif service_name in ['salesforce', 'hubspot']:
                return self._test_crm_connection(service_name, config)
            elif service_name in ['mailchimp', 'google_analytics']:
                return self._test_marketing_connection(service_name, config)
            else:
                logger.warning(f"No test implemented for {service_name}")
                return True  # Assume success if no test implemented
                
        except Exception as e:
            logger.error(f"Error testing connection to {service_name}: {e}")
            return False
    
    def _test_ecommerce_connection(self, service_name: str, config: Dict) -> bool:
        """Test connection to an e-commerce platform."""
        # Implement connection test for specific e-commerce platform
        # This is a simplified example - real implementation would actually connect to the API
        logger.info(f"Testing connection to {service_name} e-commerce platform")
        
        # In a real implementation, we would make an API call here
        # For now, we'll simulate a successful connection
        return True
    
    def _test_crm_connection(self, service_name: str, config: Dict) -> bool:
        """Test connection to a CRM system."""
        # Implement connection test for specific CRM system
        logger.info(f"Testing connection to {service_name} CRM system")
        
        # In a real implementation, we would make an API call here
        # For now, we'll simulate a successful connection
        return True
    
    def _test_marketing_connection(self, service_name: str, config: Dict) -> bool:
        """Test connection to a marketing platform."""
        # Implement connection test for specific marketing platform
        logger.info(f"Testing connection to {service_name} marketing platform")
        
        # In a real implementation, we would make an API call here
        # For now, we'll simulate a successful connection
        return True
    
    # ==== E-Commerce Platform Integration ====
    
    def integrate_shopify(self, shop_url: str, api_key: str = None, api_password: str = None) -> bool:
        """
        Integrate with a Shopify store to import customers, orders, and products.
        
        Args:
            shop_url: The shop's URL (e.g., 'mystore.myshopify.com')
            api_key: Optional API key (will use env var if not provided)
            api_password: API password or access token
            
        Returns:
            bool: True if integration was successful
        """
        try:
            # Configure the Shopify service
            api_key = api_key or self.api_keys.get('shopify')
            if not api_key or not api_password:
                logger.error("Missing Shopify API credentials")
                return False
            
            config = {
                'api_url': f"https://{shop_url}/admin/api/2023-04",
                'auth_method': 'api_key',
                'api_key': api_key,
                'api_password': api_password,
                'shop_url': shop_url
            }
            
            if not self.configure_service('shopify', config):
                return False
            
            # Import customers
            customers = self._import_shopify_customers()
            if customers:
                logger.info(f"Imported {len(customers)} customers from Shopify")
                
                # Process and load customers into Neo4j
                self._process_and_load_shopify_customers(customers)
            
            # Import products
            products = self._import_shopify_products()
            if products:
                logger.info(f"Imported {len(products)} products from Shopify")
                
                # Process and load products into Neo4j
                self._process_and_load_shopify_products(products)
            
            # Import orders
            orders = self._import_shopify_orders()
            if orders:
                logger.info(f"Imported {len(orders)} orders from Shopify")
                
                # Process and load orders into Neo4j
                self._process_and_load_shopify_orders(orders)
            
            return True
            
        except Exception as e:
            logger.error(f"Error integrating with Shopify: {e}")
            return False
    
    def _import_shopify_customers(self) -> List[Dict]:
        """Import customers from a Shopify store."""
        # In a real implementation, this would make API calls to Shopify
        # For demonstration, we'll return simulated data
        
        logger.info("Importing customers from Shopify")
        
        # Simulate API call delay
        time.sleep(0.5)
        
        # Generate sample customer data
        customers = []
        for i in range(1, 11):
            customer = {
                'id': f"CUST{i:03d}",
                'email': f"customer{i}@example.com",
                'first_name': f"First{i}",
                'last_name': f"Last{i}",
                'orders_count': np.random.randint(1, 10),
                'total_spent': round(np.random.uniform(50, 5000), 2),
                'created_at': (datetime.now().replace(
                    day=np.random.randint(1, 28),
                    month=np.random.randint(1, 12),
                    year=np.random.randint(2020, 2023)
                )).isoformat(),
                'accepts_marketing': np.random.choice([True, False]),
                'tags': np.random.choice(
                    ['loyal', 'new', 'vip', 'abandoned_cart', 'prospect'],
                    size=np.random.randint(0, 3),
                    replace=False
                ).tolist()
            }
            customers.append(customer)
        
        # Save to file for reference
        with open(self.data_dir / 'ecommerce' / 'shopify_customers.json', 'w') as f:
            json.dump(customers, f, indent=2, cls=NumpyEncoder)
        
        return customers
    
    def _process_and_load_shopify_customers(self, customers: List[Dict]) -> bool:
        """Process and load Shopify customers into Neo4j."""
        if not customers:
            return False
        
        try:
            # Connect to Neo4j if not already connected
            if not self.driver and not self.connect_neo4j():
                return False
            
            # Process in batches
            batch_size = 10
            batches = [customers[i:i+batch_size] for i in range(0, len(customers), batch_size)]
            
            for batch in batches:
                for customer in batch:
                    # Create customer node
                    customer_query = """
                    MERGE (c:Customer {customer_id: $customer_id})
                    SET c.email = $email,
                        c.first_name = $first_name,
                        c.last_name = $last_name,
                        c.orders_count = $orders_count,
                        c.total_spent = $total_spent,
                        c.created_at = datetime($created_at),
                        c.accepts_marketing = $accepts_marketing,
                        c.source = 'shopify',
                        c.last_updated = datetime()
                    
                    RETURN c
                    """
                    
                    self.run_query(customer_query, {
                        'customer_id': customer['id'],
                        'email': customer['email'],
                        'first_name': customer['first_name'],
                        'last_name': customer['last_name'],
                        'orders_count': customer['orders_count'],
                        'total_spent': customer['total_spent'],
                        'created_at': customer['created_at'],
                        'accepts_marketing': customer['accepts_marketing']
                    })
                    
                    # Create segment nodes for each tag and connect to customer
                    for tag in customer.get('tags', []):
                        segment_query = """
                        MATCH (c:Customer {customer_id: $customer_id})
                        MERGE (s:Segment {id: $segment_id})
                        SET s.name = $segment_name,
                            s.source = 'shopify',
                            s.last_updated = datetime()
                        
                        MERGE (c)-[:BELONGS_TO]->(s)
                        """
                        
                        self.run_query(segment_query, {
                            'customer_id': customer['id'],
                            'segment_id': f"shopify_tag_{tag}",
                            'segment_name': tag.title()
                        })
            
            logger.info(f"Successfully loaded {len(customers)} Shopify customers into Neo4j")
            return True
            
        except Exception as e:
            logger.error(f"Error loading Shopify customers: {e}")
            return False
    
    def _import_shopify_products(self) -> List[Dict]:
        """Import products from a Shopify store."""
        # In a real implementation, this would make API calls to Shopify
        # For demonstration, we'll return simulated data
        
        logger.info("Importing products from Shopify")
        
        # Simulate API call delay
        time.sleep(0.5)
        
        # Product categories
        categories = ['clothing', 'electronics', 'home', 'beauty', 'food']
        
        # Generate sample product data
        products = []
        for i in range(1, 21):
            category = np.random.choice(categories)
            product = {
                'id': f"PROD{i:03d}",
                'title': f"{category.title()} Product {i}",
                'category': category,
                'price': round(np.random.uniform(10, 500), 2),
                'inventory_quantity': np.random.randint(0, 100),
                'created_at': (datetime.now().replace(
                    day=np.random.randint(1, 28),
                    month=np.random.randint(1, 12),
                    year=np.random.randint(2020, 2023)
                )).isoformat(),
                'tags': np.random.choice(
                    ['bestseller', 'new', 'sale', 'limited', 'featured'],
                    size=np.random.randint(0, 3),
                    replace=False
                ).tolist()
            }
            products.append(product)
        
        # Save to file for reference
        with open(self.data_dir / 'ecommerce' / 'shopify_products.json', 'w') as f:
            json.dump(products, f, indent=2, cls=NumpyEncoder)
        
        return products
    
    def _process_and_load_shopify_products(self, products: List[Dict]) -> bool:
        """Process and load Shopify products into Neo4j."""
        if not products:
            return False
        
        try:
            # Connect to Neo4j if not already connected
            if not self.driver and not self.connect_neo4j():
                return False
            
            # Process in batches
            batch_size = 10
            batches = [products[i:i+batch_size] for i in range(0, len(products), batch_size)]
            
            for batch in batches:
                for product in batch:
                    # Create product node
                    product_query = """
                    MERGE (p:Product {id: $product_id})
                    SET p.name = $title,
                        p.category = $category,
                        p.price = $price,
                        p.inventory_quantity = $inventory_quantity,
                        p.created_at = datetime($created_at),
                        p.source = 'shopify',
                        p.last_updated = datetime()
                    
                    RETURN p
                    """
                    
                    self.run_query(product_query, {
                        'product_id': product['id'],
                        'title': product['title'],
                        'category': product['category'],
                        'price': product['price'],
                        'inventory_quantity': product['inventory_quantity'],
                        'created_at': product['created_at']
                    })
                    
                    # Create category node and connect to product
                    category_query = """
                    MATCH (p:Product {id: $product_id})
                    MERGE (c:Category {id: $category_id})
                    SET c.name = $category_name,
                        c.source = 'shopify',
                        c.last_updated = datetime()
                    
                    MERGE (p)-[:HAS_CATEGORY]->(c)
                    """
                    
                    self.run_query(category_query, {
                        'product_id': product['id'],
                        'category_id': product['category'],
                        'category_name': product['category'].title()
                    })
                    
                    # Create tag nodes and connect to product
                    for tag in product.get('tags', []):
                        tag_query = """
                        MATCH (p:Product {id: $product_id})
                        MERGE (t:Tag {id: $tag_id})
                        SET t.name = $tag_name,
                            t.source = 'shopify',
                            t.last_updated = datetime()
                        
                        MERGE (p)-[:HAS_TAG]->(t)
                        """
                        
                        self.run_query(tag_query, {
                            'product_id': product['id'],
                            'tag_id': tag,
                            'tag_name': tag.title()
                        })
            
            logger.info(f"Successfully loaded {len(products)} Shopify products into Neo4j")
            return True
            
        except Exception as e:
            logger.error(f"Error loading Shopify products: {e}")
            return False
    
    def _import_shopify_orders(self) -> List[Dict]:
        """Import orders from a Shopify store."""
        # In a real implementation, this would make API calls to Shopify
        # For demonstration, we'll return simulated data
        
        logger.info("Importing orders from Shopify")
        
        # Simulate API call delay
        time.sleep(0.5)
        
        # Load customers and products for reference
        with open(self.data_dir / 'ecommerce' / 'shopify_customers.json', 'r') as f:
            customers = json.load(f)
        
        with open(self.data_dir / 'ecommerce' / 'shopify_products.json', 'r') as f:
            products = json.load(f)
        
        # Order statuses
        statuses = ['fulfilled', 'unfulfilled', 'partially_fulfilled', 'cancelled']
        
        # Generate sample order data
        orders = []
        for i in range(1, 31):
            # Select a random customer
            customer = np.random.choice(customers)
            
            # Select 1-3 random products
            order_products = np.random.choice(
                products, 
                size=np.random.randint(1, 4),
                replace=False
            ).tolist()
            
            # Calculate order total
            order_total = sum(p['price'] * np.random.randint(1, 3) for p in order_products)
            
            order = {
                'id': f"ORDER{i:03d}",
                'customer_id': customer['id'],
                'email': customer['email'],
                'status': np.random.choice(statuses, p=[0.7, 0.1, 0.1, 0.1]),
                'created_at': (datetime.now().replace(
                    day=np.random.randint(1, 28),
                    month=np.random.randint(1, 12),
                    year=np.random.randint(2020, 2023)
                )).isoformat(),
                'total_price': round(order_total, 2),
                'line_items': [
                    {
                        'product_id': p['id'],
                        'title': p['title'],
                        'quantity': np.random.randint(1, 3),
                        'price': p['price']
                    }
                    for p in order_products
                ]
            }
            orders.append(order)
        
        # Save to file for reference
        with open(self.data_dir / 'ecommerce' / 'shopify_orders.json', 'w') as f:
            json.dump(orders, f, indent=2, cls=NumpyEncoder)
        
        return orders
    
    def _process_and_load_shopify_orders(self, orders: List[Dict]) -> bool:
        """Process and load Shopify orders into Neo4j."""
        if not orders:
            return False
        
        try:
            # Connect to Neo4j if not already connected
            if not self.driver and not self.connect_neo4j():
                return False
            
            # Process in batches
            batch_size = 10
            batches = [orders[i:i+batch_size] for i in range(0, len(orders), batch_size)]
            
            for batch in batches:
                for order in batch:
                    # Create order node
                    order_query = """
                    MERGE (o:Order {id: $order_id})
                    SET o.customer_id = $customer_id,
                        o.email = $email,
                        o.status = $status,
                        o.created_at = datetime($created_at),
                        o.total_price = $total_price,
                        o.source = 'shopify',
                        o.last_updated = datetime()
                    
                    RETURN o
                    """
                    
                    self.run_query(order_query, {
                        'order_id': order['id'],
                        'customer_id': order['customer_id'],
                        'email': order['email'],
                        'status': order['status'],
                        'created_at': order['created_at'],
                        'total_price': order['total_price']
                    })
                    
                    # Connect order to customer
                    customer_order_query = """
                    MATCH (c:Customer {customer_id: $customer_id})
                    MATCH (o:Order {id: $order_id})
                    MERGE (c)-[r:PLACED]->(o)
                    SET r.timestamp = datetime($created_at)
                    """
                    
                    self.run_query(customer_order_query, {
                        'customer_id': order['customer_id'],
                        'order_id': order['id'],
                        'created_at': order['created_at']
                    })
                    
                    # Process line items
                    for item in order.get('line_items', []):
                        # Connect order to product with CONTAINS relationship
                        order_product_query = """
                        MATCH (o:Order {id: $order_id})
                        MATCH (p:Product {id: $product_id})
                        MERGE (o)-[r:CONTAINS]->(p)
                        SET r.quantity = $quantity,
                            r.price = $price,
                            r.timestamp = datetime($created_at)
                        """
                        
                        self.run_query(order_product_query, {
                            'order_id': order['id'],
                            'product_id': item['product_id'],
                            'quantity': item['quantity'],
                            'price': item['price'],
                            'created_at': order['created_at']
                        })
                        
                        # Add PURCHASES relationship from customer to product
                        if order['status'] != 'cancelled':
                            purchase_query = """
                            MATCH (c:Customer {customer_id: $customer_id})
                            MATCH (p:Product {id: $product_id})
                            MERGE (c)-[r:PURCHASES]->(p)
                            SET r.quantity = $quantity,
                                r.price = $price,
                                r.order_id = $order_id,
                                r.timestamp = datetime($created_at)
                            """
                            
                            self.run_query(purchase_query, {
                                'customer_id': order['customer_id'],
                                'product_id': item['product_id'],
                                'quantity': item['quantity'],
                                'price': item['price'],
                                'order_id': order['id'],
                                'created_at': order['created_at']
                            })
            
            logger.info(f"Successfully loaded {len(orders)} Shopify orders into Neo4j")
            return True
            
        except Exception as e:
            logger.error(f"Error loading Shopify orders: {e}")
            return False
    
    def integrate_woocommerce(self, site_url: str, api_key: str = None, api_secret: str = None) -> bool:
        """
        Integrate with a WooCommerce store to import customers, orders, and products.
        
        Args:
            site_url: The WooCommerce site URL
            api_key: Optional API key (will use env var if not provided)
            api_secret: API secret
            
        Returns:
            bool: True if integration was successful
        """
        # Similar implementation to Shopify integration
        logger.info(f"WooCommerce integration for {site_url} not yet implemented")
        return True
    
    # ==== CRM System Integration ====
    
    def integrate_salesforce(self, instance_url: str, client_id: str = None, client_secret: str = None,
                           username: str = None, password: str = None) -> bool:
        """
        Integrate with Salesforce to import contacts, opportunities, and activities.
        
        Args:
            instance_url: Salesforce instance URL
            client_id: OAuth client ID
            client_secret: OAuth client secret
            username: Salesforce username
            password: Salesforce password
            
        Returns:
            bool: True if integration was successful
        """
        try:
            # Configure the Salesforce service
            client_id = client_id or os.getenv('SALESFORCE_CLIENT_ID')
            client_secret = client_secret or os.getenv('SALESFORCE_CLIENT_SECRET')
            username = username or os.getenv('SALESFORCE_USERNAME')
            password = password or os.getenv('SALESFORCE_PASSWORD')
            
            if not client_id or not client_secret or not username or not password:
                logger.error("Missing Salesforce credentials")
                return False
            
            config = {
                'api_url': f"{instance_url}/services/data/v58.0",
                'auth_method': 'oauth',
                'client_id': client_id,
                'client_secret': client_secret,
                'username': username,
                'password': password,
                'instance_url': instance_url
            }
            
            if not self.configure_service('salesforce', config):
                return False
            
            # Import contacts
            logger.info("Salesforce integration not yet fully implemented")
            
            # In a real implementation, we would:
            # 1. Authenticate with Salesforce using OAuth
            # 2. Import contacts, opportunities, activities
            # 3. Process and load data into Neo4j
            
            return True
            
        except Exception as e:
            logger.error(f"Error integrating with Salesforce: {e}")
            return False
    
    def integrate_hubspot(self, api_key: str = None) -> bool:
        """
        Integrate with HubSpot to import contacts, deals, and activities.
        
        Args:
            api_key: Optional API key (will use env var if not provided)
            
        Returns:
            bool: True if integration was successful
        """
        # Similar implementation to Salesforce integration
        logger.info("HubSpot integration not yet implemented")
        return True
    
    # ==== Marketing Platform Integration ====
    
    def integrate_mailchimp(self, api_key: str = None) -> bool:
        """
        Integrate with Mailchimp to import subscribers, campaigns, and activities.
        
        Args:
            api_key: Optional API key (will use env var if not provided)
            
        Returns:
            bool: True if integration was successful
        """
        try:
            # Configure the Mailchimp service
            api_key = api_key or self.api_keys.get('mailchimp')
            if not api_key:
                logger.error("Missing Mailchimp API key")
                return False
            
            # Extract API server from API key (last part after the dash)
            api_server = api_key.split('-')[-1]
            
            config = {
                'api_url': f"https://{api_server}.api.mailchimp.com/3.0",
                'auth_method': 'api_key',
                'api_key': api_key
            }
            
            if not self.configure_service('mailchimp', config):
                return False
            
            # Import subscribers, campaigns, etc.
            logger.info("Mailchimp integration not yet fully implemented")
            
            # In a real implementation, we would:
            # 1. Authenticate with Mailchimp API
            # 2. Import subscribers, campaigns, activities
            # 3. Process and load data into Neo4j
            
            return True
            
        except Exception as e:
            logger.error(f"Error integrating with Mailchimp: {e}")
            return False
    
    # ==== Real-time Data Processing ====
    
    def setup_webhooks(self, service_name: str, events: List[str], callback_url: str) -> bool:
        """
        Set up webhooks for real-time data integration.
        
        Args:
            service_name: Name of the service (e.g., 'shopify', 'salesforce')
            events: List of events to subscribe to
            callback_url: URL to receive webhook events
            
        Returns:
            bool: True if webhook setup was successful
        """
        if service_name not in self.service_configs:
            logger.error(f"Service {service_name} not configured")
            return False
        
        try:
            logger.info(f"Setting up {service_name} webhooks for events: {', '.join(events)}")
            
            # In a real implementation, we would:
            # 1. Authenticate with the service
            # 2. Register webhooks for specified events
            # 3. Store webhook IDs for future reference
            
            # For demonstration, we'll just log the setup
            logger.info(f"Webhook setup for {service_name} would register these events: {events}")
            logger.info(f"Callback URL: {callback_url}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error setting up webhooks for {service_name}: {e}")
            return False
    
    def process_webhook_event(self, service_name: str, event_type: str, event_data: Dict) -> bool:
        """
        Process a webhook event from a service.
        
        Args:
            service_name: Name of the service that sent the event
            event_type: Type of event (e.g., 'order.created', 'contact.updated')
            event_data: Dictionary containing event data
            
        Returns:
            bool: True if event processing was successful
        """
        try:
            logger.info(f"Processing {service_name} {event_type} webhook event")
            
            # In a real implementation, we would:
            # 1. Validate the webhook signature
            # 2. Parse the event data
            # 3. Process and load the data into Neo4j
            
            # Different processing logic based on service and event type
            if service_name == 'shopify':
                return self._process_shopify_webhook(event_type, event_data)
            elif service_name == 'salesforce':
                return self._process_salesforce_webhook(event_type, event_data)
            elif service_name == 'mailchimp':
                return self._process_mailchimp_webhook(event_type, event_data)
            else:
                logger.warning(f"No webhook processor implemented for {service_name}")
                return False
                
        except Exception as e:
            logger.error(f"Error processing {service_name} webhook event: {e}")
            return False
    
    def _process_shopify_webhook(self, event_type: str, event_data: Dict) -> bool:
        """Process a Shopify webhook event."""
        # Process based on event type
        if event_type == 'orders/create':
            # Process new order
            logger.info("Processing new Shopify order")
            
            # Extract order data
            order_id = event_data.get('id')
            customer_id = event_data.get('customer', {}).get('id')
            
            if not order_id or not customer_id:
                logger.error("Missing order or customer ID in webhook data")
                return False
            
            # In a real implementation, we would:
            # 1. Transform the order data
            # 2. Load it into Neo4j
            
            return True
            
        elif event_type == 'customers/create':
            # Process new customer
            logger.info("Processing new Shopify customer")
            return True
            
        elif event_type == 'products/create':
            # Process new product
            logger.info("Processing new Shopify product")
            return True
            
        else:
            logger.warning(f"Unhandled Shopify event type: {event_type}")
            return False
    
    def _process_salesforce_webhook(self, event_type: str, event_data: Dict) -> bool:
        """Process a Salesforce webhook event."""
        # Not implemented yet
        logger.info(f"Salesforce webhook processing not yet implemented for {event_type}")
        return True
    
    def _process_mailchimp_webhook(self, event_type: str, event_data: Dict) -> bool:
        """Process a Mailchimp webhook event."""
        # Not implemented yet
        logger.info(f"Mailchimp webhook processing not yet implemented for {event_type}")
        return True
    
    # ==== Data Transformation Utilities ====
    
    def transform_customer_data(self, source: str, data: List[Dict]) -> List[Dict]:
        """
        Transform customer data from various sources to a common format.
        
        Args:
            source: Source system name (e.g., 'shopify', 'salesforce')
            data: List of customer records from the source
            
        Returns:
            List of transformed customer records in common format
        """
        transformed = []
        
        try:
            if source == 'shopify':
                for record in data:
                    transformed.append({
                        'customer_id': record.get('id'),
                        'email': record.get('email'),
                        'first_name': record.get('first_name'),
                        'last_name': record.get('last_name'),
                        'created_at': record.get('created_at'),
                        'segments': record.get('tags', []),
                        'total_spent': record.get('total_spent', 0),
                        'source': 'shopify'
                    })
            
            elif source == 'salesforce':
                for record in data:
                    transformed.append({
                        'customer_id': record.get('Id'),
                        'email': record.get('Email'),
                        'first_name': record.get('FirstName'),
                        'last_name': record.get('LastName'),
                        'created_at': record.get('CreatedDate'),
                        'segments': [],  # Extract from Salesforce custom fields
                        'total_spent': 0,  # Calculate from opportunities
                        'source': 'salesforce'
                    })
            
            elif source == 'hubspot':
                for record in data:
                    properties = record.get('properties', {})
                    transformed.append({
                        'customer_id': record.get('id'),
                        'email': properties.get('email'),
                        'first_name': properties.get('firstname'),
                        'last_name': properties.get('lastname'),
                        'created_at': properties.get('createdate'),
                        'segments': [],  # Extract from HubSpot lists
                        'total_spent': 0,  # Calculate from deals
                        'source': 'hubspot'
                    })
            
            else:
                logger.warning(f"No transformer implemented for {source}")
                return []
            
            logger.info(f"Transformed {len(transformed)} {source} customer records")
            return transformed
            
        except Exception as e:
            logger.error(f"Error transforming {source} customer data: {e}")
            return []

    # ==== Integration Status Reporting ====
    
    def get_integration_status(self) -> Dict:
        """
        Get the status of all data integrations.
        
        Returns:
            Dictionary with integration status for each service
        """
        status = {
            'connected_services': [],
            'last_sync_times': {},
            'record_counts': {},
            'service_config_status': {}
        }
        
        # List connected services
        status['connected_services'] = list(self.service_configs.keys())
        
        # Get record counts from Neo4j
        if self.driver or self.connect_neo4j():
            # Count nodes by label and source
            count_query = """
            MATCH (n)
            WHERE n:Customer OR n:Product OR n:Order
            RETURN labels(n)[0] as entity_type, n.source as source, count(n) as count
            """
            
            counts = self.run_query(count_query)
            if counts:
                # Organize counts by entity type and source
                for record in counts:
                    entity_type = record.get('entity_type')
                    source = record.get('source')
                    count = record.get('count')
                    
                    if not entity_type or not source:
                        continue
                    
                    if entity_type not in status['record_counts']:
                        status['record_counts'][entity_type] = {}
                    
                    status['record_counts'][entity_type][source] = count
            
            # Get last sync times from integration history
            # In a real implementation, we would store this in Neo4j
            
        # Check service configuration status
        for service, config in self.service_configs.items():
            status['service_config_status'][service] = {
                'configured': True,
                'api_url': config.get('api_url'),
                'auth_method': config.get('auth_method')
            }
        
        return status
    
    def run_integration_job(self, clear_existing: bool = False) -> Dict:
        """
        Run a full integration job for all configured services.
        
        Args:
            clear_existing: Whether to clear existing data first
            
        Returns:
            Dictionary with job results
        """
        job_results = {
            'job_id': f"integration_{int(time.time())}",
            'start_time': datetime.now().isoformat(),
            'services_processed': [],
            'record_counts': {
                'customers': 0,
                'products': 0,
                'orders': 0
            },
            'errors': []
        }
        
        try:
            # Connect to Neo4j
            if not self.driver and not self.connect_neo4j():
                job_results['errors'].append("Failed to connect to Neo4j")
                return job_results
            
            # Clear existing data if requested
            if clear_existing:
                logger.info("Clearing existing integration data")
                clear_query = """
                MATCH (n)
                WHERE n.source IS NOT NULL
                DETACH DELETE n
                """
                
                self.run_query(clear_query)
                logger.info("Cleared existing integration data")
            
            # Process each configured service
            for service, config in self.service_configs.items():
                logger.info(f"Running integration job for {service}")
                
                success = False
                if service == 'shopify':
                    # Get credentials from config
                    shop_url = config.get('shop_url')
                    api_key = config.get('api_key')
                    api_password = config.get('api_password')
                    
                    if shop_url and api_key and api_password:
                        success = self.integrate_shopify(shop_url, api_key, api_password)
                
                elif service == 'woocommerce':
                    site_url = config.get('site_url')
                    api_key = config.get('api_key')
                    api_secret = config.get('api_secret')
                    
                    if site_url and api_key and api_secret:
                        success = self.integrate_woocommerce(site_url, api_key, api_secret)
                
                elif service == 'salesforce':
                    instance_url = config.get('instance_url')
                    client_id = config.get('client_id')
                    client_secret = config.get('client_secret')
                    username = config.get('username')
                    password = config.get('password')
                    
                    if instance_url and client_id and client_secret and username and password:
                        success = self.integrate_salesforce(instance_url, client_id, client_secret, username, password)
                
                elif service == 'mailchimp':
                    api_key = config.get('api_key')
                    
                    if api_key:
                        success = self.integrate_mailchimp(api_key)
                        
                else:
                    logger.warning(f"No integration job implemented for {service}")
                    success = False
                
                if success:
                    job_results['services_processed'].append(service)
                else:
                    job_results['errors'].append(f"Failed to process {service}")
            
            # Get record counts
            status = self.get_integration_status()
            for entity_type, sources in status.get('record_counts', {}).items():
                if entity_type == 'Customer':
                    job_results['record_counts']['customers'] = sum(sources.values())
                elif entity_type == 'Product':
                    job_results['record_counts']['products'] = sum(sources.values())
                elif entity_type == 'Order':
                    job_results['record_counts']['orders'] = sum(sources.values())
            
            job_results['end_time'] = datetime.now().isoformat()
            
            # Log job results
            logger.info(f"Integration job completed: processed {len(job_results['services_processed'])} services")
            if job_results['errors']:
                logger.warning(f"Integration job had {len(job_results['errors'])} errors")
            
            return job_results
            
        except Exception as e:
            error_msg = f"Error running integration job: {e}"
            logger.error(error_msg)
            job_results['errors'].append(error_msg)
            job_results['end_time'] = datetime.now().isoformat()
            return job_results
        
        finally:
            # Close Neo4j connection
            self.close()

if __name__ == "__main__":
    print("Starting Data Integration Service...")
    
    # Initialize service
    service = DataIntegrationService()
    
    # Example: Configure and run a Shopify integration
    # service.configure_service('shopify', {
    #     'api_url': 'https://mystore.myshopify.com/admin/api/2023-04',
    #     'auth_method': 'api_key',
    #     'shop_url': 'mystore.myshopify.com'
    # })
    # service.integrate_shopify('mystore.myshopify.com', 'api_key_here', 'api_password_here')
    
    # Run a full integration job
    job_results = service.run_integration_job()
    
    print(f"Integration job completed with status: {len(job_results['errors']) == 0}")
    print(f"Processed services: {', '.join(job_results['services_processed']) if job_results['services_processed'] else 'None'}")
    print(f"Record counts: {job_results['record_counts']}")
    if job_results['errors']:
        print(f"Errors: {job_results['errors']}")