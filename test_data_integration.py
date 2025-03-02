#!/usr/bin/env python3
"""
Test script for the Data Integration Service.
This script tests the core functionality of the data integration service.
"""

import os
import sys
import logging
import json
from pathlib import Path
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('test_integration.log')
    ]
)

logger = logging.getLogger("TestDataIntegration")

# Load environment variables
load_dotenv()

# Import the Data Integration Service
try:
    from data_integration_service import DataIntegrationService
except ImportError:
    logger.error("Failed to import DataIntegrationService. Make sure data_integration_service.py is in the current directory.")
    sys.exit(1)

def test_neo4j_connection():
    """Test connection to Neo4j database."""
    logger.info("Testing Neo4j connection...")
    service = DataIntegrationService()
    
    result = service.connect_neo4j()
    if result:
        logger.info("Successfully connected to Neo4j")
    else:
        logger.error("Failed to connect to Neo4j")
    
    service.close()
    return result

def test_service_configuration():
    """Test service configuration."""
    logger.info("Testing service configuration...")
    service = DataIntegrationService()
    
    # Configure a test service
    config = {
        'api_url': 'https://api.example.com',
        'auth_method': 'api_key'
    }
    
    result = service.configure_service('test_service', config)
    if result:
        logger.info("Successfully configured test service")
    else:
        logger.error("Failed to configure test service")
    
    service.close()
    return result

def test_shopify_integration():
    """Test Shopify data integration."""
    logger.info("Testing Shopify integration...")
    service = DataIntegrationService()
    
    # This is a simulation, not a real Shopify integration
    # In a real test, we would use actual credentials
    
    # Mock data will be generated internally
    service.connect_neo4j()
    
    result = service.integrate_shopify('test-store.myshopify.com', 'test_api_key', 'test_api_password')
    
    if result:
        logger.info("Successfully simulated Shopify integration")
        
        # Get integration status
        status = service.get_integration_status()
        logger.info(f"Integration status: {json.dumps(status, indent=2)}")
    else:
        logger.error("Failed to simulate Shopify integration")
    
    service.close()
    return result

def test_data_transformation():
    """Test data transformation utilities."""
    logger.info("Testing data transformation...")
    service = DataIntegrationService()
    
    # Create test source data
    shopify_customers = [
        {
            'id': 'CUST001',
            'email': 'test1@example.com',
            'first_name': 'Test',
            'last_name': 'User',
            'created_at': '2023-01-01T12:00:00',
            'tags': ['vip', 'new'],
            'total_spent': 1000
        },
        {
            'id': 'CUST002',
            'email': 'test2@example.com',
            'first_name': 'Another',
            'last_name': 'User',
            'created_at': '2023-02-01T12:00:00',
            'tags': ['loyal'],
            'total_spent': 2000
        }
    ]
    
    # Transform the data
    transformed = service.transform_customer_data('shopify', shopify_customers)
    
    if transformed and len(transformed) == 2:
        logger.info("Successfully transformed customer data")
        logger.info(f"Sample transformed record: {json.dumps(transformed[0], indent=2)}")
        result = True
    else:
        logger.error("Failed to transform customer data")
        result = False
    
    service.close()
    return result

def test_integration_job():
    """Test running a full integration job."""
    logger.info("Testing full integration job...")
    service = DataIntegrationService()
    
    # Connect to Neo4j
    if not service.connect_neo4j():
        logger.error("Failed to connect to Neo4j, skipping integration job test")
        return False
    
    # Configure test services
    service.configure_service('shopify', {
        'api_url': 'https://test-store.myshopify.com/admin/api/2023-04',
        'auth_method': 'api_key',
        'api_key': 'test_api_key',
        'api_password': 'test_api_password',
        'shop_url': 'test-store.myshopify.com'
    })
    
    # Run integration job
    job_results = service.run_integration_job()
    
    if job_results and 'shopify' in job_results.get('services_processed', []):
        logger.info("Successfully ran integration job")
        logger.info(f"Job results: {json.dumps(job_results, indent=2)}")
        result = True
    else:
        logger.error("Failed to run integration job")
        logger.error(f"Job errors: {job_results.get('errors', [])}")
        result = False
    
    service.close()
    return result

def run_all_tests():
    """Run all tests."""
    logger.info("Running all tests...")
    
    # Create results dictionary
    results = {
        'neo4j_connection': False,
        'service_configuration': False,
        'shopify_integration': False,
        'data_transformation': False,
        'integration_job': False
    }
    
    # Run tests
    results['neo4j_connection'] = test_neo4j_connection()
    results['service_configuration'] = test_service_configuration()
    results['shopify_integration'] = test_shopify_integration()
    results['data_transformation'] = test_data_transformation()
    results['integration_job'] = test_integration_job()
    
    # Print results
    logger.info("Test results:")
    for test, result in results.items():
        logger.info(f"{test}: {'Success' if result else 'Failure'}")
    
    # Check if all tests passed
    all_passed = all(results.values())
    logger.info(f"Overall result: {'Success' if all_passed else 'Failure'}")
    
    return all_passed

if __name__ == "__main__":
    print("Running Data Integration Service tests...")
    success = run_all_tests()
    print(f"Tests completed with {'success' if success else 'failures'}")
    sys.exit(0 if success else 1)