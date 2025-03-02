#!/usr/bin/env python3
"""
Predictive Models module for Phase 4 of the marketing ontology project.
This module implements machine learning models to forecast customer behavior
and leverages Neo4j's Graph Data Science (GDS) library.
"""

import os
import json
import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier, AdaBoostClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    mean_absolute_error, mean_squared_error, r2_score,
    roc_auc_score, confusion_matrix, classification_report
)
from neo4j import GraphDatabase

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('predictive_models.log')
    ]
)

class PredictiveModels:
    """Class for building and managing predictive models using Neo4j GDS."""
    
    def __init__(self, uri=None, username=None, password=None, database=None):
        """Initialize the PredictiveModels class with Neo4j connection details."""
        self.uri = uri or os.getenv('NEO4J_URI', 'bolt://localhost:7687')
        self.username = username or os.getenv('NEO4J_USERNAME', 'neo4j')
        self.password = password or os.getenv('NEO4J_PASSWORD', 'neo4j')
        self.database = database or os.getenv('NEO4J_DATABASE', 'neo4j')
        self.driver = None
        self.models = {}
        self.feature_importances = {}
        self.model_metrics = {}
        self.ensemble_models = {}
        
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
                    # Check if GDS is available
                    gds_available = self._check_gds_available()
                    if gds_available:
                        logging.info("Successfully connected to Neo4j database with GDS")
                    else:
                        logging.warning("Connected to Neo4j but Graph Data Science library is not available")
                    return gds_available
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
    
    def _check_gds_available(self):
        """Check if Neo4j Graph Data Science library is available."""
        try:
            query = """
            CALL gds.list()
            YIELD name
            RETURN count(name) > 0 AS gds_available
            """
            result = self.run_query(query)
            if result and len(result) > 0:
                return result[0].get("gds_available", False)
            return False
        except Exception:
            return False
    
    def _extract_customer_features(self):
        """Extract features for all customers from Neo4j for model training."""
        # First, run a query to get customer data with journey statistics
        query = """
        MATCH (c:Customer)
        
        // Count interactions by type
        OPTIONAL MATCH (c)-[r:VIEWS]->()
        WITH c, count(r) as view_count
        
        OPTIONAL MATCH (c)-[r:CLICKS_ON]->()
        WITH c, view_count, count(r) as click_count
        
        OPTIONAL MATCH (c)-[r:VISITS]->()
        WITH c, view_count, click_count, count(r) as visit_count
        
        OPTIONAL MATCH (c)-[r:ADDS_TO_CART]->()
        WITH c, view_count, click_count, visit_count, count(r) as cart_add_count
        
        OPTIONAL MATCH (c)-[r:ABANDONS]->()
        WITH c, view_count, click_count, visit_count, cart_add_count, count(r) as cart_abandon_count
        
        OPTIONAL MATCH (c)-[r:PURCHASES]->()
        WITH c, view_count, click_count, visit_count, cart_add_count, cart_abandon_count, count(r) as purchase_count
        
        // Get most recent activity timestamp
        OPTIONAL MATCH (c)-[r]->()
        WHERE r.timestamp IS NOT NULL
        WITH c, view_count, click_count, visit_count, cart_add_count, cart_abandon_count, purchase_count,
             max(r.timestamp) as last_activity
        
        // Get all segments
        OPTIONAL MATCH (c)-[:BELONGS_TO]->(s:Segment)
        WITH c, view_count, click_count, visit_count, cart_add_count, cart_abandon_count, purchase_count,
             last_activity, collect(s.id) as segments
        
        // Get all devices
        OPTIONAL MATCH (c)-[:USES]->(d:Device)
        WITH c, view_count, click_count, visit_count, cart_add_count, cart_abandon_count, purchase_count,
             last_activity, segments, collect(d.id) as devices
        
        // Get churn status
        OPTIONAL MATCH (c)-[churn:CHURNED_AT]->()
        
        RETURN 
            c.customer_id as customer_id,
            c.lifetime_value as lifetime_value,
            view_count,
            click_count,
            visit_count,
            cart_add_count,
            cart_abandon_count,
            purchase_count,
            CASE WHEN last_activity IS NULL THEN 0 
                 ELSE duration.inDays(datetime(last_activity), datetime()).days
            END as days_since_activity,
            size(segments) as segment_count,
            size(devices) as device_count,
            CASE WHEN churn IS NOT NULL THEN 1 ELSE 0 END as is_churned
        """
        
        # Execute the query
        customer_data = self.run_query(query)
        
        if not customer_data:
            logging.error("No customer data found for feature extraction")
            return None
        
        # Convert to DataFrame
        df = pd.DataFrame(customer_data)
        
        # Fill missing values
        df = df.fillna({
            'lifetime_value': 0,
            'view_count': 0,
            'click_count': 0,
            'visit_count': 0,
            'cart_add_count': 0,
            'cart_abandon_count': 0,
            'purchase_count': 0,
            'days_since_activity': 999,  # Very high number for inactive customers
            'segment_count': 0,
            'device_count': 0
        })
        
        # Add derived features
        df['cart_abandonment_rate'] = df.apply(
            lambda x: x['cart_abandon_count'] / x['cart_add_count'] 
            if x['cart_add_count'] > 0 else 0,
            axis=1
        )
        df['conversion_rate'] = df.apply(
            lambda x: x['purchase_count'] / x['visit_count'] 
            if x['visit_count'] > 0 else 0,
            axis=1
        )
        df['click_through_rate'] = df.apply(
            lambda x: x['click_count'] / x['view_count'] 
            if x['view_count'] > 0 else 0,
            axis=1
        )
        
        logging.info(f"Extracted features for {len(df)} customers")
        return df
    
    def _extract_customer_lifetime_features(self):
        """Extract features specifically for customer lifetime value prediction."""
        # Run a query to get historical purchase data
        query = """
        MATCH (c:Customer)-[p:PURCHASES]->(product:Product)
        WHERE p.timestamp IS NOT NULL
        RETURN 
            c.customer_id as customer_id,
            c.lifetime_value as current_lifetime_value,
            count(p) as purchase_count,
            min(p.timestamp) as first_purchase,
            max(p.timestamp) as last_purchase,
            avg(p.amount) as avg_purchase_amount,
            sum(p.amount) as total_spend,
            collect(p.timestamp) as purchase_dates
        ORDER BY c.customer_id
        """
        
        # Execute the query
        purchase_data = self.run_query(query)
        
        if not purchase_data:
            logging.error("No purchase data found for CLV prediction")
            return None
        
        # Convert to DataFrame
        df = pd.DataFrame(purchase_data)
        
        # Calculate additional features
        df['recency'] = df.apply(
            lambda x: (datetime.now() - datetime.fromisoformat(x['last_purchase'])).days
            if x['last_purchase'] else 999,
            axis=1
        )
        
        df['frequency'] = df['purchase_count']
        
        df['monetary'] = df['avg_purchase_amount']
        
        # Calculate time between purchases
        def calculate_purchase_interval(purchase_dates):
            if len(purchase_dates) < 2:
                return 0
            
            dates = [datetime.fromisoformat(date) for date in purchase_dates]
            dates.sort()
            
            intervals = [(dates[i+1] - dates[i]).days for i in range(len(dates)-1)]
            return sum(intervals) / len(intervals) if intervals else 0
        
        df['avg_purchase_interval'] = df['purchase_dates'].apply(calculate_purchase_interval)
        
        # Keep only relevant columns for modeling
        df = df[[
            'customer_id', 'current_lifetime_value', 'purchase_count', 
            'recency', 'frequency', 'monetary', 'avg_purchase_interval'
        ]]
        
        logging.info(f"Extracted CLV features for {len(df)} customers")
        return df
    
    def _extract_next_purchase_features(self):
        """Extract features for predicting next purchase timing and products."""
        query = """
        MATCH (c:Customer)-[p:PURCHASES]->(product:Product)
        WHERE p.timestamp IS NOT NULL
        WITH c, product, p.timestamp as purchase_date
        ORDER BY purchase_date DESC
        WITH c, collect({product: product.id, date: purchase_date})[0..10] as recent_purchases
        
        OPTIONAL MATCH (c)-[v:VIEWS]->(viewed_product:Product)
        WHERE NOT (c)-[:PURCHASES]->(viewed_product)
        WITH c, recent_purchases, collect(viewed_product.id) as viewed_not_purchased
        
        OPTIONAL MATCH (c)-[:BELONGS_TO]->(segment:Segment)
        WITH c, recent_purchases, viewed_not_purchased, collect(segment.id) as segments
        
        RETURN 
            c.customer_id as customer_id,
            recent_purchases,
            viewed_not_purchased,
            segments
        """
        
        # Execute the query
        purchase_history = self.run_query(query)
        
        if not purchase_history:
            logging.error("No purchase history found for next purchase prediction")
            return None
        
        # Process into a more structured format
        processed_data = []
        
        for customer in purchase_history:
            customer_id = customer['customer_id']
            recent_purchases = customer['recent_purchases']
            viewed_products = customer['viewed_not_purchased']
            segments = customer['segments']
            
            # Calculate days since last purchase
            if recent_purchases:
                last_purchase_date = datetime.fromisoformat(recent_purchases[0]['date'])
                days_since_last_purchase = (datetime.now() - last_purchase_date).days
            else:
                days_since_last_purchase = 999  # High number for customers with no purchases
            
            # Count total purchases
            purchase_count = len(recent_purchases)
            
            # Calculate average time between purchases
            if len(recent_purchases) >= 2:
                purchase_dates = [datetime.fromisoformat(p['date']) for p in recent_purchases]
                purchase_dates.sort()
                intervals = [(purchase_dates[i+1] - purchase_dates[i]).days for i in range(len(purchase_dates)-1)]
                avg_interval = sum(intervals) / len(intervals) if intervals else 0
            else:
                avg_interval = 0
            
            # Predict days until next purchase (target variable)
            # For already purchased customers, use their average interval
            # For new customers, use a default value (e.g., 30 days)
            if avg_interval > 0:
                days_until_next_purchase = max(1, avg_interval - days_since_last_purchase)
            else:
                days_until_next_purchase = 30  # Default assumption
            
            # Collect data
            customer_data = {
                'customer_id': customer_id,
                'purchase_count': purchase_count,
                'days_since_last_purchase': days_since_last_purchase,
                'avg_purchase_interval': avg_interval,
                'viewed_product_count': len(viewed_products),
                'segment_count': len(segments),
                'days_until_next_purchase': days_until_next_purchase
            }
            
            processed_data.append(customer_data)
        
        # Convert to DataFrame
        df = pd.DataFrame(processed_data)
        
        logging.info(f"Extracted next purchase features for {len(df)} customers")
        return df
    
    def train_churn_prediction_model(self, use_ensemble=True):
        """
        Train a machine learning model to predict customer churn.
        If use_ensemble is True, creates a multi-model ensemble with:
        - Random Forest
        - Gradient Boosting
        - Logistic Regression
        - AdaBoost
        """
        try:
            # Extract features
            features_df = self._extract_customer_features()
            
            if features_df is None or len(features_df) < 10:  # Need minimum data for training
                logging.error("Insufficient data for churn model training")
                return False
            
            # Prepare features and target
            X = features_df.drop(['customer_id', 'is_churned'], axis=1)
            y = features_df['is_churned']
            
            # Split into training and test sets
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )
            
            if not use_ensemble:
                # Train a random forest classifier (original approach)
                model = RandomForestClassifier(
                    n_estimators=100,
                    max_depth=None,
                    min_samples_split=2,
                    random_state=42
                )
                
                model.fit(X_train, y_train)
                
                # Evaluate the model
                y_pred = model.predict(X_test)
                
                metrics = {
                    'accuracy': accuracy_score(y_test, y_pred),
                    'precision': precision_score(y_test, y_pred, zero_division=0),
                    'recall': recall_score(y_test, y_pred, zero_division=0),
                    'f1': f1_score(y_test, y_pred, zero_division=0)
                }
                
                # Get feature importances
                feature_importances = dict(zip(X.columns, model.feature_importances_))
                
                # Store the model and metrics
                self.models['churn_prediction'] = model
                self.model_metrics['churn_prediction'] = metrics
                self.feature_importances['churn_prediction'] = feature_importances
                
                logging.info(f"Churn prediction model trained successfully. Accuracy: {metrics['accuracy']:.4f}")
                
                # Store model metadata in Neo4j
                self._store_model_metadata('churn_prediction', metrics, feature_importances)
                
                return True
            
            else:
                # Multi-model ensemble approach
                logging.info("Training multi-model ensemble for churn prediction...")
                
                # 1. Train individual models
                rf_model = RandomForestClassifier(
                    n_estimators=100, 
                    max_depth=None,
                    min_samples_split=2, 
                    random_state=42
                )
                
                gb_model = GradientBoostingClassifier(
                    n_estimators=100,
                    learning_rate=0.1,
                    max_depth=3,
                    random_state=42
                )
                
                lr_model = LogisticRegression(
                    C=1.0,
                    max_iter=1000,
                    class_weight='balanced',
                    random_state=42
                )
                
                ada_model = AdaBoostClassifier(
                    n_estimators=50,
                    learning_rate=0.1,
                    random_state=42
                )
                
                # Store individual models
                self.ensemble_models['churn_rf'] = rf_model
                self.ensemble_models['churn_gb'] = gb_model
                self.ensemble_models['churn_lr'] = lr_model
                self.ensemble_models['churn_ada'] = ada_model
                
                # 2. Create voting ensemble
                ensemble = VotingClassifier(
                    estimators=[
                        ('rf', rf_model),
                        ('gb', gb_model),
                        ('lr', lr_model),
                        ('ada', ada_model)
                    ],
                    voting='soft'  # Use probability estimates for voting
                )
                
                # 3. Train the ensemble
                ensemble.fit(X_train, y_train)
                
                # 4. Evaluate ensemble
                y_pred = ensemble.predict(X_test)
                y_prob = ensemble.predict_proba(X_test)[:, 1]  # Probability of class 1
                
                metrics = {
                    'accuracy': accuracy_score(y_test, y_pred),
                    'precision': precision_score(y_test, y_pred, zero_division=0),
                    'recall': recall_score(y_test, y_pred, zero_division=0),
                    'f1': f1_score(y_test, y_pred, zero_division=0),
                    'roc_auc': roc_auc_score(y_test, y_prob)
                }
                
                # 5. Get feature importances from the Random Forest component
                # Since ensemble doesn't provide feature importances directly
                rf_model.fit(X_train, y_train)
                feature_importances = dict(zip(X.columns, rf_model.feature_importances_))
                
                # 6. Store the ensemble model and metrics
                self.models['churn_prediction'] = ensemble
                self.model_metrics['churn_prediction'] = metrics
                self.feature_importances['churn_prediction'] = feature_importances
                
                # 7. Train individual models for completeness
                for name, model in self.ensemble_models.items():
                    model.fit(X_train, y_train)
                    model_preds = model.predict(X_test)
                    
                    model_metrics = {
                        'accuracy': accuracy_score(y_test, model_preds),
                        'precision': precision_score(y_test, model_preds, zero_division=0),
                        'recall': recall_score(y_test, model_preds, zero_division=0),
                        'f1': f1_score(y_test, model_preds, zero_division=0)
                    }
                    
                    if hasattr(model, 'predict_proba'):
                        model_probs = model.predict_proba(X_test)[:, 1]
                        model_metrics['roc_auc'] = roc_auc_score(y_test, model_probs)
                    
                    self.model_metrics[name] = model_metrics
                
                # 8. Log confusion matrix and classification report
                conf_matrix = confusion_matrix(y_test, y_pred)
                class_report = classification_report(y_test, y_pred)
                
                logging.info(f"Churn prediction ensemble model trained successfully.")
                logging.info(f"Ensemble metrics: Accuracy: {metrics['accuracy']:.4f}, ROC-AUC: {metrics['roc_auc']:.4f}")
                logging.info(f"Confusion matrix:\n{conf_matrix}")
                logging.info(f"Classification report:\n{class_report}")
                
                # 9. Store model metadata in Neo4j
                self._store_model_metadata('churn_prediction', metrics, feature_importances)
                
                # 10. Store comparison of models
                comparison = {
                    'ensemble': metrics,
                    'rf': self.model_metrics['churn_rf'],
                    'gb': self.model_metrics['churn_gb'],
                    'lr': self.model_metrics['churn_lr'],
                    'ada': self.model_metrics['churn_ada']
                }
                
                self._store_model_metadata('churn_model_comparison', comparison, {})
                
                return True
            
        except Exception as e:
            logging.error(f"Error training churn prediction model: {e}")
            return False
    
    def train_clv_prediction_model(self):
        """
        Train a machine learning model to predict customer lifetime value.
        Implements probabilistic modeling approach with cross-validation
        and hyperparameter tuning.
        """
        try:
            # Extract features
            features_df = self._extract_customer_lifetime_features()
            
            if features_df is None or len(features_df) < 10:  # Need minimum data for training
                logging.error("Insufficient data for CLV model training")
                return False
            
            # Prepare features and target
            X = features_df.drop(['customer_id', 'current_lifetime_value'], axis=1)
            y = features_df['current_lifetime_value']
            
            # Split into training and test sets
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )
            
            # Advanced approach: Random Forest with hyperparameter tuning
            logging.info("Training CLV prediction model with hyperparameter tuning...")
            
            # Define hyperparameter grid
            param_grid = {
                'n_estimators': [50, 100, 200],
                'max_depth': [None, 10, 20, 30],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            }
            
            # Create base model
            base_model = RandomForestRegressor(random_state=42)
            
            # Set up grid search with cross-validation
            grid_search = GridSearchCV(
                estimator=base_model,
                param_grid=param_grid,
                cv=3,  # 3-fold cross-validation
                n_jobs=-1,  # Use all available processors
                scoring='neg_mean_squared_error',
                verbose=1
            )
            
            # Fit grid search
            grid_search.fit(X_train, y_train)
            
            # Get best model
            best_model = grid_search.best_estimator_
            
            # Evaluate the model
            y_pred = best_model.predict(X_test)
            
            # Calculate metrics
            metrics = {
                'mae': mean_absolute_error(y_test, y_pred),
                'mse': mean_squared_error(y_test, y_pred),
                'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
                'r2': r2_score(y_test, y_pred)
            }
            
            # Add probabilistic confidence intervals using bootstrap
            # This simulates a probabilistic approach by generating multiple predictions
            n_bootstraps = 100
            bootstrap_predictions = np.zeros((len(X_test), n_bootstraps))
            
            for i in range(n_bootstraps):
                # Create bootstrap sample (random sampling with replacement)
                indices = np.random.choice(len(X_train), len(X_train), replace=True)
                X_bootstrap = X_train.iloc[indices]
                y_bootstrap = y_train.iloc[indices]
                
                # Train model on bootstrap sample
                bootstrap_model = RandomForestRegressor(**grid_search.best_params_, random_state=i)
                bootstrap_model.fit(X_bootstrap, y_bootstrap)
                
                # Predict
                bootstrap_predictions[:, i] = bootstrap_model.predict(X_test)
            
            # Calculate prediction intervals
            lower_bound = np.percentile(bootstrap_predictions, 5, axis=1)
            upper_bound = np.percentile(bootstrap_predictions, 95, axis=1)
            
            # Calculate average interval width as a measure of uncertainty
            avg_interval_width = np.mean(upper_bound - lower_bound)
            
            # Add to metrics
            metrics['avg_uncertainty'] = avg_interval_width
            metrics['best_params'] = grid_search.best_params_
            
            # Get feature importances
            feature_importances = dict(zip(X.columns, best_model.feature_importances_))
            
            # Store the model and metrics
            self.models['clv_prediction'] = best_model
            self.model_metrics['clv_prediction'] = metrics
            self.feature_importances['clv_prediction'] = feature_importances
            
            logging.info(f"Probabilistic CLV model trained successfully. R²: {metrics['r2']:.4f}")
            logging.info(f"Best parameters: {grid_search.best_params_}")
            logging.info(f"Average 90% prediction interval width: ${avg_interval_width:.2f}")
            
            # Store model metadata in Neo4j
            self._store_model_metadata('clv_prediction', metrics, feature_importances)
            
            return True
            
        except Exception as e:
            logging.error(f"Error training CLV prediction model: {e}")
            return False
    
    def train_next_purchase_model(self):
        """Train a machine learning model to predict days until next purchase."""
        try:
            # Extract features
            features_df = self._extract_next_purchase_features()
            
            if features_df is None or len(features_df) < 10:  # Need minimum data for training
                logging.error("Insufficient data for next purchase model training")
                return False
            
            # Prepare features and target
            X = features_df.drop(['customer_id', 'days_until_next_purchase'], axis=1)
            y = features_df['days_until_next_purchase']
            
            # Split into training and test sets
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )
            
            # Train a random forest regressor
            model = RandomForestRegressor(
                n_estimators=100,
                max_depth=None,
                min_samples_split=2,
                random_state=42
            )
            
            model.fit(X_train, y_train)
            
            # Evaluate the model
            y_pred = model.predict(X_test)
            
            metrics = {
                'mae': mean_absolute_error(y_test, y_pred),
                'mse': mean_squared_error(y_test, y_pred),
                'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
                'r2': r2_score(y_test, y_pred)
            }
            
            # Get feature importances
            feature_importances = dict(zip(X.columns, model.feature_importances_))
            
            # Store the model and metrics
            self.models['next_purchase'] = model
            self.model_metrics['next_purchase'] = metrics
            self.feature_importances['next_purchase'] = feature_importances
            
            logging.info(f"Next purchase prediction model trained successfully. RMSE: {metrics['rmse']:.4f} days")
            
            # Store model metadata in Neo4j
            self._store_model_metadata('next_purchase', metrics, feature_importances)
            
            return True
            
        except Exception as e:
            logging.error(f"Error training next purchase prediction model: {e}")
            return False
    
    def _store_model_metadata(self, model_name, metrics, feature_importances):
        """Store model metadata in Neo4j."""
        try:
            # Convert metrics and feature importances to JSON strings
            metrics_json = json.dumps(metrics)
            feature_importances_json = json.dumps(feature_importances)
            
            # Create or update model metadata node
            query = """
            MERGE (m:PredictiveModel {name: $model_name})
            SET m.last_updated = datetime(),
                m.metrics = $metrics,
                m.feature_importances = $feature_importances
            RETURN m
            """
            
            self.run_query(query, {
                'model_name': model_name,
                'metrics': metrics_json,
                'feature_importances': feature_importances_json
            })
            
            logging.info(f"Stored metadata for {model_name} model in Neo4j")
            return True
            
        except Exception as e:
            logging.error(f"Error storing model metadata: {e}")
            return False
    
    def run_dynamic_customer_segmentation(self, num_clusters=5):
        """
        Use K-means clustering to dynamically segment customers based on behavior.
        Creates segments in Neo4j based on the clustering results.
        """
        try:
            # Extract features for clustering
            features_df = self._extract_customer_features()
            
            if features_df is None or len(features_df) < num_clusters:
                logging.error("Insufficient data for customer segmentation")
                return False
            
            # Select features for clustering
            cluster_features = features_df[[
                'view_count', 'click_count', 'visit_count', 
                'cart_add_count', 'purchase_count', 'days_since_activity',
                'cart_abandonment_rate', 'conversion_rate'
            ]]
            
            # Scale the features
            scaler = StandardScaler()
            scaled_features = scaler.fit_transform(cluster_features)
            
            # Apply K-means clustering
            kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init=10)
            cluster_labels = kmeans.fit_predict(scaled_features)
            
            # Add cluster labels to the DataFrame
            features_df['cluster'] = cluster_labels
            
            # Analyze clusters
            cluster_profiles = []
            for i in range(num_clusters):
                cluster_data = features_df[features_df['cluster'] == i]
                profile = {
                    'cluster_id': i,
                    'size': len(cluster_data),
                    'percentage': len(cluster_data) / len(features_df) * 100,
                    'avg_purchase_count': cluster_data['purchase_count'].mean(),
                    'avg_cart_abandonment_rate': cluster_data['cart_abandonment_rate'].mean(),
                    'avg_conversion_rate': cluster_data['conversion_rate'].mean(),
                    'avg_days_since_activity': cluster_data['days_since_activity'].mean(),
                    'customer_ids': cluster_data['customer_id'].tolist()
                }
                cluster_profiles.append(profile)
            
            # Determine segment names based on characteristics
            segment_names = self._generate_segment_names(cluster_profiles)
            
            # Store clusters in Neo4j
            self._store_cluster_results(cluster_profiles, segment_names)
            
            logging.info(f"Dynamic customer segmentation completed with {num_clusters} clusters")
            return cluster_profiles
            
        except Exception as e:
            logging.error(f"Error running dynamic customer segmentation: {e}")
            return False
    
    def _generate_segment_names(self, cluster_profiles):
        """Generate meaningful names for customer segments based on their characteristics."""
        segment_names = {}
        
        for profile in cluster_profiles:
            cluster_id = profile['cluster_id']
            
            # Determine key characteristics
            high_value = profile['avg_purchase_count'] > 3
            high_engagement = profile['avg_conversion_rate'] > 0.2
            high_abandonment = profile['avg_cart_abandonment_rate'] > 0.5
            recent_activity = profile['avg_days_since_activity'] < 30
            inactive = profile['avg_days_since_activity'] > 90
            
            # Generate name based on characteristics
            if high_value and high_engagement and recent_activity:
                name = "High-Value Active"
            elif high_value and not recent_activity:
                name = "High-Value Dormant"
            elif high_engagement and recent_activity:
                name = "Active Engaged"
            elif high_abandonment:
                name = "Cart Abandoners"
            elif inactive:
                name = "Inactive"
            else:
                name = "Casual Browsers"
            
            # Add cluster number to ensure uniqueness
            segment_names[cluster_id] = f"{name} (Segment {cluster_id})"
        
        return segment_names
    
    def _store_cluster_results(self, cluster_profiles, segment_names):
        """Store clustering results in Neo4j."""
        try:
            # First, remove old dynamic segments
            cleanup_query = """
            MATCH (s:Segment)
            WHERE s.is_dynamic = true
            DETACH DELETE s
            """
            
            self.run_query(cleanup_query)
            
            # Create new segment nodes
            for profile in cluster_profiles:
                cluster_id = profile['cluster_id']
                segment_name = segment_names[cluster_id]
                
                # Create segment node
                create_segment_query = """
                CREATE (s:Segment {
                    id: $segment_id,
                    name: $segment_name,
                    is_dynamic: true,
                    created_at: datetime(),
                    size: $size,
                    avg_purchase_count: $avg_purchase_count,
                    avg_cart_abandonment_rate: $avg_cart_abandonment_rate,
                    avg_conversion_rate: $avg_conversion_rate,
                    avg_days_since_activity: $avg_days_since_activity
                })
                RETURN s
                """
                
                self.run_query(create_segment_query, {
                    'segment_id': f"DYNAMIC_SEGMENT_{cluster_id}",
                    'segment_name': segment_name,
                    'size': profile['size'],
                    'avg_purchase_count': profile['avg_purchase_count'],
                    'avg_cart_abandonment_rate': profile['avg_cart_abandonment_rate'],
                    'avg_conversion_rate': profile['avg_conversion_rate'],
                    'avg_days_since_activity': profile['avg_days_since_activity']
                })
                
                # Connect customers to segment
                for customer_id in profile['customer_ids']:
                    connect_query = """
                    MATCH (c:Customer {customer_id: $customer_id})
                    MATCH (s:Segment {id: $segment_id})
                    MERGE (c)-[:BELONGS_TO {is_dynamic: true, assigned_at: datetime()}]->(s)
                    """
                    
                    self.run_query(connect_query, {
                        'customer_id': customer_id,
                        'segment_id': f"DYNAMIC_SEGMENT_{cluster_id}"
                    })
            
            logging.info(f"Stored {len(cluster_profiles)} dynamic segments in Neo4j")
            return True
            
        except Exception as e:
            logging.error(f"Error storing cluster results: {e}")
            return False
    
    def detect_anomalies(self, z_score_threshold=3.0, use_advanced_methods=True):
        """
        Detect anomalous customer behavior using advanced techniques.
        Flags customers with unusual patterns using:
        - Statistical methods (z-scores)
        - Isolation Forests (unsupervised anomaly detection)
        - Autoencoders (if data volume permits)
        """
        try:
            # Query to get customer activity by segment with more features for advanced anomaly detection
            query = """
            MATCH (c:Customer)-[:BELONGS_TO]->(s:Segment)
            
            // Count recent interactions by type (last 30 days)
            OPTIONAL MATCH (c)-[r:VIEWS]->()
            WHERE r.timestamp IS NOT NULL AND 
                  duration.inDays(datetime(r.timestamp), datetime()).days <= 30
            WITH c, s, count(r) as view_count
            
            OPTIONAL MATCH (c)-[r:CLICKS_ON]->()
            WHERE r.timestamp IS NOT NULL AND 
                  duration.inDays(datetime(r.timestamp), datetime()).days <= 30
            WITH c, s, view_count, count(r) as click_count
            
            OPTIONAL MATCH (c)-[r:ADDS_TO_CART]->()
            WHERE r.timestamp IS NOT NULL AND 
                  duration.inDays(datetime(r.timestamp), datetime()).days <= 30
            WITH c, s, view_count, click_count, count(r) as cart_add_count
            
            OPTIONAL MATCH (c)-[r:ABANDONS]->()
            WHERE r.timestamp IS NOT NULL AND 
                  duration.inDays(datetime(r.timestamp), datetime()).days <= 30
            WITH c, s, view_count, click_count, cart_add_count, count(r) as cart_abandon_count
            
            // Get all recent interactions
            OPTIONAL MATCH (c)-[r]->()
            WHERE r.timestamp IS NOT NULL AND 
                  duration.inDays(datetime(r.timestamp), datetime()).days <= 30
            WITH c, s, view_count, click_count, cart_add_count, cart_abandon_count, count(r) as recent_activity_count
            
            // Get purchase amounts
            OPTIONAL MATCH (c)-[p:PURCHASES]->()
            WHERE p.amount IS NOT NULL AND
                  duration.inDays(datetime(p.timestamp), datetime()).days <= 30
            WITH c, s, view_count, click_count, cart_add_count, cart_abandon_count, recent_activity_count,
                 sum(p.amount) as recent_purchase_amount, count(p) as purchase_count
            
            // Get last activity time
            OPTIONAL MATCH (c)-[r]->()
            WHERE r.timestamp IS NOT NULL
            WITH c, s, view_count, click_count, cart_add_count, cart_abandon_count, 
                 recent_activity_count, recent_purchase_amount, purchase_count,
                 max(r.timestamp) as last_activity
            
            RETURN 
                c.customer_id as customer_id,
                s.id as segment_id,
                view_count,
                click_count,
                cart_add_count,
                cart_abandon_count,
                recent_activity_count,
                purchase_count,
                CASE WHEN recent_purchase_amount IS NULL THEN 0 ELSE recent_purchase_amount END as recent_purchase_amount,
                CASE WHEN last_activity IS NULL THEN 999 
                     ELSE duration.inDays(datetime(last_activity), datetime()).days
                END as days_since_activity
            """
            
            result = self.run_query(query)
            
            if not result:
                logging.error("No data available for anomaly detection")
                return False
            
            # Convert to DataFrame
            df = pd.DataFrame(result)
            
            # Add derived features
            df['avg_purchase_value'] = df.apply(
                lambda x: x['recent_purchase_amount'] / x['purchase_count'] if x['purchase_count'] > 0 else 0, 
                axis=1
            )
            
            df['cart_abandonment_rate'] = df.apply(
                lambda x: x['cart_abandon_count'] / x['cart_add_count'] if x['cart_add_count'] > 0 else 0,
                axis=1
            )
            
            # Basic Approach: Z-score anomaly detection by segment
            # ===============================================================
            
            # Group by segment to calculate segment averages and std devs
            segment_stats = df.groupby('segment_id').agg({
                'recent_activity_count': ['mean', 'std'],
                'recent_purchase_amount': ['mean', 'std'],
                'avg_purchase_value': ['mean', 'std'],
                'cart_abandonment_rate': ['mean', 'std'],
                'days_since_activity': ['mean', 'std']
            })
            
            # Detect anomalies using z-scores
            z_score_anomalies = []
            
            for _, row in df.iterrows():
                customer_id = row['customer_id']
                segment_id = row['segment_id']
                
                # Get segment statistics and calculate z-scores for multiple features
                try:
                    anomaly_features = {}
                    anomaly_types = []
                    
                    # Check each feature for anomalies
                    for feature in ['recent_activity_count', 'recent_purchase_amount', 
                                   'avg_purchase_value', 'cart_abandonment_rate']:
                        
                        feature_value = row[feature]
                        segment_mean = segment_stats.loc[segment_id, (feature, 'mean')]
                        segment_std = segment_stats.loc[segment_id, (feature, 'std')]
                        
                        # Prevent division by zero
                        segment_std = max(segment_std, 0.0001)
                        
                        # Calculate Z-score
                        z_score = (feature_value - segment_mean) / segment_std
                        
                        # Store z-score
                        anomaly_features[f"{feature}_z_score"] = z_score
                        
                        # Check if anomalous
                        if abs(z_score) > z_score_threshold:
                            if z_score > z_score_threshold:
                                anomaly_types.append(f"HIGH_{feature.upper()}")
                            else:
                                anomaly_types.append(f"LOW_{feature.upper()}")
                    
                    # Days since activity is special - only high is anomalous
                    feature = 'days_since_activity'
                    feature_value = row[feature]
                    segment_mean = segment_stats.loc[segment_id, (feature, 'mean')]
                    segment_std = segment_stats.loc[segment_id, (feature, 'std')]
                    segment_std = max(segment_std, 0.0001)
                    z_score = (feature_value - segment_mean) / segment_std
                    anomaly_features[f"{feature}_z_score"] = z_score
                    
                    if z_score > z_score_threshold:
                        anomaly_types.append("INACTIVE")
                    
                    if anomaly_types:
                        # Create anomaly entry
                        anomaly = {
                            'customer_id': customer_id,
                            'segment_id': segment_id,
                            'detection_method': 'z_score',
                            'anomaly_type': anomaly_types,
                            'anomaly_scores': anomaly_features,
                            'feature_values': {
                                'recent_activity_count': row['recent_activity_count'],
                                'recent_purchase_amount': row['recent_purchase_amount'],
                                'avg_purchase_value': row['avg_purchase_value'],
                                'cart_abandonment_rate': row['cart_abandonment_rate'],
                                'days_since_activity': row['days_since_activity']
                            }
                        }
                        z_score_anomalies.append(anomaly)
                        
                except Exception as e:
                    logging.warning(f"Error processing z-score anomaly for customer {customer_id}: {e}")
                    continue
            
            # Advanced Anomaly Detection using Isolation Forest
            # ===============================================================
            if use_advanced_methods and len(df) >= 30:  # Need minimum data for meaningful unsupervised detection
                try:
                    from sklearn.ensemble import IsolationForest
                    
                    logging.info("Running advanced anomaly detection with Isolation Forest")
                    
                    # Prepare data for Isolation Forest
                    features_for_isolation = [
                        'view_count', 'click_count', 'cart_add_count', 'cart_abandon_count',
                        'recent_activity_count', 'purchase_count', 'recent_purchase_amount',
                        'avg_purchase_value', 'cart_abandonment_rate', 'days_since_activity'
                    ]
                    
                    # Filter out features if they're all zeros
                    usable_features = []
                    for feature in features_for_isolation:
                        if df[feature].sum() > 0:
                            usable_features.append(feature)
                    
                    if len(usable_features) < 3:
                        logging.warning("Not enough non-zero features for Isolation Forest")
                    else:
                        # Scale features
                        scaler = StandardScaler()
                        X = scaler.fit_transform(df[usable_features])
                        
                        # Run Isolation Forest
                        isolation_forest = IsolationForest(
                            n_estimators=100,
                            contamination=0.05,  # Expect ~5% anomalies
                            random_state=42
                        )
                        
                        # Fit and predict
                        anomaly_scores = isolation_forest.fit_predict(X)
                        
                        # Decision function - lower values are more anomalous
                        anomaly_decision = isolation_forest.decision_function(X)
                        
                        # Find anomalies (where score is -1)
                        for i, score in enumerate(anomaly_scores):
                            if score == -1:  # Anomaly
                                customer_id = df.iloc[i]['customer_id']
                                segment_id = df.iloc[i]['segment_id']
                                
                                # Create feature dictionary
                                feature_values = {}
                                for feature in usable_features:
                                    feature_values[feature] = df.iloc[i][feature]
                                
                                # Add to anomalies list
                                anomaly = {
                                    'customer_id': customer_id,
                                    'segment_id': segment_id,
                                    'detection_method': 'isolation_forest',
                                    'anomaly_type': ["BEHAVIORAL_OUTLIER"],
                                    'anomaly_scores': {
                                        'isolation_score': float(anomaly_decision[i])
                                    },
                                    'feature_values': feature_values
                                }
                                
                                # Check if this customer is already in z_score_anomalies
                                existing_anomaly = next(
                                    (a for a in z_score_anomalies if a['customer_id'] == customer_id), 
                                    None
                                )
                                
                                if existing_anomaly:
                                    # Update existing entry
                                    existing_anomaly['detection_method'] += ',isolation_forest'
                                    existing_anomaly['anomaly_type'].append("BEHAVIORAL_OUTLIER")
                                    existing_anomaly['anomaly_scores']['isolation_score'] = float(anomaly_decision[i])
                                else:
                                    # Add new entry
                                    z_score_anomalies.append(anomaly)
                except ImportError:
                    logging.warning("Isolation Forest not available, skipping advanced anomaly detection")
                except Exception as e:
                    logging.error(f"Error in Isolation Forest anomaly detection: {e}")
            
            # Convert all NumPy types to Python native types for JSON serialization
            for anomaly in z_score_anomalies:
                for key, value in anomaly['anomaly_scores'].items():
                    if isinstance(value, (np.int_, np.intc, np.intp, np.int8, np.int16, np.int32, np.int64)):
                        anomaly['anomaly_scores'][key] = int(value)
                    elif isinstance(value, (np.float_, np.float16, np.float32, np.float64)):
                        anomaly['anomaly_scores'][key] = float(value)
            
            # Store anomalies in Neo4j
            if z_score_anomalies:
                self._store_anomalies(z_score_anomalies)
            
            logging.info(f"Detected {len(z_score_anomalies)} customer behavior anomalies using multiple methods")
            return z_score_anomalies
            
        except Exception as e:
            logging.error(f"Error detecting anomalies: {e}")
            return False
    
    def _store_anomalies(self, anomalies):
        """Store detected anomalies in Neo4j with enhanced data structure."""
        try:
            # Remove old anomaly nodes
            cleanup_query = """
            MATCH (a:Anomaly)
            DETACH DELETE a
            """
            
            self.run_query(cleanup_query)
            
            # Create anomaly nodes and relationships
            for anomaly in anomalies:
                # Extract data with proper defaults
                customer_id = anomaly.get('customer_id', '')
                segment_id = anomaly.get('segment_id', '')
                detection_method = anomaly.get('detection_method', 'unknown')
                anomaly_types = anomaly.get('anomaly_type', [])
                anomaly_scores = anomaly.get('anomaly_scores', {})
                feature_values = anomaly.get('feature_values', {})
                
                # Convert any nested dictionaries to strings for Neo4j storage
                anomaly_scores_json = json.dumps(anomaly_scores)
                feature_values_json = json.dumps(feature_values)
                
                # Create enriched anomaly node
                create_anomaly_query = """
                MATCH (c:Customer {customer_id: $customer_id})
                CREATE (a:Anomaly {
                    detected_at: datetime(),
                    detection_method: $detection_method,
                    segment_id: $segment_id,
                    anomaly_types: $anomaly_types,
                    anomaly_scores: $anomaly_scores,
                    feature_values: $feature_values,
                    severity: $severity
                })
                CREATE (c)-[:HAS_ANOMALY {detected_at: datetime()}]->(a)
                
                // Also connect to segment if possible
                WITH a
                MATCH (s:Segment {id: $segment_id})
                MERGE (a)-[:BELONGS_TO_SEGMENT]->(s)
                
                RETURN a
                """
                
                # Calculate severity based on the highest absolute z-score or isolation score
                max_score = 0
                for _, score in anomaly_scores.items():
                    if isinstance(score, (int, float)) and abs(score) > max_score:
                        max_score = abs(score)
                
                # Map to severity levels
                severity = "Low"
                if max_score > 5.0:
                    severity = "Critical"
                elif max_score > 4.0:
                    severity = "High"
                elif max_score > 3.0:
                    severity = "Medium"
                
                # Execute query
                self.run_query(create_anomaly_query, {
                    'customer_id': customer_id,
                    'segment_id': segment_id,
                    'detection_method': detection_method,
                    'anomaly_types': anomaly_types,
                    'anomaly_scores': anomaly_scores_json,
                    'feature_values': feature_values_json,
                    'severity': severity
                })
            
            logging.info(f"Stored {len(anomalies)} enhanced anomalies in Neo4j")
            return True
            
        except Exception as e:
            logging.error(f"Error storing anomalies: {e}")
            return False
    
    def predict_customer_insights(self, customer_id):
        """Generate predictive insights for a specific customer."""
        try:
            # First, ensure all models are trained
            if 'churn_prediction' not in self.models:
                self.train_churn_prediction_model()
            
            if 'clv_prediction' not in self.models:
                self.train_clv_prediction_model()
            
            if 'next_purchase' not in self.models:
                self.train_next_purchase_model()
            
            # Get customer features
            features_query = """
            MATCH (c:Customer {customer_id: $customer_id})
            
            // Count interactions by type
            OPTIONAL MATCH (c)-[r:VIEWS]->()
            WITH c, count(r) as view_count
            
            OPTIONAL MATCH (c)-[r:CLICKS_ON]->()
            WITH c, view_count, count(r) as click_count
            
            OPTIONAL MATCH (c)-[r:VISITS]->()
            WITH c, view_count, click_count, count(r) as visit_count
            
            OPTIONAL MATCH (c)-[r:ADDS_TO_CART]->()
            WITH c, view_count, click_count, visit_count, count(r) as cart_add_count
            
            OPTIONAL MATCH (c)-[r:ABANDONS]->()
            WITH c, view_count, click_count, visit_count, cart_add_count, count(r) as cart_abandon_count
            
            OPTIONAL MATCH (c)-[r:PURCHASES]->()
            WITH c, view_count, click_count, visit_count, cart_add_count, cart_abandon_count, count(r) as purchase_count
            
            // Get most recent activity timestamp
            OPTIONAL MATCH (c)-[r]->()
            WHERE r.timestamp IS NOT NULL
            WITH c, view_count, click_count, visit_count, cart_add_count, cart_abandon_count, purchase_count,
                 max(r.timestamp) as last_activity
            
            // Get segments count
            OPTIONAL MATCH (c)-[:BELONGS_TO]->(s:Segment)
            WITH c, view_count, click_count, visit_count, cart_add_count, cart_abandon_count, purchase_count,
                 last_activity, count(s) as segment_count
            
            // Get device count
            OPTIONAL MATCH (c)-[:USES]->(d:Device)
            WITH c, view_count, click_count, visit_count, cart_add_count, cart_abandon_count, purchase_count,
                 last_activity, segment_count, count(d) as device_count
            
            // Get purchase history
            OPTIONAL MATCH (c)-[p:PURCHASES]->(product:Product)
            WHERE p.timestamp IS NOT NULL
            WITH c, view_count, click_count, visit_count, cart_add_count, cart_abandon_count, purchase_count,
                 last_activity, segment_count, device_count, 
                 collect(p.timestamp) as purchase_dates,
                 avg(p.amount) as avg_purchase_amount
            
            RETURN 
                c.customer_id as customer_id,
                c.lifetime_value as lifetime_value,
                view_count,
                click_count,
                visit_count,
                cart_add_count,
                cart_abandon_count,
                purchase_count,
                CASE WHEN last_activity IS NULL THEN 999 
                     ELSE duration.inDays(datetime(last_activity), datetime()).days
                END as days_since_activity,
                segment_count,
                device_count,
                purchase_dates,
                avg_purchase_amount
            """
            
            result = self.run_query(features_query, {'customer_id': customer_id})
            
            if not result or len(result) == 0:
                logging.error(f"No features found for customer {customer_id}")
                return None
            
            # Extract customer data
            customer_data = result[0]
            
            # Prepare features for prediction
            churn_features = pd.DataFrame([{
                'lifetime_value': customer_data.get('lifetime_value', 0),
                'view_count': customer_data.get('view_count', 0),
                'click_count': customer_data.get('click_count', 0),
                'visit_count': customer_data.get('visit_count', 0),
                'cart_add_count': customer_data.get('cart_add_count', 0),
                'cart_abandon_count': customer_data.get('cart_abandon_count', 0),
                'purchase_count': customer_data.get('purchase_count', 0),
                'days_since_activity': customer_data.get('days_since_activity', 999),
                'segment_count': customer_data.get('segment_count', 0),
                'device_count': customer_data.get('device_count', 0),
                'cart_abandonment_rate': customer_data.get('cart_abandon_count', 0) / max(1, customer_data.get('cart_add_count', 1)),
                'conversion_rate': customer_data.get('purchase_count', 0) / max(1, customer_data.get('visit_count', 1)),
                'click_through_rate': customer_data.get('click_count', 0) / max(1, customer_data.get('view_count', 1))
            }])
            
            # Calculate CLV features
            purchase_dates = customer_data.get('purchase_dates', [])
            
            if purchase_dates:
                dates = [datetime.fromisoformat(date) for date in purchase_dates]
                dates.sort()
                
                recency = (datetime.now() - dates[-1]).days
                frequency = len(dates)
                monetary = customer_data.get('avg_purchase_amount', 0)
                
                # Calculate time between purchases
                if len(dates) >= 2:
                    intervals = [(dates[i+1] - dates[i]).days for i in range(len(dates)-1)]
                    avg_purchase_interval = sum(intervals) / len(intervals)
                else:
                    avg_purchase_interval = 30  # Default assumption
            else:
                recency = 999
                frequency = 0
                monetary = 0
                avg_purchase_interval = 0
            
            clv_features = pd.DataFrame([{
                'purchase_count': frequency,
                'recency': recency,
                'frequency': frequency,
                'monetary': monetary,
                'avg_purchase_interval': avg_purchase_interval
            }])
            
            next_purchase_features = pd.DataFrame([{
                'purchase_count': customer_data.get('purchase_count', 0),
                'days_since_last_purchase': recency,
                'avg_purchase_interval': avg_purchase_interval,
                'viewed_product_count': customer_data.get('view_count', 0),
                'segment_count': customer_data.get('segment_count', 0)
            }])
            
            # Make predictions
            insights = {
                'customer_id': customer_id,
                'timestamp': datetime.now().isoformat()
            }
            
            # Churn prediction
            if 'churn_prediction' in self.models:
                churn_prob = self.models['churn_prediction'].predict_proba(churn_features)[0][1]
                insights['churn_probability'] = churn_prob
                insights['churn_risk_level'] = 'High' if churn_prob > 0.7 else ('Medium' if churn_prob > 0.3 else 'Low')
            
            # CLV prediction
            if 'clv_prediction' in self.models:
                predicted_clv = self.models['clv_prediction'].predict(clv_features)[0]
                insights['predicted_lifetime_value'] = predicted_clv
                insights['current_lifetime_value'] = customer_data.get('lifetime_value', 0)
                insights['lifetime_value_growth'] = predicted_clv - customer_data.get('lifetime_value', 0)
            
            # Next purchase prediction
            if 'next_purchase' in self.models:
                days_until_next_purchase = self.models['next_purchase'].predict(next_purchase_features)[0]
                insights['days_until_next_purchase'] = max(1, int(days_until_next_purchase))
                insights['predicted_next_purchase_date'] = (datetime.now() + timedelta(days=max(1, int(days_until_next_purchase)))).isoformat()
            
            # Store predictions in Neo4j
            self._store_customer_predictions(customer_id, insights)
            
            logging.info(f"Generated predictive insights for customer {customer_id}")
            return insights
            
        except Exception as e:
            logging.error(f"Error generating predictive insights: {e}")
            return None
    
    def _store_customer_predictions(self, customer_id, insights):
        """Store customer predictions in Neo4j."""
        try:
            # Create or update predictions node
            query = """
            MATCH (c:Customer {customer_id: $customer_id})
            
            MERGE (p:Prediction {customer_id: $customer_id})
            SET p.timestamp = datetime(),
                p.churn_probability = $churn_probability,
                p.churn_risk_level = $churn_risk_level,
                p.predicted_lifetime_value = $predicted_lifetime_value,
                p.lifetime_value_growth = $lifetime_value_growth,
                p.days_until_next_purchase = $days_until_next_purchase,
                p.predicted_next_purchase_date = $predicted_next_purchase_date
            
            MERGE (c)-[:HAS_PREDICTION]->(p)
            
            RETURN p
            """
            
            self.run_query(query, {
                'customer_id': customer_id,
                'churn_probability': insights.get('churn_probability', 0),
                'churn_risk_level': insights.get('churn_risk_level', 'Unknown'),
                'predicted_lifetime_value': insights.get('predicted_lifetime_value', 0),
                'lifetime_value_growth': insights.get('lifetime_value_growth', 0),
                'days_until_next_purchase': insights.get('days_until_next_purchase', 0),
                'predicted_next_purchase_date': insights.get('predicted_next_purchase_date', '')
            })
            
            logging.info(f"Stored predictions for customer {customer_id} in Neo4j")
            return True
            
        except Exception as e:
            logging.error(f"Error storing customer predictions: {e}")
            return False
    
    def setup_gds_projections(self):
        """
        Set up Neo4j Graph Data Science projections for advanced graph algorithms.
        This enables community detection, centrality measures, and path finding.
        """
        if not self._check_gds_available():
            logging.error("Neo4j Graph Data Science library is not available")
            return False
        
        try:
            # Create customer journey projection
            journey_projection_query = """
            CALL gds.graph.project(
              'customer_journey',
              ['Customer', 'Product', 'Page', 'Advertisement', 'Cart', 'Email', 'Device', 'Browser', 'Location', 'Segment', 'Persona'],
              {
                INTERACTS_WITH: {
                  type: 'INTERACTS_WITH',
                  properties: ['timestamp', 'weight']
                },
                VIEWS: {
                  type: 'VIEWS',
                  properties: ['timestamp']
                },
                CLICKS_ON: {
                  type: 'CLICKS_ON',
                  properties: ['timestamp']
                },
                VISITS: {
                  type: 'VISITS',
                  properties: ['timestamp']
                },
                ADDS_TO_CART: {
                  type: 'ADDS_TO_CART',
                  properties: ['timestamp']
                },
                PURCHASES: {
                  type: 'PURCHASES',
                  properties: ['timestamp', 'amount']
                },
                ABANDONS: {
                  type: 'ABANDONS',
                  properties: ['timestamp']
                },
                BELONGS_TO: {
                  type: 'BELONGS_TO',
                  properties: ['timestamp']
                },
                USES: {
                  type: 'USES',
                  properties: ['timestamp']
                },
                HAS_PERSONA: {
                  type: 'HAS_PERSONA',
                  properties: ['timestamp']
                }
              }
            )
            YIELD graphName, nodeCount, relationshipCount
            RETURN graphName, nodeCount, relationshipCount
            """
            
            journey_result = self.run_query(journey_projection_query)
            
            if journey_result:
                logging.info(f"Created customer journey projection with {journey_result[0]['nodeCount']} nodes and {journey_result[0]['relationshipCount']} relationships")
            
            # Create customer similarity projection
            similarity_projection_query = """
            CALL gds.graph.project(
              'customer_similarity',
              ['Customer', 'Segment', 'Persona', 'Product', 'Page'],
              {
                BELONGS_TO: {
                  type: 'BELONGS_TO',
                  properties: ['timestamp']
                },
                HAS_PERSONA: {
                  type: 'HAS_PERSONA',
                  properties: ['timestamp']
                },
                PURCHASES: {
                  type: 'PURCHASES',
                  properties: ['timestamp', 'amount']
                },
                VIEWS: {
                  type: 'VIEWS',
                  properties: ['timestamp']
                }
              }
            )
            YIELD graphName, nodeCount, relationshipCount
            RETURN graphName, nodeCount, relationshipCount
            """
            
            similarity_result = self.run_query(similarity_projection_query)
            
            if similarity_result:
                logging.info(f"Created customer similarity projection with {similarity_result[0]['nodeCount']} nodes and {similarity_result[0]['relationshipCount']} relationships")
            
            return True
            
        except Exception as e:
            logging.error(f"Error setting up GDS projections: {e}")
            return False
    
    def run_community_detection(self):
        """Run community detection using Louvain algorithm to find natural customer clusters."""
        if not self._check_gds_available():
            logging.error("Neo4j Graph Data Science library is not available")
            return False
        
        try:
            # Set up projection if not already done
            self.setup_gds_projections()
            
            # Run Louvain community detection
            louvain_query = """
            CALL gds.louvain.write(
              'customer_similarity',
              {
                relationshipWeightProperty: null,
                writeProperty: 'louvainCommunity',
                seedProperty: null,
                maxLevels: 10,
                maxIterations: 10
              }
            )
            YIELD communityCount, modularity, modularities
            RETURN communityCount, modularity, modularities
            """
            
            louvain_result = self.run_query(louvain_query)
            
            if louvain_result:
                community_count = louvain_result[0]['communityCount']
                modularity = louvain_result[0]['modularity']
                logging.info(f"Louvain community detection found {community_count} communities with modularity {modularity}")
                
                # Create named segments based on communities
                self._create_community_segments()
                
                return {
                    'community_count': community_count,
                    'modularity': modularity,
                    'algorithm': 'louvain'
                }
            else:
                logging.error("Louvain community detection failed")
                return False
            
        except Exception as e:
            logging.error(f"Error running community detection: {e}")
            return False
    
    def _create_community_segments(self):
        """Create named segments based on detected communities."""
        try:
            # Query to get community information
            query = """
            MATCH (c:Customer)
            WHERE exists(c.louvainCommunity)
            WITH c.louvainCommunity AS community, collect(c) AS customers
            
            // Get purchase patterns
            OPTIONAL MATCH (customer)-[:PURCHASES]->(p:Product)
            WHERE customer IN customers
            WITH community, customers, collect(DISTINCT p.id) AS common_products
            
            // Get page visits
            OPTIONAL MATCH (customer)-[:VISITS]->(page:Page)
            WHERE customer IN customers
            WITH community, customers, common_products, collect(DISTINCT page.id) AS common_pages
            
            // Get segments
            OPTIONAL MATCH (customer)-[:BELONGS_TO]->(s:Segment)
            WHERE customer IN customers AND NOT s.is_dynamic
            WITH community, customers, common_products, common_pages, 
                 collect(DISTINCT s.id) AS common_segments
            
            RETURN 
                community,
                size(customers) AS community_size,
                common_products,
                common_pages,
                common_segments,
                [customer IN customers | customer.customer_id] AS customer_ids
            ORDER BY community_size DESC
            """
            
            result = self.run_query(query)
            
            if not result:
                logging.error("No community data found")
                return False
            
            # Create segments for each community
            for community_data in result:
                community_id = community_data['community']
                community_size = community_data['community_size']
                customer_ids = community_data['customer_ids']
                
                # Generate a descriptive name based on common attributes
                common_products = community_data['common_products']
                common_pages = community_data['common_pages']
                common_segments = community_data['common_segments']
                
                segment_name = self._generate_community_name(
                    community_id, common_products, common_pages, common_segments
                )
                
                # Create segment in Neo4j
                create_segment_query = """
                CREATE (s:Segment {
                    id: $segment_id,
                    name: $segment_name,
                    is_dynamic: true,
                    is_community: true,
                    created_at: datetime(),
                    community_id: $community_id,
                    size: $size
                })
                RETURN s
                """
                
                self.run_query(create_segment_query, {
                    'segment_id': f"COMMUNITY_{community_id}",
                    'segment_name': segment_name,
                    'community_id': community_id,
                    'size': community_size
                })
                
                # Connect customers to segment
                batch_size = 100  # Process in batches to avoid large transactions
                for i in range(0, len(customer_ids), batch_size):
                    batch = customer_ids[i:i+batch_size]
                    
                    connect_query = """
                    UNWIND $customer_ids AS customer_id
                    MATCH (c:Customer {customer_id: customer_id})
                    MATCH (s:Segment {id: $segment_id})
                    MERGE (c)-[:BELONGS_TO {is_dynamic: true, is_community: true, assigned_at: datetime()}]->(s)
                    """
                    
                    self.run_query(connect_query, {
                        'customer_ids': batch,
                        'segment_id': f"COMMUNITY_{community_id}"
                    })
            
            logging.info(f"Created segments for {len(result)} detected communities")
            return True
            
        except Exception as e:
            logging.error(f"Error creating community segments: {e}")
            return False
    
    def _generate_community_name(self, community_id, products, pages, segments):
        """Generate a descriptive name for a detected community."""
        # Start with a base name
        name = f"Community {community_id}"
        
        # Add product-based description if available
        if products and len(products) <= 3:
            product_part = ', '.join(products[:3])
            name = f"{name} - Product Interest: {product_part}"
        
        # Add page-based description if available and no product info
        elif pages and len(pages) <= 3:
            page_part = ', '.join(pages[:3])
            name = f"{name} - Page Interest: {page_part}"
        
        # Add segment-based description if available
        elif segments and len(segments) <= 2:
            segment_part = ', '.join(segments[:2])
            name = f"{name} - Similar to: {segment_part}"
        
        return name
    
    def run_phase4_modeling(self, use_ensemble=True, use_advanced_anomaly=True):
        """
        Run all Phase 4 predictive modeling tasks with enhanced functionality.
        
        Parameters:
        - use_ensemble: If True, uses multi-model ensemble for churn prediction
        - use_advanced_anomaly: If True, uses advanced anomaly detection methods
        """
        results = {}
        
        # Connect to Neo4j
        if not self.connect():
            return {"status": "error", "message": "Failed to connect to Neo4j database"}
        
        try:
            # Train predictive models
            logging.info("Training churn prediction model...")
            if use_ensemble:
                logging.info("Using multi-model ensemble approach for churn prediction")
            churn_result = self.train_churn_prediction_model(use_ensemble=use_ensemble)
            results['churn_prediction'] = {
                "status": "success" if churn_result else "error",
                "ensemble_used": use_ensemble
            }
            
            logging.info("Training probabilistic CLV prediction model...")
            clv_result = self.train_clv_prediction_model()
            results['clv_prediction'] = {
                "status": "success" if clv_result else "error",
                "model_type": "probabilistic"
            }
            
            logging.info("Training next purchase prediction model...")
            next_purchase_result = self.train_next_purchase_model()
            results['next_purchase_prediction'] = {"status": "success" if next_purchase_result else "error"}
            
            # Run dynamic segmentation
            logging.info("Running dynamic customer segmentation...")
            segmentation_result = self.run_dynamic_customer_segmentation()
            results['dynamic_segmentation'] = {"status": "success" if segmentation_result else "error"}
            
            # Run community detection if GDS is available
            if self._check_gds_available():
                logging.info("Running community detection...")
                community_result = self.run_community_detection()
                results['community_detection'] = {"status": "success" if community_result else "error"}
            
            # Run advanced anomaly detection
            logging.info("Running anomaly detection...")
            if use_advanced_anomaly:
                logging.info("Using advanced anomaly detection methods (Isolation Forest)")
            anomaly_result = self.detect_anomalies(use_advanced_methods=use_advanced_anomaly)
            results['anomaly_detection'] = {
                "status": "success" if anomaly_result else "error",
                "advanced_methods": use_advanced_anomaly
            }
            
            # Get model performance statistics
            if 'churn_prediction' in self.model_metrics:
                churn_metrics = self.model_metrics['churn_prediction']
                results['churn_metrics'] = churn_metrics
                
                # Also include ensemble component metrics if available
                if use_ensemble:
                    ensemble_components = {}
                    for model_name in ['churn_rf', 'churn_gb', 'churn_lr', 'churn_ada']:
                        if model_name in self.model_metrics:
                            ensemble_components[model_name] = self.model_metrics[model_name]
                    
                    if ensemble_components:
                        results['ensemble_metrics'] = ensemble_components
            
            if 'clv_prediction' in self.model_metrics:
                clv_metrics = self.model_metrics['clv_prediction']
                results['clv_metrics'] = {
                    'r2': clv_metrics.get('r2', 0),
                    'rmse': clv_metrics.get('rmse', 0),
                    'avg_uncertainty': clv_metrics.get('avg_uncertainty', 0)
                }
            
            # Final status
            all_successful = all(
                r.get("status") == "success" if isinstance(r, dict) else False 
                for r in results.values() if isinstance(r, dict)
            )
            results['overall_status'] = "success" if all_successful else "partial_success"
            results['timestamp'] = datetime.now().isoformat()
            
            # Store overall modeling report in Neo4j
            try:
                report_query = """
                MERGE (r:ModelingReport {timestamp: datetime()})
                SET r.metrics = $metrics,
                    r.success = $success,
                    r.models_trained = $models_trained
                RETURN r
                """
                
                self.run_query(report_query, {
                    'metrics': json.dumps(results),
                    'success': all_successful,
                    'models_trained': list(results.keys())
                })
            except Exception as e:
                logging.error(f"Error storing modeling report: {e}")
            
            return results
        
        except Exception as e:
            logging.error(f"Error running Phase 4 modeling: {e}")
            return {"status": "error", "message": str(e)}
        
        finally:
            self.close()

if __name__ == "__main__":
    import argparse
    
    # Command-line argument parsing for flexible model configuration
    parser = argparse.ArgumentParser(description='Run Phase 4 predictive models')
    parser.add_argument('--no-ensemble', action='store_true', help='Disable multi-model ensemble for churn prediction')
    parser.add_argument('--no-advanced-anomaly', action='store_true', help='Disable advanced anomaly detection methods')
    parser.add_argument('--uri', type=str, help='Neo4j URI (default: from env or bolt://localhost:7687)')
    parser.add_argument('--username', type=str, help='Neo4j username (default: from env or neo4j)')
    parser.add_argument('--password', type=str, help='Neo4j password (default: from env)')
    parser.add_argument('--database', type=str, help='Neo4j database name (default: from env or neo4j)')
    
    args = parser.parse_args()
    
    print("Starting Phase 4 Predictive Models with Advanced Functionality...")
    
    # Initialize with optional parameters
    predictor = PredictiveModels(
        uri=args.uri,
        username=args.username,
        password=args.password,
        database=args.database
    )
    
    # Run with command-line configuration
    results = predictor.run_phase4_modeling(
        use_ensemble=not args.no_ensemble,
        use_advanced_anomaly=not args.no_advanced_anomaly
    )
    
    print(f"\nPhase 4 modeling completed with status: {results.get('overall_status', 'unknown')}")
    print("\nResults:")
    
    # Detailed result reporting
    for model, result in results.items():
        if model == 'overall_status' or model == 'timestamp' or not isinstance(result, dict):
            continue
            
        status = result.get('status', 'unknown')
        status_symbol = "✅" if status == "success" else "❌"
        
        # Print basic status
        print(f"  {status_symbol} {model}: {status}")
        
        # Print extra details if available
        if model == 'churn_prediction' and result.get('ensemble_used'):
            print(f"    - Using multi-model ensemble approach")
            
        if model == 'anomaly_detection' and result.get('advanced_methods'):
            print(f"    - Using advanced anomaly detection methods")
            
    # Print metrics if available
    if 'churn_metrics' in results:
        print("\nChurn Prediction Metrics:")
        metrics = results['churn_metrics']
        for metric, value in metrics.items():
            if isinstance(value, (int, float)):
                print(f"  - {metric}: {value:.4f}")
    
    if 'clv_metrics' in results:
        print("\nCLV Prediction Metrics:")
        metrics = results['clv_metrics']
        for metric, value in metrics.items():
            if isinstance(value, (int, float)):
                print(f"  - {metric}: {value:.4f}")
                
    print("\nModeling complete. Results stored in Neo4j database.")