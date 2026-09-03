"""
Feature Store Architecture Module
Implements feature store for managing and serving ML features.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta
import json
from pathlib import Path
from config.logging_config import get_logger

logger = get_logger(__name__)


class FeatureStore:
    """
    Feature store for managing and serving ML features.
    
    Features:
    - Feature registration and versioning
    - Feature computation and storage
    - Feature retrieval for training and inference
    - Feature lineage tracking
    - Time-travel queries (historical feature values)
    """
    
    def __init__(self, store_path: str = "data/feature_store"):
        """
        Initialize feature store.
        
        Args:
            store_path: Path to feature store directory
        """
        self.store_path = Path(store_path)
        self.store_path.mkdir(parents=True, exist_ok=True)
        
        self.feature_registry = {}
        self.feature_versions = {}
        
        # Load existing registry if available
        registry_path = self.store_path / "registry.json"
        if registry_path.exists():
            with open(registry_path, 'r') as f:
                self.feature_registry = json.load(f)
        
        logger.info(f"Feature store initialized at {store_path}")
    
    def register_feature(
        self,
        feature_name: str,
        feature_type: str,
        description: str,
        source_table: str,
        computation_logic: str = None
    ) -> Dict:
        """
        Register a new feature in the feature store.
        
        Args:
            feature_name: Name of the feature
            feature_type: Type of feature (numerical, categorical, etc.)
            description: Feature description
            source_table: Source table for the feature
            computation_logic: Computation logic (optional)
        
        Returns:
            Dictionary with registration result
        """
        logger.info(f"Registering feature: {feature_name}")
        
        feature_info = {
            'name': feature_name,
            'type': feature_type,
            'description': description,
            'source_table': source_table,
            'computation_logic': computation_logic,
            'created_at': datetime.now().isoformat(),
            'version': 1
        }
        
        self.feature_registry[feature_name] = feature_info
        
        # Save registry
        self._save_registry()
        
        logger.info(f"Feature {feature_name} registered successfully")
        
        return feature_info
    
    def compute_and_store_feature(
        self,
        feature_name: str,
        data: pd.DataFrame,
        entity_id_col: str = 'customer_id',
        timestamp_col: str = 'event_date'
    ) -> Dict:
        """
        Compute and store feature values.
        
        Args:
            feature_name: Name of the feature
            data: DataFrame with feature values
            entity_id_col: Entity ID column name
            timestamp_col: Timestamp column name
        
        Returns:
            Dictionary with storage result
        """
        logger.info(f"Computing and storing feature: {feature_name}")
        
        if feature_name not in self.feature_registry:
            raise ValueError(f"Feature {feature_name} not registered")
        
        # Create feature directory
        feature_dir = self.store_path / feature_name
        feature_dir.mkdir(exist_ok=True)
        
        # Store feature data
        feature_file = feature_dir / f"v{self.feature_registry[feature_name]['version']}.parquet"
        data.to_parquet(feature_file, index=False)
        
        # Update metadata
        self.feature_registry[feature_name]['last_updated'] = datetime.now().isoformat()
        self.feature_registry[feature_name]['entity_id_col'] = entity_id_col
        self.feature_registry[feature_name]['timestamp_col'] = timestamp_col
        self.feature_registry[feature_name]['n_records'] = len(data)
        
        # Save registry
        self._save_registry()
        
        logger.info(f"Feature {feature_name} stored with {len(data)} records")
        
        return {
            'feature_name': feature_name,
            'version': self.feature_registry[feature_name]['version'],
            'n_records': len(data),
            'storage_path': str(feature_file)
        }
    
    def get_feature(
        self,
        feature_name: str,
        entity_ids: List[str] = None,
        version: int = None,
        as_of: datetime = None
    ) -> pd.DataFrame:
        """
        Retrieve feature values.
        
        Args:
            feature_name: Name of the feature
            entity_ids: List of entity IDs to retrieve (optional)
            version: Feature version (optional)
            as_of: As-of timestamp for time-travel (optional)
        
        Returns:
            DataFrame with feature values
        """
        logger.info(f"Retrieving feature: {feature_name}")
        
        if feature_name not in self.feature_registry:
            raise ValueError(f"Feature {feature_name} not registered")
        
        # Get version
        if version is None:
            version = self.feature_registry[feature_name]['version']
        
        # Load feature data
        feature_dir = self.store_path / feature_name
        feature_file = feature_dir / f"v{version}.parquet"
        
        if not feature_file.exists():
            raise ValueError(f"Feature version {version} not found")
        
        data = pd.read_parquet(feature_file)
        
        # Filter by entity IDs if provided
        if entity_ids is not None:
            entity_id_col = self.feature_registry[feature_name].get('entity_id_col', 'entity_id')
            if entity_id_col in data.columns:
                data = data[data[entity_id_col].isin(entity_ids)]
        
        # Time-travel query if as-of timestamp provided
        if as_of is not None:
            timestamp_col = self.feature_registry[feature_name].get('timestamp_col', 'timestamp')
            if timestamp_col in data.columns:
                data[timestamp_col] = pd.to_datetime(data[timestamp_col])
                data = data[data[timestamp_col] <= as_of]
                # Get latest value per entity
                entity_id_col = self.feature_registry[feature_name].get('entity_id_col', 'entity_id')
                if entity_id_col in data.columns:
                    data = data.sort_values(timestamp_col).groupby(entity_id_col).tail(1)
        
        logger.info(f"Retrieved {len(data)} feature values")
        
        return data
    
    def get_feature_set(
        self,
        feature_names: List[str],
        entity_ids: List[str] = None,
        join_on: str = 'customer_id'
    ) -> pd.DataFrame:
        """
        Retrieve multiple features as a feature set.
        
        Args:
            feature_names: List of feature names
            entity_ids: List of entity IDs to retrieve (optional)
            join_on: Column to join features on
        
        Returns:
            DataFrame with feature set
        """
        logger.info(f"Retrieving feature set with {len(feature_names)} features")
        
        feature_data = {}
        
        for feature_name in feature_names:
            try:
                feature_df = self.get_feature(feature_name, entity_ids)
                feature_data[feature_name] = feature_df
            except ValueError as e:
                logger.warning(f"Could not retrieve feature {feature_name}: {e}")
        
        if not feature_data:
            raise ValueError("No features could be retrieved")
        
        # Join features
        result_df = None
        for feature_name, df in feature_data.items():
            if result_df is None:
                result_df = df
            else:
                result_df = result_df.merge(df, on=join_on, how='outer', suffixes=('', f'_{feature_name}'))
        
        logger.info(f"Retrieved feature set with {len(result_df)} records")
        
        return result_df
    
    def create_training_dataset(
        self,
        feature_names: List[str],
        target_df: pd.DataFrame,
        target_col: str,
        entity_id_col: str = 'customer_id'
    ) -> pd.DataFrame:
        """
        Create training dataset from features and target.
        
        Args:
            feature_names: List of feature names
            target_df: DataFrame with target values
            target_col: Target column name
            entity_id_col: Entity ID column name
        
        Returns:
            DataFrame with training dataset
        """
        logger.info(f"Creating training dataset with {len(feature_names)} features")
        
        # Get feature set
        entity_ids = target_df[entity_id_col].unique().tolist()
        feature_df = self.get_feature_set(feature_names, entity_ids, entity_id_col)
        
        # Merge with target
        training_df = feature_df.merge(target_df, on=entity_id_col, how='inner')
        
        logger.info(f"Created training dataset with {len(training_df)} records")
        
        return training_df
    
    def get_feature_lineage(self, feature_name: str) -> Dict:
        """
        Get lineage information for a feature.
        
        Args:
            feature_name: Name of the feature
        
        Returns:
            Dictionary with lineage information
        """
        logger.info(f"Getting lineage for feature: {feature_name}")
        
        if feature_name not in self.feature_registry:
            raise ValueError(f"Feature {feature_name} not registered")
        
        feature_info = self.feature_registry[feature_name]
        
        lineage = {
            'feature_name': feature_name,
            'source_table': feature_info.get('source_table'),
            'computation_logic': feature_info.get('computation_logic'),
            'created_at': feature_info.get('created_at'),
            'last_updated': feature_info.get('last_updated'),
            'version': feature_info.get('version'),
            'n_records': feature_info.get('n_records')
        }
        
        return lineage
    
    def list_features(self) -> List[Dict]:
        """
        List all registered features.
        
        Returns:
            List of feature information dictionaries
        """
        features = []
        
        for feature_name, feature_info in self.feature_registry.items():
            features.append({
                'name': feature_name,
                'type': feature_info.get('type'),
                'description': feature_info.get('description'),
                'version': feature_info.get('version'),
                'last_updated': feature_info.get('last_updated')
            })
        
        return features
    
    def _save_registry(self):
        """Save feature registry to disk."""
        registry_path = self.store_path / "registry.json"
        with open(registry_path, 'w') as f:
            json.dump(self.feature_registry, f, indent=2)


def run_feature_store_pipeline(
    customers_df: pd.DataFrame,
    orders_df: pd.DataFrame
) -> Tuple[FeatureStore, Dict]:
    """
    Convenience function to run feature store pipeline.
    
    Args:
        customers_df: Customer data
        orders_df: Order data
    
    Returns:
        Tuple of (feature_store, results)
    """
    store = FeatureStore()
    
    # Register customer features
    store.register_feature(
        'customer_total_spend',
        'numerical',
        'Total spend per customer',
        'orders',
        'SUM(order_total) GROUP BY customer_id'
    )
    
    store.register_feature(
        'customer_order_count',
        'numerical',
        'Number of orders per customer',
        'orders',
        'COUNT(*) GROUP BY customer_id'
    )
    
    store.register_feature(
        'customer_avg_order_value',
        'numerical',
        'Average order value per customer',
        'orders',
        'AVG(order_total) GROUP BY customer_id'
    )
    
    # Compute and store features
    customer_spend = orders_df.groupby('customer_id')['order_total'].sum().reset_index()
    customer_spend.columns = ['customer_id', 'customer_total_spend']
    store.compute_and_store_feature('customer_total_spend', customer_spend)
    
    order_count = orders_df.groupby('customer_id').size().reset_index()
    order_count.columns = ['customer_id', 'customer_order_count']
    store.compute_and_store_feature('customer_order_count', order_count)
    
    avg_order = orders_df.groupby('customer_id')['order_total'].mean().reset_index()
    avg_order.columns = ['customer_id', 'customer_avg_order_value']
    store.compute_and_store_feature('customer_avg_order_value', avg_order)
    
    # List features
    features = store.list_features()
    
    results = {
        'registered_features': features,
        'n_features': len(features)
    }
    
    return store, results
