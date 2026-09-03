"""
Churn Feature Engineering with Temporal Cutoff
Prevents target leakage by using proper observation point methodology.

Architecture:
- Historical behavior window (features)
- Observation point (T0)
- Future window (target)
- Feature timestamp <= Prediction timestamp
"""

from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from pathlib import Path

from config.logging_config import get_logger
from config.settings import settings

logger = get_logger(__name__)


class ChurnFeatureEngineer:
    """
    Churn feature engineering with proper temporal separation.
    
    Prevents leakage by ensuring:
    1. All features are computed from data BEFORE the observation point
    2. Target is computed from data AFTER the observation point
    3. No future information leaks into features
    """
    
    def __init__(
        self,
        observation_date: str = "2024-09-30",
        feature_window_days: int = 180,
        target_window_days: int = 90
    ):
        """
        Initialize churn feature engineer.
        
        Args:
            observation_date: The cutoff date (T0) for feature/target separation
            feature_window_days: How many days before T0 to use for features
            target_window_days: How many days after T0 to use for target
        """
        self.observation_date = pd.to_datetime(observation_date)
        self.feature_window_start = self.observation_date - timedelta(days=feature_window_days)
        self.target_window_end = self.observation_date + timedelta(days=target_window_days)
        
        logger.info(f"Churn Feature Engineer initialized:")
        logger.info(f"  Observation date (T0): {self.observation_date}")
        logger.info(f"  Feature window: {self.feature_window_start} to {self.observation_date}")
        logger.info(f"  Target window: {self.observation_date} to {self.target_window_end}")
    
    def compute_features(
        self,
        orders_df: pd.DataFrame,
        customers_df: pd.DataFrame,
        sessions_df: Optional[pd.DataFrame] = None,
        reviews_df: Optional[pd.DataFrame] = None
    ) -> pd.DataFrame:
        """
        Compute churn features using only data BEFORE observation date.
        
        Args:
            orders_df: Orders data
            customers_df: Customers data
            sessions_df: Optional website sessions data
            reviews_df: Optional reviews data
        
        Returns:
            DataFrame with one row per customer and feature columns
        """
        logger.info("Computing churn features...")
        
        # Filter orders to feature window only
        orders_before = orders_df[
            (orders_df['order_date'] >= self.feature_window_start) &
            (orders_df['order_date'] < self.observation_date)
        ].copy()
        
        # Customer-level features from orders
        customer_features = self._compute_order_features(orders_before)
        
        # Merge with customer demographics
        features = customers_df.merge(
            customer_features,
            on='customer_id',
            how='left'
        )
        
        # Add session features if available
        if sessions_df is not None:
            sessions_before = sessions_df[
                (sessions_df['session_date'] >= self.feature_window_start) &
                (sessions_df['session_date'] < self.observation_date)
            ].copy()
            session_features = self._compute_session_features(sessions_before)
            features = features.merge(session_features, on='customer_id', how='left')
        
        # Add review features if available
        if reviews_df is not None:
            reviews_before = reviews_df[
                (reviews_df['review_date'] >= self.feature_window_start) &
                (reviews_df['review_date'] < self.observation_date)
            ].copy()
            review_features = self._compute_review_features(reviews_before)
            features = features.merge(review_features, on='customer_id', how='left')
        
        # Compute days since last order (as of observation date)
        features['days_since_last_order'] = self._compute_days_since_last_order(
            orders_before, features['customer_id']
        )
        
        # Compute customer tenure as of observation date
        features['customer_tenure_days'] = (
            self.observation_date - pd.to_datetime(features['signup_date'])
        ).dt.days
        
        # Fill missing values
        features = self._fill_missing_features(features)
        
        logger.info(f"Computed features for {len(features)} customers")
        return features
    
    def _compute_order_features(self, orders_df: pd.DataFrame) -> pd.DataFrame:
        """Compute order-based features."""
        if orders_df.empty:
            return pd.DataFrame(columns=['customer_id', 'total_orders', 'avg_order_value', 
                                        'total_spend', 'return_rate', 'avg_discount_amount'])
        
        features = orders_df.groupby('customer_id').agg({
            'order_id': 'nunique',
            'order_total': ['sum', 'mean'],
            'discount_amount': 'mean',
            'order_status': lambda x: (x == 'Returned').sum() / len(x) if len(x) > 0 else 0
        }).reset_index()
        
        features.columns = ['customer_id', 'total_orders', 'total_spend', 
                           'avg_order_value', 'avg_discount_amount', 'return_rate']
        
        return features
    
    def _compute_session_features(self, sessions_df: pd.DataFrame) -> pd.DataFrame:
        """Compute session-based features."""
        if sessions_df.empty:
            return pd.DataFrame(columns=['customer_id', 'total_sessions', 
                                        'avg_page_views', 'avg_session_sec', 'checkout_rate'])
        
        features = sessions_df.groupby('customer_id').agg({
            'session_id': 'count',
            'page_views': 'mean',
            'session_duration_sec': 'mean',
            'checkout_completed': 'mean'
        }).reset_index()
        
        features.columns = ['customer_id', 'total_sessions', 'avg_page_views',
                           'avg_session_sec', 'checkout_rate']
        
        return features
    
    def _compute_review_features(self, reviews_df: pd.DataFrame) -> pd.DataFrame:
        """Compute review-based features."""
        if reviews_df.empty:
            return pd.DataFrame(columns=['customer_id', 'avg_review_rating', 'total_reviews'])
        
        features = reviews_df.groupby('customer_id').agg({
            'rating': ['mean', 'count']
        }).reset_index()
        
        features.columns = ['customer_id', 'avg_review_rating', 'total_reviews']
        
        return features
    
    def _compute_days_since_last_order(
        self, 
        orders_df: pd.DataFrame, 
        customer_ids: pd.Series
    ) -> pd.Series:
        """Compute days since last order as of observation date."""
        if orders_df.empty:
            return pd.Series([np.nan] * len(customer_ids))
        
        last_orders = orders_df.groupby('customer_id')['order_date'].max().reset_index()
        last_orders['days_since_last_order'] = (
            self.observation_date - pd.to_datetime(last_orders['order_date'])
        ).dt.days
        
        result = pd.DataFrame({'customer_id': customer_ids})
        result = result.merge(
            last_orders[['customer_id', 'days_since_last_order']],
            on='customer_id',
            how='left'
        )
        
        return result['days_since_last_order']
    
    def _fill_missing_features(self, features_df: pd.DataFrame) -> pd.DataFrame:
        """Fill missing feature values with appropriate defaults."""
        # Numeric features - fill with 0
        numeric_cols = ['total_orders', 'avg_order_value', 'total_spend', 
                       'return_rate', 'avg_discount_amount', 'total_sessions',
                       'avg_page_views', 'avg_session_sec', 'checkout_rate',
                       'avg_review_rating', 'total_reviews']
        
        for col in numeric_cols:
            if col in features_df.columns:
                features_df[col] = features_df[col].fillna(0)
        
        # Days since last order - fill with large value (inactive customers)
        if 'days_since_last_order' in features_df.columns:
            features_df['days_since_last_order'] = features_df['days_since_last_order'].fillna(999)
        
        return features_df
    
    def compute_target(
        self,
        orders_df: pd.DataFrame,
        customers_df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Compute churn target using only data AFTER observation date.
        
        Churn definition: Customer did NOT place any order in the target window
        AND had at least one order before the observation date.
        
        Args:
            orders_df: Orders data
            customers_df: Customers data
        
        Returns:
            DataFrame with customer_id and churn_label_90d
        """
        logger.info("Computing churn target...")
        
        # Get orders in target window
        orders_after = orders_df[
            (orders_df['order_date'] >= self.observation_date) &
            (orders_df['order_date'] <= self.target_window_end)
        ].copy()
        
        # Get customers who had orders before observation date
        orders_before = orders_df[orders_df['order_date'] < self.observation_date].copy()
        customers_with_orders = set(orders_before['customer_id'].unique())
        
        # Identify who churned (no orders in target window)
        customers_with_orders_after = set(orders_after['customer_id'].unique())
        
        # Create target dataframe
        target_df = pd.DataFrame({'customer_id': customers_df['customer_id']})
        
        # Churn = had orders before, but no orders in target window
        target_df['churn_label_90d'] = target_df['customer_id'].apply(
            lambda x: 1 if (x in customers_with_orders and x not in customers_with_orders_after) else 0
        )
        
        logger.info(f"Churn rate: {target_df['churn_label_90d'].mean():.2%}")
        return target_df
    
    def create_churn_dataset(
        self,
        orders_df: pd.DataFrame,
        customers_df: pd.DataFrame,
        sessions_df: Optional[pd.DataFrame] = None,
        reviews_df: Optional[pd.DataFrame] = None
    ) -> pd.DataFrame:
        """
        Create complete churn dataset with features and target.
        
        Args:
            orders_df: Orders data
            customers_df: Customers data
            sessions_df: Optional website sessions data
            reviews_df: Optional reviews data
        
        Returns:
            DataFrame with features and target, ready for ML
        """
        logger.info("Creating churn dataset with temporal cutoff...")
        
        # Compute features (before T0)
        features = self.compute_features(orders_df, customers_df, sessions_df, reviews_df)
        
        # Compute target (after T0)
        target = self.compute_target(orders_df, customers_df)
        
        # Merge features and target
        dataset = features.merge(target, on='customer_id', how='inner')
        
        # Filter to customers with at least one order (meaningful churn prediction)
        dataset = dataset[dataset['total_orders'] >= 1].copy()
        
        logger.info(f"Final dataset: {len(dataset)} customers, {dataset.shape[1]} columns")
        logger.info(f"Churn rate: {dataset['churn_label_90d'].mean():.2%}")
        
        return dataset
    
    def create_temporal_splits(
        self,
        dataset: pd.DataFrame,
        train_ratio: float = 0.6,
        val_ratio: float = 0.2
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Create temporal train/validation/test splits.
        
        Splits are based on customer signup date to prevent leakage.
        Earliest customers go to train, latest to test.
        
        Args:
            dataset: Churn dataset with signup_date
            train_ratio: Proportion for training
            val_ratio: Proportion for validation
        
        Returns:
            Tuple of (train_df, val_df, test_df)
        """
        logger.info("Creating temporal splits...")
        
        # Sort by signup date
        dataset_sorted = dataset.sort_values('signup_date').copy()
        
        n = len(dataset_sorted)
        train_end = int(n * train_ratio)
        val_end = int(n * (train_ratio + val_ratio))
        
        train_df = dataset_sorted.iloc[:train_end].copy()
        val_df = dataset_sorted.iloc[train_end:val_end].copy()
        test_df = dataset_sorted.iloc[val_end:].copy()
        
        logger.info(f"Train: {len(train_df)} ({len(train_df)/n:.1%})")
        logger.info(f"Val: {len(val_df)} ({len(val_df)/n:.1%})")
        logger.info(f"Test: {len(test_df)} ({len(test_df)/n:.1%})")
        
        logger.info(f"Train churn rate: {train_df['churn_label_90d'].mean():.2%}")
        logger.info(f"Val churn rate: {val_df['churn_label_90d'].mean():.2%}")
        logger.info(f"Test churn rate: {test_df['churn_label_90d'].mean():.2%}")
        
        return train_df, val_df, test_df


def run_churn_feature_engineering_pipeline(
    observation_date: str = "2024-09-30",
    feature_window_days: int = 180,
    target_window_days: int = 90
) -> pd.DataFrame:
    """
    Convenience function to run the complete churn feature engineering pipeline.
    
    Args:
        observation_date: The cutoff date for feature/target separation
        feature_window_days: Days before observation for features
        target_window_days: Days after observation for target
    
    Returns:
        Churn dataset ready for ML
    """
    engineer = ChurnFeatureEngineer(
        observation_date=observation_date,
        feature_window_days=feature_window_days,
        target_window_days=target_window_days
    )
    
    # Load data (placeholder - implement actual data loading)
    # orders_df = pd.read_csv(settings.PROCESSED_DATA_DIR / "orders.csv")
    # customers_df = pd.read_csv(settings.PROCESSED_DATA_DIR / "customers.csv")
    # sessions_df = pd.read_csv(settings.PROCESSED_DATA_DIR / "website_sessions.csv")
    # reviews_df = pd.read_csv(settings.PROCESSED_DATA_DIR / "reviews.csv")
    
    # dataset = engineer.create_churn_dataset(orders_df, customers_df, sessions_df, reviews_df)
    
    # For now, return empty dataframe with expected structure
    logger.warning("Data loading not implemented - returning empty dataset")
    return pd.DataFrame()
