"""
Learning to Rank Module
Implements ranking models for product recommendations using Learning to Rank techniques.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split, GroupKFold
from sklearn.metrics import ndcg_score
import joblib
from pathlib import Path
from config.logging_config import get_logger

logger = get_logger(__name__)


class LearningToRankRecommender:
    """
    Learning to Rank recommender for product recommendations.
    
    Uses ranking models to optimize the order of recommendations:
    - Pointwise: Predict relevance score for each item
    - Pairwise: Predict relative ordering between item pairs
    - Listwise: Optimize entire ranking list
    
    Features:
    - Customer-item interaction features
    - Customer features (RFM, behavioral)
    - Product features (category, price, popularity)
    - Contextual features (time, season)
    """
    
    def __init__(self, ranking_method: str = 'pointwise'):
        """
        Initialize Learning to Rank recommender.
        
        Args:
            ranking_method: Ranking method ('pointwise', 'pairwise', 'listwise')
        """
        self.ranking_method = ranking_method
        self.model = None
        self.feature_names = []
    
    def prepare_ranking_features(
        self,
        customers_df: pd.DataFrame,
        products_df: pd.DataFrame,
        orders_df: pd.DataFrame,
        rfm_df: pd.DataFrame,
        behavioral_features_df: pd.DataFrame = None
    ) -> pd.DataFrame:
        """
        Prepare features for learning to rank.
        
        Args:
            customers_df: Customer data
            products_df: Product data
            orders_df: Order data
            rfm_df: RFM segmentation data
            behavioral_features_df: Behavioral features (optional)
        
        Returns:
            DataFrame with ranking features and relevance labels
        """
        logger.info("Preparing ranking features...")
        
        # Calculate relevance (purchase frequency as proxy for relevance)
        item_interactions = orders_df.groupby(['customer_id', 'product_id']).size().reset_index()
        item_interactions.columns = ['customer_id', 'product_id', 'interaction_count']
        
        # Normalize relevance to 0-1
        max_interactions = item_interactions['interaction_count'].max()
        item_interactions['relevance'] = item_interactions['interaction_count'] / max_interactions
        
        # Create customer-item pairs for all customers and products
        all_customers = customers_df['customer_id'].unique()
        all_products = products_df['product_id'].unique()
        
        # Sample negative examples (customer-item pairs with no interaction)
        existing_pairs = set(zip(item_interactions['customer_id'], item_interactions['product_id']))
        
        negative_samples = []
        n_negatives = len(item_interactions)  # Balance positive and negative
        
        for _ in range(n_negatives):
            customer_id = np.random.choice(all_customers)
            product_id = np.random.choice(all_products)
            
            if (customer_id, product_id) not in existing_pairs:
                negative_samples.append({
                    'customer_id': customer_id,
                    'product_id': product_id,
                    'interaction_count': 0,
                    'relevance': 0
                })
        
        negative_df = pd.DataFrame(negative_samples)
        
        # Combine positive and negative samples
        ranking_data = pd.concat([item_interactions, negative_df], ignore_index=True)
        
        # Add customer features
        ranking_data = ranking_data.merge(rfm_df, on='customer_id', how='left')
        
        # Add product features
        ranking_data = ranking_data.merge(
            products_df[['product_id', 'category_id', 'selling_price', 'brand_name']],
            on='product_id',
            how='left'
        )
        
        # Calculate product popularity
        product_popularity = orders_df.groupby('product_id').size()
        ranking_data['product_popularity'] = ranking_data['product_id'].map(product_popularity).fillna(0)
        
        # Add behavioral features if available
        if behavioral_features_df is not None:
            behavioral_cols = [
                'purchase_frequency', 'purchase_acceleration', 'avg_basket_value',
                'discount_dependency', 'unique_categories', 'unique_brands'
            ]
            available_behavioral = [col for col in behavioral_cols if col in behavioral_features_df.columns]
            
            if available_behavioral:
                ranking_data = ranking_data.merge(
                    behavioral_features_df[['customer_id'] + available_behavioral],
                    on='customer_id',
                    how='left'
                )
        
        # Fill missing values
        ranking_data = ranking_data.fillna(0)
        
        logger.info(f"Prepared {len(ranking_data)} ranking samples")
        
        return ranking_data
    
    def train_pointwise_model(
        self,
        ranking_data: pd.DataFrame,
        feature_cols: List[str],
        model_type: str = 'gradient_boosting'
    ) -> Dict:
        """
        Train pointwise ranking model (regression on relevance scores).
        
        Args:
            ranking_data: DataFrame with features and relevance
            feature_cols: List of feature column names
            model_type: Model type ('gradient_boosting', 'random_forest', 'ridge')
        
        Returns:
            Dictionary with training results
        """
        logger.info(f"Training pointwise ranking model with {model_type}...")
        
        # Prepare data
        X = ranking_data[feature_cols].values
        y = ranking_data['relevance'].values
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Initialize model
        if model_type == 'gradient_boosting':
            self.model = GradientBoostingRegressor(
                n_estimators=100,
                max_depth=5,
                learning_rate=0.1,
                random_state=42
            )
        elif model_type == 'random_forest':
            self.model = RandomForestRegressor(
                n_estimators=100,
                max_depth=10,
                random_state=42
            )
        elif model_type == 'ridge':
            self.model = Ridge(alpha=1.0, random_state=42)
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        # Train model
        self.model.fit(X_train, y_train)
        
        # Predictions
        y_train_pred = self.model.predict(X_train)
        y_test_pred = self.model.predict(X_test)
        
        # Metrics
        train_mse = np.mean((y_train - y_train_pred) ** 2)
        test_mse = np.mean((y_test - y_test_pred) ** 2)
        
        # Calculate NDCG (need to group by customer)
        ranking_data_with_pred = ranking_data.copy()
        ranking_data_with_pred['predicted_score'] = self.model.predict(ranking_data[feature_cols].values)
        
        # Calculate NDCG@k for each customer
        ndcg_scores = []
        for customer_id in ranking_data_with_pred['customer_id'].unique():
            customer_data = ranking_data_with_pred[ranking_data_with_pred['customer_id'] == customer_id]
            
            if len(customer_data) > 1:
                true_relevance = customer_data['relevance'].values.reshape(1, -1)
                predicted_scores = customer_data['predicted_score'].values.reshape(1, -1)
                
                try:
                    ndcg = ndcg_score(true_relevance, predicted_scores, k=10)
                    ndcg_scores.append(ndcg)
                except:
                    pass
        
        avg_ndcg = np.mean(ndcg_scores) if ndcg_scores else 0
        
        self.feature_names = feature_cols
        
        results = {
            'model_type': model_type,
            'train_mse': float(train_mse),
            'test_mse': float(test_mse),
            'avg_ndcg@10': float(avg_ndcg),
            'n_features': len(feature_cols),
            'n_samples': len(ranking_data)
        }
        
        logger.info(f"Pointwise model trained. NDCG@10: {avg_ndcg:.3f}")
        
        return results
    
    def predict_ranking(
        self,
        customer_id: int,
        candidate_products: pd.DataFrame,
        customer_features: pd.DataFrame,
        product_features: pd.DataFrame,
        top_k: int = 10
    ) -> pd.DataFrame:
        """
        Predict ranking for a customer's candidate products.
        
        Args:
            customer_id: Customer ID
            candidate_products: DataFrame with candidate product IDs
            customer_features: Customer features
            product_features: Product features
            top_k: Number of top recommendations to return
        
        Returns:
            DataFrame with ranked recommendations
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train_pointwise_model first.")
        
        # Prepare features for each candidate product
        recommendations = []
        
        for _, product_row in candidate_products.iterrows():
            product_id = product_row['product_id']
            
            # Create feature vector
            feature_vector = {}
            
            # Add customer features
            for col in self.feature_names:
                if col in customer_features.columns:
                    feature_vector[col] = customer_features[col].values[0]
                elif col in product_features.columns:
                    feature_vector[col] = product_row[col]
                else:
                    feature_vector[col] = 0
            
            # Predict score
            X = np.array([list(feature_vector.values())])
            score = self.model.predict(X)[0]
            
            recommendations.append({
                'customer_id': customer_id,
                'product_id': product_id,
                'score': score
            })
        
        # Rank by score
        recommendations_df = pd.DataFrame(recommendations)
        recommendations_df = recommendations_df.sort_values('score', ascending=False).head(top_k)
        
        # Add rank
        recommendations_df['rank'] = range(1, len(recommendations_df) + 1)
        
        return recommendations_df
    
    def get_feature_importance(self, top_n: int = 10) -> pd.DataFrame:
        """
        Get feature importance from the ranking model.
        
        Args:
            top_n: Number of top features to return
        
        Returns:
            DataFrame with feature importance
        """
        if self.model is None:
            raise ValueError("Model not trained")
        
        if hasattr(self.model, 'feature_importances_'):
            importance = dict(zip(self.feature_names, self.model.feature_importances_))
        elif hasattr(self.model, 'coef_'):
            importance = dict(zip(self.feature_names, np.abs(self.model.coef_)))
        else:
            raise ValueError("Model does not support feature importance")
        
        importance_df = pd.DataFrame([
            {'feature': k, 'importance': v}
            for k, v in importance.items()
        ])
        
        importance_df = importance_df.sort_values('importance', ascending=False).head(top_n)
        
        return importance_df
    
    def save_model(self, save_path: str = "models/ranking_model.joblib"):
        """
        Save trained ranking model.
        
        Args:
            save_path: Path to save model
        """
        if self.model is None:
            raise ValueError("No model to save")
        
        model_data = {
            'model': self.model,
            'feature_names': self.feature_names,
            'ranking_method': self.ranking_method
        }
        
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(model_data, save_path)
        
        logger.info(f"Model saved to {save_path}")
    
    def load_model(self, load_path: str = "models/ranking_model.joblib"):
        """
        Load trained ranking model.
        
        Args:
            load_path: Path to load model from
        """
        model_data = joblib.load(load_path)
        
        self.model = model_data['model']
        self.feature_names = model_data['feature_names']
        self.ranking_method = model_data['ranking_method']
        
        logger.info(f"Model loaded from {load_path}")


def run_learning_to_rank_pipeline(
    customers_df: pd.DataFrame,
    products_df: pd.DataFrame,
    orders_df: pd.DataFrame,
    rfm_df: pd.DataFrame,
    behavioral_features_df: pd.DataFrame = None,
    model_type: str = 'gradient_boosting'
) -> Tuple[LearningToRankRecommender, Dict]:
    """
    Convenience function to run complete learning to rank pipeline.
    
    Args:
        customers_df: Customer data
        products_df: Product data
        orders_df: Order data
        rfm_df: RFM segmentation data
        behavioral_features_df: Behavioral features (optional)
        model_type: Model type for ranking
    
    Returns:
        Tuple of (trained recommender, training results)
    """
    recommender = LearningToRankRecommender(ranking_method='pointwise')
    
    # Prepare features
    ranking_data = recommender.prepare_ranking_features(
        customers_df, products_df, orders_df, rfm_df, behavioral_features_df
    )
    
    # Get feature columns
    exclude_cols = ['customer_id', 'product_id', 'interaction_count', 'relevance']
    feature_cols = [col for col in ranking_data.columns if col not in exclude_cols]
    
    # Train model
    results = recommender.train_pointwise_model(ranking_data, feature_cols, model_type)
    
    return recommender, results
