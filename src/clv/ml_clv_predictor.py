"""
ML-based CLV Predictor
Implements machine learning models for Customer Lifetime Value prediction to complement BG/NBD + Gamma-Gamma.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge, Lasso
from sklearn.model_selection import train_test_split, cross_val_score, TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
from pathlib import Path
from config.logging_config import get_logger

logger = get_logger(__name__)


class MLCLVPredictor:
    """
    Machine Learning-based CLV prediction model.
    
    Complements the probabilistic BG/NBD + Gamma-Gamma model with:
    - Feature-based ML approach
    - Multiple model types (Gradient Boosting, Random Forest, Ridge, Lasso)
    - Multi-horizon predictions (90d, 180d, 365d)
    - Model comparison and selection
    - Feature importance analysis
    """
    
    def __init__(self, analysis_as_of_date: str = "2024-12-31"):
        """
        Initialize ML CLV predictor.
        
        Args:
            analysis_as_of_date: Date for analysis cutoff
        """
        self.analysis_as_of_date = pd.to_datetime(analysis_as_of_date)
        self.models = {}
        self.scalers = {}
        self.feature_importance = {}
    
    def prepare_clv_features(
        self,
        customers_df: pd.DataFrame,
        orders_df: pd.DataFrame,
        order_items_df: pd.DataFrame,
        rfm_df: pd.DataFrame,
        behavioral_features_df: pd.DataFrame = None
    ) -> pd.DataFrame:
        """
        Prepare features for ML-based CLV prediction.
        
        Args:
            customers_df: Customer data
            orders_df: Order data
            order_items_df: Order items data
            rfm_df: RFM segmentation data
            behavioral_features_df: Behavioral features (optional)
        
        Returns:
            DataFrame with CLV features and target variables
        """
        logger.info("Preparing ML CLV features...")
        
        # Prepare orders data
        orders_df['order_date'] = pd.to_datetime(orders_df['order_date'])
        orders_df = orders_df[orders_df['order_date'] <= self.analysis_as_of_date]
        
        # Calculate historical CLV targets
        cutoff_90d = self.analysis_as_of_date - timedelta(days=90)
        cutoff_180d = self.analysis_as_of_date - timedelta(days=180)
        cutoff_365d = self.analysis_as_of_date - timedelta(days=365)
        
        # Calculate future spend for each horizon
        future_spend_90d = orders_df[orders_df['order_date'] > cutoff_90d].groupby('customer_id')['order_total'].sum()
        future_spend_180d = orders_df[orders_df['order_date'] > cutoff_180d].groupby('customer_id')['order_total'].sum()
        future_spend_365d = orders_df[orders_df['order_date'] > cutoff_365d].groupby('customer_id')['order_total'].sum()
        
        # Calculate historical spend (before cutoffs)
        historical_spend = orders_df[orders_df['order_date'] <= cutoff_365d].groupby('customer_id')['order_total'].sum()
        
        # Base features from RFM
        features = rfm_df[['customer_id', 'recency_days', 'frequency', 'monetary',
                           'recency_score', 'frequency_score', 'monetary_score']].copy()
        
        # Add historical CLV as target
        features['historical_clv'] = features['customer_id'].map(historical_spend).fillna(0)
        
        # Add future CLV targets
        features['future_clv_90d'] = features['customer_id'].map(future_spend_90d).fillna(0)
        features['future_clv_180d'] = features['customer_id'].map(future_spend_180d).fillna(0)
        features['future_clv_365d'] = features['customer_id'].map(future_spend_365d).fillna(0)
        
        # Add behavioral features if available
        if behavioral_features_df is not None:
            # Select key behavioral features
            behavioral_cols = [
                'purchase_frequency', 'purchase_acceleration', 'avg_basket_size',
                'avg_basket_value', 'discount_dependency', 'return_rate',
                'customer_tenure_days', 'days_since_last_order', 'unique_categories',
                'unique_brands'
            ]
            available_behavioral = [col for col in behavioral_cols if col in behavioral_features_df.columns]
            
            if available_behavioral:
                features = features.merge(
                    behavioral_features_df[['customer_id'] + available_behavioral],
                    on='customer_id',
                    how='left'
                )
        
        # Calculate additional features
        # Order value trend
        order_stats = orders_df.groupby('customer_id')['order_total'].agg(['mean', 'std', 'min', 'max'])
        features = features.merge(order_stats, left_on='customer_id', right_index=True, how='left')
        features.columns = [c if c in ['customer_id'] else f'order_{c}' for c in features.columns]
        
        # Fill missing values
        features = features.fillna(0)
        
        logger.info(f"Prepared {len(features)} samples with {len(features.columns) - 1} features")
        
        return features
    
    def train_model(
        self,
        features_df: pd.DataFrame,
        target_horizon: str = '90d',
        model_type: str = 'gradient_boosting',
        test_size: float = 0.2
    ) -> Dict:
        """
        Train ML model for CLV prediction.
        
        Args:
            features_df: DataFrame with features and targets
            target_horizon: Target horizon ('90d', '180d', '365d')
            model_type: Model type ('gradient_boosting', 'random_forest', 'ridge', 'lasso')
            test_size: Test set size for validation
        
        Returns:
            Dictionary with training results
        """
        logger.info(f"Training {model_type} model for {target_horizon} CLV prediction...")
        
        target_col = f'future_clv_{target_horizon}'
        
        # Prepare features (exclude customer_id and targets)
        exclude_cols = ['customer_id', 'future_clv_90d', 'future_clv_180d', 'future_clv_365d', 'historical_clv']
        feature_cols = [col for col in features_df.columns if col not in exclude_cols]
        
        X = features_df[feature_cols].values
        y = features_df[target_col].values
        
        # Split data (temporal split if possible, otherwise random)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42
        )
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Initialize model
        if model_type == 'gradient_boosting':
            model = GradientBoostingRegressor(
                n_estimators=100,
                max_depth=5,
                learning_rate=0.1,
                random_state=42
            )
        elif model_type == 'random_forest':
            model = RandomForestRegressor(
                n_estimators=100,
                max_depth=10,
                random_state=42
            )
        elif model_type == 'ridge':
            model = Ridge(alpha=1.0, random_state=42)
        elif model_type == 'lasso':
            model = Lasso(alpha=1.0, random_state=42)
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        # Train model
        model.fit(X_train_scaled, y_train)
        
        # Predictions
        y_train_pred = model.predict(X_train_scaled)
        y_test_pred = model.predict(X_test_scaled)
        
        # Metrics
        train_mae = mean_absolute_error(y_train, y_train_pred)
        train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
        train_r2 = r2_score(y_train, y_train_pred)
        
        test_mae = mean_absolute_error(y_test, y_test_pred)
        test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
        test_r2 = r2_score(y_test, y_test_pred)
        
        # Cross-validation
        cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5, scoring='neg_mean_absolute_error')
        cv_mae = -cv_scores.mean()
        
        # Feature importance
        if hasattr(model, 'feature_importances_'):
            importance = dict(zip(feature_cols, model.feature_importances_))
            self.feature_importance[f'{model_type}_{target_horizon}'] = importance
        elif hasattr(model, 'coef_'):
            importance = dict(zip(feature_cols, np.abs(model.coef_)))
            self.feature_importance[f'{model_type}_{target_horizon}'] = importance
        
        # Store model and scaler
        model_key = f'{model_type}_{target_horizon}'
        self.models[model_key] = model
        self.scalers[model_key] = scaler
        
        results = {
            'model_type': model_type,
            'target_horizon': target_horizon,
            'train_mae': train_mae,
            'train_rmse': train_rmse,
            'train_r2': train_r2,
            'test_mae': test_mae,
            'test_rmse': test_rmse,
            'test_r2': test_r2,
            'cv_mae': cv_mae,
            'n_features': len(feature_cols),
            'n_samples': len(features_df)
        }
        
        logger.info(f"Model trained. Test MAE: {test_mae:.2f}, Test R²: {test_r2:.3f}")
        
        return results
    
    def compare_models(
        self,
        features_df: pd.DataFrame,
        target_horizon: str = '90d',
        model_types: List[str] = None
    ) -> pd.DataFrame:
        """
        Compare multiple ML models for CLV prediction.
        
        Args:
            features_df: DataFrame with features and targets
            target_horizon: Target horizon ('90d', '180d', '365d')
            model_types: List of model types to compare
        
        Returns:
            DataFrame with model comparison results
        """
        if model_types is None:
            model_types = ['gradient_boosting', 'random_forest', 'ridge', 'lasso']
        
        results = []
        
        for model_type in model_types:
            try:
                result = self.train_model(features_df, target_horizon, model_type)
                results.append(result)
            except Exception as e:
                logger.error(f"Error training {model_type}: {e}")
        
        comparison_df = pd.DataFrame(results)
        
        # Rank models by test MAE
        comparison_df = comparison_df.sort_values('test_mae')
        comparison_df['rank'] = range(1, len(comparison_df) + 1)
        
        return comparison_df
    
    def predict_clv(
        self,
        customer_features: pd.DataFrame,
        model_type: str = 'gradient_boosting',
        horizon: str = '90d'
    ) -> np.ndarray:
        """
        Predict CLV for customers using trained ML model.
        
        Args:
            customer_features: DataFrame with customer features
            model_type: Model type to use
            horizon: Prediction horizon ('90d', '180d', '365d')
        
        Returns:
            Array of predicted CLV values
        """
        model_key = f'{model_type}_{horizon}'
        
        if model_key not in self.models:
            raise ValueError(f"Model {model_key} not trained. Call train_model first.")
        
        # Prepare features (exclude customer_id and targets)
        exclude_cols = ['customer_id', 'future_clv_90d', 'future_clv_180d', 'future_clv_365d', 'historical_clv']
        feature_cols = [col for col in customer_features.columns if col not in exclude_cols]
        
        X = customer_features[feature_cols].values
        
        # Scale features
        scaler = self.scalers[model_key]
        X_scaled = scaler.transform(X)
        
        # Predict
        model = self.models[model_key]
        predictions = model.predict(X_scaled)
        
        # Ensure non-negative predictions
        predictions = np.maximum(predictions, 0)
        
        return predictions
    
    def get_feature_importance(
        self,
        model_type: str = 'gradient_boosting',
        horizon: str = '90d',
        top_n: int = 10
    ) -> pd.DataFrame:
        """
        Get feature importance for a trained model.
        
        Args:
            model_type: Model type
            horizon: Prediction horizon
            top_n: Number of top features to return
        
        Returns:
            DataFrame with feature importance
        """
        model_key = f'{model_type}_{horizon}'
        
        if model_key not in self.feature_importance:
            raise ValueError(f"Feature importance not available for {model_key}")
        
        importance = self.feature_importance[model_key]
        importance_df = pd.DataFrame([
            {'feature': k, 'importance': v}
            for k, v in importance.items()
        ])
        
        importance_df = importance_df.sort_values('importance', ascending=False).head(top_n)
        
        return importance_df
    
    def save_models(self, save_dir: str = "models/clv"):
        """
        Save trained models and scalers to disk.
        
        Args:
            save_dir: Directory to save models
        """
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)
        
        for model_key, model in self.models.items():
            model_path = save_path / f"{model_key}.joblib"
            joblib.dump(model, model_path)
            
            scaler_path = save_path / f"{model_key}_scaler.joblib"
            joblib.dump(self.scalers[model_key], scaler_path)
        
        logger.info(f"Saved {len(self.models)} models to {save_dir}")
    
    def load_models(self, save_dir: str = "models/clv"):
        """
        Load trained models and scalers from disk.
        
        Args:
            save_dir: Directory to load models from
        """
        save_path = Path(save_dir)
        
        model_files = list(save_path.glob("*_scaler.joblib"))
        
        for scaler_file in model_files:
            model_key = scaler_file.stem.replace("_scaler", "")
            
            model_file = save_path / f"{model_key}.joblib"
            if model_file.exists():
                self.models[model_key] = joblib.load(model_file)
                self.scalers[model_key] = joblib.load(scaler_file)
        
        logger.info(f"Loaded {len(self.models)} models from {save_dir}")


def compare_clv_approaches(
    probabilistic_clv: pd.DataFrame,
    ml_clv: pd.DataFrame,
    historical_clv: pd.DataFrame
) -> pd.DataFrame:
    """
    Compare CLV predictions from different approaches.
    
    Args:
        probabilistic_clv: BG/NBD + Gamma-Gamma CLV predictions
        ml_clv: ML-based CLV predictions
        historical_clv: Historical CLV (actual spend)
    
    Returns:
        DataFrame with comparison of CLV approaches
    """
    comparison = probabilistic_clv[['customer_id']].copy()
    
    # Add CLV from each approach
    comparison['probabilistic_clv'] = probabilistic_clv['predicted_clv_365d']
    comparison['ml_clv'] = ml_clv['predicted_clv_365d']
    comparison['historical_clv'] = historical_clv['historical_clv']
    
    # Calculate differences ratios
    comparison['probabilistic_vs_historical'] = (
        comparison['probabilistic_clv'] / comparison['historical_clv']
    ).replace([np.inf, -np.inf], np.nan)
    
    comparison['ml_vs_historical'] = (
        comparison['ml_clv'] / comparison['historical_clv']
    ).replace([np.inf, -np.inf], np.nan)
    
    comparison['ml_vs_probabilistic'] = (
        comparison['ml_clv'] / comparison['probabilistic_clv']
    ).replace([np.inf, -np.inf], np.nan)
    
    # Calculate agreement metrics
    comparison['approach_agreement'] = (
        (comparison['probabilistic_clv'] > comparison['historical_clv']) == 
        (comparison['ml_clv'] > comparison['historical_clv'])
    ).astype(int)
    
    return comparison


def run_ml_clv_pipeline(
    customers_df: pd.DataFrame,
    orders_df: pd.DataFrame,
    order_items_df: pd.DataFrame,
    rfm_df: pd.DataFrame,
    behavioral_features_df: pd.DataFrame = None,
    analysis_as_of_date: str = "2024-12-31"
) -> Tuple[MLCLVPredictor, Dict]:
    """
    Convenience function to run complete ML CLV pipeline.
    
    Args:
        customers_df: Customer data
        orders_df: Order data
        order_items_df: Order items data
        rfm_df: RFM segmentation data
        behavioral_features_df: Behavioral features (optional)
        analysis_as_of_date: Date for analysis cutoff
    
    Returns:
        Tuple of (trained predictor, analysis results)
    """
    predictor = MLCLVPredictor(analysis_as_of_date)
    
    # Prepare features
    features_df = predictor.prepare_clv_features(
        customers_df, orders_df, order_items_df, rfm_df, behavioral_features_df
    )
    
    # Compare models for each horizon
    results = {}
    for horizon in ['90d', '180d', '365d']:
        comparison = predictor.compare_models(features_df, horizon)
        best_model = comparison.iloc[0]
        results[horizon] = {
            'best_model': best_model['model_type'],
            'best_mae': best_model['test_mae'],
            'best_r2': best_model['test_r2'],
            'comparison': comparison.to_dict()
        }
    
    return predictor, results
