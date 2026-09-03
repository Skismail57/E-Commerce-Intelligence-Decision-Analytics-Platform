"""
Test suite for Churn Prediction module
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.ml.churn_predictor import ChurnPredictor


class TestChurnPredictor:
    """Test ChurnPredictor class"""
    
    @pytest.fixture
    def sample_churn_data(self):
        """Create sample churn feature data"""
        np.random.seed(42)
        n_samples = 1000
        
        return pd.DataFrame({
            'customer_id': range(1, n_samples + 1),
            'days_since_last_order': np.random.randint(1, 365, n_samples),
            'total_orders': np.random.randint(1, 50, n_samples),
            'avg_order_value': np.random.uniform(1000, 10000, n_samples),
            'total_spend': np.random.uniform(1000, 500000, n_samples),
            'return_rate': np.random.uniform(0, 0.3, n_samples),
            'discount_usage_pct': np.random.uniform(0, 50, n_samples),
            'sessions_count': np.random.randint(0, 100, n_samples),
            'customer_tenure_days': np.random.randint(30, 730, n_samples),
            'review_count': np.random.randint(0, 20, n_samples),
            'total_units_bought': np.random.randint(1, 200, n_samples),
            'customer_segment': np.random.choice(['Regular', 'Premium', 'VIP'], n_samples),
            'gender': np.random.choice(['M', 'F'], n_samples),
            'state': np.random.choice(['Maharashtra', 'Karnataka', 'Tamil Nadu', 'Delhi'], n_samples),
            'churn_label_90d': np.random.choice([0, 1], n_samples, p=[0.8, 0.2])
        })
    
    @pytest.fixture
    def churn_predictor(self, tmp_path):
        """Create ChurnPredictor instance with temp directory"""
        return ChurnPredictor(data_dir=tmp_path, processed_dir=tmp_path)
    
    def test_initialization(self, churn_predictor):
        """Test ChurnPredictor initialization"""
        assert churn_predictor is not None
        assert churn_predictor.model is None
        assert churn_predictor.feature_importance is None
    
    def test_prepare_features(self, churn_predictor, sample_churn_data):
        """Test feature preparation"""
        feature_df, y, feature_names = churn_predictor._prepare_features(sample_churn_data)
        
        assert feature_df is not None
        assert len(y) > 0
        assert len(feature_names) > 0
        assert len(feature_df) == len(sample_churn_data)
    
    def test_train_model(self, churn_predictor, sample_churn_data):
        """Test model training"""
        result = churn_predictor.train_model(sample_churn_data, test_size=0.2, random_state=42)
        
        assert 'metrics' in result
        assert 'status' in result
        assert churn_predictor.model is not None
        assert churn_predictor._metrics is not None
    
    def test_train_model_temporal_split(self, churn_predictor, sample_churn_data):
        """Test model training with temporal split"""
        result = churn_predictor.train_model(
            sample_churn_data, 
            test_size=0.2, 
            random_state=42,
            use_temporal_split=True
        )
        
        assert 'metrics' in result
        assert churn_predictor.model is not None
    
    def test_predict(self, churn_predictor, sample_churn_data):
        """Test prediction"""
        # Train first
        churn_predictor.train_model(sample_churn_data, test_size=0.2, random_state=42)
        
        # Predict on a subset
        test_data = sample_churn_data.head(10)
        predictions = churn_predictor.predict(test_data)
        
        assert predictions is not None
        assert len(predictions) == len(test_data)
        assert 'churn_probability' in predictions.columns
        assert 'risk_tier' in predictions.columns
    
    def test_feature_importance(self, churn_predictor, sample_churn_data):
        """Test feature importance extraction"""
        churn_predictor.train_model(sample_churn_data, test_size=0.2, random_state=42)
        
        importance = churn_predictor.feature_importance
        assert importance is not None
        assert 'feature' in importance.columns
        assert 'importance' in importance.columns
        assert len(importance) > 0
    
    def test_heuristic_fallback(self, churn_predictor):
        """Test heuristic fallback when insufficient data"""
        # Create minimal data with single class
        minimal_data = pd.DataFrame({
            'customer_id': range(1, 11),
            'days_since_last_order': [10] * 10,
            'total_orders': [5] * 10,
            'avg_order_value': [1000] * 10,
            'total_spend': [5000] * 10,
            'return_rate': [0.1] * 10,
            'discount_usage_pct': [5] * 10,
            'sessions_count': [10] * 10,
            'customer_tenure_days': [100] * 10,
            'review_count': [2] * 10,
            'total_units_bought': [10] * 10,
            'customer_segment': ['Regular'] * 10,
            'gender': ['M'] * 10,
            'state': ['Maharashtra'] * 10,
            'churn_label_90d': [0] * 10  # All same class
        })
        
        result = churn_predictor.train_model(minimal_data, test_size=0.2, random_state=42)
        
        assert result['status'] == 'heuristic_fallback'
        assert churn_predictor.model is None


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
