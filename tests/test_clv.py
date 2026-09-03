"""
Test suite for CLV Prediction module
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import sys
from datetime import datetime

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.clv.clv_predictor import CLVPredictor


class TestCLVPredictor:
    """Test CLVPredictor class"""
    
    @pytest.fixture
    def sample_customers(self):
        """Create sample customer data"""
        np.random.seed(42)
        return pd.DataFrame({
            'customer_id': range(1, 101),
            'first_name': ['John'] * 100,
            'last_name': ['Doe'] * 100,
            'signup_date': pd.date_range('2022-01-01', periods=100, freq='D'),
            'customer_segment': np.random.choice(['Regular', 'Premium', 'VIP'], 100),
            'city': ['Mumbai'] * 100,
            'state': ['Maharashtra'] * 100,
            'country': ['India'] * 100
        })
    
    @pytest.fixture
    def sample_orders(self):
        """Create sample order data"""
        np.random.seed(42)
        return pd.DataFrame({
            'order_id': range(1, 501),
            'customer_id': np.random.randint(1, 101, 500),
            'order_date': pd.date_range('2023-01-01', periods=500, freq='H'),
            'order_status': np.random.choice(['Delivered', 'Shipped', 'Processing', 'Returned', 'Cancelled'], 500),
            'order_total': np.random.uniform(1000, 10000, 500)
        })
    
    @pytest.fixture
    def sample_order_items(self):
        """Create sample order items data"""
        np.random.seed(42)
        return pd.DataFrame({
            'order_item_id': range(1, 1001),
            'order_id': np.random.randint(1, 501, 1000),
            'product_id': np.random.randint(1, 51, 1000),
            'quantity': np.random.randint(1, 5, 1000),
            'unit_price': np.random.uniform(1000, 10000, 1000),
            'line_total': np.random.uniform(1000, 20000, 1000)
        })
    
    @pytest.fixture
    def clv_predictor(self, tmp_path):
        """Create CLVPredictor instance with temp directory"""
        return CLVPredictor(data_dir=tmp_path)
    
    def test_initialization(self, clv_predictor):
        """Test CLVPredictor initialization"""
        assert clv_predictor is not None
        assert clv_predictor.bg_nbd_params is None
        assert clv_predictor.gamma_gamma_params is None
    
    def test_compute_rfm_data(self, clv_predictor, sample_orders):
        """Test RFM data computation"""
        clv_predictor._dfs["orders"] = sample_orders

        rfm_data = clv_predictor.compute_rfm_data()
        
        assert 'customer_id' in rfm_data.columns
        assert 'frequency' in rfm_data.columns
        assert 'recency' in rfm_data.columns
        assert 'T' in rfm_data.columns
        assert 'monetary_value' in rfm_data.columns
        assert len(rfm_data) > 0
    
    def test_fit_bg_nbd(self, clv_predictor):
        """Test BG/NBD model fitting"""
        # Create sample RFM data
        rfm_data = pd.DataFrame({
            'customer_id': range(1, 101),
            'frequency': np.random.randint(1, 50, 100),
            'recency': np.random.randint(1, 365, 100),
            'T': np.random.randint(30, 730, 100),
            'monetary_value': np.random.uniform(1000, 10000, 100)
        })
        
        params = clv_predictor.fit_bg_nbd(rfm_data)
        
        assert 'r' in params
        assert 'alpha' in params
        assert 'a' in params
        assert 'b' in params
        assert all(v > 0 for v in params.values())
    
    def test_fit_gamma_gamma(self, clv_predictor):
        """Test Gamma-Gamma model fitting"""
        # Create sample RFM data
        rfm_data = pd.DataFrame({
            'customer_id': range(1, 101),
            'frequency': np.random.randint(1, 50, 100),
            'recency': np.random.randint(1, 365, 100),
            'T': np.random.randint(30, 730, 100),
            'monetary_value': np.random.uniform(1000, 10000, 100)
        })
        
        params = clv_predictor.fit_gamma_gamma(rfm_data)
        
        assert 'p' in params
        assert 'q' in params
        assert 'gamma' in params
        assert all(v > 0 for v in params.values())
    
    def test_predict_p_alive(self, clv_predictor):
        """Test p_alive prediction"""
        # Create sample RFM data and fit model
        rfm_data = pd.DataFrame({
            'customer_id': range(1, 101),
            'frequency': np.random.randint(1, 50, 100),
            'recency': np.random.randint(1, 365, 100),
            'T': np.random.randint(30, 730, 100),
            'monetary_value': np.random.uniform(1000, 10000, 100)
        })
        
        clv_predictor.fit_bg_nbd(rfm_data)
        p_alive = clv_predictor.predict_p_alive(rfm_data, prediction_period=90)
        
        assert len(p_alive) == len(rfm_data)
        assert all(p_alive >= 0)
        assert all(p_alive <= 1)
    
    def test_predict_expected_transactions(self, clv_predictor):
        """Test expected transactions prediction"""
        # Create sample RFM data and fit model
        rfm_data = pd.DataFrame({
            'customer_id': range(1, 101),
            'frequency': np.random.randint(1, 50, 100),
            'recency': np.random.randint(1, 365, 100),
            'T': np.random.randint(30, 730, 100),
            'monetary_value': np.random.uniform(1000, 10000, 100)
        })
        
        clv_predictor.fit_bg_nbd(rfm_data)
        expected_tx = clv_predictor.predict_expected_transactions(rfm_data, prediction_period=90)
        
        assert len(expected_tx) == len(rfm_data)
        assert all(expected_tx >= 0)
    
    def test_predict_clv(self, clv_predictor):
        """Test CLV prediction"""
        # Create sample RFM data
        rfm_data = pd.DataFrame({
            'customer_id': range(1, 101),
            'frequency': np.random.randint(1, 50, 100),
            'recency': np.random.randint(1, 365, 100),
            'T': np.random.randint(30, 730, 100),
            'monetary_value': np.random.uniform(1000, 10000, 100)
        })
        
        clv_predictions = clv_predictor.predict_clv(rfm_data, prediction_period=90)
        
        assert 'customer_id' in clv_predictions.columns
        assert 'expected_transactions' in clv_predictions.columns
        assert 'expected_monetary_value' in clv_predictions.columns
        assert 'clv' in clv_predictions.columns
        assert 'clv_discounted' in clv_predictions.columns
        assert 'clv_tier' in clv_predictions.columns
        assert len(clv_predictions) == len(rfm_data)
        assert all(clv_predictions['clv'] >= 0)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
