"""
Test suite for Demand Forecasting module
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.forecasting.demand_forecaster import DemandForecaster


class TestDemandForecaster:
    """Test DemandForecaster class"""
    
    @pytest.fixture
    def sample_orders(self):
        """Create sample order data"""
        np.random.seed(42)
        dates = pd.date_range('2024-01-01', periods=365, freq='D')
        return pd.DataFrame({
            'order_id': range(1, 366),
            'customer_id': np.random.randint(1, 51, 365),
            'order_date': dates,
            'order_status': np.random.choice(['Delivered', 'Shipped', 'Processing', 'Returned', 'Cancelled'], 365),
            'order_total': np.random.uniform(1000, 10000, 365)
        })
    
    @pytest.fixture
    def sample_order_items(self):
        """Create sample order items data"""
        np.random.seed(42)
        return pd.DataFrame({
            'order_item_id': range(1, 731),
            'order_id': np.random.randint(1, 366, 730),
            'product_id': np.random.randint(1, 21, 730),
            'quantity': np.random.randint(1, 5, 730),
            'unit_price': np.random.uniform(1000, 10000, 730),
            'line_total': np.random.uniform(1000, 20000, 730)
        })
    
    @pytest.fixture
    def demand_forecaster(self, tmp_path):
        """Create DemandForecaster instance with temp directory"""
        return DemandForecaster(data_dir=tmp_path, processed_dir=tmp_path)
    
    def test_initialization(self, demand_forecaster):
        """Test DemandForecaster initialization"""
        assert demand_forecaster is not None
        assert demand_forecaster.data_dir == tmp_path
    
    def test_build_ts_from_orders(self, demand_forecaster, sample_orders, sample_order_items):
        """Test time series building from orders"""
        ts = demand_forecaster._build_ts_from_orders(sample_orders, sample_order_items, granularity="D")
        
        assert 'date' in ts.columns
        assert 'units_sold' in ts.columns
        assert 'revenue_inr' in ts.columns
        assert len(ts) > 0
    
    def test_forecast_moving_average(self, demand_forecaster, sample_orders, sample_order_items):
        """Test moving average forecast"""
        ts = demand_forecaster._build_ts_from_orders(sample_orders, sample_order_items, granularity="D")
        result = demand_forecaster.forecast_moving_average(ts, horizon=30)
        
        assert 'forecast' in result
        assert len(result['forecast']) == 30
        assert all(result['forecast'] >= 0)
    
    def test_forecast_exponential_smoothing(self, demand_forecaster, sample_orders, sample_order_items):
        """Test exponential smoothing forecast"""
        ts = demand_forecaster._build_ts_from_orders(sample_orders, sample_order_items, granularity="D")
        result = demand_forecaster.forecast_exponential_smoothing(ts, horizon=30)
        
        assert 'forecast' in result
        assert len(result['forecast']) == 30
    
    def test_forecast_snaive(self, demand_forecaster, sample_orders, sample_order_items):
        """Test seasonal naive forecast"""
        ts = demand_forecaster._build_ts_from_orders(sample_orders, sample_order_items, granularity="D")
        result = demand_forecaster.forecast_snaive(ts, horizon=30)
        
        assert 'forecast' in result
        assert len(result['forecast']) == 30
    
    def test_time_series_cross_validation(self, demand_forecaster, sample_orders, sample_order_items):
        """Test time series cross-validation"""
        ts = demand_forecaster._build_ts_from_orders(sample_orders, sample_order_items, granularity="D")
        
        cv_results = demand_forecaster.time_series_cross_validation(
            ts, 
            lambda ts_, h_: demand_forecaster.forecast_moving_average(ts_, h_),
            n_splits=3,
            horizon=30
        )
        
        assert 'mae_mean' in cv_results
        assert 'rmse_mean' in cv_results
        assert 'mape_mean' in cv_results
        assert 'n_splits' in cv_results
    
    def test_mae(self, demand_forecaster):
        """Test MAE calculation"""
        y_true = np.array([100, 200, 300, 400, 500])
        y_pred = np.array([110, 190, 310, 390, 510])
        
        mae = demand_forecaster.mae(y_true, y_pred)
        assert mae == pytest.approx(10, abs=1)
    
    def test_rmse(self, demand_forecaster):
        """Test RMSE calculation"""
        y_true = np.array([100, 200, 300, 400, 500])
        y_pred = np.array([110, 190, 310, 390, 510])
        
        rmse = demand_forecaster.rmse(y_true, y_pred)
        assert rmse > 0
    
    def test_mape(self, demand_forecaster):
        """Test MAPE calculation"""
        y_true = np.array([100, 200, 300, 400, 500])
        y_pred = np.array([110, 190, 310, 390, 510])
        
        mape = demand_forecaster.mape(y_true, y_pred)
        assert mape > 0
        assert mape < 100


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
