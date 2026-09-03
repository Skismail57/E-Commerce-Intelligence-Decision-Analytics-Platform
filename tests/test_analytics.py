"""
Test suite for analytics modules
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.analytics.customer_analytics import CustomerIntelligence
from src.analytics.product_analytics import ProductIntelligence
from src.analytics.marketing_analytics import MarketingAnalyzer
from src.analytics.statistical_analytics import StatisticalAnalyzer
from src.transformation.feature_engineering import FeatureEngineer


class TestFeatureEngineering:
    """Test FeatureEngineer class"""
    
    @pytest.fixture
    def sample_customers(self):
        """Create sample customer data"""
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
        return pd.DataFrame({
            'order_id': range(1, 501),
            'customer_id': np.random.randint(1, 101, 500),
            'order_date': pd.date_range('2023-01-01', periods=500, freq='H'),
            'order_status': np.random.choice(['Delivered', 'Shipped', 'Processing', 'Returned', 'Cancelled'], 500),
            'order_total': np.random.uniform(1000, 10000, 500),
            'discount_amount': np.random.uniform(0, 500, 500)
        })
    
    @pytest.fixture
    def sample_products(self):
        """Create sample product data"""
        return pd.DataFrame({
            'product_id': range(1, 51),
            'product_name': [f'Product {i}' for i in range(1, 51)],
            'category_id': np.random.randint(1, 11, 50),
            'supplier_id': np.random.randint(1, 6, 50),
            'cost_price': np.random.uniform(500, 5000, 50),
            'selling_price': np.random.uniform(1000, 10000, 50),
            'launch_date': pd.date_range('2022-01-01', periods=50, freq='W')
        })
    
    @pytest.fixture
    def sample_order_items(self):
        """Create sample order items data"""
        return pd.DataFrame({
            'order_item_id': range(1, 1001),
            'order_id': np.random.randint(1, 501, 1000),
            'product_id': np.random.randint(1, 51, 1000),
            'quantity': np.random.randint(1, 5, 1000),
            'unit_price': np.random.uniform(1000, 10000, 1000),
            'line_total': np.random.uniform(1000, 20000, 1000),
            'discount': np.random.uniform(0, 1000, 1000)
        })
    
    def test_compute_rfm(self, sample_customers, sample_orders):
        """Test RFM computation"""
        rfm = FeatureEngineer.compute_rfm(sample_customers, sample_orders)
        
        assert 'customer_id' in rfm.columns
        assert 'recency_days' in rfm.columns
        assert 'frequency' in rfm.columns
        assert 'monetary_value' in rfm.columns
        assert 'r_score' in rfm.columns
        assert 'f_score' in rfm.columns
        assert 'm_score' in rfm.columns
        assert 'rfm_segment' in rfm.columns
        
        assert len(rfm) == len(sample_customers)
        assert rfm['r_score'].between(1, 5).all()
        assert rfm['f_score'].between(1, 5).all()
        assert rfm['m_score'].between(1, 5).all()
    
    def test_compute_clv(self, sample_customers, sample_orders, sample_order_items, sample_products):
        """Test CLV computation"""
        clv = FeatureEngineer.compute_clv(
            sample_customers, sample_orders, sample_order_items, sample_products
        )
        
        assert 'customer_id' in clv.columns
        assert 'clv' in clv.columns
        assert 'avg_order_value' in clv.columns
        assert 'purchase_frequency' in clv.columns
        assert 'customer_value_tier' in clv.columns
        
        assert len(clv) == len(sample_customers)
        assert (clv['clv'] >= 0).all()
    
    def test_compute_product_matrix(self, sample_products, sample_order_items, sample_orders):
        """Test product matrix computation"""
        categories = pd.DataFrame({
            'category_id': range(1, 11),
            'category_name': [f'Category {i}' for i in range(1, 11)],
            'subcategory': ['Subcategory A'] * 10
        })
        
        matrix = FeatureEngineer.compute_product_matrix(
            sample_products, sample_order_items, sample_orders, categories
        )
        
        assert 'product_id' in matrix.columns
        assert 'revenue_inr' in matrix.columns
        assert 'profit_inr' in matrix.columns
        assert 'margin_ratio' in matrix.columns
        assert 'quadrant' in matrix.columns
        
        assert len(matrix) == len(sample_products)
        assert matrix['quadrant'].isin(['Stars', 'Volume', 'Remove', 'Premium']).all()


class TestStatisticalAnalyzer:
    """Test StatisticalAnalyzer class"""
    
    @pytest.fixture
    def sample_series(self):
        """Create sample series for testing"""
        return pd.Series(np.random.normal(100, 15, 1000))
    
    def test_descriptive_statistics(self, sample_series):
        """Test descriptive statistics"""
        analyzer = StatisticalAnalyzer()
        stats = analyzer.descriptive_statistics(sample_series)
        
        assert 'count' in stats
        assert 'mean' in stats
        assert 'median' in stats
        assert 'std' in stats
        assert 'min' in stats
        assert 'max' in stats
        
        assert stats['count'] == 1000
        assert stats['mean'] == pytest.approx(100, abs=5)
    
    def test_detect_outliers_iqr(self, sample_series):
        """Test IQR outlier detection"""
        analyzer = StatisticalAnalyzer()
        outlier_mask, clean_series = analyzer.detect_outliers(sample_series, method='iqr')
        
        assert len(outlier_mask) == len(sample_series)
        assert isinstance(outlier_mask, pd.Series)
    
    def test_detect_outliers_zscore(self, sample_series):
        """Test Z-score outlier detection"""
        analyzer = StatisticalAnalyzer()
        outlier_mask, clean_series = analyzer.detect_outliers(sample_series, method='zscore')
        
        assert len(outlier_mask) == len(sample_series)
        assert isinstance(outlier_mask, pd.Series)
    
    def test_correlation_analysis(self):
        """Test correlation analysis"""
        analyzer = StatisticalAnalyzer()
        df = pd.DataFrame({
            'A': np.random.normal(0, 1, 100),
            'B': np.random.normal(0, 1, 100),
            'C': np.random.normal(0, 1, 100)
        })
        
        corr = analyzer.correlation_analysis(df, method='pearson')
        
        assert corr.shape == (3, 3)
        assert np.diag(corr.values).sum() == pytest.approx(3, abs=0.1)


class TestCustomerIntelligence:
    """Test CustomerIntelligence class"""
    
    @pytest.fixture
    def customer_intelligence(self, tmp_path):
        """Create CustomerIntelligence instance with temp directory"""
        return CustomerIntelligence(data_dir=tmp_path)
    
    def test_initialization(self, customer_intelligence):
        """Test CustomerIntelligence initialization"""
        assert customer_intelligence is not None
        assert customer_intelligence.feature_engineer is not None


class TestProductIntelligence:
    """Test ProductIntelligence class"""
    
    @pytest.fixture
    def product_intelligence(self, tmp_path):
        """Create ProductIntelligence instance with temp directory"""
        return ProductIntelligence(data_dir=tmp_path)
    
    def test_initialization(self, product_intelligence):
        """Test ProductIntelligence initialization"""
        assert product_intelligence is not None
        assert product_intelligence.fe is not None


class TestMarketingAnalyzer:
    """Test MarketingAnalyzer class"""
    
    @pytest.fixture
    def marketing_analyzer(self, tmp_path):
        """Create MarketingAnalyzer instance with temp directory"""
        return MarketingAnalyzer(data_dir=tmp_path)
    
    def test_initialization(self, marketing_analyzer):
        """Test MarketingAnalyzer initialization"""
        assert marketing_analyzer is not None


class TestDataIntegrity:
    """Test data integrity and validation"""
    
    def test_customer_data_schema(self):
        """Test customer data schema requirements"""
        required_columns = [
            'customer_id', 'first_name', 'last_name', 'signup_date',
            'customer_segment', 'city', 'state', 'country'
        ]
        
        # This would be used with actual data
        # for col in required_columns:
        #     assert col in customer_df.columns
    
    def test_order_data_schema(self):
        """Test order data schema requirements"""
        required_columns = [
            'order_id', 'customer_id', 'order_date', 'order_status',
            'order_total'
        ]
        
        # This would be used with actual data
        # for col in required_columns:
        #     assert col in order_df.columns
    
    def test_product_data_schema(self):
        """Test product data schema requirements"""
        required_columns = [
            'product_id', 'product_name', 'category_id', 'supplier_id',
            'cost_price', 'selling_price'
        ]
        
        # This would be used with actual data
        # for col in required_columns:
        #     assert col in product_df.columns


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
