"""
Customer Behavioral Features Engine
Computes advanced behavioral features for customer intelligence and fingerprinting.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional
from datetime import datetime, timedelta
from scipy import stats
from config.logging_config import get_logger

logger = get_logger(__name__)


class CustomerBehavioralFeatures:
    """
    Advanced customer behavioral feature engineering.
    
    Computes sophisticated behavioral metrics beyond basic RFM:
    - Purchase patterns (frequency, acceleration, deceleration)
    - Basket characteristics (size, diversity, value)
    - Engagement metrics (sessions, cart abandonment)
    - Price sensitivity and discount dependency
    - Return propensity and quality
    - Temporal patterns (tenure, interpurchase time)
    """
    
    def __init__(self, analysis_as_of_date: str = "2024-12-31"):
        """
        Initialize behavioral features engine.
        
        Args:
            analysis_as_of_date: Date for analysis cutoff
        """
        self.analysis_as_of_date = pd.to_datetime(analysis_as_of_date)
    
    def compute_behavioral_features(
        self,
        customers_df: pd.DataFrame,
        orders_df: pd.DataFrame,
        order_items_df: pd.DataFrame,
        sessions_df: pd.DataFrame,
        products_df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Compute comprehensive behavioral features for all customers.
        
        Args:
            customers_df: Customer data
            orders_df: Order data
            order_items_df: Order items data
            sessions_df: Website sessions data
            products_df: Product data
        
        Returns:
            DataFrame with behavioral features for each customer
        """
        logger.info("Computing customer behavioral features...")
        
        # Prepare data
        orders_df['order_date'] = pd.to_datetime(orders_df['order_date'])
        orders_df = orders_df[orders_df['order_date'] <= self.analysis_as_of_date]
        
        # Merge order items with products
        order_items_products = order_items_df.merge(
            products_df[['product_id', 'category_id', 'brand_name', 'selling_price']],
            on='product_id',
            how='left'
        )
        
        # Initialize features DataFrame
        features = customers_df[['customer_id']].copy()
        
        # Compute feature groups
        features = self._compute_purchase_patterns(features, orders_df)
        features = self._compute_basket_characteristics(features, orders_df, order_items_products)
        features = self._compute_engagement_metrics(features, sessions_df, orders_df)
        features = self._compute_price_sensitivity(features, orders_df, order_items_products)
        features = self._compute_return_propensity(features, orders_df, order_items_products)
        features = self._compute_temporal_patterns(features, orders_df)
        features = self._compute_diversity_metrics(features, order_items_products)
        
        logger.info(f"Computed {len(features.columns)} behavioral features for {len(features)} customers")
        
        return features
    
    def _compute_purchase_patterns(self, features: pd.DataFrame, orders_df: pd.DataFrame) -> pd.DataFrame:
        """Compute purchase frequency and pattern features"""
        # Group by customer
        customer_orders = orders_df.groupby('customer_id').agg({
            'order_id': 'count',
            'order_date': ['min', 'max', 'std']
        }).reset_index()
        customer_orders.columns = ['customer_id', 'total_orders', 'first_order_date', 
                                  'last_order_date', 'order_date_std']
        
        # Calculate purchase frequency (orders per day active)
        customer_orders['customer_tenure_days'] = (
            customer_orders['last_order_date'] - customer_orders['first_order_date']
        ).dt.days + 1
        customer_orders['purchase_frequency'] = (
            customer_orders['total_orders'] / customer_orders['customer_tenure_days']
        )
        
        # Purchase acceleration/deceleration (compare recent vs historical)
        cutoff_date = self.analysis_as_of_date - timedelta(days=90)
        recent_orders = orders_df[orders_df['order_date'] >= cutoff_date].groupby('customer_id').size()
        historical_orders = orders_df[orders_df['order_date'] < cutoff_date].groupby('customer_id').size()
        
        customer_orders['recent_orders_90d'] = customer_orders['customer_id'].map(recent_orders).fillna(0)
        customer_orders['historical_orders_before_90d'] = customer_orders['customer_id'].map(historical_orders).fillna(0)
        
        # Calculate acceleration ratio
        customer_orders['purchase_acceleration'] = np.where(
            customer_orders['historical_orders_before_90d'] > 0,
            customer_orders['recent_orders_90d'] / customer_orders['historical_orders_before_90d'],
            customer_orders['recent_orders_90d']
        )
        
        # Merge
        features = features.merge(
            customer_orders[['customer_id', 'total_orders', 'purchase_frequency', 
                            'purchase_acceleration', 'order_date_std']],
            on='customer_id',
            how='left'
        )
        
        return features
    
    def _compute_basket_characteristics(self, features: pd.DataFrame, 
                                       orders_df: pd.DataFrame,
                                       order_items_products: pd.DataFrame) -> pd.DataFrame:
        """Compute basket size and value characteristics"""
        # Average basket size (items per order)
        basket_sizes = order_items_products.groupby('order_id').size()
        orders_with_basket = orders_df.merge(
            basket_sizes.rename('basket_size'),
            left_on='order_id',
            right_index=True,
            how='left'
        )
        
        avg_basket_size = orders_with_basket.groupby('customer_id')['basket_size'].mean().reset_index()
        avg_basket_size.columns = ['customer_id', 'avg_basket_size']
        
        # Average basket value
        avg_basket_value = orders_df.groupby('customer_id')['order_total'].mean().reset_index()
        avg_basket_value.columns = ['customer_id', 'avg_basket_value']
        
        # Basket value trend (recent vs historical)
        cutoff_date = self.analysis_as_of_date - timedelta(days=90)
        recent_basket = orders_df[orders_df['order_date'] >= cutoff_date].groupby('customer_id')['order_total'].mean()
        historical_basket = orders_df[orders_df['order_date'] < cutoff_date].groupby('customer_id')['order_total'].mean()
        
        basket_trend = pd.DataFrame({
            'customer_id': orders_df['customer_id'].unique()
        })
        basket_trend['recent_avg_basket'] = basket_trend['customer_id'].map(recent_basket)
        basket_trend['historical_avg_basket'] = basket_trend['customer_id'].map(historical_basket)
        basket_trend['basket_value_trend'] = (
            basket_trend['recent_avg_basket'] / basket_trend['historical_avg_basket']
        ).fillna(1)
        
        # Merge
        features = features.merge(avg_basket_size, on='customer_id', how='left')
        features = features.merge(avg_basket_value, on='customer_id', how='left')
        features = features.merge(
            basket_trend[['customer_id', 'basket_value_trend']],
            on='customer_id',
            how='left'
        )
        
        return features
    
    def _compute_engagement_metrics(self, features: pd.DataFrame,
                                    sessions_df: pd.DataFrame,
                                    orders_df: pd.DataFrame) -> pd.DataFrame:
        """Compute session engagement and cart abandonment metrics"""
        if sessions_df.empty:
            # Fill with defaults if no session data
            features['total_sessions'] = 0
            features['avg_session_duration'] = 0
            features['cart_abandonment_rate'] = 0
            features['checkout_abandonment_rate'] = 0
            return features
        
        # Total sessions
        total_sessions = sessions_df.groupby('customer_id').size().reset_index()
        total_sessions.columns = ['customer_id', 'total_sessions']
        
        # Average session duration
        avg_duration = sessions_df.groupby('customer_id')['session_duration'].mean().reset_index()
        avg_duration.columns = ['customer_id', 'avg_session_duration']
        
        # Cart abandonment (sessions with add_to_cart but no purchase)
        sessions_with_cart = sessions_df[sessions_df['added_to_cart'] == True]
        sessions_with_purchase = sessions_df[sessions_df['made_purchase'] == True]
        
        cart_abandonment = pd.DataFrame({
            'customer_id': sessions_df['customer_id'].unique()
        })
        cart_abandonment['sessions_with_cart'] = cart_abandonment['customer_id'].map(
            sessions_with_cart.groupby('customer_id').size()
        ).fillna(0)
        cart_abandonment['sessions_with_purchase'] = cart_abandonment['customer_id'].map(
            sessions_with_purchase.groupby('customer_id').size()
        ).fillna(0)
        cart_abandonment['cart_abandonment_rate'] = np.where(
            cart_abandonment['sessions_with_cart'] > 0,
            (cart_abandonment['sessions_with_cart'] - cart_abandonment['sessions_with_purchase']) / 
            cart_abandonment['sessions_with_cart'],
            0
        )
        
        # Checkout abandonment (sessions with checkout_start but no purchase)
        sessions_with_checkout = sessions_df[sessions_df['checkout_started'] == True]
        checkout_abandonment = pd.DataFrame({
            'customer_id': sessions_df['customer_id'].unique()
        })
        checkout_abandonment['sessions_with_checkout'] = checkout_abandonment['customer_id'].map(
            sessions_with_checkout.groupby('customer_id').size()
        ).fillna(0)
        checkout_abandonment['checkout_abandonment_rate'] = np.where(
            checkout_abandonment['sessions_with_checkout'] > 0,
            (checkout_abandonment['sessions_with_checkout'] - checkout_abandonment['customer_id'].map(
                sessions_with_purchase.groupby('customer_id').size()
            ).fillna(0)) / checkout_abandonment['sessions_with_checkout'],
            0
        )
        
        # Merge
        features = features.merge(total_sessions, on='customer_id', how='left')
        features = features.merge(avg_duration, on='customer_id', how='left')
        features = features.merge(
            cart_abandonment[['customer_id', 'cart_abandonment_rate']],
            on='customer_id',
            how='left'
        )
        features = features.merge(
            checkout_abandonment[['customer_id', 'checkout_abandonment_rate']],
            on='customer_id',
            how='left'
        )
        
        # Fill NaN values
        features['total_sessions'] = features['total_sessions'].fillna(0)
        features['avg_session_duration'] = features['avg_session_duration'].fillna(0)
        features['cart_abandonment_rate'] = features['cart_abandonment_rate'].fillna(0)
        features['checkout_abandonment_rate'] = features['checkout_abandonment_rate'].fillna(0)
        
        return features
    
    def _compute_price_sensitivity(self, features: pd.DataFrame,
                                  orders_df: pd.DataFrame,
                                  order_items_products: pd.DataFrame) -> pd.DataFrame:
        """Compute price sensitivity and discount dependency"""
        # Discount usage rate
        items_with_discount = order_items_products[order_items_products['discount_pct'] > 0]
        total_items = order_items_products.groupby('customer_id').size()
        discounted_items = items_with_discount.groupby('customer_id').size()
        
        discount_dependency = pd.DataFrame({
            'customer_id': order_items_products['customer_id'].unique()
        })
        discount_dependency['discount_dependency'] = (
            discount_dependency['customer_id'].map(discounted_items) / 
            discount_dependency['customer_id'].map(total_items)
        ).fillna(0)
        
        # Average discount percentage
        avg_discount = order_items_products.groupby('customer_id')['discount_pct'].mean().reset_index()
        avg_discount.columns = ['customer_id', 'avg_discount_pct']
        
        # Price sensitivity (correlation between price and quantity)
        price_sensitivity = []
        for customer_id in order_items_products['customer_id'].unique():
            customer_items = order_items_products[order_items_products['customer_id'] == customer_id]
            if len(customer_items) > 1:
                corr = customer_items['selling_price'].corr(customer_items['quantity'])
                price_sensitivity.append({'customer_id': customer_id, 'price_sensitivity': corr})
            else:
                price_sensitivity.append({'customer_id': customer_id, 'price_sensitivity': 0})
        
        price_sensitivity_df = pd.DataFrame(price_sensitivity)
        
        # Merge
        features = features.merge(discount_dependency, on='customer_id', how='left')
        features = features.merge(avg_discount, on='customer_id', how='left')
        features = features.merge(price_sensitivity_df, on='customer_id', how='left')
        
        # Fill NaN values
        features['discount_dependency'] = features['discount_dependency'].fillna(0)
        features['avg_discount_pct'] = features['avg_discount_pct'].fillna(0)
        features['price_sensitivity'] = features['price_sensitivity'].fillna(0)
        
        return features
    
    def _compute_return_propensity(self, features: pd.DataFrame,
                                  orders_df: pd.DataFrame,
                                  order_items_products: pd.DataFrame) -> pd.DataFrame:
        """Compute return propensity and quality metrics"""
        # Return rate by customer
        returned_orders = orders_df[orders_df['order_status'] == 'Returned']
        total_orders = orders_df.groupby('customer_id').size()
        returned_count = returned_orders.groupby('customer_id').size()
        
        return_propensity = pd.DataFrame({
            'customer_id': orders_df['customer_id'].unique()
        })
        return_propensity['return_rate'] = (
            return_propensity['customer_id'].map(returned_count) / 
            return_propensity['customer_id'].map(total_orders)
        ).fillna(0)
        
        # Return rate by category (if available)
        # This would require returns data - simplified version here
        
        # Merge
        features = features.merge(
            return_propensity[['customer_id', 'return_rate']],
            on='customer_id',
            how='left'
        )
        
        features['return_rate'] = features['return_rate'].fillna(0)
        
        return features
    
    def _compute_temporal_patterns(self, features: pd.DataFrame,
                                  orders_df: pd.DataFrame) -> pd.DataFrame:
        """Compute temporal patterns like interpurchase time and seasonality"""
        # Customer tenure
        customer_tenure = orders_df.groupby('customer_id').agg({
            'order_date': ['min', 'max']
        }).reset_index()
        customer_tenure.columns = ['customer_id', 'first_order_date', 'last_order_date']
        
        customer_tenure['customer_tenure_days'] = (
            self.analysis_as_of_date - customer_tenure['first_order_date']
        ).dt.days
        customer_tenure['days_since_last_order'] = (
            self.analysis_as_of_date - customer_tenure['last_order_date']
        ).dt.days
        
        # Average interpurchase time
        interpurchase_times = []
        for customer_id in orders_df['customer_id'].unique():
            customer_orders = orders_df[orders_df['customer_id'] == customer_id].sort_values('order_date')
            if len(customer_orders) > 1:
                time_diffs = customer_orders['order_date'].diff().dt.days.dropna()
                avg_interpurchase = time_diffs.mean()
                std_interpurchase = time_diffs.std()
            else:
                avg_interpurchase = customer_tenure[customer_tenure['customer_id'] == customer_id]['customer_tenure_days'].values[0]
                std_interpurchase = 0
            
            interpurchase_times.append({
                'customer_id': customer_id,
                'avg_interpurchase_days': avg_interpurchase,
                'std_interpurchase_days': std_interpurchase
            })
        
        interpurchase_df = pd.DataFrame(interpurchase_times)
        
        # Merge
        features = features.merge(
            customer_tenure[['customer_id', 'customer_tenure_days', 'days_since_last_order']],
            on='customer_id',
            how='left'
        )
        features = features.merge(
            interpurchase_df[['customer_id', 'avg_interpurchase_days', 'std_interpurchase_days']],
            on='customer_id',
            how='left'
        )
        
        # Fill NaN values
        features['customer_tenure_days'] = features['customer_tenure_days'].fillna(0)
        features['days_since_last_order'] = features['days_since_last_order'].fillna(0)
        features['avg_interpurchase_days'] = features['avg_interpurchase_days'].fillna(0)
        features['std_interpurchase_days'] = features['std_interpurchase_days'].fillna(0)
        
        return features
    
    def _compute_diversity_metrics(self, features: pd.DataFrame,
                                   order_items_products: pd.DataFrame) -> pd.DataFrame:
        """Compute category and brand diversity metrics"""
        # Category diversity (number of unique categories)
        category_diversity = order_items_products.groupby('customer_id')['category_id'].nunique().reset_index()
        category_diversity.columns = ['customer_id', 'unique_categories']
        
        # Brand diversity (number of unique brands)
        brand_diversity = order_items_products.groupby('customer_id')['brand_name'].nunique().reset_index()
        brand_diversity.columns = ['customer_id', 'unique_brands']
        
        # Category concentration (Herfindahl index)
        category_concentration = []
        for customer_id in order_items_products['customer_id'].unique():
            customer_items = order_items_products[order_items_products['customer_id'] == customer_id]
            category_counts = customer_items['category_id'].value_counts(normalize=True)
            hhi = (category_counts ** 2).sum()  # Herfindahl-Hirschman Index
            category_concentration.append({'customer_id': customer_id, 'category_concentration': hhi})
        
        category_concentration_df = pd.DataFrame(category_concentration)
        
        # Merge
        features = features.merge(category_diversity, on='customer_id', how='left')
        features = features.merge(brand_diversity, on='customer_id', how='left')
        features = features.merge(category_concentration_df, on='customer_id', how='left')
        
        # Fill NaN values
        features['unique_categories'] = features['unique_categories'].fillna(0)
        features['unique_brands'] = features['unique_brands'].fillna(0)
        features['category_concentration'] = features['category_concentration'].fillna(0)
        
        return features
    
    def generate_customer_fingerprint(self, features_df: pd.DataFrame, customer_id: int) -> Dict:
        """
        Generate a behavioral fingerprint for a specific customer.
        
        Args:
            features_df: DataFrame with behavioral features
            customer_id: Customer ID to fingerprint
        
        Returns:
            Dictionary with customer behavioral fingerprint
        """
        customer_data = features_df[features_df['customer_id'] == customer_id]
        
        if len(customer_data) == 0:
            return {"error": "Customer not found"}
        
        row = customer_data.iloc[0]
        
        # Determine behavioral patterns
        purchase_pattern = self._classify_purchase_pattern(row)
        price_sensitivity_level = self._classify_price_sensitivity(row)
        engagement_level = self._classify_engagement(row)
        diversity_level = self._classify_diversity(row)
        
        fingerprint = {
            "customer_id": customer_id,
            "purchase_pattern": purchase_pattern,
            "price_sensitivity": price_sensitivity_level,
            "engagement_level": engagement_level,
            "diversity_level": diversity_level,
            "metrics": {
                "total_orders": int(row.get('total_orders', 0)),
                "purchase_frequency": float(row.get('purchase_frequency', 0)),
                "purchase_acceleration": float(row.get('purchase_acceleration', 0)),
                "avg_basket_size": float(row.get('avg_basket_size', 0)),
                "avg_basket_value": float(row.get('avg_basket_value', 0)),
                "discount_dependency": float(row.get('discount_dependency', 0)),
                "return_rate": float(row.get('return_rate', 0)),
                "days_since_last_order": int(row.get('days_since_last_order', 0)),
                "unique_categories": int(row.get('unique_categories', 0)),
                "unique_brands": int(row.get('unique_brands', 0)),
            },
            "recommendations": self._generate_recommendations(row, purchase_pattern, 
                                                          price_sensitivity_level, engagement_level)
        }
        
        return fingerprint
    
    def _classify_purchase_pattern(self, row: pd.Series) -> str:
        """Classify customer purchase pattern"""
        if row.get('purchase_acceleration', 0) > 1.2:
            return "Accelerating"
        elif row.get('purchase_acceleration', 0) < 0.8:
            return "Decelerating"
        elif row.get('purchase_frequency', 0) > 0.1:
            return "High Frequency"
        elif row.get('purchase_frequency', 0) > 0.05:
            return "Medium Frequency"
        else:
            return "Low Frequency"
    
    def _classify_price_sensitivity(self, row: pd.Series) -> str:
        """Classify customer price sensitivity"""
        if row.get('discount_dependency', 0) > 0.7:
            return "High"
        elif row.get('discount_dependency', 0) > 0.4:
            return "Medium"
        else:
            return "Low"
    
    def _classify_engagement(self, row: pd.Series) -> str:
        """Classify customer engagement level"""
        if row.get('total_sessions', 0) > 20:
            return "High"
        elif row.get('total_sessions', 0) > 10:
            return "Medium"
        else:
            return "Low"
    
    def _classify_diversity(self, row: pd.Series) -> str:
        """Classify customer purchase diversity"""
        if row.get('unique_categories', 0) > 5:
            return "High"
        elif row.get('unique_categories', 0) > 3:
            return "Medium"
        else:
            return "Low"
    
    def _generate_recommendations(self, row: pd.Series, purchase_pattern: str,
                                price_sensitivity: str, engagement: str) -> List[str]:
        """Generate personalized recommendations based on behavioral fingerprint"""
        recommendations = []
        
        # Purchase pattern recommendations
        if purchase_pattern == "Decelerating":
            recommendations.append("Re-engagement campaign recommended")
        elif purchase_pattern == "Accelerating":
            recommendations.append("Upsell opportunity - customer is increasing activity")
        
        # Price sensitivity recommendations
        if price_sensitivity == "High":
            recommendations.append("Target with discount offers")
        elif price_sensitivity == "Low":
            recommendations.append("Focus on value proposition rather than discounts")
        
        # Engagement recommendations
        if engagement == "Low":
            recommendations.append("Improve onboarding and engagement")
        
        # Return rate recommendations
        if row.get('return_rate', 0) > 0.2:
            recommendations.append("Investigate product quality or fit issues")
        
        # Cart abandonment recommendations
        if row.get('cart_abandonment_rate', 0) > 0.7:
            recommendations.append("Implement cart recovery emails")
        
        return recommendations


def compute_behavioral_features_pipeline(
    customers_df: pd.DataFrame,
    orders_df: pd.DataFrame,
    order_items_df: pd.DataFrame,
    sessions_df: pd.DataFrame,
    products_df: pd.DataFrame,
    analysis_as_of_date: str = "2024-12-31"
) -> pd.DataFrame:
    """
    Convenience function to compute behavioral features.
    
    Args:
        customers_df: Customer data
        orders_df: Order data
        order_items_df: Order items data
        sessions_df: Website sessions data
        products_df: Product data
        analysis_as_of_date: Date for analysis cutoff
    
    Returns:
        DataFrame with behavioral features
    """
    engine = CustomerBehavioralFeatures(analysis_as_of_date)
    return engine.compute_behavioral_features(
        customers_df, orders_df, order_items_df, sessions_df, products_df
    )
