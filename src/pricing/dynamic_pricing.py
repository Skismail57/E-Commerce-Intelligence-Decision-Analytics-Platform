"""
Dynamic Pricing Engine
Implements dynamic pricing strategies based on demand elasticity, competitor pricing, and inventory levels.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
import joblib
from pathlib import Path
from config.logging_config import get_logger

logger = get_logger(__name__)


class DynamicPricingEngine:
    """
    Dynamic pricing engine for optimal price optimization.
    
    Features:
    - Demand elasticity estimation
    - Price optimization based on demand curves
    - Competitor price monitoring
    - Inventory-aware pricing
    - Time-based pricing (seasonality, promotions)
    """
    
    def __init__(self):
        """Initialize dynamic pricing engine"""
        self.demand_model = None
        self.elasticity_model = None
        self.price_history = {}
    
    def estimate_price_elasticity(
        self,
        sales_data: pd.DataFrame,
        product_id: int,
        window_days: int = 30
    ) -> float:
        """
        Estimate price elasticity of demand for a product.
        
        Args:
            sales_data: DataFrame with sales data (date, product_id, price, quantity)
            product_id: Product ID
            window_days: Time window for analysis
        
        Returns:
            Price elasticity coefficient
        """
        logger.info(f"Estimating price elasticity for product {product_id}...")
        
        # Filter data for product and time window
        product_sales = sales_data[
            (sales_data['product_id'] == product_id)
        ].copy()
        
        if len(product_sales) < 10:
            logger.warning(f"Insufficient data for product {product_id}")
            return -1.0  # Default elasticity
        
        # Calculate log-log regression for elasticity
        # log(quantity) = alpha + beta * log(price)
        product_sales['log_price'] = np.log(product_sales['price'])
        product_sales['log_quantity'] = np.log(product_sales['quantity'].replace(0, 1))
        
        # Remove outliers
        product_sales = product_sales[
            (product_sales['log_price'] > np.percentile(product_sales['log_price'], 5)) &
            (product_sales['log_price'] < np.percentile(product_sales['log_price'], 95))
        ]
        
        if len(product_sales) < 5:
            return -1.0
        
        # Fit linear regression
        X = product_sales['log_price'].values.reshape(-1, 1)
        y = product_sales['log_quantity'].values
        
        model = LinearRegression()
        model.fit(X, y)
        
        elasticity = model.coef_[0]
        
        logger.info(f"Price elasticity for product {product_id}: {elasticity:.3f}")
        
        return elasticity
    
    def optimize_price(
        self,
        product_id: int,
        current_price: float,
        elasticity: float,
        cost_price: float,
        demand_at_current_price: float,
        min_price: float = None,
        max_price: float = None,
        objective: str = 'profit'
    ) -> Dict:
        """
        Optimize price based on elasticity and objective.
        
        Args:
            product_id: Product ID
            current_price: Current selling price
            elasticity: Price elasticity of demand
            cost_price: Cost price
            demand_at_current_price: Demand at current price
            min_price: Minimum allowed price
            max_price: Maximum allowed price
            objective: Optimization objective ('profit', 'revenue', 'demand')
        
        Returns:
            Dictionary with optimal price and metrics
        """
        logger.info(f"Optimizing price for product {product_id}...")
        
        # Set price bounds
        if min_price is None:
            min_price = cost_price * 1.1  # At least 10% margin
        if max_price is None:
            max_price = current_price * 2  # Max 2x current price
        
        # Demand function: Q(p) = Q0 * (p/p0)^elasticity
        def demand_at_price(price):
            return demand_at_current_price * (price / current_price) ** elasticity
        
        # Revenue function: R(p) = p * Q(p)
        def revenue_at_price(price):
            return price * demand_at_price(price)
        
        # Profit function: P(p) = (p - cost) * Q(p)
        def profit_at_price(price):
            return (price - cost_price) * demand_at_price(price)
        
        # Optimize based on objective
        prices = np.linspace(min_price, max_price, 100)
        
        if objective == 'profit':
            values = [profit_at_price(p) for p in prices]
        elif objective == 'revenue':
            values = [revenue_at_price(p) for p in prices]
        elif objective == 'demand':
            values = [demand_at_price(p) for p in prices]
        else:
            raise ValueError(f"Unknown objective: {objective}")
        
        optimal_idx = np.argmax(values)
        optimal_price = prices[optimal_idx]
        
        # Calculate metrics at optimal price
        optimal_demand = demand_at_price(optimal_price)
        optimal_revenue = revenue_at_price(optimal_price)
        optimal_profit = profit_at_price(optimal_price)
        
        # Calculate metrics at current price
        current_revenue = revenue_at_price(current_price)
        current_profit = profit_at_price(current_price)
        
        # Calculate improvement
        revenue_improvement = (optimal_revenue - current_revenue) / current_revenue if current_revenue > 0 else 0
        profit_improvement = (optimal_profit - current_profit) / current_profit if current_profit > 0 else 0
        
        results = {
            'product_id': product_id,
            'current_price': current_price,
            'optimal_price': optimal_price,
            'price_change_pct': float((optimal_price - current_price) / current_price * 100),
            'elasticity': elasticity,
            'optimal_demand': float(optimal_demand),
            'optimal_revenue': float(optimal_revenue),
            'optimal_profit': float(optimal_profit),
            'revenue_improvement_pct': float(revenue_improvement * 100),
            'profit_improvement_pct': float(profit_improvement * 100),
            'objective': objective
        }
        
        logger.info(f"Optimal price: {optimal_price:.2f} (change: {results['price_change_pct']:.1f}%)")
        
        return results
    
    def train_demand_model(
        self,
        sales_data: pd.DataFrame,
        feature_cols: List[str]
    ) -> Dict:
        """
        Train ML model to predict demand based on price and other features.
        
        Args:
            sales_data: DataFrame with sales data and features
            feature_cols: List of feature column names
        
        Returns:
            Dictionary with training results
        """
        logger.info("Training demand prediction model...")
        
        # Prepare data
        X = sales_data[feature_cols].values
        y = sales_data['quantity'].values
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Train model
        self.demand_model = GradientBoostingRegressor(
            n_estimators=100,
            max_depth=5,
            learning_rate=0.1,
            random_state=42
        )
        
        self.demand_model.fit(X_train, y_train)
        
        # Predictions
        y_train_pred = self.demand_model.predict(X_train)
        y_test_pred = self.demand_model.predict(X_test)
        
        # Metrics
        train_mae = mean_absolute_error(y_train, y_train_pred)
        test_mae = mean_absolute_error(y_test, y_test_pred)
        train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
        test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
        
        results = {
            'train_mae': float(train_mae),
            'test_mae': float(test_mae),
            'train_rmse': float(train_rmse),
            'test_rmse': float(test_rmse),
            'n_features': len(feature_cols),
            'n_samples': len(sales_data)
        }
        
        logger.info(f"Demand model trained. Test MAE: {test_mae:.2f}")
        
        return results
    
    def predict_demand(
        self,
        product_features: pd.DataFrame,
        feature_cols: List[str]
    ) -> np.ndarray:
        """
        Predict demand using trained model.
        
        Args:
            product_features: DataFrame with product features
            feature_cols: List of feature column names
        
        Returns:
            Array of predicted demand values
        """
        if self.demand_model is None:
            raise ValueError("Demand model not trained. Call train_demand_model first.")
        
        X = product_features[feature_cols].values
        predictions = self.demand_model.predict(X)
        
        # Ensure non-negative predictions
        predictions = np.maximum(predictions, 0)
        
        return predictions
    
    def batch_optimize_prices(
        self,
        products_df: pd.DataFrame,
        sales_data: pd.DataFrame,
        cost_prices: pd.DataFrame,
        objective: str = 'profit'
    ) -> pd.DataFrame:
        """
        Optimize prices for multiple products.
        
        Args:
            products_df: DataFrame with product data
            sales_data: DataFrame with sales data
            cost_prices: DataFrame with cost prices
            objective: Optimization objective
        
        Returns:
            DataFrame with optimal prices for all products
        """
        logger.info(f"Batch optimizing prices for {len(products_df)} products...")
        
        results = []
        
        for _, product in products_df.iterrows():
            product_id = product['product_id']
            current_price = product['selling_price']
            cost_price = cost_prices[cost_prices['product_id'] == product_id]['cost_price'].values[0]
            
            # Estimate elasticity
            elasticity = self.estimate_price_elasticity(sales_data, product_id)
            
            if elasticity == -1.0:
                # Use default elasticity if estimation failed
                elasticity = -1.5  # Typical for e-commerce
            
            # Get current demand
            product_sales = sales_data[sales_data['product_id'] == product_id]
            demand_at_current = product_sales['quantity'].mean() if len(product_sales) > 0 else 10
            
            # Optimize price
            optimization_result = self.optimize_price(
                product_id, current_price, elasticity, cost_price,
                demand_at_current, objective=objective
            )
            
            results.append(optimization_result)
        
        results_df = pd.DataFrame(results)
        
        logger.info(f"Batch optimization complete. Avg profit improvement: {results_df['profit_improvement_pct'].mean():.1f}%")
        
        return results_df
    
    def get_price_recommendations(
        self,
        product_id: int,
        current_price: float,
        competitor_prices: List[float],
        elasticity: float,
        inventory_level: int,
        inventory_threshold: int = 10
    ) -> Dict:
        """
        Get comprehensive price recommendations considering multiple factors.
        
        Args:
            product_id: Product ID
            current_price: Current price
            competitor_prices: List of competitor prices
            elasticity: Price elasticity
            inventory_level: Current inventory level
            inventory_threshold: Low inventory threshold
        
        Returns:
            Dictionary with price recommendations
        """
        recommendations = {
            'product_id': product_id,
            'current_price': current_price,
            'competitor_avg_price': float(np.mean(competitor_prices)),
            'competitor_min_price': float(np.min(competitor_prices)),
            'competitor_max_price': float(np.max(competitor_prices)),
            'elasticity': elasticity,
            'inventory_level': inventory_level,
            'inventory_status': 'normal'
        }
        
        # Determine inventory status
        if inventory_level < inventory_threshold:
            recommendations['inventory_status'] = 'low'
            recommendations['inventory_action'] = 'increase_price_to_reduce_demand'
        elif inventory_level > inventory_threshold * 10:
            recommendations['inventory_status'] = 'high'
            recommendations['inventory_action'] = 'decrease_price_to_clear_inventory'
        
        # Competitor-based pricing
        if current_price > recommendations['competitor_avg_price'] * 1.2:
            recommendations['competitor_position'] = 'premium'
            recommendations['competitor_action'] = 'consider_price_reduction'
        elif current_price < recommendations['competitor_avg_price'] * 0.8:
            recommendations['competitor_position'] = 'discounted'
            recommendations['competitor_action'] = 'consider_price_increase'
        else:
            recommendations['competitor_position'] = 'competitive'
            recommendations['competitor_action'] = 'maintain_current_price'
        
        # Elasticity-based recommendation
        if abs(elasticity) > 2:
            recommendations['elasticity_sensitivity'] = 'high'
            recommendations['elasticity_action'] = 'small_price_changes_have_large_impact'
        elif abs(elasticity) > 1:
            recommendations['elasticity_sensitivity'] = 'medium'
            recommendations['elasticity_action'] = 'moderate_price_changes_effective'
        else:
            recommendations['elasticity_sensitivity'] = 'low'
            recommendations['elasticity_action'] = 'price_changes_have_limited_impact'
        
        # Overall recommendation
        recommendations['recommended_action'] = self._combine_recommendations(recommendations)
        
        return recommendations
    
    def _combine_recommendations(self, recommendations: Dict) -> str:
        """Combine multiple factors into overall recommendation"""
        actions = []
        
        if recommendations['inventory_status'] == 'low':
            actions.append('increase_price')
        elif recommendations['inventory_status'] == 'high':
            actions.append('decrease_price')
        
        if recommendations['competitor_position'] == 'premium':
            actions.append('monitor_competitors')
        elif recommendations['competitor_position'] == 'discounted':
            actions.append('consider_increase')
        
        if not actions:
            return 'maintain_current_price'
        
        return ', '.join(actions)
    
    def save_model(self, save_path: str = "models/dynamic_pricing.joblib"):
        """Save trained pricing model"""
        if self.demand_model is None:
            raise ValueError("No model to save")
        
        model_data = {
            'demand_model': self.demand_model,
            'price_history': self.price_history
        }
        
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(model_data, save_path)
        
        logger.info(f"Model saved to {save_path}")
    
    def load_model(self, load_path: str = "models/dynamic_pricing.joblib"):
        """Load trained pricing model"""
        model_data = joblib.load(load_path)
        
        self.demand_model = model_data['demand_model']
        self.price_history = model_data['price_history']
        
        logger.info(f"Model loaded from {load_path}")


def run_dynamic_pricing_pipeline(
    sales_data: pd.DataFrame,
    products_df: pd.DataFrame,
    cost_prices: pd.DataFrame,
    objective: str = 'profit'
) -> Tuple[DynamicPricingEngine, pd.DataFrame]:
    """
    Convenience function to run complete dynamic pricing pipeline.
    
    Args:
        sales_data: Sales data
        products_df: Product data
        cost_prices: Cost price data
        objective: Optimization objective
    
    Returns:
        Tuple of (engine, optimization results)
    """
    engine = DynamicPricingEngine()
    
    # Batch optimize prices
    results = engine.batch_optimize_prices(
        products_df, sales_data, cost_prices, objective
    )
    
    return engine, results
