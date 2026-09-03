"""
Prescriptive Optimization Engine Module
Implements prescriptive analytics for optimal decision making.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from scipy.optimize import minimize, linprog
from config.logging_config import get_logger

logger = get_logger(__name__)


class PrescriptiveOptimizer:
    """
    Prescriptive optimization engine for decision making.
    
    Features:
    - Linear programming optimization
    - Revenue optimization
    - Cost minimization
    - Resource allocation
    - Multi-objective optimization
    """
    
    def __init__(self):
        """Initialize prescriptive optimizer"""
        self.optimization_results = {}
        logger.info("Prescriptive optimizer initialized")
    
    def optimize_pricing(
        self,
        products_df: pd.DataFrame,
        demand_model: callable,
        cost_col: str = 'cost_price',
        current_price_col: str = 'selling_price',
        price_bounds: Tuple[float, float] = (0.5, 2.0)
    ) -> Dict:
        """
        Optimize prices to maximize revenue.
        
        Args:
            products_df: DataFrame with product data
            demand_model: Function that predicts demand given price
            cost_col: Cost column name
            current_price_col: Current price column name
            price_bounds: Price bounds as (min_multiplier, max_multiplier)
        
        Returns:
            Dictionary with optimal prices
        """
        logger.info("Optimizing prices for revenue maximization...")
        
        n_products = len(products_df)
        current_prices = products_df[current_price_col].values
        costs = products_df[cost_col].values
        
        # Define objective function (negative revenue for minimization)
        def objective(price_multipliers):
            prices = current_prices * price_multipliers
            demands = np.array([demand_model(p) for p in prices])
            revenues = (prices - costs) * demands
            return -np.sum(revenues)
        
        # Bounds for price multipliers
        bounds = [price_bounds] * n_products
        
        # Initial guess (current prices)
        x0 = np.ones(n_products)
        
        # Optimize
        result = minimize(objective, x0, bounds=bounds, method='L-BFGS-B')
        
        # Calculate optimal prices
        optimal_multipliers = result.x
        optimal_prices = current_prices * optimal_multipliers
        
        # Calculate expected revenue
        optimal_demands = np.array([demand_model(p) for p in optimal_prices])
        optimal_revenue = np.sum((optimal_prices - costs) * optimal_demands)
        
        # Calculate current revenue
        current_demands = np.array([demand_model(p) for p in current_prices])
        current_revenue = np.sum((current_prices - costs) * current_demands)
        
        results = {
            'optimal_prices': optimal_prices.tolist(),
            'optimal_multipliers': optimal_multipliers.tolist(),
            'optimal_revenue': float(optimal_revenue),
            'current_revenue': float(current_revenue),
            'revenue_increase': float(optimal_revenue - current_revenue),
            'revenue_increase_pct': float((optimal_revenue - current_revenue) / current_revenue * 100) if current_revenue > 0 else 0,
            'optimization_success': result.success,
            'n_products': n_products
        }
        
        logger.info(f"Price optimization complete. Revenue increase: {results['revenue_increase_pct']:.1%}")
        
        return results
    
    def optimize_inventory_allocation(
        self,
        products_df: pd.DataFrame,
        total_budget: float,
        demand_col: str = 'demand',
        cost_col: str = 'cost_price',
        price_col: str = 'selling_price'
    ) -> Dict:
        """
        Optimize inventory allocation within budget constraints.
        
        Args:
            products_df: DataFrame with product data
            total_budget: Total budget for inventory
            demand_col: Demand column name
            cost_col: Cost column name
            price_col: Selling price column name
        
        Returns:
            Dictionary with optimal allocation
        """
        logger.info(f"Optimizing inventory allocation with ${total_budget} budget...")
        
        n_products = len(products_df)
        costs = products_df[cost_col].values
        prices = products_df[price_col].values
        demands = products_df[demand_col].values
        
        # Profit per unit
        profits = prices - costs
        
        # Linear programming: maximize profit subject to budget constraint
        # Objective: maximize sum(profit * quantity)
        # Constraint: sum(cost * quantity) <= budget
        
        c = -profits  # Negative for minimization
        A_ub = costs.reshape(1, -1)
        b_ub = [total_budget]
        bounds = [(0, demand) for demand in demands]
        
        result = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
        
        if result.success:
            optimal_quantities = result.x
            optimal_cost = np.sum(optimal_quantities * costs)
            optimal_profit = np.sum(optimal_quantities * profits)
            
            results = {
                'optimal_quantities': optimal_quantities.tolist(),
                'total_cost': float(optimal_cost),
                'total_profit': float(optimal_profit),
                'budget_utilization': float(optimal_cost / total_budget * 100),
                'optimization_success': True,
                'n_products': n_products
            }
        else:
            results = {
                'optimization_success': False,
                'message': result.message
            }
        
        logger.info(f"Inventory allocation optimization complete. Success: {result.success}")
        
        return results
    
    def optimize_marketing_allocation(
        self,
        channels: List[str],
        channel_roi: Dict[str, float],
        channel_costs: Dict[str, float],
        total_budget: float,
        min_allocation_pct: float = 0.05
    ) -> Dict:
        """
        Optimize marketing budget allocation across channels.
        
        Args:
            channels: List of marketing channels
            channel_roi: ROI for each channel
            channel_costs: Cost per unit for each channel
            total_budget: Total marketing budget
            min_allocation_pct: Minimum allocation percentage per channel
        
        Returns:
            Dictionary with optimal allocation
        """
        logger.info(f"Optimizing marketing allocation for {len(channels)} channels...")
        
        n_channels = len(channels)
        
        # ROI values
        roi_values = np.array([channel_roi[ch] for ch in channels])
        
        # Define objective function (negative return for minimization)
        def objective(allocations):
            returns = allocations * roi_values
            return -np.sum(returns)
        
        # Budget constraint
        def budget_constraint(allocations):
            return total_budget - np.sum(allocations)
        
        # Minimum allocation constraint
        min_allocation = total_budget * min_allocation_pct
        bounds = [(min_allocation, total_budget) for _ in channels]
        
        constraints = {'type': 'eq', 'fun': budget_constraint}
        
        # Initial guess (equal allocation)
        x0 = np.array([total_budget / n_channels] * n_channels)
        
        # Optimize
        result = minimize(objective, x0, bounds=bounds, constraints=constraints, method='SLSQP')
        
        if result.success:
            optimal_allocations = result.x
            expected_returns = optimal_allocations * roi_values
            total_return = np.sum(expected_returns)
            
            allocation_results = {}
            for i, channel in enumerate(channels):
                allocation_results[channel] = {
                    'budget': float(optimal_allocations[i]),
                    'percentage': float(optimal_allocations[i] / total_budget * 100),
                    'expected_return': float(expected_returns[i]),
                    'roi': channel_roi[channel]
                }
            
            results = {
                'allocations': allocation_results,
                'total_return': float(total_return),
                'overall_roi': float(total_return / total_budget * 100),
                'optimization_success': True,
                'n_channels': n_channels
            }
        else:
            results = {
                'optimization_success': False,
                'message': result.message
            }
        
        logger.info(f"Marketing allocation optimization complete. Success: {result.success}")
        
        return results
    
    def optimize_staffing(
        self,
        demand_forecast: List[int],
        hourly_cost: float,
        service_level_target: float = 0.95,
        max_staff_per_hour: int = 20
    ) -> Dict:
        """
        Optimize staffing levels based on demand forecast.
        
        Args:
            demand_forecast: Hourly demand forecast
            hourly_cost: Cost per staff hour
            service_level_target: Target service level
            max_staff_per_hour: Maximum staff per hour
        
        Returns:
            Dictionary with optimal staffing
        """
        logger.info(f"Optimizing staffing for {len(demand_forecast)} hours...")
        
        n_hours = len(demand_forecast)
        
        # Define objective function (minimize cost)
        def objective(staffing):
            return np.sum(staffing) * hourly_cost
        
        # Service level constraint
        def service_level_constraint(staffing):
            # Simple model: service level = min(1, staff / demand)
            service_levels = np.minimum(1, staffing / np.array(demand_forecast, dtype=float))
            avg_service_level = np.mean(service_levels)
            return avg_service_level - service_level_target
        
        # Bounds for staffing
        bounds = [(0, max_staff_per_hour)] * n_hours
        
        constraints = {'type': 'ineq', 'fun': service_level_constraint}
        
        # Initial guess (demand-based)
        x0 = np.array([min(d, max_staff_per_hour) for d in demand_forecast])
        
        # Optimize
        result = minimize(objective, x0, bounds=bounds, constraints=constraints, method='SLSQP')
        
        if result.success:
            optimal_staffing = result.x
            total_cost = np.sum(optimal_staffing) * hourly_cost
            
            # Calculate actual service levels
            service_levels = np.minimum(1, optimal_staffing / np.array(demand_forecast, dtype=float))
            avg_service_level = np.mean(service_levels)
            
            results = {
                'optimal_staffing': optimal_staffing.tolist(),
                'total_cost': float(total_cost),
                'avg_service_level': float(avg_service_level),
                'service_levels': service_levels.tolist(),
                'optimization_success': True,
                'n_hours': n_hours
            }
        else:
            results = {
                'optimization_success': False,
                'message': result.message
            }
        
        logger.info(f"Staffing optimization complete. Success: {result.success}")
        
        return results
    
    def multi_objective_optimization(
        self,
        objectives: List[callable],
        weights: List[float],
        initial_guess: np.ndarray,
        bounds: List[Tuple],
        constraints: List = None
    ) -> Dict:
        """
        Perform multi-objective optimization using weighted sum method.
        
        Args:
            objectives: List of objective functions to minimize
            weights: Weights for each objective
            initial_guess: Initial guess for optimization
            bounds: Bounds for each variable
            constraints: Optimization constraints
        
        Returns:
            Dictionary with optimization results
        """
        logger.info(f"Running multi-objective optimization with {len(objectives)} objectives...")
        
        # Normalize weights
        weights = np.array(weights) / np.sum(weights)
        
        # Combined objective function
        def combined_objective(x):
            combined = 0
            for i, obj in enumerate(objectives):
                combined += weights[i] * obj(x)
            return combined
        
        # Optimize
        result = minimize(combined_objective, initial_guess, bounds=bounds, 
                         constraints=constraints, method='SLSQP')
        
        if result.success:
            # Calculate individual objective values
            objective_values = [obj(result.x) for obj in objectives]
            
            results = {
                'optimal_solution': result.x.tolist(),
                'objective_values': [float(v) for v in objective_values],
                'combined_objective': float(result.fun),
                'weights': weights.tolist(),
                'optimization_success': True
            }
        else:
            results = {
                'optimization_success': False,
                'message': result.message
            }
        
        logger.info(f"Multi-objective optimization complete. Success: {result.success}")
        
        return results
    
    def generate_action_plan(
        self,
        optimization_results: Dict,
        current_state: Dict,
        priority_threshold: float = 0.1
    ) -> List[Dict]:
        """
        Generate actionable plan from optimization results.
        
        Args:
            optimization_results: Results from optimization
            current_state: Current state of the system
            priority_threshold: Threshold for high-priority actions
        
        Returns:
            List of action items
        """
        logger.info("Generating action plan from optimization results...")
        
        actions = []
        
        # Generate pricing actions
        if 'optimal_prices' in optimization_results:
            current_prices = current_state.get('prices', [])
            optimal_prices = optimization_results['optimal_prices']
            
            for i, (current, optimal) in enumerate(zip(current_prices, optimal_prices)):
                change_pct = (optimal - current) / current if current > 0 else 0
                
                if abs(change_pct) > priority_threshold:
                    priority = 'high' if abs(change_pct) > 0.2 else 'medium'
                    actions.append({
                        'action_type': 'price_change',
                        'item_id': i,
                        'current_price': current,
                        'optimal_price': optimal,
                        'change_pct': float(change_pct * 100),
                        'priority': priority
                    })
        
        # Generate inventory actions
        if 'optimal_quantities' in optimization_results:
            current_quantities = current_state.get('quantities', [])
            optimal_quantities = optimization_results['optimal_quantities']
            
            for i, (current, optimal) in enumerate(zip(current_quantities, optimal_quantities)):
                change = optimal - current
                
                if abs(change) > priority_threshold * current:
                    priority = 'high' if abs(change) > 0.5 * current else 'medium'
                    actions.append({
                        'action_type': 'inventory_adjustment',
                        'item_id': i,
                        'current_quantity': current,
                        'optimal_quantity': optimal,
                        'change': float(change),
                        'priority': priority
                    })
        
        # Sort by priority
        priority_order = {'high': 0, 'medium': 1, 'low': 2}
        actions.sort(key=lambda x: priority_order.get(x['priority'], 3))
        
        logger.info(f"Generated {len(actions)} action items")
        
        return actions


def run_prescriptive_pipeline(
    products_df: pd.DataFrame,
    total_budget: float = 100000
) -> Tuple[PrescriptiveOptimizer, Dict]:
    """
    Convenience function to run complete prescriptive pipeline.
    
    Args:
        products_df: Product data
        total_budget: Total budget
    
    Returns:
        Tuple of (optimizer, results)
    """
    optimizer = PrescriptiveOptimizer()
    
    # Simple demand model for demonstration
    def demand_model(price):
        return 1000 - 5 * price
    
    # Optimize pricing
    pricing_results = optimizer.optimize_pricing(products_df, demand_model)
    
    # Optimize inventory allocation
    inventory_results = optimizer.optimize_inventory_allocation(
        products_df, total_budget
    )
    
    results = {
        'pricing_optimization': pricing_results,
        'inventory_optimization': inventory_results
    }
    
    return optimizer, results
