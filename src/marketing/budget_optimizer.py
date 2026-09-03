"""
Marketing Budget Optimization Module
Implements budget optimization algorithms for maximizing marketing ROI.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from scipy.optimize import minimize
from config.logging_config import get_logger

logger = get_logger(__name__)


class BudgetOptimizer:
    """
    Marketing budget optimization engine.
    
    Features:
    - ROI-based budget allocation
    - Constraint optimization (min/max per channel)
    - Scenario analysis
    - Budget reallocation recommendations
    """
    
    def __init__(self):
        """Initialize budget optimizer"""
        self.optimization_results = {}
    
    def optimize_by_roi(
        self,
        channel_roi: Dict[str, float],
        total_budget: float,
        min_allocation_pct: float = 0.05,
        max_allocation_pct: float = 0.5
    ) -> Dict:
        """
        Optimize budget allocation based on ROI.
        
        Args:
            channel_roi: Dictionary mapping channels to ROI
            total_budget: Total budget to allocate
            min_allocation_pct: Minimum allocation percentage per channel
            max_allocation_pct: Maximum allocation percentage per channel
        
        Returns:
            Dictionary with optimized allocation
        """
        logger.info("Optimizing budget by ROI...")
        
        channels = list(channel_roi.keys())
        n_channels = len(channels)
        
        # Convert ROI to weights (handle negative ROI)
        roi_values = np.array([channel_roi[ch] for ch in channels])
        roi_values = np.maximum(roi_values, 0)  # Set negative ROI to 0
        
        if roi_values.sum() > 0:
            weights = roi_values / roi_values.sum()
        else:
            weights = np.ones(n_channels) / n_channels
        
        # Apply constraints
        min_budget = total_budget * min_allocation_pct
        max_budget = total_budget * max_allocation_pct
        
        allocations = {}
        for i, channel in enumerate(channels):
            # Calculate allocation based on weight
            allocation = weights[i] * total_budget
            
            # Apply constraints
            allocation = max(min_budget, min(max_budget, allocation))
            
            allocations[channel] = {
                'budget': float(allocation),
                'percentage': float(allocation / total_budget * 100),
                'roi': channel_roi[channel]
            }
        
        # Normalize to ensure total budget is met
        total_allocated = sum(a['budget'] for a in allocations.values())
        if total_allocated != total_budget:
            scale_factor = total_budget / total_allocated
            for channel in allocations:
                allocations[channel]['budget'] *= scale_factor
                allocations[channel]['percentage'] *= scale_factor
        
        logger.info(f"Budget optimization complete. Allocated to {len(channels)} channels")
        
        return allocations
    
    def optimize_with_constraints(
        self,
        channel_roi: Dict[str, float],
        channel_costs: Dict[str, float],
        total_budget: float,
        min_spend: Dict[str, float] = None,
        max_spend: Dict[str, float] = None
    ) -> Dict:
        """
        Optimize budget with custom constraints using optimization.
        
        Args:
            channel_roi: Dictionary mapping channels to ROI
            channel_costs: Dictionary mapping channels to cost per unit
            total_budget: Total budget
            min_spend: Minimum spend per channel (optional)
            max_spend: Maximum spend per channel (optional)
        
        Returns:
            Dictionary with optimized allocation
        """
        logger.info("Optimizing budget with constraints...")
        
        channels = list(channel_roi.keys())
        n_channels = len(channels)
        
        # Set default constraints
        if min_spend is None:
            min_spend = {ch: total_budget * 0.05 for ch in channels}
        if max_spend is None:
            max_spend = {ch: total_budget * 0.5 for ch in channels}
        
        # Define objective function (maximize total ROI)
        def objective(x):
            total_roi = 0
            for i, channel in enumerate(channels):
                total_roi += x[i] * channel_roi[channel]
            return -total_roi  # Minimize negative ROI
        
        # Define constraints
        constraints = [
            {'type': 'eq', 'fun': lambda x: np.sum(x) - total_budget}  # Total budget constraint
        ]
        
        # Bounds
        bounds = [(min_spend[ch], max_spend[ch]) for ch in channels]
        
        # Initial guess (equal allocation)
        x0 = np.array([total_budget / n_channels] * n_channels)
        
        # Optimize
        result = minimize(objective, x0, bounds=bounds, constraints=constraints, method='SLSQP')
        
        # Extract results
        allocations = {}
        for i, channel in enumerate(channels):
            allocations[channel] = {
                'budget': float(result.x[i]),
                'percentage': float(result.x[i] / total_budget * 100),
                'roi': channel_roi[channel],
                'expected_return': float(result.x[i] * channel_roi[channel])
            }
        
        logger.info(f"Constrained optimization complete. Total ROI: {-result.fun:.2f}")
        
        return allocations
    
    def scenario_analysis(
        self,
        base_allocation: Dict,
        channel_roi: Dict[str, float],
        scenarios: List[Dict]
    ) -> pd.DataFrame:
        """
        Perform scenario analysis on budget allocations.
        
        Args:
            base_allocation: Base budget allocation
            channel_roi: Channel ROI values
            scenarios: List of scenario configurations
        
        Returns:
            DataFrame with scenario comparison
        """
        logger.info(f"Performing scenario analysis for {len(scenarios)} scenarios...")
        
        results = []
        
        # Base scenario
        base_total_roi = sum(
            base_allocation[ch]['budget'] * channel_roi.get(ch, 0)
            for ch in base_allocation
        )
        
        results.append({
            'scenario': 'base',
            'total_budget': sum(a['budget'] for a in base_allocation.values()),
            'total_roi': base_total_roi,
            'allocation': base_allocation
        })
        
        # Analyze each scenario
        for scenario in scenarios:
            scenario_allocation = {}
            total_budget = scenario.get('total_budget', sum(a['budget'] for a in base_allocation.values()))
            
            # Apply scenario adjustments
            for channel in base_allocation:
                if channel in scenario.get('adjustments', {}):
                    adjustment = scenario['adjustments'][channel]
                    scenario_allocation[channel] = base_allocation[channel].copy()
                    scenario_allocation[channel]['budget'] *= adjustment
                    scenario_allocation[channel]['percentage'] *= adjustment
                else:
                    scenario_allocation[channel] = base_allocation[channel].copy()
            
            # Calculate total ROI
            scenario_total_roi = sum(
                scenario_allocation[ch]['budget'] * channel_roi.get(ch, 0)
                for ch in scenario_allocation
            )
            
            results.append({
                'scenario': scenario.get('name', 'unnamed'),
                'total_budget': total_budget,
                'total_roi': scenario_total_roi,
                'roi_change_pct': ((scenario_total_roi - base_total_roi) / base_total_roi * 100) if base_total_roi > 0 else 0,
                'allocation': scenario_allocation
            })
        
        results_df = pd.DataFrame(results)
        
        logger.info(f"Scenario analysis complete")
        
        return results_df
    
    def recommend_reallocation(
        self,
        current_allocation: Dict,
        channel_roi: Dict[str, float],
        channel_performance: Dict[str, Dict]
    ) -> Dict:
        """
        Recommend budget reallocation based on performance.
        
        Args:
            current_allocation: Current budget allocation
            channel_roi: Channel ROI values
            channel_performance: Channel performance metrics
        
        Returns:
            Dictionary with reallocation recommendations
        """
        logger.info("Generating reallocation recommendations...")
        
        recommendations = []
        total_budget = sum(a['budget'] for a in current_allocation.values())
        
        for channel, allocation in current_allocation.items():
            roi = channel_roi.get(channel, 0)
            performance = channel_performance.get(channel, {})
            
            # Determine recommendation
            if roi < 0:
                action = 'decrease'
                reason = 'negative_roi'
                suggested_change = -0.5  # Reduce by 50%
            elif roi < performance.get('target_roi', 0.1):
                action = 'decrease'
                reason = 'below_target'
                suggested_change = -0.2  # Reduce by 20%
            elif roi > performance.get('target_roi', 0.1) * 1.5:
                action = 'increase'
                reason = 'above_target'
                suggested_change = 0.3  # Increase by 30%
            else:
                action = 'maintain'
                reason = 'on_target'
                suggested_change = 0
            
            current_budget = allocation['budget']
            suggested_budget = current_budget * (1 + suggested_change)
            
            recommendations.append({
                'channel': channel,
                'current_budget': current_budget,
                'current_percentage': allocation['percentage'],
                'roi': roi,
                'action': action,
                'reason': reason,
                'suggested_budget': suggested_budget,
                'suggested_percentage': suggested_budget / total_budget * 100,
                'change_pct': suggested_change * 100
            })
        
        # Normalize suggested budgets to total budget
        total_suggested = sum(r['suggested_budget'] for r in recommendations)
        if total_suggested != total_budget:
            scale_factor = total_budget / total_suggested
            for rec in recommendations:
                rec['suggested_budget'] *= scale_factor
                rec['suggested_percentage'] *= scale_factor
        
        result = {
            'recommendations': recommendations,
            'total_budget': total_budget,
            'n_channels': len(recommendations)
        }
        
        logger.info(f"Reallocation recommendations generated for {len(recommendations)} channels")
        
        return result


def run_budget_optimization_pipeline(
    channel_roi: Dict[str, float],
    total_budget: float,
    channel_costs: Dict[str, float] = None,
    scenarios: List[Dict] = None
) -> Tuple[BudgetOptimizer, Dict]:
    """
    Convenience function to run complete budget optimization pipeline.
    
    Args:
        channel_roi: Channel ROI values
        total_budget: Total budget
        channel_costs: Channel costs (optional)
        scenarios: Scenario configurations (optional)
    
    Returns:
        Tuple of (optimizer, results)
    """
    optimizer = BudgetOptimizer()
    
    # Simple ROI-based optimization
    roi_allocation = optimizer.optimize_by_roi(channel_roi, total_budget)
    
    # Constrained optimization if costs provided
    constrained_allocation = None
    if channel_costs:
        constrained_allocation = optimizer.optimize_with_constraints(
            channel_roi, channel_costs, total_budget
        )
    
    # Scenario analysis if scenarios provided
    scenario_results = None
    if scenarios:
        scenario_results = optimizer.scenario_analysis(roi_allocation, channel_roi, scenarios)
    
    results = {
        'roi_allocation': roi_allocation,
        'constrained_allocation': constrained_allocation,
        'scenario_results': scenario_results
    }
    
    return optimizer, results
