"""
Monte Carlo Simulation Module
Implements Monte Carlo simulation for risk analysis and decision making under uncertainty.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Callable
from scipy import stats
from config.logging_config import get_logger

logger = get_logger(__name__)


class MonteCarloSimulator:
    """
    Monte Carlo simulation engine for risk analysis.
    
    Features:
    - Revenue simulation with uncertainty
    - Demand forecasting with confidence intervals
    - Risk analysis and VaR calculation
    - Scenario generation
    - Sensitivity analysis
    """
    
    def __init__(self, random_seed: int = 42):
        """
        Initialize Monte Carlo simulator.
        
        Args:
            random_seed: Random seed for reproducibility
        """
        np.random.seed(random_seed)
        logger.info("Monte Carlo simulator initialized")
    
    def simulate_revenue(
        self,
        base_revenue: float,
        revenue_std: float,
        n_simulations: int = 10000,
        distribution: str = 'normal'
    ) -> Dict:
        """
        Simulate revenue with uncertainty.
        
        Args:
            base_revenue: Base revenue value
            revenue_std: Standard deviation of revenue
            n_simulations: Number of simulations
            distribution: Distribution type ('normal', 'lognormal', 'triangular')
        
        Returns:
            Dictionary with simulation results
        """
        logger.info(f"Running {n_simulations} revenue simulations...")
        
        if distribution == 'normal':
            simulated_revenues = np.random.normal(base_revenue, revenue_std, n_simulations)
        elif distribution == 'lognormal':
            # Convert normal parameters to lognormal
            mu = np.log(base_revenue**2 / np.sqrt(base_revenue**2 + revenue_std**2))
            sigma = np.sqrt(np.log(1 + revenue_std**2 / base_revenue**2))
            simulated_revenues = np.random.lognormal(mu, sigma, n_simulations)
        elif distribution == 'triangular':
            # Triangular distribution with mode at base_revenue
            low = base_revenue - 2 * revenue_std
            high = base_revenue + 2 * revenue_std
            simulated_revenues = np.random.triangular(low, base_revenue, high, n_simulations)
        else:
            raise ValueError(f"Unknown distribution: {distribution}")
        
        # Calculate statistics
        results = {
            'mean': float(np.mean(simulated_revenues)),
            'std': float(np.std(simulated_revenues)),
            'median': float(np.median(simulated_revenues)),
            'min': float(np.min(simulated_revenues)),
            'max': float(np.max(simulated_revenues)),
            'percentile_5': float(np.percentile(simulated_revenues, 5)),
            'percentile_25': float(np.percentile(simulated_revenues, 25)),
            'percentile_75': float(np.percentile(simulated_revenues, 75)),
            'percentile_95': float(np.percentile(simulated_revenues, 95)),
            'n_simulations': n_simulations,
            'simulations': simulated_revenues
        }
        
        logger.info(f"Revenue simulation complete. Mean: ${results['mean']:,.2f}")
        
        return results
    
    def simulate_demand(
        self,
        base_demand: float,
        demand_std: float,
        price_elasticity: float,
        price_changes: List[float],
        n_simulations: int = 10000
    ) -> pd.DataFrame:
        """
        Simulate demand under different price scenarios.
        
        Args:
            base_demand: Base demand
            demand_std: Standard deviation of demand
            price_elasticity: Price elasticity of demand
            price_changes: List of price change percentages
            n_simulations: Number of simulations per scenario
        
        Returns:
            DataFrame with demand simulations
        """
        logger.info(f"Simulating demand for {len(price_changes)} price scenarios...")
        
        results = []
        
        for price_change in price_changes:
            # Calculate demand change based on elasticity
            demand_change = -price_elasticity * price_change
            expected_demand = base_demand * (1 + demand_change)
            
            # Simulate demand with uncertainty
            simulated_demands = np.random.normal(expected_demand, demand_std, n_simulations)
            simulated_demands = np.maximum(simulated_demands, 0)  # No negative demand
            
            results.append({
                'price_change_pct': price_change,
                'expected_demand': float(expected_demand),
                'mean_demand': float(np.mean(simulated_demands)),
                'std_demand': float(np.std(simulated_demands)),
                'percentile_5': float(np.percentile(simulated_demands, 5)),
                'percentile_95': float(np.percentile(simulated_demands, 95))
            })
        
        results_df = pd.DataFrame(results)
        
        logger.info(f"Demand simulation complete for {len(results_df)} scenarios")
        
        return results_df
    
    def calculate_var(
        self,
        returns: np.ndarray,
        confidence_level: float = 0.95
    ) -> Dict:
        """
        Calculate Value at Risk (VaR).
        
        Args:
            returns: Array of returns
            confidence_level: Confidence level (e.g., 0.95 for 95% VaR)
        
        Returns:
            Dictionary with VaR metrics
        """
        logger.info(f"Calculating VaR at {confidence_level * 100}% confidence level...")
        
        # Calculate VaR
        var = np.percentile(returns, (1 - confidence_level) * 100)
        
        # Calculate Conditional VaR (Expected Shortfall)
        cvar = returns[returns <= var].mean()
        
        results = {
            'var': float(var),
            'cvar': float(cvar),
            'confidence_level': confidence_level,
            'n_observations': len(returns)
        }
        
        logger.info(f"VaR calculated: {var:.2%}, CVaR: {cvar:.2%}")
        
        return results
    
    def simulate_inventory_costs(
        self,
        demand_mean: float,
        demand_std: float,
        holding_cost: float,
        ordering_cost: float,
        stockout_cost: float,
        order_quantities: List[float],
        n_simulations: int = 10000
    ) -> pd.DataFrame:
        """
        Simulate inventory costs under different order quantities.
        
        Args:
            demand_mean: Mean demand
            demand_std: Standard deviation of demand
            holding_cost: Cost per unit held
            ordering_cost: Fixed ordering cost
            stockout_cost: Cost per stockout
            order_quantities: List of order quantities to test
            n_simulations: Number of simulations
        
        Returns:
            DataFrame with cost simulations
        """
        logger.info(f"Simulating inventory costs for {len(order_quantities)} order quantities...")
        
        results = []
        
        for order_qty in order_quantities:
            total_costs = []
            
            for _ in range(n_simulations):
                # Simulate demand
                demand = np.random.normal(demand_mean, demand_std)
                demand = max(demand, 0)
                
                # Calculate costs
                if demand <= order_qty:
                    # No stockout
                    holding_units = order_qty - demand
                    holding_cost_total = holding_units * holding_cost
                    stockout_cost_total = 0
                else:
                    # Stockout occurs
                    holding_cost_total = 0
                    stockout_units = demand - order_qty
                    stockout_cost_total = stockout_units * stockout_cost
                
                total_cost = ordering_cost + holding_cost_total + stockout_cost_total
                total_costs.append(total_cost)
            
            results.append({
                'order_quantity': order_qty,
                'mean_cost': float(np.mean(total_costs)),
                'std_cost': float(np.std(total_costs)),
                'percentile_5': float(np.percentile(total_costs, 5)),
                'percentile_95': float(np.percentile(total_costs, 95))
            })
        
        results_df = pd.DataFrame(results)
        
        logger.info(f"Inventory cost simulation complete")
        
        return results_df
    
    def sensitivity_analysis(
        self,
        base_value: float,
        parameter_ranges: Dict[str, Tuple[float, float]],
        model_function: Callable,
        n_samples: int = 1000
    ) -> pd.DataFrame:
        """
        Perform sensitivity analysis on model parameters.
        
        Args:
            base_value: Base value for the output
            parameter_ranges: Dictionary mapping parameter names to (min, max) ranges
            model_function: Function that takes parameters and returns output
            n_samples: Number of samples
        
        Returns:
            DataFrame with sensitivity results
        """
        logger.info(f"Performing sensitivity analysis on {len(parameter_ranges)} parameters...")
        
        # Generate random samples for each parameter
        samples = {}
        for param, (min_val, max_val) in parameter_ranges.items():
            samples[param] = np.random.uniform(min_val, max_val, n_samples)
        
        # Calculate outputs for each sample
        outputs = []
        for i in range(n_samples):
            params = {param: samples[param][i] for param in parameter_ranges}
            output = model_function(params)
            outputs.append(output)
        
        outputs = np.array(outputs)
        
        # Calculate sensitivity indices (simple correlation-based)
        sensitivity_results = []
        for param in parameter_ranges:
            correlation = np.corrcoef(samples[param], outputs)[0, 1]
            sensitivity_results.append({
                'parameter': param,
                'correlation': float(correlation),
                'absolute_correlation': float(abs(correlation))
            })
        
        sensitivity_df = pd.DataFrame(sensitivity_results)
        sensitivity_df = sensitivity_df.sort_values('absolute_correlation', ascending=False)
        
        results = {
            'sensitivity': sensitivity_df,
            'mean_output': float(np.mean(outputs)),
            'std_output': float(np.std(outputs)),
            'n_samples': n_samples
        }
        
        logger.info(f"Sensitivity analysis complete")
        
        return results
    
    def simulate_clv_scenarios(
        self,
        customer_segments: Dict[str, Dict],
        n_simulations: int = 10000
    ) -> pd.DataFrame:
        """
        Simulate Customer Lifetime Value across scenarios.
        
        Args:
            customer_segments: Dictionary of segment parameters
            n_simulations: Number of simulations per segment
        
        Returns:
            DataFrame with CLV simulations
        """
        logger.info(f"Simulating CLV for {len(customer_segments)} segments...")
        
        results = []
        
        for segment, params in customer_segments.items():
            # Extract parameters
            avg_purchase_value = params.get('avg_purchase_value', 50)
            purchase_frequency = params.get('purchase_frequency', 12)  # per year
            retention_rate = params.get('retention_rate', 0.8)
            customer_lifetime_years = params.get('customer_lifetime_years', 5)
            discount_rate = params.get('discount_rate', 0.1)
            
            clv_values = []
            
            for _ in range(n_simulations):
                # Simulate with uncertainty
                simulated_purchase_value = np.random.lognormal(
                    np.log(avg_purchase_value), 0.2
                )
                simulated_frequency = np.random.poisson(purchase_frequency)
                simulated_retention = np.random.beta(retention_rate * 10, (1 - retention_rate) * 10)
                
                # Calculate CLV
                clv = 0
                for year in range(1, customer_lifetime_years + 1):
                    if np.random.random() < simulated_retention:
                        annual_value = simulated_purchase_value * simulated_frequency
                        discounted_value = annual_value / ((1 + discount_rate) ** year)
                        clv += discounted_value
                    else:
                        break
                
                clv_values.append(clv)
            
            results.append({
                'segment': segment,
                'mean_clv': float(np.mean(clv_values)),
                'median_clv': float(np.median(clv_values)),
                'std_clv': float(np.std(clv_values)),
                'percentile_5': float(np.percentile(clv_values, 5)),
                'percentile_95': float(np.percentile(clv_values, 95))
            })
        
        results_df = pd.DataFrame(results)
        
        logger.info(f"CLV simulation complete for {len(results_df)} segments")
        
        return results_df


def run_monte_carlo_pipeline(
    base_revenue: float = 1000000,
    revenue_std: float = 100000,
    n_simulations: int = 10000
) -> Tuple[MonteCarloSimulator, Dict]:
    """
    Convenience function to run complete Monte Carlo pipeline.
    
    Args:
        base_revenue: Base revenue
        revenue_std: Revenue standard deviation
        n_simulations: Number of simulations
    
    Returns:
        Tuple of (simulator, results)
    """
    simulator = MonteCarloSimulator()
    
    # Simulate revenue
    revenue_results = simulator.simulate_revenue(base_revenue, revenue_std, n_simulations)
    
    # Calculate VaR from revenue simulations
    returns = (revenue_results['simulations'] - base_revenue) / base_revenue
    var_results = simulator.calculate_var(returns)
    
    results = {
        'revenue_simulation': revenue_results,
        'var_analysis': var_results
    }
    
    return simulator, results
