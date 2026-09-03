"""
What-If Analysis Engine Module
Implements what-if scenario analysis for business decision support.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Callable
from config.logging_config import get_logger

logger = get_logger(__name__)


class WhatIfAnalyzer:
    """
    What-if analysis engine for scenario planning.
    
    Features:
    - Pricing scenario analysis
    - Marketing campaign impact simulation
    - Inventory level scenarios
    - Customer retention scenarios
    - Multi-variable scenario analysis
    """
    
    def __init__(self):
        """Initialize what-if analyzer"""
        self.scenarios = {}
        logger.info("What-if analyzer initialized")
    
    def analyze_pricing_scenario(
        self,
        current_price: float,
        current_demand: float,
        price_elasticity: float,
        new_price: float,
        cost_per_unit: float = None
    ) -> Dict:
        """
        Analyze impact of price change.
        
        Args:
            current_price: Current price
            current_demand: Current demand
            price_elasticity: Price elasticity of demand
            new_price: New price to analyze
            cost_per_unit: Cost per unit (optional)
        
        Returns:
            Dictionary with scenario results
        """
        logger.info(f"Analyzing pricing scenario: ${current_price} -> ${new_price}")
        
        # Calculate price change percentage
        price_change_pct = (new_price - current_price) / current_price
        
        # Calculate demand change based on elasticity
        demand_change_pct = -price_elasticity * price_change_pct
        new_demand = current_demand * (1 + demand_change_pct)
        
        # Calculate revenue
        current_revenue = current_price * current_demand
        new_revenue = new_price * new_demand
        revenue_change = new_revenue - current_revenue
        revenue_change_pct = revenue_change / current_revenue if current_revenue > 0 else 0
        
        # Calculate profit if cost provided
        profit_change = None
        profit_change_pct = None
        if cost_per_unit is not None:
            current_profit = (current_price - cost_per_unit) * current_demand
            new_profit = (new_price - cost_per_unit) * new_demand
            profit_change = new_profit - current_profit
            profit_change_pct = profit_change / current_profit if current_profit > 0 else 0
        
        results = {
            'current_price': current_price,
            'new_price': new_price,
            'price_change_pct': float(price_change_pct * 100),
            'current_demand': current_demand,
            'new_demand': float(new_demand),
            'demand_change_pct': float(demand_change_pct * 100),
            'current_revenue': float(current_revenue),
            'new_revenue': float(new_revenue),
            'revenue_change': float(revenue_change),
            'revenue_change_pct': float(revenue_change_pct * 100),
            'profit_change': float(profit_change) if profit_change is not None else None,
            'profit_change_pct': float(profit_change_pct * 100) if profit_change_pct is not None else None
        }
        
        logger.info(f"Pricing scenario analysis complete. Revenue change: {revenue_change_pct:.1%}")
        
        return results
    
    def analyze_marketing_scenario(
        self,
        current_customers: int,
        current_revenue: float,
        marketing_spend: float,
        expected_acquisition_rate: float,
        expected_ltv: float = 100
    ) -> Dict:
        """
        Analyze impact of marketing campaign.
        
        Args:
            current_customers: Current number of customers
            current_revenue: Current revenue
            marketing_spend: Marketing spend
            expected_acquisition_rate: Expected customers acquired per $1000 spend
            expected_ltv: Expected lifetime value per new customer
        
        Returns:
            Dictionary with scenario results
        """
        logger.info(f"Analyzing marketing scenario with ${marketing_spend} spend")
        
        # Calculate expected new customers
        new_customers = (marketing_spend / 1000) * expected_acquisition_rate
        
        # Calculate expected revenue from new customers
        new_revenue = new_customers * expected_ltv
        
        # Calculate ROI
        roi = (new_revenue - marketing_spend) / marketing_spend if marketing_spend > 0 else 0
        
        # Calculate total impact
        total_customers = current_customers + new_customers
        total_revenue = current_revenue + new_revenue
        
        results = {
            'marketing_spend': marketing_spend,
            'expected_new_customers': float(new_customers),
            'expected_ltv': expected_ltv,
            'expected_new_revenue': float(new_revenue),
            'roi': float(roi),
            'current_customers': current_customers,
            'total_customers': float(total_customers),
            'customer_growth_pct': float((new_customers / current_customers) * 100) if current_customers > 0 else 0,
            'current_revenue': current_revenue,
            'total_revenue': float(total_revenue),
            'revenue_growth_pct': float((new_revenue / current_revenue) * 100) if current_revenue > 0 else 0
        }
        
        logger.info(f"Marketing scenario analysis complete. ROI: {roi:.1%}")
        
        return results
    
    def analyze_inventory_scenario(
        self,
        current_stock: int,
        daily_demand: int,
        lead_time_days: int,
        service_level: float = 0.95,
        demand_std: float = 10
    ) -> Dict:
        """
        Analyze inventory scenario with different service levels.
        
        Args:
            current_stock: Current stock level
            daily_demand: Average daily demand
            lead_time_days: Lead time in days
            service_level: Desired service level
            demand_std: Standard deviation of daily demand
        
        Returns:
            Dictionary with inventory scenario results
        """
        logger.info(f"Analyzing inventory scenario at {service_level:.0%} service level")
        
        from scipy import stats
        
        # Calculate safety stock
        z_score = stats.norm.ppf(service_level)
        safety_stock = z_score * demand_std * np.sqrt(lead_time_days)
        
        # Calculate reorder point
        reorder_point = (daily_demand * lead_time_days) + safety_stock
        
        # Calculate economic order quantity (simplified)
        holding_cost_pct = 0.25  # 25% annual holding cost
        ordering_cost = 50  # Fixed ordering cost
        annual_demand = daily_demand * 365
        
        eoq = np.sqrt((2 * ordering_cost * annual_demand) / (holding_cost_pct * daily_demand))
        
        # Calculate stockout probability
        if current_stock > 0:
            stockout_prob = 1 - stats.norm.cdf((current_stock - daily_demand * lead_time_days) / 
                                               (demand_std * np.sqrt(lead_time_days)))
        else:
            stockout_prob = 1.0
        
        results = {
            'current_stock': current_stock,
            'daily_demand': daily_demand,
            'lead_time_days': lead_time_days,
            'service_level': service_level,
            'safety_stock': float(safety_stock),
            'reorder_point': float(reorder_point),
            'eoq': float(eoq),
            'stockout_probability': float(stockout_prob),
            'z_score': float(z_score)
        }
        
        logger.info(f"Inventory scenario analysis complete. Safety stock: {safety_stock:.0f}")
        
        return results
    
    def analyze_retention_scenario(
        self,
        current_customers: int,
        current_retention_rate: float,
        avg_revenue_per_customer: float,
        new_retention_rate: float,
        time_horizon_months: int = 12
    ) -> Dict:
        """
        Analyze impact of retention rate improvement.
        
        Args:
            current_customers: Current number of customers
            current_retention_rate: Current monthly retention rate
            avg_revenue_per_customer: Average monthly revenue per customer
            new_retention_rate: New retention rate to analyze
            time_horizon_months: Time horizon for analysis
        
        Returns:
            Dictionary with retention scenario results
        """
        logger.info(f"Analyzing retention scenario: {current_retention_rate:.1%} -> {new_retention_rate:.1%}")
        
        # Simulate customer base over time
        current_customers_over_time = []
        new_customers_over_time = []
        current_revenue_over_time = []
        new_revenue_over_time = []
        
        customers_current = current_customers
        customers_new = current_customers
        
        for month in range(time_horizon_months):
            # Current scenario
            customers_current = customers_current * current_retention_rate
            current_customers_over_time.append(customers_current)
            current_revenue_over_time.append(customers_current * avg_revenue_per_customer)
            
            # New scenario
            customers_new = customers_new * new_retention_rate
            new_customers_over_time.append(customers_new)
            new_revenue_over_time.append(customers_new * avg_revenue_per_customer)
        
        # Calculate total revenue difference
        total_current_revenue = sum(current_revenue_over_time)
        total_new_revenue = sum(new_revenue_over_time)
        revenue_lift = total_new_revenue - total_current_revenue
        revenue_lift_pct = revenue_lift / total_current_revenue if total_current_revenue > 0 else 0
        
        results = {
            'current_retention_rate': current_retention_rate,
            'new_retention_rate': new_retention_rate,
            'retention_improvement': float((new_retention_rate - current_retention_rate) * 100),
            'time_horizon_months': time_horizon_months,
            'total_current_revenue': float(total_current_revenue),
            'total_new_revenue': float(total_new_revenue),
            'revenue_lift': float(revenue_lift),
            'revenue_lift_pct': float(revenue_lift_pct * 100),
            'final_customers_current': float(current_customers_over_time[-1]),
            'final_customers_new': float(new_customers_over_time[-1]),
            'customer_retention_lift': float(new_customers_over_time[-1] - current_customers_over_time[-1])
        }
        
        logger.info(f"Retention scenario analysis complete. Revenue lift: {revenue_lift_pct:.1%}")
        
        return results
    
    def analyze_multi_variable_scenario(
        self,
        base_metrics: Dict[str, float],
        scenario_changes: Dict[str, float],
        impact_model: Callable
    ) -> Dict:
        """
        Analyze scenario with multiple variable changes.
        
        Args:
            base_metrics: Dictionary of base metric values
            scenario_changes: Dictionary of percentage changes for each metric
            impact_model: Function that calculates impact from metrics
        
        Returns:
            Dictionary with multi-variable scenario results
        """
        logger.info(f"Analyzing multi-variable scenario with {len(scenario_changes)} changes")
        
        # Calculate new metric values
        new_metrics = {}
        for metric, base_value in base_metrics.items():
            if metric in scenario_changes:
                change_pct = scenario_changes[metric]
                new_metrics[metric] = base_value * (1 + change_pct)
            else:
                new_metrics[metric] = base_value
        
        # Calculate base and new impacts
        base_impact = impact_model(base_metrics)
        new_impact = impact_model(new_metrics)
        
        # Calculate change
        impact_change = new_impact - base_impact
        impact_change_pct = impact_change / base_impact if base_impact != 0 else 0
        
        results = {
            'base_metrics': base_metrics,
            'new_metrics': new_metrics,
            'scenario_changes': scenario_changes,
            'base_impact': float(base_impact),
            'new_impact': float(new_impact),
            'impact_change': float(impact_change),
            'impact_change_pct': float(impact_change_pct * 100)
        }
        
        logger.info(f"Multi-variable scenario analysis complete. Impact change: {impact_change_pct:.1%}")
        
        return results
    
    def compare_scenarios(
        self,
        scenarios: List[Dict],
        metric_name: str = 'revenue'
    ) -> pd.DataFrame:
        """
        Compare multiple scenarios.
        
        Args:
            scenarios: List of scenario result dictionaries
            metric_name: Metric to compare
        
        Returns:
            DataFrame with scenario comparison
        """
        logger.info(f"Comparing {len(scenarios)} scenarios")
        
        comparison_data = []
        
        for i, scenario in enumerate(scenarios):
            comparison_data.append({
                'scenario_id': i,
                'scenario_name': scenario.get('name', f'Scenario {i}'),
                metric_name: scenario.get(metric_name, 0)
            })
        
        comparison_df = pd.DataFrame(comparison_data)
        comparison_df = comparison_df.sort_values(metric_name, ascending=False)
        
        logger.info(f"Scenario comparison complete")
        
        return comparison_df
    
    def generate_scenario_report(
        self,
        scenarios: List[Dict],
        base_scenario: Dict
    ) -> Dict:
        """
        Generate comprehensive scenario report.
        
        Args:
            scenarios: List of scenario results
            base_scenario: Base scenario for comparison
        
        Returns:
            Dictionary with scenario report
        """
        logger.info("Generating scenario report")
        
        report = {
            'base_scenario': base_scenario,
            'n_scenarios': len(scenarios),
            'scenarios': scenarios,
            'best_scenario': max(scenarios, key=lambda x: x.get('revenue', 0)) if scenarios else None,
            'worst_scenario': min(scenarios, key=lambda x: x.get('revenue', 0)) if scenarios else None
        }
        
        logger.info(f"Scenario report generated for {len(scenarios)} scenarios")
        
        return report


def run_what_if_pipeline(
    current_price: float = 100,
    current_demand: float = 1000,
    price_elasticity: float = 1.5,
    price_scenarios: List[float] = None
) -> Tuple[WhatIfAnalyzer, Dict]:
    """
    Convenience function to run what-if analysis pipeline.
    
    Args:
        current_price: Current price
        current_demand: Current demand
        price_elasticity: Price elasticity
        price_scenarios: List of price scenarios to analyze
    
    Returns:
        Tuple of (analyzer, results)
    """
    analyzer = WhatIfAnalyzer()
    
    if price_scenarios is None:
        price_scenarios = [90, 95, 105, 110]
    
    # Analyze pricing scenarios
    pricing_results = []
    for new_price in price_scenarios:
        result = analyzer.analyze_pricing_scenario(
            current_price, current_demand, price_elasticity, new_price
        )
        result['name'] = f'Price ${new_price}'
        pricing_results.append(result)
    
    # Compare scenarios
    comparison = analyzer.compare_scenarios(pricing_results)
    
    # Generate report
    report = analyzer.generate_scenario_report(pricing_results, {'price': current_price, 'demand': current_demand})
    
    results = {
        'pricing_scenarios': pricing_results,
        'scenario_comparison': comparison,
        'scenario_report': report
    }
    
    return analyzer, results
