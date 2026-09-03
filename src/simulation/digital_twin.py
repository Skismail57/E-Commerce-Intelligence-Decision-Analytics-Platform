"""
Digital Twin Implementation Module
Creates a simulation model of the e-commerce business for what-if analysis.

Architecture:
- Agent-based modeling of customers, products, and inventory
- Demand simulation with seasonality and trends
- Supply chain simulation with lead times
- What-if scenario testing (price changes, promotions, disruptions)
- KPI forecasting under different scenarios
"""

from typing import Dict, List, Optional, Tuple
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path

from config.logging_config import get_logger
from config.settings import settings

logger = get_logger(__name__)


class DigitalTwin:
    """
    Digital twin of the e-commerce business.
    
    Simulates the business to test scenarios:
    - What if we increase prices by 10%?
    - What if a supplier is disrupted for 2 weeks?
    - What if we run a promotion?
    - What if demand increases by 20%?
    """
    
    def __init__(self, start_date: str = "2024-01-01"):
        """
        Initialize digital twin.
        
        Args:
            start_date: Simulation start date
        """
        self.start_date = pd.to_datetime(start_date)
        self.current_date = self.start_date
        self.state = {}
        logger.info(f"Digital Twin initialized with start date: {self.start_date}")
    
    def initialize_state(
        self,
        customers_df: pd.DataFrame,
        products_df: pd.DataFrame,
        inventory_df: pd.DataFrame,
        orders_df: pd.DataFrame
    ) -> None:
        """
        Initialize digital twin state from historical data.
        
        Args:
            customers_df: Customer data
            products_df: Product data
            inventory_df: Inventory data
            orders_df: Historical orders
        """
        logger.info("Initializing digital twin state...")
        
        self.state = {
            'customers': customers_df.copy(),
            'products': products_df.copy(),
            'inventory': inventory_df.copy(),
            'historical_orders': orders_df.copy(),
            'current_orders': pd.DataFrame(),
            'revenue': 0,
            'orders_count': 0,
            'customers_active': 0
        }
        
        # Calculate baseline metrics
        self.state['baseline_daily_revenue'] = orders_df.groupby(
            pd.to_datetime(orders_df['order_date']).dt.date
        )['order_total'].sum().mean()
        
        self.state['baseline_daily_orders'] = orders_df.groupby(
            pd.to_datetime(orders_df['order_date']).dt.date
        )['order_id'].count().mean()
        
        logger.info("Digital twin state initialized")
    
    def simulate_demand(
        self,
        date: pd.Timestamp,
        scenario_params: Optional[Dict] = None
    ) -> float:
        """
        Simulate demand for a given date.
        
        Args:
            date: Date to simulate
            scenario_params: Scenario parameters (price_change, demand_multiplier, etc.)
        
        Returns:
            Simulated demand (number of orders)
        """
        scenario_params = scenario_params or {}
        
        # Base demand from historical patterns
        day_of_week = date.dayofweek
        day_of_month = date.day
        month = date.month
        
        # Seasonality factors (simplified)
        seasonality = 1.0
        if month in [11, 12]:  # Holiday season
            seasonality = 1.5
        elif month in [1, 2]:  # Post-holiday
            seasonality = 0.8
        
        # Day of week factors
        dow_factors = {0: 0.8, 1: 1.0, 2: 1.0, 3: 1.0, 4: 1.1, 5: 1.2, 6: 0.9}
        dow_factor = dow_factors.get(day_of_week, 1.0)
        
        # Scenario multipliers
        demand_multiplier = scenario_params.get('demand_multiplier', 1.0)
        price_change = scenario_params.get('price_change', 0)
        
        # Price elasticity (simplified)
        price_elasticity = -1.5
        price_factor = 1 + (price_change * price_elasticity / 100)
        
        # Calculate simulated demand
        base_demand = self.state.get('baseline_daily_orders', 100)
        simulated_demand = base_demand * seasonality * dow_factor * demand_multiplier * price_factor
        
        # Add random noise
        noise = np.random.normal(0, 0.1 * simulated_demand)
        simulated_demand = max(0, simulated_demand + noise)
        
        return simulated_demand
    
    def simulate_orders(
        self,
        date: pd.Timestamp,
        demand: float,
        scenario_params: Optional[Dict] = None
    ) -> pd.DataFrame:
        """
        Simulate orders for a given date.
        
        Args:
            date: Date to simulate
            demand: Number of orders to simulate
            scenario_params: Scenario parameters
        
        Returns:
            DataFrame with simulated orders
        """
        scenario_params = scenario_params or {}
        
        orders = []
        n_orders = int(demand)
        
        for i in range(n_orders):
            # Random customer
            customer_id = np.random.choice(self.state['customers']['customer_id'])
            
            # Random product
            product_id = np.random.choice(self.state['products']['product_id'])
            
            # Get product price
            product = self.state['products'][self.state['products']['product_id'] == product_id].iloc[0]
            base_price = product['selling_price']
            
            # Apply price change from scenario
            price_change = scenario_params.get('price_change', 0)
            price = base_price * (1 + price_change / 100)
            
            # Random quantity
            quantity = np.random.poisson(2) + 1
            
            # Calculate order total
            order_total = price * quantity
            
            orders.append({
                'order_id': f"sim_{date.strftime('%Y%m%d')}_{i}",
                'customer_id': customer_id,
                'product_id': product_id,
                'order_date': date,
                'quantity': quantity,
                'unit_price': price,
                'order_total': order_total,
                'order_status': 'Delivered'
            })
        
        return pd.DataFrame(orders)
    
    def simulate_inventory_update(
        self,
        orders_df: pd.DataFrame,
        scenario_params: Optional[Dict] = None
    ) -> pd.DataFrame:
        """
        Simulate inventory updates after orders.
        
        Args:
            orders_df: Orders to process
            scenario_params: Scenario parameters (supply_disruption, etc.)
        
        Returns:
            Updated inventory DataFrame
        """
        scenario_params = scenario_params or {}
        
        inventory = self.state['inventory'].copy()
        
        # Apply supply disruption
        supply_disruption = scenario_params.get('supply_disruption', False)
        if supply_disruption:
            # Reduce inventory replenishment
            inventory['units_in_stock'] = inventory['units_in_stock'] * 0.9
        
        # Deduct inventory for orders
        for _, order in orders_df.iterrows():
            product_id = order['product_id']
            quantity = order['quantity']
            
            idx = inventory[inventory['product_id'] == product_id].index
            if len(idx) > 0:
                inventory.loc[idx, 'units_in_stock'] -= quantity
        
        # Ensure no negative inventory
        inventory['units_in_stock'] = inventory['units_in_stock'].clip(lower=0)
        
        return inventory
    
    def simulate_day(
        self,
        date: pd.Timestamp,
        scenario_params: Optional[Dict] = None
    ) -> Dict[str, float]:
        """
        Simulate a single day of operations.
        
        Args:
            date: Date to simulate
            scenario_params: Scenario parameters
        
        Returns:
            Dictionary with daily metrics
        """
        # Simulate demand
        demand = self.simulate_demand(date, scenario_params)
        
        # Simulate orders
        orders = self.simulate_orders(date, demand, scenario_params)
        
        # Update inventory
        inventory = self.simulate_inventory_update(orders, scenario_params)
        self.state['inventory'] = inventory
        
        # Calculate metrics
        daily_revenue = orders['order_total'].sum()
        daily_orders = len(orders)
        daily_customers = orders['customer_id'].nunique()
        
        # Update state
        self.state['current_orders'] = pd.concat([
            self.state['current_orders'], orders
        ], ignore_index=True)
        self.state['revenue'] += daily_revenue
        self.state['orders_count'] += daily_orders
        self.state['customers_active'] = daily_customers
        
        return {
            'date': date,
            'demand': demand,
            'orders': daily_orders,
            'revenue': daily_revenue,
            'customers': daily_customers
        }
    
    def simulate_scenario(
        self,
        scenario_name: str,
        scenario_params: Dict,
        duration_days: int = 30
    ) -> pd.DataFrame:
        """
        Simulate a complete scenario over multiple days.
        
        Args:
            scenario_name: Name of the scenario
            scenario_params: Scenario parameters
            duration_days: Number of days to simulate
        
        Returns:
            DataFrame with daily simulation results
        """
        logger.info(f"Simulating scenario: {scenario_name} for {duration_days} days")
        
        results = []
        current_date = self.current_date
        
        for day in range(duration_days):
            date = current_date + timedelta(days=day)
            
            daily_metrics = self.simulate_day(date, scenario_params)
            daily_metrics['scenario'] = scenario_name
            daily_metrics['day'] = day + 1
            
            results.append(daily_metrics)
        
        results_df = pd.DataFrame(results)
        
        # Calculate scenario summary
        summary = {
            'scenario': scenario_name,
            'total_revenue': results_df['revenue'].sum(),
            'total_orders': results_df['orders'].sum(),
            'avg_daily_revenue': results_df['revenue'].mean(),
            'avg_daily_orders': results_df['orders'].mean(),
            'duration_days': duration_days
        }
        
        logger.info(f"Scenario {scenario_name} complete: ₹{summary['total_revenue']:,.2f} revenue")
        
        return results_df
    
    def compare_scenarios(
        self,
        scenarios: Dict[str, Dict],
        duration_days: int = 30
    ) -> pd.DataFrame:
        """
        Compare multiple scenarios.
        
        Args:
            scenarios: Dictionary mapping scenario name to parameters
            duration_days: Number of days to simulate each scenario
        
        Returns:
            DataFrame with scenario comparison
        """
        logger.info(f"Comparing {len(scenarios)} scenarios...")
        
        comparison_results = []
        
        for scenario_name, scenario_params in scenarios.items():
            # Reset state for each scenario
            self.current_date = self.start_date
            self.state['current_orders'] = pd.DataFrame()
            self.state['revenue'] = 0
            self.state['orders_count'] = 0
            
            # Simulate scenario
            results_df = self.simulate_scenario(scenario_name, scenario_params, duration_days)
            
            # Calculate summary
            summary = {
                'scenario': scenario_name,
                'total_revenue': results_df['revenue'].sum(),
                'total_orders': results_df['orders'].sum(),
                'avg_daily_revenue': results_df['revenue'].mean(),
                'avg_daily_orders': results_df['orders'].mean()
            }
            comparison_results.append(summary)
        
        comparison_df = pd.DataFrame(comparison_results)
        
        # Calculate percentage changes vs baseline
        if 'baseline' in comparison_df['scenario'].values:
            baseline = comparison_df[comparison_df['scenario'] == 'baseline'].iloc[0]
            comparison_df['revenue_change_pct'] = (
                (comparison_df['total_revenue'] - baseline['total_revenue']) / 
                baseline['total_revenue'] * 100
            )
        
        logger.info("Scenario comparison complete")
        return comparison_df
    
    def generate_what_if_report(
        self,
        comparison_df: pd.DataFrame,
        output_path: Optional[Path] = None
    ) -> str:
        """
        Generate what-if analysis report.
        
        Args:
            comparison_df: Scenario comparison results
            output_path: Optional path to save report
        
        Returns:
            Report string
        """
        report = f"""
Digital Twin What-If Analysis Report
{'=' * 60}

Scenario Comparison:
"""
        
        for _, row in comparison_df.iterrows():
            report += f"""
{row['scenario'].upper()}:
- Total Revenue: ₹{row['total_revenue']:,.2f}
- Total Orders: {row['total_orders']:.0f}
- Avg Daily Revenue: ₹{row['avg_daily_revenue']:,.2f}
- Avg Daily Orders: {row['avg_daily_orders']:.1f}
"""
            if 'revenue_change_pct' in row:
                change = row['revenue_change_pct']
                direction = "increase" if change > 0 else "decrease"
                report += f"- Revenue Change: {abs(change):.1f}% {direction}\n"
        
        report += f"""
Recommendations:
"""
        
        # Find best scenario
        if 'revenue_change_pct' in comparison_df.columns:
            best_scenario = comparison_df.loc[comparison_df['revenue_change_pct'].idxmax()]
            report += f"- Best Scenario: {best_scenario['scenario']} (+{best_scenario['revenue_change_pct']:.1f}% revenue)\n"
        
        report += f"""
Interpretation:
- Digital twin simulates business operations under different scenarios
- Compare revenue, orders, and other KPIs across scenarios
- Use results to inform strategic decisions
- Scenarios can include price changes, demand shifts, supply disruptions
"""
        
        if output_path:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(report)
            logger.info(f"What-if report saved to {output_path}")
        
        return report


def run_digital_twin_simulation(
    customers_df: pd.DataFrame,
    products_df: pd.DataFrame,
    inventory_df: pd.DataFrame,
    orders_df: pd.DataFrame,
    scenarios: Dict[str, Dict],
    duration_days: int = 30
) -> Dict[str, any]:
    """
    Convenience function to run digital twin simulation.
    
    Args:
        customers_df: Customer data
        products_df: Product data
        inventory_df: Inventory data
        orders_df: Historical orders
        scenarios: Dictionary of scenarios to simulate
        duration_days: Duration of each scenario
    
    Returns:
        Dictionary with simulation results
    """
    twin = DigitalTwin()
    
    # Initialize state
    twin.initialize_state(customers_df, products_df, inventory_df, orders_df)
    
    # Compare scenarios
    comparison = twin.compare_scenarios(scenarios, duration_days)
    
    return {
        'scenario_comparison': comparison,
        'digital_twin': twin
    }
