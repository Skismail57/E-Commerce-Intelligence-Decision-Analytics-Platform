"""
Inventory Optimization Module
Implements EOQ (Economic Order Quantity) and safety stock calculations for optimal inventory management.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from scipy import stats
from config.logging_config import get_logger

logger = get_logger(__name__)


class InventoryOptimizer:
    """
    Inventory optimization engine for optimal stock levels.
    
    Features:
    - Economic Order Quantity (EOQ) calculation
    - Safety stock calculation with service level targets
    - Reorder point determination
    - Inventory cost analysis
    - Multi-period inventory planning
    """
    
    def __init__(self):
        """Initialize inventory optimizer"""
        self.inventory_policies = {}
    
    def calculate_eoq(
        self,
        annual_demand: float,
        ordering_cost: float,
        holding_cost_per_unit: float
    ) -> Dict:
        """
        Calculate Economic Order Quantity (EOQ).
        
        Args:
            annual_demand: Annual demand in units
            ordering_cost: Cost per order
            holding_cost_per_unit: Annual holding cost per unit
        
        Returns:
            Dictionary with EOQ and related metrics
        """
        logger.info("Calculating EOQ...")
        
        # EOQ formula: sqrt(2 * D * S / H)
        # D = annual demand, S = ordering cost, H = holding cost per unit
        eoq = np.sqrt(2 * annual_demand * ordering_cost / holding_cost_per_unit)
        
        # Calculate related metrics
        orders_per_year = annual_demand / eoq
        time_between_orders = 365 / orders_per_year  # days
        total_ordering_cost = orders_per_year * ordering_cost
        total_holding_cost = (eoq / 2) * holding_cost_per_unit
        total_cost = total_ordering_cost + total_holding_cost
        
        results = {
            'eoq': float(eoq),
            'orders_per_year': float(orders_per_year),
            'time_between_orders_days': float(time_between_orders),
            'total_ordering_cost': float(total_ordering_cost),
            'total_holding_cost': float(total_holding_cost),
            'total_annual_cost': float(total_cost),
            'annual_demand': annual_demand,
            'ordering_cost': ordering_cost,
            'holding_cost_per_unit': holding_cost_per_unit
        }
        
        logger.info(f"EOQ: {eoq:.2f}, Total annual cost: {total_cost:.2f}")
        
        return results
    
    def calculate_safety_stock(
        self,
        daily_demand: float,
        demand_std: float,
        lead_time_days: int,
        lead_time_std: float,
        service_level: float = 0.95
    ) -> Dict:
        """
        Calculate safety stock with service level target.
        
        Args:
            daily_demand: Average daily demand
            demand_std: Standard deviation of daily demand
            lead_time_days: Average lead time in days
            lead_time_std: Standard deviation of lead time
            service_level: Target service level (e.g., 0.95 for 95%)
        
        Returns:
            Dictionary with safety stock and related metrics
        """
        logger.info(f"Calculating safety stock at {service_level:.0%} service level...")
        
        # Z-score for service level
        z_score = stats.norm.ppf(service_level)
        
        # Safety stock formula with demand and lead time uncertainty
        # SS = z * sqrt(L * sigma_d^2 + D^2 * sigma_L^2)
        # L = lead time, sigma_d = demand std, D = daily demand, sigma_L = lead time std
        safety_stock = z_score * np.sqrt(
            lead_time_days * demand_std**2 + daily_demand**2 * lead_time_std**2
        )
        
        # Simplified version (only demand uncertainty)
        safety_stock_demand_only = z_score * demand_std * np.sqrt(lead_time_days)
        
        # Reorder point
        reorder_point = daily_demand * lead_time_days + safety_stock
        
        # Calculate expected stockout probability
        stockout_probability = 1 - service_level
        
        results = {
            'safety_stock': float(safety_stock),
            'safety_stock_demand_only': float(safety_stock_demand_only),
            'reorder_point': float(reorder_point),
            'z_score': float(z_score),
            'service_level': service_level,
            'stockout_probability': float(stockout_probability),
            'daily_demand': daily_demand,
            'demand_std': demand_std,
            'lead_time_days': lead_time_days,
            'lead_time_std': lead_time_std
        }
        
        logger.info(f"Safety stock: {safety_stock:.2f}, Reorder point: {reorder_point:.2f}")
        
        return results
    
    def calculate_reorder_point(
        self,
        daily_demand: float,
        lead_time_days: int,
        safety_stock: float
    ) -> float:
        """
        Calculate reorder point.
        
        Args:
            daily_demand: Average daily demand
            lead_time_days: Lead time in days
            safety_stock: Safety stock level
        
        Returns:
            Reorder point
        """
        reorder_point = daily_demand * lead_time_days + safety_stock
        return float(reorder_point)
    
    def optimize_inventory_policy(
        self,
        product_id: int,
        demand_data: pd.DataFrame,
        ordering_cost: float,
        holding_cost_per_unit: float,
        lead_time_days: int,
        lead_time_std: float,
        service_level: float = 0.95
    ) -> Dict:
        """
        Calculate complete inventory policy for a product.
        
        Args:
            product_id: Product ID
            demand_data: DataFrame with historical demand data
            ordering_cost: Cost per order
            holding_cost_per_unit: Annual holding cost per unit
            lead_time_days: Average lead time
            lead_time_std: Standard deviation of lead time
            service_level: Target service level
        
        Returns:
            Dictionary with complete inventory policy
        """
        logger.info(f"Optimizing inventory policy for product {product_id}...")
        
        # Calculate demand statistics
        daily_demand = demand_data['quantity'].mean()
        demand_std = demand_data['quantity'].std()
        annual_demand = daily_demand * 365
        
        # Calculate EOQ
        eoq_results = self.calculate_eoq(annual_demand, ordering_cost, holding_cost_per_unit)
        
        # Calculate safety stock
        safety_stock_results = self.calculate_safety_stock(
            daily_demand, demand_std, lead_time_days, lead_time_std, service_level
        )
        
        # Calculate reorder point
        reorder_point = self.calculate_reorder_point(
            daily_demand, lead_time_days, safety_stock_results['safety_stock']
        )
        
        # Combine results
        policy = {
            'product_id': product_id,
            'eoq': eoq_results['eoq'],
            'safety_stock': safety_stock_results['safety_stock'],
            'reorder_point': reorder_point,
            'service_level': service_level,
            'orders_per_year': eoq_results['orders_per_year'],
            'total_annual_cost': eoq_results['total_annual_cost'],
            'daily_demand': daily_demand,
            'demand_std': demand_std,
            'lead_time_days': lead_time_days
        }
        
        # Store policy
        self.inventory_policies[product_id] = policy
        
        logger.info(f"Inventory policy for product {product_id}: EOQ={policy['eoq']:.0f}, ROP={policy['reorder_point']:.0f}")
        
        return policy
    
    def batch_optimize_inventory(
        self,
        products_df: pd.DataFrame,
        demand_data: pd.DataFrame,
        ordering_cost: float,
        holding_cost_per_unit: float,
        lead_time_days: int,
        lead_time_std: float,
        service_level: float = 0.95
    ) -> pd.DataFrame:
        """
        Optimize inventory policies for multiple products.
        
        Args:
            products_df: DataFrame with product data
            demand_data: DataFrame with demand data
            ordering_cost: Cost per order
            holding_cost_per_unit: Annual holding cost per unit
            lead_time_days: Average lead time
            lead_time_std: Standard deviation of lead time
            service_level: Target service level
        
        Returns:
            DataFrame with inventory policies for all products
        """
        logger.info(f"Batch optimizing inventory for {len(products_df)} products...")
        
        policies = []
        
        for _, product in products_df.iterrows():
            product_id = product['product_id']
            
            # Get demand data for this product
            product_demand = demand_data[demand_data['product_id'] == product_id]
            
            if len(product_demand) < 10:
                logger.warning(f"Insufficient demand data for product {product_id}")
                continue
            
            # Optimize policy
            policy = self.optimize_inventory_policy(
                product_id, product_demand, ordering_cost,
                holding_cost_per_unit, lead_time_days, lead_time_std, service_level
            )
            
            policies.append(policy)
        
        policies_df = pd.DataFrame(policies)
        
        logger.info(f"Batch optimization complete for {len(policies_df)} products")
        
        return policies_df
    
    def calculate_inventory_kpis(
        self,
        current_inventory: pd.DataFrame,
        policies_df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Calculate inventory KPIs based on current inventory and policies.
        
        Args:
            current_inventory: DataFrame with current inventory levels
            policies_df: DataFrame with inventory policies
        
        Returns:
            DataFrame with inventory KPIs
        """
        logger.info("Calculating inventory KPIs...")
        
        # Merge current inventory with policies
        kpis_df = current_inventory.merge(policies_df, on='product_id', how='left')
        
        # Calculate days of stock remaining
        kpis_df['days_of_stock'] = kpis_df['current_stock'] / kpis_df['daily_demand']
        
        # Calculate stock status
        kpis_df['stock_status'] = np.where(
            kpis_df['current_stock'] < kpis_df['reorder_point'],
            'below_rop',
            np.where(
                kpis_df['current_stock'] < kpis_df['reorder_point'] + kpis_df['eoq'],
                'normal',
                'above_target'
            )
        )
        
        # Calculate stockout risk
        kpis_df['stockout_risk'] = np.where(
            kpis_df['current_stock'] < kpis_df['safety_stock'],
            'high',
            np.where(
                kpis_df['current_stock'] < kpis_df['reorder_point'],
                'medium',
                'low'
            )
        )
        
        # Calculate holding cost
        kpis_df['annual_holding_cost'] = kpis_df['current_stock'] * kpis_df.get('holding_cost_per_unit', 10)
        
        # Calculate order urgency
        kpis_df['order_urgency'] = np.where(
            kpis_df['days_of_stock'] < kpis_df['lead_time_days'],
            'urgent',
            np.where(
                kpis_df['days_of_stock'] < kpis_df['lead_time_days'] * 2,
                'soon',
                'normal'
            )
        )
        
        logger.info(f"Calculated KPIs for {len(kpis_df)} products")
        
        return kpis_df
    
    def simulate_inventory_levels(
        self,
        product_id: int,
        initial_stock: int,
        daily_demand: float,
        demand_std: float,
        lead_time_days: int,
        eoq: int,
        reorder_point: int,
        simulation_days: int = 365
    ) -> pd.DataFrame:
        """
        Simulate inventory levels over time.
        
        Args:
            product_id: Product ID
            initial_stock: Initial inventory level
            daily_demand: Average daily demand
            demand_std: Standard deviation of demand
            lead_time_days: Lead time
            eoq: Economic order quantity
            reorder_point: Reorder point
            simulation_days: Number of days to simulate
        
        Returns:
            DataFrame with daily inventory levels
        """
        logger.info(f"Simulating inventory for product {product_id} over {simulation_days} days...")
        
        inventory = initial_stock
        on_order = 0
        order_arrival_day = None
        
        simulation_data = []
        
        for day in range(simulation_days):
            # Simulate demand (normal distribution)
            daily_demand_sim = max(0, np.random.normal(daily_demand, demand_std))
            
            # Check for order arrival
            if order_arrival_day == day:
                inventory += on_order
                on_order = 0
                order_arrival_day = None
            
            # Place order if below reorder point and no pending order
            if inventory < reorder_point and on_order == 0:
                on_order = eoq
                order_arrival_day = day + lead_time_days
            
            # Satisfy demand
            sales = min(inventory, daily_demand_sim)
            inventory -= sales
            stockout = daily_demand_sim - sales
            
            simulation_data.append({
                'day': day,
                'inventory': inventory,
                'on_order': on_order,
                'sales': sales,
                'stockout': stockout,
                'order_placed': on_order > 0 and order_arrival_day == day + lead_time_days
            })
        
        simulation_df = pd.DataFrame(simulation_data)
        
        # Calculate summary statistics
        total_sales = simulation_df['sales'].sum()
        total_stockouts = simulation_df['stockout'].sum()
        service_level = total_sales / (total_sales + total_stockouts) if (total_sales + total_stockouts) > 0 else 1
        
        logger.info(f"Simulation complete. Service level: {service_level:.2%}, Total stockouts: {total_stockouts:.0f}")
        
        return simulation_df
    
    def get_reorder_recommendations(
        self,
        kpis_df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Get reorder recommendations based on inventory KPIs.
        
        Args:
            kpis_df: DataFrame with inventory KPIs
        
        Returns:
            DataFrame with reorder recommendations
        """
        recommendations = kpis_df.copy()
        
        # Generate recommendations
        recommendations['action'] = np.where(
            kpis_df['stock_status'] == 'below_rop',
            'reorder_now',
            np.where(
                kpis_df['order_urgency'] == 'urgent',
                'reorder_now',
                np.where(
                    kpis_df['order_urgency'] == 'soon',
                    'plan_reorder',
                    'no_action'
                )
            )
        )
        
        # Calculate recommended order quantity
        recommendations['recommended_order_qty'] = np.where(
            kpis_df['action'] == 'reorder_now',
            kpis_df['eoq'],
            0
        )
        
        # Calculate expected delivery date
        recommendations['expected_delivery_days'] = np.where(
            kpis_df['action'] == 'reorder_now',
            kpis_df['lead_time_days'],
            0
        )
        
        return recommendations


def run_inventory_optimization_pipeline(
    products_df: pd.DataFrame,
    demand_data: pd.DataFrame,
    current_inventory: pd.DataFrame,
    ordering_cost: float = 50,
    holding_cost_per_unit: float = 10,
    lead_time_days: int = 7,
    lead_time_std: float = 2,
    service_level: float = 0.95
) -> Tuple[InventoryOptimizer, pd.DataFrame]:
    """
    Convenience function to run complete inventory optimization pipeline.
    
    Args:
        products_df: Product data
        demand_data: Demand data
        current_inventory: Current inventory levels
        ordering_cost: Cost per order
        holding_cost_per_unit: Annual holding cost per unit
        lead_time_days: Average lead time
        lead_time_std: Standard deviation of lead time
        service_level: Target service level
    
    Returns:
        Tuple of (optimizer, recommendations)
    """
    optimizer = InventoryOptimizer()
    
    # Optimize policies
    policies_df = optimizer.batch_optimize_inventory(
        products_df, demand_data, ordering_cost, holding_cost_per_unit,
        lead_time_days, lead_time_std, service_level
    )
    
    # Calculate KPIs
    kpis_df = optimizer.calculate_inventory_kpis(current_inventory, policies_df)
    
    # Get recommendations
    recommendations = optimizer.get_reorder_recommendations(kpis_df)
    
    return optimizer, recommendations
