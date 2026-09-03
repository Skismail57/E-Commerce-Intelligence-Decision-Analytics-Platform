"""
Multi-Echelon Inventory Optimization Module
Implements inventory optimization across multiple supply chain echelons.

Architecture:
- Multi-echelon inventory model (warehouse → regional stores → local stores)
- Safety stock optimization with service level constraints
- Reorder point calculation with lead time variability
- Bullwhip effect mitigation
- Cross-docking optimization
"""

from typing import Dict, List, Optional, Tuple
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path

from config.logging_config import get_logger
from config.settings import settings

logger = get_logger(__name__)


class MultiEchelonInventoryOptimizer:
    """
    Multi-echelon inventory optimization.
    
    Optimizes inventory across multiple supply chain levels:
    - Central warehouse
    - Regional distribution centers
    - Retail stores
    
    Uses service level constraints and cost optimization.
    """
    
    def __init__(self):
        """Initialize multi-echelon inventory optimizer."""
        logger.info("Multi-Echelon Inventory Optimizer initialized")
    
    def calculate_safety_stock(
        self,
        demand_std: float,
        lead_time_std: float,
        average_demand: float,
        average_lead_time: float,
        service_level: float = 0.95
    ) -> float:
        """
        Calculate safety stock using service level approach.
        
        Safety Stock = Z * sqrt((LT * σd²) + (D² * σlt²))
        
        Args:
            demand_std: Standard deviation of demand
            lead_time_std: Standard deviation of lead time
            average_demand: Average demand per period
            average_lead_time: Average lead time in periods
            service_level: Target service level (0-1)
        
        Returns:
            Safety stock quantity
        """
        # Z-score for service level
        from scipy.stats import norm
        z_score = norm.ppf(service_level)
        
        # Calculate safety stock
        safety_stock = z_score * np.sqrt(
            (average_lead_time * demand_std ** 2) + 
            (average_demand ** 2 * lead_time_std ** 2)
        )
        
        return safety_stock
    
    def calculate_reorder_point(
        self,
        average_demand: float,
        average_lead_time: float,
        safety_stock: float
    ) -> float:
        """
        Calculate reorder point.
        
        Reorder Point = (Average Demand * Average Lead Time) + Safety Stock
        
        Args:
            average_demand: Average demand per period
            average_lead_time: Average lead time in periods
            safety_stock: Safety stock quantity
        
        Returns:
            Reorder point
        """
        reorder_point = (average_demand * average_lead_time) + safety_stock
        return reorder_point
    
    def optimize_economic_order_quantity(
        self,
        annual_demand: float,
        ordering_cost: float,
        holding_cost: float
    ) -> float:
        """
        Calculate Economic Order Quantity (EOQ).
        
        EOQ = sqrt(2 * D * S / H)
        
        Args:
            annual_demand: Annual demand
            ordering_cost: Cost per order
            holding_cost: Holding cost per unit per year
        
        Returns:
            Optimal order quantity
        """
        eoq = np.sqrt((2 * annual_demand * ordering_cost) / holding_cost)
        return eoq
    
    def calculate_total_inventory_cost(
        self,
        order_quantity: float,
        annual_demand: float,
        ordering_cost: float,
        holding_cost: float,
        unit_cost: float,
        safety_stock: float
    ) -> Dict[str, float]:
        """
        Calculate total inventory cost components.
        
        Args:
            order_quantity: Order quantity
            annual_demand: Annual demand
            ordering_cost: Cost per order
            holding_cost: Holding cost per unit per year
            unit_cost: Cost per unit
            safety_stock: Safety stock quantity
        
        Returns:
            Dictionary with cost components
        """
        # Ordering cost
        number_of_orders = annual_demand / order_quantity
        total_ordering_cost = number_of_orders * ordering_cost
        
        # Holding cost (cycle stock + safety stock)
        average_inventory = (order_quantity / 2) + safety_stock
        total_holding_cost = average_inventory * holding_cost
        
        # Purchase cost
        total_purchase_cost = annual_demand * unit_cost
        
        # Total cost
        total_cost = total_ordering_cost + total_holding_cost + total_purchase_cost
        
        costs = {
            'ordering_cost': total_ordering_cost,
            'holding_cost': total_holding_cost,
            'purchase_cost': total_purchase_cost,
            'total_cost': total_cost,
            'number_of_orders': number_of_orders,
            'average_inventory': average_inventory
        }
        
        return costs
    
    def optimize_multi_echelon(
        self,
        demand_data: pd.DataFrame,
        echelons: List[str],
        lead_times: Dict[str, float],
        service_levels: Dict[str, float],
        holding_costs: Dict[str, float],
        ordering_costs: Dict[str, float]
    ) -> Dict[str, Dict[str, float]]:
        """
        Optimize inventory across multiple echelons.
        
        Args:
            demand_data: Demand data by echelon
            echelons: List of echelon names (e.g., ['warehouse', 'regional', 'store'])
            lead_times: Lead time for each echelon
            service_levels: Target service level for each echelon
            holding_costs: Holding cost for each echelon
            ordering_costs: Ordering cost for each echelon
        
        Returns:
            Dictionary mapping echelon to optimization results
        """
        logger.info("Optimizing multi-echelon inventory...")
        
        echelon_results = {}
        
        for echelon in echelons:
            if echelon not in demand_data.columns:
                continue
            
            # Calculate demand statistics
            demand = demand_data[echelon].dropna()
            average_demand = demand.mean()
            demand_std = demand.std()
            
            annual_demand = average_demand * 365  # Assuming daily demand
            
            # Calculate safety stock
            safety_stock = self.calculate_safety_stock(
                demand_std=demand_std,
                lead_time_std=lead_times.get(echelon, 1) * 0.2,  # 20% lead time variability
                average_demand=average_demand,
                average_lead_time=lead_times.get(echelon, 1),
                service_level=service_levels.get(echelon, 0.95)
            )
            
            # Calculate reorder point
            reorder_point = self.calculate_reorder_point(
                average_demand=average_demand,
                average_lead_time=lead_times.get(echelon, 1),
                safety_stock=safety_stock
            )
            
            # Calculate EOQ
            eoq = self.optimize_economic_order_quantity(
                annual_demand=annual_demand,
                ordering_cost=ordering_costs.get(echelon, 100),
                holding_cost=holding_costs.get(echelon, 0.1)
            )
            
            # Calculate total cost
            costs = self.calculate_total_inventory_cost(
                order_quantity=eoq,
                annual_demand=annual_demand,
                ordering_cost=ordering_costs.get(echelon, 100),
                holding_cost=holding_costs.get(echelon, 0.1),
                unit_cost=10,  # Placeholder
                safety_stock=safety_stock
            )
            
            echelon_results[echelon] = {
                'safety_stock': safety_stock,
                'reorder_point': reorder_point,
                'economic_order_quantity': eoq,
                'total_cost': costs['total_cost'],
                'holding_cost': costs['holding_cost'],
                'ordering_cost': costs['ordering_cost'],
                'average_inventory': costs['average_inventory']
            }
        
        logger.info(f"Multi-echelon optimization complete for {len(echelon_results)} echelons")
        return echelon_results
    
    def mitigate_bullwhip_effect(
        self,
        demand_data: pd.DataFrame,
        echelon_order: List[str]
    ) -> Dict[str, float]:
        """
        Analyze and quantify bullwhip effect across echelons.
        
        Bullwhip effect = variance of orders / variance of demand
        
        Args:
            demand_data: Demand/order data by echelon
            echelon_order: Order of echelons (upstream to downstream)
        
        Returns:
            Dictionary with bullwhip metrics
        """
        logger.info("Analyzing bullwhip effect...")
        
        bullwhip_ratios = {}
        
        for i in range(len(echelon_order) - 1):
            downstream = echelon_order[i]
            upstream = echelon_order[i + 1]
            
            if downstream in demand_data.columns and upstream in data.columns:
                downstream_var = data[downstream].var()
                upstream_var = data[upstream].var()
                
                if downstream_var > 0:
                    bullwhip_ratio = upstream_var / downstream_var
                else:
                    bullwhip_ratio = 1
                
                bullwhip_ratios[f"{downstream}_to_{upstream}"] = bullwhip_ratio
        
        logger.info(f"Bullwhip ratios: {bullwhip_ratios}")
        return bullwhip_ratios
    
    def optimize_cross_docking(
        self,
        demand_data: pd.DataFrame,
        products: List[str],
        cross_dock_capacity: float = 1000
    ) -> Dict[str, float]:
        """
        Optimize cross-docking operations.
        
        Cross-docking reduces inventory by transferring goods directly
        from inbound to outbound without storage.
        
        Args:
            demand_data: Demand data
            products: List of products
            cross_dock_capacity: Cross-dock capacity
        
        Returns:
            Dictionary with cross-docking recommendations
        """
        logger.info("Optimizing cross-docking operations...")
        
        # Calculate total demand
        total_demand = demand_data.sum().sum()
        
        # Calculate potential savings from cross-docking
        # Assume cross-docking reduces holding time by 50%
        holding_cost_reduction = 0.5
        
        # Determine which products to cross-dock
        # High-volume, low-variability products are best candidates
        product_stats = {}
        for product in products:
            if product in demand_data.columns:
                product_demand = data[product]
                product_stats[product] = {
                    'mean': product_demand.mean(),
                    'std': product_demand.std(),
                    'cv': product_demand.std() / product_demand.mean() if product_demand.mean() > 0 else 0
                }
        
        # Select products for cross-docking (low CV, high volume)
        cross_dock_products = [
            p for p, stats in product_stats.items()
            if stats['cv'] < 0.3 and stats['mean'] > 100
        ]
        
        recommendations = {
            'total_demand': total_demand,
            'cross_dock_capacity': cross_dock_capacity,
            'cross_dock_products': cross_dock_products,
            'n_cross_dock_products': len(cross_dock_products),
            'holding_cost_reduction_pct': holding_cost_reduction * 100
        }
        
        logger.info(f"Cross-docking: {len(cross_dock_products)} products recommended")
        return recommendations
    
    def generate_inventory_report(
        self,
        echelon_results: Dict[str, Dict[str, float]],
        bullwhip_ratios: Dict[str, float],
        cross_dock_recommendations: Dict[str, float],
        output_path: Optional[Path] = None
    ) -> str:
        """
        Generate multi-echelon inventory optimization report.
        
        Args:
            echelon_results: Optimization results by echelon
            bullwhip_ratios: Bullwhip effect metrics
            cross_dock_recommendations: Cross-docking recommendations
            output_path: Optional path to save report
        
        Returns:
            Report string
        """
        report = f"""
Multi-Echelon Inventory Optimization Report
{'=' * 60}

Echelon Optimization Results:
"""
        
        for echelon, results in echelon_results.items():
            report += f"""
{echelon.upper()}:
- Safety Stock: {results['safety_stock']:.2f} units
- Reorder Point: {results['reorder_point']:.2f} units
- Economic Order Quantity: {results['economic_order_quantity']:.2f} units
- Total Cost: ₹{results['total_cost']:,.2f}
- Average Inventory: {results['average_inventory']:.2f} units
"""
        
        report += f"""
Bullwhip Effect Analysis:
"""
        
        for link, ratio in bullwhip_ratios.items():
            severity = "Severe" if ratio > 2 else "Moderate" if ratio > 1.5 else "Minimal"
            report += f"- {link}: {ratio:.2f}x ({severity})\n"
        
        report += f"""
Cross-Docking Recommendations:
- Total Demand: {cross_dock_recommendations['total_demand']:.2f} units
- Cross-Dock Capacity: {cross_dock_recommendations['cross_dock_capacity']:.2f} units
- Products Recommended: {cross_dock_recommendations['n_cross_dock_products']}
- Holding Cost Reduction: {cross_dock_recommendations['holding_cost_reduction_pct']:.1f}%

Interpretation:
- Safety Stock: Buffer against demand and lead time uncertainty
- Reorder Point: When to place new orders
- EOQ: Optimal order quantity to minimize costs
- Bullwhip Ratio: > 1.5 indicates significant amplification
- Cross-Docking: Reduces inventory holding time for fast-moving items
"""
        
        if output_path:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(report)
            logger.info(f"Inventory report saved to {output_path}")
        
        return report


def run_multi_echelon_optimization(
    demand_data: pd.DataFrame,
    echelons: List[str],
    lead_times: Dict[str, float],
    service_levels: Dict[str, float]
) -> Dict[str, any]:
    """
    Convenience function to run complete multi-echelon optimization.
    
    Args:
        demand_data: Demand data by echelon
        echelons: List of echelon names
        lead_times: Lead time for each echelon
        service_levels: Target service level for each echelon
    
    Returns:
        Dictionary with optimization results
    """
    optimizer = MultiEchelonInventoryOptimizer()
    
    # Default costs
    holding_costs = {e: 0.1 for e in echelons}
    ordering_costs = {e: 100 for e in echelons}
    
    # Optimize multi-echelon
    echelon_results = optimizer.optimize_multi_echelon(
        demand_data, echelons, lead_times, service_levels, holding_costs, ordering_costs
    )
    
    # Analyze bullwhip effect
    bullwhip_ratios = optimizer.mitigate_bullwhip_effect(demand_data, echelons)
    
    # Cross-docking recommendations
    products = [col for col in demand_data.columns if col not in echelons]
    cross_dock_recommendations = optimizer.optimize_cross_docking(demand_data, products)
    
    return {
        'echelon_results': echelon_results,
        'bullwhip_ratios': bullwhip_ratios,
        'cross_dock_recommendations': cross_dock_recommendations
    }
