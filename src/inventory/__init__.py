"""
Inventory Module
Provides inventory optimization using EOQ (Economic Order Quantity) and safety stock calculations.
"""

from .optimization import InventoryOptimizer, run_inventory_optimization_pipeline

__all__ = [
    'InventoryOptimizer',
    'run_inventory_optimization_pipeline',
]
