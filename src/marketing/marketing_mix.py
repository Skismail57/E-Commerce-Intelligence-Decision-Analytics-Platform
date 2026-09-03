"""
Marketing Mix Modeling Module
Implements marketing mix modeling to measure the effectiveness of marketing channels.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from config.logging_config import get_logger

logger = get_logger(__name__)


class MarketingMixModeler:
    """
    Marketing mix modeling engine.
    
    Features:
    - Channel effectiveness measurement
    - ROI calculation for marketing channels
    - Adstock transformation for carryover effects
    - Saturation curves for diminishing returns
    - Budget allocation optimization
    """
    
    def __init__(self):
        """Initialize marketing mix modeler"""
        self.model = None
        self.channel_coefficients = {}
        self.adstock_params = {}
    
    def apply_adstock(
        self,
        spend: np.ndarray,
        decay_rate: float = 0.5
    ) -> np.ndarray:
        """
        Apply adstock transformation to model carryover effects.
        
        Args:
            spend: Array of spend values
            decay_rate: Decay rate for adstock (0-1)
        
        Returns:
            Array with adstock applied
        """
        adstock = np.zeros_like(spend)
        adstock[0] = spend[0]
        
        for i in range(1, len(spend)):
            adstock[i] = spend[i] + decay_rate * adstock[i-1]
        
        return adstock
    
    def apply_saturation(
        self,
        spend: np.ndarray,
        saturation_point: float = 1000
    ) -> np.ndarray:
        """
        Apply saturation curve to model diminishing returns.
        
        Args:
            spend: Array of spend values
            saturation_point: Point where saturation begins
        
        Returns:
            Array with saturation applied
        """
        return spend / (spend + saturation_point)
    
    def prepare_mmm_data(
        self,
        spend_data: pd.DataFrame,
        sales_data: pd.DataFrame,
        channels: List[str],
        date_col: str = 'date',
        sales_col: str = 'sales'
    ) -> pd.DataFrame:
        """
        Prepare data for marketing mix modeling.
        
        Args:
            spend_data: DataFrame with marketing spend by channel
            sales_data: DataFrame with sales data
            channels: List of marketing channel names
            date_col: Date column name
            sales_col: Sales column name
        
        Returns:
            DataFrame with prepared MMM data
        """
        logger.info("Preparing marketing mix modeling data...")
        
        # Merge spend and sales data
        mmm_data = spend_data.merge(sales_data, on=date_col, how='inner')
        
        # Apply transformations for each channel
        for channel in channels:
            if channel in mmm_data.columns:
                # Apply adstock
                adstock = self.apply_adstock(mmm_data[channel].values)
                mmm_data[f'{channel}_adstock'] = adstock
                
                # Apply saturation
                saturated = self.apply_saturation(adstock)
                mmm_data[f'{channel}_saturated'] = saturated
        
        logger.info(f"MMM data prepared with {len(mmm_data)} observations")
        
        return mmm_data
    
    def fit_mmm_model(
        self,
        mmm_data: pd.DataFrame,
        channels: List[str],
        sales_col: str = 'sales',
        model_type: str = 'linear'
    ) -> Dict:
        """
        Fit marketing mix model.
        
        Args:
            mmm_data: Prepared MMM data
            channels: List of marketing channel names
            sales_col: Sales column name
            model_type: Model type ('linear', 'ridge', 'gradient_boosting')
        
        Returns:
            Dictionary with model results
        """
        logger.info(f"Fitting {model_type} marketing mix model...")
        
        # Prepare features
        feature_cols = [f'{channel}_saturated' for channel in channels]
        X = mmm_data[feature_cols].values
        y = mmm_data[sales_col].values
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Initialize model
        if model_type == 'linear':
            self.model = LinearRegression()
        elif model_type == 'ridge':
            self.model = Ridge(alpha=1.0)
        elif model_type == 'gradient_boosting':
            self.model = GradientBoostingRegressor(n_estimators=100, random_state=42)
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        # Fit model
        self.model.fit(X_train, y_train)
        
        # Predictions
        y_train_pred = self.model.predict(X_train)
        y_test_pred = self.model.predict(X_test)
        
        # Metrics
        train_mae = mean_absolute_error(y_train, y_train_pred)
        test_mae = mean_absolute_error(y_test, y_test_pred)
        train_r2 = r2_score(y_train, y_train_pred)
        test_r2 = r2_score(y_test, y_test_pred)
        
        # Extract channel coefficients
        if hasattr(self.model, 'coef_'):
            for i, channel in enumerate(channels):
                self.channel_coefficients[channel] = float(self.model.coef_[i])
        elif hasattr(self.model, 'feature_importances_'):
            for i, channel in enumerate(channels):
                self.channel_coefficients[channel] = float(self.model.feature_importances_[i])
        
        results = {
            'model_type': model_type,
            'train_mae': float(train_mae),
            'test_mae': float(test_mae),
            'train_r2': float(train_r2),
            'test_r2': float(test_r2),
            'channel_coefficients': self.channel_coefficients,
            'n_channels': len(channels)
        }
        
        logger.info(f"MMM model fitted. Test R²: {test_r2:.3f}")
        
        return results
    
    def calculate_channel_roi(
        self,
        mmm_data: pd.DataFrame,
        channels: List[str],
        sales_col: str = 'sales'
    ) -> Dict:
        """
        Calculate ROI for each marketing channel.
        
        Args:
            mmm_data: MMM data
            channels: List of marketing channel names
            sales_col: Sales column name
        
        Returns:
            Dictionary with channel ROI
        """
        logger.info("Calculating channel ROI...")
        
        roi_results = {}
        
        for channel in channels:
            # Total spend
            total_spend = mmm_data[channel].sum()
            
            # Contribution from channel (using coefficient)
            if channel in self.channel_coefficients:
                coefficient = self.channel_coefficients[channel]
                contribution = coefficient * mmm_data[f'{channel}_saturated'].sum()
                
                # ROI = (Revenue - Cost) / Cost
                roi = (contribution - total_spend) / total_spend if total_spend > 0 else 0
                
                roi_results[channel] = {
                    'total_spend': float(total_spend),
                    'estimated_contribution': float(contribution),
                    'roi': float(roi),
                    'coefficient': coefficient
                }
        
        return roi_results
    
    def optimize_budget_allocation(
        self,
        total_budget: float,
        channels: List[str],
        min_allocation_pct: float = 0.05
    ) -> Dict:
        """
        Optimize budget allocation across channels.
        
        Args:
            total_budget: Total marketing budget
            channels: List of marketing channel names
            min_allocation_pct: Minimum allocation percentage per channel
        
        Returns:
            Dictionary with optimal budget allocation
        """
        logger.info("Optimizing budget allocation...")
        
        # Get channel coefficients
        coefficients = np.array([self.channel_coefficients.get(channel, 0) for channel in channels])
        
        # Normalize coefficients to get allocation weights
        if coefficients.sum() > 0:
            weights = coefficients / coefficients.sum()
        else:
            weights = np.ones(len(channels)) / len(channels)
        
        # Apply minimum allocation constraint
        min_allocation = total_budget * min_allocation_pct
        remaining_budget = total_budget - (min_allocation * len(channels))
        
        # Distribute remaining budget based on weights
        allocations = {}
        for i, channel in enumerate(channels):
            allocation = min_allocation + (weights[i] * remaining_budget)
            allocations[channel] = {
                'budget': float(allocation),
                'percentage': float(allocation / total_budget * 100),
                'coefficient': float(coefficients[i])
            }
        
        logger.info(f"Budget allocation optimized for {len(channels)} channels")
        
        return allocations


def run_marketing_mix_pipeline(
    spend_data: pd.DataFrame,
    sales_data: pd.DataFrame,
    channels: List[str],
    total_budget: float = 100000
) -> Tuple[MarketingMixModeler, Dict]:
    """
    Convenience function to run complete marketing mix pipeline.
    
    Args:
        spend_data: Marketing spend data
        sales_data: Sales data
        channels: List of marketing channels
        total_budget: Total budget for optimization
    
    Returns:
        Tuple of (modeler, results)
    """
    modeler = MarketingMixModeler()
    
    # Prepare data
    mmm_data = modeler.prepare_mmm_data(spend_data, sales_data, channels)
    
    # Fit model
    model_results = modeler.fit_mmm_model(mmm_data, channels)
    
    # Calculate ROI
    roi_results = modeler.calculate_channel_roi(mmm_data, channels)
    
    # Optimize budget
    allocation = modeler.optimize_budget_allocation(total_budget, channels)
    
    results = {
        'model_results': model_results,
        'roi_results': roi_results,
        'budget_allocation': allocation
    }
    
    return modeler, results
