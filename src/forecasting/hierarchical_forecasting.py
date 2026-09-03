"""
Hierarchical and Probabilistic Forecasting Module
Implements hierarchical time series forecasting and probabilistic predictions with prediction intervals.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from prophet import Prophet
from sklearn.metrics import mean_absolute_error, mean_squared_error
import joblib
from pathlib import Path
from config.logging_config import get_logger

logger = get_logger(__name__)


class HierarchicalForecaster:
    """
    Hierarchical forecasting engine for multi-level time series.
    
    Features:
    - Hierarchical reconciliation (bottom-up, top-down, optimal combination)
    - Probabilistic forecasting with prediction intervals
    - Multiple hierarchy levels (category, subcategory, product)
    - Forecast aggregation and disaggregation
    """
    
    def __init__(self):
        """Initialize hierarchical forecaster"""
        self.base_models = {}
        self.hierarchy_structure = {}
    
    def define_hierarchy(
        self,
        hierarchy_config: Dict[str, List[str]]
    ) -> None:
        """
        Define the hierarchy structure.
        
        Args:
            hierarchy_config: Dictionary mapping level names to column names
                Example: {'category': 'category_id', 'product': 'product_id'}
        """
        logger.info("Defining hierarchy structure...")
        self.hierarchy_structure = hierarchy_config
        logger.info(f"Hierarchy defined with {len(hierarchy_config)} levels")
    
    def fit_bottom_up(
        self,
        data: pd.DataFrame,
        date_col: str,
        value_col: str,
        hierarchy_cols: List[str],
        forecast_horizon: int = 30
    ) -> Dict:
        """
        Fit bottom-up forecasting model.
        
        Args:
            data: DataFrame with time series data
            date_col: Date column name
            value_col: Value column name
            hierarchy_cols: List of hierarchy columns
            forecast_horizon: Number of periods to forecast
        
        Returns:
            Dictionary with forecasting results
        """
        logger.info("Fitting bottom-up hierarchical forecasting model...")
        
        # Prepare data
        data = data.copy()
        data[date_col] = pd.to_datetime(data[date_col])
        
        # Fit models at the lowest level (most granular)
        lowest_level = hierarchy_cols[-1]
        unique_entities = data[lowest_level].unique()
        
        forecasts = {}
        
        for entity in unique_entities:
            entity_data = data[data[lowest_level] == entity].copy()
            entity_data = entity_data.sort_values(date_col)
            
            # Fit Prophet model
            prophet_data = entity_data[[date_col, value_col]].copy()
            prophet_data.columns = ['ds', 'y']
            
            model = Prophet(
                yearly_seasonality=True,
                weekly_seasonality=True,
                daily_seasonality=False,
                interval_width=0.8  # 80% prediction interval
            )
            
            model.fit(prophet_data)
            
            # Make forecast
            future = model.make_future_dataframe(periods=forecast_horizon)
            forecast = model.predict(future)
            
            self.base_models[entity] = model
            forecasts[entity] = forecast
        
        # Aggregate forecasts up the hierarchy
        aggregated_forecasts = self._aggregate_forecasts(forecasts, hierarchy_cols, data)
        
        results = {
            'method': 'bottom_up',
            'n_models': len(self.base_models),
            'forecast_horizon': forecast_horizon,
            'forecasts': aggregated_forecasts
        }
        
        logger.info(f"Bottom-up forecasting complete. Fitted {len(self.base_models)} models")
        
        return results
    
    def fit_top_down(
        self,
        data: pd.DataFrame,
        date_col: str,
        value_col: str,
        hierarchy_cols: List[str],
        forecast_horizon: int = 30
    ) -> Dict:
        """
        Fit top-down forecasting model.
        
        Args:
            data: DataFrame with time series data
            date_col: Date column name
            value_col: Value column name
            hierarchy_cols: List of hierarchy columns
            forecast_horizon: Number of periods to forecast
        
        Returns:
            Dictionary with forecasting results
        """
        logger.info("Fitting top-down hierarchical forecasting model...")
        
        # Prepare data
        data = data.copy()
        data[date_col] = pd.to_datetime(data[date_col])
        
        # Fit model at the highest level (aggregated)
        highest_level = hierarchy_cols[0]
        
        # Aggregate data at highest level
        aggregated_data = data.groupby([date_col, highest_level])[value_col].sum().reset_index()
        
        forecasts = {}
        proportions = {}
        
        for level_value in aggregated_data[highest_level].unique():
            level_data = aggregated_data[aggregated_data[highest_level] == level_value].copy()
            level_data = level_data.sort_values(date_col)
            
            # Fit Prophet model
            prophet_data = level_data[[date_col, value_col]].copy()
            prophet_data.columns = ['ds', 'y']
            
            model = Prophet(
                yearly_seasonality=True,
                weekly_seasonality=True,
                daily_seasonality=False,
                interval_width=0.8
            )
            
            model.fit(prophet_data)
            
            # Make forecast
            future = model.make_future_dataframe(periods=forecast_horizon)
            forecast = model.predict(future)
            
            self.base_models[f"{highest_level}_{level_value}"] = model
            forecasts[f"{highest_level}_{level_value}"] = forecast
            
            # Calculate historical proportions for disaggregation
            historical_total = level_data[value_col].sum()
            proportions[f"{highest_level}_{level_value}"] = {}
            
            # Calculate proportions for lower levels
            for lower_level in hierarchy_cols[1:]:
                for entity in data[lower_level].unique():
                    entity_data = data[data[lower_level] == entity]
                    entity_value = entity_data[value_col].sum()
                    proportions[f"{highest_level}_{level_value}"][entity] = entity_value / historical_total
        
        # Disaggregate forecasts down the hierarchy
        disaggregated_forecasts = self._disaggregate_forecasts(forecasts, proportions, hierarchy_cols)
        
        results = {
            'method': 'top_down',
            'n_models': len(self.base_models),
            'forecast_horizon': forecast_horizon,
            'forecasts': disaggregated_forecasts,
            'proportions': proportions
        }
        
        logger.info(f"Top-down forecasting complete. Fitted {len(self.base_models)} models")
        
        return results
    
    def _aggregate_forecasts(
        self,
        forecasts: Dict,
        hierarchy_cols: List[str],
        historical_data: pd.DataFrame
    ) -> Dict:
        """Aggregate forecasts from bottom level up hierarchy"""
        aggregated = {}
        
        # Start with lowest level forecasts
        lowest_level = hierarchy_cols[-1]
        
        for entity, forecast in forecasts.items():
            aggregated[f"{lowest_level}_{entity}"] = forecast
        
        # Aggregate up the hierarchy
        for i in range(len(hierarchy_cols) - 2, -1, -1):
            level = hierarchy_cols[i]
            next_level = hierarchy_cols[i + 1]
            
            # Get entities at this level
            level_entities = historical_data[level].unique()
            
            for entity in level_entities:
                # Get all entities at next level that belong to this entity
                next_level_entities = historical_data[
                    historical_data[level] == entity
                ][next_level].unique()
                
                # Aggregate forecasts
                entity_forecasts = []
                for next_entity in next_level_entities:
                    key = f"{next_level}_{next_entity}"
                    if key in aggregated:
                        entity_forecasts.append(aggregated[key])
                
                if entity_forecasts:
                    # Sum the forecasts
                    combined_forecast = entity_forecasts[0].copy()
                    for forecast in entity_forecasts[1:]:
                        combined_forecast['yhat'] += forecast['yhat']
                        combined_forecast['yhat_lower'] += forecast['yhat_lower']
                        combined_forecast['yhat_upper'] += forecast['yhat_upper']
                    
                    aggregated[f"{level}_{entity}"] = combined_forecast
        
        return aggregated
    
    def _disaggregate_forecasts(
        self,
        forecasts: Dict,
        proportions: Dict,
        hierarchy_cols: List[str]
    ) -> Dict:
        """Disaggregate forecasts from top level down hierarchy"""
        disaggregated = {}
        
        # Start with highest level forecasts
        highest_level = hierarchy_cols[0]
        
        for key, forecast in forecasts.items():
            disaggregated[key] = forecast
        
        # Disaggregate down the hierarchy
        for i in range(len(hierarchy_cols) - 1):
            current_level = hierarchy_cols[i]
            next_level = hierarchy_cols[i + 1]
            
            for key, forecast in list(disaggregated.items()):
                if key.startswith(f"{current_level}_"):
                    level_value = key.split("_")[1]
                    
                    # Get proportions for this level
                    if key in proportions:
                        for entity, proportion in proportions[key].items():
                            # Create disaggregated forecast
                            entity_forecast = forecast.copy()
                            entity_forecast['yhat'] *= proportion
                            entity_forecast['yhat_lower'] *= proportion
                            entity_forecast['yhat_upper'] *= proportion
                            
                            disaggregated[f"{next_level}_{entity}"] = entity_forecast
        
        return disaggregated
    
    def get_probabilistic_forecast(
        self,
        entity_id: str,
        forecast_horizon: int = 30
    ) -> Dict:
        """
        Get probabilistic forecast with prediction intervals.
        
        Args:
            entity_id: Entity identifier
            forecast_horizon: Number of periods to forecast
        
        Returns:
            Dictionary with probabilistic forecast
        """
        if entity_id not in self.base_models:
            raise ValueError(f"No model found for entity {entity_id}")
        
        model = self.base_models[entity_id]
        
        # Make future dataframe
        future = model.make_future_dataframe(periods=forecast_horizon)
        forecast = model.predict(future)
        
        # Extract forecast and prediction intervals
        forecast_values = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(forecast_horizon)
        
        # Calculate prediction interval width
        forecast_values['interval_width'] = forecast_values['yhat_upper'] - forecast_values['yhat_lower']
        
        # Calculate uncertainty (coefficient of variation)
        forecast_values['uncertainty_cv'] = forecast_values['interval_width'] / (2 * forecast_values['yhat'])
        
        result = {
            'entity_id': entity_id,
            'forecast': forecast_values.to_dict('records'),
            'mean_forecast': float(forecast_values['yhat'].mean()),
            'forecast_std': float(forecast_values['yhat'].std()),
            'mean_interval_width': float(forecast_values['interval_width'].mean()),
            'mean_uncertainty_cv': float(forecast_values['uncertainty_cv'].mean())
        }
        
        return result
    
    def evaluate_forecast_accuracy(
        self,
        actual_data: pd.DataFrame,
        forecast_data: pd.DataFrame,
        date_col: str,
        value_col: str
    ) -> Dict:
        """
        Evaluate forecast accuracy.
        
        Args:
            actual_data: Actual values
            forecast_data: Forecasted values
            date_col: Date column name
            value_col: Value column name
        
        Returns:
            Dictionary with accuracy metrics
        """
        # Merge actual and forecast
        merged = actual_data.merge(forecast_data, on=date_col, how='inner')
        
        # Calculate metrics
        mae = mean_absolute_error(merged[value_col + '_actual'], merged[value_col + '_forecast'])
        rmse = np.sqrt(mean_squared_error(merged[value_col + '_actual'], merged[value_col + '_forecast']))
        
        # Mean Absolute Percentage Error
        mape = np.mean(np.abs((merged[value_col + '_actual'] - merged[value_col + '_forecast']) / merged[value_col + '_actual'])) * 100
        
        # Symmetric Mean Absolute Percentage Error
        smape = np.mean(2 * np.abs(merged[value_col + '_actual'] - merged[value_col + '_forecast']) / 
                       (np.abs(merged[value_col + '_actual']) + np.abs(merged[value_col + '_forecast']))) * 100
        
        results = {
            'mae': float(mae),
            'rmse': float(rmse),
            'mape': float(mape),
            'smape': float(smape),
            'n_observations': len(merged)
        }
        
        return results
    
    def save_models(self, save_dir: str = "models/hierarchical_forecasting"):
        """Save trained models"""
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)
        
        for entity_id, model in self.base_models.items():
            model_path = save_path / f"{entity_id}.joblib"
            joblib.dump(model, model_path)
        
        logger.info(f"Saved {len(self.base_models)} models to {save_dir}")
    
    def load_models(self, load_dir: str = "models/hierarchical_forecasting"):
        """Load trained models"""
        load_path = Path(load_dir)
        
        model_files = list(load_path.glob("*.joblib"))
        
        for model_file in model_files:
            entity_id = model_file.stem
            self.base_models[entity_id] = joblib.load(model_file)
        
        logger.info(f"Loaded {len(self.base_models)} models from {load_dir}")


class ProbabilisticForecaster:
    """
    Probabilistic forecasting engine with prediction intervals.
    
    Features:
    - Probabilistic forecasts with prediction intervals
    - Monte Carlo simulation for uncertainty quantification
    - Scenario analysis
    - Risk assessment
    """
    
    def __init__(self):
        """Initialize probabilistic forecaster"""
        self.models = {}
    
    def fit_probabilistic_model(
        self,
        data: pd.DataFrame,
        date_col: str,
        value_col: str,
        entity_id: str,
        uncertainty_samples: int = 1000
    ) -> Dict:
        """
        Fit probabilistic forecasting model.
        
        Args:
            data: DataFrame with time series data
            date_col: Date column name
            value_col: Value column name
            entity_id: Entity identifier
            uncertainty_samples: Number of samples for uncertainty quantification
        
        Returns:
            Dictionary with fitting results
        """
        logger.info(f"Fitting probabilistic model for {entity_id}...")
        
        # Prepare data
        data = data.copy()
        data[date_col] = pd.to_datetime(data[date_col])
        
        # Fit Prophet model with MCMC for full uncertainty
        prophet_data = data[[date_col, value_col]].copy()
        prophet_data.columns = ['ds', 'y']
        
        model = Prophet(
            yearly_seasonality=True,
            weekly_seasonality=True,
            daily_seasonality=False,
            mcmc_samples=uncertainty_samples,
            interval_width=0.95  # 95% prediction interval
        )
        
        model.fit(prophet_data)
        
        self.models[entity_id] = model
        
        results = {
            'entity_id': entity_id,
            'uncertainty_samples': uncertainty_samples,
            'interval_width': 0.95
        }
        
        logger.info(f"Probabilistic model fitted for {entity_id}")
        
        return results
    
    def generate_scenarios(
        self,
        entity_id: str,
        forecast_horizon: int = 30,
        n_scenarios: int = 100
    ) -> Dict:
        """
        Generate multiple forecast scenarios.
        
        Args:
            entity_id: Entity identifier
            forecast_horizon: Number of periods to forecast
            n_scenarios: Number of scenarios to generate
        
        Returns:
            Dictionary with scenario forecasts
        """
        if entity_id not in self.models:
            raise ValueError(f"No model found for entity {entity_id}")
        
        model = self.models[entity_id]
        
        # Make future dataframe
        future = model.make_future_dataframe(periods=forecast_horizon)
        
        # Generate multiple samples
        scenarios = []
        for i in range(n_scenarios):
            forecast = model.predict(future)
            forecast_values = forecast['yhat'].tail(forecast_horizon).values
            scenarios.append(forecast_values)
        
        scenarios_array = np.array(scenarios)
        
        # Calculate statistics
        mean_forecast = scenarios_array.mean(axis=0)
        std_forecast = scenarios_array.std(axis=0)
        percentiles = {
            'p10': np.percentile(scenarios_array, 10, axis=0),
            'p25': np.percentile(scenarios_array, 25, axis=0),
            'p50': np.percentile(scenarios_array, 50, axis=0),
            'p75': np.percentile(scenarios_array, 75, axis=0),
            'p90': np.percentile(scenarios_array, 90, axis=0)
        }
        
        result = {
            'entity_id': entity_id,
            'n_scenarios': n_scenarios,
            'forecast_horizon': forecast_horizon,
            'mean_forecast': mean_forecast.tolist(),
            'std_forecast': std_forecast.tolist(),
            'percentiles': {k: v.tolist() for k, v in percentiles.items()}
        }
        
        return result


def run_hierarchical_forecasting_pipeline(
    data: pd.DataFrame,
    date_col: str,
    value_col: str,
    hierarchy_cols: List[str],
    method: str = 'bottom_up',
    forecast_horizon: int = 30
) -> Tuple[HierarchicalForecaster, Dict]:
    """
    Convenience function to run hierarchical forecasting pipeline.
    
    Args:
        data: Time series data
        date_col: Date column name
        value_col: Value column name
        hierarchy_cols: List of hierarchy columns
        method: Forecasting method ('bottom_up' or 'top_down')
        forecast_horizon: Number of periods to forecast
    
    Returns:
        Tuple of (forecaster, results)
    """
    forecaster = HierarchicalForecaster()
    forecaster.define_hierarchy({f'level_{i}': col for i, col in enumerate(hierarchy_cols)})
    
    if method == 'bottom_up':
        results = forecaster.fit_bottom_up(data, date_col, value_col, hierarchy_cols, forecast_horizon)
    else:
        results = forecaster.fit_top_down(data, date_col, value_col, hierarchy_cols, forecast_horizon)
    
    return forecaster, results
