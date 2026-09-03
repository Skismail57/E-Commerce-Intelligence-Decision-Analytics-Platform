"""
CLV Backtesting Module
Implements temporal validation for CLV predictions to ensure models are actually predictive.
"""

from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from pathlib import Path

from config.logging_config import get_logger
from config.settings import settings

logger = get_logger(__name__)


class CLVBacktester:
    """
    Backtests CLV predictions using temporal holdout validation.
    
    Architecture:
    - Training window: Historical data used to fit CLV models
    - Prediction date: Point where predictions are made
    - Holdout window: Future period where actual revenue is measured
    - Evaluation: Compare predicted vs actual revenue in holdout
    """
    
    def __init__(
        self,
        train_end_date: str = "2024-06-30",
        prediction_date: str = "2024-06-30",
        holdout_end_date: str = "2024-12-31"
    ):
        """
        Initialize CLV backtester.
        
        Args:
            train_end_date: End date for training data
            prediction_date: Date when CLV predictions are made
            holdout_end_date: End date for holdout evaluation period
        """
        self.train_end_date = pd.to_datetime(train_end_date)
        self.prediction_date = pd.to_datetime(prediction_date)
        self.holdout_end_date = pd.to_datetime(holdout_end_date)
        
        logger.info(f"CLV Backtester initialized:")
        logger.info(f"  Training window: up to {self.train_end_date}")
        logger.info(f"  Prediction date: {self.prediction_date}")
        logger.info(f"  Holdout window: {self.prediction_date} to {self.holdout_end_date}")
    
    def prepare_training_data(
        self,
        orders_df: pd.DataFrame,
        customers_df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Prepare training data for CLV model fitting.
        
        Uses only orders BEFORE train_end_date.
        
        Args:
            orders_df: Orders data
            customers_df: Customers data
        
        Returns:
            DataFrame in RFM format for CLV modeling
        """
        logger.info("Preparing CLV training data...")
        
        # Filter to training period
        orders_train = orders_df[
            orders_df['order_date'] < self.train_end_date
        ].copy()
        
        # Calculate RFM metrics as of train_end_date
        rfm_data = orders_train.groupby('customer_id').agg({
            'order_date': ['min', 'max', 'count'],
            'order_total': 'sum'
        }).reset_index()
        
        rfm_data.columns = ['customer_id', 'first_order', 'last_order', 'frequency', 'monetary']
        
        # Calculate recency (days since last order as of train_end_date)
        rfm_data['recency'] = (
            self.train_end_date - pd.to_datetime(rfm_data['last_order'])
        ).dt.days
        
        # Calculate T (customer age as of train_end_date)
        rfm_data['T'] = (
            self.train_end_date - pd.to_datetime(rfm_data['first_order'])
        ).dt.days
        
        # Select relevant columns for BG/NBD
        training_data = rfm_data[['customer_id', 'frequency', 'recency', 'T', 'monetary']].copy()
        
        # Filter to customers with at least 2 purchases (required for CLV modeling)
        training_data = training_data[training_data['frequency'] >= 2].copy()
        
        logger.info(f"Training data: {len(training_data)} customers with >=2 purchases")
        return training_data
    
    def prepare_holdout_data(
        self,
        orders_df: pd.DataFrame,
        customers_df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Prepare holdout data for evaluation.
        
        Uses orders in holdout window (prediction_date to holdout_end_date).
        
        Args:
            orders_df: Orders data
            customers_df: Customers data
        
        Returns:
            DataFrame with actual revenue in holdout period
        """
        logger.info("Preparing CLV holdout data...")
        
        # Filter to holdout period
        orders_holdout = orders_df[
            (orders_df['order_date'] >= self.prediction_date) &
            (orders_df['order_date'] <= self.holdout_end_date)
        ].copy()
        
        # Calculate actual revenue in holdout period
        holdout_revenue = orders_holdout.groupby('customer_id')['order_total'].sum().reset_index()
        holdout_revenue.columns = ['customer_id', 'actual_holdout_revenue']
        
        logger.info(f"Holdout data: {len(holdout_revenue)} customers with purchases in holdout")
        return holdout_revenue
    
    def evaluate_clv_predictions(
        self,
        predicted_clv: pd.DataFrame,
        holdout_revenue: pd.DataFrame,
        prediction_horizon_days: int = 180
    ) -> Dict[str, float]:
        """
        Evaluate CLV predictions against actual holdout revenue.
        
        Args:
            predicted_clv: DataFrame with customer_id and predicted CLV
            holdout_revenue: DataFrame with customer_id and actual revenue
            prediction_horizon_days: Number of days CLV was predicted for
        
        Returns:
            Dictionary with evaluation metrics
        """
        logger.info("Evaluating CLV predictions...")
        
        # Merge predictions with actuals
        evaluation_data = predicted_clv.merge(
            holdout_revenue,
            on='customer_id',
            how='inner'
        )
        
        if len(evaluation_data) == 0:
            logger.warning("No overlap between predictions and holdout data")
            return {}
        
        # Scale predicted CLV to match holdout period
        # If CLV is annual, scale to holdout period
        clv_column = None
        for col in predicted_clv.columns:
            if 'clv' in col.lower() and 'predicted' in col.lower():
                clv_column = col
                break
        
        if clv_column is None:
            clv_column = predicted_clv.columns[1]  # Assume second column is CLV
        
        # Scale CLV to holdout period (assuming CLV is annual)
        scale_factor = prediction_horizon_days / 365
        evaluation_data['scaled_predicted_clv'] = evaluation_data[clv_column] * scale_factor
        
        # Calculate metrics
        actuals = evaluation_data['actual_holdout_revenue'].values
        predictions = evaluation_data['scaled_predicted_clv'].values
        
        metrics = {
            'n_customers': len(evaluation_data),
            'mae': np.mean(np.abs(actuals - predictions)),
            'rmse': np.sqrt(np.mean((actuals - predictions) ** 2)),
            'mape': np.mean(np.abs((actuals - predictions) / (actuals + 1e-6))) * 100,
            'total_actual_revenue': actuals.sum(),
            'total_predicted_revenue': predictions.sum(),
            'revenue_bias': (predictions.sum() - actuals.sum()) / actuals.sum() * 100,
        }
        
        # Calculate R-squared
        ss_res = np.sum((actuals - predictions) ** 2)
        ss_tot = np.sum((actuals - np.mean(actuals)) ** 2)
        metrics['r_squared'] = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
        
        # Calculate decile lift
        evaluation_data['decile'] = pd.qcut(
            evaluation_data['scaled_predicted_clv'],
            10,
            labels=False,
            duplicates='drop'
        )
        
        decile_performance = evaluation_data.groupby('decile').agg({
            'actual_holdout_revenue': 'sum',
            'scaled_predicted_clv': 'sum'
        })
        
        metrics['decile_lift'] = {}
        for decile in decile_performance.index:
            actual = decile_performance.loc[decile, 'actual_holdout_revenue']
            predicted = decile_performance.loc[decile, 'scaled_predicted_clv']
            metrics['decile_lift'][f'decile_{decile}'] = actual / predicted if predicted > 0 else 0
        
        logger.info(f"CLV Evaluation Results:")
        logger.info(f"  MAE: ₹{metrics['mae']:,.2f}")
        logger.info(f"  RMSE: ₹{metrics['rmse']:,.2f}")
        logger.info(f"  MAPE: {metrics['mape']:.2f}%")
        logger.info(f"  R²: {metrics['r_squared']:.3f}")
        logger.info(f"  Revenue Bias: {metrics['revenue_bias']:.2f}%")
        
        return metrics
    
    def run_backtest(
        self,
        orders_df: pd.DataFrame,
        customers_df: pd.DataFrame,
        clv_predictor,
        prediction_horizon_days: int = 180
    ) -> Dict[str, float]:
        """
        Run complete CLV backtest pipeline.
        
        Args:
            orders_df: Orders data
            customers_df: Customers data
            clv_predictor: CLVPredictor instance
            prediction_horizon_days: Number of days CLV was predicted for
        
        Returns:
            Dictionary with evaluation metrics
        """
        logger.info("Running CLV backtest pipeline...")
        
        # Prepare training data
        training_data = self.prepare_training_data(orders_df, customers_df)
        
        # Prepare holdout data
        holdout_revenue = self.prepare_holdout_data(orders_df, customers_df)
        
        # Fit CLV model on training data
        # (This would call the actual CLV predictor)
        # For now, we'll create placeholder predictions
        logger.info("Fitting CLV model on training data...")
        # clv_predictor.fit(training_data)
        
        # Generate predictions
        # predicted_clv = clv_predictor.predict(horizon_days=prediction_horizon_days)
        
        # Placeholder: create dummy predictions for testing
        customer_ids = training_data['customer_id'].unique()
        predicted_clv = pd.DataFrame({
            'customer_id': customer_ids,
            'predicted_clv_180d': np.random.uniform(1000, 10000, len(customer_ids))
        })
        
        # Evaluate predictions
        metrics = self.evaluate_clv_predictions(
            predicted_clv,
            holdout_revenue,
            prediction_horizon_days
        )
        
        return metrics
    
    def generate_backtest_report(
        self,
        metrics: Dict[str, float],
        output_path: Optional[Path] = None
    ) -> str:
        """
        Generate a human-readable backtest report.
        
        Args:
            metrics: Evaluation metrics dictionary
            output_path: Optional path to save report
        
        Returns:
            Report string
        """
        report = f"""
CLV Backtest Report
{'=' * 50}

Configuration:
- Training End Date: {self.train_end_date}
- Prediction Date: {self.prediction_date}
- Holdout End Date: {self.holdout_end_date}
- Holdout Period: {(self.holdout_end_date - self.prediction_date).days} days

Evaluation Metrics:
- Customers Evaluated: {metrics.get('n_customers', 0)}
- Mean Absolute Error (MAE): ₹{metrics.get('mae', 0):,.2f}
- Root Mean Squared Error (RMSE): ₹{metrics.get('rmse', 0):,.2f}
- Mean Absolute Percentage Error (MAPE): {metrics.get('mape', 0):.2f}%
- R-Squared: {metrics.get('r_squared', 0):.3f}

Revenue Comparison:
- Total Actual Revenue: ₹{metrics.get('total_actual_revenue', 0):,.2f}
- Total Predicted Revenue: ₹{metrics.get('total_predicted_revenue', 0):,.2f}
- Revenue Bias: {metrics.get('revenue_bias', 0):.2f}%

Decile Lift:
"""
        
        for decile, lift in metrics.get('decile_lift', {}).items():
            report += f"- {decile}: {lift:.2f}x\n"
        
        report += f"""
Interpretation:
- MAE < ₹1,000: Excellent accuracy
- MAE < ₹5,000: Good accuracy
- MAE < ₹10,000: Acceptable accuracy
- R² > 0.5: Strong predictive power
- R² > 0.3: Moderate predictive power
- Revenue Bias within ±10%: Well-calibrated
"""
        
        if output_path:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(report)
            logger.info(f"Backtest report saved to {output_path}")
        
        return report


def run_clv_backtesting_pipeline(
    train_end_date: str = "2024-06-30",
    prediction_date: str = "2024-06-30",
    holdout_end_date: str = "2024-12-31"
) -> Dict[str, float]:
    """
    Convenience function to run CLV backtesting pipeline.
    
    Args:
        train_end_date: End date for training data
        prediction_date: Date when CLV predictions are made
        holdout_end_date: End date for holdout evaluation period
    
    Returns:
        Dictionary with evaluation metrics
    """
    backtester = CLVBacktester(
        train_end_date=train_end_date,
        prediction_date=prediction_date,
        holdout_end_date=holdout_end_date
    )
    
    # Load data (placeholder - implement actual data loading)
    # orders_df = pd.read_csv(settings.PROCESSED_DATA_DIR / "orders.csv")
    # customers_df = pd.read_csv(settings.PROCESSED_DATA_DIR / "customers.csv")
    # clv_predictor = CLVPredictor()
    
    # metrics = backtester.run_backtest(orders_df, customers_df, clv_predictor)
    
    # report = backtester.generate_backtest_report(metrics)
    
    logger.warning("Data loading not implemented - returning empty metrics")
    return {}
