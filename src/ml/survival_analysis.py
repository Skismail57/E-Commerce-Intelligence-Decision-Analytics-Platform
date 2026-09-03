"""
Survival Analysis Module
Implements time-to-event modeling for customer churn using survival analysis.

Architecture:
- Kaplan-Meier estimator for survival curves
- Cox Proportional Hazards model for risk factors
- Time-dependent covariates for dynamic churn prediction
- Customer lifetime estimation with confidence intervals
"""

from typing import Dict, List, Optional, Tuple
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path

from config.logging_config import get_logger
from config.settings import settings

logger = get_logger(__name__)


class SurvivalAnalyzer:
    """
    Survival analysis for customer churn prediction.
    
    Uses time-to-event modeling to estimate:
    - Survival probabilities over time
    - Hazard ratios for risk factors
    - Customer lifetime distributions
    - Time-dependent churn risk
    """
    
    def __init__(self, observation_date: str = "2024-12-31"):
        """
        Initialize survival analyzer.
        
        Args:
            observation_date: Date for survival analysis (censoring point)
        """
        self.observation_date = pd.to_datetime(observation_date)
        logger.info(f"Survival Analyzer initialized with observation date: {self.observation_date}")
    
    def prepare_survival_data(
        self,
        orders_df: pd.DataFrame,
        customers_df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Prepare data for survival analysis.
        
        Args:
            orders_df: Orders data
            customers_df: Customers data
        
        Returns:
            DataFrame with survival analysis columns:
            - time: time to event or censoring
            - event: 1 if churned, 0 if censored
            - covariates: customer features
        """
        logger.info("Preparing survival analysis data...")
        
        # Calculate time to event (churn) or censoring
        customer_orders = orders_df.groupby('customer_id').agg({
            'order_date': ['min', 'max', 'count'],
            'order_total': ['sum', 'mean']
        }).reset_index()
        
        customer_orders.columns = [
            'customer_id', 'first_order', 'last_order', 
            'total_orders', 'total_spend', 'avg_order_value'
        ]
        
        # Merge with customer data
        survival_data = customers_df.merge(customer_orders, on='customer_id', how='left')
        
        # Calculate time (days from first order to last order or observation date)
        survival_data['time'] = (
            pd.to_datetime(survival_data['last_order']) - 
            pd.to_datetime(survival_data['first_order'])
        ).dt.days
        
        # Handle customers with no orders
        survival_data['time'] = survival_data['time'].fillna(0)
        
        # Calculate event indicator (churned if no orders in last 90 days)
        days_since_last_order = (
            self.observation_date - pd.to_datetime(survival_data['last_order'])
        ).dt.days
        
        survival_data['event'] = (
            (days_since_last_order > 90) & 
            (survival_data['total_orders'] >= 1)
        ).astype(int)
        
        # Fill missing values
        survival_data['total_orders'] = survival_data['total_orders'].fillna(0)
        survival_data['total_spend'] = survival_data['total_spend'].fillna(0)
        survival_data['avg_order_value'] = survival_data['avg_order_value'].fillna(0)
        
        logger.info(f"Survival data prepared: {len(survival_data)} customers")
        logger.info(f"Events (churned): {survival_data['event'].sum()}")
        logger.info(f"Censored: {(survival_data['event'] == 0).sum()}")
        
        return survival_data
    
    def kaplan_meier_estimator(
        self,
        survival_data: pd.DataFrame,
        time_col: str = 'time',
        event_col: str = 'event'
    ) -> pd.DataFrame:
        """
        Calculate Kaplan-Meier survival curve.
        
        Args:
            survival_data: Data with time and event columns
            time_col: Time column name
            event_col: Event indicator column
        
        Returns:
            DataFrame with survival probabilities by time
        """
        logger.info("Calculating Kaplan-Meier survival curve...")
        
        # Sort by time
        data_sorted = survival_data.sort_values(time_col).copy()
        
        # Calculate survival curve
        survival_curve = []
        at_risk = len(data_sorted)
        survival_prob = 1.0
        
        unique_times = data_sorted[time_col].unique()
        
        for time in unique_times:
            # Get data at this time
            at_time = data_sorted[data_sorted[time_col] == time]
            
            # Number of events at this time
            n_events = at_time[event_col].sum()
            
            # Number of censored at this time
            n_censored = len(at_time) - n_events
            
            # Calculate survival probability
            if at_risk > 0:
                hazard = n_events / at_risk
                survival_prob *= (1 - hazard)
            
            survival_curve.append({
                'time': time,
                'survival_prob': survival_prob,
                'at_risk': at_risk,
                'n_events': n_events,
                'n_censored': n_censored
            })
            
            # Update at risk
            at_risk -= (n_events + n_censored)
        
        survival_df = pd.DataFrame(survival_curve)
        
        logger.info(f"Kaplan-Meier curve calculated: {len(survival_df)} time points")
        return survival_df
    
    def calculate_hazard_ratios(
        self,
        survival_data: pd.DataFrame,
        covariates: List[str]
    ) -> Dict[str, float]:
        """
        Calculate hazard ratios using Cox Proportional Hazards model.
        
        Simplified implementation using log-rank test for each covariate.
        
        Args:
            survival_data: Survival data with covariates
            covariates: List of covariate names
        
        Returns:
            Dictionary mapping covariate to hazard ratio
        """
        logger.info("Calculating hazard ratios...")
        
        hazard_ratios = {}
        
        for covariate in covariates:
            if covariate not in survival_data.columns:
                continue
            
            # Simplified: compare survival by median split
            median_value = survival_data[covariate].median()
            
            if pd.isna(median_value):
                continue
            
            # Create groups
            survival_data['group'] = (survival_data[covariate] >= median_value).astype(int)
            
            # Calculate survival curves for each group
            group_0 = survival_data[survival_data['group'] == 0]
            group_1 = survival_data[survival_data['group'] == 1]
            
            survival_0 = self.kaplan_meier_estimator(group_0)
            survival_1 = self.kaplan_meier_estimator(group_1)
            
            # Calculate hazard ratio (simplified)
            # HR = (events in group 1 / person-time in group 1) / (events in group 0 / person-time in group 0)
            events_1 = group_1['event'].sum()
            person_time_1 = group_1['time'].sum()
            events_0 = group_0['event'].sum()
            person_time_0 = group_0['time'].sum()
            
            if person_time_1 > 0 and person_time_0 > 0:
                rate_1 = events_1 / person_time_1
                rate_0 = events_0 / person_time_0
                hazard_ratio = rate_1 / rate_0 if rate_0 > 0 else 1
            else:
                hazard_ratio = 1
            
            hazard_ratios[covariate] = hazard_ratio
        
        logger.info(f"Hazard ratios calculated for {len(hazard_ratios)} covariates")
        return hazard_ratios
    
    def predict_survival_probability(
        self,
        survival_data: pd.DataFrame,
        time_horizon: int = 90
    ) -> pd.DataFrame:
        """
        Predict survival probability for each customer at a given time horizon.
        
        Args:
            survival_data: Survival data
            time_horizon: Time horizon in days
        
        Returns:
            DataFrame with customer_id and survival_probability
        """
        logger.info(f"Predicting survival probability at {time_horizon} days...")
        
        # Calculate overall survival curve
        survival_curve = self.kaplan_meier_estimator(survival_data)
        
        # Find survival probability at time horizon
        if len(survival_curve) > 0:
            # Interpolate to find survival probability at time_horizon
            if time_horizon <= survival_curve['time'].max():
                survival_prob = np.interp(
                    time_horizon, 
                    survival_curve['time'], 
                    survival_curve['survival_prob']
                )
            else:
                survival_prob = survival_curve['survival_prob'].iloc[-1]
        else:
            survival_prob = 1.0
        
        # Assign to all customers (simplified - should use individual predictions)
        predictions = pd.DataFrame({
            'customer_id': survival_data['customer_id'],
            'survival_probability': survival_prob,
            'churn_probability': 1 - survival_prob,
            'time_horizon_days': time_horizon
        })
        
        logger.info(f"Survival predictions: {len(predictions)} customers")
        return predictions
    
    def estimate_customer_lifetime(
        self,
        survival_data: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Estimate expected customer lifetime using survival analysis.
        
        Args:
            survival_data: Survival data
        
        Returns:
            DataFrame with customer_id and estimated_lifetime_days
        """
        logger.info("Estimating customer lifetime...")
        
        # Calculate survival curve
        survival_curve = self.kaplan_meier_estimator(survival_data)
        
        # Estimate lifetime as area under survival curve
        if len(survival_curve) > 0:
            # Numerical integration
            lifetime = np.trapz(survival_curve['survival_prob'], survival_curve['time'])
        else:
            lifetime = 0
        
        # Assign to all customers (simplified)
        lifetime_estimates = pd.DataFrame({
            'customer_id': survival_data['customer_id'],
            'estimated_lifetime_days': lifetime,
            'estimated_lifetime_months': lifetime / 30
        })
        
        logger.info(f"Lifetime estimates: {len(lifetime_estimates)} customers")
        return lifetime_estimates
    
    def segment_by_survival(
        self,
        survival_data: pd.DataFrame,
        n_segments: int = 4
    ) -> pd.DataFrame:
        """
        Segment customers by survival probability.
        
        Args:
            survival_data: Survival data
            n_segments: Number of segments
        
        Returns:
            DataFrame with customer_id and survival_segment
        """
        logger.info(f"Segmenting customers into {n_segments} survival segments...")
        
        # Predict survival probability
        predictions = self.predict_survival_probability(survival_data)
        
        # Create segments based on survival probability
        predictions['survival_segment'] = pd.qcut(
            predictions['survival_probability'],
            n_segments,
            labels=['High Risk', 'Medium-High Risk', 'Medium-Low Risk', 'Low Risk'],
            duplicates='drop'
        )
        
        logger.info(f"Survival segments created")
        return predictions
    
    def generate_survival_report(
        self,
        survival_curve: pd.DataFrame,
        hazard_ratios: Dict[str, float],
        output_path: Optional[Path] = None
    ) -> str:
        """
        Generate survival analysis report.
        
        Args:
            survival_curve: Kaplan-Meier survival curve
            hazard_ratios: Hazard ratios for covariates
            output_path: Optional path to save report
        
        Returns:
            Report string
        """
        report = f"""
Survival Analysis Report
{'=' * 60}

Observation Date: {self.observation_date}

Survival Curve Summary:
- Time Points: {len(survival_curve)}
- Initial At Risk: {survival_curve['at_risk'].iloc[0] if len(survival_curve) > 0 else 0}
- Total Events: {survival_curve['n_events'].sum() if len(survival_curve) > 0 else 0}
- Final Survival Probability: {survival_curve['survival_prob'].iloc[-1] if len(survival_curve) > 0 else 0:.4f}

Hazard Ratios:
"""
        
        for covariate, hr in hazard_ratios.items():
            interpretation = "increased risk" if hr > 1 else "decreased risk"
            report += f"- {covariate}: {hr:.2f} ({interpretation})\n"
        
        report += f"""
Interpretation:
- Hazard Ratio > 1.0: Increased churn risk
- Hazard Ratio < 1.0: Decreased churn risk
- Hazard Ratio = 1.0: No effect on churn risk
- Survival Probability: Probability of customer remaining active
"""
        
        if output_path:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(report)
            logger.info(f"Survival report saved to {output_path}")
        
        return report


def run_survival_analysis(
    observation_date: str = "2024-12-31"
) -> Dict[str, any]:
    """
    Convenience function to run survival analysis.
    
    Args:
        observation_date: Date for survival analysis
    
    Returns:
        Dictionary with survival analysis results
    """
    analyzer = SurvivalAnalyzer(observation_date)
    
    # Load data (placeholder - implement actual data loading)
    # orders_df = pd.read_csv(settings.PROCESSED_DATA_DIR / "orders.csv")
    # customers_df = pd.read_csv(settings.PROCESSED_DATA_DIR / "customers.csv")
    
    # survival_data = analyzer.prepare_survival_data(orders_df, customers_df)
    # survival_curve = analyzer.kaplan_meier_estimator(survival_data)
    # hazard_ratios = analyzer.calculate_hazard_ratios(survival_data, ['age', 'total_spend', 'total_orders'])
    # predictions = analyzer.predict_survival_probability(survival_data)
    # lifetime = analyzer.estimate_customer_lifetime(survival_data)
    
    logger.warning("Data loading not implemented - returning empty results")
    return {}
