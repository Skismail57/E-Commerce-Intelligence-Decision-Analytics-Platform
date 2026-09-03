"""
Survival Analysis Module
Implements Kaplan-Meier survival curves and Cox Proportional Hazards model for customer churn prediction.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
from lifelines import KaplanMeierFitter, CoxPHFitter
from lifelines.utils import concordance_index
import matplotlib.pyplot as plt
from config.logging_config import get_logger

logger = get_logger(__name__)


class CustomerSurvivalAnalysis:
    """
    Survival analysis for customer churn prediction.
    
    Instead of binary churn prediction, survival analysis estimates:
    - When a customer is likely to churn
    - Survival probability over time
    - Hazard rates based on customer characteristics
    
    Methods:
    - Kaplan-Meier: Non-parametric survival curve estimation
    - Cox Proportional Hazards: Semi-parametric model with covariates
    """
    
    def __init__(self, analysis_as_of_date: str = "2024-12-31"):
        """
        Initialize survival analysis engine.
        
        Args:
            analysis_as_of_date: Date for analysis cutoff
        """
        self.analysis_as_of_date = pd.to_datetime(analysis_as_of_date)
        self.km_fitter = KaplanMeierFitter()
        self.cox_fitter = CoxPHFitter()
        self.km_model = None
        self.cox_model = None
    
    def prepare_survival_data(
        self,
        customers_df: pd.DataFrame,
        orders_df: pd.DataFrame,
        features_df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Prepare data for survival analysis.
        
        Args:
            customers_df: Customer data
            orders_df: Order data
            features_df: Customer features (RFM, behavioral, etc.)
        
        Returns:
            DataFrame with survival analysis columns:
            - duration: Time to event or censoring
            - event: Churn indicator (1=churned, 0=censored)
            - covariates: Customer characteristics
        """
        logger.info("Preparing survival analysis data...")
        
        # Prepare orders data
        orders_df['order_date'] = pd.to_datetime(orders_df['order_date'])
        orders_df = orders_df[orders_df['order_date'] <= self.analysis_as_of_date]
        
        # Calculate customer tenure and last order date
        customer_activity = orders_df.groupby('customer_id').agg({
            'order_date': ['min', 'max', 'count']
        }).reset_index()
        customer_activity.columns = ['customer_id', 'first_order_date', 
                                   'last_order_date', 'total_orders']
        
        # Merge with customer registration
        survival_data = customers_df[['customer_id', 'registration_date']].copy()
        survival_data['registration_date'] = pd.to_datetime(survival_data['registration_date'])
        survival_data = survival_data.merge(customer_activity, on='customer_id', how='left')
        
        # Fill missing values for customers with no orders
        survival_data['first_order_date'] = survival_data['first_order_date'].fillna(
            survival_data['registration_date']
        )
        survival_data['last_order_date'] = survival_data['last_order_date'].fillna(
            survival_data['registration_date']
        )
        survival_data['total_orders'] = survival_data['total_orders'].fillna(0)
        
        # Calculate duration (time from first order to analysis date or last order)
        survival_data['duration'] = (
            survival_data['last_order_date'] - survival_data['first_order_date']
        ).dt.days + 1
        
        # Define churn event (no order in last 90 days)
        churn_cutoff = self.analysis_as_of_date - timedelta(days=90)
        survival_data['event'] = (
            (survival_data['last_order_date'] < churn_cutoff) & 
            (survival_data['total_orders'] > 0)
        ).astype(int)
        
        # Merge with features
        if features_df is not None:
            survival_data = survival_data.merge(features_df, on='customer_id', how='left')
        
        logger.info(f"Prepared survival data for {len(survival_data)} customers")
        logger.info(f"Churn rate: {survival_data['event'].mean():.2%}")
        
        return survival_data
    
    def fit_kaplan_meier(self, survival_data: pd.DataFrame) -> KaplanMeierFitter:
        """
        Fit Kaplan-Meier survival curve.
        
        Args:
            survival_data: DataFrame with 'duration' and 'event' columns
        
        Returns:
            Fitted KaplanMeierFitter model
        """
        logger.info("Fitting Kaplan-Meier survival curve...")
        
        self.km_model = self.km_fitter.fit(
            survival_data['duration'],
            survival_data['event'],
            label='Customer Survival'
        )
        
        logger.info(f"Kaplan-Meier model fitted. Median survival: {self.km_model.median_survival_time_:.0f} days")
        
        return self.km_model
    
    def fit_cox_proportional_hazards(
        self,
        survival_data: pd.DataFrame,
        covariates: List[str]
    ) -> CoxPHFitter:
        """
        Fit Cox Proportional Hazards model.
        
        Args:
            survival_data: DataFrame with survival data and covariates
            covariates: List of covariate column names
        
        Returns:
            Fitted CoxPHFitter model
        """
        logger.info(f"Fitting Cox Proportional Hazards model with {len(covariates)} covariates...")
        
        # Prepare data for Cox model
        cox_data = survival_data[['customer_id', 'duration', 'event'] + covariates].copy()
        
        # Handle missing values
        cox_data = cox_data.dropna()
        
        # Fit Cox model
        self.cox_model = self.cox_fitter.fit(
            cox_data[['duration', 'event'] + covariates],
            duration_col='duration',
            event_col='event'
        )
        
        # Calculate concordance index
        c_index = self.cox_model.concordance_index_
        logger.info(f"Cox model fitted. Concordance index: {c_index:.3f}")
        
        return self.cox_model
    
    def predict_survival_probability(
        self,
        customer_id: int,
        time_points: List[int] = None
    ) -> Dict[int, float]:
        """
        Predict survival probability for a customer at specific time points.
        
        Args:
            customer_id: Customer ID
            time_points: List of time points (days) to predict survival probability
        
        Returns:
            Dictionary mapping time points to survival probabilities
        """
        if time_points is None:
            time_points = [30, 60, 90, 180, 365]
        
        if self.km_model is None:
            raise ValueError("Kaplan-Meier model not fitted. Call fit_kaplan_meier first.")
        
        # Get survival probabilities from KM curve
        survival_probs = {}
        for t in time_points:
            if t in self.km_model.survival_function_.index:
                survival_probs[t] = float(self.km_model.survival_function_.loc[t].values[0])
            else:
                # Interpolate
                survival_probs[t] = float(self.km_model.predict(t))
        
        return survival_probs
    
    def predict_hazard_ratio(
        self,
        customer_features: pd.DataFrame
    ) -> float:
        """
        Predict hazard ratio for a customer using Cox model.
        
        Args:
            customer_features: DataFrame with customer covariates
        
        Returns:
            Hazard ratio (relative risk compared to baseline)
        """
        if self.cox_model is None:
            raise ValueError("Cox model not fitted. Call fit_cox_proportional_hazards first.")
        
        hazard_ratio = self.cox_model.predict_partial_hazard(customer_features)
        return float(hazard_ratio.values[0])
    
    def get_survival_curve_data(self) -> pd.DataFrame:
        """
        Get survival curve data for plotting.
        
        Returns:
            DataFrame with time points and survival probabilities
        """
        if self.km_model is None:
            raise ValueError("Kaplan-Meier model not fitted. Call fit_kaplan_meier first.")
        
        survival_df = self.km_model.survival_function_.reset_index()
        survival_df.columns = ['days', 'survival_probability']
        
        return survival_df
    
    def get_cox_coefficients(self) -> pd.DataFrame:
        """
        Get Cox model coefficients and their interpretation.
        
        Returns:
            DataFrame with coefficients, p-values, and hazard ratios
        """
        if self.cox_model is None:
            raise ValueError("Cox model not fitted. Call fit_cox_proportional_hazards first.")
        
        summary = self.cox_model.summary
        summary['hazard_ratio'] = np.exp(summary['coef'])
        summary['interpretation'] = summary['hazard_ratio'].apply(
            lambda x: 'Increased risk' if x > 1 else 'Decreased risk'
        )
        
        return summary
    
    def segment_survival_curves(
        self,
        survival_data: pd.DataFrame,
        segment_column: str
    ) -> Dict[str, KaplanMeierFitter]:
        """
        Fit separate survival curves for customer segments.
        
        Args:
            survival_data: DataFrame with survival data
            segment_column: Column name to segment by
        
        Returns:
            Dictionary mapping segment names to fitted KM models
        """
        logger.info(f"Fitting survival curves by {segment_column}...")
        
        segment_models = {}
        
        for segment in survival_data[segment_column].unique():
            segment_data = survival_data[survival_data[segment_column] == segment]
            
            km = KaplanMeierFitter()
            km.fit(
                segment_data['duration'],
                segment_data['event'],
                label=f'{segment_column}={segment}'
            )
            
            segment_models[segment] = km
        
        return segment_models
    
    def generate_customer_survival_report(
        self,
        customer_id: int,
        survival_data: pd.DataFrame,
        customer_features: pd.DataFrame = None
    ) -> Dict:
        """
        Generate comprehensive survival report for a customer.
        
        Args:
            customer_id: Customer ID
            survival_data: Survival analysis data
            customer_features: Customer covariates for Cox model
        
        Returns:
            Dictionary with survival analysis results
        """
        customer_data = survival_data[survival_data['customer_id'] == customer_id]
        
        if len(customer_data) == 0:
            return {"error": "Customer not found"}
        
        row = customer_data.iloc[0]
        
        # Predict survival probabilities
        survival_probs = self.predict_survival_probability(customer_id)
        
        # Calculate hazard ratio if Cox model is fitted
        hazard_ratio = None
        risk_factors = []
        if self.cox_model is not None and customer_features is not None:
            try:
                hazard_ratio = self.predict_hazard_ratio(customer_features)
                
                # Get top risk factors
                coefficients = self.get_cox_coefficients()
                coefficients = coefficients.sort_values('hazard_ratio', ascending=False)
                risk_factors = [
                    {
                        'factor': idx,
                        'hazard_ratio': row['hazard_ratio'],
                        'interpretation': row['interpretation']
                    }
                    for idx, row in coefficients.head(5).iterrows()
                ]
            except Exception as e:
                logger.warning(f"Could not calculate hazard ratio: {e}")
        
        # Determine survival tier
        survival_90d = survival_probs.get(90, 0)
        if survival_90d > 0.8:
            survival_tier = "Low Risk"
        elif survival_90d > 0.5:
            survival_tier = "Medium Risk"
        else:
            survival_tier = "High Risk"
        
        report = {
            "customer_id": customer_id,
            "survival_tier": survival_tier,
            "duration_days": int(row['duration']),
            "churned": bool(row['event']),
            "survival_probabilities": survival_probs,
            "hazard_ratio": hazard_ratio,
            "risk_factors": risk_factors,
            "recommendations": self._generate_survival_recommendations(
                survival_tier, survival_probs, hazard_ratio
            )
        }
        
        return report
    
    def _generate_survival_recommendations(
        self,
        survival_tier: str,
        survival_probs: Dict[int, float],
        hazard_ratio: Optional[float]
    ) -> List[str]:
        """Generate recommendations based on survival analysis"""
        recommendations = []
        
        if survival_tier == "High Risk":
            recommendations.append("Immediate retention intervention required")
            recommendations.append("Consider personalized offers to re-engage")
        elif survival_tier == "Medium Risk":
            recommendations.append("Monitor customer engagement closely")
            recommendations.append("Proactive communication recommended")
        else:
            recommendations.append("Customer stable - maintain engagement")
        
        if hazard_ratio and hazard_ratio > 2:
            recommendations.append("Customer has high relative risk - investigate specific factors")
        
        # Check survival probability drop
        if survival_probs.get(30, 1) - survival_probs.get(90, 1) > 0.3:
            recommendations.append("Rapid survival probability decline detected")
        
        return recommendations


def run_survival_analysis_pipeline(
    customers_df: pd.DataFrame,
    orders_df: pd.DataFrame,
    features_df: pd.DataFrame,
    analysis_as_of_date: str = "2024-12-31",
    cox_covariates: List[str] = None
) -> Tuple[CustomerSurvivalAnalysis, Dict]:
    """
    Convenience function to run complete survival analysis pipeline.
    
    Args:
        customers_df: Customer data
        orders_df: Order data
        features_df: Customer features
        analysis_as_of_date: Date for analysis cutoff
        cox_covariates: List of covariate names for Cox model
    
    Returns:
        Tuple of (fitted model, analysis results)
    """
    analyzer = CustomerSurvivalAnalysis(analysis_as_of_date)
    
    # Prepare data
    survival_data = analyzer.prepare_survival_data(customers_df, orders_df, features_df)
    
    # Fit Kaplan-Meier
    km_model = analyzer.fit_kaplan_meier(survival_data)
    
    # Fit Cox model if covariates provided
    if cox_covariates:
        cox_model = analyzer.fit_cox_proportional_hazards(survival_data, cox_covariates)
    
    # Generate summary statistics
    results = {
        'total_customers': len(survival_data),
        'churned_customers': int(survival_data['event'].sum()),
        'churn_rate': float(survival_data['event'].mean()),
        'median_survival_time': float(km_model.median_survival_time_),
        'survival_probabilities': {
            30: float(km_model.predict(30)),
            60: float(km_model.predict(60)),
            90: float(km_model.predict(90)),
            180: float(km_model.predict(180)),
            365: float(km_model.predict(365)),
        }
    }
    
    if cox_covariates:
        results['concordance_index'] = float(cox_model.concordance_index_)
        results['cox_coefficients'] = analyzer.get_cox_coefficients().to_dict()
    
    return analyzer, results
