"""
Uplift Modeling Module
Implements T-learner and X-learner for estimating treatment effects in marketing campaigns.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, mean_squared_error
import matplotlib.pyplot as plt
from config.logging_config import get_logger

logger = get_logger(__name__)


class UpliftModeler:
    """
    Uplift modeling for marketing campaign effectiveness.
    
    Estimates the causal effect of treatments (e.g., marketing campaigns)
    on outcomes (e.g., purchase, spend) using meta-learners:
    - T-learner: Separate models for treatment and control groups
    - X-learner: Cross-fitting with propensity scores
    
    Applications:
    - Marketing campaign targeting optimization
    - Personalized treatment assignment
    - Heterogeneous treatment effect estimation
    """
    
    def __init__(self, outcome_type: str = 'continuous'):
        """
        Initialize uplift modeler.
        
        Args:
            outcome_type: Type of outcome ('continuous' or 'binary')
        """
        self.outcome_type = outcome_type
        self.t_learner = None
        self.t_learner_control = None
        self.t_learner_treatment = None
        self.x_learner = None
        self.propensity_model = None
        self.feature_importance = {}
    
    def prepare_uplift_data(
        self,
        customers_df: pd.DataFrame,
        campaigns_df: pd.DataFrame,
        orders_df: pd.DataFrame,
        features_df: pd.DataFrame = None
    ) -> pd.DataFrame:
        """
        Prepare data for uplift modeling.
        
        Args:
            customers_df: Customer data
            campaigns_df: Campaign assignment data (customer_id, campaign_id, treatment)
            orders_df: Order data for outcome calculation
            features_df: Customer features (optional)
        
        Returns:
            DataFrame with treatment, outcome, and features
        """
        logger.info("Preparing uplift modeling data...")
        
        # Prepare orders data
        orders_df['order_date'] = pd.to_datetime(orders_df['order_date'])
        
        # Calculate outcome (e.g., purchase after campaign)
        # For binary outcome: whether customer made a purchase
        # For continuous outcome: total spend after campaign
        
        # Merge campaigns with customers
        uplift_data = campaigns_df.merge(
            customers_df[['customer_id', 'registration_date']],
            on='customer_id',
            how='left'
        )
        
        # Calculate outcome based on campaign date
        # Assuming campaigns_df has campaign_date
        if 'campaign_date' in campaigns_df.columns:
            campaigns_df['campaign_date'] = pd.to_datetime(campaigns_df['campaign_date'])
            
            # Calculate post-campaign purchases
            for _, row in campaigns_df.iterrows():
                customer_id = row['customer_id']
                campaign_date = row['campaign_date']
                
                # Get orders after campaign date (within 30 days)
                post_campaign_orders = orders_df[
                    (orders_df['customer_id'] == customer_id) &
                    (orders_df['order_date'] > campaign_date) &
                    (orders_df['order_date'] <= campaign_date + pd.Timedelta(days=30))
                ]
                
                if self.outcome_type == 'binary':
                    outcome = 1 if len(post_campaign_orders) > 0 else 0
                else:
                    outcome = post_campaign_orders['order_total'].sum()
                
                uplift_data.loc[uplift_data['customer_id'] == customer_id, 'outcome'] = outcome
        
        # Add features if available
        if features_df is not None:
            uplift_data = uplift_data.merge(features_df, on='customer_id', how='left')
        
        # Ensure treatment is binary (0=control, 1=treatment)
        uplift_data['treatment'] = uplift_data['treatment'].astype(int)
        
        logger.info(f"Prepared {len(uplift_data)} samples for uplift modeling")
        logger.info(f"Treatment group: {uplift_data['treatment'].sum()}, Control group: {(1-uplift_data['treatment']).sum()}")
        
        return uplift_data
    
    def fit_t_learner(
        self,
        data: pd.DataFrame,
        feature_cols: List[str],
        model_type: str = 'gradient_boosting'
    ) -> Dict:
        """
        Fit T-learner (Two-model approach).
        
        Args:
            data: DataFrame with treatment, outcome, and features
            feature_cols: List of feature column names
            model_type: Model type ('gradient_boosting', 'random_forest', 'linear')
        
        Returns:
            Dictionary with training results
        """
        logger.info(f"Fitting T-learner with {model_type}...")
        
        # Split data by treatment
        treatment_data = data[data['treatment'] == 1]
        control_data = data[data['treatment'] == 0]
        
        # Initialize models
        if model_type == 'gradient_boosting':
            model_class = GradientBoostingRegressor
            model_params = {'n_estimators': 100, 'max_depth': 5, 'random_state': 42}
        elif model_type == 'random_forest':
            model_class = RandomForestRegressor
            model_params = {'n_estimators': 100, 'max_depth': 10, 'random_state': 42}
        elif model_type == 'linear':
            model_class = LinearRegression
            model_params = {}
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        # Fit separate models for treatment and control
        self.t_learner_treatment = model_class(**model_params)
        self.t_learner_treatment.fit(treatment_data[feature_cols], treatment_data['outcome'])
        
        self.t_learner_control = model_class(**model_params)
        self.t_learner_control.fit(control_data[feature_cols], control_data['outcome'])
        
        # Calculate uplift on training data
        treatment_pred = self.t_learner_treatment.predict(data[feature_cols])
        control_pred = self.t_learner_control.predict(data[feature_cols])
        uplift = treatment_pred - control_pred
        
        # Store feature importance if available
        if hasattr(self.t_learner_treatment, 'feature_importances_'):
            self.feature_importance['t_learner'] = dict(zip(
                feature_cols, self.t_learner_treatment.feature_importances_
            ))
        
        results = {
            'model_type': model_type,
            'n_treatment': len(treatment_data),
            'n_control': len(control_data),
            'mean_uplift': float(np.mean(uplift)),
            'std_uplift': float(np.std(uplift)),
            'uplift_percentile_25': float(np.percentile(uplift, 25)),
            'uplift_percentile_50': float(np.percentile(uplift, 50)),
            'uplift_percentile_75': float(np.percentile(uplift, 75)),
        }
        
        logger.info(f"T-learner fitted. Mean uplift: {results['mean_uplift']:.4f}")
        
        return results
    
    def fit_x_learner(
        self,
        data: pd.DataFrame,
        feature_cols: List[str],
        model_type: str = 'gradient_boosting'
    ) -> Dict:
        """
        Fit X-learner (Cross-fitting with propensity scores).
        
        Args:
            data: DataFrame with treatment, outcome, and features
            feature_cols: List of feature column names
            model_type: Model type for outcome models
        
        Returns:
            Dictionary with training results
        """
        logger.info(f"Fitting X-learner with {model_type}...")
        
        # Fit propensity model
        self.propensity_model = LogisticRegression(random_state=42, max_iter=1000)
        self.propensity_model.fit(data[feature_cols], data['treatment'])
        
        # Get propensity scores
        propensity_scores = self.propensity_model.predict_proba(data[feature_cols])[:, 1]
        
        # Split data by treatment
        treatment_data = data[data['treatment'] == 1].copy()
        control_data = data[data['treatment'] == 0].copy()
        
        # Initialize outcome models
        if model_type == 'gradient_boosting':
            model_class = GradientBoostingRegressor
            model_params = {'n_estimators': 100, 'max_depth': 5, 'random_state': 42}
        elif model_type == 'random_forest':
            model_class = RandomForestRegressor
            model_params = {'n_estimators': 100, 'max_depth': 10, 'random_state': 42}
        elif model_type == 'linear':
            model_class = LinearRegression
            model_params = {}
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        # Fit outcome models on opposite groups
        # Model for control group (trained on treatment data)
        mu_c = model_class(**model_params)
        mu_c.fit(treatment_data[feature_cols], treatment_data['outcome'])
        
        # Model for treatment group (trained on control data)
        mu_t = model_class(**model_params)
        mu_t.fit(control_data[feature_cols], control_data['outcome'])
        
        # Calculate imputed treatment effects
        # For treated units: D_i = Y_i - mu_c(X_i)
        treatment_data['imputed_effect'] = treatment_data['outcome'] - mu_c.predict(treatment_data[feature_cols])
        
        # For control units: D_i = mu_t(X_i) - Y_i
        control_data['imputed_effect'] = mu_t.predict(control_data[feature_cols]) - control_data['outcome']
        
        # Combine imputed effects
        imputed_effects = pd.concat([
            treatment_data[['customer_id', 'imputed_effect', 'treatment']],
            control_data[['customer_id', 'imputed_effect', 'treatment']]
        ])
        
        # Fit final models on imputed effects
        # Model for treated units (weighted by propensity)
        g_t = model_class(**model_params)
        g_t.fit(treatment_data[feature_cols], treatment_data['imputed_effect'],
                sample_weight=1 / propensity_scores[treatment_data.index])
        
        # Model for control units (weighted by 1 - propensity)
        g_c = model_class(**model_params)
        g_c.fit(control_data[feature_cols], control_data['imputed_effect'],
                sample_weight=1 / (1 - propensity_scores[control_data.index]))
        
        # Calculate final uplift
        uplift_treated = g_t.predict(data[feature_cols])
        uplift_control = g_c.predict(data[feature_cols])
        
        # Combine using propensity scores
        final_uplift = propensity_scores * uplift_treated + (1 - propensity_scores) * uplift_control
        
        # Store models
        self.x_learner = {
            'mu_c': mu_c,
            'mu_t': mu_t,
            'g_t': g_t,
            'g_c': g_c
        }
        
        results = {
            'model_type': model_type,
            'n_treatment': len(treatment_data),
            'n_control': len(control_data),
            'mean_uplift': float(np.mean(final_uplift)),
            'std_uplift': float(np.std(final_uplift)),
            'uplift_percentile_25': float(np.percentile(final_uplift, 25)),
            'uplift_percentile_50': float(np.percentile(final_uplift, 50)),
            'uplift_percentile_75': float(np.percentile(final_uplift, 75)),
        }
        
        logger.info(f"X-learner fitted. Mean uplift: {results['mean_uplift']:.4f}")
        
        return results
    
    def predict_uplift(
        self,
        data: pd.DataFrame,
        feature_cols: List[str],
        method: str = 't_learner'
    ) -> np.ndarray:
        """
        Predict uplift for customers.
        
        Args:
            data: DataFrame with features
            feature_cols: List of feature column names
            method: Uplift method ('t_learner' or 'x_learner')
        
        Returns:
            Array of predicted uplift values
        """
        if method == 't_learner':
            if self.t_learner_treatment is None or self.t_learner_control is None:
                raise ValueError("T-learner not fitted. Call fit_t_learner first.")
            
            treatment_pred = self.t_learner_treatment.predict(data[feature_cols])
            control_pred = self.t_learner_control.predict(data[feature_cols])
            uplift = treatment_pred - control_pred
            
        elif method == 'x_learner':
            if self.x_learner is None or self.propensity_model is None:
                raise ValueError("X-learner not fitted. Call fit_x_learner first.")
            
            # Get propensity scores
            propensity_scores = self.propensity_model.predict_proba(data[feature_cols])[:, 1]
            
            # Get predictions from both models
            uplift_treated = self.x_learner['g_t'].predict(data[feature_cols])
            uplift_control = self.x_learner['g_c'].predict(data[feature_cols])
            
            # Combine using propensity scores
            uplift = propensity_scores * uplift_treated + (1 - propensity_scores) * uplift_control
            
        else:
            raise ValueError(f"Unknown method: {method}")
        
        return uplift
    
    def get_uplift_segments(
        self,
        data: pd.DataFrame,
        feature_cols: List[str],
        method: str = 't_learner',
        n_segments: int = 4
    ) -> pd.DataFrame:
        """
        Segment customers by uplift.
        
        Args:
            data: DataFrame with features
            feature_cols: List of feature column names
            method: Uplift method ('t_learner' or 'x_learner')
            n_segments: Number of segments
        
        Returns:
            DataFrame with customer uplift predictions and segments
        """
        uplift = self.predict_uplift(data, feature_cols, method)
        
        # Create segments based on uplift percentiles
        percentiles = np.linspace(0, 100, n_segments + 1)
        bins = np.percentile(uplift, percentiles)
        
        # Assign segments
        segments = []
        for u in uplift:
            for i in range(len(bins) - 1):
                if bins[i] <= u < bins[i + 1]:
                    segments.append(i)
                    break
            else:
                segments.append(n_segments - 1)  # Last segment
        
        result_df = data[['customer_id']].copy()
        result_df['uplift'] = uplift
        result_df['uplift_segment'] = segments
        
        # Add segment labels
        segment_labels = {
            0: 'Lost Causes',
            1: 'Persuadables',
            2: 'Sure Things',
            3: 'Sleeping Dogs'
        }
        result_df['segment_label'] = result_df['uplift_segment'].map(segment_labels)
        
        return result_df
    
    def calculate_uplift_metrics(
        self,
        data: pd.DataFrame,
        feature_cols: List[str],
        method: str = 't_learner',
        target_percentile: float = 0.3
    ) -> Dict:
        """
        Calculate uplift metrics for model evaluation.
        
        Args:
            data: DataFrame with treatment, outcome, and features
            feature_cols: List of feature column names
            method: Uplift method ('t_learner' or 'x_learner')
            target_percentile: Percentile for targeting (e.g., 0.3 = top 30%)
        
        Returns:
            Dictionary with uplift metrics
        """
        uplift = self.predict_uplift(data, feature_cols, method)
        
        # Sort by uplift
        data_with_uplift = data.copy()
        data_with_uplift['uplift'] = uplift
        data_with_uplift = data_with_uplift.sort_values('uplift', ascending=False)
        
        # Calculate metrics at different targeting levels
        metrics = {}
        
        for percentile in [0.1, 0.2, 0.3, 0.4, 0.5]:
            n_target = int(len(data_with_uplift) * percentile)
            target_data = data_with_uplift.head(n_target)
            
            # Calculate average treatment effect in target group
            treatment_outcome = target_data[target_data['treatment'] == 1]['outcome'].mean()
            control_outcome = target_data[target_data['treatment'] == 0]['outcome'].mean()
            ate = treatment_outcome - control_outcome
            
            metrics[f'ate_at_{int(percentile*100)}pct'] = float(ate)
        
        # Calculate overall ATE
        overall_treatment = data[data['treatment'] == 1]['outcome'].mean()
        overall_control = data[data['treatment'] == 0]['outcome'].mean()
        metrics['overall_ate'] = float(overall_treatment - overall_control)
        
        # Calculate uplift gain
        target_ate = metrics[f'ate_at_{int(target_percentile*100)}pct']
        metrics['uplift_gain'] = float(target_ate - metrics['overall_ate'])
        metrics['uplift_gain_pct'] = float((target_ate / metrics['overall_ate'] - 1) * 100) if metrics['overall_ate'] != 0 else 0
        
        return metrics
    
    def get_feature_importance(self, method: str = 't_learner', top_n: int = 10) -> pd.DataFrame:
        """
        Get feature importance for uplift model.
        
        Args:
            method: Uplift method
            top_n: Number of top features to return
        
        Returns:
            DataFrame with feature importance
        """
        if method == 't_learner' and 't_learner' in self.feature_importance:
            importance = self.feature_importance['t_learner']
        else:
            raise ValueError(f"Feature importance not available for {method}")
        
        importance_df = pd.DataFrame([
            {'feature': k, 'importance': v}
            for k, v in importance.items()
        ])
        
        importance_df = importance_df.sort_values('importance', ascending=False).head(top_n)
        
        return importance_df


def run_uplift_pipeline(
    customers_df: pd.DataFrame,
    campaigns_df: pd.DataFrame,
    orders_df: pd.DataFrame,
    features_df: pd.DataFrame = None,
    outcome_type: str = 'continuous'
) -> Tuple[UpliftModeler, Dict]:
    """
    Convenience function to run complete uplift modeling pipeline.
    
    Args:
        customers_df: Customer data
        campaigns_df: Campaign assignment data
        orders_df: Order data
        features_df: Customer features (optional)
        outcome_type: Type of outcome ('continuous' or 'binary')
    
    Returns:
        Tuple of (fitted modeler, analysis results)
    """
    modeler = UpliftModeler(outcome_type)
    
    # Prepare data
    data = modeler.prepare_uplift_data(customers_df, campaigns_df, orders_df, features_df)
    
    # Get feature columns
    exclude_cols = ['customer_id', 'treatment', 'outcome', 'campaign_id', 'campaign_date']
    feature_cols = [col for col in data.columns if col not in exclude_cols]
    
    # Fit T-learner
    t_results = modeler.fit_t_learner(data, feature_cols)
    
    # Fit X-learner
    x_results = modeler.fit_x_learner(data, feature_cols)
    
    # Calculate uplift metrics
    t_metrics = modeler.calculate_uplift_metrics(data, feature_cols, 't_learner')
    x_metrics = modeler.calculate_uplift_metrics(data, feature_cols, 'x_learner')
    
    results = {
        't_learner': {**t_results, **t_metrics},
        'x_learner': {**x_results, **x_metrics},
        'n_features': len(feature_cols),
        'n_samples': len(data)
    }
    
    return modeler, results
