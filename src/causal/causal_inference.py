"""
Causal Inference Module
Implements Propensity Score Matching (PSM), Inverse Probability Weighting (IPW), and Difference-in-Differences (DiD).
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import pairwise_distances
from scipy import stats
import matplotlib.pyplot as plt
from config.logging_config import get_logger

logger = get_logger(__name__)


class CausalInferenceEngine:
    """
    Causal inference engine for estimating treatment effects.
    
    Methods:
    - Propensity Score Matching (PSM): Match treated and control units
    - Inverse Probability Weighting (IPW): Weight observations by propensity scores
    - Difference-in-Differences (DiD): Compare changes over time between groups
    
    Applications:
    - Marketing campaign effectiveness
    - Policy impact evaluation
    - Feature rollout analysis
    """
    
    def __init__(self):
        """Initialize causal inference engine"""
        self.propensity_model = None
        self.propensity_scores = None
        self.matched_pairs = None
        self.weights = None
    
    def estimate_propensity_scores(
        self,
        data: pd.DataFrame,
        treatment_col: str,
        feature_cols: List[str]
    ) -> np.ndarray:
        """
        Estimate propensity scores using logistic regression.
        
        Args:
            data: DataFrame with treatment and features
            treatment_col: Column name for treatment indicator
            feature_cols: List of feature column names
        
        Returns:
            Array of propensity scores
        """
        logger.info("Estimating propensity scores...")
        
        X = data[feature_cols].values
        y = data[treatment_col].values
        
        self.propensity_model = LogisticRegression(random_state=42, max_iter=1000)
        self.propensity_model.fit(X, y)
        
        self.propensity_scores = self.propensity_model.predict_proba(X)[:, 1]
        
        logger.info(f"Propensity scores estimated. Mean: {self.propensity_scores.mean():.3f}")
        
        return self.propensity_scores
    
    def propensity_score_matching(
        self,
        data: pd.DataFrame,
        treatment_col: str,
        outcome_col: str,
        feature_cols: List[str],
        method: str = 'nearest',
        caliper: float = 0.1,
        ratio: int = 1
    ) -> Dict:
        """
        Perform Propensity Score Matching.
        
        Args:
            data: DataFrame with treatment, outcome, and features
            treatment_col: Column name for treatment indicator
            outcome_col: Column name for outcome
            feature_cols: List of feature column names
            method: Matching method ('nearest', 'caliper')
            caliper: Caliper for matching (max allowed propensity score difference)
            ratio: Number of control units to match per treated unit
        
        Returns:
            Dictionary with matching results and treatment effect estimate
        """
        logger.info(f"Performing PSM with {method} matching...")
        
        # Estimate propensity scores
        propensity_scores = self.estimate_propensity_scores(data, treatment_col, feature_cols)
        
        # Split data
        treated = data[data[treatment_col] == 1].copy()
        control = data[data[treatment_col] == 0].copy()
        
        treated['propensity_score'] = propensity_scores[data[treatment_col] == 1]
        control['propensity_score'] = propensity_scores[data[treatment_col] == 0]
        
        # Match treated units to control units
        matched_pairs = []
        
        for idx, treated_unit in treated.iterrows():
            # Calculate distance to all control units
            distances = np.abs(control['propensity_score'].values - treated_unit['propensity_score'])
            
            if method == 'caliper':
                # Filter by caliper
                eligible = distances <= caliper
                if eligible.sum() == 0:
                    continue
                distances[~eligible] = np.inf
            
            # Find nearest matches
            n_matches = min(ratio, eligible.sum() if method == 'caliper' else len(control))
            match_indices = np.argsort(distances)[:n_matches]
            
            for match_idx in match_indices:
                matched_pairs.append({
                    'treated_idx': idx,
                    'treated_outcome': treated_unit[outcome_col],
                    'control_idx': control.index[match_idx],
                    'control_outcome': control.iloc[match_idx][outcome_col],
                    'propensity_diff': distances[match_idx]
                })
        
        self.matched_pairs = pd.DataFrame(matched_pairs)
        
        if len(self.matched_pairs) == 0:
            logger.warning("No matches found")
            return {'ate': None, 'att': None, 'n_matches': 0}
        
        # Calculate treatment effects
        self.matched_pairs['individual_effect'] = (
            self.matched_pairs['treated_outcome'] - self.matched_pairs['control_outcome']
        )
        
        # Average Treatment Effect on the Treated (ATT)
        att = self.matched_pairs['individual_effect'].mean()
        
        # Average Treatment Effect (ATE) - using matched pairs
        ate = att  # In PSM, ATT is typically reported
        
        # Standard error
        se = self.matched_pairs['individual_effect'].std() / np.sqrt(len(self.matched_pairs))
        
        # T-statistic and p-value
        t_stat = att / se
        p_value = 2 * (1 - stats.t.cdf(abs(t_stat), df=len(self.matched_pairs) - 1))
        
        results = {
            'method': 'propensity_score_matching',
            'matching_method': method,
            'n_treated': len(treated),
            'n_control': len(control),
            'n_matches': len(self.matched_pairs),
            'att': float(att),
            'ate': float(ate),
            'standard_error': float(se),
            't_statistic': float(t_stat),
            'p_value': float(p_value),
            'is_significant': p_value < 0.05,
            'mean_propensity_diff': float(self.matched_pairs['propensity_diff'].mean())
        }
        
        logger.info(f"PSM completed. ATT: {att:.4f}, p-value: {p_value:.4f}")
        
        return results
    
    def inverse_probability_weighting(
        self,
        data: pd.DataFrame,
        treatment_col: str,
        outcome_col: str,
        feature_cols: List[str]
    ) -> Dict:
        """
        Perform Inverse Probability Weighting.
        
        Args:
            data: DataFrame with treatment, outcome, and features
            treatment_col: Column name for treatment indicator
            outcome_col: Column name for outcome
            feature_cols: List of feature column names
        
        Returns:
            Dictionary with IPW results and treatment effect estimate
        """
        logger.info("Performing Inverse Probability Weighting...")
        
        # Estimate propensity scores
        propensity_scores = self.estimate_propensity_scores(data, treatment_col, feature_cols)
        
        # Calculate IPW weights
        # For treated: weight = 1 / propensity_score
        # For control: weight = 1 / (1 - propensity_score)
        weights = np.where(
            data[treatment_col] == 1,
            1 / propensity_scores,
            1 / (1 - propensity_scores)
        )
        
        # Trim extreme weights (top and bottom 1%)
        weight_99 = np.percentile(weights, 99)
        weight_1 = np.percentile(weights, 1)
        weights = np.clip(weights, weight_1, weight_99)
        
        self.weights = weights
        
        # Calculate weighted outcomes
        weighted_treatment_outcome = np.sum(
            data[data[treatment_col] == 1][outcome_col].values * 
            weights[data[treatment_col] == 1]
        ) / np.sum(weights[data[treatment_col] == 1])
        
        weighted_control_outcome = np.sum(
            data[data[treatment_col] == 0][outcome_col].values * 
            weights[data[treatment_col] == 0]
        ) / np.sum(weights[data[treatment_col] == 0])
        
        # Average Treatment Effect (ATE)
        ate = weighted_treatment_outcome - weighted_control_outcome
        
        # Calculate standard error using bootstrap
        n_bootstrap = 100
        bootstrap_ates = []
        
        for _ in range(n_bootstrap):
            # Resample with replacement
            indices = np.random.choice(len(data), size=len(data), replace=True)
            sample = data.iloc[indices]
            sample_weights = weights[indices]
            
            sample_treatment_outcome = np.sum(
                sample[sample[treatment_col] == 1][outcome_col].values * 
                sample_weights[sample[treatment_col] == 1]
            ) / np.sum(sample_weights[sample[treatment_col] == 1])
            
            sample_control_outcome = np.sum(
                sample[sample[treatment_col] == 0][outcome_col].values * 
                sample_weights[sample[treatment_col] == 0]
            ) / np.sum(sample_weights[sample[treatment_col] == 0])
            
            bootstrap_ates.append(sample_treatment_outcome - sample_control_outcome)
        
        se = np.std(bootstrap_ates)
        
        # T-statistic and p-value
        t_stat = ate / se
        p_value = 2 * (1 - stats.t.cdf(abs(t_stat), df=len(data) - 1))
        
        results = {
            'method': 'inverse_probability_weighting',
            'n_treated': int((data[treatment_col] == 1).sum()),
            'n_control': int((data[treatment_col] == 0).sum()),
            'ate': float(ate),
            'weighted_treatment_outcome': float(weighted_treatment_outcome),
            'weighted_control_outcome': float(weighted_control_outcome),
            'standard_error': float(se),
            't_statistic': float(t_stat),
            'p_value': float(p_value),
            'is_significant': p_value < 0.05,
            'mean_weight': float(weights.mean()),
            'max_weight': float(weights.max())
        }
        
        logger.info(f"IPW completed. ATE: {ate:.4f}, p-value: {p_value:.4f}")
        
        return results
    
    def difference_in_differences(
        self,
        data: pd.DataFrame,
        treatment_col: str,
        outcome_col: str,
        time_col: str,
        treatment_time: str
    ) -> Dict:
        """
        Perform Difference-in-Differences analysis.
        
        Args:
            data: DataFrame with treatment, outcome, and time
            treatment_col: Column name for treatment indicator
            outcome_col: Column name for outcome
            time_col: Column name for time period
            treatment_time: Time when treatment was applied
        
        Returns:
            Dictionary with DiD results
        """
        logger.info("Performing Difference-in-Differences analysis...")
        
        # Ensure time column is datetime
        data[time_col] = pd.to_datetime(data[time_col])
        treatment_time = pd.to_datetime(treatment_time)
        
        # Create post-treatment indicator
        data['post'] = (data[time_col] > treatment_time).astype(int)
        
        # Create interaction term (treatment * post)
        data['treatment_post'] = data[treatment_col] * data['post']
        
        # Calculate means by group and time
        group_means = data.groupby([treatment_col, 'post'])[outcome_col].mean().reset_index()
        
        # Pre-treatment means
        pre_treatment = group_means[(group_means['post'] == 0) & (group_means[treatment_col] == 1)][outcome_col].values[0]
        pre_control = group_means[(group_means['post'] == 0) & (group_means[treatment_col] == 0)][outcome_col].values[0]
        
        # Post-treatment means
        post_treatment = group_means[(group_means['post'] == 1) & (group_means[treatment_col] == 1)][outcome_col].values[0]
        post_control = group_means[(group_means['post'] == 1) & (group_means[treatment_col] == 0)][outcome_col].values[0]
        
        # Calculate DiD estimator
        # DiD = (post_treatment - pre_treatment) - (post_control - pre_control)
        treatment_diff = post_treatment - pre_treatment
        control_diff = post_control - pre_control
        did = treatment_diff - control_diff
        
        # Linear regression for standard errors
        import statsmodels.formula.api as smf
        model = smf.ols(f"{outcome_col} ~ {treatment_col} + post + treatment_post", data=data).fit()
        
        # Get DiD coefficient and its statistics
        did_coef = model.params['treatment_post']
        did_se = model.bse['treatment_post']
        did_p_value = model.pvalues['treatment_post']
        
        results = {
            'method': 'difference_in_differences',
            'pre_treatment_mean': float(pre_treatment),
            'pre_control_mean': float(pre_control),
            'post_treatment_mean': float(post_treatment),
            'post_control_mean': float(post_control),
            'treatment_diff': float(treatment_diff),
            'control_diff': float(control_diff),
            'did_estimator': float(did),
            'did_coef': float(did_coef),
            'did_se': float(did_se),
            'did_p_value': float(did_p_value),
            'is_significant': did_p_value < 0.05,
            'parallel_trend_assumption': 'Not tested - requires pre-treatment periods'
        }
        
        logger.info(f"DiD completed. DiD: {did:.4f}, p-value: {did_p_value:.4f}")
        
        return results
    
    def compare_methods(
        self,
        data: pd.DataFrame,
        treatment_col: str,
        outcome_col: str,
        feature_cols: List[str],
        time_col: str = None,
        treatment_time: str = None
    ) -> pd.DataFrame:
        """
        Compare causal inference methods.
        
        Args:
            data: DataFrame with treatment, outcome, and features
            treatment_col: Column name for treatment indicator
            outcome_col: Column name for outcome
            feature_cols: List of feature column names
            time_col: Column name for time (for DiD)
            treatment_time: Treatment time (for DiD)
        
        Returns:
            DataFrame with comparison of methods
        """
        results = []
        
        # PSM
        try:
            psm_result = self.propensity_score_matching(
                data, treatment_col, outcome_col, feature_cols
            )
            results.append({
                'method': 'PSM',
                'effect_estimate': psm_result.get('att'),
                'standard_error': psm_result.get('standard_error'),
                'p_value': psm_result.get('p_value'),
                'is_significant': psm_result.get('is_significant')
            })
        except Exception as e:
            logger.error(f"PSM failed: {e}")
        
        # IPW
        try:
            ipw_result = self.inverse_probability_weighting(
                data, treatment_col, outcome_col, feature_cols
            )
            results.append({
                'method': 'IPW',
                'effect_estimate': ipw_result.get('ate'),
                'standard_error': ipw_result.get('standard_error'),
                'p_value': ipw_result.get('p_value'),
                'is_significant': ipw_result.get('is_significant')
            })
        except Exception as e:
            logger.error(f"IPW failed: {e}")
        
        # DiD (if time data available)
        if time_col and treatment_time:
            try:
                did_result = self.difference_in_differences(
                    data, treatment_col, outcome_col, time_col, treatment_time
                )
                results.append({
                    'method': 'DiD',
                    'effect_estimate': did_result.get('did_estimator'),
                    'standard_error': did_result.get('did_se'),
                    'p_value': did_result.get('did_p_value'),
                    'is_significant': did_result.get('is_significant')
                })
            except Exception as e:
                logger.error(f"DiD failed: {e}")
        
        return pd.DataFrame(results)


def run_causal_inference_pipeline(
    data: pd.DataFrame,
    treatment_col: str,
    outcome_col: str,
    feature_cols: List[str],
    methods: List[str] = None,
    time_col: str = None,
    treatment_time: str = None
) -> Tuple[CausalInferenceEngine, Dict]:
    """
    Convenience function to run causal inference pipeline.
    
    Args:
        data: DataFrame with treatment, outcome, and features
        treatment_col: Column name for treatment indicator
        outcome_col: Column name for outcome
        feature_cols: List of feature column names
        methods: List of methods to run ('psm', 'ipw', 'did')
        time_col: Column name for time (for DiD)
        treatment_time: Treatment time (for DiD)
    
    Returns:
        Tuple of (engine, results dictionary)
    """
    if methods is None:
        methods = ['psm', 'ipw']
    
    engine = CausalInferenceEngine()
    results = {}
    
    if 'psm' in methods:
        results['psm'] = engine.propensity_score_matching(
            data, treatment_col, outcome_col, feature_cols
        )
    
    if 'ipw' in methods:
        results['ipw'] = engine.inverse_probability_weighting(
            data, treatment_col, outcome_col, feature_cols
        )
    
    if 'did' in methods and time_col and treatment_time:
        results['did'] = engine.difference_in_differences(
            data, treatment_col, outcome_col, time_col, treatment_time
        )
    
    return engine, results
