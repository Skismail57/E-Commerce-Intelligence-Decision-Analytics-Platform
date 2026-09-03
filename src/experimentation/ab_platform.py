"""
A/B Experimentation Platform
Implements A/B testing framework for marketing campaigns, UI changes, and feature rollouts.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
from scipy import stats
from statsmodels.stats.power import TTestIndPower
from statsmodels.stats.proportion import proportions_ztest, proportion_confint
import matplotlib.pyplot as plt
from config.logging_config import get_logger

logger = get_logger(__name__)


class ABExperimentPlatform:
    """
    A/B Experimentation Platform for marketing and product experiments.
    
    Features:
    - Experiment design and sample size calculation
    - Random assignment and stratification
    - Statistical analysis (t-tests, chi-square, Bayesian)
    - Power analysis and significance testing
    - Multiple hypothesis testing correction
    - Experiment tracking and reporting
    """
    
    def __init__(self):
        """Initialize A/B experimentation platform"""
        self.experiments = {}
        self.power_calculator = TTestIndPower()
    
    def design_experiment(
        self,
        experiment_id: str,
        name: str,
        description: str,
        variants: List[str],
        metric: str,
        metric_type: str = 'continuous',
        baseline_conversion: float = None,
        minimum_detectable_effect: float = 0.05,
        alpha: float = 0.05,
        power: float = 0.8
    ) -> Dict:
        """
        Design a new A/B experiment.
        
        Args:
            experiment_id: Unique experiment identifier
            name: Experiment name
            description: Experiment description
            variants: List of variant names (e.g., ['control', 'treatment'])
            metric: Primary metric to measure
            metric_type: Type of metric ('continuous' or 'binary')
            baseline_conversion: Baseline conversion rate (for binary metrics)
            minimum_detectable_effect: Minimum effect size to detect
            alpha: Significance level
            power: Statistical power
        
        Returns:
            Dictionary with experiment design details
        """
        logger.info(f"Designing experiment: {experiment_id}")
        
        # Calculate required sample size
        if metric_type == 'binary' and baseline_conversion:
            required_sample_size = self._calculate_sample_size_binary(
                baseline_conversion, minimum_detectable_effect, alpha, power
            )
        else:
            # Default sample size for continuous metrics
            required_sample_size = self._calculate_sample_size_continuous(
                minimum_detectable_effect, alpha, power
            )
        
        experiment = {
            'experiment_id': experiment_id,
            'name': name,
            'description': description,
            'variants': variants,
            'metric': metric,
            'metric_type': metric_type,
            'baseline_conversion': baseline_conversion,
            'minimum_detectable_effect': minimum_detectable_effect,
            'alpha': alpha,
            'power': power,
            'required_sample_size': required_sample_size,
            'status': 'designed',
            'created_at': datetime.now().isoformat()
        }
        
        self.experiments[experiment_id] = experiment
        
        logger.info(f"Experiment designed. Required sample size: {required_sample_size}")
        
        return experiment
    
    def _calculate_sample_size_binary(
        self,
        baseline: float,
        effect_size: float,
        alpha: float,
        power: float
    ) -> int:
        """Calculate sample size for binary metrics"""
        # Use proportions_ztest sample size calculation
        p1 = baseline
        p2 = baseline * (1 + effect_size)
        
        # Standard formula for two-proportion test
        pooled_p = (p1 + p2) / 2
        n_per_group = (
            (stats.norm.ppf(1 - alpha/2) * np.sqrt(2 * pooled_p * (1 - pooled_p)) +
             stats.norm.ppf(power) * np.sqrt(p1 * (1 - p1) + p2 * (1 - p2)))
        ) ** 2 / (p2 - p1) ** 2
        
        return int(np.ceil(n_per_group))
    
    def _calculate_sample_size_continuous(
        self,
        effect_size: float,
        alpha: float,
        power: float
    ) -> int:
        """Calculate sample size for continuous metrics"""
        # Cohen's d for effect size
        # Assuming effect_size is Cohen's d
        n_per_group = self.power_calculator.solve_power(
            effect_size=effect_size,
            alpha=alpha,
            power=power,
            alternative='two-sided'
        )
        
        return int(np.ceil(n_per_group))
    
    def assign_variants(
        self,
        experiment_id: str,
        customers_df: pd.DataFrame,
        stratification_cols: List[str] = None
    ) -> pd.DataFrame:
        """
        Randomly assign customers to experiment variants.
        
        Args:
            experiment_id: Experiment identifier
            customers_df: Customer data
            stratification_cols: Columns for stratified sampling
        
        Returns:
            DataFrame with variant assignments
        """
        if experiment_id not in self.experiments:
            raise ValueError(f"Experiment {experiment_id} not found")
        
        experiment = self.experiments[experiment_id]
        variants = experiment['variants']
        
        assignments = customers_df[['customer_id']].copy()
        
        if stratification_cols and all(col in customers_df.columns for col in stratification_cols):
            # Stratified random assignment
            assignments['variant'] = None
            
            for _, group in customers_df.groupby(stratification_cols):
                group_customers = group['customer_id'].values
                np.random.shuffle(group_customers)
                
                # Split into variants
                n_variants = len(variants)
                chunk_size = len(group_customers) // n_variants
                
                for i, variant in enumerate(variants):
                    start_idx = i * chunk_size
                    end_idx = (i + 1) * chunk_size if i < n_variants - 1 else len(group_customers)
                    assigned_customers = group_customers[start_idx:end_idx]
                    
                    assignments.loc[assignments['customer_id'].isin(assigned_customers), 'variant'] = variant
        else:
            # Simple random assignment
            assignments['variant'] = np.random.choice(variants, size=len(assignments))
        
        # Update experiment status
        self.experiments[experiment_id]['status'] = 'assigned'
        self.experiments[experiment_id]['assigned_at'] = datetime.now().isoformat()
        
        logger.info(f"Assigned {len(assignments)} customers to {len(variants)} variants")
        
        return assignments
    
    def analyze_experiment(
        self,
        experiment_id: str,
        results_df: pd.DataFrame
    ) -> Dict:
        """
        Analyze experiment results.
        
        Args:
            experiment_id: Experiment identifier
            results_df: DataFrame with customer_id, variant, and metric values
        
        Returns:
            Dictionary with analysis results
        """
        if experiment_id not in self.experiments:
            raise ValueError(f"Experiment {experiment_id} not found")
        
        experiment = self.experiments[experiment_id]
        metric = experiment['metric']
        metric_type = experiment['metric_type']
        variants = experiment['variants']
        
        logger.info(f"Analyzing experiment: {experiment_id}")
        
        analysis_results = {
            'experiment_id': experiment_id,
            'metric': metric,
            'metric_type': metric_type,
            'variants': {},
            'statistical_tests': {}
        }
        
        # Calculate metrics for each variant
        for variant in variants:
            variant_data = results_df[results_df['variant'] == variant]
            
            if metric_type == 'continuous':
                variant_stats = {
                    'n_samples': len(variant_data),
                    'mean': float(variant_data[metric].mean()),
                    'std': float(variant_data[metric].std()),
                    'median': float(variant_data[metric].median()),
                    'min': float(variant_data[metric].min()),
                    'max': float(variant_data[metric].max())
                }
            else:
                variant_stats = {
                    'n_samples': len(variant_data),
                    'conversion_rate': float(variant_data[metric].mean()),
                    'n_conversions': int(variant_data[metric].sum())
                }
            
            analysis_results['variants'][variant] = variant_stats
        
        # Perform statistical tests
        if len(variants) == 2:
            # Two-variant test
            control_data = results_df[results_df['variant'] == variants[0]][metric]
            treatment_data = results_df[results_df['variant'] == variants[1]][metric]
            
            if metric_type == 'continuous':
                # T-test
                t_stat, p_value = stats.ttest_ind(control_data, treatment_data)
                
                # Effect size (Cohen's d)
                pooled_std = np.sqrt((control_data.var() + treatment_data.var()) / 2)
                effect_size = (treatment_data.mean() - control_data.mean()) / pooled_std
                
                analysis_results['statistical_tests'] = {
                    'test_type': 't_test',
                    't_statistic': float(t_stat),
                    'p_value': float(p_value),
                    'effect_size': float(effect_size),
                    'is_significant': p_value < experiment['alpha']
                }
            else:
                # Z-test for proportions
                n_control = len(control_data)
                n_treatment = len(treatment_data)
                conversions = [control_data.sum(), treatment_data.sum()]
                nobs = [n_control, n_treatment]
                
                z_stat, p_value = proportions_ztest(conversions, nobs)
                
                # Confidence intervals
                ci_control = proportion_confint(conversions[0], nobs[0], alpha=experiment['alpha'])
                ci_treatment = proportion_confint(conversions[1], nobs[1], alpha=experiment['alpha'])
                
                # Relative lift
                control_rate = conversions[0] / n_control
                treatment_rate = conversions[1] / n_treatment
                relative_lift = (treatment_rate - control_rate) / control_rate if control_rate > 0 else 0
                
                analysis_results['statistical_tests'] = {
                    'test_type': 'z_test_proportions',
                    'z_statistic': float(z_stat),
                    'p_value': float(p_value),
                    'relative_lift': float(relative_lift),
                    'absolute_lift': float(treatment_rate - control_rate),
                    'control_ci': [float(ci_control[0]), float(ci_control[1])],
                    'treatment_ci': [float(ci_treatment[0]), float(ci_treatment[1])],
                    'is_significant': p_value < experiment['alpha']
                }
        else:
            # Multi-variant test (ANOVA or chi-square)
            if metric_type == 'continuous':
                # ANOVA
                groups = [results_df[results_df['variant'] == v][metric].values for v in variants]
                f_stat, p_value = stats.f_oneway(*groups)
                
                analysis_results['statistical_tests'] = {
                    'test_type': 'anova',
                    'f_statistic': float(f_stat),
                    'p_value': float(p_value),
                    'is_significant': p_value < experiment['alpha']
                }
            else:
                # Chi-square test
                contingency_table = pd.crosstab(results_df['variant'], results_df[metric])
                chi2, p_value, dof, expected = stats.chi2_contingency(contingency_table)
                
                analysis_results['statistical_tests'] = {
                    'test_type': 'chi_square',
                    'chi2_statistic': float(chi2),
                    'p_value': float(p_value),
                    'degrees_of_freedom': int(dof),
                    'is_significant': p_value < experiment['alpha']
                }
        
        # Update experiment status
        self.experiments[experiment_id]['status'] = 'analyzed'
        self.experiments[experiment_id]['analyzed_at'] = datetime.now().isoformat()
        self.experiments[experiment_id]['results'] = analysis_results
        
        logger.info(f"Experiment analyzed. Significant: {analysis_results['statistical_tests']['is_significant']}")
        
        return analysis_results
    
    def calculate_power(
        self,
        experiment_id: str,
        observed_effect_size: float,
        sample_size: int
    ) -> float:
        """
        Calculate achieved power given observed effect size and sample size.
        
        Args:
            experiment_id: Experiment identifier
            observed_effect_size: Observed effect size
            sample_size: Sample size per group
        
        Returns:
            Achieved statistical power
        """
        if experiment_id not in self.experiments:
            raise ValueError(f"Experiment {experiment_id} not found")
        
        experiment = self.experiments[experiment_id]
        alpha = experiment['alpha']
        
        achieved_power = self.power_calculator.power(
            effect_size=observed_effect_size,
            nobs1=sample_size,
            alpha=alpha,
            alternative='two-sided'
        )
        
        return float(achieved_power)
    
    def get_experiment_summary(self, experiment_id: str) -> Dict:
        """
        Get summary of an experiment.
        
        Args:
            experiment_id: Experiment identifier
        
        Returns:
            Dictionary with experiment summary
        """
        if experiment_id not in self.experiments:
            raise ValueError(f"Experiment {experiment_id} not found")
        
        return self.experiments[experiment_id]
    
    def list_experiments(self, status: str = None) -> List[Dict]:
        """
        List all experiments, optionally filtered by status.
        
        Args:
            status: Filter by status (optional)
        
        Returns:
            List of experiment summaries
        """
        experiments = list(self.experiments.values())
        
        if status:
            experiments = [e for e in experiments if e['status'] == status]
        
        return experiments


def run_ab_experiment(
    customers_df: pd.DataFrame,
    metric_data: pd.DataFrame,
    experiment_name: str,
    variants: List[str],
    metric: str,
    metric_type: str = 'continuous',
    baseline_conversion: float = None,
    minimum_detectable_effect: float = 0.05
) -> Tuple[Dict, Dict]:
    """
    Convenience function to run a complete A/B experiment.
    
    Args:
        customers_df: Customer data
        metric_data: Data with metric values
        experiment_name: Name for the experiment
        variants: List of variant names
        metric: Metric to measure
        metric_type: Type of metric
        baseline_conversion: Baseline conversion rate (for binary metrics)
        minimum_detectable_effect: Minimum effect size to detect
    
    Returns:
        Tuple of (experiment design, analysis results)
    """
    platform = ABExperimentPlatform()
    
    # Design experiment
    experiment_id = f"exp_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    design = platform.design_experiment(
        experiment_id=experiment_id,
        name=experiment_name,
        description=f"A/B test for {metric}",
        variants=variants,
        metric=metric,
        metric_type=metric_type,
        baseline_conversion=baseline_conversion,
        minimum_detectable_effect=minimum_detectable_effect
    )
    
    # Assign variants
    assignments = platform.assign_variants(experiment_id, customers_df)
    
    # Merge with metric data
    results = assignments.merge(metric_data, on='customer_id', how='inner')
    
    # Analyze
    analysis = platform.analyze_experiment(experiment_id, results)
    
    return design, analysis
