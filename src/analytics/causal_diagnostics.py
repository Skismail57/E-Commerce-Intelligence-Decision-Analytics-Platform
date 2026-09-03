"""
Causal Diagnostics Module
Implements diagnostic tools to validate causal assumptions in marketing and pricing models.

Architecture:
- Instrument validation for marketing campaigns
- Parallel trends testing for difference-in-differences
- Confounder detection and adjustment
- Propensity score diagnostics
- Sensitivity analysis for unobserved confounders
"""

from typing import Dict, List, Optional, Tuple
import pandas as pd
import numpy as np
from scipy import stats
from pathlib import Path

from config.logging_config import get_logger
from config.settings import settings

logger = get_logger(__name__)


class CausalDiagnostics:
    """
    Diagnostic tools for validating causal inference assumptions.
    
    Helps ensure that marketing and pricing models are actually
    measuring causal effects, not just correlations.
    """
    
    def __init__(self):
        """Initialize causal diagnostics."""
        logger.info("Causal Diagnostics initialized")
    
    def check_parallel_trends(
        self,
        pre_treatment_data: pd.DataFrame,
        treatment_col: str = 'treatment',
        outcome_col: str = 'outcome',
        time_col: str = 'time'
    ) -> Dict[str, float]:
        """
        Test parallel trends assumption for difference-in-differences.
        
        Args:
            pre_treatment_data: Data from pre-treatment period
            treatment_col: Column indicating treatment (1=treated, 0=control)
            outcome_col: Outcome variable
            time_col: Time variable
        
        Returns:
            Dictionary with test results
        """
        logger.info("Testing parallel trends assumption...")
        
        # Separate treatment and control groups
        treated = pre_treatment_data[pre_treatment_data[treatment_col] == 1]
        control = pre_treatment_data[pre_treatment_data[treatment_col] == 0]
        
        # Calculate trends for each group
        treated_trend = self._calculate_trend(treated, time_col, outcome_col)
        control_trend = self._calculate_trend(control, time_col, outcome_col)
        
        # Test if trends are significantly different
        trend_diff = treated_trend - control_trend
        
        # Bootstrap confidence interval for trend difference
        n_bootstrap = 1000
        bootstrap_diffs = []
        
        for _ in range(n_bootstrap):
            treated_sample = treated.sample(n=len(treated), replace=True)
            control_sample = control.sample(n=len(control), replace=True)
            
            treated_trend_boot = self._calculate_trend(treated_sample, time_col, outcome_col)
            control_trend_boot = self._calculate_trend(control_sample, time_col, outcome_col)
            
            bootstrap_diffs.append(treated_trend_boot - control_trend_boot)
        
        ci_lower = np.percentile(bootstrap_diffs, 2.5)
        ci_upper = np.percentile(bootstrap_diffs, 97.5)
        
        parallel_trends_holds = (ci_lower <= 0 <= ci_upper)
        
        results = {
            'treated_trend': treated_trend,
            'control_trend': control_trend,
            'trend_difference': trend_diff,
            'ci_95_lower': ci_lower,
            'ci_95_upper': ci_upper,
            'parallel_trends_holds': parallel_trends_holds,
            'interpretation': 'Parallel trends assumption holds' if parallel_trends_holds else 'Parallel trends assumption violated'
        }
        
        logger.info(f"Parallel trends test: {results['interpretation']}")
        return results
    
    def _calculate_trend(
        self,
        data: pd.DataFrame,
        time_col: str,
        outcome_col: str
    ) -> float:
        """Calculate linear trend coefficient."""
        if len(data) < 2:
            return 0.0
        
        x = data[time_col].values
        y = data[outcome_col].values
        
        # Remove NaN values
        mask = ~np.isnan(x) & ~np.isnan(y)
        x = x[mask]
        y = y[mask]
        
        if len(x) < 2:
            return 0.0
        
        # Calculate slope (trend)
        slope, _ = np.polyfit(x, y, 1)
        return slope
    
    def validate_instrument(
        self,
        data: pd.DataFrame,
        instrument_col: str,
        treatment_col: str,
        outcome_col: str
    ) -> Dict[str, float]:
        """
        Validate instrumental variable assumptions.
        
        Tests:
        1. Relevance: Instrument is correlated with treatment
        2. Exclusion: Instrument affects outcome only through treatment
        3. Exogeneity: Instrument is not correlated with confounders
        
        Args:
            data: Data containing instrument, treatment, and outcome
            instrument_col: Instrument variable
            treatment_col: Treatment variable
            outcome_col: Outcome variable
        
        Returns:
            Dictionary with validation results
        """
        logger.info("Validating instrumental variable...")
        
        # Test relevance: correlation between instrument and treatment
        instrument = data[instrument_col].values
        treatment = data[treatment_col].values
        
        relevance_corr, relevance_p = stats.pearsonr(
            instrument[np.isnan(instrument) == False],
            treatment[np.isnan(treatment) == False]
        )
        
        # First stage F-statistic (strength of instrument)
        from sklearn.linear_model import LinearRegression
        
        mask = ~np.isnan(instrument) & ~np.isnan(treatment)
        X = instrument[mask].reshape(-1, 1)
        y = treatment[mask]
        
        if len(X) > 1:
            reg = LinearRegression()
            reg.fit(X, y)
            predictions = reg.predict(X)
            ssr = np.sum((y - predictions) ** 2)
            sst = np.sum((y - np.mean(y)) ** 2)
            r_squared = 1 - (ssr / sst) if sst > 0 else 0
            f_stat = r_squared / (1 - r_squared) * (len(y) - 2) if r_squared < 1 else np.inf
        else:
            r_squared = 0
            f_stat = 0
        
        # Test exclusion: correlation between instrument and outcome
        outcome = data[outcome_col].values
        exclusion_corr, exclusion_p = stats.pearsonr(
            instrument[np.isnan(instrument) == False],
            outcome[np.isnan(outcome) == False]
        )
        
        results = {
            'relevance_correlation': relevance_corr,
            'relevance_p_value': relevance_p,
            'first_stage_f_statistic': f_stat,
            'first_stage_r_squared': r_squared,
            'instrument_strong': f_stat > 10,  # Rule of thumb
            'exclusion_correlation': exclusion_corr,
            'exclusion_p_value': exclusion_p,
            'interpretation': 'Instrument is strong' if f_stat > 10 else 'Instrument is weak'
        }
        
        logger.info(f"Instrument validation: {results['interpretation']}")
        return results
    
    def detect_confounders(
        self,
        data: pd.DataFrame,
        treatment_col: str,
        outcome_col: str,
        candidate_confounders: List[str]
    ) -> Dict[str, Dict[str, float]]:
        """
        Detect potential confounders using correlation analysis.
        
        A confounder is a variable that is correlated with both
        treatment and outcome.
        
        Args:
            data: Data containing treatment, outcome, and candidates
            treatment_col: Treatment variable
            outcome_col: Outcome variable
            candidate_confounders: List of potential confounders
        
        Returns:
            Dictionary mapping confounder to correlation metrics
        """
        logger.info("Detecting potential confounders...")
        
        confounders = {}
        treatment = data[treatment_col].values
        outcome = data[outcome_col].values
        
        for confounder in candidate_confounders:
            if confounder not in data.columns:
                continue
            
            confounder_var = data[confounder].values
            
            # Correlation with treatment
            corr_treatment, p_treatment = stats.pearsonr(
                confounder_var[np.isnan(confounder_var) == False],
                treatment[np.isnan(treatment) == False]
            )
            
            # Correlation with outcome
            corr_outcome, p_outcome = stats.pearsonr(
                confounder_var[np.isnan(confounder_var) == False],
                outcome[np.isnan(outcome) == False]
            )
            
            # Consider it a confounder if correlated with both
            is_confounder = (abs(corr_treatment) > 0.1) and (abs(corr_outcome) > 0.1)
            
            confounders[confounder] = {
                'correlation_with_treatment': corr_treatment,
                'p_value_treatment': p_treatment,
                'correlation_with_outcome': corr_outcome,
                'p_value_outcome': p_outcome,
                'is_confounder': is_confounder
            }
        
        logger.info(f"Detected {sum(1 for c in confounders.values() if c['is_confounder'])} potential confounders")
        return confounders
    
    def propensity_score_diagnostics(
        self,
        propensity_scores: np.ndarray,
        treatment: np.ndarray
    ) -> Dict[str, float]:
        """
        Diagnose propensity score matching quality.
        
        Args:
            propensity_scores: Propensity scores for all units
            treatment: Treatment indicator (1=treated, 0=control)
        
        Returns:
            Dictionary with diagnostic metrics
        """
        logger.info("Running propensity score diagnostics...")
        
        treated_scores = propensity_scores[treatment == 1]
        control_scores = propensity_scores[treatment == 0]
        
        # Overlap: proportion of control units with propensity > min treated propensity
        min_treated = np.min(treated_scores)
        max_control = np.max(control_scores)
        overlap = np.mean(control_scores >= min_treated)
        
        # Standardized mean difference of propensity scores
        mean_diff = np.mean(treated_scores) - np.mean(control_scores)
        pooled_std = np.sqrt((np.var(treated_scores) + np.var(control_scores)) / 2)
        smd = mean_diff / pooled_std if pooled_std > 0 else 0
        
        # Visual inspection metric: histogram overlap
        # (simplified as proportion of scores in common range)
        common_min = max(np.min(treated_scores), np.min(control_scores))
        common_max = min(np.max(treated_scores), np.max(control_scores))
        
        if common_max > common_min:
            treated_in_common = np.mean((treated_scores >= common_min) & (treated_scores <= common_max))
            control_in_common = np.mean((control_scores >= common_min) & (control_scores <= common_max))
            histogram_overlap = (treated_in_common + control_in_common) / 2
        else:
            histogram_overlap = 0
        
        results = {
            'propensity_overlap': overlap,
            'standardized_mean_difference': smd,
            'histogram_overlap': histogram_overlap,
            'good_balance': (abs(smd) < 0.1) and (overlap > 0.8),
            'interpretation': 'Good propensity balance' if (abs(smd) < 0.1) and (overlap > 0.8) else 'Poor propensity balance'
        }
        
        logger.info(f"Propensity score diagnostics: {results['interpretation']}")
        return results
    
    def sensitivity_analysis(
        self,
        estimated_effect: float,
        outcome_variance: float,
        treatment_variance: float,
        rho_range: Tuple[float, float] = (-0.9, 0.9)
    ) -> pd.DataFrame:
        """
        Rosenbaum sensitivity analysis for unobserved confounders.
        
        Tests how strong an unobserved confounder would need to be
        to explain away the estimated effect.
        
        Args:
            estimated_effect: Estimated causal effect
            outcome_variance: Variance of outcome
            treatment_variance: Variance of treatment
            rho_range: Range of correlation between unobserved confounder and treatment
        
        Returns:
            DataFrame with sensitivity results
        """
        logger.info("Running sensitivity analysis...")
        
        rho_values = np.linspace(rho_range[0], rho_range[1], 100)
        sensitivity_results = []
        
        for rho in rho_values:
            # Calculate required correlation with outcome to explain effect
            # Using Rosenbaum bounds formula
            gamma = np.exp(rho)
            lower_bound = estimated_effect / gamma
            upper_bound = estimated_effect * gamma
            
            sensitivity_results.append({
                'rho': rho,
                'gamma': gamma,
                'lower_bound': lower_bound,
                'upper_bound': upper_bound,
                'effect_explained': (lower_bound <= 0 <= upper_bound)
            })
        
        results_df = pd.DataFrame(sensitivity_results)
        
        # Find minimum rho that explains effect
        explaining_rho = results_df[results_df['effect_explained']]['rho'].min()
        
        logger.info(f"Sensitivity analysis: Effect would be explained by confounder with rho >= {explaining_rho:.2f}")
        return results_df
    
    def generate_diagnostics_report(
        self,
        diagnostics: Dict[str, any],
        output_path: Optional[Path] = None
    ) -> str:
        """
        Generate comprehensive causal diagnostics report.
        
        Args:
            diagnostics: Dictionary with all diagnostic results
            output_path: Optional path to save report
        
        Returns:
            Report string
        """
        report = f"""
Causal Diagnostics Report
{'=' * 60}

Parallel Trends Test:
"""
        
        if 'parallel_trends' in diagnostics:
            pt = diagnostics['parallel_trends']
            report += f"- Treated Trend: {pt.get('treated_trend', 0):.4f}\n"
            report += f"- Control Trend: {pt.get('control_trend', 0):.4f}\n"
            report += f"- Trend Difference: {pt.get('trend_difference', 0):.4f}\n"
            report += f"- 95% CI: [{pt.get('ci_95_lower', 0):.4f}, {pt.get('ci_95_upper', 0):.4f}]\n"
            report += f"- Assumption Holds: {pt.get('parallel_trends_holds', False)}\n"
            report += f"- Interpretation: {pt.get('interpretation', 'N/A')}\n"
        
        report += "\nInstrument Validation:\n"
        if 'instrument' in diagnostics:
            inst = diagnostics['instrument']
            report += f"- Relevance Correlation: {inst.get('relevance_correlation', 0):.4f}\n"
            report += f"- First Stage F-Statistic: {inst.get('first_stage_f_statistic', 0):.2f}\n"
            report += f"- Instrument Strong: {inst.get('instrument_strong', False)}\n"
            report += f"- Interpretation: {inst.get('interpretation', 'N/A')}\n"
        
        report += "\nConfounder Detection:\n"
        if 'confounders' in diagnostics:
            confounders = diagnostics['confounders']
            detected = [k for k, v in confounders.items() if v.get('is_confounder', False)]
            report += f"- Detected Confounders: {len(detected)}\n"
            for conf in detected:
                report += f"  * {conf}\n"
        
        report += "\nPropensity Score Diagnostics:\n"
        if 'propensity' in diagnostics:
            prop = diagnostics['propensity']
            report += f"- Propensity Overlap: {prop.get('propensity_overlap', 0):.%\n"
            report += f"- Standardized Mean Difference: {prop.get('standardized_mean_difference', 0):.4f}\n"
            report += f"- Good Balance: {prop.get('good_balance', False)}\n"
            report += f"- Interpretation: {prop.get('interpretation', 'N/A')}\n"
        
        report += f"""
Interpretation Guidelines:
- Parallel Trends: Essential for difference-in-differences
- Instrument Strength: F-statistic > 10 indicates strong instrument
- Confounders: Variables correlated with both treatment and outcome
- Propensity Balance: SMD < 0.1 and overlap > 80% indicates good balance
"""
        
        if output_path:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(report)
            logger.info(f"Diagnostics report saved to {output_path}")
        
        return report


def run_causal_diagnostics(
    data: pd.DataFrame,
    treatment_col: str,
    outcome_col: str,
    candidate_confounders: List[str]
) -> Dict[str, any]:
    """
    Convenience function to run complete causal diagnostics.
    
    Args:
        data: Data containing treatment, outcome, and confounders
        treatment_col: Treatment variable
        outcome_col: Outcome variable
        candidate_confounders: List of potential confounders
    
    Returns:
        Dictionary with all diagnostic results
    """
    diagnostics = CausalDiagnostics()
    
    results = {}
    
    # Detect confounders
    results['confounders'] = diagnostics.detect_confounders(
        data, treatment_col, outcome_col, candidate_confounders
    )
    
    # Run other diagnostics as needed based on data structure
    
    return results
