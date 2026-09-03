"""
Uplift Ground-Truth Evaluation Module
Implements evaluation for uplift modeling using randomized experiment data.

Architecture:
- Requires randomized treatment assignment (A/B test data)
- Estimates true causal effect from ground truth
- Compares predicted uplift vs actual uplift
- Metrics: AUUC, Qini coefficient, uplift calibration
"""

from typing import Dict, List, Optional, Tuple
import pandas as pd
import numpy as np
from scipy import stats
from pathlib import Path

from config.logging_config import get_logger
from config.settings import settings

logger = get_logger(__name__)


class UpliftEvaluator:
    """
    Evaluator for uplift models using ground-truth experimental data.
    
    Requires data from randomized controlled trials (A/B tests)
    to validate that uplift predictions match actual causal effects.
    """
    
    def __init__(self):
        """Initialize uplift evaluator."""
        logger.info("Uplift Evaluator initialized")
    
    def calculate_true_uplift(
        self,
        data: pd.DataFrame,
        treatment_col: str = 'treatment',
        outcome_col: str = 'outcome',
        group_col: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Calculate true uplift from experimental data.
        
        True uplift = E[outcome | treatment=1] - E[outcome | treatment=0]
        
        Args:
            data: Experimental data with treatment and outcome
            treatment_col: Treatment indicator (1=treated, 0=control)
            outcome_col: Outcome variable
            group_col: Optional grouping variable for subgroup analysis
        
        Returns:
            DataFrame with true uplift estimates
        """
        logger.info("Calculating true uplift from experimental data...")
        
        if group_col is None:
            # Overall uplift
            treated = data[data[treatment_col] == 1][outcome_col]
            control = data[data[treatment_col] == 0][outcome_col]
            
            true_uplift = treated.mean() - control.mean()
            
            # Calculate standard error
            treated_var = treated.var(ddof=1) / len(treated)
            control_var = control.var(ddof=1) / len(control)
            se = np.sqrt(treated_var + control_var)
            
            # Calculate p-value
            t_stat = true_uplift / se if se > 0 else 0
            p_value = 2 * (1 - stats.norm.cdf(abs(t_stat)))
            
            result = pd.DataFrame({
                'group': 'overall',
                'true_uplift': true_uplift,
                'standard_error': se,
                't_statistic': t_stat,
                'p_value': p_value,
                't-treated': len(treated),
                'n-control': len(control)
            }, index=[0])
        else:
            # Uplift by group
            results = []
            for group in data[group_col].unique():
                group_data = data[data[group_col] == group]
                treated = group_data[group_data[treatment_col] == 1][outcome_col]
                control = group_data[group_data[treatment_col] == 0][outcome_col]
                
                if len(treated) == 0 or len(control) == 0:
                    continue
                
                true_uplift = treated.mean() - control.mean()
                
                treated_var = treated.var(ddof=1) / len(treated)
                control_var = control.var(ddof=1) / len(control)
                se = np.sqrt(treated_var + control_var)
                
                t_stat = true_uplift / se if se > 0 else 0
                p_value = 2 * (1 - stats.norm.cdf(abs(t_stat)))
                
                results.append({
                    'group': group,
                    'true_uplift': true_uplift,
                    'standard_error': se,
                    't_statistic': t_stat,
                    'p_value': p_value,
                    'n-treated': len(treated),
                    'n-control': len(control)
                })
            
            result = pd.DataFrame(results)
        
        logger.info(f"True uplift calculated: {result['true_uplift'].mean():.4f}")
        return result
    
    def evaluate_uplift_predictions(
        self,
        data: pd.DataFrame,
        treatment_col: str = 'treatment',
        outcome_col: str = 'outcome',
        uplift_col: str = 'predicted_uplift'
    ) -> Dict[str, float]:
        """
        Evaluate uplift predictions against ground truth.
        
        Args:
            data: Data with treatment, outcome, and predicted uplift
            treatment_col: Treatment indicator
            outcome_col: Outcome variable
            uplift_col: Predicted uplift column
        
        Returns:
            Dictionary with evaluation metrics
        """
        logger.info("Evaluating uplift predictions...")
        
        # Separate treated and control
        treated = data[data[treatment_col] == 1].copy()
        control = data[data[treatment_col] == 0].copy()
        
        # Calculate actual uplift for each unit (using matching)
        # Simplified: compare average outcome in uplift quantiles
        
        # Create uplift deciles
        data['uplift_decile'] = pd.qcut(
            data[uplift_col],
            10,
            labels=False,
            duplicates='drop'
        )
        
        # Calculate actual uplift by decile
        decile_uplifts = []
        for decile in sorted(data['uplift_decile'].unique()):
            decile_data = data[data['uplift_decile'] == decile]
            
            decile_treated = decile_data[decile_data[treatment_col] == 1][outcome_col]
            decile_control = decile_data[decile_data[treatment_col] == 0][outcome_col]
            
            if len(decile_treated) > 0 and len(decile_control) > 0:
                actual_uplift = decile_treated.mean() - decile_control.mean()
                predicted_uplift = decile_data[uplift_col].mean()
                decile_uplifts.append({
                    'decile': decile,
                    'predicted_uplift': predicted_uplift,
                    'actual_uplift': actual_uplift
                })
        
        decile_df = pd.DataFrame(decile_uplifts)
        
        # Calculate correlation between predicted and actual uplift
        if len(decile_df) > 1:
            uplift_correlation = decile_df['predicted_uplift'].corr(decile_df['actual_uplift'])
        else:
            uplift_correlation = 0
        
        # Calculate AUUC (Area Under Uplift Curve)
        auuc = self._calculate_auuc(data, treatment_col, outcome_col, uplift_col)
        
        # Calculate Qini coefficient
        qini = self._calculate_qini(data, treatment_col, outcome_col, uplift_col)
        
        # Calculate uplift calibration error
        calibration_error = np.mean(np.abs(
            decile_df['predicted_uplift'] - decile_df['actual_uplift']
        )) if len(decile_df) > 0 else 0
        
        metrics = {
            'uplift_correlation': uplift_correlation,
            'auuc': auuc,
            'qini_coefficient': qini,
            'calibration_error': calibration_error,
            'n_deciles': len(decile_df)
        }
        
        logger.info(f"Uplift evaluation: AUUC={auuc:.4f}, Qini={qini:.4f}")
        return metrics
    
    def _calculate_auuc(
        self,
        data: pd.DataFrame,
        treatment_col: str,
        outcome_col: str,
        uplift_col: str
    ) -> float:
        """
        Calculate Area Under Uplift Curve.
        
        Sorts by predicted uplift and calculates cumulative uplift.
        """
        # Sort by predicted uplift (descending)
        data_sorted = data.sort_values(uplift_col, ascending=False).copy()
        
        # Calculate cumulative uplift
        cumulative_uplift = []
        n_treated = 0
        n_control = 0
        sum_treated = 0
        sum_control = 0
        
        for _, row in data_sorted.iterrows():
            if row[treatment_col] == 1:
                n_treated += 1
                sum_treated += row[outcome_col]
            else:
                n_control += 1
                sum_control += row[outcome_col]
            
            if n_treated > 0 and n_control > 0:
                uplift = (sum_treated / n_treated) - (sum_control / n_control)
                cumulative_uplift.append(uplift)
        
        # Calculate area under curve
        if len(cumulative_uplift) > 0:
            auuc = np.mean(cumulative_uplift)
        else:
            auuc = 0
        
        return auuc
    
    def _calculate_qini(
        self,
        data: pd.DataFrame,
        treatment_col: str,
        outcome_col: str,
        uplift_col: str
    ) -> float:
        """
        Calculate Qini coefficient.
        
        Similar to AUUC but uses incremental gain.
        """
        # Sort by predicted uplift (descending)
        data_sorted = data.sort_values(uplift_col, ascending=False).copy()
        
        # Calculate incremental gain
        total_treated = data[data[treatment_col] == 1][outcome_col].sum()
        total_control = data[data[treatment_col] == 0][outcome_col].sum()
        overall_uplift = (total_treated / len(data[data[treatment_col] == 1])) - \
                        (total_control / len(data[data[treatment_col] == 0]))
        
        incremental_gains = []
        n_treated = 0
        n_control = 0
        sum_treated = 0
        sum_control = 0
        
        for _, row in data_sorted.iterrows():
            if row[treatment_col] == 1:
                n_treated += 1
                sum_treated += row[outcome_col]
            else:
                n_control += 1
                sum_control += row[outcome_col]
            
            if n_treated > 0 and n_control > 0:
                current_uplift = (sum_treated / n_treated) - (sum_control / n_control)
                incremental_gain = current_uplift - overall_uplift
                incremental_gains.append(incremental_gain)
        
        # Calculate Qini (sum of incremental gains)
        qini = np.sum(incremental_gains) if incremental_gains else 0
        
        return qini
    
    def evaluate_uplift_by_segment(
        self,
        data: pd.DataFrame,
        treatment_col: str = 'treatment',
        outcome_col: str = 'outcome',
        uplift_col: str = 'predicted_uplift',
        segment_col: str = 'segment'
    ) -> Dict[str, Dict[str, float]]:
        """
        Evaluate uplift predictions by customer segment.
        
        Args:
            data: Data with treatment, outcome, predicted uplift, and segment
            treatment_col: Treatment indicator
            outcome_col: Outcome variable
            uplift_col: Predicted uplift column
            segment_col: Segment variable
        
        Returns:
            Dictionary mapping segment to evaluation metrics
        """
        logger.info("Evaluating uplift by segment...")
        
        segment_metrics = {}
        
        for segment in data[segment_col].unique():
            segment_data = data[data[segment_col] == segment].copy()
            
            if len(segment_data) < 100:  # Skip small segments
                continue
            
            metrics = self.evaluate_uplift_predictions(
                segment_data, treatment_col, outcome_col, uplift_col
            )
            
            segment_metrics[segment] = metrics
        
        logger.info(f"Evaluated {len(segment_metrics)} segments")
        return segment_metrics
    
    def generate_uplift_report(
        self,
        true_uplift: pd.DataFrame,
        uplift_metrics: Dict[str, float],
        output_path: Optional[Path] = None
    ) -> str:
        """
        Generate comprehensive uplift evaluation report.
        
        Args:
            true_uplift: DataFrame with true uplift estimates
            uplift_metrics: Dictionary with evaluation metrics
            output_path: Optional path to save report
        
        Returns:
            Report string
        """
        report = f"""
Uplift Ground-Truth Evaluation Report
{'=' * 60}

True Uplift Estimates:
"""
        
        for _, row in true_uplift.iterrows():
            report += f"""
Group: {row['group']}
- True Uplift: {row['true_uplift']:.4f}
- Standard Error: {row['standard_error']:.4f}
- T-Statistic: {row['t_statistic']:.4f}
- P-Value: {row['p_value']:.4f}
- Sample Size: {row['n-treated'] + row['n-control']}
"""
        
        report += f"""
Uplift Prediction Evaluation:
- Uplift Correlation: {uplift_metrics.get('uplift_correlation', 0):.4f}
- AUUC (Area Under Uplift Curve): {uplift_metrics.get('auuc', 0):.4f}
- Qini Coefficient: {uplift_metrics.get('qini_coefficient', 0):.4f}
- Calibration Error: {uplift_metrics.get('calibration_error', 0):.4f}
- Deciles Evaluated: {uplift_metrics.get('n_deciles', 0)}

Interpretation:
- AUUC > 0.5: Good uplift model performance
- AUUC > 0.3: Acceptable performance
- AUUC < 0.3: Poor performance
- Qini > 0: Model adds value over random targeting
- Calibration Error < 0.1: Well-calibrated predictions
- Uplift Correlation > 0.5: Strong correlation with true uplift
"""
        
        if output_path:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(report)
            logger.info(f"Uplift report saved to {output_path}")
        
        return report


def run_uplift_evaluation(
    data: pd.DataFrame,
    treatment_col: str = 'treatment',
    outcome_col: str = 'outcome',
    uplift_col: str = 'predicted_uplift'
) -> Dict[str, any]:
    """
    Convenience function to run complete uplift evaluation.
    
    Args:
        data: Experimental data with treatment, outcome, and predicted uplift
        treatment_col: Treatment indicator
        outcome_col: Outcome variable
        uplift_col: Predicted uplift column
    
    Returns:
        Dictionary with true uplift and evaluation metrics
    """
    evaluator = UpliftEvaluator()
    
    # Calculate true uplift
    true_uplift = evaluator.calculate_true_uplift(
        data, treatment_col, outcome_col
    )
    
    # Evaluate predictions
    uplift_metrics = evaluator.evaluate_uplift_predictions(
        data, treatment_col, outcome_col, uplift_col
    )
    
    return {
        'true_uplift': true_uplift,
        'uplift_metrics': uplift_metrics
    }
