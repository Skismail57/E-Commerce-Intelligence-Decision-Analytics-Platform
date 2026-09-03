"""
Data Drift Monitoring Module
Implements data drift detection and monitoring for ML models.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from scipy import stats
from sklearn.metrics import pairwise_distances
from config.logging_config import get_logger

logger = get_logger(__name__)


class DataDriftMonitor:
    """
    Data drift monitoring engine.
    
    Features:
    - Statistical drift detection (KS test, Chi-square)
    - Population Stability Index (PSI)
    - Feature distribution comparison
    - Drift alerting
    - Drift visualization data
    """
    
    def __init__(self):
        """Initialize data drift monitor"""
        self.baseline_stats = {}
        self.drift_history = []
        logger.info("Data drift monitor initialized")
    
    def calculate_baseline(
        self,
        data: pd.DataFrame,
        feature_cols: List[str] = None
    ) -> Dict:
        """
        Calculate baseline statistics for drift detection.
        
        Args:
            data: Baseline data
            feature_cols: List of feature columns to monitor
        
        Returns:
            Dictionary with baseline statistics
        """
        logger.info("Calculating baseline statistics...")
        
        if feature_cols is None:
            feature_cols = data.select_dtypes(include=[np.number]).columns.tolist()
        
        baseline = {}
        
        for col in feature_cols:
            if col in data.columns:
                col_data = data[col].dropna()
                
                baseline[col] = {
                    'mean': float(col_data.mean()),
                    'std': float(col_data.std()),
                    'min': float(col_data.min()),
                    'max': float(col_data.max()),
                    'median': float(col_data.median()),
                    'q25': float(col_data.quantile(0.25)),
                    'q75': float(col_data.quantile(0.75)),
                    'n_samples': len(col_data)
                }
        
        self.baseline_stats = baseline
        
        logger.info(f"Baseline calculated for {len(baseline)} features")
        
        return baseline
    
    def calculate_psi(
        self,
        baseline: np.ndarray,
        current: np.ndarray,
        bins: int = 10
    ) -> float:
        """
        Calculate Population Stability Index (PSI).
        
        Args:
            baseline: Baseline data
            current: Current data
            bins: Number of bins for PSI calculation
        
        Returns:
            PSI value
        """
        # Create bins based on baseline
        _, bin_edges = np.histogram(baseline, bins=bins)
        
        # Calculate distributions
        baseline_dist, _ = np.histogram(baseline, bins=bin_edges)
        current_dist, _ = np.histogram(current, bins=bin_edges)
        
        # Normalize to percentages
        baseline_pct = baseline_dist / len(baseline)
        current_pct = current_dist / len(current)
        
        # Add small value to avoid division by zero
        baseline_pct = np.maximum(baseline_pct, 0.0001)
        current_pct = np.maximum(current_pct, 0.0001)
        
        # Calculate PSI
        psi = np.sum((current_pct - baseline_pct) * np.log(current_pct / baseline_pct))
        
        return float(psi)
    
    def detect_statistical_drift(
        self,
        current_data: pd.DataFrame,
        feature_cols: List[str] = None,
        significance_level: float = 0.05
    ) -> pd.DataFrame:
        """
        Detect statistical drift using hypothesis tests.
        
        Args:
            current_data: Current data to compare
            feature_cols: List of feature columns
            significance_level: Significance level for tests
        
        Returns:
            DataFrame with drift detection results
        """
        logger.info("Detecting statistical drift...")
        
        if not self.baseline_stats:
            raise ValueError("Baseline not calculated. Call calculate_baseline first.")
        
        if feature_cols is None:
            feature_cols = list(self.baseline_stats.keys())
        
        results = []
        
        for col in feature_cols:
            if col not in current_data.columns:
                continue
            
            baseline_mean = self.baseline_stats[col]['mean']
            baseline_std = self.baseline_stats[col]['std']
            
            current_values = current_data[col].dropna()
            
            if len(current_values) < 10:
                continue
            
            # T-test for mean shift
            t_stat, p_value = stats.ttest_1samp(current_values, baseline_mean)
            
            # Calculate PSI
            psi = self.calculate_psi(
                np.random.normal(baseline_mean, baseline_std, 1000),
                current_values
            )
            
            # Determine drift status
            is_drift = p_value < significance_level
            drift_severity = 'high' if psi > 0.5 else 'medium' if psi > 0.2 else 'low' if psi > 0.1 else 'none'
            
            results.append({
                'feature': col,
                'baseline_mean': baseline_mean,
                'current_mean': float(current_values.mean()),
                'mean_shift': float(current_values.mean() - baseline_mean),
                't_statistic': float(t_stat),
                'p_value': float(p_value),
                'psi': float(psi),
                'is_drift': is_drift,
                'drift_severity': drift_severity
            })
        
        results_df = pd.DataFrame(results)
        
        logger.info(f"Statistical drift detection complete. Drift detected in {results_df['is_drift'].sum()} features")
        
        return results_df
    
    def detect_distribution_drift(
        self,
        current_data: pd.DataFrame,
        feature_cols: List[str] = None
    ) -> pd.DataFrame:
        """
        Detect distribution drift using KS test.
        
        Args:
            current_data: Current data
            feature_cols: List of feature columns
        
        Returns:
            DataFrame with distribution drift results
        """
        logger.info("Detecting distribution drift...")
        
        if not self.baseline_stats:
            raise ValueError("Baseline not calculated. Call calculate_baseline first.")
        
        if feature_cols is None:
            feature_cols = list(self.baseline_stats.keys())
        
        results = []
        
        for col in feature_cols:
            if col not in current_data.columns:
                continue
            
            baseline_mean = self.baseline_stats[col]['mean']
            baseline_std = self.baseline_stats[col]['std']
            
            current_values = current_data[col].dropna()
            
            if len(current_values) < 10:
                continue
            
            # Generate synthetic baseline distribution
            baseline_sample = np.random.normal(baseline_mean, baseline_std, len(current_values))
            
            # KS test for distribution difference
            ks_stat, ks_pvalue = stats.ks_2samp(baseline_sample, current_values)
            
            # Determine drift
            is_drift = ks_pvalue < 0.05
            drift_severity = 'high' if ks_stat > 0.3 else 'medium' if ks_stat > 0.15 else 'low' if ks_stat > 0.05 else 'none'
            
            results.append({
                'feature': col,
                'ks_statistic': float(ks_stat),
                'ks_pvalue': float(ks_pvalue),
                'is_drift': is_drift,
                'drift_severity': drift_severity
            })
        
        results_df = pd.DataFrame(results)
        
        logger.info(f"Distribution drift detection complete")
        
        return results_df
    
    def calculate_drift_score(
        self,
        drift_results: pd.DataFrame
    ) -> Dict:
        """
        Calculate overall drift score.
        
        Args:
            drift_results: DataFrame with drift detection results
        
        Returns:
            Dictionary with drift score
        """
        logger.info("Calculating overall drift score...")
        
        if drift_results.empty:
            return {'overall_drift_score': 0.0, 'n_features_drifted': 0}
        
        # Count drifted features
        n_drifted = drift_results['is_drift'].sum()
        n_total = len(drift_results)
        
        # Calculate weighted drift score
        severity_weights = {'high': 3, 'medium': 2, 'low': 1, 'none': 0}
        drift_scores = drift_results['drift_severity'].map(severity_weights)
        
        overall_score = drift_scores.mean() / 3.0  # Normalize to 0-1
        
        results = {
            'overall_drift_score': float(overall_score),
            'n_features_drifted': int(n_drifted),
            'n_total_features': int(n_total),
            'drift_percentage': float(n_drifted / n_total * 100) if n_total > 0 else 0
        }
        
        logger.info(f"Overall drift score: {overall_score:.3f}")
        
        return results
    
    def generate_drift_alert(
        self,
        drift_score: float,
        threshold: float = 0.3
    ) -> Dict:
        """
        Generate drift alert if threshold exceeded.
        
        Args:
            drift_score: Overall drift score
            threshold: Alert threshold
        
        Returns:
            Dictionary with alert information
        """
        alert = {
            'drift_score': drift_score,
            'threshold': threshold,
            'alert_triggered': drift_score > threshold,
            'alert_level': 'critical' if drift_score > 0.5 else 'warning' if drift_score > threshold else 'none'
        }
        
        if alert['alert_triggered']:
            logger.warning(f"Drift alert triggered! Score: {drift_score:.3f}")
        
        return alert
    
    def monitor_drift_over_time(
        self,
        data_stream: List[pd.DataFrame],
        feature_cols: List[str] = None
    ) -> pd.DataFrame:
        """
        Monitor drift over time from data stream.
        
        Args:
            data_stream: List of DataFrames representing time periods
            feature_cols: List of feature columns
        
        Returns:
            DataFrame with drift over time
        """
        logger.info(f"Monitoring drift over {len(data_stream)} time periods...")
        
        drift_history = []
        
        for i, data in enumerate(data_stream):
            # Detect drift
            drift_results = self.detect_statistical_drift(data, feature_cols)
            drift_score = self.calculate_drift_score(drift_results)
            
            drift_history.append({
                'time_period': i,
                'overall_drift_score': drift_score['overall_drift_score'],
                'n_features_drifted': drift_score['n_features_drifted'],
                'drift_percentage': drift_score['drift_percentage']
            })
        
        drift_history_df = pd.DataFrame(drift_history)
        self.drift_history = drift_history
        
        logger.info(f"Drift monitoring complete")
        
        return drift_history_df


def run_drift_monitoring_pipeline(
    baseline_data: pd.DataFrame,
    current_data: pd.DataFrame,
    feature_cols: List[str] = None
) -> Tuple[DataDriftMonitor, Dict]:
    """
    Convenience function to run drift monitoring pipeline.
    
    Args:
        baseline_data: Baseline data
        current_data: Current data
        feature_cols: Feature columns to monitor
    
    Returns:
        Tuple of (monitor, results)
    """
    monitor = DataDriftMonitor()
    
    # Calculate baseline
    baseline = monitor.calculate_baseline(baseline_data, feature_cols)
    
    # Detect statistical drift
    statistical_drift = monitor.detect_statistical_drift(current_data, feature_cols)
    
    # Detect distribution drift
    distribution_drift = monitor.detect_distribution_drift(current_data, feature_cols)
    
    # Calculate drift score
    drift_score = monitor.calculate_drift_score(statistical_drift)
    
    # Generate alert
    alert = monitor.generate_drift_alert(drift_score['overall_drift_score'])
    
    results = {
        'baseline': baseline,
        'statistical_drift': statistical_drift,
        'distribution_drift': distribution_drift,
        'drift_score': drift_score,
        'alert': alert
    }
    
    return monitor, results
