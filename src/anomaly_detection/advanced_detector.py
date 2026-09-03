"""
Advanced Anomaly Detection Module
Implements anomaly detection with root cause analysis for KPI monitoring.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from scipy import stats
from config.logging_config import get_logger

logger = get_logger(__name__)


class AdvancedAnomalyDetector:
    """
    Advanced anomaly detection engine with root cause analysis.
    
    Features:
    - Statistical anomaly detection (z-score, IQR)
    - ML-based anomaly detection (Isolation Forest)
    - Root cause analysis using feature attribution
    - Anomaly scoring and severity classification
    - Multi-dimensional anomaly detection
    """
    
    def __init__(self):
        """Initialize anomaly detector"""
        self.isolation_forest = None
        self.scaler = None
        self.baseline_stats = {}
    
    def calculate_baseline_stats(
        self,
        data: pd.DataFrame,
        value_col: str,
        window_size: int = 30
    ) -> Dict:
        """
        Calculate baseline statistics for anomaly detection.
        
        Args:
            data: DataFrame with time series data
            value_col: Column name for values
            window_size: Window size for rolling statistics
        
        Returns:
            Dictionary with baseline statistics
        """
        logger.info("Calculating baseline statistics...")
        
        values = data[value_col].values
        
        # Basic statistics
        baseline = {
            'mean': float(np.mean(values)),
            'std': float(np.std(values)),
            'median': float(np.median(values)),
            'q25': float(np.percentile(values, 25)),
            'q75': float(np.percentile(values, 75)),
            'min': float(np.min(values)),
            'max': float(np.max(values))
        }
        
        # Rolling statistics
        rolling_mean = data[value_col].rolling(window=window_size).mean()
        rolling_std = data[value_col].rolling(window=window_size).std()
        
        baseline['rolling_mean'] = float(rolling_mean.iloc[-1]) if len(rolling_mean) > 0 else baseline['mean']
        baseline['rolling_std'] = float(rolling_std.iloc[-1]) if len(rolling_std) > 0 else baseline['std']
        
        # IQR
        baseline['iqr'] = baseline['q75'] - baseline['q25']
        baseline['lower_bound'] = baseline['q25'] - 1.5 * baseline['iqr']
        baseline['upper_bound'] = baseline['q75'] + 1.5 * baseline['iqr']
        
        self.baseline_stats = baseline
        
        logger.info(f"Baseline calculated. Mean: {baseline['mean']:.2f}, Std: {baseline['std']:.2f}")
        
        return baseline
    
    def detect_statistical_anomalies(
        self,
        data: pd.DataFrame,
        value_col: str,
        method: str = 'zscore',
        threshold: float = 3.0
    ) -> pd.DataFrame:
        """
        Detect anomalies using statistical methods.
        
        Args:
            data: DataFrame with time series data
            value_col: Column name for values
            method: Detection method ('zscore', 'iqr', 'modified_zscore')
            threshold: Anomaly threshold
        
        Returns:
            DataFrame with anomaly flags
        """
        logger.info(f"Detecting statistical anomalies using {method}...")
        
        result_df = data.copy()
        values = data[value_col].values
        
        if method == 'zscore':
            # Z-score method
            if self.baseline_stats:
                mean = self.baseline_stats['mean']
                std = self.baseline_stats['std']
            else:
                mean = np.mean(values)
                std = np.std(values)
            
            z_scores = (values - mean) / std if std > 0 else np.zeros_like(values)
            result_df['z_score'] = z_scores
            result_df['is_anomaly'] = np.abs(z_scores) > threshold
            result_df['anomaly_score'] = np.abs(z_scores)
        
        elif method == 'iqr':
            # IQR method
            if self.baseline_stats:
                lower_bound = self.baseline_stats['lower_bound']
                upper_bound = self.baseline_stats['upper_bound']
            else:
                q25 = np.percentile(values, 25)
                q75 = np.percentile(values, 75)
                iqr = q75 - q25
                lower_bound = q25 - 1.5 * iqr
                upper_bound = q75 + 1.5 * iqr
            
            result_df['is_anomaly'] = (values < lower_bound) | (values > upper_bound)
            result_df['anomaly_score'] = np.where(
                values < lower_bound,
                (lower_bound - values) / (upper_bound - lower_bound),
                np.where(
                    values > upper_bound,
                    (values - upper_bound) / (upper_bound - lower_bound),
                    0
                )
            )
        
        elif method == 'modified_zscore':
            # Modified Z-score (median-based)
            median = np.median(values)
            mad = np.median(np.abs(values - median))
            modified_z_scores = 0.6745 * (values - median) / mad if mad > 0 else np.zeros_like(values)
            
            result_df['modified_z_score'] = modified_z_scores
            result_df['is_anomaly'] = np.abs(modified_z_scores) > threshold
            result_df['anomaly_score'] = np.abs(modified_z_scores)
        
        # Classify severity
        result_df['severity'] = np.where(
            result_df['anomaly_score'] > threshold * 2,
            'critical',
            np.where(
                result_df['anomaly_score'] > threshold * 1.5,
                'high',
                np.where(
                    result_df['is_anomaly'],
                    'medium',
                    'none'
                )
            )
        )
        
        n_anomalies = result_df['is_anomaly'].sum()
        logger.info(f"Statistical anomaly detection complete. {n_anomalies} anomalies detected")
        
        return result_df
    
    def fit_isolation_forest(
        self,
        data: pd.DataFrame,
        feature_cols: List[str],
        contamination: float = 0.1
    ) -> Dict:
        """
        Fit Isolation Forest model for anomaly detection.
        
        Args:
            data: DataFrame with features
            feature_cols: List of feature column names
            contamination: Expected proportion of anomalies
        
        Returns:
            Dictionary with fitting results
        """
        logger.info("Fitting Isolation Forest model...")
        
        # Prepare data
        X = data[feature_cols].values
        
        # Scale features
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)
        
        # Fit Isolation Forest
        self.isolation_forest = IsolationForest(
            contamination=contamination,
            random_state=42,
            n_estimators=100
        )
        
        self.isolation_forest.fit(X_scaled)
        
        # Get anomaly scores
        anomaly_scores = self.isolation_forest.decision_function(X_scaled)
        
        results = {
            'n_samples': len(data),
            'n_features': len(feature_cols),
            'contamination': contamination,
            'mean_anomaly_score': float(np.mean(anomaly_scores)),
            'std_anomaly_score': float(np.std(anomaly_scores))
        }
        
        logger.info(f"Isolation Forest fitted. Mean anomaly score: {results['mean_anomaly_score']:.3f}")
        
        return results
    
    def detect_ml_anomalies(
        self,
        data: pd.DataFrame,
        feature_cols: List[str]
    ) -> pd.DataFrame:
        """
        Detect anomalies using trained Isolation Forest.
        
        Args:
            data: DataFrame with features
            feature_cols: List of feature column names
        
        Returns:
            DataFrame with anomaly flags
        """
        if self.isolation_forest is None:
            raise ValueError("Isolation Forest not fitted. Call fit_isolation_forest first.")
        
        logger.info("Detecting anomalies using Isolation Forest...")
        
        result_df = data.copy()
        
        # Prepare data
        X = data[feature_cols].values
        X_scaled = self.scaler.transform(X)
        
        # Predict anomalies
        predictions = self.isolation_forest.predict(X_scaled)
        anomaly_scores = self.isolation_forest.decision_function(X_scaled)
        
        result_df['is_anomaly'] = predictions == -1
        result_df['anomaly_score'] = -anomaly_scores  # Convert to positive (higher = more anomalous)
        
        # Classify severity
        score_threshold = np.percentile(result_df['anomaly_score'], 90)
        result_df['severity'] = np.where(
            result_df['anomaly_score'] > score_threshold * 1.5,
            'critical',
            np.where(
                result_df['anomaly_score'] > score_threshold,
                'high',
                np.where(
                    result_df['is_anomaly'],
                    'medium',
                    'none'
                )
            )
        )
        
        n_anomalies = result_df['is_anomaly'].sum()
        logger.info(f"ML anomaly detection complete. {n_anomalies} anomalies detected")
        
        return result_df
    
    def analyze_root_cause(
        self,
        anomaly_data: pd.DataFrame,
        feature_cols: List[str],
        baseline_data: pd.DataFrame = None
    ) -> Dict:
        """
        Analyze root cause of anomalies using feature attribution.
        
        Args:
            anomaly_data: Data containing anomalies
            feature_cols: List of feature column names
            baseline_data: Baseline data for comparison (optional)
        
        Returns:
            Dictionary with root cause analysis
        """
        logger.info("Analyzing root causes...")
        
        if baseline_data is None:
            # Use overall statistics as baseline
            baseline_data = anomaly_data
        
        # Calculate feature deviations
        root_causes = []
        
        for col in feature_cols:
            if col in anomaly_data.columns:
                anomaly_values = anomaly_data[col].values
                baseline_values = baseline_data[col].values
                
                # Calculate deviation
                anomaly_mean = np.mean(anomaly_values)
                baseline_mean = np.mean(baseline_values)
                baseline_std = np.std(baseline_values)
                
                if baseline_std > 0:
                    deviation = (anomaly_mean - baseline_mean) / baseline_std
                else:
                    deviation = 0
                
                # Calculate contribution to anomaly
                contribution = abs(deviation)
                
                root_causes.append({
                    'feature': col,
                    'anomaly_mean': float(anomaly_mean),
                    'baseline_mean': float(baseline_mean),
                    'deviation': float(deviation),
                    'contribution': float(contribution)
                })
        
        # Sort by contribution
        root_causes.sort(key=lambda x: x['contribution'], reverse=True)
        
        result = {
            'root_causes': root_causes[:5],  # Top 5 causes
            'primary_cause': root_causes[0]['feature'] if root_causes else None,
            'n_features_analyzed': len(root_causes)
        }
        
        logger.info(f"Root cause analysis complete. Primary cause: {result['primary_cause']}")
        
        return result
    
    def detect_multidimensional_anomalies(
        self,
        data: pd.DataFrame,
        feature_cols: List[str],
        method: str = 'isolation_forest'
    ) -> pd.DataFrame:
        """
        Detect anomalies in multi-dimensional data.
        
        Args:
            data: DataFrame with features
            feature_cols: List of feature column names
            method: Detection method ('isolation_forest', 'mahalanobis')
        
        Returns:
            DataFrame with anomaly flags
        """
        logger.info(f"Detecting multi-dimensional anomalies using {method}...")
        
        if method == 'isolation_forest':
            return self.detect_ml_anomalies(data, feature_cols)
        
        elif method == 'mahalanobis':
            result_df = data.copy()
            
            # Calculate Mahalanobis distance
            X = data[feature_cols].values
            
            # Calculate covariance matrix
            cov_matrix = np.cov(X.T)
            
            # Handle singular matrix
            if np.linalg.det(cov_matrix) == 0:
                cov_matrix += np.eye(len(feature_cols)) * 1e-6
            
            inv_cov_matrix = np.linalg.inv(cov_matrix)
            mean = np.mean(X, axis=0)
            
            # Calculate distances
            diff = X - mean
            mahalanobis_dist = np.sqrt(np.sum(diff @ inv_cov_matrix * diff, axis=1))
            
            # Threshold using chi-square distribution
            threshold = stats.chi2.ppf(0.95, df=len(feature_cols))
            
            result_df['mahalanobis_distance'] = mahalanobis_dist
            result_df['is_anomaly'] = mahalanobis_dist > threshold
            result_df['anomaly_score'] = mahalanobis_dist / threshold
            result_df['severity'] = np.where(
                result_df['anomaly_score'] > 2,
                'critical',
                np.where(
                    result_df['anomaly_score'] > 1.5,
                    'high',
                    np.where(
                        result_df['is_anomaly'],
                        'medium',
                        'none'
                    )
                )
            )
            
            n_anomalies = result_df['is_anomaly'].sum()
            logger.info(f"Multi-dimensional anomaly detection complete. {n_anomalies} anomalies detected")
            
            return result_df
        
        else:
            raise ValueError(f"Unknown method: {method}")


def run_anomaly_detection_pipeline(
    data: pd.DataFrame,
    value_col: str,
    feature_cols: List[str],
    method: str = 'isolation_forest'
) -> Tuple[AdvancedAnomalyDetector, Dict]:
    """
    Convenience function to run complete anomaly detection pipeline.
    
    Args:
        data: Time series data
        value_col: Value column name
        feature_cols: Feature column names
        method: Detection method
    
    Returns:
        Tuple of (detector, results)
    """
    detector = AdvancedAnomalyDetector()
    
    # Calculate baseline
    baseline = detector.calculate_baseline_stats(data, value_col)
    
    # Detect anomalies
    if method == 'isolation_forest':
        detector.fit_isolation_forest(data, feature_cols)
        anomaly_results = detector.detect_ml_anomalies(data, feature_cols)
    else:
        anomaly_results = detector.detect_statistical_anomalies(data, value_col, method)
    
    # Analyze root causes for anomalies
    anomalies = anomaly_results[anomaly_results['is_anomaly']]
    root_cause = None
    if len(anomalies) > 0:
        root_cause = detector.analyze_root_cause(anomalies, feature_cols)
    
    results = {
        'baseline': baseline,
        'anomaly_results': anomaly_results,
        'root_cause': root_cause,
        'n_anomalies': int(anomaly_results['is_anomaly'].sum())
    }
    
    return detector, results
