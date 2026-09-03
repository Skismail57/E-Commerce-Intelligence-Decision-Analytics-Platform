"""
Model Monitoring and Auto-Retraining Module
Implements model performance monitoring and automatic retraining.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from config.logging_config import get_logger

logger = get_logger(__name__)


class ModelMonitor:
    """
    Model performance monitoring and auto-retraining engine.
    
    Features:
    - Performance metrics tracking
    - Model degradation detection
    - Automatic retraining triggers
    - Model versioning
    - Performance trend analysis
    """
    
    def __init__(self):
        """Initialize model monitor"""
        self.performance_history = {}
        self.model_versions = {}
        self.retraining_triggers = {}
        logger.info("Model monitor initialized")
    
    def log_performance(
        self,
        model_id: str,
        metrics: Dict[str, float],
        timestamp: datetime = None
    ) -> Dict:
        """
        Log model performance metrics.
        
        Args:
            model_id: Model identifier
            metrics: Dictionary of performance metrics
            timestamp: Timestamp of the measurement
        
        Returns:
            Dictionary with logging result
        """
        if timestamp is None:
            timestamp = datetime.now()
        
        if model_id not in self.performance_history:
            self.performance_history[model_id] = []
        
        performance_record = {
            'timestamp': timestamp.isoformat(),
            'metrics': metrics
        }
        
        self.performance_history[model_id].append(performance_record)
        
        logger.info(f"Performance logged for model {model_id}")
        
        return performance_record
    
    def detect_performance_degradation(
        self,
        model_id: str,
        metric_name: str = 'accuracy',
        threshold: float = 0.05,
        window_size: int = 10
    ) -> Dict:
        """
        Detect if model performance has degraded.
        
        Args:
            model_id: Model identifier
            metric_name: Metric to monitor
            threshold: Degradation threshold
            window_size: Window size for comparison
        
        Returns:
            Dictionary with degradation detection result
        """
        if model_id not in self.performance_history:
            raise ValueError(f"No performance history for model {model_id}")
        
        history = self.performance_history[model_id]
        
        if len(history) < window_size * 2:
            return {
                'model_id': model_id,
                'is_degraded': False,
                'reason': 'Insufficient data'
            }
        
        # Get recent and baseline performance
        recent = history[-window_size:]
        baseline = history[-(window_size * 2):-window_size]
        
        recent_values = [r['metrics'].get(metric_name, 0) for r in recent]
        baseline_values = [r['metrics'].get(metric_name, 0) for r in baseline]
        
        recent_mean = np.mean(recent_values)
        baseline_mean = np.mean(baseline_values)
        
        # Calculate degradation
        degradation = (baseline_mean - recent_mean) / baseline_mean if baseline_mean > 0 else 0
        
        is_degraded = degradation > threshold
        
        result = {
            'model_id': model_id,
            'metric_name': metric_name,
            'baseline_mean': float(baseline_mean),
            'recent_mean': float(recent_mean),
            'degradation_pct': float(degradation * 100),
            'is_degraded': is_degraded,
            'threshold': threshold
        }
        
        if is_degraded:
            logger.warning(f"Performance degradation detected for {model_id}: {degradation:.1%}")
        
        return result
    
    def calculate_performance_trend(
        self,
        model_id: str,
        metric_name: str = 'accuracy'
    ) -> Dict:
        """
        Calculate performance trend over time.
        
        Args:
            model_id: Model identifier
            metric_name: Metric to analyze
        
        Returns:
            Dictionary with trend analysis
        """
        if model_id not in self.performance_history:
            raise ValueError(f"No performance history for model {model_id}")
        
        history = self.performance_history[model_id]
        values = [r['metrics'].get(metric_name, 0) for r in history]
        
        if len(values) < 2:
            return {
                'model_id': model_id,
                'trend': 'insufficient_data',
                'slope': 0.0
            }
        
        # Calculate linear trend
        x = np.arange(len(values))
        slope, intercept = np.polyfit(x, values, 1)
        
        # Determine trend direction
        if slope > 0.001:
            trend = 'improving'
        elif slope < -0.001:
            trend = 'declining'
        else:
            trend = 'stable'
        
        # Calculate volatility
        volatility = np.std(values)
        
        result = {
            'model_id': model_id,
            'metric_name': metric_name,
            'trend': trend,
            'slope': float(slope),
            'volatility': float(volatility),
            'current_value': float(values[-1]),
            'mean_value': float(np.mean(values)),
            'n_observations': len(values)
        }
        
        return result
    
    def should_retrain(
        self,
        model_id: str,
        degradation_threshold: float = 0.05,
        min_observations: int = 20
    ) -> Dict:
        """
        Determine if model should be retrained.
        
        Args:
            model_id: Model identifier
            degradation_threshold: Degradation threshold
            min_observations: Minimum observations required
        
        Returns:
            Dictionary with retraining decision
        """
        if model_id not in self.performance_history:
            return {
                'model_id': model_id,
                'should_retrain': False,
                'reason': 'No performance history'
            }
        
        history = self.performance_history[model_id]
        
        if len(history) < min_observations:
            return {
                'model_id': model_id,
                'should_retrain': False,
                'reason': f'Insufficient observations ({len(history)} < {min_observations})'
            }
        
        # Check for degradation
        degradation_result = self.detect_performance_degradation(
            model_id, threshold=degradation_threshold
        )
        
        # Check trend
        trend_result = self.calculate_performance_trend(model_id)
        
        # Decision logic
        should_retrain = (
            degradation_result['is_degraded'] or
            trend_result['trend'] == 'declining'
        )
        
        reasons = []
        if degradation_result['is_degraded']:
            reasons.append(f"Performance degraded by {degradation_result['degradation_pct']:.1%}")
        if trend_result['trend'] == 'declining':
            reasons.append("Performance trend is declining")
        
        result = {
            'model_id': model_id,
            'should_retrain': should_retrain,
            'reasons': reasons,
            'degradation_result': degradation_result,
            'trend_result': trend_result
        }
        
        if should_retrain:
            logger.info(f"Retraining recommended for model {model_id}: {', '.join(reasons)}")
        
        return result
    
    def trigger_retraining(
        self,
        model_id: str,
        training_function: callable,
        training_data: pd.DataFrame,
        target_col: str
    ) -> Dict:
        """
        Trigger model retraining.
        
        Args:
            model_id: Model identifier
            training_function: Function to train the model
            training_data: Training data
            target_col: Target column name
        
        Returns:
            Dictionary with retraining result
        """
        logger.info(f"Triggering retraining for model {model_id}")
        
        try:
            # Train new model
            new_model = training_function(training_data, target_col)
            
            # Increment version
            if model_id not in self.model_versions:
                self.model_versions[model_id] = 0
            self.model_versions[model_id] += 1
            
            # Record retraining
            retraining_record = {
                'model_id': model_id,
                'version': self.model_versions[model_id],
                'timestamp': datetime.now().isoformat(),
                'status': 'success'
            }
            
            if model_id not in self.retraining_triggers:
                self.retraining_triggers[model_id] = []
            self.retraining_triggers[model_id].append(retraining_record)
            
            logger.info(f"Retraining successful for model {model_id}. New version: {self.model_versions[model_id]}")
            
            return {
                'success': True,
                'model_id': model_id,
                'new_version': self.model_versions[model_id],
                'new_model': new_model
            }
        
        except Exception as e:
            logger.error(f"Retraining failed for model {model_id}: {e}")
            
            return {
                'success': False,
                'model_id': model_id,
                'error': str(e)
            }
    
    def get_model_health_report(
        self,
        model_id: str
    ) -> Dict:
        """
        Generate comprehensive model health report.
        
        Args:
            model_id: Model identifier
        
        Returns:
            Dictionary with health report
        """
        logger.info(f"Generating health report for model {model_id}")
        
        # Get performance trend
        trend_result = self.calculate_performance_trend(model_id)
        
        # Check if retraining is needed
        retrain_decision = self.should_retrain(model_id)
        
        # Get version info
        current_version = self.model_versions.get(model_id, 0)
        
        # Get retraining history
        retraining_history = self.retraining_triggers.get(model_id, [])
        
        # Calculate health score
        health_score = 1.0
        if trend_result['trend'] == 'declining':
            health_score -= 0.3
        if retrain_decision['should_retrain']:
            health_score -= 0.4
        if trend_result['volatility'] > 0.1:
            health_score -= 0.2
        health_score = max(0.0, health_score)
        
        # Determine health status
        if health_score >= 0.8:
            health_status = 'healthy'
        elif health_score >= 0.5:
            health_status = 'warning'
        else:
            health_status = 'critical'
        
        report = {
            'model_id': model_id,
            'current_version': current_version,
            'health_score': float(health_score),
            'health_status': health_status,
            'performance_trend': trend_result,
            'retrain_decision': retrain_decision,
            'n_retrainings': len(retraining_history),
            'last_retrained': retraining_history[-1]['timestamp'] if retraining_history else None
        }
        
        logger.info(f"Health report generated for {model_id}: {health_status}")
        
        return report
    
    def compare_model_versions(
        self,
        model_id: str,
        version1: int,
        version2: int
    ) -> Dict:
        """
        Compare performance between two model versions.
        
        Args:
            model_id: Model identifier
            version1: First version number
            version2: Second version number
        
        Returns:
            Dictionary with version comparison
        """
        logger.info(f"Comparing versions {version1} and {version2} for model {model_id}")
        
        # This would typically retrieve performance data for each version
        # For now, return a placeholder
        result = {
            'model_id': model_id,
            'version1': version1,
            'version2': version2,
            'comparison': 'Version comparison requires version-specific performance tracking'
        }
        
        return result


def run_model_monitoring_pipeline(
    model_id: str,
    performance_data: List[Dict[str, float]]
) -> Tuple[ModelMonitor, Dict]:
    """
    Convenience function to run model monitoring pipeline.
    
    Args:
        model_id: Model identifier
        performance_data: List of performance metric dictionaries
    
    Returns:
        Tuple of (monitor, results)
    """
    monitor = ModelMonitor()
    
    # Log performance data
    for metrics in performance_data:
        monitor.log_performance(model_id, metrics)
    
    # Calculate trend
    trend = monitor.calculate_performance_trend(model_id)
    
    # Check for degradation
    degradation = monitor.detect_performance_degradation(model_id)
    
    # Check if retraining needed
    retrain_decision = monitor.should_retrain(model_id)
    
    # Generate health report
    health_report = monitor.get_model_health_report(model_id)
    
    results = {
        'performance_trend': trend,
        'degradation_detection': degradation,
        'retrain_decision': retrain_decision,
        'health_report': health_report
    }
    
    return monitor, results
