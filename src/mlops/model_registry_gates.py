"""
Model Registry Promotion Gates Module
Implements automated promotion gates for model deployment.

Architecture:
- Stage-based model promotion (Staging → Production)
- Automated quality gates (metrics thresholds)
- Data drift detection
- Model performance monitoring
- Rollback capabilities
"""

from typing import Dict, List, Optional, Tuple, Callable
import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path
import mlflow
from mlflow.tracking import MlflowClient

from config.logging_config import get_logger
from config.settings import settings

logger = get_logger(__name__)


class ModelPromotionGates:
    """
    Model promotion gates for automated deployment decisions.
    
    Implements quality gates that models must pass before
    being promoted to production.
    """
    
    def __init__(self, model_name: str):
        """
        Initialize model promotion gates.
        
        Args:
            model_name: Name of the model in registry
        """
        self.model_name = model_name
        self.client = MlflowClient()
        
        # Define quality gates
        self.quality_gates = {
            'accuracy': {'min': 0.8, 'critical': True},
            'precision': {'min': 0.75, 'critical': True},
            'recall': {'min': 0.7, 'critical': True},
            'f1_score': {'min': 0.75, 'critical': True},
            'auc_roc': {'min': 0.8, 'critical': True},
            'data_drift_score': {'max': 0.3, 'critical': False}
        }
        
        logger.info(f"Model Promotion Gates initialized for: {model_name}")
    
    def set_quality_gate(
        self,
        metric_name: str,
        threshold: float,
        min_max: str = 'min',
        critical: bool = True
    ) -> None:
        """
        Set a quality gate for a metric.
        
        Args:
            metric_name: Name of the metric
            threshold: Threshold value
            min_max: 'min' for minimum threshold, 'max' for maximum
            critical: Whether this is a critical gate
        """
        self.quality_gates[metric_name] = {
            'min' if min_max == 'min' else 'max': threshold,
            'critical': critical
        }
        logger.info(f"Set quality gate: {metric_name} {min_max}={threshold}")
    
    def check_quality_gates(
        self,
        metrics: Dict[str, float]
    ) -> Tuple[bool, Dict[str, str]]:
        """
        Check if model metrics pass quality gates.
        
        Args:
            metrics: Dictionary of model metrics
        
        Returns:
            Tuple of (all_passed, gate_results)
        """
        gate_results = {}
        all_passed = True
        
        for metric_name, gate_config in self.quality_gates.items():
            if metric_name not in metrics:
                gate_results[metric_name] = 'SKIPPED'
                continue
            
            metric_value = metrics[metric_name]
            
            # Check minimum threshold
            if 'min' in gate_config:
                threshold = gate_config['min']
                passed = metric_value >= threshold
                gate_results[metric_name] = 'PASSED' if passed else 'FAILED'
                
                if gate_config.get('critical', False) and not passed:
                    all_passed = False
            
            # Check maximum threshold
            elif 'max' in gate_config:
                threshold = gate_config['max']
                passed = metric_value <= threshold
                gate_results[metric_name] = 'PASSED' if passed else 'FAILED'
                
                if gate_config.get('critical', False) and not passed:
                    all_passed = False
        
        logger.info(f"Quality gates check: {'PASSED' if all_passed else 'FAILED'}")
        return all_passed, gate_results
    
    def register_model(
        self,
        run_id: str,
        model_name: Optional[str] = None
    ) -> str:
        """
        Register a model from a run.
        
        Args:
            run_id: MLflow run ID
            model_name: Optional custom model name
        
        Returns:
            Registered model version
        """
        model_name = model_name or self.model_name
        
        # Register model
        model_uri = f"runs:/{run_id}/model"
        model_version = mlflow.register_model(
            model_uri,
            model_name
        )
        
        logger.info(f"Registered model {model_name} version {model_version.version}")
        return model_version.version
    
    def promote_to_staging(
        self,
        model_version: str,
        metrics: Dict[str, float]
    ) -> Tuple[bool, str]:
        """
        Promote model to staging after quality gate check.
        
        Args:
            model_version: Model version to promote
            metrics: Model metrics for quality gate check
        
        Returns:
            Tuple of (success, message)
        """
        # Check quality gates
        gates_passed, gate_results = self.check_quality_gates(metrics)
        
        if not gates_passed:
            failed_gates = [k for k, v in gate_results.items() if v == 'FAILED']
            message = f"Quality gates failed: {failed_gates}"
            logger.warning(message)
            return False, message
        
        # Promote to staging
        try:
            self.client.transition_model_version_stage(
                name=self.model_name,
                version=model_version,
                stage="Staging"
            )
            message = f"Model {model_version} promoted to Staging"
            logger.info(message)
            return True, message
        except Exception as e:
            message = f"Failed to promote to Staging: {e}"
            logger.error(message)
            return False, message
    
    def promote_to_production(
        self,
        model_version: str,
        require_staging: bool = True
    ) -> Tuple[bool, str]:
        """
        Promote model to production.
        
        Args:
            model_version: Model version to promote
            require_staging: Whether model must be in Staging first
        
        Returns:
            Tuple of (success, message)
        """
        # Check if model is in staging (if required)
        if require_staging:
            model_version_info = self.client.get_model_version(
                name=self.model_name,
                version=model_version
            )
            
            current_stage = model_version_info.current_stage
            if current_stage != "Staging":
                message = f"Model must be in Staging before production (current: {current_stage})"
                logger.warning(message)
                return False, message
        
        # Promote to production
        try:
            self.client.transition_model_version_stage(
                name=self.model_name,
                version=model_version,
                stage="Production",
                archive_existing_versions=True
            )
            message = f"Model {model_version} promoted to Production"
            logger.info(message)
            return True, message
        except Exception as e:
            message = f"Failed to promote to Production: {e}"
            logger.error(message)
            return False, message
    
    def rollback_model(
        self,
        from_version: str,
        to_version: str
    ) -> Tuple[bool, str]:
        """
        Rollback to a previous model version.
        
        Args:
            from_version: Current production version
            to_version: Version to rollback to
        
        Returns:
            Tuple of (success, message)
        """
        try:
            # Archive current production model
            self.client.transition_model_version_stage(
                name=self.model_name,
                version=from_version,
                stage="Archived"
            )
            
            # Promote rollback version to production
            self.client.transition_model_version_stage(
                name=self.model_name,
                version=to_version,
                stage="Production"
            )
            
            message = f"Rolled back from version {from_version} to {to_version}"
            logger.info(message)
            return True, message
        except Exception as e:
            message = f"Rollback failed: {e}"
            logger.error(message)
            return False, message
    
    def get_production_model(self) -> Optional[str]:
        """
        Get the current production model version.
        
        Returns:
            Production model version or None
        """
        try:
            production_models = self.client.get_latest_versions(
                name=self.model_name,
                stages=["Production"]
            )
            
            if production_models:
                return production_models[0].version
            else:
                return None
        except Exception as e:
            logger.error(f"Error getting production model: {e}")
            return None
    
    def compare_model_versions(
        self,
        version1: str,
        version2: str
    ) -> Dict[str, Dict[str, float]]:
        """
        Compare metrics between two model versions.
        
        Args:
            version1: First model version
            version2: Second model version
        
        Returns:
            Dictionary with metrics comparison
        """
        comparison = {}
        
        for version in [version1, version2]:
            model_version = self.client.get_model_version(
                name=self.model_name,
                version=version
            )
            
            run_id = model_version.run_id
            run = self.client.get_run(run_id)
            
            comparison[f"version_{version}"] = run.data.metrics
        
        # Calculate differences
        metrics_v1 = comparison[f"version_{version1}"]
        metrics_v2 = comparison[f"version_{version2}"]
        
        comparison['differences'] = {}
        for metric in set(metrics_v1.keys()) & set(metrics_v2.keys()):
            diff = metrics_v2[metric] - metrics_v1[metric]
            comparison['differences'][metric] = diff
        
        logger.info(f"Compared versions {version1} and {version2}")
        return comparison
    
    def generate_promotion_report(
        self,
        gate_results: Dict[str, str],
        metrics: Dict[str, float],
        output_path: Optional[Path] = None
    ) -> str:
        """
        Generate model promotion report.
        
        Args:
            gate_results: Quality gate results
            metrics: Model metrics
            output_path: Optional path to save report
        
        Returns:
            Report string
        """
        report = f"""
Model Promotion Report
{'=' * 60}

Model: {self.model_name}
Timestamp: {datetime.now().isoformat()}

Quality Gates Results:
"""
        
        for metric, result in gate_results.items():
            threshold = self.quality_gates.get(metric, {})
            threshold_str = ""
            if 'min' in threshold:
                threshold_str = f" (min: {threshold['min']})"
            elif 'max' in threshold:
                threshold_str = f" (max: {threshold['max']})"
            
            report += f"- {metric}: {result}{threshold_str}\n"
        
        report += f"""
Model Metrics:
"""
        
        for metric, value in metrics.items():
            report += f"- {metric}: {value:.4f}\n"
        
        report += f"""
Recommendation:
"""
        
        failed_gates = [k for k, v in gate_results.items() if v == 'FAILED']
        if failed_gates:
            report += f"DO NOT PROMOTE - Failed gates: {failed_gates}\n"
        else:
            report += "APPROVED FOR PROMOTION - All quality gates passed\n"
        
        if output_path:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(report)
            logger.info(f"Promotion report saved to {output_path}")
        
        return report


def run_model_promotion_pipeline(
    model_name: str,
    run_id: str,
    metrics: Dict[str, float],
    promote_to_prod: bool = False
) -> Dict[str, any]:
    """
    Convenience function to run model promotion pipeline.
    
    Args:
        model_name: Name of the model
        run_id: MLflow run ID
        metrics: Model metrics
        promote_to_prod: Whether to promote to production
    
    Returns:
        Dictionary with promotion results
    """
    gates = ModelPromotionGates(model_name)
    
    # Register model
    version = gates.register_model(run_id)
    
    # Promote to staging
    staging_success, staging_message = gates.promote_to_staging(version, metrics)
    
    results = {
        'model_version': version,
        'staging_success': staging_success,
        'staging_message': staging_message
    }
    
    # Promote to production if requested and staging succeeded
    if promote_to_prod and staging_success:
        prod_success, prod_message = gates.promote_to_production(version)
        results['production_success'] = prod_success
        results['production_message'] = prod_message
    
    return results
