"""
MLflow Tracking for MLOps
Provides experiment tracking, model registry, and model versioning for ML models.
"""

from pathlib import Path
from typing import Dict, Optional, Any
import mlflow
import mlflow.sklearn
import mlflow.pyfunc
import pandas as pd
import numpy as np
import json
from datetime import datetime

from config.logging_config import get_logger

logger = get_logger(__name__)


class MLflowTracker:
    """
    MLflow tracker for experiment tracking and model registry.
    
    Features:
    - Experiment tracking for all ML models
    - Model versioning and registry
    - Hyperparameter logging
    - Metric tracking
    - Artifact management
    """
    
    def __init__(
        self,
        experiment_name: str = "ecommerce_analytics",
        tracking_uri: Optional[str] = None,
        model_registry_uri: Optional[str] = None
    ):
        """
        Initialize MLflow tracker.
        
        Args:
            experiment_name: Name of the MLflow experiment
            tracking_uri: MLflow tracking server URI (default: local file://mlruns)
            model_registry_uri: MLflow model registry URI
        """
        self.experiment_name = experiment_name
        self.tracking_uri = tracking_uri or "file://mlruns"
        self.model_registry_uri = model_registry_uri
        
        # Set tracking URI
        mlflow.set_tracking_uri(self.tracking_uri)
        
        # Set or create experiment
        try:
            self.experiment = mlflow.get_experiment_by_name(experiment_name)
            if self.experiment is None:
                self.experiment_id = mlflow.create_experiment(experiment_name)
                logger.info(f"Created new experiment: {experiment_name}")
            else:
                self.experiment_id = self.experiment.experiment_id
                logger.info(f"Using existing experiment: {experiment_name}")
        except Exception as e:
            logger.error(f"Error setting up MLflow experiment: {e}")
            raise
    
    def start_run(self, run_name: Optional[str] = None) -> mlflow.ActiveRun:
        """
        Start a new MLflow run.
        
        Args:
            run_name: Name for the run
        
        Returns:
            Active MLflow run
        """
        if run_name:
            run = mlflow.start_run(run_name=run_name, experiment_id=self.experiment_id)
        else:
            run = mlflow.start_run(experiment_id=self.experiment_id)
        
        logger.info(f"Started MLflow run: {run.info.run_id}")
        return run
    
    def log_params(self, params: Dict[str, Any]) -> None:
        """
        Log parameters to the current run.
        
        Args:
            params: Dictionary of parameters
        """
        for key, value in params.items():
            mlflow.log_param(key, value)
        logger.info(f"Logged {len(params)} parameters")
    
    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None) -> None:
        """
        Log metrics to the current run.
        
        Args:
            metrics: Dictionary of metrics
            step: Step number for metrics
        """
        for key, value in metrics.items():
            mlflow.log_metric(key, value, step=step)
        logger.info(f"Logged {len(metrics)} metrics")
    
    def log_model(
        self,
        model: Any,
        model_name: str,
        artifact_path: str = "model",
        registered_model_name: Optional[str] = None,
        **kwargs
    ) -> None:
        """
        Log a model to the current run.
        
        Args:
            model: Trained model object
            model_name: Name for the model
            artifact_path: Path within the run's artifact directory
            registered_model_name: Name for registered model (optional)
            **kwargs: Additional arguments for log_model
        """
        mlflow.sklearn.log_model(
            model,
            artifact_path=artifact_path,
            registered_model_name=registered_model_name,
            **kwargs
        )
        logger.info(f"Logged model: {model_name}")
    
    def log_artifact(self, file_path: str, artifact_path: Optional[str] = None) -> None:
        """
        Log an artifact (file) to the current run.
        
        Args:
            file_path: Path to the file
            artifact_path: Path within the run's artifact directory
        """
        mlflow.log_artifact(file_path, artifact_path)
        logger.info(f"Logged artifact: {file_path}")
    
    def log_dataframe(self, df: pd.DataFrame, artifact_path: str) -> None:
        """
        Log a DataFrame as a CSV artifact.
        
        Args:
            df: DataFrame to log
            artifact_path: Path within the run's artifact directory
        """
        temp_path = f"temp_{artifact_path}.csv"
        df.to_csv(temp_path, index=False)
        mlflow.log_artifact(temp_path, artifact_path)
        Path(temp_path).unlink()
        logger.info(f"Logged DataFrame: {artifact_path}")
    
    def log_dict(self, data: Dict[str, Any], artifact_path: str) -> None:
        """
        Log a dictionary as a JSON artifact.
        
        Args:
            data: Dictionary to log
            artifact_path: Path within the run's artifact directory
        """
        temp_path = f"temp_{artifact_path}.json"
        with open(temp_path, 'w') as f:
            json.dump(data, f, indent=2, default=str)
        mlflow.log_artifact(temp_path, artifact_path)
        Path(temp_path).unlink()
        logger.info(f"Logged dictionary: {artifact_path}")
    
    def end_run(self, status: str = "FINISHED") -> None:
        """
        End the current MLflow run.
        
        Args:
            status: Run status (FINISHED, FAILED, KILLED)
        """
        mlflow.end_run(status=status)
        logger.info(f"Ended MLflow run with status: {status}")
    
    def register_model(
        self,
        model_name: str,
        model_path: str,
        description: Optional[str] = None
    ) -> mlflow.entities.model_registry.RegisteredModel:
        """
        Register a model in the MLflow Model Registry.
        
        Args:
            model_name: Name for the registered model
            model_path: Path to the model artifact
            description: Model description
        
        Returns:
            Registered model
        """
        model_uri = f"runs:/{mlflow.active_run().info.run_id}/{model_path}"
        
        try:
            model_version = mlflow.register_model(
                model_uri,
                model_name,
                description=description
            )
            logger.info(f"Registered model: {model_name} version {model_version.version}")
            return model_version
        except Exception as e:
            logger.error(f"Error registering model: {e}")
            raise
    
    def load_model(self, model_name: str, model_version: Optional[str] = None, stage: Optional[str] = None):
        """
        Load a model from the MLflow Model Registry.
        
        Args:
            model_name: Name of the registered model
            model_version: Specific version to load
            stage: Stage to load from (Production, Staging, None)
        
        Returns:
            Loaded model
        """
        if model_version:
            model_uri = f"models:/{model_name}/{model_version}"
        elif stage:
            model_uri = f"models:/{model_name}/{stage}"
        else:
            model_uri = f"models:/{model_name}"
        
        model = mlflow.sklearn.load_model(model_uri)
        logger.info(f"Loaded model: {model_uri}")
        return model
    
    def transition_model_stage(
        self,
        model_name: str,
        version: str,
        stage: str,
        archive_existing_versions: bool = False
    ) -> None:
        """
        Transition a model version to a new stage.
        
        Args:
            model_name: Name of the registered model
            version: Model version
            stage: Target stage (Production, Staging, Archived)
            archive_existing_versions: Whether to archive existing versions in the target stage
        """
        from mlflow.tracking import MlflowClient
        client = MlflowClient()
        
        client.transition_model_version_stage(
            name=model_name,
            version=version,
            stage=stage,
            archive_existing_versions=archive_existing_versions
        )
        logger.info(f"Transitioned model {model_name} v{version} to {stage}")
    
    def get_model_info(self, model_name: str) -> Dict[str, Any]:
        """
        Get information about a registered model.
        
        Args:
            model_name: Name of the registered model
        
        Returns:
            Dictionary with model information
        """
        from mlflow.tracking import MlflowClient
        client = MlflowClient()
        
        model = client.get_registered_model(model_name)
        
        versions = client.get_latest_versions(model_name, stages=["Production", "Staging", "Archived", "None"])
        
        return {
            "name": model.name,
            "creation_timestamp": model.creation_timestamp,
            "last_updated_timestamp": model.last_updated_timestamp,
            "description": model.description,
            "versions": [
                {
                    "version": v.version,
                    "stage": v.current_stage,
                    "run_id": v.run_id,
                    "creation_timestamp": v.creation_timestamp
                }
                for v in versions
            ]
        }
    
    def search_runs(
        self,
        filter_string: str = "",
        max_results: int = 100,
        order_by: Optional[list] = None
    ) -> list:
        """
        Search for runs in the experiment.
        
        Args:
            filter_string: Filter string for search
            max_results: Maximum number of results
            order_by: Order by clause
        
        Returns:
            List of runs
        """
        runs = mlflow.search_runs(
            experiment_ids=[self.experiment_id],
            filter_string=filter_string,
            max_results=max_results,
            order_by=order_by
        )
        return runs
    
    def delete_run(self, run_id: str) -> None:
        """
        Delete a run.
        
        Args:
            run_id: Run ID to delete
        """
        mlflow.delete_run(run_id)
        logger.info(f"Deleted run: {run_id}")
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        if exc_type is not None:
            self.end_run(status="FAILED")
        else:
            self.end_run(status="FINISHED")


# Convenience functions for common MLflow operations

def track_churn_model(
    model: Any,
    params: Dict[str, Any],
    metrics: Dict[str, float],
    feature_importance: pd.DataFrame,
    model_name: str = "churn_predictor",
    run_name: Optional[str] = None
) -> str:
    """
    Track a churn prediction model with MLflow.
    
    Args:
        model: Trained model
        params: Model hyperparameters
        metrics: Model metrics
        feature_importance: Feature importance DataFrame
        model_name: Name for the model
        run_name: Name for the run
    
    Returns:
        Run ID
    """
    tracker = MLflowTracker(experiment_name="churn_prediction")
    
    with tracker.start_run(run_name=run_name):
        tracker.log_params(params)
        tracker.log_metrics(metrics)
        tracker.log_model(model, model_name, registered_model_name=model_name)
        tracker.log_dataframe(feature_importance, "feature_importance")
        
        run_id = mlflow.active_run().info.run_id
    
    return run_id


def track_forecasting_model(
    model: Any,
    params: Dict[str, Any],
    metrics: Dict[str, float],
    forecast_df: pd.DataFrame,
    model_name: str = "demand_forecaster",
    run_name: Optional[str] = None
) -> str:
    """
    Track a forecasting model with MLflow.
    
    Args:
        model: Trained model (or model info for non-sklearn models)
        params: Model hyperparameters
        metrics: Model metrics
        forecast_df: Forecast results DataFrame
        model_name: Name for the model
        run_name: Name for the run
    
    Returns:
        Run ID
    """
    tracker = MLflowTracker(experiment_name="demand_forecasting")
    
    with tracker.start_run(run_name=run_name):
        tracker.log_params(params)
        tracker.log_metrics(metrics)
        tracker.log_dataframe(forecast_df, "forecast_results")
        
        # For non-sklearn models, log as artifact
        if not hasattr(model, 'predict'):
            tracker.log_dict({"model_type": str(type(model))}, "model_info")
        else:
            tracker.log_model(model, model_name, registered_model_name=model_name)
        
        run_id = mlflow.active_run().info.run_id
    
    return run_id


def track_clv_model(
    params: Dict[str, Any],
    metrics: Dict[str, float],
    clv_predictions: pd.DataFrame,
    model_name: str = "clv_predictor",
    run_name: Optional[str] = None
) -> str:
    tracker = MLflowTracker(experiment_name="clv_prediction")
    with tracker.start_run(run_name=run_name):
        tracker.log_params(params)
        tracker.log_metrics(metrics)
        tracker.log_dataframe(clv_predictions, "clv_predictions")
        run_id = mlflow.active_run().info.run_id
    return run_id
