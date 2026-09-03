"""
MLflow Pipeline Integration Module
Integrates MLflow for experiment tracking, model management, and pipeline orchestration.

Architecture:
- MLflow experiment tracking for all ML models
- Pipeline orchestration with MLflow Projects
- Model versioning and registry integration
- Hyperparameter tracking
- Artifact logging (models, plots, metrics)
"""

from typing import Dict, List, Optional, Any
import pandas as pd
import numpy as np
from pathlib import Path
import mlflow
import mlflow.sklearn
import mlflow.xgboost
from datetime import datetime

from config.logging_config import get_logger
from config.settings import settings

logger = get_logger(__name__)


class MLflowPipeline:
    """
    MLflow pipeline integration for experiment tracking and model management.
    
    Provides:
    - Experiment tracking
    - Model logging
    - Hyperparameter tracking
    - Artifact management
    - Pipeline orchestration
    """
    
    def __init__(self, experiment_name: str = "ecommerce_analytics"):
        """
        Initialize MLflow pipeline.
        
        Args:
            experiment_name: Name of the MLflow experiment
        """
        self.experiment_name = experiment_name
        self.experiment_id = None
        self.run_id = None
        
        # Set MLflow tracking URI
        mlflow.set_tracking_uri(settings.MLFLOW_TRACKING_URI if hasattr(settings, 'MLFLOW_TRACKING_URI') else "file:///mlruns")
        
        # Create or get experiment
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
            self.experiment_id = None
        
        mlflow.set_experiment(experiment_name)
    
    def start_run(self, run_name: Optional[str] = None) -> str:
        """
        Start a new MLflow run.
        
        Args:
            run_name: Optional name for the run
        
        Returns:
            Run ID
        """
        if run_name is None:
            run_name = f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        mlflow.start_run(run_name=run_name)
        self.run_id = mlflow.active_run().info.run_id
        
        logger.info(f"Started MLflow run: {self.run_id}")
        return self.run_id
    
    def end_run(self, status: str = "FINISHED") -> None:
        """
        End the current MLflow run.
        
        Args:
            status: Run status (FINISHED, FAILED, KILLED)
        """
        if mlflow.active_run():
            mlflow.end_run(status=status)
            logger.info(f"Ended MLflow run with status: {status}")
    
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
            step: Optional step number
        """
        for key, value in metrics.items():
            mlflow.log_metric(key, value, step=step)
        logger.info(f"Logged {len(metrics)} metrics")
    
    def log_artifact(self, artifact_path: str, artifact: Optional[Any] = None) -> None:
        """
        Log an artifact to the current run.
        
        Args:
            artifact_path: Path to the artifact
            artifact: Optional artifact object (e.g., model, dataframe)
        """
        if artifact is not None:
            mlflow.log_artifact(artifact_path)
        else:
            mlflow.log_artifact(artifact_path)
        logger.info(f"Logged artifact: {artifact_path}")
    
    def log_model(
        self,
        model: Any,
        model_name: str,
        model_type: str = "sklearn"
    ) -> None:
        """
        Log a model to the current run.
        
        Args:
            model: Trained model object
            model_name: Name for the model
            model_type: Type of model (sklearn, xgboost, etc.)
        """
        if model_type == "sklearn":
            mlflow.sklearn.log_model(model, model_name)
        elif model_type == "xgboost":
            mlflow.xgboost.log_model(model, model_name)
        else:
            mlflow.log_model(model, model_name)
        
        logger.info(f"Logged model: {model_name} ({model_type})")
    
    def log_dataframe(self, df: pd.DataFrame, artifact_name: str) -> None:
        """
        Log a dataframe as an artifact.
        
        Args:
            df: DataFrame to log
            artifact_name: Name for the artifact
        """
        temp_path = f"/tmp/{artifact_name}.csv"
        df.to_csv(temp_path, index=False)
        mlflow.log_artifact(temp_path, artifact_name)
        logger.info(f"Logged dataframe: {artifact_name}")
    
    def log_figure(self, figure, artifact_name: str) -> None:
        """
        Log a matplotlib figure as an artifact.
        
        Args:
            figure: Matplotlib figure object
            artifact_name: Name for the artifact
        """
        temp_path = f"/tmp/{artifact_name}.png"
        figure.savefig(temp_path)
        mlflow.log_artifact(temp_path, artifact_name)
        logger.info(f"Logged figure: {artifact_name}")
    
    def track_churn_model(
        self,
        model: Any,
        params: Dict[str, Any],
        metrics: Dict[str, float],
        feature_importance: Optional[pd.DataFrame] = None
    ) -> str:
        """
        Track a churn model training run.
        
        Args:
            model: Trained churn model
            params: Model hyperparameters
            metrics: Evaluation metrics
            feature_importance: Optional feature importance dataframe
        
        Returns:
            Run ID
        """
        run_name = f"churn_model_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.start_run(run_name)
        
        # Log parameters
        self.log_params(params)
        
        # Log metrics
        self.log_metrics(metrics)
        
        # Log model
        self.log_model(model, "churn_model", "sklearn")
        
        # Log feature importance if provided
        if feature_importance is not None:
            self.log_dataframe(feature_importance, "feature_importance")
        
        self.end_run()
        
        return self.run_id
    
    def track_forecast_model(
        self,
        model: Any,
        params: Dict[str, Any],
        metrics: Dict[str, float],
        forecast_plot: Optional[Any] = None
    ) -> str:
        """
        Track a forecast model training run.
        
        Args:
            model: Trained forecast model
            params: Model hyperparameters
            metrics: Evaluation metrics
            forecast_plot: Optional forecast plot
        
        Returns:
            Run ID
        """
        run_name = f"forecast_model_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.start_run(run_name)
        
        # Log parameters
        self.log_params(params)
        
        # Log metrics
        self.log_metrics(metrics)
        
        # Log model
        self.log_model(model, "forecast_model", "sklearn")
        
        # Log forecast plot if provided
        if forecast_plot is not None:
            self.log_figure(forecast_plot, "forecast_plot")
        
        self.end_run()
        
        return self.run_id
    
    def track_clv_model(
        self,
        params: Dict[str, Any],
        metrics: Dict[str, float],
        clv_predictions: pd.DataFrame
    ) -> str:
        """
        Track a CLV model training run.
        
        Args:
            params: Model parameters
            metrics: Evaluation metrics
            clv_predictions: CLV predictions dataframe
        
        Returns:
            Run ID
        """
        run_name = f"clv_model_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.start_run(run_name)
        
        # Log parameters
        self.log_params(params)
        
        # Log metrics
        self.log_metrics(metrics)
        
        # Log predictions
        self.log_dataframe(clv_predictions, "clv_predictions")
        
        self.end_run()
        
        return self.run_id
    
    def compare_runs(
        self,
        run_ids: List[str],
        metric_name: str
    ) -> pd.DataFrame:
        """
        Compare multiple runs by a specific metric.
        
        Args:
            run_ids: List of run IDs to compare
            metric_name: Metric to compare
        
        Returns:
            DataFrame with run comparison
        """
        comparison_data = []
        
        for run_id in run_ids:
            run = mlflow.get_run(run_id)
            metrics = run.data.metrics
            params = run.data.params
            
            comparison_data.append({
                'run_id': run_id,
                'run_name': run.data.tags.get('mlflow.runName', 'unknown'),
                metric_name: metrics.get(metric_name, None),
                **params
            })
        
        comparison_df = pd.DataFrame(comparison_data)
        
        # Sort by metric (descending)
        if metric_name in comparison_df.columns:
            comparison_df = comparison_df.sort_values(metric_name, ascending=False)
        
        logger.info(f"Compared {len(run_ids)} runs by {metric_name}")
        return comparison_df
    
    def get_best_run(
        self,
        metric_name: str,
        ascending: bool = False
    ) -> Optional[str]:
        """
        Get the best run for a given metric.
        
        Args:
            metric_name: Metric to optimize
            ascending: Whether lower is better
        
        Returns:
            Best run ID
        """
        runs = mlflow.search_runs(
            experiment_ids=[self.experiment_id],
            order_by=[f"metrics.{metric_name} {'ASC' if ascending else 'DESC'}"]
        )
        
        if len(runs) > 0:
            best_run_id = runs.iloc[0]['run_id']
            logger.info(f"Best run for {metric_name}: {best_run_id}")
            return best_run_id
        else:
            logger.warning(f"No runs found for experiment {self.experiment_name}")
            return None
    
    def load_model(self, run_id: str, model_name: str = "model"):
        """
        Load a model from a run.
        
        Args:
            run_id: Run ID
            model_name: Name of the logged model
        
        Returns:
            Loaded model
        """
        model_uri = f"runs:/{run_id}/{model_name}"
        model = mlflow.sklearn.load_model(model_uri)
        
        logger.info(f"Loaded model from run {run_id}")
        return model


class PipelineOrchestrator:
    """
    Orchestrates ML pipelines with MLflow tracking.
    
    Coordinates multiple ML pipeline steps with automatic tracking.
    """
    
    def __init__(self, pipeline_name: str):
        """
        Initialize pipeline orchestrator.
        
        Args:
            pipeline_name: Name of the pipeline
        """
        self.pipeline_name = pipeline_name
        self.mlflow = MLflowPipeline(experiment_name=pipeline_name)
        self.steps = []
    
    def add_step(self, step_name: str, step_func: callable) -> None:
        """
        Add a step to the pipeline.
        
        Args:
            step_name: Name of the step
            step_func: Function to execute for this step
        """
        self.steps.append({
            'name': step_name,
            'func': step_func
        })
        logger.info(f"Added step: {step_name}")
    
    def run_pipeline(self, **kwargs) -> Dict[str, Any]:
        """
        Run the complete pipeline.
        
        Args:
            **kwargs: Arguments to pass to pipeline steps
        
        Returns:
            Dictionary with pipeline results
        """
        logger.info(f"Running pipeline: {self.pipeline_name}")
        
        self.mlflow.start_run(f"pipeline_{self.pipeline_name}")
        
        results = {}
        
        for step in self.steps:
            step_name = step['name']
            step_func = step['func']
            
            logger.info(f"Executing step: {step_name}")
            
            try:
                # Execute step
                step_result = step_func(**kwargs)
                results[step_name] = step_result
                
                # Log step metrics if available
                if isinstance(step_result, dict) and 'metrics' in step_result:
                    self.mlflow.log_metrics(step_result['metrics'])
                
                logger.info(f"Step completed: {step_name}")
                
            except Exception as e:
                logger.error(f"Step failed: {step_name} - {e}")
                self.mlflow.end_run(status="FAILED")
                raise
        
        self.mlflow.end_run()
        
        logger.info(f"Pipeline completed: {self.pipeline_name}")
        return results


def run_mlflow_pipeline(
    pipeline_name: str,
    steps: List[Tuple[str, callable]],
    **kwargs
) -> Dict[str, Any]:
    """
    Convenience function to run an MLflow pipeline.
    
    Args:
        pipeline_name: Name of the pipeline
        steps: List of (step_name, step_func) tuples
        **kwargs: Arguments to pass to pipeline steps
    
    Returns:
        Dictionary with pipeline results
    """
    orchestrator = PipelineOrchestrator(pipeline_name)
    
    for step_name, step_func in steps:
        orchestrator.add_step(step_name, step_func)
    
    results = orchestrator.run_pipeline(**kwargs)
    
    return results
