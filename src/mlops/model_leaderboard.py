"""
Model Leaderboard Module
Tracks and compares model performance across experiments.

Architecture:
- Leaderboard for all model types (churn, forecast, CLV, recommendation)
- Performance comparison and ranking
- Model metadata tracking
- Best model selection
- Performance trends over time
"""

from typing import Dict, List, Optional, Any
import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path
import mlflow
from mlflow.tracking import MlflowClient

from config.logging_config import get_logger
from config.settings import settings

logger = get_logger(__name__)


class ModelLeaderboard:
    """
    Model leaderboard for tracking and comparing model performance.
    
    Maintains rankings for different model types and provides
    tools for model selection and comparison.
    """
    
    def __init__(self, experiment_name: str = "ecommerce_analytics"):
        """
        Initialize model leaderboard.
        
        Args:
            experiment_name: MLflow experiment name
        """
        self.experiment_name = experiment_name
        self.client = MlflowClient()
        self.leaderboard_data = {}
        
        logger.info(f"Model Leaderboard initialized for: {experiment_name}")
    
    def refresh_leaderboard(self) -> None:
        """Refresh leaderboard data from MLflow."""
        logger.info("Refreshing leaderboard data...")
        
        # Get all runs from experiment
        runs = self.client.search_runs(
            experiment_ids=[self.client.get_experiment_by_name(self.experiment_name).experiment_id]
        )
        
        # Organize by model type
        self.leaderboard_data = {
            'churn': [],
            'forecast': [],
            'clv': [],
            'recommendation': [],
            'all': []
        }
        
        for run in runs:
            run_data = {
                'run_id': run.info.run_id,
                'run_name': run.data.tags.get('mlflow.runName', 'unknown'),
                'start_time': datetime.fromtimestamp(run.info.start_time / 1000),
                'status': run.info.status,
                'metrics': run.data.metrics,
                'params': run.data.params
            }
            
            # Categorize by model type based on run name or tags
            run_name_lower = run_data['run_name'].lower()
            
            if 'churn' in run_name_lower:
                self.leaderboard_data['churn'].append(run_data)
            elif 'forecast' in run_name_lower:
                self.leaderboard_data['forecast'].append(run_data)
            elif 'clv' in run_name_lower:
                self.leaderboard_data['clv'].append(run_data)
            elif 'recommendation' in run_name_lower:
                self.leaderboard_data['recommendation'].append(run_data)
            
            self.leaderboard_data['all'].append(run_data)
        
        logger.info(f"Leaderboard refreshed: {len(self.leaderboard_data['all'])} total runs")
    
    def get_churn_leaderboard(self, metric: str = 'f1_score') -> pd.DataFrame:
        """
        Get churn model leaderboard sorted by metric.
        
        Args:
            metric: Metric to sort by (default: f1_score)
        
        Returns:
            DataFrame with churn model rankings
        """
        if not self.leaderboard_data.get('churn'):
            return pd.DataFrame()
        
        leaderboard = []
        
        for run in self.leaderboard_data['churn']:
            leaderboard.append({
                'run_id': run['run_id'],
                'run_name': run['run_name'],
                'start_time': run['start_time'],
                'status': run['status'],
                'accuracy': run['metrics'].get('accuracy', None),
                'precision': run['metrics'].get('precision', None),
                'recall': run['metrics'].get('recall', None),
                'f1_score': run['metrics'].get('f1_score', None),
                'auc_roc': run['metrics'].get('auc_roc', None),
                **run['params']
            })
        
        df = pd.DataFrame(leaderboard)
        
        if metric in df.columns:
            df = df.sort_values(metric, ascending=False)
        
        df['rank'] = range(1, len(df) + 1)
        
        return df
    
    def get_forecast_leaderboard(self, metric: str = 'rmse') -> pd.DataFrame:
        """
        Get forecast model leaderboard sorted by metric.
        
        Args:
            metric: Metric to sort by (default: rmse, lower is better)
        
        Returns:
            DataFrame with forecast model rankings
        """
        if not self.leaderboard_data.get('forecast'):
            return pd.DataFrame()
        
        leaderboard = []
        
        for run in self.leaderboard_data['forecast']:
            leaderboard.append({
                'run_id': run['run_id'],
                'run_name': run['run_name'],
                'start_time': run['start_time'],
                'status': run['status'],
                'mae': run['metrics'].get('mae', None),
                'rmse': run['metrics'].get('rmse', None),
                'mape': run['metrics'].get('mape', None),
                'rmsse': run['metrics'].get('rmsse', None),
                **run['params']
            })
        
        df = pd.DataFrame(leaderboard)
        
        if metric in df.columns:
            # For forecast metrics, lower is usually better
            df = df.sort_values(metric, ascending=True)
        
        df['rank'] = range(1, len(df) + 1)
        
        return df
    
    def get_clv_leaderboard(self, metric: str = 'r_squared') -> pd.DataFrame:
        """
        Get CLV model leaderboard sorted by metric.
        
        Args:
            metric: Metric to sort by (default: r_squared)
        
        Returns:
            DataFrame with CLV model rankings
        """
        if not self.leaderboard_data.get('clv'):
            return pd.DataFrame()
        
        leaderboard = []
        
        for run in self.leaderboard_data['clv']:
            leaderboard.append({
                'run_id': run['run_id'],
                'run_name': run['run_name'],
                'start_time': run['start_time'],
                'status': run['status'],
                'mae': run['metrics'].get('mae', None),
                'rmse': run['metrics'].get('rmse', None),
                'r_squared': run['metrics'].get('r_squared', None),
                **run['params']
            })
        
        df = pd.DataFrame(leaderboard)
        
        if metric in df.columns:
            df = df.sort_values(metric, ascending=False)
        
        df['rank'] = range(1, len(df) + 1)
        
        return df
    
    def get_recommendation_leaderboard(self, metric: str = 'ndcg_at_10') -> pd.DataFrame:
        """
        Get recommendation model leaderboard sorted by metric.
        
        Args:
            metric: Metric to sort by (default: ndcg_at_10)
        
        Returns:
            DataFrame with recommendation model rankings
        """
        if not self.leaderboard_data.get('recommendation'):
            return pd.DataFrame()
        
        leaderboard = []
        
        for run in self.leaderboard_data['recommendation']:
            leaderboard.append({
                'run_id': run['run_id'],
                'run_name': run['run_name'],
                'start_time': run['start_time'],
                'status': run['status'],
                'recall_at_10': run['metrics'].get('recall_at_10', None),
                'precision_at_10': run['metrics'].get('precision_at_10', None),
                'ndcg_at_10': run['metrics'].get('ndcg_at_10', None),
                'hit_rate_at_10': run['metrics'].get('hit_rate_at_10', None),
                **run['params']
            })
        
        df = pd.DataFrame(leaderboard)
        
        if metric in df.columns:
            df = df.sort_values(metric, ascending=False)
        
        df['rank'] = range(1, len(df) + 1)
        
        return df
    
    def get_best_model(
        self,
        model_type: str,
        metric: str = None
    ) -> Optional[Dict[str, Any]]:
        """
        Get the best model for a given type.
        
        Args:
            model_type: Type of model (churn, forecast, clv, recommendation)
            metric: Metric to optimize (uses default if None)
        
        Returns:
            Dictionary with best model info or None
        """
        if metric is None:
            metric_defaults = {
                'churn': 'f1_score',
                'forecast': 'rmse',
                'clv': 'r_squared',
                'recommendation': 'ndcg_at_10'
            }
            metric = metric_defaults.get(model_type)
        
        if model_type == 'churn':
            leaderboard = self.get_churn_leaderboard(metric)
        elif model_type == 'forecast':
            leaderboard = self.get_forecast_leaderboard(metric)
        elif model_type == 'clv':
            leaderboard = self.get_clv_leaderboard(metric)
        elif model_type == 'recommendation':
            leaderboard = self.get_recommendation_leaderboard(metric)
        else:
            logger.warning(f"Unknown model type: {model_type}")
            return None
        
        if len(leaderboard) == 0:
            return None
        
        best_model = leaderboard.iloc[0].to_dict()
        
        logger.info(f"Best {model_type} model: {best_model['run_name']} ({metric}={best_model.get(metric, 'N/A')})")
        return best_model
    
    def compare_models(
        self,
        run_ids: List[str],
        metrics: List[str]
    ) -> pd.DataFrame:
        """
        Compare specific models by metrics.
        
        Args:
            run_ids: List of run IDs to compare
            metrics: List of metrics to compare
        
        Returns:
            DataFrame with model comparison
        """
        comparison = []
        
        for run_id in run_ids:
            run = self.client.get_run(run_id)
            
            row = {
                'run_id': run_id,
                'run_name': run.data.tags.get('mlflow.runName', 'unknown')
            }
            
            for metric in metrics:
                row[metric] = run.data.metrics.get(metric, None)
            
            comparison.append(row)
        
        df = pd.DataFrame(comparison)
        
        logger.info(f"Compared {len(run_ids)} models")
        return df
    
    def get_performance_trend(
        self,
        model_type: str,
        metric: str,
        days: int = 30
    ) -> pd.DataFrame:
        """
        Get performance trend over time for a model type.
        
        Args:
            model_type: Type of model
            metric: Metric to track
            days: Number of days to look back
        
        Returns:
            DataFrame with performance trend
        """
        cutoff_date = datetime.now() - timedelta(days=days)
        
        if model_type not in self.leaderboard_data:
            return pd.DataFrame()
        
        trend_data = []
        
        for run in self.leaderboard_data[model_type]:
            if run['start_time'] < cutoff_date:
                continue
            
            metric_value = run['metrics'].get(metric)
            if metric_value is not None:
                trend_data.append({
                    'run_id': run['run_id'],
                    'run_name': run['run_name'],
                    'date': run['start_time'],
                    metric: metric_value
                })
        
        df = pd.DataFrame(trend_data)
        df = df.sort_values('date')
        
        logger.info(f"Performance trend: {len(df)} data points")
        return df
    
    def generate_leaderboard_report(
        self,
        output_path: Optional[Path] = None
    ) -> str:
        """
        Generate comprehensive leaderboard report.
        
        Args:
            output_path: Optional path to save report
        
        Returns:
            Report string
        """
        self.refresh_leaderboard()
        
        report = f"""
Model Leaderboard Report
{'=' * 60}

Experiment: {self.experiment_name}
Generated: {datetime.now().isoformat()}

CHURN MODELS (Ranked by F1 Score):
"""
        
        churn_lb = self.get_churn_leaderboard()
        if len(churn_lb) > 0:
            for _, row in churn_lb.head(5).iterrows():
                report += f"{int(row['rank'])}. {row['run_name']} - F1: {row['f1_score']:.4f}, AUC: {row['auc_roc']:.4f}\n"
        else:
            report += "No churn models found\n"
        
        report += f"""
FORECAST MODELS (Ranked by RMSE):
"""
        
        forecast_lb = self.get_forecast_leaderboard()
        if len(forecast_lb) > 0:
            for _, row in forecast_lb.head(5).iterrows():
                report += f"{int(row['rank'])}. {row['run_name']} - RMSE: {row['rmse']:.4f}, MAE: {row['mae']:.4f}\n"
        else:
            report += "No forecast models found\n"
        
        report += f"""
CLV MODELS (Ranked by R²):
"""
        
        clv_lb = self.get_clv_leaderboard()
        if len(clv_lb) > 0:
            for _, row in clv_lb.head(5).iterrows():
                report += f"{int(row['rank'])}. {row['run_name']} - R²: {row['r_squared']:.4f}, MAE: {row['mae']:.4f}\n"
        else:
            report += "No CLV models found\n"
        
        report += f"""
RECOMMENDATION MODELS (Ranked by NDCG@10):
"""
        
        rec_lb = self.get_recommendation_leaderboard()
        if len(rec_lb) > 0:
            for _, row in rec_lb.head(5).iterrows():
                report += f"{int(row['rank'])}. {row['run_name']} - NDCG: {row['ndcg_at_10']:.4f}, Recall: {row['recall_at_10']:.4f}\n"
        else:
            report += "No recommendation models found\n"
        
        report += f"""
TOTAL RUNS: {len(self.leaderboard_data['all'])}
"""
        
        if output_path:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(report)
            logger.info(f"Leaderboard report saved to {output_path}")
        
        return report


def get_model_leaderboard(experiment_name: str = "ecommerce_analytics") -> ModelLeaderboard:
    """
    Convenience function to get model leaderboard.
    
    Args:
        experiment_name: MLflow experiment name
    
    Returns:
        ModelLeaderboard instance
    """
    leaderboard = ModelLeaderboard(experiment_name)
    leaderboard.refresh_leaderboard()
    return leaderboard
