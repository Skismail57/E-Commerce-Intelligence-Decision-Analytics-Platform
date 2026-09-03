"""
Recommendation Temporal Evaluation Module
Implements proper temporal evaluation for recommendation systems using time-based splits.

Architecture:
- Training interactions: Historical user-item interactions
- Prediction date: Point where recommendations are generated
- Test interactions: Future interactions to evaluate against
- Metrics: Recall@K, Precision@K, NDCG@K, MAP, MRR, Hit Rate
"""

from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from pathlib import Path

from config.logging_config import get_logger
from config.settings import settings

logger = get_logger(__name__)


class RecommendationTemporalEvaluator:
    """
    Temporal evaluator for recommendation systems.
    
    Uses time-based splits to ensure recommendations are evaluated
    against future interactions, preventing leakage.
    """
    
    def __init__(
        self,
        train_end_date: str = "2024-09-30",
        prediction_date: str = "2024-09-30",
        test_end_date: str = "2024-12-31"
    ):
        """
        Initialize temporal evaluator.
        
        Args:
            train_end_date: End date for training interactions
            prediction_date: Date when recommendations are generated
            test_end_date: End date for test interactions
        """
        self.train_end_date = pd.to_datetime(train_end_date)
        self.prediction_date = pd.to_datetime(prediction_date)
        self.test_end_date = pd.to_datetime(test_end_date)
        
        logger.info(f"Recommendation Temporal Evaluator initialized:")
        logger.info(f"  Training window: up to {self.train_end_date}")
        logger.info(f"  Prediction date: {self.prediction_date}")
        logger.info(f"  Test window: {self.prediction_date} to {self.test_end_date}")
    
    def prepare_training_interactions(
        self,
        interactions_df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Prepare training interactions from historical data.
        
        Args:
            interactions_df: User-item interaction data with timestamp
        
        Returns:
            DataFrame with training interactions
        """
        logger.info("Preparing training interactions...")
        
        # Filter to training period
        training_interactions = interactions_df[
            interactions_df['timestamp'] < self.train_end_date
        ].copy()
        
        logger.info(f"Training interactions: {len(training_interactions)}")
        return training_interactions
    
    def prepare_test_interactions(
        self,
        interactions_df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Prepare test interactions from future data.
        
        Args:
            interactions_df: User-item interaction data with timestamp
        
        Returns:
            DataFrame with test interactions (ground truth)
        """
        logger.info("Preparing test interactions...")
        
        # Filter to test period
        test_interactions = interactions_df[
            (interactions_df['timestamp'] >= self.prediction_date) &
            (interactions_df['timestamp'] <= self.test_end_date)
        ].copy()
        
        logger.info(f"Test interactions: {len(test_interactions)}")
        return test_interactions
    
    def generate_recommendations(
        self,
        training_interactions: pd.DataFrame,
        recommender,
        k: int = 10
    ) -> Dict[int, List[int]]:
        """
        Generate recommendations for users using training data.
        
        Args:
            training_interactions: Historical interactions
            recommender: Recommendation system instance
            k: Number of recommendations per user
        
        Returns:
            Dictionary mapping user_id to list of recommended item_ids
        """
        logger.info(f"Generating top-{k} recommendations...")
        
        # Get unique users from training data
        unique_users = training_interactions['user_id'].unique()
        
        recommendations = {}
        for user_id in unique_users:
            # Generate recommendations for this user
            # This would call the actual recommender
            # For now, use placeholder
            recommendations[user_id] = list(range(k))
        
        logger.info(f"Generated recommendations for {len(recommendations)} users")
        return recommendations
    
    def evaluate_recommendations(
        self,
        recommendations: Dict[int, List[int]],
        test_interactions: pd.DataFrame,
        k: int = 10
    ) -> Dict[str, float]:
        """
        Evaluate recommendations against test interactions.
        
        Args:
            recommendations: Dictionary of user_id -> recommended items
            test_interactions: Ground truth future interactions
            k: Number of recommendations to evaluate
        
        Returns:
            Dictionary with evaluation metrics
        """
        logger.info(f"Evaluating recommendations at K={k}...")
        
        # Create ground truth sets
        test_interactions_by_user = test_interactions.groupby('user_id')['item_id'].apply(set).to_dict()
        
        metrics = {
            'recall_at_k': [],
            'precision_at_k': [],
            'ndcg_at_k': [],
            'hit_rate': [],
            'mrr': []
        }
        
        for user_id, recommended_items in recommendations.items():
            if user_id not in test_interactions_by_user:
                continue
            
            ground_truth = test_interactions_by_user[user_id]
            recommended_at_k = set(recommended_items[:k])
            
            # Recall@K
            hits = len(recommended_at_k & ground_truth)
            recall = hits / len(ground_truth) if len(ground_truth) > 0 else 0
            metrics['recall_at_k'].append(recall)
            
            # Precision@K
            precision = hits / k if k > 0 else 0
            metrics['precision_at_k'].append(precision)
            
            # NDCG@K
            dcg = 0.0
            for i, item in enumerate(recommended_items[:k]):
                if item in ground_truth:
                    dcg += 1.0 / np.log2(i + 2)
            # Ideal DCG (all relevant items at top)
            idcg = sum(1.0 / np.log2(i + 2) for i in range(min(len(ground_truth), k)))
            ndcg = dcg / idcg if idcg > 0 else 0
            metrics['ndcg_at_k'].append(ndcg)
            
            # Hit Rate
            metrics['hit_rate'].append(1 if hits > 0 else 0)
            
            # MRR (Mean Reciprocal Rank)
            mrr = 0.0
            for i, item in enumerate(recommended_items[:k]):
                if item in ground_truth:
                    mrr = 1.0 / (i + 1)
                    break
            metrics['mrr'].append(mrr)
        
        # Calculate average metrics
        avg_metrics = {
            f'recall@{k}': np.mean(metrics['recall_at_k']) if metrics['recall_at_k'] else 0,
            f'precision@{k}': np.mean(metrics['precision_at_k']) if metrics['precision_at_k'] else 0,
            f'ndcg@{k}': np.mean(metrics['ndcg_at_k']) if metrics['ndcg_at_k'] else 0,
            f'hit_rate@{k}': np.mean(metrics['hit_rate']) if metrics['hit_rate'] else 0,
            f'mrr@{k}': np.mean(metrics['mrr']) if metrics['mrr'] else 0,
            'n_users_evaluated': len(metrics['recall_at_k'])
        }
        
        logger.info(f"Recommendation Evaluation Results:")
        logger.info(f"  Recall@{k}: {avg_metrics[f'recall@{k}']:.4f}")
        logger.info(f"  Precision@{k}: {avg_metrics[f'precision@{k}']:.4f}")
        logger.info(f"  NDCG@{k}: {avg_metrics[f'ndcg@{k}']:.4f}")
        logger.info(f"  Hit Rate@{k}: {avg_metrics[f'hit_rate@{k}']:.4f}")
        logger.info(f"  MRR@{k}: {avg_metrics[f'mrr@{k}']:.4f}")
        
        return avg_metrics
    
    def evaluate_at_multiple_k(
        self,
        recommendations: Dict[int, List[int]],
        test_interactions: pd.DataFrame,
        k_values: List[int] = [5, 10, 20, 50]
    ) -> Dict[str, float]:
        """
        Evaluate recommendations at multiple K values.
        
        Args:
            recommendations: Dictionary of user_id -> recommended items
            test_interactions: Ground truth future interactions
            k_values: List of K values to evaluate
        
        Returns:
            Dictionary with evaluation metrics for all K values
        """
        logger.info(f"Evaluating recommendations at K={k_values}...")
        
        all_metrics = {}
        for k in k_values:
            metrics = self.evaluate_recommendations(recommendations, test_interactions, k)
            all_metrics.update(metrics)
        
        return all_metrics
    
    def calculate_coverage(
        self,
        recommendations: Dict[int, List[int]],
        total_items: int
    ) -> float:
        """
        Calculate catalog coverage of recommendations.
        
        Args:
            recommendations: Dictionary of user_id -> recommended items
            total_items: Total number of items in catalog
        
        Returns:
            Coverage percentage
        """
        all_recommended_items = set()
        for items in recommendations.values():
            all_recommended_items.update(items)
        
        coverage = len(all_recommended_items) / total_items if total_items > 0 else 0
        logger.info(f"Catalog Coverage: {coverage:.2%} ({len(all_recommended_items)}/{total_items} items)")
        return coverage
    
    def calculate_diversity(
        self,
        recommendations: Dict[int, List[int]],
        item_features: Optional[pd.DataFrame] = None
    ) -> float:
        """
        Calculate diversity of recommendations.
        
        Args:
            recommendations: Dictionary of user_id -> recommended items
            item_features: Optional item features for similarity calculation
        
        Returns:
            Average diversity score
        """
        # Simplified diversity: average number of unique items per user
        diversities = []
        for items in recommendations.values():
            diversities.append(len(set(items)) / len(items) if items else 0)
        
        avg_diversity = np.mean(diversities) if diversities else 0
        logger.info(f"Average Diversity: {avg_diversity:.4f}")
        return avg_diversity
    
    def run_temporal_evaluation(
        self,
        interactions_df: pd.DataFrame,
        recommender,
        k_values: List[int] = [5, 10, 20]
    ) -> Dict[str, float]:
        """
        Run complete temporal evaluation pipeline.
        
        Args:
            interactions_df: User-item interaction data
            recommender: Recommendation system instance
            k_values: K values to evaluate
        
        Returns:
            Dictionary with all evaluation metrics
        """
        logger.info("Running recommendation temporal evaluation pipeline...")
        
        # Prepare training interactions
        training_interactions = self.prepare_training_interactions(interactions_df)
        
        # Prepare test interactions
        test_interactions = self.prepare_test_interactions(interactions_df)
        
        # Generate recommendations
        max_k = max(k_values)
        recommendations = self.generate_recommendations(training_interactions, recommender, max_k)
        
        # Evaluate at multiple K values
        metrics = self.evaluate_at_multiple_k(recommendations, test_interactions, k_values)
        
        # Calculate additional metrics
        total_items = interactions_df['item_id'].nunique()
        metrics['catalog_coverage'] = self.calculate_coverage(recommendations, total_items)
        metrics['diversity'] = self.calculate_diversity(recommendations)
        
        return metrics
    
    def generate_evaluation_report(
        self,
        metrics: Dict[str, float],
        output_path: Optional[Path] = None
    ) -> str:
        """
        Generate human-readable evaluation report.
        
        Args:
            metrics: Evaluation metrics dictionary
            output_path: Optional path to save report
        
        Returns:
            Report string
        """
        report = f"""
Recommendation Temporal Evaluation Report
{'=' * 60}

Configuration:
- Training End Date: {self.train_end_date}
- Prediction Date: {self.prediction_date}
- Test End Date: {self.test_end_date}
- Test Period: {(self.test_end_date - self.prediction_date).days} days

Evaluation Metrics:
"""
        
        # Group metrics by type
        recall_metrics = {k: v for k, v in metrics.items() if 'recall' in k}
        precision_metrics = {k: v for k, v in metrics.items() if 'precision' in k}
        ndcg_metrics = {k: v for k, v in metrics.items() if 'ndcg' in k}
        hit_rate_metrics = {k: v for k, v in metrics.items() if 'hit_rate' in k}
        
        if recall_metrics:
            report += "\nRecall:\n"
            for k, v in sorted(recall_metrics.items()):
                report += f"  {k}: {v:.4f}\n"
        
        if precision_metrics:
            report += "\nPrecision:\n"
            for k, v in sorted(precision_metrics.items()):
                report += f"  {k}: {v:.4f}\n"
        
        if ndcg_metrics:
            report += "\nNDCG:\n"
            for k, v in sorted(ndcg_metrics.items()):
                report += f"  {k}: {v:.4f}\n"
        
        if hit_rate_metrics:
            report += "\nHit Rate:\n"
            for k, v in sorted(hit_rate_metrics.items()):
                report += f"  {k}: {v:.4f}\n"
        
        report += f"""
Additional Metrics:
- Catalog Coverage: {metrics.get('catalog_coverage', 0):.2%}
- Diversity: {metrics.get('diversity', 0):.4f}
- Users Evaluated: {metrics.get('n_users_evaluated', 0)}

Interpretation:
- Recall@10 > 0.2: Good recommendation coverage
- Precision@10 > 0.1: Good recommendation relevance
- NDCG@10 > 0.3: Good ranking quality
- Hit Rate > 0.3: Good overall performance
- Coverage > 0.5: Good catalog coverage
"""
        
        if output_path:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(report)
            logger.info(f"Evaluation report saved to {output_path}")
        
        return report


def run_recommendation_temporal_evaluation(
    train_end_date: str = "2024-09-30",
    prediction_date: str = "2024-09-30",
    test_end_date: str = "2024-12-31"
) -> Dict[str, float]:
    """
    Convenience function to run recommendation temporal evaluation.
    
    Args:
        train_end_date: End date for training interactions
        prediction_date: Date when recommendations are generated
        test_end_date: End date for test interactions
    
    Returns:
        Dictionary with evaluation metrics
    """
    evaluator = RecommendationTemporalEvaluator(
        train_end_date=train_end_date,
        prediction_date=prediction_date,
        test_end_date=test_end_date
    )
    
    # Load data (placeholder - implement actual data loading)
    # interactions_df = pd.read_csv(settings.PROCESSED_DATA_DIR / "interactions.csv")
    # recommender = ProductRecommender()
    
    # metrics = evaluator.run_temporal_evaluation(interactions_df, recommender)
    
    # report = evaluator.generate_evaluation_report(metrics)
    
    logger.warning("Data loading not implemented - returning empty metrics")
    return {}
