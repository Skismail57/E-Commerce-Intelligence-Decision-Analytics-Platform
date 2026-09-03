"""
Cost-Sensitive Decision Making Module
Implements cost-sensitive learning for optimizing business decisions.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from config.logging_config import get_logger

logger = get_logger(__name__)


class CostSensitiveLearner:
    """
    Cost-sensitive learning engine for optimizing business decisions.
    
    Features:
    - Cost matrix definition
    - Cost-sensitive model training
    - Threshold optimization for cost minimization
    - Expected cost calculation
    - Cost-benefit analysis
    """
    
    def __init__(self):
        """Initialize cost-sensitive learner"""
        self.cost_matrix = {}
        self.cost_sensitive_models = {}
        self.optimal_thresholds = {}
        logger.info("Cost-sensitive learner initialized")
    
    def define_cost_matrix(
        self,
        tp_cost: float = 0,
        tn_cost: float = 0,
        fp_cost: float = 1,
        fn_cost: float = 5
    ) -> Dict:
        """
        Define cost matrix for classification.
        
        Args:
            tp_cost: Cost of true positive
            tn_cost: Cost of true negative
            fp_cost: Cost of false positive
            fn_cost: Cost of false negative
        
        Returns:
            Dictionary with cost matrix
        """
        self.cost_matrix = {
            'TP': tp_cost,
            'TN': tn_cost,
            'FP': fp_cost,
            'FN': fn_cost
        }
        
        logger.info(f"Cost matrix defined: TP={tp_cost}, TN={tn_cost}, FP={fp_cost}, FN={fn_cost}")
        
        return self.cost_matrix
    
    def calculate_expected_cost(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray
    ) -> float:
        """
        Calculate expected cost of predictions.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
        
        Returns:
            Expected cost
        """
        if not self.cost_matrix:
            raise ValueError("Cost matrix not defined. Call define_cost_matrix first.")
        
        # Calculate confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        
        # Extract values
        tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)
        
        # Calculate expected cost
        expected_cost = (
            tp * self.cost_matrix['TP'] +
            tn * self.cost_matrix['TN'] +
            fp * self.cost_matrix['FP'] +
            fn * self.cost_matrix['FN']
        )
        
        return float(expected_cost)
    
    def find_optimal_threshold(
        self,
        y_true: np.ndarray,
        y_prob: np.ndarray,
        threshold_range: Tuple[float, float] = (0.0, 1.0),
        n_steps: int = 100
    ) -> Dict:
        """
        Find optimal decision threshold to minimize cost.
        
        Args:
            y_true: True labels
            y_prob: Predicted probabilities
            threshold_range: Range of thresholds to search
            n_steps: Number of steps in search
        
        Returns:
            Dictionary with optimal threshold and cost
        """
        if not self.cost_matrix:
            raise ValueError("Cost matrix not defined. Call define_cost_matrix first.")
        
        logger.info("Finding optimal threshold...")
        
        best_threshold = 0.5
        best_cost = float('inf')
        
        # Search for optimal threshold
        for threshold in np.linspace(threshold_range[0], threshold_range[1], n_steps):
            y_pred = (y_prob >= threshold).astype(int)
            cost = self.calculate_expected_cost(y_true, y_pred)
            
            if cost < best_cost:
                best_cost = cost
                best_threshold = threshold
        
        result = {
            'optimal_threshold': best_threshold,
            'min_cost': best_cost,
            'cost_matrix': self.cost_matrix
        }
        
        logger.info(f"Optimal threshold found: {best_threshold:.3f} with cost {best_cost:.2f}")
        
        return result
    
    def train_cost_sensitive_model(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        model_type: str = 'random_forest',
        class_weights: Dict[int, float] = None
    ) -> Any:
        """
        Train a cost-sensitive model using class weights.
        
        Args:
            X_train: Training features
            y_train: Training labels
            model_type: Type of model to train
            class_weights: Class weights (optional)
        
        Returns:
            Trained model
        """
        logger.info(f"Training cost-sensitive {model_type} model...")
        
        # Calculate class weights from cost matrix if not provided
        if class_weights is None and self.cost_matrix:
            # Weight classes inversely proportional to their misclassification costs
            fn_cost = self.cost_matrix.get('FN', 1)
            fp_cost = self.cost_matrix.get('FP', 1)
            
            # Higher cost for FN means higher weight for positive class
            class_weights = {0: fp_cost, 1: fn_cost}
        
        # Initialize model
        if model_type == 'random_forest':
            model = RandomForestClassifier(
                class_weight=class_weights,
                random_state=42,
                n_estimators=100
            )
        elif model_type == 'gradient_boosting':
            model = GradientBoostingClassifier(
                random_state=42,
                n_estimators=100
            )
        elif model_type == 'logistic_regression':
            model = LogisticRegression(
                class_weight=class_weights,
                random_state=42,
                max_iter=1000
            )
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        # Train model
        model.fit(X_train, y_train)
        
        self.cost_sensitive_models[model_type] = model
        
        logger.info(f"Cost-sensitive {model_type} model trained")
        
        return model
    
    def evaluate_cost_sensitive_model(
        self,
        model: Any,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        threshold: float = 0.5
    ) -> Dict:
        """
        Evaluate cost-sensitive model.
        
        Args:
            model: Trained model
            X_test: Test features
            y_test: Test labels
            threshold: Decision threshold
        
        Returns:
            Dictionary with evaluation results
        """
        logger.info("Evaluating cost-sensitive model...")
        
        # Get predictions
        y_prob = model.predict_proba(X_test)[:, 1]
        y_pred = (y_prob >= threshold).astype(int)
        
        # Calculate confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)
        
        # Calculate metrics
        results = {
            'threshold': threshold,
            'true_positives': int(tp),
            'true_negatives': int(tn),
            'false_positives': int(fp),
            'false_negatives': int(fn),
            'accuracy': float((tp + tn) / (tp + tn + fp + fn)) if (tp + tn + fp + fn) > 0 else 0,
            'precision': float(tp / (tp + fp)) if (tp + fp) > 0 else 0,
            'recall': float(tp / (tp + fn)) if (tp + fn) > 0 else 0
        }
        
        # Calculate cost if cost matrix defined
        if self.cost_matrix:
            expected_cost = self.calculate_expected_cost(y_test, y_pred)
            results['expected_cost'] = expected_cost
        
        logger.info(f"Model evaluation complete. Expected cost: {results.get('expected_cost', 'N/A')}")
        
        return results
    
    def compare_thresholds(
        self,
        y_true: np.ndarray,
        y_prob: np.ndarray,
        thresholds: List[float] = None
    ) -> pd.DataFrame:
        """
        Compare costs at different thresholds.
        
        Args:
            y_true: True labels
            y_prob: Predicted probabilities
            thresholds: List of thresholds to test
        
        Returns:
            DataFrame with threshold comparison
        """
        if not self.cost_matrix:
            raise ValueError("Cost matrix not defined. Call define_cost_matrix first.")
        
        if thresholds is None:
            thresholds = np.linspace(0.1, 0.9, 9)
        
        results = []
        
        for threshold in thresholds:
            y_pred = (y_prob >= threshold).astype(int)
            cost = self.calculate_expected_cost(y_true, y_pred)
            
            # Calculate confusion matrix
            cm = confusion_matrix(y_true, y_pred)
            tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)
            
            results.append({
                'threshold': threshold,
                'cost': cost,
                'tp': int(tp),
                'tn': int(tn),
                'fp': int(fp),
                'fn': int(fn)
            })
        
        results_df = pd.DataFrame(results)
        
        logger.info(f"Threshold comparison complete for {len(thresholds)} thresholds")
        
        return results_df
    
    def cost_benefit_analysis(
        self,
        y_true: np.ndarray,
        y_prob: np.ndarray,
        benefit_tp: float = 100,
        benefit_tn: float = 0,
        cost_fp: float = 10,
        cost_fn: float = 50
    ) -> Dict:
        """
        Perform cost-benefit analysis.
        
        Args:
            y_true: True labels
            y_prob: Predicted probabilities
            benefit_tp: Benefit of true positive
            benefit_tn: Benefit of true negative
            cost_fp: Cost of false positive
            cost_fn: Cost of false negative
        
        Returns:
            Dictionary with cost-benefit analysis
        """
        logger.info("Performing cost-benefit analysis...")
        
        # Define cost-benefit matrix
        self.cost_matrix = {
            'TP': -benefit_tp,  # Negative because it's a benefit
            'TN': -benefit_tn,
            'FP': cost_fp,
            'FN': cost_fn
        }
        
        # Find optimal threshold
        optimal_result = self.find_optimal_threshold(y_true, y_prob)
        
        # Calculate net benefit at optimal threshold
        y_pred_optimal = (y_prob >= optimal_result['optimal_threshold']).astype(int)
        cm = confusion_matrix(y_true, y_pred_optimal)
        tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)
        
        net_benefit = (
            tp * benefit_tp +
            tn * benefit_tn -
            fp * cost_fp -
            fn * cost_fn
        )
        
        results = {
            'optimal_threshold': optimal_result['optimal_threshold'],
            'net_benefit': float(net_benefit),
            'benefit_tp': benefit_tp,
            'benefit_tn': benefit_tn,
            'cost_fp': cost_fp,
            'cost_fn': cost_fn,
            'confusion_matrix': {
                'tp': int(tp),
                'tn': int(tn),
                'fp': int(fp),
                'fn': int(fn)
            }
        }
        
        logger.info(f"Cost-benefit analysis complete. Net benefit: ${net_benefit:.2f}")
        
        return results


def run_cost_sensitive_pipeline(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    tp_cost: float = 0,
    tn_cost: float = 0,
    fp_cost: float = 1,
    fn_cost: float = 5
) -> Tuple[CostSensitiveLearner, Dict]:
    """
    Convenience function to run cost-sensitive pipeline.
    
    Args:
        X_train: Training features
        y_train: Training labels
        X_test: Test features
        y_test: Test labels
        tp_cost: Cost of true positive
        tn_cost: Cost of true negative
        fp_cost: Cost of false positive
        fn_cost: Cost of false negative
    
    Returns:
        Tuple of (learner, results)
    """
    learner = CostSensitiveLearner()
    
    # Define cost matrix
    learner.define_cost_matrix(tp_cost, tn_cost, fp_cost, fn_cost)
    
    # Train cost-sensitive model
    model = learner.train_cost_sensitive_model(X_train, y_train)
    
    # Get predictions
    y_prob = model.predict_proba(X_test)[:, 1]
    
    # Find optimal threshold
    optimal_threshold = learner.find_optimal_threshold(y_test.values, y_prob)
    
    # Evaluate at optimal threshold
    evaluation = learner.evaluate_cost_sensitive_model(
        model, X_test, y_test, optimal_threshold['optimal_threshold']
    )
    
    # Compare thresholds
    threshold_comparison = learner.compare_thresholds(y_test.values, y_prob)
    
    results = {
        'cost_matrix': learner.cost_matrix,
        'optimal_threshold': optimal_threshold,
        'evaluation': evaluation,
        'threshold_comparison': threshold_comparison
    }
    
    return learner, results
