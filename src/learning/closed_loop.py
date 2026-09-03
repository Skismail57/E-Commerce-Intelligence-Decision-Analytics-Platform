"""
Closed-Loop Learning System Module
Implements closed-loop learning system for continuous model improvement.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Callable
from datetime import datetime, timedelta
from config.logging_config import get_logger

logger = get_logger(__name__)


class ClosedLoopLearningSystem:
    """
    Closed-loop learning system for continuous model improvement.
    
    Features:
    - Feedback collection and storage
    - Prediction-outcome tracking
    - Performance monitoring
    - Automatic retraining triggers
    - Continuous improvement
    - Learning from mistakes
    """
    
    def __init__(self):
        """Initialize closed-loop learning system"""
        self.feedback_data = []
        self.predictions_log = []
        self.performance_history = {}
        self.retraining_history = []
        self.learning_metrics = {}
        logger.info("Closed-loop learning system initialized")
    
    def log_prediction(
        self,
        model_id: str,
        prediction: Any,
        features: Dict,
        timestamp: datetime = None
    ) -> Dict:
        """
        Log a model prediction.
        
        Args:
            model_id: Model identifier
            prediction: Model prediction
            features: Input features
            timestamp: Prediction timestamp
        
        Returns:
            Dictionary with prediction log
        """
        if timestamp is None:
            timestamp = datetime.now()
        
        prediction_log = {
            'model_id': model_id,
            'prediction': prediction,
            'features': features,
            'timestamp': timestamp.isoformat(),
            'outcome': None,
            'feedback_received': False
        }
        
        self.predictions_log.append(prediction_log)
        
        logger.info(f"Prediction logged for model {model_id}")
        
        return prediction_log
    
    def record_feedback(
        self,
        prediction_id: int,
        actual_outcome: Any,
        feedback_type: str = 'explicit'
    ) -> Dict:
        """
        Record feedback for a prediction.
        
        Args:
            prediction_id: ID of the prediction
            actual_outcome: Actual outcome
            feedback_type: Type of feedback (explicit, implicit)
        
        Returns:
            Dictionary with feedback record
        """
        if prediction_id >= len(self.predictions_log):
            raise ValueError(f"Prediction ID {prediction_id} not found")
        
        prediction_log = self.predictions_log[prediction_id]
        prediction_log['outcome'] = actual_outcome
        prediction_log['feedback_received'] = True
        prediction_log['feedback_type'] = feedback_type
        prediction_log['feedback_timestamp'] = datetime.now().isoformat()
        
        # Calculate feedback metrics
        prediction = prediction_log['prediction']
        
        # For classification
        if isinstance(prediction, (int, float)) and prediction in [0, 1]:
            is_correct = (prediction == actual_outcome)
            feedback_metrics = {
                'is_correct': is_correct,
                'prediction': prediction,
                'actual': actual_outcome
            }
        else:
            # For regression or other types
            try:
                error = abs(prediction - actual_outcome)
                feedback_metrics = {
                    'error': float(error),
                    'prediction': prediction,
                    'actual': actual_outcome
                }
            except:
                feedback_metrics = {
                    'prediction': prediction,
                    'actual': actual_outcome
                }
        
        feedback_record = {
            'prediction_id': prediction_id,
            'model_id': prediction_log['model_id'],
            'actual_outcome': actual_outcome,
            'feedback_type': feedback_type,
            'feedback_metrics': feedback_metrics,
            'timestamp': datetime.now().isoformat()
        }
        
        self.feedback_data.append(feedback_record)
        
        logger.info(f"Feedback recorded for prediction {prediction_id}")
        
        return feedback_record
    
    def calculate_learning_metrics(
        self,
        model_id: str,
        window_size: int = 100
    ) -> Dict:
        """
        Calculate learning metrics for a model.
        
        Args:
            model_id: Model identifier
            window_size: Window size for metrics calculation
        
        Returns:
            Dictionary with learning metrics
        """
        logger.info(f"Calculating learning metrics for model {model_id}")
        
        # Get recent feedback for this model
        model_feedback = [
            f for f in self.feedback_data
            if f['model_id'] == model_id
        ][-window_size:]
        
        if not model_feedback:
            return {'error': 'No feedback data available'}
        
        # Calculate metrics
        n_feedback = len(model_feedback)
        
        # Calculate accuracy if classification
        if 'is_correct' in model_feedback[0]['feedback_metrics']:
            correct_count = sum(
                1 for f in model_feedback
                if f['feedback_metrics'].get('is_correct', False)
            )
            accuracy = correct_count / n_feedback if n_feedback > 0 else 0
            
            metrics = {
                'accuracy': float(accuracy),
                'n_correct': correct_count,
                'n_total': n_feedback
            }
        else:
            # Calculate error metrics for regression
            errors = [
                f['feedback_metrics'].get('error', 0)
                for f in model_feedback
                if 'error' in f['feedback_metrics']
            ]
            
            if errors:
                metrics = {
                    'mean_error': float(np.mean(errors)),
                    'max_error': float(np.max(errors)),
                    'min_error': float(np.min(errors)),
                    'std_error': float(np.std(errors))
                }
            else:
                metrics = {}
        
        # Calculate feedback rate
        model_predictions = [
            p for p in self.predictions_log
            if p['model_id'] == model_id
        ]
        feedback_rate = n_feedback / len(model_predictions) if model_predictions else 0
        
        metrics.update({
            'n_feedback': n_feedback,
            'feedback_rate': float(feedback_rate),
            'window_size': window_size,
            'calculated_at': datetime.now().isoformat()
        })
        
        self.learning_metrics[model_id] = metrics
        
        logger.info(f"Learning metrics calculated for {model_id}")
        
        return metrics
    
    def detect_performance_degradation(
        self,
        model_id: str,
        threshold: float = 0.1
    ) -> Dict:
        """
        Detect if model performance has degraded.
        
        Args:
            model_id: Model identifier
            threshold: Degradation threshold
        
        Returns:
            Dictionary with degradation detection result
        """
        logger.info(f"Detecting performance degradation for {model_id}")
        
        if model_id not in self.learning_metrics:
            metrics = self.calculate_learning_metrics(model_id)
        else:
            metrics = self.learning_metrics[model_id]
        
        if 'error' in metrics:
            return {'error': 'Could not calculate metrics'}
        
        # Get historical performance
        if model_id not in self.performance_history:
            self.performance_history[model_id] = []
        
        history = self.performance_history[model_id]
        
        if len(history) < 2:
            # Not enough history
            current_metric = metrics.get('accuracy', 1 - metrics.get('mean_error', 0))
            self.performance_history[model_id].append({
                'timestamp': datetime.now().isoformat(),
                'metric': current_metric
            })
            return {
                'is_degraded': False,
                'reason': 'Insufficient history'
            }
        
        # Compare with previous performance
        previous_metric = history[-1]['metric']
        current_metric = metrics.get('accuracy', 1 - metrics.get('mean_error', 0))
        
        degradation = (previous_metric - current_metric) / previous_metric if previous_metric > 0 else 0
        is_degraded = degradation > threshold
        
        # Update history
        self.performance_history[model_id].append({
            'timestamp': datetime.now().isoformat(),
            'metric': current_metric
        })
        
        result = {
            'model_id': model_id,
            'previous_metric': float(previous_metric),
            'current_metric': float(current_metric),
            'degradation': float(degradation),
            'is_degraded': is_degraded,
            'threshold': threshold
        }
        
        if is_degraded:
            logger.warning(f"Performance degradation detected for {model_id}: {degradation:.1%}")
        
        return result
    
    def trigger_retraining(
        self,
        model_id: str,
        retraining_function: Callable,
        training_data: pd.DataFrame,
        target_col: str
    ) -> Dict:
        """
        Trigger model retraining.
        
        Args:
            model_id: Model identifier
            retraining_function: Function to retrain the model
            training_data: Training data
            target_col: Target column name
        
        Returns:
            Dictionary with retraining result
        """
        logger.info(f"Triggering retraining for model {model_id}")
        
        try:
            # Retrain model
            new_model = retraining_function(training_data, target_col)
            
            # Record retraining
            retraining_record = {
                'model_id': model_id,
                'timestamp': datetime.now().isoformat(),
                'status': 'success',
                'training_samples': len(training_data)
            }
            
            self.retraining_history.append(retraining_record)
            
            logger.info(f"Retraining successful for model {model_id}")
            
            return {
                'success': True,
                'model_id': model_id,
                'new_model': new_model,
                'retraining_record': retraining_record
            }
        
        except Exception as e:
            logger.error(f"Retraining failed for model {model_id}: {e}")
            
            retraining_record = {
                'model_id': model_id,
                'timestamp': datetime.now().isoformat(),
                'status': 'failed',
                'error': str(e)
            }
            
            self.retraining_history.append(retraining_record)
            
            return {
                'success': False,
                'model_id': model_id,
                'error': str(e)
            }
    
    def analyze_mistakes(
        self,
        model_id: str,
        n_mistakes: int = 10
    ) -> List[Dict]:
        """
        Analyze common mistakes made by the model.
        
        Args:
            model_id: Model identifier
            n_mistakes: Number of mistakes to analyze
        
        Returns:
            List of mistake analyses
        """
        logger.info(f"Analyzing mistakes for model {model_id}")
        
        # Get incorrect predictions
        mistakes = []
        
        for i, prediction_log in enumerate(self.predictions_log):
            if prediction_log['model_id'] == model_id and prediction_log['feedback_received']:
                feedback_metrics = self.feedback_data[i]['feedback_metrics'] if i < len(self.feedback_data) else {}
                
                if not feedback_metrics.get('is_correct', True):
                    mistakes.append({
                        'prediction_id': i,
                        'prediction': prediction_log['prediction'],
                        'actual': prediction_log['outcome'],
                        'features': prediction_log['features']
                    })
        
        # Analyze common patterns
        mistake_analysis = []
        
        if mistakes:
            # Get feature importance from mistakes
            feature_errors = {}
            
            for mistake in mistakes[:n_mistakes]:
                for feature, value in mistake['features'].items():
                    if feature not in feature_errors:
                        feature_errors[feature] = []
                    feature_errors[feature].append(value)
            
            # Calculate statistics for each feature
            for feature, values in feature_errors.items():
                if values:
                    mistake_analysis.append({
                        'feature': feature,
                        'avg_value': float(np.mean(values)),
                        'std_value': float(np.std(values)),
                        'n_occurrences': len(values)
                    })
        
        logger.info(f"Mistake analysis complete for {model_id}")
        
        return mistake_analysis
    
    def generate_learning_report(self) -> Dict:
        """
        Generate comprehensive learning report.
        
        Returns:
            Dictionary with learning report
        """
        logger.info("Generating learning report...")
        
        report = {
            'n_predictions': len(self.predictions_log),
            'n_feedback': len(self.feedback_data),
            'feedback_rate': float(len(self.feedback_data) / len(self.predictions_log)) if self.predictions_log else 0,
            'n_models': len(set(p['model_id'] for p in self.predictions_log)),
            'n_retrainings': len(self.retraining_history),
            'learning_metrics': self.learning_metrics,
            'retraining_history': self.retraining_history[-5:] if self.retraining_history else [],
            'generated_at': datetime.now().isoformat()
        }
        
        logger.info("Learning report generated")
        
        return report
    
    def run_learning_cycle(
        self,
        model_id: str,
        retraining_function: Callable,
        training_data: pd.DataFrame,
        target_col: str,
        degradation_threshold: float = 0.1
    ) -> Dict:
        """
        Run a complete learning cycle.
        
        Args:
            model_id: Model identifier
            retraining_function: Function to retrain the model
            training_data: Training data
            target_col: Target column name
            degradation_threshold: Threshold for triggering retraining
        
        Returns:
            Dictionary with cycle results
        """
        logger.info(f"Running learning cycle for model {model_id}")
        
        # Calculate learning metrics
        metrics = self.calculate_learning_metrics(model_id)
        
        # Detect degradation
        degradation = self.detect_performance_degradation(model_id, degradation_threshold)
        
        # Trigger retraining if degraded
        retraining_result = None
        if degradation.get('is_degraded', False):
            retraining_result = self.trigger_retraining(
                model_id, retraining_function, training_data, target_col
            )
        
        # Analyze mistakes
        mistakes = self.analyze_mistakes(model_id)
        
        # Generate report
        report = self.generate_learning_report()
        
        cycle_results = {
            'model_id': model_id,
            'learning_metrics': metrics,
            'degradation_detection': degradation,
            'retraining_result': retraining_result,
            'mistake_analysis': mistakes,
            'learning_report': report
        }
        
        logger.info(f"Learning cycle complete for {model_id}")
        
        return cycle_results


def run_closed_loop_pipeline(
    model_id: str,
    predictions: List[Any],
    features_list: List[Dict],
    outcomes: List[Any],
    retraining_function: Callable = None
) -> Tuple[ClosedLoopLearningSystem, Dict]:
    """
    Convenience function to run closed-loop learning pipeline.
    
    Args:
        model_id: Model identifier
        predictions: List of predictions
        features_list: List of feature dictionaries
        outcomes: List of actual outcomes
        retraining_function: Optional retraining function
    
    Returns:
        Tuple of (system, results)
    """
    system = ClosedLoopLearningSystem()
    
    # Log predictions
    prediction_ids = []
    for prediction, features in zip(predictions, features_list):
        log = system.log_prediction(model_id, prediction, features)
        prediction_ids.append(len(system.predictions_log) - 1)
    
    # Record feedback
    for pred_id, outcome in zip(prediction_ids, outcomes):
        system.record_feedback(pred_id, outcome)
    
    # Calculate learning metrics
    metrics = system.calculate_learning_metrics(model_id)
    
    # Detect degradation
    degradation = system.detect_performance_degradation(model_id)
    
    # Analyze mistakes
    mistakes = system.analyze_mistakes(model_id)
    
    # Generate report
    report = system.generate_learning_report()
    
    results = {
        'learning_metrics': metrics,
        'degradation_detection': degradation,
        'mistake_analysis': mistakes,
        'learning_report': report
    }
    
    return system, results
