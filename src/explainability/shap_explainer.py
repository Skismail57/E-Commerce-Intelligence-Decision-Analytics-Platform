"""
Explainable AI (SHAP) Module
Implements SHAP-based model explainability for interpretability.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from config.logging_config import get_logger

logger = get_logger(__name__)


class SHAPExplainer:
    """
    SHAP-based model explainer for interpretability.
    
    Features:
    - Feature importance calculation
    - SHAP value computation
    - Local and global explanations
    - Feature contribution analysis
    - Interaction effects
    """
    
    def __init__(self):
        """Initialize SHAP explainer"""
        self.explainer = None
        self.shap_values = None
        self.feature_importance = {}
        logger.info("SHAP explainer initialized")
    
    def calculate_feature_importance(
        self,
        model: Any,
        X: pd.DataFrame,
        method: str = 'permutation'
    ) -> pd.DataFrame:
        """
        Calculate feature importance using various methods.
        
        Args:
            model: Trained model
            X: Feature DataFrame
            method: Method for importance calculation ('permutation', 'gain', 'shap')
        
        Returns:
            DataFrame with feature importance
        """
        logger.info(f"Calculating feature importance using {method} method...")
        
        if method == 'permutation':
            # Permutation importance
            from sklearn.inspection import permutation_importance
            from sklearn.metrics import accuracy_score
            
            def scorer(model, X, y):
                return accuracy_score(y, model.predict(X))
            
            # Need labels for permutation importance
            # For now, use model's feature_importances_ if available
            if hasattr(model, 'feature_importances_'):
                importance = model.feature_importances_
                feature_names = X.columns.tolist()
            else:
                # Fallback to SHAP-based importance
                importance = self._calculate_shap_importance(model, X)
                feature_names = X.columns.tolist()
        
        elif method == 'gain':
            # Gain importance (for tree-based models)
            if hasattr(model, 'feature_importances_'):
                importance = model.feature_importances_
                feature_names = X.columns.tolist()
            else:
                importance = self._calculate_shap_importance(model, X)
                feature_names = X.columns.tolist()
        
        elif method == 'shap':
            importance = self._calculate_shap_importance(model, X)
            feature_names = X.columns.tolist()
        
        else:
            raise ValueError(f"Unknown method: {method}")
        
        # Create DataFrame
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': importance
        }).sort_values('importance', ascending=False)
        
        self.feature_importance[method] = importance_df
        
        logger.info(f"Feature importance calculated for {len(importance_df)} features")
        
        return importance_df
    
    def _calculate_shap_importance(
        self,
        model: Any,
        X: pd.DataFrame
    ) -> np.ndarray:
        """
        Calculate SHAP-based feature importance.
        
        Args:
            model: Trained model
            X: Feature DataFrame
        
        Returns:
            Array of importance values
        """
        try:
            import shap
            
            # Create explainer based on model type
            if hasattr(model, 'predict_proba'):
                explainer = shap.TreeExplainer(model)
            else:
                explainer = shap.Explainer(model, X)
            
            # Calculate SHAP values
            shap_values = explainer.shap_values(X)
            
            # Calculate mean absolute SHAP values as importance
            if isinstance(shap_values, list):
                # For classification, use the first class
                importance = np.mean(np.abs(shap_values[0]), axis=0)
            else:
                importance = np.mean(np.abs(shap_values), axis=0)
            
            return importance
        
        except ImportError:
            logger.warning("SHAP not installed. Using fallback importance.")
            # Fallback to model's feature_importances_ if available
            if hasattr(model, 'feature_importances_'):
                return model.feature_importances_
            else:
                # Return uniform importance as fallback
                return np.ones(X.shape[1])
    
    def explain_prediction(
        self,
        model: Any,
        X: pd.DataFrame,
        instance_idx: int = 0
    ) -> Dict:
        """
        Explain a single prediction using SHAP.
        
        Args:
            model: Trained model
            X: Feature DataFrame
            instance_idx: Index of instance to explain
        
        Returns:
            Dictionary with explanation
        """
        logger.info(f"Explaining prediction for instance {instance_idx}...")
        
        try:
            import shap
            
            # Create explainer
            if hasattr(model, 'predict_proba'):
                explainer = shap.TreeExplainer(model)
            else:
                explainer = shap.Explainer(model, X)
            
            # Calculate SHAP values
            shap_values = explainer.shap_values(X)
            
            # Get instance SHAP values
            if isinstance(shap_values, list):
                instance_shap = shap_values[0][instance_idx]
            else:
                instance_shap = shap_values[instance_idx]
            
            # Get base value
            base_value = explainer.expected_value
            if isinstance(base_value, list):
                base_value = base_value[0]
            
            # Get feature values
            instance_features = X.iloc[instance_idx].to_dict()
            
            # Calculate feature contributions
            contributions = {}
            for i, feature in enumerate(X.columns):
                contributions[feature] = {
                    'shap_value': float(instance_shap[i]),
                    'feature_value': float(instance_features[feature])
                }
            
            # Sort by absolute SHAP value
            sorted_contributions = sorted(
                contributions.items(),
                key=lambda x: abs(x[1]['shap_value']),
                reverse=True
            )
            
            result = {
                'instance_idx': instance_idx,
                'base_value': float(base_value),
                'prediction': float(model.predict(X.iloc[[instance_idx]])[0]),
                'feature_contributions': dict(sorted_contributions),
                'top_positive_features': [
                    (f, v) for f, v in sorted_contributions if v['shap_value'] > 0
                ],
                'top_negative_features': [
                    (f, v) for f, v in sorted_contributions if v['shap_value'] < 0
                ]
            }
            
            logger.info(f"Prediction explanation complete")
            
            return result
        
        except ImportError:
            logger.warning("SHAP not installed. Returning basic explanation.")
            return {
                'instance_idx': instance_idx,
                'prediction': float(model.predict(X.iloc[[instance_idx]])[0]),
                'note': 'SHAP not installed. Install shap package for detailed explanations.'
            }
    
    def explain_global(
        self,
        model: Any,
        X: pd.DataFrame
    ) -> Dict:
        """
        Generate global model explanation.
        
        Args:
            model: Trained model
            X: Feature DataFrame
        
        Returns:
            Dictionary with global explanation
        """
        logger.info("Generating global model explanation...")
        
        # Calculate feature importance
        importance_df = self.calculate_feature_importance(model, X, method='shap')
        
        # Calculate SHAP values for all instances
        try:
            import shap
            
            if hasattr(model, 'predict_proba'):
                explainer = shap.TreeExplainer(model)
            else:
                explainer = shap.Explainer(model, X)
            
            shap_values = explainer.shap_values(X)
            
            if isinstance(shap_values, list):
                shap_values = shap_values[0]
            
            # Calculate summary statistics
            mean_abs_shap = np.mean(np.abs(shap_values), axis=0)
            std_shap = np.std(shap_values, axis=0)
            
            summary_stats = []
            for i, feature in enumerate(X.columns):
                summary_stats.append({
                    'feature': feature,
                    'mean_abs_shap': float(mean_abs_shap[i]),
                    'std_shap': float(std_shap[i]),
                    'importance_rank': int(i + 1)
                })
            
            summary_df = pd.DataFrame(summary_stats).sort_values('mean_abs_shap', ascending=False)
            
        except ImportError:
            summary_df = importance_df.rename(columns={'importance': 'mean_abs_shap'})
        
        result = {
            'feature_importance': importance_df.to_dict('records'),
            'summary_statistics': summary_df.to_dict('records'),
            'n_features': len(X.columns),
            'n_instances': len(X)
        }
        
        logger.info(f"Global explanation complete for {len(X.columns)} features")
        
        return result
    
    def analyze_feature_interactions(
        self,
        model: Any,
        X: pd.DataFrame,
        max_features: int = 10
    ) -> pd.DataFrame:
        """
        Analyze feature interactions using SHAP interaction values.
        
        Args:
            model: Trained model
            X: Feature DataFrame
            max_features: Maximum number of features to analyze
        
        Returns:
            DataFrame with interaction analysis
        """
        logger.info("Analyzing feature interactions...")
        
        try:
            import shap
            
            # Create explainer
            if hasattr(model, 'predict_proba'):
                explainer = shap.TreeExplainer(model)
            else:
                explainer = shap.Explainer(model, X)
            
            # Calculate SHAP interaction values
            shap_interaction_values = explainer.shap_interaction_values(X)
            
            if isinstance(shap_interaction_values, list):
                shap_interaction_values = shap_interaction_values[0]
            
            # Calculate mean absolute interaction values
            mean_abs_interaction = np.mean(np.abs(shap_interaction_values), axis=0)
            
            # Get top features
            top_features = X.columns[:max_features].tolist()
            
            # Create interaction matrix
            interactions = []
            for i, feat1 in enumerate(top_features):
                for j, feat2 in enumerate(top_features):
                    if i < j:  # Avoid duplicates
                        interactions.append({
                            'feature1': feat1,
                            'feature2': feat2,
                            'interaction_strength': float(mean_abs_interaction[i, j])
                        })
            
            interaction_df = pd.DataFrame(interactions).sort_values(
                'interaction_strength', ascending=False
            )
            
            logger.info(f"Feature interaction analysis complete")
            
            return interaction_df
        
        except ImportError:
            logger.warning("SHAP not installed. Cannot analyze interactions.")
            return pd.DataFrame()
    
    def generate_explanation_report(
        self,
        model: Any,
        X: pd.DataFrame,
        y: pd.Series = None,
        n_instances: int = 5
    ) -> Dict:
        """
        Generate comprehensive explanation report.
        
        Args:
            model: Trained model
            X: Feature DataFrame
            y: Target labels (optional)
            n_instances: Number of instances to explain
        
        Returns:
            Dictionary with explanation report
        """
        logger.info("Generating comprehensive explanation report...")
        
        # Global explanation
        global_explanation = self.explain_global(model, X)
        
        # Local explanations for sample instances
        local_explanations = []
        for i in range(min(n_instances, len(X))):
            local_explanation = self.explain_prediction(model, X, i)
            local_explanations.append(local_explanation)
        
        # Feature interaction analysis
        interaction_analysis = self.analyze_feature_interactions(model, X)
        
        # Model performance if labels provided
        performance = None
        if y is not None:
            from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
            y_pred = model.predict(X)
            performance = {
                'accuracy': float(accuracy_score(y, y_pred)),
                'precision': float(precision_score(y, y_pred, average='weighted')),
                'recall': float(recall_score(y, y_pred, average='weighted')),
                'f1_score': float(f1_score(y, y_pred, average='weighted'))
            }
        
        report = {
            'global_explanation': global_explanation,
            'local_explanations': local_explanations,
            'feature_interactions': interaction_analysis.to_dict('records') if not interaction_analysis.empty else [],
            'model_performance': performance,
            'n_instances_explained': len(local_explanations)
        }
        
        logger.info(f"Explanation report generated")
        
        return report


def run_shap_pipeline(
    model: Any,
    X: pd.DataFrame,
    y: pd.Series = None
) -> Tuple[SHAPExplainer, Dict]:
    """
    Convenience function to run SHAP explanation pipeline.
    
    Args:
        model: Trained model
        X: Feature DataFrame
        y: Target labels (optional)
    
    Returns:
        Tuple of (explainer, results)
    """
    explainer = SHAPExplainer()
    
    # Generate comprehensive report
    report = explainer.generate_explanation_report(model, X, y)
    
    results = {
        'explanation_report': report
    }
    
    return explainer, results
