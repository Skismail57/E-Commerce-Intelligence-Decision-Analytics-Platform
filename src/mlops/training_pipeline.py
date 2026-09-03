"""
ML Training Pipeline with Prefect Module
Implements ML training pipelines using Prefect for orchestration.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from config.logging_config import get_logger

logger = get_logger(__name__)


class MLTrainingPipeline:
    """
    ML training pipeline for model development.
    
    Features:
    - Data preprocessing and feature engineering
    - Model training with multiple algorithms
    - Model evaluation and comparison
    - Hyperparameter tuning
    - Model versioning and tracking
    """
    
    def __init__(self):
        """Initialize ML training pipeline"""
        self.models = {}
        self.model_metrics = {}
        self.training_history = []
        logger.info("ML training pipeline initialized")
    
    def preprocess_data(
        self,
        data: pd.DataFrame,
        target_col: str,
        categorical_cols: List[str] = None,
        numerical_cols: List[str] = None,
        handle_missing: bool = True
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Preprocess data for training.
        
        Args:
            data: Input DataFrame
            target_col: Target column name
            categorical_cols: List of categorical columns
            numerical_cols: List of numerical columns
            handle_missing: Whether to handle missing values
        
        Returns:
            Tuple of (features DataFrame, target Series)
        """
        logger.info("Preprocessing data...")
        
        # Separate features and target
        y = data[target_col]
        X = data.drop(columns=[target_col])
        
        # Handle missing values
        if handle_missing:
            # Fill numerical columns with median
            if numerical_cols:
                for col in numerical_cols:
                    if col in X.columns:
                        X[col] = X[col].fillna(X[col].median())
            
            # Fill categorical columns with mode
            if categorical_cols:
                for col in categorical_cols:
                    if col in X.columns:
                        X[col] = X[col].fillna(X[col].mode()[0] if not X[col].mode().empty else 'missing')
        
        # Encode categorical variables
        if categorical_cols:
            for col in categorical_cols:
                if col in X.columns:
                    X = pd.get_dummies(X, columns=[col], prefix=col, drop_first=True)
        
        # Scale numerical features
        if numerical_cols:
            from sklearn.preprocessing import StandardScaler
            scaler = StandardScaler()
            num_cols_present = [col for col in numerical_cols if col in X.columns]
            if num_cols_present:
                X[num_cols_present] = scaler.fit_transform(X[num_cols_present])
        
        logger.info(f"Data preprocessing complete. Features: {X.shape[1]}, Samples: {X.shape[0]}")
        
        return X, y
    
    def train_model(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        model_type: str = 'random_forest',
        hyperparameters: Dict = None
    ) -> Any:
        """
        Train a machine learning model.
        
        Args:
            X_train: Training features
            y_train: Training target
            model_type: Type of model to train
            hyperparameters: Model hyperparameters
        
        Returns:
            Trained model
        """
        logger.info(f"Training {model_type} model...")
        
        if hyperparameters is None:
            hyperparameters = {}
        
        # Initialize model based on type
        if model_type == 'random_forest':
            model = RandomForestClassifier(
                n_estimators=hyperparameters.get('n_estimators', 100),
                max_depth=hyperparameters.get('max_depth', 10),
                random_state=42
            )
        elif model_type == 'gradient_boosting':
            model = GradientBoostingClassifier(
                n_estimators=hyperparameters.get('n_estimators', 100),
                learning_rate=hyperparameters.get('learning_rate', 0.1),
                max_depth=hyperparameters.get('max_depth', 3),
                random_state=42
            )
        elif model_type == 'logistic_regression':
            model = LogisticRegression(
                C=hyperparameters.get('C', 1.0),
                max_iter=hyperparameters.get('max_iter', 1000),
                random_state=42
            )
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        # Train model
        model.fit(X_train, y_train)
        
        logger.info(f"{model_type} model trained successfully")
        
        return model
    
    def evaluate_model(
        self,
        model: Any,
        X_test: pd.DataFrame,
        y_test: pd.Series
    ) -> Dict:
        """
        Evaluate a trained model.
        
        Args:
            model: Trained model
            X_test: Test features
            y_test: Test target
        
        Returns:
            Dictionary with evaluation metrics
        """
        logger.info("Evaluating model...")
        
        # Make predictions
        y_pred = model.predict(X_test)
        
        # Get prediction probabilities if available
        try:
            y_pred_proba = model.predict_proba(X_test)[:, 1]
            has_proba = True
        except:
            y_pred_proba = None
            has_proba = False
        
        # Calculate metrics
        metrics = {
            'accuracy': float(accuracy_score(y_test, y_pred)),
            'precision': float(precision_score(y_test, y_pred, average='weighted')),
            'recall': float(recall_score(y_test, y_pred, average='weighted')),
            'f1_score': float(f1_score(y_test, y_pred, average='weighted'))
        }
        
        # Add AUC if probabilities available
        if has_proba:
            try:
                metrics['roc_auc'] = float(roc_auc_score(y_test, y_pred_proba))
            except:
                pass
        
        logger.info(f"Model evaluation complete. Accuracy: {metrics['accuracy']:.3f}")
        
        return metrics
    
    def compare_models(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        model_types: List[str] = None
    ) -> pd.DataFrame:
        """
        Compare multiple model types.
        
        Args:
            X_train: Training features
            y_train: Training target
            X_test: Test features
            y_test: Test target
            model_types: List of model types to compare
        
        Returns:
            DataFrame with model comparison
        """
        if model_types is None:
            model_types = ['random_forest', 'gradient_boosting', 'logistic_regression']
        
        logger.info(f"Comparing {len(model_types)} models...")
        
        results = []
        
        for model_type in model_types:
            try:
                # Train model
                model = self.train_model(X_train, y_train, model_type)
                
                # Evaluate model
                metrics = self.evaluate_model(model, X_test, y_test)
                
                # Store model
                self.models[model_type] = model
                self.model_metrics[model_type] = metrics
                
                results.append({
                    'model_type': model_type,
                    **metrics
                })
            except Exception as e:
                logger.error(f"Error training {model_type}: {e}")
                results.append({
                    'model_type': model_type,
                    'error': str(e)
                })
        
        results_df = pd.DataFrame(results)
        
        logger.info(f"Model comparison complete")
        
        return results_df
    
    def hyperparameter_tuning(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        model_type: str = 'random_forest',
        param_grid: Dict = None
    ) -> Dict:
        """
        Perform simple grid search for hyperparameter tuning.
        
        Args:
            X_train: Training features
            y_train: Training target
            X_test: Test features
            y_test: Test target
            model_type: Type of model
            param_grid: Parameter grid for tuning
        
        Returns:
            Dictionary with best parameters and metrics
        """
        logger.info(f"Hyperparameter tuning for {model_type}...")
        
        if param_grid is None:
            # Default parameter grids
            if model_type == 'random_forest':
                param_grid = {
                    'n_estimators': [50, 100, 200],
                    'max_depth': [5, 10, 15]
                }
            elif model_type == 'gradient_boosting':
                param_grid = {
                    'n_estimators': [50, 100, 200],
                    'learning_rate': [0.01, 0.1, 0.2],
                    'max_depth': [3, 5, 7]
                }
            elif model_type == 'logistic_regression':
                param_grid = {
                    'C': [0.1, 1.0, 10.0],
                    'max_iter': [500, 1000, 2000]
                }
        
        # Grid search
        best_score = 0
        best_params = None
        best_model = None
        
        from itertools import product
        
        param_names = list(param_grid.keys())
        param_values = list(param_grid.values())
        
        for combination in product(*param_values):
            params = dict(zip(param_names, combination))
            
            try:
                model = self.train_model(X_train, y_train, model_type, params)
                metrics = self.evaluate_model(model, X_test, y_test)
                score = metrics.get('accuracy', 0)
                
                if score > best_score:
                    best_score = score
                    best_params = params
                    best_model = model
            except Exception as e:
                logger.warning(f"Error with params {params}: {e}")
        
        results = {
            'model_type': model_type,
            'best_params': best_params,
            'best_score': float(best_score),
            'best_model': best_model
        }
        
        logger.info(f"Hyperparameter tuning complete. Best score: {best_score:.3f}")
        
        return results
    
    def run_training_pipeline(
        self,
        data: pd.DataFrame,
        target_col: str,
        categorical_cols: List[str] = None,
        numerical_cols: List[str] = None,
        test_size: float = 0.2,
        model_types: List[str] = None,
        tune_hyperparameters: bool = False
    ) -> Dict:
        """
        Run complete training pipeline.
        
        Args:
            data: Input data
            target_col: Target column name
            categorical_cols: Categorical columns
            numerical_cols: Numerical columns
            test_size: Test set size
            model_types: Model types to train
            tune_hyperparameters: Whether to tune hyperparameters
        
        Returns:
            Dictionary with pipeline results
        """
        logger.info("Running complete training pipeline...")
        
        # Preprocess data
        X, y = self.preprocess_data(data, target_col, categorical_cols, numerical_cols)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )
        
        # Compare models
        comparison_df = self.compare_models(X_train, y_train, X_test, y_test, model_types)
        
        # Hyperparameter tuning if requested
        tuning_results = None
        if tune_hyperparameters and model_types:
            tuning_results = {}
            for model_type in model_types:
                try:
                    tuning_result = self.hyperparameter_tuning(
                        X_train, y_train, X_test, y_test, model_type
                    )
                    tuning_results[model_type] = tuning_result
                except Exception as e:
                    logger.error(f"Error tuning {model_type}: {e}")
        
        # Select best model
        if not comparison_df.empty and 'accuracy' in comparison_df.columns:
            best_model_type = comparison_df.loc[comparison_df['accuracy'].idxmax(), 'model_type']
            best_model = self.models.get(best_model_type)
        else:
            best_model_type = None
            best_model = None
        
        # Record training history
        training_record = {
            'timestamp': datetime.now().isoformat(),
            'n_samples': len(data),
            'n_features': X.shape[1],
            'test_size': test_size,
            'best_model_type': best_model_type,
            'model_comparison': comparison_df.to_dict('records'),
            'hyperparameter_tuning': tuning_results
        }
        self.training_history.append(training_record)
        
        results = {
            'training_record': training_record,
            'model_comparison': comparison_df,
            'best_model_type': best_model_type,
            'best_model': best_model,
            'hyperparameter_tuning': tuning_results
        }
        
        logger.info("Training pipeline complete")
        
        return results


def run_ml_training_pipeline(
    data: pd.DataFrame,
    target_col: str,
    categorical_cols: List[str] = None,
    numerical_cols: List[str] = None
) -> Tuple[MLTrainingPipeline, Dict]:
    """
    Convenience function to run ML training pipeline.
    
    Args:
        data: Input data
        target_col: Target column name
        categorical_cols: Categorical columns
        numerical_cols: Numerical columns
    
    Returns:
        Tuple of (pipeline, results)
    """
    pipeline = MLTrainingPipeline()
    
    results = pipeline.run_training_pipeline(
        data, target_col, categorical_cols, numerical_cols
    )
    
    return pipeline, results
