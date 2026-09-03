"""
MLflow Model Registry Module
Implements upgraded MLflow model registry for model lifecycle management.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime
import joblib
import json
from pathlib import Path
from config.logging_config import get_logger

logger = get_logger(__name__)


class ModelRegistry:
    """
    MLflow-style model registry for model lifecycle management.
    
    Features:
    - Model registration and versioning
    - Model stage management (Staging, Production, Archived)
    - Model metadata tracking
    - Model loading and deployment
    - Model comparison and selection
    """
    
    def __init__(self, registry_path: str = "data/model_registry"):
        """
        Initialize model registry.
        
        Args:
            registry_path: Path to model registry directory
        """
        self.registry_path = Path(registry_path)
        self.registry_path.mkdir(parents=True, exist_ok=True)
        
        self.registered_models = {}
        self.model_versions = {}
        
        # Load existing registry
        self._load_registry()
        
        logger.info(f"Model registry initialized at {registry_path}")
    
    def _load_registry(self):
        """Load existing model registry from disk."""
        registry_file = self.registry_path / "registry.json"
        if registry_file.exists():
            with open(registry_file, 'r') as f:
                self.registered_models = json.load(f)
    
    def _save_registry(self):
        """Save model registry to disk."""
        registry_file = self.registry_path / "registry.json"
        with open(registry_file, 'w') as f:
            json.dump(self.registered_models, f, indent=2)
    
    def register_model(
        self,
        model_name: str,
        model: Any,
        model_type: str = 'sklearn',
        metrics: Dict = None,
        hyperparameters: Dict = None,
        tags: Dict = None,
        description: str = None
    ) -> Dict:
        """
        Register a new model in the registry.
        
        Args:
            model_name: Name of the model
            model: Trained model object
            model_type: Type of model (sklearn, tensorflow, pytorch, etc.)
            metrics: Model performance metrics
            hyperparameters: Model hyperparameters
            tags: Model tags for organization
            description: Model description
        
        Returns:
            Dictionary with registration result
        """
        logger.info(f"Registering model: {model_name}")
        
        # Initialize model entry if not exists
        if model_name not in self.registered_models:
            self.registered_models[model_name] = {
                'name': model_name,
                'versions': [],
                'current_stage': 'None',
                'created_at': datetime.now().isoformat()
            }
        
        # Generate new version
        current_version = len(self.registered_models[model_name]['versions']) + 1
        version_name = f"v{current_version}"
        
        # Save model
        model_dir = self.registry_path / model_name / version_name
        model_dir.mkdir(parents=True, exist_ok=True)
        
        model_file = model_dir / "model.pkl"
        joblib.dump(model, model_file)
        
        # Create version entry
        version_info = {
            'version': version_name,
            'version_number': current_version,
            'model_type': model_type,
            'stage': 'Staging',
            'created_at': datetime.now().isoformat(),
            'metrics': metrics or {},
            'hyperparameters': hyperparameters or {},
            'tags': tags or {},
            'description': description,
            'model_path': str(model_file)
        }
        
        self.registered_models[model_name]['versions'].append(version_info)
        
        # Save registry
        self._save_registry()
        
        logger.info(f"Model {model_name} registered as version {version_name}")
        
        return version_info
    
    def transition_model_stage(
        self,
        model_name: str,
        version: str,
        new_stage: str
    ) -> Dict:
        """
        Transition model to a new stage.
        
        Args:
            model_name: Name of the model
            version: Model version
            new_stage: New stage (Production, Staging, Archived)
        
        Returns:
            Dictionary with transition result
        """
        logger.info(f"Transitioning {model_name} {version} to {new_stage}")
        
        if model_name not in self.registered_models:
            raise ValueError(f"Model {model_name} not found")
        
        # Find version
        version_info = None
        for v in self.registered_models[model_name]['versions']:
            if v['version'] == version:
                version_info = v
                break
        
        if not version_info:
            raise ValueError(f"Version {version} not found for model {model_name}")
        
        # Archive current production model if transitioning to production
        if new_stage == 'Production':
            for v in self.registered_models[model_name]['versions']:
                if v['stage'] == 'Production':
                    v['stage'] = 'Archived'
        
        # Update stage
        version_info['stage'] = new_stage
        version_info['last_updated'] = datetime.now().isoformat()
        
        # Update current stage
        if new_stage == 'Production':
            self.registered_models[model_name]['current_stage'] = 'Production'
        
        # Save registry
        self._save_registry()
        
        logger.info(f"Model {model_name} {version} transitioned to {new_stage}")
        
        return version_info
    
    def load_model(
        self,
        model_name: str,
        version: str = None,
        stage: str = None
    ) -> Any:
        """
        Load a model from the registry.
        
        Args:
            model_name: Name of the model
            version: Specific version to load (optional)
            stage: Load model from specific stage (optional)
        
        Returns:
            Loaded model object
        """
        logger.info(f"Loading model: {model_name}")
        
        if model_name not in self.registered_models:
            raise ValueError(f"Model {model_name} not found")
        
        # Determine which version to load
        if version:
            # Load specific version
            version_info = None
            for v in self.registered_models[model_name]['versions']:
                if v['version'] == version:
                    version_info = v
                    break
            if not version_info:
                raise ValueError(f"Version {version} not found")
        elif stage:
            # Load from stage
            version_info = None
            for v in self.registered_models[model_name]['versions']:
                if v['stage'] == stage:
                    version_info = v
                    break
            if not version_info:
                raise ValueError(f"No model found in stage {stage}")
        else:
            # Load latest version
            version_info = self.registered_models[model_name]['versions'][-1]
        
        # Load model from disk
        model_path = version_info['model_path']
        model = joblib.load(model_path)
        
        logger.info(f"Model {model_name} loaded from {version_info['version']}")
        
        return model
    
    def list_models(self) -> List[Dict]:
        """
        List all registered models.
        
        Returns:
            List of model information
        """
        models = []
        
        for model_name, model_info in self.registered_models.items():
            latest_version = model_info['versions'][-1] if model_info['versions'] else None
            
            models.append({
                'name': model_name,
                'n_versions': len(model_info['versions']),
                'current_stage': model_info.get('current_stage', 'None'),
                'latest_version': latest_version['version'] if latest_version else None,
                'created_at': model_info.get('created_at')
            })
        
        return models
    
    def get_model_info(
        self,
        model_name: str,
        version: str = None
    ) -> Dict:
        """
        Get detailed information about a model.
        
        Args:
            model_name: Name of the model
            version: Specific version (optional)
        
        Returns:
            Dictionary with model information
        """
        if model_name not in self.registered_models:
            raise ValueError(f"Model {model_name} not found")
        
        if version:
            # Get specific version info
            version_info = None
            for v in self.registered_models[model_name]['versions']:
                if v['version'] == version:
                    version_info = v
                    break
            if not version_info:
                raise ValueError(f"Version {version} not found")
            return version_info
        else:
            # Return all versions
            return self.registered_models[model_name]
    
    def compare_models(
        self,
        model_name: str,
        metric: str = 'accuracy'
    ) -> pd.DataFrame:
        """
        Compare different versions of a model.
        
        Args:
            model_name: Name of the model
            metric: Metric to compare
        
        Returns:
            DataFrame with model comparison
        """
        if model_name not in self.registered_models:
            raise ValueError(f"Model {model_name} not found")
        
        comparison_data = []
        
        for version_info in self.registered_models[model_name]['versions']:
            comparison_data.append({
                'version': version_info['version'],
                'stage': version_info['stage'],
                'created_at': version_info['created_at'],
                'metric_value': version_info['metrics'].get(metric, 0),
                'metric_name': metric
            })
        
        comparison_df = pd.DataFrame(comparison_data)
        
        return comparison_df
    
    def delete_model(
        self,
        model_name: str,
        version: str = None
    ) -> Dict:
        """
        Delete a model or specific version.
        
        Args:
            model_name: Name of the model
            version: Specific version to delete (optional)
        
        Returns:
            Dictionary with deletion result
        """
        logger.info(f"Deleting model: {model_name}")
        
        if model_name not in self.registered_models:
            raise ValueError(f"Model {model_name} not found")
        
        if version:
            # Delete specific version
            self.registered_models[model_name]['versions'] = [
                v for v in self.registered_models[model_name]['versions']
                if v['version'] != version
            ]
            
            # Delete model files
            model_dir = self.registry_path / model_name / version
            if model_dir.exists():
                import shutil
                shutil.rmtree(model_dir)
        else:
            # Delete entire model
            del self.registered_models[model_name]
            
            # Delete model directory
            model_dir = self.registry_path / model_name
            if model_dir.exists():
                import shutil
                shutil.rmtree(model_dir)
        
        # Save registry
        self._save_registry()
        
        logger.info(f"Model {model_name} deleted")
        
        return {'success': True, 'model_name': model_name, 'version': version}


def run_model_registry_pipeline(
    model: Any,
    model_name: str,
    metrics: Dict = None
) -> Tuple[ModelRegistry, Dict]:
    """
    Convenience function to run model registry pipeline.
    
    Args:
        model: Trained model
        model_name: Name for the model
        metrics: Model performance metrics
    
    Returns:
        Tuple of (registry, results)
    """
    registry = ModelRegistry()
    
    # Register model
    registration_result = registry.register_model(
        model_name, model, metrics=metrics
    )
    
    # Transition to production
    registry.transition_model_stage(model_name, registration_result['version'], 'Production')
    
    # List models
    models = registry.list_models()
    
    results = {
        'registration': registration_result,
        'models_list': models
    }
    
    return registry, results
