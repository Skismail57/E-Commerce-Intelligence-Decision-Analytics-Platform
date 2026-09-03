"""
Reproducibility Features Module
Implements tools for ensuring reproducible experiments and analyses.

Architecture:
- Random seed management
- Environment tracking
- Dependency versioning
- Configuration serialization
- Experiment logging
"""

from typing import Dict, List, Optional, Any
import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path
import json
import hashlib
import sys
import os

from config.logging_config import get_logger
from config.settings import settings

logger = get_logger(__name__)


class ReproducibilityManager:
    """
    Manages reproducibility features for experiments.
    
    Ensures that experiments can be reproduced by tracking:
    - Random seeds
    - Environment details
    - Package versions
    - Configuration parameters
    """
    
    def __init__(self, experiment_name: str):
        """
        Initialize reproducibility manager.
        
        Args:
            experiment_name: Name of the experiment
        """
        self.experiment_name = experiment_name
        self.random_seed = 42
        self.environment_info = {}
        self.package_versions = {}
        self.config = {}
        
        # Capture environment info
        self._capture_environment()
        
        logger.info(f"Reproducibility Manager initialized for: {experiment_name}")
    
    def set_random_seed(self, seed: int = 42) -> None:
        """
        Set random seed for reproducibility.
        
        Args:
            seed: Random seed value
        """
        self.random_seed = seed
        
        # Set seeds for common libraries
        np.random.seed(seed)
        
        # Try to set seed for other libraries if available
        try:
            import random
            random.seed(seed)
        except ImportError:
            pass
        
        try:
            import tensorflow as tf
            tf.random.set_seed(seed)
        except ImportError:
            pass
        
        try:
            import torch
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
        except ImportError:
            pass
        
        logger.info(f"Random seed set to: {seed}")
    
    def _capture_environment(self) -> None:
        """Capture environment information."""
        self.environment_info = {
            'python_version': sys.version,
            'platform': sys.platform,
            'hostname': os.uname().nodename if hasattr(os, 'uname') else 'unknown',
            'working_directory': os.getcwd(),
            'timestamp': datetime.now().isoformat()
        }
        
        # Capture package versions
        self.package_versions = {
            'numpy': np.__version__,
            'pandas': pd.__version__,
        }
        
        # Try to capture other common packages
        common_packages = [
            'scikit-learn', 'xgboost', 'lightgbm', 'catboost',
            'tensorflow', 'torch', 'mlflow', 'streamlit'
        ]
        
        for package in common_packages:
            try:
                __import__(package.replace('-', '_'))
                self.package_versions[package] = sys.modules[package.replace('-', '_')].__version__
            except (ImportError, AttributeError):
                pass
        
        logger.info("Environment information captured")
    
    def save_config(self, config: Dict[str, Any]) -> None:
        """
        Save experiment configuration.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        logger.info(f"Configuration saved: {len(config)} parameters")
    
    def get_config_hash(self) -> str:
        """
        Get hash of configuration for identification.
        
        Returns:
            SHA256 hash of configuration
        """
        config_str = json.dumps(self.config, sort_keys=True)
        config_hash = hashlib.sha256(config_str.encode()).hexdigest()
        return config_hash
    
    def save_reproducibility_report(
        self,
        output_path: Optional[Path] = None
    ) -> str:
        """
        Save reproducibility report.
        
        Args:
            output_path: Optional path to save report
        
        Returns:
            Report string
        """
        report = {
            'experiment_name': self.experiment_name,
            'random_seed': self.random_seed,
            'config_hash': self.get_config_hash(),
            'environment': self.environment_info,
            'package_versions': self.package_versions,
            'configuration': self.config,
            'timestamp': datetime.now().isoformat()
        }
        
        if output_path:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, 'w') as f:
                json.dump(report, f, indent=2)
            logger.info(f"Reproducibility report saved to {output_path}")
        
        return json.dumps(report, indent=2)
    
    def load_reproducibility_report(
        self,
        report_path: Path
    ) -> Dict[str, Any]:
        """
        Load reproducibility report.
        
        Args:
            report_path: Path to report file
        
        Returns:
            Dictionary with reproducibility information
        """
        with open(report_path, 'r') as f:
            report = json.load(f)
        
        logger.info(f"Reproducibility report loaded from {report_path}")
        return report
    
    def verify_reproducibility(
        self,
        report_path: Path
    ) -> Dict[str, bool]:
        """
        Verify that current environment matches saved report.
        
        Args:
            report_path: Path to report file
        
        Returns:
            Dictionary with verification results
        """
        saved_report = self.load_reproducibility_report(report_path)
        
        verification = {
            'python_version_match': saved_report['environment']['python_version'] == self.environment_info['python_version'],
            'platform_match': saved_report['environment']['platform'] == self.environment_info['platform'],
            'package_versions_match': True,
            'config_hash_match': saved_report['config_hash'] == self.get_config_hash()
        }
        
        # Check package versions
        for package, version in saved_report['package_versions'].items():
            if package in self.package_versions:
                if self.package_versions[package] != version:
                    verification['package_versions_match'] = False
        
        all_match = all(verification.values())
        
        logger.info(f"Reproducibility verification: {'PASSED' if all_match else 'FAILED'}")
        return verification


class ExperimentLogger:
    """
    Logs experiment details for reproducibility.
    
    Tracks:
    - Experiment parameters
    - Data sources
    - Preprocessing steps
    - Model configurations
    - Results
    """
    
    def __init__(self, experiment_name: str):
        """
        Initialize experiment logger.
        
        Args:
            experiment_name: Name of the experiment
        """
        self.experiment_name = experiment_name
        self.log = {
            'experiment_name': experiment_name,
            'start_time': datetime.now().isoformat(),
            'steps': [],
            'data_sources': [],
            'parameters': {},
            'results': {}
        }
        
        logger.info(f"Experiment Logger initialized for: {experiment_name}")
    
    def log_step(self, step_name: str, step_details: Dict[str, Any]) -> None:
        """
        Log a processing step.
        
        Args:
            step_name: Name of the step
            step_details: Details of the step
        """
        step_entry = {
            'name': step_name,
            'timestamp': datetime.now().isoformat(),
            'details': step_details
        }
        self.log['steps'].append(step_entry)
        logger.info(f"Logged step: {step_name}")
    
    def log_data_source(self, source_name: str, source_path: str, hash_value: str) -> None:
        """
        Log a data source.
        
        Args:
            source_name: Name of the data source
            source_path: Path to the data
            hash_value: Hash of the data file
        """
        data_entry = {
            'name': source_name,
            'path': source_path,
            'hash': hash_value,
            'timestamp': datetime.now().isoformat()
        }
        self.log['data_sources'].append(data_entry)
        logger.info(f"Logged data source: {source_name}")
    
    def log_parameters(self, parameters: Dict[str, Any]) -> None:
        """
        Log experiment parameters.
        
        Args:
            parameters: Parameter dictionary
        """
        self.log['parameters'].update(parameters)
        logger.info(f"Logged {len(parameters)} parameters")
    
    def log_results(self, results: Dict[str, Any]) -> None:
        """
        Log experiment results.
        
        Args:
            results: Results dictionary
        """
        self.log['results'].update(results)
        logger.info("Logged results")
    
    def finalize_experiment(self) -> None:
        """Finalize experiment log."""
        self.log['end_time'] = datetime.now().isoformat()
        logger.info("Experiment finalized")
    
    def save_log(self, output_path: Path) -> None:
        """
        Save experiment log to file.
        
        Args:
            output_path: Path to save log
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(self.log, f, indent=2)
        
        logger.info(f"Experiment log saved to {output_path}")
    
    def load_log(self, log_path: Path) -> Dict[str, Any]:
        """
        Load experiment log from file.
        
        Args:
            log_path: Path to log file
        
        Returns:
            Experiment log dictionary
        """
        with open(log_path, 'r') as f:
            self.log = json.load(f)
        
        logger.info(f"Experiment log loaded from {log_path}")
        return self.log


def calculate_file_hash(file_path: Path) -> str:
    """
    Calculate SHA256 hash of a file.
    
    Args:
        file_path: Path to file
    
    Returns:
        SHA256 hash
    """
    sha256_hash = hashlib.sha256()
    
    with open(file_path, 'rb') as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    
    return sha256_hash.hexdigest()


def setup_reproducible_experiment(
    experiment_name: str,
    random_seed: int = 42,
    config: Optional[Dict[str, Any]] = None
) -> ReproducibilityManager:
    """
    Convenience function to setup a reproducible experiment.
    
    Args:
        experiment_name: Name of the experiment
        random_seed: Random seed
        config: Optional configuration
    
    Returns:
        ReproducibilityManager instance
    """
    repro_manager = ReproducibilityManager(experiment_name)
    repro_manager.set_random_seed(random_seed)
    
    if config:
        repro_manager.save_config(config)
    
    return repro_manager
