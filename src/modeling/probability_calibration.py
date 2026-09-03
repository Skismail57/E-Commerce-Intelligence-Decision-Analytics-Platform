"""
Probability Calibration Module
Implements probability calibration for improving model prediction reliability.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.metrics import brier_score_loss, log_loss
from sklearn.isotonic import IsotonicRegression
from config.logging_config import get_logger

logger = get_logger(__name__)


class ProbabilityCalibrator:
    """
    Probability calibration engine for improving prediction reliability.
    
    Features:
    - Platt scaling calibration
    - Isotonic regression calibration
    - Calibration curve analysis
    - Brier score calculation
    - Expected calibration error
    """
    
    def __init__(self):
        """Initialize probability calibrator"""
        self.calibrated_models = {}
        self.calibration_methods = ['sigmoid', 'isotonic']
        logger.info("Probability calibrator initialized")
    
    def calculate_calibration_curve(
        self,
        y_true: np.ndarray,
        y_prob: np.ndarray,
        n_bins: int = 10
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculate calibration curve.
        
        Args:
            y_true: True labels
            y_prob: Predicted probabilities
            n_bins: Number of bins for calibration curve
        
        Returns:
            Tuple of (fraction of positives, mean predicted probability)
        """
        fraction_of_positives, mean_predicted_value = calibration_curve(
            y_true, y_prob, n_bins=n_bins
        )
        
        return fraction_of_positives, mean_predicted_value
    
    def calculate_brier_score(
        self,
        y_true: np.ndarray,
        y_prob: np.ndarray
    ) -> float:
        """
        Calculate Brier score for probability calibration.
        
        Args:
            y_true: True labels
            y_prob: Predicted probabilities
        
        Returns:
            Brier score
        """
        brier_score = brier_score_loss(y_true, y_prob)
        return float(brier_score)
    
    def calculate_expected_calibration_error(
        self,
        y_true: np.ndarray,
        y_prob: np.ndarray,
        n_bins: int = 10
    ) -> float:
        """
        Calculate Expected Calibration Error (ECE).
        
        Args:
            y_true: True labels
            y_prob: Predicted probabilities
            n_bins: Number of bins
        
        Returns:
            Expected calibration error
        """
        fraction_of_positives, mean_predicted_value = self.calculate_calibration_curve(
            y_true, y_prob, n_bins
        )
        
        # Calculate weights for each bin
        bin_edges = np.linspace(0, 1, n_bins + 1)
        bin_indices = np.digitize(y_prob, bin_edges) - 1
        bin_weights = np.bincount(bin_indices, minlength=n_bins)
        bin_weights = bin_weights / len(y_true)
        
        # Calculate weighted calibration error
        calibration_error = np.abs(fraction_of_positives - mean_predicted_value)
        ece = np.sum(bin_weights * calibration_error)
        
        return float(ece)
    
    def calibrate_model(
        self,
        model: Any,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_calib: pd.DataFrame,
        y_calib: pd.Series,
        method: str = 'sigmoid'
    ) -> Any:
        """
        Calibrate a classifier.
        
        Args:
            model: Trained classifier
            X_train: Training features
            y_train: Training labels
            X_calib: Calibration features
            y_calib: Calibration labels
            method: Calibration method ('sigmoid' or 'isotonic')
        
        Returns:
            Calibrated model
        """
        logger.info(f"Calibrating model using {method} method...")
        
        if method not in self.calibration_methods:
            raise ValueError(f"Unknown calibration method: {method}")
        
        # Create calibrated classifier
        calibrated_model = CalibratedClassifierCV(
            model, method=method, cv='prefit'
        )
        
        # Fit on calibration data
        calibrated_model.fit(X_calib, y_calib)
        
        logger.info(f"Model calibrated successfully using {method}")
        
        return calibrated_model
    
    def compare_calibration_methods(
        self,
        model: Any,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_calib: pd.DataFrame,
        y_calib: pd.Series,
        X_test: pd.DataFrame,
        y_test: pd.Series
    ) -> pd.DataFrame:
        """
        Compare different calibration methods.
        
        Args:
            model: Trained classifier
            X_train: Training features
            y_train: Training labels
            X_calib: Calibration features
            y_calib: Calibration labels
            X_test: Test features
            y_test: Test labels
        
        Returns:
            DataFrame with calibration comparison
        """
        logger.info("Comparing calibration methods...")
        
        results = []
        
        # Get uncalibrated predictions
        y_prob_uncalibrated = model.predict_proba(X_test)[:, 1]
        
        # Calculate metrics for uncalibrated model
        results.append({
            'method': 'uncalibrated',
            'brier_score': self.calculate_brier_score(y_test.values, y_prob_uncalibrated),
            'ece': self.calculate_expected_calibration_error(y_test.values, y_prob_uncalibrated),
            'log_loss': log_loss(y_test, y_prob_uncalibrated)
        })
        
        # Test each calibration method
        for method in self.calibration_methods:
            try:
                # Calibrate model
                calibrated_model = self.calibrate_model(
                    model, X_train, y_train, X_calib, y_calib, method
                )
                
                # Get calibrated predictions
                y_prob_calibrated = calibrated_model.predict_proba(X_test)[:, 1]
                
                # Calculate metrics
                results.append({
                    'method': method,
                    'brier_score': self.calculate_brier_score(y_test.values, y_prob_calibrated),
                    'ece': self.calculate_expected_calibration_error(y_test.values, y_prob_calibrated),
                    'log_loss': log_loss(y_test, y_prob_calibrated)
                })
                
                # Store calibrated model
                self.calibrated_models[method] = calibrated_model
                
            except Exception as e:
                logger.error(f"Error with {method} calibration: {e}")
                results.append({
                    'method': method,
                    'error': str(e)
                })
        
        results_df = pd.DataFrame(results)
        
        logger.info("Calibration method comparison complete")
        
        return results_df
    
    def apply_temperature_scaling(
        self,
        y_prob: np.ndarray,
        temperature: float = 1.0
    ) -> np.ndarray:
        """
        Apply temperature scaling to probabilities.
        
        Args:
            y_prob: Original probabilities
            temperature: Temperature parameter
        
        Returns:
            Scaled probabilities
        """
        # Apply temperature scaling
        log_prob = np.log(y_prob + 1e-10)
        scaled_log_prob = log_prob / temperature
        scaled_prob = np.exp(scaled_log_prob)
        
        # Normalize
        scaled_prob = scaled_prob / np.sum(scaled_prob, axis=1, keepdims=True)
        
        return scaled_prob
    
    def find_optimal_temperature(
        self,
        y_true: np.ndarray,
        y_prob: np.ndarray,
        temperature_range: Tuple[float, float] = (0.1, 10.0)
    ) -> float:
        """
        Find optimal temperature for temperature scaling.
        
        Args:
            y_true: True labels
            y_prob: Predicted probabilities
            temperature_range: Range of temperatures to search
        
        Returns:
            Optimal temperature
        """
        logger.info("Finding optimal temperature...")
        
        best_temperature = 1.0
        best_score = float('inf')
        
        # Search for optimal temperature
        for temp in np.linspace(temperature_range[0], temperature_range[1], 100):
            scaled_prob = self.apply_temperature_scaling(y_prob.reshape(-1, 1), temp)
            score = self.calculate_brier_score(y_true, scaled_prob.flatten())
            
            if score < best_score:
                best_score = score
                best_temperature = temp
        
        logger.info(f"Optimal temperature found: {best_temperature:.3f}")
        
        return best_temperature
    
    def generate_calibration_report(
        self,
        y_true: np.ndarray,
        y_prob: np.ndarray,
        n_bins: int = 10
    ) -> Dict:
        """
        Generate comprehensive calibration report.
        
        Args:
            y_true: True labels
            y_prob: Predicted probabilities
            n_bins: Number of bins
        
        Returns:
            Dictionary with calibration report
        """
        logger.info("Generating calibration report...")
        
        # Calculate calibration metrics
        brier_score = self.calculate_brier_score(y_true, y_prob)
        ece = self.calculate_expected_calibration_error(y_true, y_prob, n_bins)
        logloss = log_loss(y_true, y_prob)
        
        # Calculate calibration curve
        fraction_of_positives, mean_predicted_value = self.calculate_calibration_curve(
            y_true, y_prob, n_bins
        )
        
        # Determine calibration quality
        if ece < 0.05:
            calibration_quality = 'excellent'
        elif ece < 0.1:
            calibration_quality = 'good'
        elif ece < 0.2:
            calibration_quality = 'fair'
        else:
            calibration_quality = 'poor'
        
        report = {
            'brier_score': brier_score,
            'expected_calibration_error': ece,
            'log_loss': logloss,
            'calibration_quality': calibration_quality,
            'n_bins': n_bins,
            'calibration_curve': {
                'fraction_of_positives': fraction_of_positives.tolist(),
                'mean_predicted_value': mean_predicted_value.tolist()
            }
        }
        
        logger.info(f"Calibration report complete. Quality: {calibration_quality}")
        
        return report


def run_calibration_pipeline(
    model: Any,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_calib: pd.DataFrame,
    y_calib: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series
) -> Tuple[ProbabilityCalibrator, Dict]:
    """
    Convenience function to run complete calibration pipeline.
    
    Args:
        model: Trained classifier
        X_train: Training features
        y_train: Training labels
        X_calib: Calibration features
        y_calib: Calibration labels
        X_test: Test features
        y_test: Test labels
    
    Returns:
        Tuple of (calibrator, results)
    """
    calibrator = ProbabilityCalibrator()
    
    # Compare calibration methods
    comparison = calibrator.compare_calibration_methods(
        model, X_train, y_train, X_calib, y_calib, X_test, y_test
    )
    
    # Generate calibration report for best method
    best_method = comparison.loc[comparison['brier_score'].idxmin(), 'method']
    
    if best_method in calibrator.calibrated_models:
        best_model = calibrator.calibrated_models[best_method]
        y_prob_best = best_model.predict_proba(X_test)[:, 1]
        report = calibrator.generate_calibration_report(y_test.values, y_prob_best)
    else:
        y_prob_uncalibrated = model.predict_proba(X_test)[:, 1]
        report = calibrator.generate_calibration_report(y_test.values, y_prob_uncalibrated)
    
    results = {
        'calibration_comparison': comparison,
        'best_method': best_method,
        'calibration_report': report
    }
    
    return calibrator, results
