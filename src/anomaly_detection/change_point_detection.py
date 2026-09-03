"""
Change-Point Detection Module
Implements change-point detection algorithms for identifying structural breaks in time series.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from scipy import stats
from ruptures import Pelt, Binseg
from config.logging_config import get_logger

logger = get_logger(__name__)


class ChangePointDetector:
    """
    Change-point detection engine for time series analysis.
    
    Features:
    - Statistical change-point detection (CUSUM, Bayesian)
    - Algorithmic detection (PELT, Binary Segmentation)
    - Multiple change-point detection
    - Change-point significance testing
    - Before/after analysis
    """
    
    def __init__(self):
        """Initialize change-point detector"""
        self.change_points = []
        self.change_point_stats = {}
    
    def detect_cusum(
        self,
        data: pd.Series,
        threshold: float = 5.0,
        drift: float = 0.0
    ) -> List[int]:
        """
        Detect change-points using CUSUM (Cumulative Sum) algorithm.
        
        Args:
            data: Time series data
            threshold: Detection threshold
            drift: Expected drift
        
        Returns:
            List of change-point indices
        """
        logger.info("Detecting change-points using CUSUM...")
        
        x = data.values.astype(float)
        n = len(x)
        
        # Calculate CUSUM
        cusum_pos = np.zeros(n)
        cusum_neg = np.zeros(n)
        
        for i in range(1, n):
            cusum_pos[i] = max(0, cusum_pos[i-1] + x[i] - drift - threshold)
            cusum_neg[i] = min(0, cusum_neg[i-1] + x[i] - drift + threshold)
        
        # Detect change-points
        change_points = []
        for i in range(1, n):
            if cusum_pos[i] == 0 and cusum_pos[i-1] > 0:
                change_points.append(i)
            if cusum_neg[i] == 0 and cusum_neg[i-1] < 0:
                change_points.append(i)
        
        logger.info(f"CUSUM detected {len(change_points)} change-points")
        
        return change_points
    
    def detect_pelt(
        self,
        data: pd.Series,
        penalty: float = 10.0,
        min_size: int = 10
    ) -> List[int]:
        """
        Detect change-points using PELT (Pruned Exact Linear Time) algorithm.
        
        Args:
            data: Time series data
            penalty: Penalty for adding change-points
            min_size: Minimum segment size
        
        Returns:
            List of change-point indices
        """
        logger.info("Detecting change-points using PELT...")
        
        x = data.values.astype(float)
        
        # Use ruptures library
        algo = Pelt(model="rbf", min_size=min_size, penalty=penalty).fit(x)
        change_points = algo.predict(pen=penalty)
        
        # Remove the last point (end of series)
        if change_points and change_points[-1] == len(x):
            change_points = change_points[:-1]
        
        logger.info(f"PELT detected {len(change_points)} change-points")
        
        return change_points
    
    def detect_binary_segmentation(
        self,
        data: pd.Series,
        n_bkps: int = 5,
        min_size: int = 10
    ) -> List[int]:
        """
        Detect change-points using Binary Segmentation.
        
        Args:
            data: Time series data
            n_bkps: Maximum number of change-points
            min_size: Minimum segment size
        
        Returns:
            List of change-point indices
        """
        logger.info(f"Detecting change-points using Binary Segmentation (max {n_bkps})...")
        
        x = data.values.astype(float)
        
        # Use ruptures library
        algo = Binseg(model="rbf", min_size=min_size).fit(x)
        change_points = algo.predict(n_bkps=n_bkps)
        
        # Remove the last point (end of series)
        if change_points and change_points[-1] == len(x):
            change_points = change_points[:-1]
        
        logger.info(f"Binary Segmentation detected {len(change_points)} change-points")
        
        return change_points
    
    def detect_statistical_change(
        self,
        data: pd.Series,
        window_size: int = 30,
        test: str = 'ks'
    ) -> List[int]:
        """
        Detect change-points using statistical tests.
        
        Args:
            data: Time series data
            window_size: Window size for comparison
            test: Statistical test ('ks', 'ttest')
        
        Returns:
            List of change-point indices
        """
        logger.info(f"Detecting change-points using {test} test...")
        
        x = data.values.astype(float)
        n = len(x)
        change_points = []
        
        for i in range(window_size, n - window_size):
            # Compare windows before and after
            before = x[i-window_size:i]
            after = x[i:i+window_size]
            
            if test == 'ks':
                # Kolmogorov-Smirnov test
                statistic, p_value = stats.ks_2samp(before, after)
            elif test == 'ttest':
                # T-test
                statistic, p_value = stats.ttest_ind(before, after)
            else:
                raise ValueError(f"Unknown test: {test}")
            
            # Significant change
            if p_value < 0.05:
                change_points.append(i)
        
        logger.info(f"Statistical test detected {len(change_points)} change-points")
        
        return change_points
    
    def analyze_change_point(
        self,
        data: pd.Series,
        change_point: int,
        window_size: int = 30
    ) -> Dict:
        """
        Analyze a specific change-point.
        
        Args:
            data: Time series data
            change_point: Index of change-point
            window_size: Window size for analysis
        
        Returns:
            Dictionary with change-point analysis
        """
        x = data.values.astype(float)
        
        # Get before and after segments
        before = x[max(0, change_point - window_size):change_point]
        after = x[change_point:min(len(x), change_point + window_size)]
        
        # Calculate statistics
        before_mean = np.mean(before)
        after_mean = np.mean(after)
        before_std = np.std(before)
        after_std = np.std(after)
        
        # Calculate change magnitude
        change_magnitude = (after_mean - before_mean) / before_mean if before_mean != 0 else 0
        
        # Statistical test
        statistic, p_value = stats.ttest_ind(before, after)
        
        result = {
            'change_point': change_point,
            'before_mean': float(before_mean),
            'after_mean': float(after_mean),
            'before_std': float(before_std),
            'after_std': float(after_std),
            'change_magnitude_pct': float(change_magnitude * 100),
            't_statistic': float(statistic),
            'p_value': float(p_value),
            'is_significant': p_value < 0.05
        }
        
        return result
    
    def detect_multiple_change_points(
        self,
        data: pd.Series,
        methods: List[str] = None
    ) -> Dict:
        """
        Detect change-points using multiple methods.
        
        Args:
            data: Time series data
            methods: List of methods to use
        
        Returns:
            Dictionary with results from all methods
        """
        if methods is None:
            methods = ['cusum', 'pelt', 'binary_segmentation']
        
        results = {}
        
        if 'cusum' in methods:
            results['cusum'] = self.detect_cusum(data)
        
        if 'pelt' in methods:
            results['pelt'] = self.detect_pelt(data)
        
        if 'binary_segmentation' in methods:
            results['binary_segmentation'] = self.detect_binary_segmentation(data)
        
        if 'statistical' in methods:
            results['statistical'] = self.detect_statistical_change(data)
        
        # Find consensus change-points
        all_change_points = []
        for method, points in results.items():
            all_change_points.extend(points)
        
        # Count occurrences
        from collections import Counter
        point_counts = Counter(all_change_points)
        
        # Get change-points detected by multiple methods
        consensus_points = [p for p, count in point_counts.items() if count >= 2]
        
        results['consensus'] = sorted(consensus_points)
        
        logger.info(f"Multi-method detection complete. Consensus: {len(consensus_points)} change-points")
        
        return results
    
    def analyze_all_change_points(
        self,
        data: pd.Series,
        change_points: List[int],
        window_size: int = 30
    ) -> List[Dict]:
        """
        Analyze all detected change-points.
        
        Args:
            data: Time series data
            change_points: List of change-point indices
            window_size: Window size for analysis
        
        Returns:
            List of change-point analyses
        """
        analyses = []
        
        for cp in change_points:
            analysis = self.analyze_change_point(data, cp, window_size)
            analyses.append(analysis)
        
        # Sort by change magnitude
        analyses.sort(key=lambda x: abs(x['change_magnitude_pct']), reverse=True)
        
        return analyses
    
    def segment_time_series(
        self,
        data: pd.Series,
        change_points: List[int]
    ) -> List[pd.Series]:
        """
        Segment time series based on change-points.
        
        Args:
            data: Time series data
            change_points: List of change-point indices
        
        Returns:
            List of segmented time series
        """
        segments = []
        start = 0
        
        for cp in sorted(change_points):
            segments.append(data.iloc[start:cp])
            start = cp
        
        # Add final segment
        segments.append(data.iloc[start:])
        
        return segments


def run_change_point_detection_pipeline(
    data: pd.Series,
    methods: List[str] = None,
    analyze: bool = True
) -> Tuple[ChangePointDetector, Dict]:
    """
    Convenience function to run complete change-point detection pipeline.
    
    Args:
        data: Time series data
        methods: Methods to use
        analyze: Whether to analyze change-points
    
    Returns:
        Tuple of (detector, results)
    """
    detector = ChangePointDetector()
    
    # Detect change-points
    detection_results = detector.detect_multiple_change_points(data, methods)
    
    # Use consensus change-points
    consensus_points = detection_results.get('consensus', [])
    
    results = {
        'detection_results': detection_results,
        'consensus_change_points': consensus_points,
        'n_change_points': len(consensus_points)
    }
    
    # Analyze change-points if requested
    if analyze and consensus_points:
        analyses = detector.analyze_all_change_points(data, consensus_points)
        results['change_point_analyses'] = analyses
        
        # Segment time series
        segments = detector.segment_time_series(data, consensus_points)
        results['segments'] = segments
    
    return detector, results
