"""
Data Lineage and Quality Scoring Module
Implements data lineage tracking and quality scoring for data governance.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime
import json
from pathlib import Path
from config.logging_config import get_logger

logger = get_logger(__name__)


class DataLineageTracker:
    """
    Data lineage tracking and quality scoring engine.
    
    Features:
    - Data lineage tracking
    - Data source mapping
    - Transformation history
    - Quality scoring
    - Data freshness monitoring
    - Data completeness tracking
    """
    
    def __init__(self, lineage_path: str = "data/lineage"):
        """
        Initialize data lineage tracker.
        
        Args:
            lineage_path: Path to lineage storage
        """
        self.lineage_path = Path(lineage_path)
        self.lineage_path.mkdir(parents=True, exist_ok=True)
        
        self.data_sources = {}
        self.transformations = {}
        self.lineage_graph = {}
        self.quality_scores = {}
        
        # Load existing lineage
        self._load_lineage()
        
        logger.info(f"Data lineage tracker initialized at {lineage_path}")
    
    def _load_lineage(self):
        """Load existing lineage from disk."""
        lineage_file = self.lineage_path / "lineage.json"
        if lineage_file.exists():
            with open(lineage_file, 'r') as f:
                data = json.load(f)
                self.data_sources = data.get('data_sources', {})
                self.transformations = data.get('transformations', {})
                self.lineage_graph = data.get('lineage_graph', {})
                self.quality_scores = data.get('quality_scores', {})
    
    def _save_lineage(self):
        """Save lineage to disk."""
        lineage_file = self.lineage_path / "lineage.json"
        data = {
            'data_sources': self.data_sources,
            'transformations': self.transformations,
            'lineage_graph': self.lineage_graph,
            'quality_scores': self.quality_scores
        }
        with open(lineage_file, 'w') as f:
            json.dump(data, f, indent=2)
    
    def register_data_source(
        self,
        source_name: str,
        source_type: str,
        location: str,
        schema: Dict = None,
        metadata: Dict = None
    ) -> Dict:
        """
        Register a data source.
        
        Args:
            source_name: Name of the data source
            source_type: Type of source (database, file, api, etc.)
            location: Source location
            schema: Data schema
            metadata: Additional metadata
        
        Returns:
            Dictionary with source information
        """
        logger.info(f"Registering data source: {source_name}")
        
        source = {
            'name': source_name,
            'type': source_type,
            'location': location,
            'schema': schema or {},
            'metadata': metadata or {},
            'registered_at': datetime.now().isoformat()
        }
        
        self.data_sources[source_name] = source
        
        # Initialize lineage graph node
        self.lineage_graph[source_name] = {
            'type': 'source',
            'downstream': [],
            'upstream': []
        }
        
        self._save_lineage()
        
        logger.info(f"Data source {source_name} registered")
        
        return source
    
    def log_transformation(
        self,
        transformation_name: str,
        input_sources: List[str],
        output_target: str,
        transformation_type: str,
        parameters: Dict = None
    ) -> Dict:
        """
        Log a data transformation.
        
        Args:
            transformation_name: Name of transformation
            input_sources: List of input source names
            output_target: Output target name
            transformation_type: Type of transformation
            parameters: Transformation parameters
        
        Returns:
            Dictionary with transformation information
        """
        logger.info(f"Logging transformation: {transformation_name}")
        
        transformation = {
            'name': transformation_name,
            'input_sources': input_sources,
            'output_target': output_target,
            'type': transformation_type,
            'parameters': parameters or {},
            'timestamp': datetime.now().isoformat()
        }
        
        self.transformations[transformation_name] = transformation
        
        # Update lineage graph
        for source in input_sources:
            if source not in self.lineage_graph:
                self.lineage_graph[source] = {'type': 'source', 'downstream': [], 'upstream': []}
            self.lineage_graph[source]['downstream'].append(output_target)
        
        if output_target not in self.lineage_graph:
            self.lineage_graph[output_target] = {'type': 'target', 'downstream': [], 'upstream': []}
        
        self.lineage_graph[output_target]['upstream'].extend(input_sources)
        
        self._save_lineage()
        
        logger.info(f"Transformation {transformation_name} logged")
        
        return transformation
    
    def calculate_quality_score(
        self,
        data: pd.DataFrame,
        data_name: str,
        completeness_weight: float = 0.3,
        uniqueness_weight: float = 0.2,
        validity_weight: float = 0.3,
        freshness_weight: float = 0.2
    ) -> Dict:
        """
        Calculate quality score for data.
        
        Args:
            data: DataFrame to score
            data_name: Name of the data
            completeness_weight: Weight for completeness score
            uniqueness_weight: Weight for uniqueness score
            validity_weight: Weight for validity score
            freshness_weight: Weight for freshness score
        
        Returns:
            Dictionary with quality scores
        """
        logger.info(f"Calculating quality score for {data_name}")
        
        # Completeness score
        completeness_score = 1 - (data.isna().sum().sum() / (len(data) * len(data.columns)))
        
        # Uniqueness score (for key columns, assume first column is key)
        key_column = data.columns[0]
        uniqueness_score = data[key_column].nunique() / len(data)
        
        # Validity score (basic checks)
        validity_score = 1.0
        for col in data.select_dtypes(include=[np.number]).columns:
            if (data[col] < 0).any():
                validity_score -= 0.1
        
        validity_score = max(0, validity_score)
        
        # Freshness score (assume current if no timestamp)
        freshness_score = 1.0
        
        # Calculate weighted score
        overall_score = (
            completeness_score * completeness_weight +
            uniqueness_score * uniqueness_weight +
            validity_score * validity_weight +
            freshness_score * freshness_weight
        )
        
        quality_info = {
            'data_name': data_name,
            'completeness_score': float(completeness_score),
            'uniqueness_score': float(uniqueness_score),
            'validity_score': float(validity_score),
            'freshness_score': float(freshness_score),
            'overall_score': float(overall_score),
            'n_rows': len(data),
            'n_columns': len(data.columns),
            'n_null_values': int(data.isna().sum().sum()),
            'calculated_at': datetime.now().isoformat()
        }
        
        self.quality_scores[data_name] = quality_info
        
        self._save_lineage()
        
        logger.info(f"Quality score calculated for {data_name}: {overall_score:.3f}")
        
        return quality_info
    
    def get_lineage(self, data_name: str) -> Dict:
        """
        Get lineage information for a data asset.
        
        Args:
            data_name: Name of the data asset
        
        Returns:
            Dictionary with lineage information
        """
        logger.info(f"Getting lineage for {data_name}")
        
        if data_name not in self.lineage_graph:
            return {'error': f'Data {data_name} not found in lineage'}
        
        node = self.lineage_graph[data_name]
        
        # Get upstream lineage
        upstream_lineage = []
        visited = set()
        
        def trace_upstream(name):
            if name in visited or name not in self.lineage_graph:
                return
            visited.add(name)
            node_info = self.lineage_graph[name]
            for upstream in node_info['upstream']:
                upstream_lineage.append(upstream)
                trace_upstream(upstream)
        
        trace_upstream(data_name)
        
        # Get downstream lineage
        downstream_lineage = []
        visited = set()
        
        def trace_downstream(name):
            if name in visited or name not in self.lineage_graph:
                return
            visited.add(name)
            node_info = self.lineage_graph[name]
            for downstream in node_info['downstream']:
                downstream_lineage.append(downstream)
                trace_downstream(downstream)
        
        trace_downstream(data_name)
        
        lineage_info = {
            'data_name': data_name,
            'type': node['type'],
            'upstream': upstream_lineage,
            'downstream': downstream_lineage,
            'n_upstream': len(upstream_lineage),
            'n_downstream': len(downstream_lineage)
        }
        
        return lineage_info
    
    def get_quality_history(self, data_name: str) -> List[Dict]:
        """
        Get quality score history for a data asset.
        
        Args:
            data_name: Name of the data asset
        
        Returns:
            List of quality scores over time
        """
        if data_name not in self.quality_scores:
            return []
        
        # For simplicity, return current score
        # In a real implementation, this would track history over time
        return [self.quality_scores[data_name]]
    
    def generate_lineage_report(self) -> Dict:
        """
        Generate comprehensive lineage report.
        
        Returns:
            Dictionary with lineage report
        """
        logger.info("Generating lineage report...")
        
        report = {
            'n_data_sources': len(self.data_sources),
            'n_transformations': len(self.transformations),
            'n_lineage_nodes': len(self.lineage_graph),
            'data_sources': list(self.data_sources.keys()),
            'transformations': list(self.transformations.keys()),
            'average_quality_score': float(np.mean([s['overall_score'] for s in self.quality_scores.values()])) if self.quality_scores else 0,
            'quality_scores': self.quality_scores
        }
        
        logger.info("Lineage report generated")
        
        return report
    
    def trace_data_flow(self, source_name: str, target_name: str) -> List[str]:
        """
        Trace data flow from source to target.
        
        Args:
            source_name: Source data name
            target_name: Target data name
        
        Returns:
            List of transformation names in the path
        """
        logger.info(f"Tracing data flow from {source_name} to {target_name}")
        
        # BFS to find path
        from collections import deque
        
        queue = deque([(source_name, [])])
        visited = set()
        
        while queue:
            current, path = queue.popleft()
            
            if current == target_name:
                return path
            
            if current in visited:
                continue
            
            visited.add(current)
            
            if current in self.lineage_graph:
                for downstream in self.lineage_graph[current]['downstream']:
                    # Find transformation that produces this downstream
                    for trans_name, trans in self.transformations.items():
                        if trans['output_target'] == downstream:
                            queue.append((downstream, path + [trans_name]))
        
        return []


def run_lineage_pipeline(
    data_sources: Dict[str, pd.DataFrame]
) -> Tuple[DataLineageTracker, Dict]:
    """
    Convenience function to run data lineage pipeline.
    
    Args:
        data_sources: Dictionary of data sources
    
    Returns:
        Tuple of (tracker, results)
    """
    tracker = DataLineageTracker()
    
    # Register data sources
    for name, data in data_sources.items():
        tracker.register_data_source(
            name,
            'dataframe',
            'memory',
            {'columns': list(data.columns), 'dtypes': data.dtypes.astype(str).to_dict()}
        )
    
    # Calculate quality scores
    quality_scores = {}
    for name, data in data_sources.items():
        quality_scores[name] = tracker.calculate_quality_score(data, name)
    
    # Generate lineage report
    report = tracker.generate_lineage_report()
    
    results = {
        'quality_scores': quality_scores,
        'lineage_report': report
    }
    
    return tracker, results
