"""
Data Versioning Module
Implements data versioning for tracking dataset changes and lineage.

Architecture:
- Dataset versioning with hash-based identification
- Data lineage tracking
- Dataset metadata management
- Version comparison and diff
- Data catalog
"""

from typing import Dict, List, Optional, Any
import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path
import json
import hashlib
import shutil

from config.logging_config import get_logger
from config.settings import settings

logger = get_logger(__name__)


class DataVersioning:
    """
    Data versioning system for tracking dataset changes.
    
    Provides:
    - Version tracking with hash-based identification
    - Data lineage
    - Metadata management
    - Version comparison
    """
    
    def __init__(self, data_registry_path: Optional[Path] = None):
        """
        Initialize data versioning system.
        
        Args:
            data_registry_path: Path to data registry file
        """
        self.data_registry_path = data_registry_path or Path(settings.DATA_DIR) / "data_registry.json"
        self.registry = self._load_registry()
        
        logger.info(f"Data Versioning initialized with registry: {self.data_registry_path}")
    
    def _load_registry(self) -> Dict[str, Any]:
        """Load data registry from file."""
        if self.data_registry_path.exists():
            with open(self.data_registry_path, 'r') as f:
                return json.load(f)
        else:
            return {
                'datasets': {},
                'last_updated': datetime.now().isoformat()
            }
    
    def _save_registry(self) -> None:
        """Save data registry to file."""
        self.registry['last_updated'] = datetime.now().isoformat()
        
        self.data_registry_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.data_registry_path, 'w') as f:
            json.dump(self.registry, f, indent=2)
        
        logger.info("Data registry saved")
    
    def _calculate_dataframe_hash(self, df: pd.DataFrame) -> str:
        """
        Calculate hash of a dataframe.
        
        Args:
            df: DataFrame to hash
        
        Returns:
            SHA256 hash
        """
        # Convert to string representation for hashing
        df_str = df.to_csv(index=False).encode()
        return hashlib.sha256(df_str).hexdigest()
    
    def register_dataset(
        self,
        dataset_name: str,
        df: pd.DataFrame,
        metadata: Optional[Dict[str, Any]] = None,
        source_path: Optional[Path] = None
    ) -> str:
        """
        Register a new dataset version.
        
        Args:
            dataset_name: Name of the dataset
            df: DataFrame to register
            metadata: Optional metadata dictionary
            source_path: Optional source file path
        
        Returns:
            Version ID (hash)
        """
        version_hash = self._calculate_dataframe_hash(df)
        
        # Create dataset entry if not exists
        if dataset_name not in self.registry['datasets']:
            self.registry['datasets'][dataset_name] = {
                'name': dataset_name,
                'versions': [],
                'current_version': None,
                'created_at': datetime.now().isoformat()
            }
        
        dataset_entry = self.registry['datasets'][dataset_name]
        
        # Check if version already exists
        existing_version = next(
            (v for v in dataset_entry['versions'] if v['hash'] == version_hash),
            None
        )
        
        if existing_version:
            logger.info(f"Dataset {dataset_name} version {version_hash} already exists")
            return version_hash
        
        # Create new version entry
        version_entry = {
            'hash': version_hash,
            'created_at': datetime.now().isoformat(),
            'rows': len(df),
            'columns': len(df.columns),
            'column_names': list(df.columns),
            'dtypes': {col: str(dtype) for col, dtype in df.dtypes.items()},
            'metadata': metadata or {},
            'source_path': str(source_path) if source_path else None
        }
        
        dataset_entry['versions'].append(version_entry)
        dataset_entry['current_version'] = version_hash
        
        self._save_registry()
        
        logger.info(f"Registered dataset {dataset_name} version {version_hash}")
        return version_hash
    
    def get_dataset_version(
        self,
        dataset_name: str,
        version_hash: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Get dataset version information.
        
        Args:
            dataset_name: Name of the dataset
            version_hash: Optional version hash (uses current if None)
        
        Returns:
            Version information dictionary or None
        """
        if dataset_name not in self.registry['datasets']:
            return None
        
        dataset_entry = self.registry['datasets'][dataset_name]
        
        if version_hash is None:
            version_hash = dataset_entry['current_version']
        
        version_entry = next(
            (v for v in dataset_entry['versions'] if v['hash'] == version_hash),
            None
        )
        
        return version_entry
    
    def get_dataset_lineage(
        self,
        dataset_name: str
    ) -> List[Dict[str, Any]]:
        """
        Get lineage (version history) of a dataset.
        
        Args:
            dataset_name: Name of the dataset
        
        Returns:
            List of version entries in chronological order
        """
        if dataset_name not in self.registry['datasets']:
            return []
        
        dataset_entry = self.registry['datasets'][dataset_name]
        versions = sorted(dataset_entry['versions'], key=lambda x: x['created_at'])
        
        return versions
    
    def compare_versions(
        self,
        dataset_name: str,
        version1_hash: str,
        version2_hash: str
    ) -> Dict[str, Any]:
        """
        Compare two dataset versions.
        
        Args:
            dataset_name: Name of the dataset
            version1_hash: First version hash
            version2_hash: Second version hash
        
        Returns:
            Dictionary with comparison results
        """
        version1 = self.get_dataset_version(dataset_name, version1_hash)
        version2 = self.get_dataset_version(dataset_name, version2_hash)
        
        if not version1 or not version2:
            return {'error': 'One or both versions not found'}
        
        comparison = {
            'version1': version1_hash,
            'version2': version2_hash,
            'rows_diff': version2['rows'] - version1['rows'],
            'columns_diff': version2['columns'] - version1['columns'],
            'columns_added': list(set(version2['column_names']) - set(version1['column_names'])),
            'columns_removed': list(set(version1['column_names']) - set(version2['column_names'])),
            'dtype_changes': {}
        }
        
        # Check for dtype changes in common columns
        common_columns = set(version1['column_names']) & set(version2['column_names'])
        for col in common_columns:
            if version1['dtypes'][col] != version2['dtypes'][col]:
                comparison['dtype_changes'][col] = {
                    'from': version1['dtypes'][col],
                    'to': version2['dtypes'][col]
                }
        
        logger.info(f"Compared versions {version1_hash} and {version2_hash}")
        return comparison
    
    def list_datasets(self) -> List[str]:
        """
        List all registered datasets.
        
        Returns:
            List of dataset names
        """
        return list(self.registry['datasets'].keys())
    
    def get_current_version(self, dataset_name: str) -> Optional[str]:
        """
        Get current version hash for a dataset.
        
        Args:
            dataset_name: Name of the dataset
        
        Returns:
            Current version hash or None
        """
        if dataset_name not in self.registry['datasets']:
            return None
        
        return self.registry['datasets'][dataset_name]['current_version']
    
    def set_current_version(
        self,
        dataset_name: str,
        version_hash: str
    ) -> bool:
        """
        Set current version for a dataset.
        
        Args:
            dataset_name: Name of the dataset
            version_hash: Version hash to set as current
        
        Returns:
            True if successful, False otherwise
        """
        if dataset_name not in self.registry['datasets']:
            return False
        
        # Verify version exists
        version_exists = any(
            v['hash'] == version_hash 
            for v in self.registry['datasets'][dataset_name]['versions']
        )
        
        if not version_exists:
            return False
        
        self.registry['datasets'][dataset_name]['current_version'] = version_hash
        self._save_registry()
        
        logger.info(f"Set current version for {dataset_name} to {version_hash}")
        return True
    
    def generate_data_catalog(self) -> pd.DataFrame:
        """
        Generate a data catalog of all datasets.
        
        Returns:
            DataFrame with dataset information
        """
        catalog_data = []
        
        for dataset_name, dataset_info in self.registry['datasets'].items():
            current_version = self.get_dataset_version(dataset_name)
            
            catalog_data.append({
                'dataset_name': dataset_name,
                'current_version': dataset_info['current_version'],
                'total_versions': len(dataset_info['versions']),
                'current_rows': current_version['rows'] if current_version else 0,
                'current_columns': current_version['columns'] if current_version else 0,
                'created_at': dataset_info['created_at'],
                'last_updated': current_version['created_at'] if current_version else None
            })
        
        catalog_df = pd.DataFrame(catalog_data)
        
        logger.info(f"Generated data catalog: {len(catalog_df)} datasets")
        return catalog_df
    
    def export_registry(self, export_path: Path) -> None:
        """
        Export data registry to file.
        
        Args:
            export_path: Path to export registry
        """
        export_path = Path(export_path)
        export_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(export_path, 'w') as f:
            json.dump(self.registry, f, indent=2)
        
        logger.info(f"Registry exported to {export_path}")
    
    def import_registry(self, import_path: Path) -> None:
        """
        Import data registry from file.
        
        Args:
            import_path: Path to import registry from
        """
        with open(import_path, 'r') as f:
            imported_registry = json.load(f)
        
        # Merge with existing registry
        for dataset_name, dataset_info in imported_registry['datasets'].items():
            if dataset_name not in self.registry['datasets']:
                self.registry['datasets'][dataset_name] = dataset_info
            else:
                # Merge versions
                existing_hashes = {
                    v['hash'] for v in self.registry['datasets'][dataset_name]['versions']
                }
                for version in dataset_info['versions']:
                    if version['hash'] not in existing_hashes:
                        self.registry['datasets'][dataset_name]['versions'].append(version)
        
        self._save_registry()
        
        logger.info(f"Registry imported from {import_path}")


class DataLineageTracker:
    """
    Tracks data lineage across processing steps.
    
    Records:
    - Data sources
    - Transformations applied
    - Output datasets
    - Dependencies
    """
    
    def __init__(self, lineage_path: Optional[Path] = None):
        """
        Initialize data lineage tracker.
        
        Args:
            lineage_path: Path to lineage file
        """
        self.lineage_path = lineage_path or Path(settings.DATA_DIR) / "data_lineage.json"
        self.lineage = self._load_lineage()
        
        logger.info(f"Data Lineage Tracker initialized: {self.lineage_path}")
    
    def _load_lineage(self) -> Dict[str, Any]:
        """Load lineage from file."""
        if self.lineage_path.exists():
            with open(self.lineage_path, 'r') as f:
                return json.load(f)
        else:
            return {
                'nodes': [],
                'edges': [],
                'last_updated': datetime.now().isoformat()
            }
    
    def _save_lineage(self) -> None:
        """Save lineage to file."""
        self.lineage['last_updated'] = datetime.now().isoformat()
        
        self.lineage_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.lineage_path, 'w') as f:
            json.dump(self.lineage, f, indent=2)
        
        logger.info("Data lineage saved")
    
    def add_node(
        self,
        node_id: str,
        node_type: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Add a node to the lineage graph.
        
        Args:
            node_id: Unique identifier for the node
            node_type: Type of node (source, transformation, output)
            metadata: Optional metadata
        """
        node = {
            'id': node_id,
            'type': node_type,
            'metadata': metadata or {},
            'created_at': datetime.now().isoformat()
        }
        
        # Check if node already exists
        if not any(n['id'] == node_id for n in self.lineage['nodes']):
            self.lineage['nodes'].append(node)
            logger.info(f"Added lineage node: {node_id}")
    
    def add_edge(
        self,
        source_id: str,
        target_id: str,
        transformation: Optional[str] = None
    ) -> None:
        """
        Add an edge to the lineage graph.
        
        Args:
            source_id: Source node ID
            target_id: Target node ID
            transformation: Optional transformation description
        """
        edge = {
            'source': source_id,
            'target': target_id,
            'transformation': transformation,
            'created_at': datetime.now().isoformat()
        }
        
        self.lineage['edges'].append(edge)
        logger.info(f"Added lineage edge: {source_id} -> {target_id}")
    
    def get_lineage_graph(self) -> Dict[str, Any]:
        """
        Get the complete lineage graph.
        
        Returns:
            Dictionary with nodes and edges
        """
        return {
            'nodes': self.lineage['nodes'],
            'edges': self.lineage['edges']
        }
    
    def get_upstream_dependencies(self, node_id: str) -> List[str]:
        """
        Get upstream dependencies for a node.
        
        Args:
            node_id: Node ID to get dependencies for
        
        Returns:
            List of upstream node IDs
        """
        upstream = []
        
        for edge in self.lineage['edges']:
            if edge['target'] == node_id:
                upstream.append(edge['source'])
                # Recursively get dependencies
                upstream.extend(self.get_upstream_dependencies(edge['source']))
        
        return list(set(upstream))
    
    def get_downstream_consumers(self, node_id: str) -> List[str]:
        """
        Get downstream consumers for a node.
        
        Args:
            node_id: Node ID to get consumers for
        
        Returns:
            List of downstream node IDs
        """
        downstream = []
        
        for edge in self.lineage['edges']:
            if edge['source'] == node_id:
                downstream.append(edge['target'])
                # Recursively get consumers
                downstream.extend(self.get_downstream_consumers(edge['target']))
        
        return list(set(downstream))


def get_data_versioning_system() -> DataVersioning:
    """
    Convenience function to get data versioning system.
    
    Returns:
        DataVersioning instance
    """
    return DataVersioning()
