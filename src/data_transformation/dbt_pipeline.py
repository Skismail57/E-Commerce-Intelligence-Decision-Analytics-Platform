"""
DBT Data Transformations Module
Implements dbt-style data transformations for data pipeline management.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Callable
from datetime import datetime
import json
from pathlib import Path
from config.logging_config import get_logger

logger = get_logger(__name__)


class DBTPipeline:
    """
    DBT-style data transformation pipeline.
    
    Features:
    - SQL-like transformations
    - Data model definitions
    - Transformation dependencies
    - Pipeline orchestration
    - Data quality checks
    """
    
    def __init__(self, project_path: str = "data/dbt_project"):
        """
        Initialize DBT pipeline.
        
        Args:
            project_path: Path to DBT project directory
        """
        self.project_path = Path(project_path)
        self.project_path.mkdir(parents=True, exist_ok=True)
        
        self.models = {}
        self.transformations = {}
        self.dependencies = {}
        
        logger.info(f"DBT pipeline initialized at {project_path}")
    
    def define_model(
        self,
        model_name: str,
        source_table: str,
        columns: List[str],
        description: str = None
    ) -> Dict:
        """
        Define a data model.
        
        Args:
            model_name: Name of the model
            source_table: Source table name
            columns: List of column definitions
            description: Model description
        
        Returns:
            Dictionary with model definition
        """
        logger.info(f"Defining model: {model_name}")
        
        model = {
            'name': model_name,
            'source_table': source_table,
            'columns': columns,
            'description': description,
            'created_at': datetime.now().isoformat()
        }
        
        self.models[model_name] = model
        
        logger.info(f"Model {model_name} defined")
        
        return model
    
    def create_transformation(
        self,
        transformation_name: str,
        sql_query: str,
        depends_on: List[str] = None,
        description: str = None
    ) -> Dict:
        """
        Create a data transformation.
        
        Args:
            transformation_name: Name of the transformation
            sql_query: SQL query for transformation
            depends_on: List of dependencies
            description: Transformation description
        
        Returns:
            Dictionary with transformation definition
        """
        logger.info(f"Creating transformation: {transformation_name}")
        
        transformation = {
            'name': transformation_name,
            'sql_query': sql_query,
            'depends_on': depends_on or [],
            'description': description,
            'created_at': datetime.now().isoformat()
        }
        
        self.transformations[transformation_name] = transformation
        
        # Track dependencies
        if depends_on:
            self.dependencies[transformation_name] = depends_on
        
        logger.info(f"Transformation {transformation_name} created")
        
        return transformation
    
    def apply_transformation(
        self,
        data: pd.DataFrame,
        transformation_name: str
    ) -> pd.DataFrame:
        """
        Apply a transformation to data.
        
        Args:
            data: Input DataFrame
            transformation_name: Name of transformation to apply
        
        Returns:
            Transformed DataFrame
        """
        logger.info(f"Applying transformation: {transformation_name}")
        
        if transformation_name not in self.transformations:
            raise ValueError(f"Transformation {transformation_name} not found")
        
        transformation = self.transformations[transformation_name]
        
        # Parse and apply SQL-like transformation
        # For simplicity, we'll implement basic operations
        result = self._apply_sql_like(data, transformation['sql_query'])
        
        logger.info(f"Transformation {transformation_name} applied")
        
        return result
    
    def _apply_sql_like(self, data: pd.DataFrame, sql_query: str) -> pd.DataFrame:
        """
        Apply SQL-like transformation to DataFrame.
        
        Args:
            data: Input DataFrame
            sql_query: SQL-like query
        
        Returns:
            Transformed DataFrame
        """
        # Simple SQL-like operations
        query_lower = sql_query.lower()
        
        # SELECT operation
        if 'select' in query_lower:
            # Extract columns
            if 'select *' in query_lower:
                result = data.copy()
            else:
                # Parse column names (simplified)
                select_part = query_lower.split('select')[1].split('from')[0]
                columns = [col.strip() for col in select_part.split(',')]
                columns = [col.replace(' as ', '_as_') for col in columns]
                result = data[columns].copy()
        
        # WHERE operation
        if 'where' in query_lower:
            where_part = query_lower.split('where')[1]
            # Simplified where clause parsing
            if '>' in where_part:
                col, val = where_part.split('>')
                col = col.strip()
                val = val.strip()
                try:
                    val = float(val)
                    result = result[result[col] > val]
                except:
                    pass
            elif '<' in where_part:
                col, val = where_part.split('<')
                col = col.strip()
                val = val.strip()
                try:
                    val = float(val)
                    result = result[result[col] < val]
                except:
                    pass
            elif '=' in where_part:
                col, val = where_part.split('=')
                col = col.strip()
                val = val.strip().strip("'\"")
                result = result[result[col] == val]
        
        # GROUP BY operation
        if 'group by' in query_lower:
            group_part = query_lower.split('group by')[1].strip()
            group_cols = [col.strip() for col in group_part.split(',')]
            result = result.groupby(group_cols).agg('sum').reset_index()
        
        # ORDER BY operation
        if 'order by' in query_lower:
            order_part = query_lower.split('order by')[1].strip()
            order_col = order_part.split()[0]
            if 'desc' in order_part:
                result = result.sort_values(order_col, ascending=False)
            else:
                result = result.sort_values(order_col)
        
        return result
    
    def run_pipeline(
        self,
        data_sources: Dict[str, pd.DataFrame],
        target_transformations: List[str] = None
    ) -> Dict:
        """
        Run the complete transformation pipeline.
        
        Args:
            data_sources: Dictionary of source data
            target_transformations: List of transformations to run
        
        Returns:
            Dictionary with pipeline results
        """
        logger.info("Running DBT pipeline...")
        
        if target_transformations is None:
            target_transformations = list(self.transformations.keys())
        
        results = {}
        intermediate_data = data_sources.copy()
        
        # Run transformations in dependency order
        for transformation_name in target_transformations:
            if transformation_name not in self.transformations:
                logger.warning(f"Transformation {transformation_name} not found, skipping")
                continue
            
            transformation = self.transformations[transformation_name]
            
            # Check dependencies
            if transformation['depends_on']:
                for dep in transformation['depends_on']:
                    if dep not in intermediate_data:
                        logger.warning(f"Dependency {dep} not found, skipping {transformation_name}")
                        continue
            
            # Determine source data
            if transformation['depends_on']:
                # Merge all dependencies
                source_data = None
                for dep in transformation['depends_on']:
                    if dep in intermediate_data:
                        if source_data is None:
                            source_data = intermediate_data[dep]
                        else:
                            # Simple merge
                            source_data = pd.concat([source_data, intermediate_data[dep]], ignore_index=True)
            else:
                # Use first available source
                source_data = list(intermediate_data.values())[0]
            
            # Apply transformation
            try:
                transformed_data = self.apply_transformation(source_data, transformation_name)
                intermediate_data[transformation_name] = transformed_data
                results[transformation_name] = {
                    'status': 'success',
                    'n_rows': len(transformed_data),
                    'n_columns': len(transformed_data.columns)
                }
            except Exception as e:
                logger.error(f"Error in transformation {transformation_name}: {e}")
                results[transformation_name] = {
                    'status': 'error',
                    'error': str(e)
                }
        
        logger.info(f"DBT pipeline complete. {len(results)} transformations executed")
        
        return results
    
    def add_data_quality_check(
        self,
        model_name: str,
        check_type: str,
        check_config: Dict
    ) -> Dict:
        """
        Add data quality check to a model.
        
        Args:
            model_name: Name of the model
            check_type: Type of check (not_null, unique, range, etc.)
            check_config: Check configuration
        
        Returns:
            Dictionary with check definition
        """
        logger.info(f"Adding data quality check to {model_name}")
        
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not found")
        
        if 'quality_checks' not in self.models[model_name]:
            self.models[model_name]['quality_checks'] = []
        
        check = {
            'type': check_type,
            'config': check_config,
            'created_at': datetime.now().isoformat()
        }
        
        self.models[model_name]['quality_checks'].append(check)
        
        logger.info(f"Data quality check added to {model_name}")
        
        return check
    
    def run_data_quality_checks(
        self,
        data: pd.DataFrame,
        model_name: str
    ) -> Dict:
        """
        Run data quality checks on data.
        
        Args:
            data: DataFrame to check
            model_name: Name of the model
        
        Returns:
            Dictionary with check results
        """
        logger.info(f"Running data quality checks for {model_name}")
        
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not found")
        
        quality_checks = self.models[model_name].get('quality_checks', [])
        results = []
        
        for check in quality_checks:
            check_type = check['type']
            config = check['config']
            
            if check_type == 'not_null':
                column = config['column']
                passed = data[column].notna().all()
                results.append({
                    'check_type': check_type,
                    'column': column,
                    'passed': passed,
                    'null_count': int(data[column].isna().sum())
                })
            
            elif check_type == 'unique':
                column = config['column']
                passed = data[column].nunique() == len(data)
                results.append({
                    'check_type': check_type,
                    'column': column,
                    'passed': passed,
                    'duplicate_count': int(len(data) - data[column].nunique())
                })
            
            elif check_type == 'range':
                column = config['column']
                min_val = config.get('min', -np.inf)
                max_val = config.get('max', np.inf)
                passed = ((data[column] >= min_val) & (data[column] <= max_val)).all()
                results.append({
                    'check_type': check_type,
                    'column': column,
                    'passed': passed,
                    'out_of_range_count': int(~((data[column] >= min_val) & (data[column] <= max_val)).sum())
                })
        
        logger.info(f"Data quality checks complete for {model_name}")
        
        return {'model_name': model_name, 'checks': results}
    
    def generate_lineage(self) -> Dict:
        """
        Generate data lineage graph.
        
        Returns:
            Dictionary with lineage information
        """
        logger.info("Generating data lineage...")
        
        lineage = {
            'models': list(self.models.keys()),
            'transformations': list(self.transformations.keys()),
            'dependencies': self.dependencies
        }
        
        logger.info("Data lineage generated")
        
        return lineage
    
    def save_project(self):
        """Save DBT project to disk."""
        project_file = self.project_path / "project.json"
        
        project_data = {
            'models': self.models,
            'transformations': self.transformations,
            'dependencies': self.dependencies
        }
        
        with open(project_file, 'w') as f:
            json.dump(project_data, f, indent=2)
        
        logger.info(f"DBT project saved to {project_file}")
    
    def load_project(self):
        """Load DBT project from disk."""
        project_file = self.project_path / "project.json"
        
        if project_file.exists():
            with open(project_file, 'r') as f:
                project_data = json.load(f)
            
            self.models = project_data.get('models', {})
            self.transformations = project_data.get('transformations', {})
            self.dependencies = project_data.get('dependencies', {})
            
            logger.info(f"DBT project loaded from {project_file}")


def run_dbt_pipeline(
    data_sources: Dict[str, pd.DataFrame]
) -> Tuple[DBTPipeline, Dict]:
    """
    Convenience function to run DBT pipeline.
    
    Args:
        data_sources: Dictionary of source data
    
    Returns:
        Tuple of (pipeline, results)
    """
    pipeline = DBTPipeline()
    
    # Define a sample model
    pipeline.define_model(
        'customer_orders',
        'orders',
        ['customer_id', 'order_id', 'order_total', 'order_date'],
        'Customer orders model'
    )
    
    # Create a sample transformation
    pipeline.create_transformation(
        'aggregate_customer_spend',
        'SELECT customer_id, SUM(order_total) as total_spend FROM customer_orders GROUP BY customer_id',
        ['customer_orders'],
        'Aggregate customer spend'
    )
    
    # Run pipeline
    results = pipeline.run_pipeline(data_sources)
    
    # Generate lineage
    lineage = pipeline.generate_lineage()
    
    pipeline_results = {
        'transformation_results': results,
        'lineage': lineage
    }
    
    return pipeline, pipeline_results
