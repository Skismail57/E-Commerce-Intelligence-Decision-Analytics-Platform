"""
Great Expectations Configuration for Data Quality
Defines expectations and validation rules for all data tables.
"""

import great_expectations as gx
from great_expectations.core import ExpectationSuite, ExpectationConfiguration
import pandas as pd
from datetime import datetime
from typing import Dict, Any
import json


class GreatExpectationsValidator:
    """
    Great Expectations validator for data quality checks.
    
    Provides comprehensive data validation with detailed expectations
    for all data tables in the platform.
    """
    
    def __init__(self, context_root: str = "gx"):
        """
        Initialize Great Expectations context.
        
        Args:
            context_root: Root directory for Great Expectations context
        """
        self.context = gx.get_context(context_root_dir=context_root)
        self.suites = {}
        self._initialize_suites()
    
    def _initialize_suites(self):
        """Initialize expectation suites for all tables"""
        self.suites = {
            'customers': self._create_customer_suite(),
            'products': self._create_product_suite(),
            'orders': self._create_order_suite(),
            'order_items': self._create_order_item_suite(),
            'rfm_segments': self._create_rfm_suite(),
            'clv_predictions': self._create_clv_suite(),
            'churn_features': self._create_churn_feature_suite(),
            'churn_predictions': self._create_churn_prediction_suite(),
            'forecast_history': self._create_forecast_history_suite(),
            'forecast_predictions': self._create_forecast_prediction_suite(),
        }
    
    def _create_customer_suite(self) -> ExpectationSuite:
        """Create expectation suite for customers table"""
        suite = ExpectationSuite(
            expectation_suite_name="customers_suite",
            data_asset_type="Dataset"
        )
        
        expectations = [
            ExpectationConfiguration(
                expectation_type="expect_column_to_exist",
                kwargs={"column": "customer_id"}
            ),
            ExpectationConfiguration(
                expectation_type="expect_column_to_exist",
                kwargs={"column": "customer_name"}
            ),
            ExpectationConfiguration(
                expectation_type="expect_column_to_exist",
                kwargs={"column": "email"}
            ),
            ExpectationConfiguration(
                expectation_type="expect_column_values_to_be_unique",
                kwargs={"column": "customer_id"}
            ),
            ExpectationConfiguration(
                expectation_type="expect_column_values_to_not_be_null",
                kwargs={"column": "customer_id"}
            ),
            ExpectationConfiguration(
                expectation_type="expect_column_values_to_not_be_null",
                kwargs={"column": "customer_name"}
            ),
            ExpectationConfiguration(
                expectation_type="expect_column_values_to_not_be_null",
                kwargs={"column": "email"}
            ),
            ExpectationConfiguration(
                expectation_type="expect_column_values_to_match_regex",
                kwargs={"column": "email", "regex": r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$"}
            ),
            ExpectationConfiguration(
                expectation_type="expect_column_values_to_be_in_set",
                kwargs={"column": "country", "value_set": ["India", "USA", "UK", "Canada", "Australia"]}
            ),
            ExpectationConfiguration(
                expectation_type="expect_column_values_to_be_between",
                kwargs={"column": "customer_id", "min_value": 1}
            ),
        ]
        
        suite.add_expectations(expectations)
        return suite
    
    def _create_product_suite(self) -> ExpectationSuite:
        """Create expectation suite for products table"""
        suite = ExpectationSuite(
            expectation_suite_name="products_suite",
            data_asset_type="Dataset"
        )
        
        expectations = [
            ExpectationConfiguration(
                expectation_type="expect_column_to_exist",
                kwargs={"column": "product_id"}
            ),
            ExpectationConfiguration(
                expectation_type="expect_column_values_to_be_unique",
                kwargs={"column": "product_id"}
            ),
            ExpectationConfiguration(
                expectation_type="expect_column_values_to_not_be_null",
                kwargs={"column": "product_name"}
            ),
            ExpectationConfiguration(
                expectation_type="expect_column_values_to_not_be_null",
                kwargs={"column": "selling_price"}
            ),
            ExpectationConfiguration(
                expectation_type="expect_column_values_to_be_between",
                kwargs={"column": "selling_price", "min_value": 0}
            ),
            ExpectationConfiguration(
                expectation_type="expect_column_values_to_be_between",
                kwargs={"column": "cost_price", "min_value": 0}
            ),
            ExpectationConfiguration(
                expectation_type="expect_column_values_to_be_between",
                kwargs={"column": "stock_quantity", "min_value": 0}
            ),
            ExpectationConfiguration(
                expectation_type="expect_column_values_to_not_be_null",
                kwargs={"column": "category_id"}
            ),
        ]
        
        suite.add_expectations(expectations)
        return suite
    
    def _create_order_suite(self) -> ExpectationSuite:
        """Create expectation suite for orders table"""
        suite = ExpectationSuite(
            expectation_suite_name="orders_suite",
            data_asset_type="Dataset"
        )
        
        expectations = [
            ExpectationConfiguration(
                expectation_type="expect_column_to_exist",
                kwargs={"column": "order_id"}
            ),
            ExpectationConfiguration(
                expectation_type="expect_column_values_to_be_unique",
                kwargs={"column": "order_id"}
            ),
            ExpectationConfiguration(
                expectation_type="expect_column_values_to_not_be_null",
                kwargs={"column": "customer_id"}
            ),
            ExpectationConfiguration(
                expectation_type="expect_column_values_to_not_be_null",
                kwargs={"column": "order_date"}
            ),
            ExpectationConfiguration(
                expectation_type="expect_column_values_to_not_be_null",
                kwargs={"column": "order_total"}
            ),
            ExpectationConfiguration(
                expectation_type="expect_column_values_to_be_between",
                kwargs={"column": "order_total", "min_value": 0}
            ),
            ExpectationConfiguration(
                expectation_type="expect_column_values_to_be_in_set",
                kwargs={
                    "column": "order_status",
                    "value_set": ["Delivered", "Shipped", "Processing", "Cancelled", "Returned"]
                }
            ),
            ExpectationConfiguration(
                expectation_type="expect_column_values_to_not_be_null",
                kwargs={"column": "order_status"}
            ),
        ]
        
        suite.add_expectations(expectations)
        return suite
    
    def _create_order_item_suite(self) -> ExpectationSuite:
        """Create expectation suite for order_items table"""
        suite = ExpectationSuite(
            expectation_suite_name="order_items_suite",
            data_asset_type="Dataset"
        )
        
        expectations = [
            ExpectationConfiguration(
                expectation_type="expect_column_to_exist",
                kwargs={"column": "order_item_id"}
            ),
            ExpectationConfiguration(
                expectation_type="expect_column_values_to_be_unique",
                kwargs={"column": "order_item_id"}
            ),
            ExpectationConfiguration(
                expectation_type="expect_column_values_to_not_be_null",
                kwargs={"column": "order_id"}
            ),
            ExpectationConfiguration(
                expectation_type="expect_column_values_to_not_be_null",
                kwargs={"column": "product_id"}
            ),
            ExpectationConfiguration(
                expectation_type="expect_column_values_to_be_between",
                kwargs={"column": "quantity", "min_value": 1}
            ),
            ExpectationConfiguration(
                expectation_type="expect_column_values_to_be_between",
                kwargs={"column": "unit_price", "min_value": 0}
            ),
            ExpectationConfiguration(
                expectation_type="expect_column_values_to_be_between",
                kwargs={"column": "discount_pct", "min_value": 0, "max_value": 100}
            ),
        ]
        
        suite.add_expectations(expectations)
        return suite
    
    def _create_rfm_suite(self) -> ExpectationSuite:
        """Create expectation suite for RFM segments table"""
        suite = ExpectationSuite(
            expectation_suite_name="rfm_segments_suite",
            data_asset_type="Dataset"
        )
        
        expectations = [
            ExpectationConfiguration(
                expectation_type="expect_column_to_exist",
                kwargs={"column": "customer_id"}
            ),
            ExpectationConfiguration(
                expectation_type="expect_column_values_to_be_unique",
                kwargs={"column": "customer_id"}
            ),
            ExpectationConfiguration(
                expectation_type="expect_column_values_to_be_between",
                kwargs={"column": "recency_days", "min_value": 0}
            ),
            ExpectationConfiguration(
                expectation_type="expect_column_values_to_be_between",
                kwargs={"column": "frequency", "min_value": 0}
            ),
            ExpectationConfiguration(
                expectation_type="expect_column_values_to_be_between",
                kwargs={"column": "monetary", "min_value": 0}
            ),
            ExpectationConfiguration(
                expectation_type="expect_column_values_to_be_between",
                kwargs={"column": "recency_score", "min_value": 1, "max_value": 5}
            ),
            ExpectationConfiguration(
                expectation_type="expect_column_values_to_be_between",
                kwargs={"column": "frequency_score", "min_value": 1, "max_value": 5}
            ),
            ExpectationConfiguration(
                expectation_type="expect_column_values_to_be_between",
                kwargs={"column": "monetary_score", "min_value": 1, "max_value": 5}
            ),
            ExpectationConfiguration(
                expectation_type="expect_column_values_to_be_in_set",
                kwargs={
                    "column": "segment",
                    "value_set": ['Champions', 'Loyal Customers', 'Potential Loyalist', 'New Customers',
                                'Promising', 'Need Attention', 'About to Sleep', 'At Risk',
                                'Cannot Lose Them', 'Hibernating', 'Lost']
                }
            ),
        ]
        
        suite.add_expectations(expectations)
        return suite
    
    def _create_clv_suite(self) -> ExpectationSuite:
        """Create expectation suite for CLV predictions table"""
        suite = ExpectationSuite(
            expectation_suite_name="clv_predictions_suite",
            data_asset_type="Dataset"
        )
        
        expectations = [
            ExpectationConfiguration(
                expectation_type="expect_column_to_exist",
                kwargs={"column": "customer_id"}
            ),
            ExpectationConfiguration(
                expectation_type="expect_column_values_to_be_unique",
                kwargs={"column": "customer_id"}
            ),
            ExpectationConfiguration(
                expectation_type="expect_column_values_to_be_between",
                kwargs={"column": "predicted_clv_90d", "min_value": 0}
            ),
            ExpectationConfiguration(
                expectation_type="expect_column_values_to_be_between",
                kwargs={"column": "predicted_clv_180d", "min_value": 0}
            ),
            ExpectationConfiguration(
                expectation_type="expect_column_values_to_be_between",
                kwargs={"column": "predicted_clv_365d", "min_value": 0}
            ),
            ExpectationConfiguration(
                expectation_type="expect_column_values_to_be_between",
                kwargs={"column": "historical_clv", "min_value": 0}
            ),
            ExpectationConfiguration(
                expectation_type="expect_column_values_to_be_in_set",
                kwargs={"column": "clv_tier", "value_set": ['Platinum', 'Gold', 'Silver', 'Bronze']}
            ),
        ]
        
        suite.add_expectations(expectations)
        return suite
    
    def _create_churn_feature_suite(self) -> ExpectationSuite:
        """Create expectation suite for churn features table"""
        suite = ExpectationSuite(
            expectation_suite_name="churn_features_suite",
            data_asset_type="Dataset"
        )
        
        expectations = [
            ExpectationConfiguration(
                expectation_type="expect_column_to_exist",
                kwargs={"column": "customer_id"}
            ),
            ExpectationConfiguration(
                expectation_type="expect_column_values_to_be_unique",
                kwargs={"column": "customer_id"}
            ),
            ExpectationConfiguration(
                expectation_type="expect_column_values_to_be_in_set",
                kwargs={"column": "churn_label_90d", "value_set": [0, 1]}
            ),
            ExpectationConfiguration(
                expectation_type="expect_column_values_to_be_between",
                kwargs={"column": "days_since_last_order", "min_value": 0}
            ),
            ExpectationConfiguration(
                expectation_type="expect_column_values_to_be_between",
                kwargs={"column": "order_frequency", "min_value": 0}
            ),
            ExpectationConfiguration(
                expectation_type="expect_column_values_to_be_between",
                kwargs={"column": "avg_order_value", "min_value": 0}
            ),
            ExpectationConfiguration(
                expectation_type="expect_column_values_to_be_between",
                kwargs={"column": "total_spend", "min_value": 0}
            ),
            ExpectationConfiguration(
                expectation_type="expect_column_values_to_be_between",
                kwargs={"column": "return_rate", "min_value": 0, "max_value": 1}
            ),
            ExpectationConfiguration(
                expectation_type="expect_column_values_to_be_between",
                kwargs={"column": "discount_usage_pct", "min_value": 0, "max_value": 1}
            ),
        ]
        
        suite.add_expectations(expectations)
        return suite
    
    def _create_churn_prediction_suite(self) -> ExpectationSuite:
        """Create expectation suite for churn predictions table"""
        suite = ExpectationSuite(
            expectation_suite_name="churn_predictions_suite",
            data_asset_type="Dataset"
        )
        
        expectations = [
            ExpectationConfiguration(
                expectation_type="expect_column_to_exist",
                kwargs={"column": "customer_id"}
            ),
            ExpectationConfiguration(
                expectation_type="expect_column_values_to_be_unique",
                kwargs={"column": "customer_id"}
            ),
            ExpectationConfiguration(
                expectation_type="expect_column_values_to_be_between",
                kwargs={"column": "churn_probability", "min_value": 0, "max_value": 1}
            ),
            ExpectationConfiguration(
                expectation_type="expect_column_values_to_be_in_set",
                kwargs={"column": "risk_tier", "value_set": ['High', 'Medium', 'Low']}
            ),
            ExpectationConfiguration(
                expectation_type="expect_column_values_to_be_between",
                kwargs={"column": "risk_score", "min_value": 0, "max_value": 1000}
            ),
        ]
        
        suite.add_expectations(expectations)
        return suite
    
    def _create_forecast_history_suite(self) -> ExpectationSuite:
        """Create expectation suite for forecast history table"""
        suite = ExpectationSuite(
            expectation_suite_name="forecast_history_suite",
            data_asset_type="Dataset"
        )
        
        expectations = [
            ExpectationConfiguration(
                expectation_type="expect_column_to_exist",
                kwargs={"column": "date"}
            ),
            ExpectationConfiguration(
                expectation_type="expect_column_values_to_not_be_null",
                kwargs={"column": "date"}
            ),
            ExpectationConfiguration(
                expectation_type="expect_column_values_to_be_between",
                kwargs={"column": "actual_units", "min_value": 0}
            ),
            ExpectationConfiguration(
                expectation_type="expect_column_values_to_be_between",
                kwargs={"column": "actual_revenue", "min_value": 0}
            ),
        ]
        
        suite.add_expectations(expectations)
        return suite
    
    def _create_forecast_prediction_suite(self) -> ExpectationSuite:
        """Create expectation suite for forecast predictions table"""
        suite = ExpectationSuite(
            expectation_suite_name="forecast_predictions_suite",
            data_asset_type="Dataset"
        )
        
        expectations = [
            ExpectationConfiguration(
                expectation_type="expect_column_to_exist",
                kwargs={"column": "date"}
            ),
            ExpectationConfiguration(
                expectation_type="expect_column_values_to_not_be_null",
                kwargs={"column": "date"}
            ),
            ExpectationConfiguration(
                expectation_type="expect_column_values_to_be_between",
                kwargs={"column": "forecast_units", "min_value": 0}
            ),
            ExpectationConfiguration(
                expectation_type="expect_column_values_to_be_between",
                kwargs={"column": "forecast_revenue", "min_value": 0}
            ),
            ExpectationConfiguration(
                expectation_type="expect_column_values_to_be_between",
                kwargs={"column": "lower_bound", "min_value": 0}
            ),
            ExpectationConfiguration(
                expectation_type="expect_column_values_to_be_between",
                kwargs={"column": "upper_bound", "min_value": 0}
            ),
            ExpectationConfiguration(
                expectation_type="expect_column_values_to_not_be_null",
                kwargs={"column": "model_type"}
            ),
        ]
        
        suite.add_expectations(expectations)
        return suite
    
    def validate_dataframe(
        self,
        df: pd.DataFrame,
        table_name: str,
        run_name: str = None
    ) -> Dict[str, Any]:
        """
        Validate a DataFrame against its expectation suite.
        
        Args:
            df: DataFrame to validate
            table_name: Name of the table/suite
            run_name: Optional name for the validation run
        
        Returns:
            Dictionary with validation results
        """
        if table_name not in self.suites:
            return {
                'success': False,
                'error': f"No expectation suite found for table: {table_name}",
                'table_name': table_name
            }
        
        suite = self.suites[table_name]
        
        try:
            batch = gx.dataset.PandasDataset(df)
            results = batch.validate(
                expectation_suite=suite,
                run_name=run_name or f"{table_name}_validation_{datetime.now().isoformat()}"
            )
            
            return {
                'success': results.success,
                'table_name': table_name,
                'statistics': results.statistics,
                'result_count': len(results.results),
                'passed_count': sum(1 for r in results.results if r.success),
                'failed_count': sum(1 for r in results.results if not r.success),
                'validation_time': datetime.now().isoformat(),
                'results': [
                    {
                        'expectation': r.expectation_config.expectation_type,
                        'success': r.success,
                        'column': r.expectation_config.kwargs.get('column'),
                        'exception': r.exception_info if not r.success else None
                    }
                    for r in results.results
                ]
            }
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'table_name': table_name,
                'validation_time': datetime.now().isoformat()
            }
    
    def validate_all(
        self,
        data_dict: Dict[str, pd.DataFrame],
        run_name: str = None
    ) -> Dict[str, Any]:
        """
        Validate multiple DataFrames against their expectation suites.
        
        Args:
            data_dict: Dictionary of table_name -> DataFrame
            run_name: Optional name for the validation run
        
        Returns:
            Dictionary with validation results for all tables
        """
        results = {}
        
        for table_name, df in data_dict.items():
            results[table_name] = self.validate_dataframe(df, table_name, run_name)
        
        # Calculate overall statistics
        total_tables = len(results)
        valid_tables = sum(1 for r in results.values() if r.get('success', False))
        
        return {
            'validation_timestamp': datetime.now().isoformat(),
            'run_name': run_name,
            'total_tables': total_tables,
            'valid_tables': valid_tables,
            'invalid_tables': total_tables - valid_tables,
            'overall_success_rate': (valid_tables / total_tables * 100) if total_tables > 0 else 0,
            'table_results': results
        }
    
    def add_suite(self, table_name: str, suite: ExpectationSuite):
        """
        Add a custom expectation suite.
        
        Args:
            table_name: Name for the suite
            suite: ExpectationSuite to add
        """
        self.suites[table_name] = suite
    
    def get_suite(self, table_name: str) -> ExpectationSuite:
        """Get expectation suite for a table"""
        if table_name not in self.suites:
            raise ValueError(f"No expectation suite found for table: {table_name}")
        return self.suites[table_name]


# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================

def create_data_quality_checkpoint(
    validator: GreatExpectationsValidator,
    table_name: str
):
    """
    Create a Great Expectations checkpoint for a table.
    
    Args:
        validator: GreatExpectationsValidator instance
        table_name: Name of the table
    """
    suite = validator.get_suite(table_name)
    
    # Add suite to context
    validator.context.add_or_update_expectation_suite(expectation_suite=suite)
    
    # Create checkpoint configuration
    checkpoint_config = {
        "name": f"{table_name}_checkpoint",
        "config_version": 1.0,
        "class_name": "SimpleCheckpoint",
        "run_name_template": "%Y%m%d-%H%M%S-" + table_name,
        "validations": [
            {
                "batch_name": table_name,
                "expectation_suite_name": f"{table_name}_suite",
            }
        ]
    }
    
    return checkpoint_config


def generate_data_quality_report(
    validation_results: Dict[str, Any]
) -> str:
    """
    Generate a human-readable data quality report.
    
    Args:
        validation_results: Results from validate_all()
    
    Returns:
        Formatted report string
    """
    report_lines = [
        "=" * 80,
        "DATA QUALITY VALIDATION REPORT",
        "=" * 80,
        f"Validation Timestamp: {validation_results['validation_timestamp']}",
        f"Run Name: {validation_results.get('run_name', 'N/A')}",
        "",
        f"Total Tables: {validation_results['total_tables']}",
        f"Valid Tables: {validation_results['valid_tables']}",
        f"Invalid Tables: {validation_results['invalid_tables']}",
        f"Overall Success Rate: {validation_results['overall_success_rate']:.2f}%",
        "",
        "-" * 80,
        "TABLE RESULTS",
        "-" * 80,
    ]
    
    for table_name, result in validation_results['table_results'].items():
        report_lines.append(f"\nTable: {table_name}")
        report_lines.append(f"Status: {'✓ PASSED' if result.get('success') else '✗ FAILED'}")
        
        if result.get('success'):
            report_lines.append(f"Passed Expectations: {result.get('passed_count', 0)}")
            report_lines.append(f"Failed Expectations: {result.get('failed_count', 0)}")
        else:
            report_lines.append(f"Error: {result.get('error', 'Unknown error')}")
        
        if 'results' in result:
            failed_results = [r for r in result['results'] if not r['success']]
            if failed_results:
                report_lines.append("\nFailed Expectations:")
                for fr in failed_results:
                    report_lines.append(f"  - {fr['expectation']} on column '{fr.get('column', 'N/A')}'")
                    if fr.get('exception'):
                        report_lines.append(f"    Exception: {fr['exception']}")
    
    report_lines.append("\n" + "=" * 80)
    
    return "\n".join(report_lines)
