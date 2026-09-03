"""
Data Quality Schemas using Pandera
Defines data contracts and validation rules for all data tables.
"""

import pandera as pa
from pandera.typing import Series, DateTime, Int, Float, String
import pandas as pd
from datetime import datetime


# ============================================================================
# CUSTOMER SCHEMAS
# ============================================================================

class CustomerSchema(pa.DataFrameModel):
    """Schema for customers table - aligned with canonical schema"""
    
    customer_id: Series[Int] = pa.Field(
        unique=True,
        coerce=True,
        description="Unique customer identifier"
    )
    
    first_name: Series[String] = pa.Field(
        nullable=False,
        description="Customer first name"
    )
    
    last_name: Series[String] = pa.Field(
        nullable=False,
        description="Customer last name"
    )
    
    email: Series[String] = pa.Field(
        nullable=False,
        description="Customer email address"
    )
    
    phone: Series[String] = pa.Field(
        nullable=True,
        description="Customer phone number"
    )
    
    gender: Series[String] = pa.Field(
        nullable=True,
        isin=['M', 'F', 'Other'],
        description="Customer gender"
    )
    
    age: Series[Int] = pa.Field(
        nullable=True,
        ge=18,
        le=120,
        coerce=True,
        description="Customer age"
    )
    
    state: Series[String] = pa.Field(
        nullable=False,
        description="Customer state"
    )
    
    city: Series[String] = pa.Field(
        nullable=False,
        description="Customer city"
    )
    
    signup_date: Series[DateTime] = pa.Field(
        nullable=False,
        coerce=True,
        description="Customer signup date"
    )
    
    customer_segment: Series[String] = pa.Field(
        nullable=False,
        isin=['Premium', 'Standard', 'Budget'],
        description="Customer segment"
    )
    
    class Config:
        strict = True
        coerce = True


# ============================================================================
# PRODUCT SCHEMAS
# ============================================================================

class ProductSchema(pa.DataFrameModel):
    """Schema for products table - aligned with canonical schema"""
    
    product_id: Series[Int] = pa.Field(
        unique=True,
        coerce=True,
        description="Unique product identifier"
    )
    
    product_name: Series[String] = pa.Field(
        nullable=False,
        description="Product name"
    )
    
    category_id: Series[Int] = pa.Field(
        nullable=False,
        coerce=True,
        description="Product category identifier"
    )
    
    brand_name: Series[String] = pa.Field(
        nullable=True,
        description="Product brand"
    )
    
    selling_price: Series[Float] = pa.Field(
        ge=0,
        coerce=True,
        description="Current selling price"
    )
    
    cost_price: Series[Float] = pa.Field(
        ge=0,
        coerce=True,
        description="Cost price"
    )
    
    units_in_stock: Series[Int] = pa.Field(
        ge=0,
        coerce=True,
        description="Available inventory"
    )
    
    class Config:
        strict = True
        coerce = True


# ============================================================================
# ORDER SCHEMAS
# ============================================================================

class OrderSchema(pa.DataFrameModel):
    """Schema for orders table - aligned with canonical schema"""
    
    order_id: Series[Int] = pa.Field(
        unique=True,
        coerce=True,
        description="Unique order identifier"
    )
    
    customer_id: Series[Int] = pa.Field(
        nullable=False,
        coerce=True,
        description="Customer who placed order"
    )
    
    order_date: Series[DateTime] = pa.Field(
        nullable=False,
        coerce=True,
        description="Order timestamp"
    )
    
    order_status: Series[String] = pa.Field(
        isin=['Pending', 'Processing', 'Shipped', 'Delivered', 'Cancelled', 'Returned'],
        nullable=False,
        description="Order status"
    )
    
    order_total: Series[Float] = pa.Field(
        ge=0,
        coerce=True,
        description="Total order amount"
    )
    
    discount_amount: Series[Float] = pa.Field(
        nullable=True,
        ge=0,
        coerce=True,
        description="Discount applied"
    )
    
    payment_method: Series[String] = pa.Field(
        nullable=True,
        description="Payment method"
    )
    
    class Config:
        strict = True
        coerce = True


# ============================================================================
# ORDER ITEM SCHEMAS
# ============================================================================

class OrderItemSchema(pa.DataFrameModel):
    """Schema for order_items table - aligned with canonical schema"""
    
    order_item_id: Series[Int] = pa.Field(
        unique=True,
        coerce=True,
        description="Unique line item identifier"
    )
    
    order_id: Series[Int] = pa.Field(
        nullable=False,
        coerce=True,
        description="Parent order"
    )
    
    product_id: Series[Int] = pa.Field(
        nullable=False,
        coerce=True,
        description="Product ordered"
    )
    
    quantity: Series[Int] = pa.Field(
        ge=1,
        coerce=True,
        description="Quantity ordered"
    )
    
    unit_price: Series[Float] = pa.Field(
        ge=0,
        coerce=True,
        description="Unit price at time of order"
    )
    
    line_total: Series[Float] = pa.Field(
        ge=0,
        coerce=True,
        description="Line item total"
    )
    
    class Config:
        strict = True
        coerce = True


# ============================================================================
# RFM SEGMENT SCHEMAS
# ============================================================================

class RFMSegmentSchema(pa.DataFrameModel):
    """Schema for RFM segmentation results"""
    
    customer_id: Series[Int] = pa.Field(
        unique=True,
        coerce=True,
        description="Customer identifier"
    )
    
    recency_days: Series[Int] = pa.Field(
        ge=0,
        coerce=True,
        description="Days since last purchase"
    )
    
    frequency: Series[Int] = pa.Field(
        ge=0,
        coerce=True,
        description="Number of purchases"
    )
    
    monetary: Series[Float] = pa.Field(
        ge=0,
        coerce=True,
        description="Total monetary value"
    )
    
    recency_score: Series[Int] = pa.Field(
        ge=1,
        le=5,
        coerce=True,
        description="Recency score (1-5)"
    )
    
    frequency_score: Series[Int] = pa.Field(
        ge=1,
        le=5,
        coerce=True,
        description="Frequency score (1-5)"
    )
    
    monetary_score: Series[Int] = pa.Field(
        ge=1,
        le=5,
        coerce=True,
        description="Monetary score (1-5)"
    )
    
    rfm_score: Series[Int] = pa.Field(
        ge=111,
        le=555,
        coerce=True,
        description="Combined RFM score"
    )
    
    segment: Series[String] = pa.Field(
        isin=['Champions', 'Loyal Customers', 'Potential Loyalist', 'New Customers',
              'Promising', 'Need Attention', 'About to Sleep', 'At Risk',
              'Cannot Lose Them', 'Hibernating', 'Lost'],
        nullable=False,
        description="RFM segment"
    )
    
    class Config:
        strict = True
        coerce = True


# ============================================================================
# CLV SCHEMAS
# ============================================================================

class CLVPredictionSchema(pa.DataFrameModel):
    """Schema for CLV predictions"""
    
    customer_id: Series[Int] = pa.Field(
        unique=True,
        coerce=True,
        description="Customer identifier"
    )
    
    predicted_clv_90d: Series[Float] = pa.Field(
        ge=0,
        coerce=True,
        description="Predicted 90-day CLV"
    )
    
    predicted_clv_180d: Series[Float] = pa.Field(
        ge=0,
        coerce=True,
        description="Predicted 180-day CLV"
    )
    
    predicted_clv_365d: Series[Float] = pa.Field(
        ge=0,
        coerce=True,
        description="Predicted 365-day CLV"
    )
    
    historical_clv: Series[Float] = pa.Field(
        ge=0,
        coerce=True,
        description="Historical CLV"
    )
    
    clv_tier: Series[String] = pa.Field(
        isin=['Platinum', 'Gold', 'Silver', 'Bronze'],
        nullable=False,
        description="CLV tier"
    )
    
    prediction_date: Series[DateTime] = pa.Field(
        nullable=False,
        coerce=True,
        description="Prediction date"
    )
    
    class Config:
        strict = True
        coerce = True


# ============================================================================
# CHURN SCHEMAS
# ============================================================================

class ChurnFeatureSchema(pa.DataFrameModel):
    """Schema for churn feature dataset"""
    
    customer_id: Series[Int] = pa.Field(
        unique=True,
        coerce=True,
        description="Customer identifier"
    )
    
    churn_label_90d: Series[Int] = pa.Field(
        ge=0,
        le=1,
        coerce=True,
        description="Churn label (0=no, 1=yes) for 90 days"
    )
    
    days_since_last_order: Series[Int] = pa.Field(
        ge=0,
        coerce=True,
        description="Days since last order"
    )
    
    order_frequency: Series[Float] = pa.Field(
        ge=0,
        coerce=True,
        description="Order frequency"
    )
    
    avg_order_value: Series[Float] = pa.Field(
        ge=0,
        coerce=True,
        description="Average order value"
    )
    
    total_spend: Series[Float] = pa.Field(
        ge=0,
        coerce=True,
        description="Total spend"
    )
    
    return_rate: Series[Float] = pa.Field(
        ge=0,
        le=1,
        coerce=True,
        description="Return rate"
    )
    
    discount_usage_pct: Series[Float] = pa.Field(
        ge=0,
        le=1,
        coerce=True,
        description="Discount usage percentage"
    )
    
    class Config:
        strict = True
        coerce = True


class ChurnPredictionSchema(pa.DataFrameModel):
    """Schema for churn predictions"""
    
    customer_id: Series[Int] = pa.Field(
        unique=True,
        coerce=True,
        description="Customer identifier"
    )
    
    churn_probability: Series[Float] = pa.Field(
        ge=0,
        le=1,
        coerce=True,
        description="Churn probability"
    )
    
    risk_tier: Series[String] = pa.Field(
        isin=['High', 'Medium', 'Low'],
        nullable=False,
        description="Risk tier"
    )
    
    risk_score: Series[Int] = pa.Field(
        ge=0,
        le=1000,
        coerce=True,
        description="Risk score (0-1000)"
    )
    
    prediction_date: Series[DateTime] = pa.Field(
        nullable=False,
        coerce=True,
        description="Prediction date"
    )
    
    class Config:
        strict = True
        coerce = True


# ============================================================================
# FORECASTING SCHEMAS
# ============================================================================

class ForecastHistorySchema(pa.DataFrameModel):
    """Schema for forecast historical data"""
    
    date: Series[DateTime] = pa.Field(
        nullable=False,
        coerce=True,
        description="Date"
    )
    
    actual_units: Series[Int] = pa.Field(
        ge=0,
        coerce=True,
        description="Actual units sold"
    )
    
    actual_revenue: Series[Float] = pa.Field(
        ge=0,
        coerce=True,
        description="Actual revenue"
    )
    
    class Config:
        strict = True
        coerce = True


class ForecastPredictionSchema(pa.DataFrameModel):
    """Schema for forecast predictions"""
    
    date: Series[DateTime] = pa.Field(
        nullable=False,
        coerce=True,
        description="Forecast date"
    )
    
    forecast_units: Series[Float] = pa.Field(
        ge=0,
        coerce=True,
        description="Forecasted units"
    )
    
    forecast_revenue: Series[Float] = pa.Field(
        ge=0,
        coerce=True,
        description="Forecasted revenue"
    )
    
    lower_bound: Series[Float] = pa.Field(
        ge=0,
        coerce=True,
        description="Lower confidence bound"
    )
    
    upper_bound: Series[Float] = pa.Field(
        ge=0,
        coerce=True,
        description="Upper confidence bound"
    )
    
    model_type: Series[String] = pa.Field(
        nullable=False,
        description="Model type used for forecast"
    )
    
    class Config:
        strict = True
        coerce = True


# ============================================================================
# RECOMMENDATION SCHEMAS
# ============================================================================

class RecommendationSchema(pa.DataFrameModel):
    """Schema for product recommendations"""
    
    customer_id: Series[Int] = pa.Field(
        nullable=False,
        coerce=True,
        description="Customer identifier"
    )
    
    product_id: Series[Int] = pa.Field(
        nullable=False,
        coerce=True,
        description="Recommended product identifier"
    )
    
    score: Series[Float] = pa.Field(
        ge=0,
        le=1,
        coerce=True,
        description="Recommendation score"
    )
    
    rank: Series[Int] = pa.Field(
        ge=1,
        coerce=True,
        description="Recommendation rank"
    )
    
    strategy: Series[String] = pa.Field(
        isin=['collaborative', 'content_based', 'hybrid', 'popular'],
        nullable=False,
        description="Recommendation strategy"
    )
    
    class Config:
        strict = True
        coerce = True


# ============================================================================
# DATA QUALITY VALIDATOR
# ============================================================================

class DataQualityValidator:
    """
    Central data quality validator using Pandera schemas.
    
    Provides validation for all data tables and generates quality reports.
    """
    
    def __init__(self):
        """Initialize validator with all schemas"""
        self.schemas = {
            'customers': CustomerSchema,
            'products': ProductSchema,
            'orders': OrderSchema,
            'order_items': OrderItemSchema,
            'rfm_segments': RFMSegmentSchema,
            'clv_predictions': CLVPredictionSchema,
            'churn_features': ChurnFeatureSchema,
            'churn_predictions': ChurnPredictionSchema,
            'forecast_history': ForecastHistorySchema,
            'forecast_predictions': ForecastPredictionSchema,
            'recommendations': RecommendationSchema,
        }
    
    def validate(self, df: pd.DataFrame, table_name: str) -> tuple[bool, str]:
        """
        Validate a DataFrame against its schema.
        
        Args:
            df: DataFrame to validate
            table_name: Name of the table/schema to validate against
        
        Returns:
            Tuple of (is_valid, error_message)
        """
        if table_name not in self.schemas:
            return False, f"No schema found for table: {table_name}"
        
        schema = self.schemas[table_name]
        
        try:
            validated_df = schema.validate(df)
            return True, "Validation passed"
        except pa.errors.SchemaError as e:
            return False, str(e)
        except Exception as e:
            return False, f"Unexpected error: {str(e)}"
    
    def get_schema(self, table_name: str) -> pa.DataFrameModel:
        """Get schema for a specific table"""
        if table_name not in self.schemas:
            raise ValueError(f"No schema found for table: {table_name}")
        return self.schemas[table_name]
    
    def validate_all(self, data_dict: dict[str, pd.DataFrame]) -> dict[str, tuple[bool, str]]:
        """
        Validate multiple DataFrames against their schemas.
        
        Args:
            data_dict: Dictionary of table_name -> DataFrame
        
        Returns:
            Dictionary of table_name -> (is_valid, error_message)
        """
        results = {}
        for table_name, df in data_dict.items():
            results[table_name] = self.validate(df, table_name)
        return results
    
    def generate_quality_report(self, data_dict: dict[str, pd.DataFrame]) -> dict:
        """
        Generate a comprehensive data quality report.
        
        Args:
            data_dict: Dictionary of table_name -> DataFrame
        
        Returns:
            Dictionary with quality metrics for each table
        """
        report = {
            'validation_timestamp': datetime.now().isoformat(),
            'tables': {}
        }
        
        validation_results = self.validate_all(data_dict)
        
        for table_name, (is_valid, error_msg) in validation_results.items():
            df = data_dict[table_name]
            
            table_report = {
                'is_valid': is_valid,
                'error_message': error_msg if not is_valid else None,
                'row_count': len(df),
                'column_count': len(df.columns),
                'missing_values': df.isnull().sum().to_dict(),
                'duplicate_rows': df.duplicated().sum(),
                'memory_usage_mb': df.memory_usage(deep=True).sum() / (1024 * 1024)
            }
            
            report['tables'][table_name] = table_report
        
        # Calculate overall quality score
        total_tables = len(validation_results)
        valid_tables = sum(1 for is_valid, _ in validation_results.values() if is_valid)
        report['overall_quality_score'] = (valid_tables / total_tables * 100) if total_tables > 0 else 0
        report['total_tables'] = total_tables
        report['valid_tables'] = valid_tables
        
        return report


# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================

def validate_dataframe(df: pd.DataFrame, schema: pa.DataFrameModel) -> tuple[bool, str, pd.DataFrame]:
    """
    Validate a DataFrame against a schema.
    
    Args:
        df: DataFrame to validate
        schema: Pandera schema to validate against
    
    Returns:
        Tuple of (is_valid, error_message, validated_dataframe)
    """
    try:
        validated_df = schema.validate(df)
        return True, "Validation passed", validated_df
    except pa.errors.SchemaError as e:
        return False, str(e), df
    except Exception as e:
        return False, f"Unexpected error: {str(e)}", df


def get_data_quality_summary(df: pd.DataFrame) -> dict:
    """
    Get a quick data quality summary for a DataFrame.
    
    Args:
        df: DataFrame to analyze
    
    Returns:
        Dictionary with quality metrics
    """
    return {
        'row_count': len(df),
        'column_count': len(df.columns),
        'missing_values': df.isnull().sum().sum(),
        'duplicate_rows': df.duplicated().sum(),
        'memory_usage_mb': df.memory_usage(deep=True).sum() / (1024 * 1024),
        'dtypes': df.dtypes.astype(str).to_dict()
    }
