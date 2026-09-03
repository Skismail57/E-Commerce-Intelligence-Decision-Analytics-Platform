"""
Data Quality Module
Provides data contracts, validation, and quality monitoring using Pandera and Great Expectations.
"""

from .schemas import (
    DataQualityValidator,
    validate_dataframe,
    get_data_quality_summary,
    CustomerSchema,
    ProductSchema,
    OrderSchema,
    OrderItemSchema,
    RFMSegmentSchema,
    CLVPredictionSchema,
    ChurnFeatureSchema,
    ChurnPredictionSchema,
    ForecastHistorySchema,
    ForecastPredictionSchema,
    RecommendationSchema,
)

from .great_expectations_config import (
    GreatExpectationsValidator,
    create_data_quality_checkpoint,
    generate_data_quality_report,
)

__all__ = [
    # Pandera schemas
    'CustomerSchema',
    'ProductSchema',
    'OrderSchema',
    'OrderItemSchema',
    'RFMSegmentSchema',
    'CLVPredictionSchema',
    'ChurnFeatureSchema',
    'ChurnPredictionSchema',
    'ForecastHistorySchema',
    'ForecastPredictionSchema',
    'RecommendationSchema',
    
    # Validators
    'DataQualityValidator',
    'GreatExpectationsValidator',
    
    # Convenience functions
    'validate_dataframe',
    'get_data_quality_summary',
    'create_data_quality_checkpoint',
    'generate_data_quality_report',
]
