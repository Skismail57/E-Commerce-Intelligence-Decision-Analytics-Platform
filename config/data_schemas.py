"""
Canonical Data Schemas
Single source of truth for all data table schemas across the platform.
Used for validation, documentation, and ensuring consistency across CSV, Pandas, PostgreSQL, and API layers.
"""

from typing import Dict, List, Any
from datetime import datetime
from enum import Enum


class DataType(Enum):
    """Supported data types for schema definition."""
    INTEGER = "integer"
    FLOAT = "float"
    STRING = "string"
    BOOLEAN = "boolean"
    DATE = "date"
    DATETIME = "datetime"
    JSON = "json"


class ColumnSchema:
    """Schema definition for a single column."""
    
    def __init__(
        self,
        name: str,
        data_type: DataType,
        nullable: bool = True,
        primary_key: bool = False,
        foreign_key: str = None,
        description: str = "",
        min_value: Any = None,
        max_value: Any = None,
        allowed_values: List[Any] = None,
        regex_pattern: str = None
    ):
        self.name = name
        self.data_type = data_type
        self.nullable = nullable
        self.primary_key = primary_key
        self.foreign_key = foreign_key
        self.description = description
        self.min_value = min_value
        self.max_value = max_value
        self.allowed_values = allowed_values
        self.regex_pattern = regex_pattern


class TableSchema:
    """Schema definition for a complete table."""
    
    def __init__(
        self,
        table_name: str,
        columns: List[ColumnSchema],
        description: str = "",
        indexes: List[str] = None,
        unique_constraints: List[List[str]] = None
    ):
        self.table_name = table_name
        self.columns = columns
        self.description = description
        self.indexes = indexes or []
        self.unique_constraints = unique_constraints or []
    
    def get_column(self, column_name: str) -> ColumnSchema:
        """Get column schema by name."""
        for col in self.columns:
            if col.name == column_name:
                return col
        raise ValueError(f"Column {column_name} not found in table {self.table_name}")
    
    def get_primary_keys(self) -> List[ColumnSchema]:
        """Get all primary key columns."""
        return [col for col in self.columns if col.primary_key]
    
    def get_foreign_keys(self) -> Dict[str, ColumnSchema]:
        """Get all foreign key columns mapped to their target tables."""
        return {col.name: col for col in self.columns if col.foreign_key}


# ============================================================================
# CANONICAL TABLE SCHEMAS
# ============================================================================

CUSTOMERS_SCHEMA = TableSchema(
    table_name="customers",
    description="Customer master data with demographic and segmentation information",
    columns=[
        ColumnSchema("customer_id", DataType.INTEGER, nullable=False, primary_key=True, description="Unique customer identifier"),
        ColumnSchema("first_name", DataType.STRING, nullable=False, description="Customer first name"),
        ColumnSchema("last_name", DataType.STRING, nullable=False, description="Customer last name"),
        ColumnSchema("email", DataType.STRING, nullable=False, description="Customer email address", regex_pattern=r"^[^@]+@[^@]+\.[^@]+$"),
        ColumnSchema("phone", DataType.STRING, nullable=True, description="Customer phone number"),
        ColumnSchema("gender", DataType.STRING, nullable=True, description="Customer gender", allowed_values=["M", "F", "Other"]),
        ColumnSchema("age", DataType.INTEGER, nullable=True, description="Customer age", min_value=18, max_value=120),
        ColumnSchema("state", DataType.STRING, nullable=False, description="Customer state/region"),
        ColumnSchema("city", DataType.STRING, nullable=False, description="Customer city"),
        ColumnSchema("signup_date", DataType.DATE, nullable=False, description="Customer signup date"),
        ColumnSchema("customer_segment", DataType.STRING, nullable=False, description="Customer segment", allowed_values=["Premium", "Standard", "Budget"]),
    ],
    indexes=["customer_id", "email", "state", "customer_segment"]
)

PRODUCTS_SCHEMA = TableSchema(
    table_name="products",
    description="Product master data with pricing and categorization",
    columns=[
        ColumnSchema("product_id", DataType.INTEGER, nullable=False, primary_key=True, description="Unique product identifier"),
        ColumnSchema("product_name", DataType.STRING, nullable=False, description="Product name"),
        ColumnSchema("category_id", DataType.INTEGER, nullable=False, foreign_key="categories.category_id", description="Product category"),
        ColumnSchema("brand_name", DataType.STRING, nullable=True, description="Product brand"),
        ColumnSchema("selling_price", DataType.FLOAT, nullable=False, description="Current selling price", min_value=0),
        ColumnSchema("cost_price", DataType.FLOAT, nullable=False, description="Cost price", min_value=0),
        ColumnSchema("units_in_stock", DataType.INTEGER, nullable=False, description="Available inventory", min_value=0),
    ],
    indexes=["product_id", "category_id", "brand_name"]
)

CATEGORIES_SCHEMA = TableSchema(
    table_name="categories",
    description="Product category hierarchy",
    columns=[
        ColumnSchema("category_id", DataType.INTEGER, nullable=False, primary_key=True, description="Unique category identifier"),
        ColumnSchema("category_name", DataType.STRING, nullable=False, description="Category name"),
        ColumnSchema("subcategory", DataType.STRING, nullable=True, description="Subcategory name"),
    ],
    indexes=["category_id"]
)

ORDERS_SCHEMA = TableSchema(
    table_name="orders",
    description="Order transaction data",
    columns=[
        ColumnSchema("order_id", DataType.INTEGER, nullable=False, primary_key=True, description="Unique order identifier"),
        ColumnSchema("customer_id", DataType.INTEGER, nullable=False, foreign_key="customers.customer_id", description="Customer who placed order"),
        ColumnSchema("order_date", DataType.DATETIME, nullable=False, description="Order timestamp"),
        ColumnSchema("order_status", DataType.STRING, nullable=False, description="Order status", allowed_values=["Pending", "Processing", "Shipped", "Delivered", "Cancelled", "Returned"]),
        ColumnSchema("order_total", DataType.FLOAT, nullable=False, description="Total order amount", min_value=0),
        ColumnSchema("discount_amount", DataType.FLOAT, nullable=True, description="Discount applied", min_value=0),
        ColumnSchema("payment_method", DataType.STRING, nullable=True, description="Payment method"),
    ],
    indexes=["order_id", "customer_id", "order_date", "order_status"]
)

ORDER_ITEMS_SCHEMA = TableSchema(
    table_name="order_items",
    description="Order line items with product and quantity details",
    columns=[
        ColumnSchema("order_item_id", DataType.INTEGER, nullable=False, primary_key=True, description="Unique line item identifier"),
        ColumnSchema("order_id", DataType.INTEGER, nullable=False, foreign_key="orders.order_id", description="Parent order"),
        ColumnSchema("product_id", DataType.INTEGER, nullable=False, foreign_key="products.product_id", description="Product ordered"),
        ColumnSchema("quantity", DataType.INTEGER, nullable=False, description="Quantity ordered", min_value=1),
        ColumnSchema("unit_price", DataType.FLOAT, nullable=False, description="Unit price at time of order", min_value=0),
        ColumnSchema("line_total", DataType.FLOAT, nullable=False, description="Line item total", min_value=0),
    ],
    indexes=["order_item_id", "order_id", "product_id"]
)

PAYMENTS_SCHEMA = TableSchema(
    table_name="payments",
    description="Payment transaction data",
    columns=[
        ColumnSchema("payment_id", DataType.INTEGER, nullable=False, primary_key=True, description="Unique payment identifier"),
        ColumnSchema("order_id", DataType.INTEGER, nullable=False, foreign_key="orders.order_id", description="Associated order"),
        ColumnSchema("payment_date", DataType.DATETIME, nullable=False, description="Payment timestamp"),
        ColumnSchema("payment_method", DataType.STRING, nullable=False, description="Payment method"),
        ColumnSchema("payment_status", DataType.STRING, nullable=False, description="Payment status", allowed_values=["Pending", "Completed", "Failed", "Refunded"]),
        ColumnSchema("amount", DataType.FLOAT, nullable=False, description="Payment amount", min_value=0),
    ],
    indexes=["payment_id", "order_id", "payment_date"]
)

RETURNS_SCHEMA = TableSchema(
    table_name="returns",
    description="Product return data",
    columns=[
        ColumnSchema("return_id", DataType.INTEGER, nullable=False, primary_key=True, description="Unique return identifier"),
        ColumnSchema("order_id", DataType.INTEGER, nullable=False, foreign_key="orders.order_id", description="Original order"),
        ColumnSchema("product_id", DataType.INTEGER, nullable=False, foreign_key="products.product_id", description="Returned product"),
        ColumnSchema("return_date", DataType.DATE, nullable=False, description="Return date"),
        ColumnSchema("return_reason", DataType.STRING, nullable=True, description="Reason for return"),
        ColumnSchema("refund_amount", DataType.FLOAT, nullable=False, description="Refund amount", min_value=0),
    ],
    indexes=["return_id", "order_id", "product_id", "return_date"]
)

REVIEWS_SCHEMA = TableSchema(
    table_name="reviews",
    description="Customer product reviews",
    columns=[
        ColumnSchema("review_id", DataType.INTEGER, nullable=False, primary_key=True, description="Unique review identifier"),
        ColumnSchema("customer_id", DataType.INTEGER, nullable=False, foreign_key="customers.customer_id", description="Reviewing customer"),
        ColumnSchema("product_id", DataType.INTEGER, nullable=False, foreign_key="products.product_id", description="Reviewed product"),
        ColumnSchema("rating", DataType.INTEGER, nullable=False, description="Product rating", min_value=1, max_value=5),
        ColumnSchema("review_text", DataType.STRING, nullable=True, description="Review text"),
        ColumnSchema("review_date", DataType.DATE, nullable=False, description="Review date"),
    ],
    indexes=["review_id", "customer_id", "product_id", "rating"]
)

WEBSITE_SESSIONS_SCHEMA = TableSchema(
    table_name="website_sessions",
    description="Website session tracking data",
    columns=[
        ColumnSchema("session_id", DataType.INTEGER, nullable=False, primary_key=True, description="Unique session identifier"),
        ColumnSchema("customer_id", DataType.INTEGER, nullable=True, foreign_key="customers.customer_id", description="Customer if logged in"),
        ColumnSchema("session_date", DataType.DATE, nullable=False, description="Session date"),
        ColumnSchema("page_views", DataType.INTEGER, nullable=False, description="Number of page views", min_value=0),
        ColumnSchema("session_duration_sec", DataType.INTEGER, nullable=False, description="Session duration in seconds", min_value=0),
        ColumnSchema("checkout_completed", DataType.BOOLEAN, nullable=False, description="Whether checkout was completed"),
    ],
    indexes=["session_id", "customer_id", "session_date"]
)

INVENTORY_SCHEMA = TableSchema(
    table_name="inventory",
    description="Inventory tracking data",
    columns=[
        ColumnSchema("inventory_id", DataType.INTEGER, nullable=False, primary_key=True, description="Unique inventory record"),
        ColumnSchema("product_id", DataType.INTEGER, nullable=False, foreign_key="products.product_id", description="Product"),
        ColumnSchema("store_id", DataType.INTEGER, nullable=True, description="Store location"),
        ColumnSchema("quantity_on_hand", DataType.INTEGER, nullable=False, description="Available quantity", min_value=0),
        ColumnSchema("reorder_point", DataType.INTEGER, nullable=False, description="Reorder threshold", min_value=0),
        ColumnSchema("last_restocked_date", DataType.DATE, nullable=True, description="Last restock date"),
    ],
    indexes=["inventory_id", "product_id", "store_id"]
)

MARKETING_SPEND_SCHEMA = TableSchema(
    table_name="marketing_spend",
    description="Marketing spend data by channel and date",
    columns=[
        ColumnSchema("spend_id", DataType.INTEGER, nullable=False, primary_key=True, description="Unique spend record"),
        ColumnSchema("campaign_id", DataType.INTEGER, nullable=True, foreign_key="campaigns.campaign_id", description="Associated campaign"),
        ColumnSchema("channel", DataType.STRING, nullable=False, description="Marketing channel"),
        ColumnSchema("spend_date", DataType.DATE, nullable=False, description="Spend date"),
        ColumnSchema("spend_amount", DataType.FLOAT, nullable=False, description="Amount spent", min_value=0),
    ],
    indexes=["spend_id", "campaign_id", "channel", "spend_date"]
)

CAMPAIGNS_SCHEMA = TableSchema(
    table_name="campaigns",
    description="Marketing campaign definitions",
    columns=[
        ColumnSchema("campaign_id", DataType.INTEGER, nullable=False, primary_key=True, description="Unique campaign identifier"),
        ColumnSchema("campaign_name", DataType.STRING, nullable=False, description="Campaign name"),
        ColumnSchema("campaign_type", DataType.STRING, nullable=False, description="Campaign type"),
        ColumnSchema("start_date", DataType.DATE, nullable=False, description="Campaign start date"),
        ColumnSchema("end_date", DataType.DATE, nullable=False, description="Campaign end date"),
        ColumnSchema("budget", DataType.FLOAT, nullable=False, description="Campaign budget", min_value=0),
    ],
    indexes=["campaign_id", "campaign_type", "start_date"]
)

# ============================================================================
# SCHEMA REGISTRY
# ============================================================================

SCHEMA_REGISTRY: Dict[str, TableSchema] = {
    "customers": CUSTOMERS_SCHEMA,
    "products": PRODUCTS_SCHEMA,
    "categories": CATEGORIES_SCHEMA,
    "orders": ORDERS_SCHEMA,
    "order_items": ORDER_ITEMS_SCHEMA,
    "payments": PAYMENTS_SCHEMA,
    "returns": RETURNS_SCHEMA,
    "reviews": REVIEWS_SCHEMA,
    "website_sessions": WEBSITE_SESSIONS_SCHEMA,
    "inventory": INVENTORY_SCHEMA,
    "marketing_spend": MARKETING_SPEND_SCHEMA,
    "campaigns": CAMPAIGNS_SCHEMA,
}


def get_schema(table_name: str) -> TableSchema:
    """Get schema for a table by name."""
    if table_name not in SCHEMA_REGISTRY:
        raise ValueError(f"Schema for table '{table_name}' not found in registry")
    return SCHEMA_REGISTRY[table_name]


def list_all_schemas() -> List[str]:
    """List all available table schemas."""
    return list(SCHEMA_REGISTRY.keys())


def validate_schema_compliance(
    table_name: str,
    data_columns: List[str],
    data_types: Dict[str, str] = None
) -> Dict[str, Any]:
    """
    Validate that data complies with the canonical schema.
    
    Args:
        table_name: Name of the table to validate against
        data_columns: List of column names in the data
        data_types: Optional mapping of column names to their data types
    
    Returns:
        Dictionary with validation results
    """
    schema = get_schema(table_name)
    result = {
        "table_name": table_name,
        "is_valid": True,
        "missing_columns": [],
        "extra_columns": [],
        "type_mismatches": [],
        "nullable_violations": [],
    }
    
    schema_columns = {col.name for col in schema.columns}
    data_columns_set = set(data_columns)
    
    # Check for missing columns
    for col in schema.columns:
        if col.name not in data_columns_set and not col.nullable:
            result["missing_columns"].append(col.name)
            result["is_valid"] = False
    
    # Check for extra columns
    extra = data_columns_set - schema_columns
    if extra:
        result["extra_columns"] = list(extra)
    
    # Check type mismatches if data types provided
    if data_types:
        for col in schema.columns:
            if col.name in data_types:
                # Simplified type checking - can be enhanced
                expected_type = col.data_type.value
                actual_type = data_types[col.name].lower()
                if expected_type not in actual_type:
                    result["type_mismatches"].append({
                        "column": col.name,
                        "expected": expected_type,
                        "actual": actual_type
                    })
                    result["is_valid"] = False
    
    return result
