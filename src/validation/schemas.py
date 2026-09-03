from pathlib import Path
from typing import Dict, Optional

import pandas as pd
import pandera as pa
from pandera import Column, DataFrameSchema, Check, Index

from config.logging_config import get_logger

logger = get_logger(__name__)


def nullable_int(min_val: int = None, max_val: int = None) -> Column:
    checks = []
    if min_val is not None:
        checks.append(Check.greater_than_or_equal_to(min_val))
    if max_val is not None:
        checks.append(Check.less_than_or_equal_to(max_val))
    return Column(int, checks=checks or None, nullable=True)


def required_int(min_val: int = None, max_val: int = None) -> Column:
    checks = []
    if min_val is not None:
        checks.append(Check.greater_than_or_equal_to(min_val))
    if max_val is not None:
        checks.append(Check.less_than_or_equal_to(max_val))
    return Column(int, checks=checks or None, nullable=False)


def nullable_float(min_val: float = None, max_val: float = None) -> Column:
    checks = []
    if min_val is not None:
        checks.append(Check.greater_than_or_equal_to(min_val))
    if max_val is not None:
        checks.append(Check.less_than_or_equal_to(max_val))
    return Column(float, checks=checks or None, nullable=True)


def required_float(min_val: float = None, max_val: float = None) -> Column:
    checks = []
    if min_val is not None:
        checks.append(Check.greater_than_or_equal_to(min_val))
    if max_val is not None:
        checks.append(Check.less_than_or_equal_to(max_val))
    return Column(float, checks=checks or None, nullable=False)


def required_str(
    allowed: Optional[tuple] = None,
    nullable: bool = False,
    regex: Optional[str] = None,
) -> Column:
    checks = []
    if allowed:
        checks.append(Check.isin(list(allowed)))
    if regex:
        checks.append(Check.str_matches(regex))
    return Column(str, checks=checks or None, nullable=nullable)


SCHEMAS = {
    "customers": DataFrameSchema(
        {
            "customer_id": required_int(min_val=1),
            "first_name": required_str(nullable=False),
            "last_name": required_str(nullable=False),
            "gender": required_str(allowed=("M", "F"), nullable=True),
            "age": nullable_int(min_val=0, max_val=120),
            "city": required_str(nullable=False),
            "state": required_str(nullable=False),
            "country": required_str(nullable=False),
            "signup_date": Column("datetime64[ns]", nullable=False),
            "customer_segment": required_str(nullable=True),
        },
        index=Index(int),
        strict=False,
        coerce=True,
    ),
    "products": DataFrameSchema(
        {
            "product_id": required_int(min_val=1),
            "product_name": required_str(nullable=False),
            "category_id": required_int(min_val=1),
            "supplier_id": required_int(min_val=1),
            "cost_price": required_float(min_val=0),
            "selling_price": required_float(min_val=0),
            "launch_date": Column("datetime64[ns]", nullable=False),
            "product_status": required_str(
                allowed=("Active", "Discontinued", "Out of Season"), nullable=True
            ),
        },
        strict=False,
        coerce=True,
    ),
    "orders": DataFrameSchema(
        {
            "order_id": required_int(min_val=1),
            "customer_id": required_int(min_val=1),
            "order_date": Column("datetime64[ns]", nullable=False),
            "order_status": required_str(
                allowed=(
                    "Delivered", "Cancelled", "Returned",
                    "Processing", "Shipped", "Pending",
                ),
                nullable=False,
            ),
            "order_total": required_float(min_val=0),
            "discount_amount": nullable_float(min_val=0),
            "tax_amount": nullable_float(min_val=0),
            "shipping_cost": nullable_float(min_val=0),
        },
        strict=False,
        coerce=True,
    ),
    "order_items": DataFrameSchema(
        {
            "order_item_id": required_int(min_val=1),
            "order_id": required_int(min_val=1),
            "product_id": required_int(min_val=1),
            "quantity": required_int(min_val=1),
            "unit_price": required_float(min_val=0),
            "discount": nullable_float(min_val=0),
            "tax": nullable_float(min_val=0),
            "line_total": required_float(min_val=0),
        },
        strict=False,
        coerce=True,
    ),
    "payments": DataFrameSchema(
        {
            "payment_id": required_int(min_val=1),
            "customer_id": required_int(min_val=1),
            "payment_method": required_str(nullable=False),
            "payment_status": required_str(
                allowed=("Success", "Failed", "Pending", "Refunded"), nullable=False
            ),
            "transaction_date": Column("datetime64[ns]", nullable=False),
            "amount": required_float(min_val=0),
        },
        strict=False,
        coerce=True,
    ),
    "categories": DataFrameSchema(
        {
            "category_id": required_int(min_val=1),
            "category_name": required_str(nullable=False),
            "subcategory": required_str(nullable=False),
        },
        strict=False,
        coerce=True,
    ),
    "suppliers": DataFrameSchema(
        {
            "supplier_id": required_int(min_val=1),
            "supplier_name": required_str(nullable=False),
            "rating": nullable_float(min_val=0, max_val=5),
            "reliability_score": nullable_float(min_val=0, max_val=100),
        },
        strict=False,
        coerce=True,
    ),
    "stores": DataFrameSchema(
        {
            "store_id": required_int(min_val=1),
            "store_name": required_str(nullable=False),
            "store_type": required_str(
                allowed=("Warehouse", "Retail", "Fulfillment Center"), nullable=True
            ),
            "city": required_str(nullable=False),
            "state": required_str(nullable=False),
        },
        strict=False,
        coerce=True,
    ),
    "employees": DataFrameSchema(
        {
            "employee_id": required_int(min_val=1),
            "first_name": required_str(nullable=False),
            "last_name": required_str(nullable=False),
            "role": required_str(nullable=False),
            "hire_date": Column("datetime64[ns]", nullable=False),
            "performance_score": nullable_float(min_val=0, max_val=100),
        },
        strict=False,
        coerce=True,
    ),
    "inventory": DataFrameSchema(
        {
            "inventory_id": required_int(min_val=1),
            "product_id": required_int(min_val=1),
            "store_id": required_int(min_val=1),
            "stock_quantity": required_int(min_val=0),
            "reorder_level": nullable_int(min_val=0),
            "safety_stock": nullable_int(min_val=0),
        },
        strict=False,
        coerce=True,
    ),
    "marketing_campaigns": DataFrameSchema(
        {
            "campaign_id": required_int(min_val=1),
            "campaign_name": required_str(nullable=False),
            "campaign_type": required_str(nullable=False),
            "channel": required_str(nullable=False),
            "start_date": Column("datetime64[ns]", nullable=False),
            "end_date": Column("datetime64[ns]", nullable=False),
            "total_budget": required_float(min_val=0),
            "status": required_str(
                allowed=("Active", "Completed", "Planned", "Cancelled"), nullable=True
            ),
        },
        strict=False,
        coerce=True,
    ),
    "returns": DataFrameSchema(
        {
            "return_id": required_int(min_val=1),
            "order_item_id": required_int(min_val=1),
            "order_id": required_int(min_val=1),
            "customer_id": required_int(min_val=1),
            "product_id": required_int(min_val=1),
            "return_date": Column("datetime64[ns]", nullable=False),
            "refund_amount": required_float(min_val=0),
            "quantity_returned": required_int(min_val=1),
        },
        strict=False,
        coerce=True,
    ),
    "reviews": DataFrameSchema(
        {
            "review_id": required_int(min_val=1),
            "customer_id": required_int(min_val=1),
            "product_id": required_int(min_val=1),
            "review_date": Column("datetime64[ns]", nullable=False),
            "rating": required_int(min_val=1, max_val=5),
            "helpful_votes": nullable_int(min_val=0),
        },
        strict=False,
        coerce=True,
    ),
    "marketing_spend": DataFrameSchema(
        {
            "spend_id": required_int(min_val=1),
            "campaign_id": required_int(min_val=1),
            "spend_date": Column("datetime64[ns]", nullable=False),
            "channel": required_str(nullable=False),
            "impressions": nullable_int(min_val=0),
            "clicks": nullable_int(min_val=0),
            "spend_amount": required_float(min_val=0),
            "ctr": nullable_float(min_val=0),
            "cpc": nullable_float(min_val=0),
        },
        strict=False,
        coerce=True,
    ),
    "website_sessions": DataFrameSchema(
        {
            "session_id": required_int(min_val=1),
            "session_date": Column("datetime64[ns]", nullable=False),
            "session_start": Column("datetime64[ns]", nullable=False),
            "page_views": nullable_int(min_val=0),
            "product_views": nullable_int(min_val=0),
            "cart_adds": nullable_int(min_val=0),
            "checkout_started": nullable_int(min_val=0, max_val=1),
            "checkout_completed": nullable_int(min_val=0, max_val=1),
            "session_duration_sec": nullable_int(min_val=0),
        },
        strict=False,
        coerce=True,
    ),
}


class DataValidator:
    def __init__(self, schemas: Dict[str, DataFrameSchema] = None):
        self.schemas = schemas or SCHEMAS

    def validate(
        self, df: pd.DataFrame, table_name: str, lazy: bool = True
    ) -> pd.DataFrame:
        if table_name not in self.schemas:
            logger.warning(f"No schema defined for {table_name}")
            return df
        logger.info(f"Validating {len(df):,} rows for table '{table_name}'...")
        try:
            validated = self.schemas[table_name].validate(df, lazy=lazy)
            logger.info(f"  Validation passed for {table_name}")
            return validated
        except pa.errors.SchemaErrors as e:
            logger.error(f"  Validation FAILED for {table_name}: {e}")
            logger.error(f"  Failure counts:\n{e.failure_cases}")
            raise
        except Exception as e:
            logger.error(f"  Validation error for {table_name}: {e}")
            raise

    def validate_file(self, csv_path: str, table_name: str) -> pd.DataFrame:
        df = pd.read_csv(csv_path, encoding="utf-8")
        return self.validate(df, table_name)

    def validate_all(
        self, data_dir: Path, tables: list = None
    ) -> Dict[str, pd.DataFrame]:
        data_dir = Path(data_dir)
        tables = tables or list(self.schemas.keys())
        results = {}
        for t in tables:
            path = data_dir / f"{t}.csv"
            if path.exists():
                results[t] = self.validate_file(str(path), t)
            else:
                logger.warning(f"Skipping {t}: file not found")
        return results
