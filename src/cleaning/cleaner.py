"""
Data Cleaning Module for E-Commerce Intelligence & Decision Analytics Platform.

Provides the DataCleaner class that performs ETL cleaning operations to transform
raw CSV data into staging-ready format. Cleaning operations include deduplication,
type coercion, missing value imputation, outlier handling (IQR capping), and
string sanitization for all supported tables.

Supported tables:
    customers, products, categories, orders, order_items, payments, returns,
    reviews, inventory, suppliers, stores, employees, marketing_campaigns,
    marketing_spend, website_sessions
"""

from config.logging_config import get_logger
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple

logger = get_logger(__name__)

PK_COLUMNS: Dict[str, str] = {
    "customers": "customer_id",
    "categories": "category_id",
    "suppliers": "supplier_id",
    "products": "product_id",
    "stores": "store_id",
    "employees": "employee_id",
    "inventory": "inventory_id",
    "marketing_campaigns": "campaign_id",
    "payments": "payment_id",
    "orders": "order_id",
    "order_items": "order_item_id",
    "returns": "return_id",
    "reviews": "review_id",
    "marketing_spend": "spend_id",
    "website_sessions": "session_id",
}

INT_ID_COLUMNS: set = {
    "customer_id", "product_id", "order_id", "payment_id", "category_id",
    "supplier_id", "store_id", "employee_id", "inventory_id", "campaign_id",
    "return_id", "review_id", "spend_id", "session_id", "order_item_id",
}

FLOAT_COLUMNS: set = {
    "cost_price", "selling_price", "discount", "tax", "line_total",
    "order_total", "shipping_cost", "refund_amount", "spend_amount", "amount",
    "unit_price", "total_budget", "ctr", "cpc", "rating", "reliability_score",
    "performance_score", "discount_amount", "tax_amount",
}

OUTLIER_COLUMNS: set = {
    "cost_price", "selling_price", "order_total", "amount", "quantity",
    "unit_price", "stock_quantity", "refund_amount", "spend_amount",
    "line_total", "total_budget", "quantity_returned",
}

NULLABLE_FK_COLUMNS: set = {
    "campaign_id",
}

EMAIL_PHONE_COLUMNS: set = {"email", "phone"}

DEFAULT_TABLES: List[str] = [
    "customers", "categories", "suppliers", "products", "stores", "employees",
    "inventory", "marketing_campaigns", "payments", "orders", "order_items",
    "returns", "reviews", "marketing_spend", "website_sessions",
]


class DataCleaner:
    """
    Performs ETL data cleaning operations on raw tables (raw -> staging).

    Applies a standardized cleaning pipeline covering deduplication, type
    coercion, missing value imputation, IQR-based outlier capping, and string
    whitespace sanitization. Produces per-table metrics on issues fixed.
    """

    def __init__(self) -> None:
        self.pk_columns = PK_COLUMNS
        self.int_id_columns = INT_ID_COLUMNS
        self.float_columns = FLOAT_COLUMNS
        self.outlier_columns = OUTLIER_COLUMNS
        self.nullable_fk = NULLABLE_FK_COLUMNS
        self.email_phone_cols = EMAIL_PHONE_COLUMNS

    def _dedup(self, df: pd.DataFrame, table_name: str) -> Tuple[pd.DataFrame, int]:
        in_rows = len(df)
        df = df.drop_duplicates()
        dup_full = in_rows - len(df)

        pk = self.pk_columns.get(table_name)
        dup_pk = 0
        if pk and pk in df.columns:
            before = len(df)
            df = df.drop_duplicates(subset=[pk], keep="first")
            dup_pk = before - len(df)

        removed = dup_full + dup_pk
        if removed > 0:
            logger.info(f"  [{table_name}] Deduplication removed {removed} rows "
                        f"(full-row={dup_full}, pk={dup_pk})")
        return df, removed

    def _coerce_types(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, int]:
        dates_coerced_errors = 0

        for col in df.columns:
            if "date" in col.lower():
                before = df[col].notna().sum()
                df[col] = pd.to_datetime(df[col], errors="coerce")
                after = df[col].notna().sum()
                dates_coerced_errors += int(before - after)
            elif col in self.int_id_columns:
                df[col] = pd.to_numeric(df[col], errors="coerce").astype("Int64")
            elif col in self.float_columns:
                df[col] = pd.to_numeric(df[col], errors="coerce").astype(float)

        if dates_coerced_errors > 0:
            logger.info(f"  Type coercion: {dates_coerced_errors} date parse errors coerced to NaT")
        return df, dates_coerced_errors

    def _impute_missing(self, df: pd.DataFrame, table_name: str) -> Tuple[pd.DataFrame, int]:
        nulls_filled = 0
        pk = self.pk_columns.get(table_name)

        for col in df.columns:
            if col == pk:
                continue

            total = len(df)
            if total == 0:
                continue
            null_count = int(df[col].isna().sum())
            if null_count == 0:
                continue
            null_pct = (null_count / total) * 100

            if df[col].dtype == "object" or pd.api.types.is_string_dtype(df[col]):
                if col.lower() in self.email_phone_cols:
                    continue
                df[col] = df[col].fillna("Unknown")
                nulls_filled += null_count
            elif col in self.nullable_fk:
                continue
            elif pd.api.types.is_datetime64_any_dtype(df[col]):
                df[col] = df[col].fillna(pd.NaT)
            elif pd.api.types.is_numeric_dtype(df[col]):
                if col in self.int_id_columns:
                    continue
                if null_pct < 5:
                    median_val = df[col].median()
                    if pd.notna(median_val):
                        df[col] = df[col].fillna(median_val)
                        nulls_filled += null_count
                else:
                    df[col] = df[col].fillna(0)
                    nulls_filled += null_count

        if nulls_filled > 0:
            logger.info(f"  [{table_name}] Imputed {nulls_filled} missing values")
        return df, nulls_filled

    def _handle_outliers_iqr(self, df: pd.DataFrame, table_name: str) -> Tuple[pd.DataFrame, int]:
        outliers_capped = 0
        cols = [c for c in df.columns if c in self.outlier_columns]

        for col in cols:
            series = pd.to_numeric(df[col], errors="coerce")
            valid = series.dropna()
            if len(valid) < 4:
                continue

            q1 = valid.quantile(0.25)
            q3 = valid.quantile(0.75)
            iqr = q3 - q1
            if iqr == 0 or pd.isna(iqr):
                continue

            floor = q1 - 1.5 * iqr
            ceiling = q3 + 1.5 * iqr

            below = (series < floor).sum()
            above = (series > ceiling).sum()
            capped = int(below + above)
            if capped > 0:
                df[col] = series.clip(lower=floor, upper=ceiling)
                outliers_capped += capped
                logger.info(f"  [{table_name}] Outlier capping on {col}: "
                            f"{capped} values clipped (floor={floor:.2f}, ceil={ceiling:.2f})")

        return df, outliers_capped

    def _sanitize_strings(self, df: pd.DataFrame) -> pd.DataFrame:
        obj_cols = df.select_dtypes(include=["object"]).columns.tolist()
        for col in obj_cols:
            df[col] = df[col].astype(str).str.strip()
            df[col] = df[col].replace({"nan": np.nan, "None": np.nan, "": np.nan})
        return df

    def clean_table(self, df: pd.DataFrame, table_name: str) -> Tuple[pd.DataFrame, int]:
        """
        Clean a single table DataFrame applying the full cleaning pipeline.

        Args:
            df: Raw input DataFrame for the table.
            table_name: Name of the table (used for PK lookup and logging).

        Returns:
            Tuple of (cleaned DataFrame, total count of issues fixed).
        """
        logger.info(f"Cleaning table '{table_name}': {len(df):,} rows")

        df, dup_removed = self._dedup(df.copy(), table_name)
        df = self._sanitize_strings(df)
        df, dates_coerced_errors = self._coerce_types(df)
        df, nulls_filled = self._impute_missing(df, table_name)
        df, outliers_capped = self._handle_outliers_iqr(df, table_name)

        issues_fixed = (
            int(dup_removed)
            + int(nulls_filled)
            + int(outliers_capped)
            + int(dates_coerced_errors)
        )

        logger.info(
            f"  [{table_name}] Cleaning complete: {len(df):,} rows "
            f"(issues_fixed={issues_fixed}, "
            f"dup={dup_removed}, nulls={nulls_filled}, "
            f"outliers={outliers_capped}, coerce_err={dates_coerced_errors})"
        )
        return df, issues_fixed

    def clean_all(
        self,
        raw_dir: Path,
        staging_dir: Path,
        tables: Optional[List[str]] = None,
    ) -> Dict[str, Dict[str, int]]:
        """
        Clean all specified tables from raw_dir and save outputs to staging_dir.

        For each table: reads the CSV from raw_dir, runs clean_table, and saves
        the result to staging_dir/{table}.csv. Creates staging_dir if it does
        not exist.

        Args:
            raw_dir: Directory containing raw CSV files.
            staging_dir: Destination directory for cleaned CSVs.
            tables: Optional list of table names to clean. If None, all
                DEFAULT_TABLES are processed.

        Returns:
            Dictionary keyed by table name, each value being a dict with keys:
                in_rows, out_rows, removed, issues_fixed.
        """
        raw_dir = Path(raw_dir)
        staging_dir = Path(staging_dir)
        staging_dir.mkdir(parents=True, exist_ok=True)

        tables = tables or DEFAULT_TABLES
        logger.info(f"Starting cleaning pipeline: {len(tables)} tables "
                    f"from {raw_dir} -> {staging_dir}")

        results: Dict[str, Dict[str, int]] = {}

        for table_name in tables:
            csv_path = raw_dir / f"{table_name}.csv"
            if not csv_path.exists():
                logger.warning(f"Skipping '{table_name}': file not found at {csv_path}")
                continue

            try:
                df = pd.read_csv(csv_path, encoding="utf-8")
            except Exception as e:
                logger.error(f"Failed to read {csv_path}: {e}")
                continue

            in_rows = len(df)
            cleaned_df, issues_fixed = self.clean_table(df, table_name)
            out_rows = len(cleaned_df)
            removed = in_rows - out_rows

            out_path = staging_dir / f"{table_name}.csv"
            cleaned_df.to_csv(out_path, index=False, encoding="utf-8")
            logger.info(f"  Saved cleaned {table_name} -> {out_path} ({out_rows:,} rows)")

            results[table_name] = {
                "in_rows": int(in_rows),
                "out_rows": int(out_rows),
                "removed": int(removed),
                "issues_fixed": int(issues_fixed),
            }

        logger.info("=" * 60)
        logger.info("CLEANING SUMMARY")
        total_in = sum(r["in_rows"] for r in results.values())
        total_out = sum(r["out_rows"] for r in results.values())
        total_removed = sum(r["removed"] for r in results.values())
        total_issues = sum(r["issues_fixed"] for r in results.values())
        for t, r in results.items():
            logger.info(
                f"  {t}: in={r['in_rows']:,}, out={r['out_rows']:,}, "
                f"removed={r['removed']:,}, issues_fixed={r['issues_fixed']:,}"
            )
        logger.info(
            f"  TOTAL: tables={len(results)}, in={total_in:,}, out={total_out:,}, "
            f"removed={total_removed:,}, issues_fixed={total_issues:,}"
        )
        logger.info("=" * 60)

        return results
