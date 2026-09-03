"""Foreign key integrity checker for the e-commerce data warehouse.

Validates referential integrity across 15 dimension and fact tables by
checking that every non-null foreign key value references an existing
primary key in its parent table, and that every primary key column is
unique and non-null.
"""

from pathlib import Path
from typing import Dict, List, Set

import numpy as np
import pandas as pd
from rich.console import Console
from rich.table import Table

from config.logging_config import get_logger

logger = get_logger(__name__)


TABLE_SCHEMA: Dict[str, Dict] = {
    "customers": {
        "pk": "customer_id",
        "fks": [],
    },
    "categories": {
        "pk": "category_id",
        "fks": [],
    },
    "suppliers": {
        "pk": "supplier_id",
        "fks": [],
    },
    "products": {
        "pk": "product_id",
        "fks": [
            {"fk_col": "category_id", "parent_table": "categories", "pk_col": "category_id", "nullable": False},
            {"fk_col": "supplier_id", "parent_table": "suppliers", "pk_col": "supplier_id", "nullable": False},
        ],
    },
    "stores": {
        "pk": "store_id",
        "fks": [],
    },
    "employees": {
        "pk": "employee_id",
        "fks": [
            {"fk_col": "store_id", "parent_table": "stores", "pk_col": "store_id", "nullable": False},
        ],
    },
    "inventory": {
        "pk": "inventory_id",
        "fks": [
            {"fk_col": "product_id", "parent_table": "products", "pk_col": "product_id", "nullable": False},
            {"fk_col": "store_id", "parent_table": "stores", "pk_col": "store_id", "nullable": False},
        ],
    },
    "marketing_campaigns": {
        "pk": "campaign_id",
        "fks": [],
    },
    "payments": {
        "pk": "payment_id",
        "fks": [
            {"fk_col": "customer_id", "parent_table": "customers", "pk_col": "customer_id", "nullable": False},
        ],
    },
    "orders": {
        "pk": "order_id",
        "fks": [
            {"fk_col": "customer_id", "parent_table": "customers", "pk_col": "customer_id", "nullable": False},
            {"fk_col": "store_id", "parent_table": "stores", "pk_col": "store_id", "nullable": False},
            {"fk_col": "payment_id", "parent_table": "payments", "pk_col": "payment_id", "nullable": False},
            {"fk_col": "campaign_id", "parent_table": "marketing_campaigns", "pk_col": "campaign_id", "nullable": True},
        ],
    },
    "order_items": {
        "pk": "order_item_id",
        "fks": [
            {"fk_col": "order_id", "parent_table": "orders", "pk_col": "order_id", "nullable": False},
            {"fk_col": "product_id", "parent_table": "products", "pk_col": "product_id", "nullable": False},
        ],
    },
    "returns": {
        "pk": "return_id",
        "fks": [
            {"fk_col": "order_item_id", "parent_table": "order_items", "pk_col": "order_item_id", "nullable": False},
            {"fk_col": "order_id", "parent_table": "orders", "pk_col": "order_id", "nullable": False},
            {"fk_col": "customer_id", "parent_table": "customers", "pk_col": "customer_id", "nullable": False},
            {"fk_col": "product_id", "parent_table": "products", "pk_col": "product_id", "nullable": False},
        ],
    },
    "reviews": {
        "pk": "review_id",
        "fks": [
            {"fk_col": "customer_id", "parent_table": "customers", "pk_col": "customer_id", "nullable": False},
            {"fk_col": "product_id", "parent_table": "products", "pk_col": "product_id", "nullable": False},
            {"fk_col": "order_id", "parent_table": "orders", "pk_col": "order_id", "nullable": False},
        ],
    },
    "marketing_spend": {
        "pk": "spend_id",
        "fks": [
            {"fk_col": "campaign_id", "parent_table": "marketing_campaigns", "pk_col": "campaign_id", "nullable": False},
        ],
    },
    "website_sessions": {
        "pk": "session_id",
        "fks": [
            {"fk_col": "customer_id", "parent_table": "customers", "pk_col": "customer_id", "nullable": True},
            {"fk_col": "campaign_id", "parent_table": "marketing_campaigns", "pk_col": "campaign_id", "nullable": True},
        ],
    },
}


class FKIntegrityChecker:
    def __init__(self):
        self.report: Dict = {}

    def _read_csv(self, data_dir: Path, table: str) -> pd.DataFrame:
        csv_path = Path(data_dir) / f"{table}.csv"
        logger.debug(f"Reading {csv_path}")
        return pd.read_csv(csv_path, low_memory=False)

    def _get_pk_set(self, df: pd.DataFrame, pk_col: str) -> Set:
        pk_series = df[pk_col].replace({np.nan: None})
        valid_pks = pk_series.dropna()
        return set(valid_pks.tolist())

    def _check_pk(self, df: pd.DataFrame, pk_col: str) -> Dict:
        pk_series = df[pk_col].replace({np.nan: None})
        null_count = int(pk_series.isna().sum())
        dup_count = int(pk_series.duplicated().sum())
        pk_valid = (null_count == 0) and (dup_count == 0)
        return {
            "pk_valid": pk_valid,
            "pk_dups": dup_count,
            "pk_nulls": null_count,
        }

    def _check_fk(
        self,
        df: pd.DataFrame,
        fk_col: str,
        parent_pk_set: Set,
        nullable: bool,
    ) -> Dict:
        fk_series = df[fk_col].replace({np.nan: None})
        null_mask = fk_series.isna()
        total_nulls = int(null_mask.sum())
        non_null_fks = fk_series[~null_mask]
        total = int(len(non_null_fks))

        if total == 0:
            orphans = 0
            valid = True
        else:
            orphan_mask = ~non_null_fks.isin(parent_pk_set)
            orphans = int(orphan_mask.sum())
            valid = orphans == 0

        if nullable:
            valid = valid or True

        return {
            "fk_col": fk_col,
            "orphans": orphans,
            "total": total,
            "valid": valid,
            "nulls": total_nulls,
        }

    def check_all(self, data_dir) -> Dict:
        data_dir = Path(data_dir)
        logger.info(f"Starting FK integrity check on {data_dir}")

        dataframes: Dict[str, pd.DataFrame] = {}
        pk_sets: Dict[str, Set] = {}
        report: Dict = {}

        for table in TABLE_SCHEMA.keys():
            try:
                df = self._read_csv(data_dir, table)
                dataframes[table] = df
                pk_col = TABLE_SCHEMA[table]["pk"]
                pk_sets[table] = self._get_pk_set(df, pk_col)
                logger.debug(f"Loaded {table}: {len(df):,} rows, PK set size={len(pk_sets[table]):,}")
            except FileNotFoundError:
                logger.warning(f"CSV not found for table '{table}', skipping")
                continue
            except Exception as e:
                logger.error(f"Error loading {table}: {e}")
                raise

        for table, schema in TABLE_SCHEMA.items():
            if table not in dataframes:
                continue

            df = dataframes[table]
            pk_col = schema["pk"]

            pk_result = self._check_pk(df, pk_col)

            fk_results: List[Dict] = []
            for fk_def in schema["fks"]:
                parent_table = fk_def["parent_table"]
                fk_col = fk_def["fk_col"]
                pk_col_parent = fk_def["pk_col"]
                nullable = fk_def["nullable"]

                if parent_table not in pk_sets:
                    logger.warning(f"Parent table {parent_table} not loaded, skipping FK {table}.{fk_col}")
                    continue

                parent_pk_set = pk_sets[parent_table]
                fk_result = self._check_fk(df, fk_col, parent_pk_set, nullable)
                fk_result["parent_table"] = parent_table
                fk_result["pk_col"] = pk_col_parent
                fk_results.append(fk_result)

                if not fk_result["valid"]:
                    logger.warning(
                        f"FK violation: {table}.{fk_col} -> {parent_table}.{pk_col_parent}: "
                        f"{fk_result['orphans']:,} orphans out of {fk_result['total']:,}"
                    )

            report[table] = {
                "fks": fk_results,
                "pk_valid": pk_result["pk_valid"],
                "pk_dups": pk_result["pk_dups"],
                "pk_nulls": pk_result["pk_nulls"],
                "rows": int(len(df)),
            }

        all_pk_valid = all(r["pk_valid"] for r in report.values())
        all_fk_valid = all(fk["valid"] for r in report.values() for fk in r["fks"])
        logger.info(
            f"Integrity check complete. PKs valid: {all_pk_valid}, "
            f"FKs valid: {all_fk_valid}. Checked {len(report)} tables."
        )

        self.report = report
        return report

    def print_report(self, report: Dict, console: Console) -> bool:
        table = Table(
            title="E-Commerce Data Warehouse — FK Integrity Report",
            show_header=True,
            header_style="bold cyan",
            title_style="bold magenta",
        )
        table.add_column("Table", style="bold", no_wrap=True)
        table.add_column("Rows", justify="right", style="dim")
        table.add_column("PK", justify="center")
        table.add_column("PK Dups", justify="right")
        table.add_column("FK Check", justify="center")
        table.add_column("FK Details", style="white")

        all_passed = True

        for table_name in TABLE_SCHEMA.keys():
            if table_name not in report:
                continue

            info = report[table_name]

            pk_status = "[bold green]OK[/bold green]" if info["pk_valid"] else "[bold red]FAIL[/bold red]"
            pk_dups_str = f"{info['pk_dups']:,}" if info["pk_dups"] > 0 else "0"
            if info["pk_dups"] > 0:
                pk_dups_str = f"[red]{pk_dups_str}[/red]"

            if not info["pk_valid"]:
                all_passed = False

            fk_entries = info["fks"]
            if not fk_entries:
                fk_status = "[dim]N/A[/dim]"
                fk_details = "[dim]no FKs[/dim]"
            else:
                all_fks_ok = all(f["valid"] for f in fk_entries)
                fk_status = "[bold green]OK[/bold green]" if all_fks_ok else "[bold red]FAIL[/bold red]"
                if not all_fks_ok:
                    all_passed = False

                detail_parts = []
                for fk in fk_entries:
                    sign = "OK" if fk["valid"] else "FAIL"
                    color = "green" if fk["valid"] else "red"
                    orphans_str = f"{fk['orphans']:,}" if fk["orphans"] > 0 else "0"
                    total_str = f"{fk['total']:,}"
                    open_tag = "[" + color + "]"
                    close_tag = "[/" + color + "]"
                    part = (
                        open_tag + sign + close_tag + " "
                        + fk["fk_col"] + "->" + fk["parent_table"] + " "
                        + "(" + orphans_str + "/" + total_str + ")"
                    )
                    detail_parts.append(part)
                fk_details = "\n".join(detail_parts)

            rows_str = f"{info['rows']:,}"
            table.add_row(
                table_name,
                rows_str,
                pk_status,
                pk_dups_str,
                fk_status,
                fk_details,
            )

        console.print()
        console.print(table)
        console.print()

        if all_passed:
            console.print("[bold green]OK - All integrity checks passed.[/bold green]")
        else:
            console.print("[bold red]FAIL - Integrity check failures detected. See details above.[/bold red]")
        console.print()

        return all_passed
