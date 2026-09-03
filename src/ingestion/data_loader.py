from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
from sqlalchemy import inspect

from config.settings import settings
from config.logging_config import get_logger
from src.ingestion.database import DatabaseManager

logger = get_logger(__name__)


TABLE_LOAD_ORDER: List[str] = [
    "customers",
    "categories",
    "suppliers",
    "products",
    "stores",
    "employees",
    "inventory",
    "marketing_campaigns",
    "payments",
    "orders",
    "order_items",
    "returns",
    "reviews",
    "marketing_spend",
    "website_sessions",
]


class DataLoader:
    def __init__(self, db: DatabaseManager = None, data_dir: Path = None):
        self.db = db or DatabaseManager()
        self.data_dir = Path(data_dir) if data_dir else settings.RAW_DATA_DIR

    def read_csv(self, table_name: str) -> pd.DataFrame:
        path = self.data_dir / f"{table_name}.csv"
        if not path.exists():
            raise FileNotFoundError(f"CSV file not found: {path}")
        logger.info(f"Reading {path}")
        df = pd.read_csv(path, encoding="utf-8")
        logger.info(f"  Loaded {len(df)} rows from {table_name}.csv")
        return df

    def write_df_to_sql(
        self,
        df: pd.DataFrame,
        table_name: str,
        if_exists: str = "append",
        chunksize: int = 10000,
        dtype: Optional[dict] = None,
    ) -> int:
        logger.info(
            f"Loading {len(df):,} rows into table '{table_name}' "
            f"(mode={if_exists}, chunksize={chunksize})"
        )
        rows = df.to_sql(
            name=table_name, con=self.db.engine, if_exists=if_exists,
            index=False, chunksize=chunksize, dtype=dtype,
        )
        logger.info(f"  Successfully loaded {len(df):,} rows into '{table_name}'")
        return rows or len(df)

    def table_exists(self, table_name: str) -> bool:
        inspector = inspect(self.db.engine)
        return inspector.has_table(table_name)

    def get_row_count(self, table_name: str) -> int:
        if not self.table_exists(table_name):
            return 0
        with self.db.get_connection() as conn:
            result = conn.execute(f"SELECT COUNT(*) FROM {table_name}")
            return int(result.scalar())

    def load_table(
        self,
        table_name: str,
        if_exists: str = "append",
        chunksize: int = 10000,
    ) -> int:
        df = self.read_csv(table_name)
        cols_lower = {c: c.lower() for c in df.columns}
        df = df.rename(columns=cols_lower)
        return self.write_df_to_sql(df, table_name, if_exists=if_exists, chunksize=chunksize)

    def load_all(
        self,
        tables: Optional[List[str]] = None,
        if_exists: str = "append",
        chunksize: int = 10000,
    ) -> Dict[str, int]:
        tables = tables or TABLE_LOAD_ORDER
        logger.info(f"Loading {len(tables)} tables into database...")

        results = {}
        for table in tables:
            try:
                rows = self.load_table(table, if_exists=if_exists, chunksize=chunksize)
                results[table] = rows
            except Exception as e:
                logger.error(f"Failed to load {table}: {e}")
                raise
        logger.info("=" * 60)
        logger.info("LOAD SUMMARY")
        for t, c in results.items():
            logger.info(f"  {t}: {c:,} rows")
        logger.info(f"  Total: {sum(results.values()):,} rows")
        logger.info("=" * 60)
        return results

    def drop_all_tables(self) -> None:
        logger.warning("Dropping all tables in reverse order...")
        for table in reversed(TABLE_LOAD_ORDER):
            try:
                self.db.execute_sql(f'DROP TABLE IF EXISTS "{table}" CASCADE')
                logger.info(f"  Dropped {table}")
            except Exception as e:
                logger.warning(f"  Error dropping {table}: {e}")

    def create_schema(self, schema_file: Path = None) -> None:
        schema_file = schema_file or (
            settings.SQL_DIR / "staging" / "001_create_schema.sql"
        )
        logger.info(f"Creating schema from {schema_file}...")
        self.db.read_sql_file(str(schema_file))
        logger.info("Schema created successfully")

    def reset_database(self, schema_file: Path = None) -> None:
        logger.warning("Resetting database (DROP ALL -> CREATE SCHEMA)...")
        self.drop_all_tables()
        self.create_schema(schema_file)
        logger.info("Database reset complete")


def main():
    loader = DataLoader()
    loader.db.test_connection()
    return loader


if __name__ == "__main__":
    main()
