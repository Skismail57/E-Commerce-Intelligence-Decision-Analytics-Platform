from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

from config.logging_config import get_logger
from src.transformation.feature_engineering import FeatureEngineer

logger = get_logger(__name__)


class ProductIntelligence:
    def __init__(self, data_dir: Optional[Path] = None):
        self.data_dir = Path(data_dir) if data_dir else None
        self.fe = FeatureEngineer(data_dir)

    def _read(self, table: str, data_dir: Optional[Path] = None) -> pd.DataFrame:
        directory = data_dir or self.data_dir
        if directory is None:
            raise ValueError("data_dir must be provided")
        df = pd.read_csv(Path(directory) / f"{table}.csv", low_memory=False)
        for dc in [c for c in df.columns if "date" in c.lower()]:
            try:
                df[dc] = pd.to_datetime(df[dc])
            except Exception:
                pass
        return df

    def build_product_360(
        self,
        data_dir: Optional[Path] = None,
    ) -> pd.DataFrame:
        directory = Path(data_dir) if data_dir else self.data_dir
        if directory is None:
            raise ValueError("data_dir must be provided")

        products = self._read("products", directory)
        categories = self._read("categories", directory)
        suppliers = self._read("suppliers", directory)
        order_items = self._read("order_items", directory)
        orders = self._read("orders", directory)
        returns = self._read("returns", directory)
        reviews = self._read("reviews", directory)
        inventory = self._read("inventory", directory)

        valid_statuses = ["Delivered", "Shipped", "Processing", "Returned"]
        valid_orders = orders[orders["order_status"].isin(valid_statuses)]
        oi_valid = order_items[order_items["order_id"].isin(valid_orders["order_id"])]

        p360 = products.merge(categories, on="category_id", how="left")
        p360 = p360.merge(
            suppliers[["supplier_id", "supplier_name", "rating"]],
            on="supplier_id", how="left", suffixes=("", "_supp"),
        )
        if "rating_supp" in p360.columns:
            p360 = p360.rename(columns={"rating_supp": "supplier_rating"})

        oi_agg = oi_valid.groupby("product_id").agg(
            total_orders=("order_id", "nunique"),
            units_sold=("quantity", "sum"),
            revenue=("line_total", "sum"),
            discount_given=("discount", "sum"),
            avg_discount_pct=("discount_pct", "mean"),
        ).reset_index()

        oi_cogs = oi_valid.merge(
            products[["product_id", "cost_price"]], on="product_id", how="left"
        )
        oi_cogs["cogs"] = oi_cogs["quantity"] * oi_cogs["cost_price"].fillna(0)
        cogs_agg = oi_cogs.groupby("product_id")["cogs"].sum().reset_index(name="cogs")
        oi_agg = oi_agg.merge(cogs_agg, on="product_id", how="left")
        oi_agg["gross_profit"] = oi_agg["revenue"] - oi_agg["cogs"].fillna(0)
        oi_agg["gross_margin_pct"] = np.where(
            oi_agg["revenue"] > 0, oi_agg["gross_profit"] / oi_agg["revenue"] * 100, 0.0
        )

        p360 = p360.merge(oi_agg, on="product_id", how="left")

        ret_agg = returns.groupby("product_id")["return_id"].nunique().reset_index(name="total_returns")
        p360 = p360.merge(ret_agg, on="product_id", how="left")
        p360["total_returns"] = p360["total_returns"].fillna(0).astype(int)
        p360["return_rate_pct"] = np.where(
            p360["units_sold"].fillna(0) > 0,
            p360["total_returns"] / p360["units_sold"] * 100,
            0.0,
        ).round(2)

        rv_agg = reviews.groupby("product_id").agg(
            total_reviews=("review_id", "nunique"),
            avg_rating=("rating", "mean"),
            five_star_reviews=("rating", lambda x: (x == 5).sum()),
        ).reset_index()
        p360 = p360.merge(rv_agg, on="product_id", how="left")
        p360["total_reviews"] = p360["total_reviews"].fillna(0).astype(int)
        p360["five_star_reviews"] = p360["five_star_reviews"].fillna(0).astype(int)
        p360["avg_rating"] = p360["avg_rating"].fillna(0.0).round(2)

        inv_agg = inventory.groupby("product_id").agg(
            current_stock_units=("stock_quantity", "sum"),
            at_risk_stores_count=("inventory_id", "count"),
        ).reset_index()
        p360 = p360.merge(inv_agg, on="product_id", how="left")
        p360["current_stock_units"] = p360["current_stock_units"].fillna(0.0)
        p360["at_risk_stores_count"] = p360["at_risk_stores_count"].fillna(0).astype(int)

        p360["inventory_value_cost"] = p360["current_stock_units"] * p360["cost_price"].fillna(0)
        p360["inventory_value_retail"] = p360["current_stock_units"] * p360["selling_price"].fillna(0)

        for c in ["total_orders", "units_sold", "revenue", "discount_given", "cogs", "gross_profit"]:
            if c in p360.columns:
                p360[c] = p360[c].fillna(0.0)
        for c in ["avg_discount_pct", "gross_margin_pct"]:
            if c in p360.columns:
                p360[c] = p360[c].fillna(0.0).round(2)

        logger.info(f"Product 360 built: {len(p360):,} products x {len(p360.columns)} cols")
        return p360

    @staticmethod
    def product_lifecycle_analysis(
        order_items_df: pd.DataFrame,
        orders_df: pd.DataFrame,
        products_df: pd.DataFrame,
    ) -> pd.DataFrame:
        valid_statuses = ["Delivered", "Shipped", "Processing", "Returned"]
        valid_orders = orders_df[orders_df["order_status"].isin(valid_statuses)]
        oi = order_items_df.merge(valid_orders[["order_id", "order_date"]], on="order_id", how="left")

        per_month = oi.merge(
            products_df[["product_id", "product_name", "launch_date"]], on="product_id", how="left"
        )
        per_month["order_month"] = pd.to_datetime(per_month["order_date"]).dt.to_period("M")

        monthly_sales = per_month.groupby(
            ["product_id", "product_name", "order_month", "launch_date"]
        ).agg(
            units_sold=("quantity", "sum"),
            revenue=("line_total", "sum"),
        ).reset_index().sort_values(["product_id", "order_month"])

        monthly_sales["month_seq"] = monthly_sales.groupby("product_id").cumcount() + 1
        monthly_sales["units_mom_growth_pct"] = monthly_sales.groupby("product_id")["units_sold"].pct_change() * 100
        monthly_sales["revenue_mom_growth_pct"] = monthly_sales.groupby("product_id")["revenue"].pct_change() * 100

        product_launch = products_df[["product_id", "launch_date"]].copy()
        product_launch["launch_month"] = pd.to_datetime(product_launch["launch_date"]).dt.to_period("M")

        product_sales_age = monthly_sales.merge(
            product_launch[["product_id", "launch_month"]], on="product_id", how="left"
        )
        product_sales_age["months_since_launch"] = (
            (product_sales_age["order_month"] - product_sales_age["launch_month"]).apply(lambda x: x.n if pd.notna(x) else None)
        )

        def _lifecycle_stage(row):
            months = row["months_since_launch"] if pd.notna(row["months_since_launch"]) else 0
            growth = row["revenue_mom_growth_pct"] if pd.notna(row["revenue_mom_growth_pct"]) else 0
            if months <= 1:
                return "Introduction"
            if months <= 3 and growth >= 10:
                return "Growth"
            if growth > 5:
                return "Growth"
            if -10 <= growth <= 5:
                return "Maturity"
            return "Decline"

        product_sales_age["lifecycle_stage"] = product_sales_age.apply(_lifecycle_stage, axis=1)

        latest = product_sales_age.sort_values("order_month").groupby("product_id").tail(1).copy()
        lifecycle = products_df[["product_id", "product_name", "launch_date"]].merge(
            latest[["product_id", "lifecycle_stage", "months_since_launch", "revenue_mom_growth_pct"]],
            on="product_id", how="left",
        )
        lifecycle["lifecycle_stage"] = lifecycle["lifecycle_stage"].fillna("Not Launched / No Sales")

        logger.info(f"Product lifecycle analysis: {len(lifecycle)} products")
        return lifecycle

    @staticmethod
    def price_elasticity_analysis(
        order_items_df: pd.DataFrame,
        orders_df: pd.DataFrame,
        products_df: pd.DataFrame,
    ) -> pd.DataFrame:
        try:
            from scipy.stats import pearsonr
        except ImportError:
            pearsonr = None

        valid_statuses = ["Delivered", "Shipped", "Processing", "Returned"]
        valid_orders = orders_df[orders_df["order_status"].isin(valid_statuses)]
        oi = order_items_df.merge(valid_orders[["order_id", "order_date"]], on="order_id", how="left")

        results = []
        for pid, group in oi.groupby("product_id"):
            if len(group) < 10:
                continue
            prices = group["unit_price"].values
            quantities = group["quantity"].values

            mean_price = float(np.mean(prices))
            mean_qty = float(np.mean(quantities))
            std_price = float(np.std(prices)) if len(prices) > 1 else 0.0

            if pearsonr is not None and std_price > 0 and np.std(quantities) > 0:
                corr, p_val = pearsonr(prices, quantities)
                elasticity = corr * (np.std(quantities) / np.std(prices)) * (mean_price / mean_qty) if mean_qty != 0 else 0.0
            else:
                corr, p_val = None, None
                elasticity = None

            pname = products_df.loc[products_df["product_id"] == pid, "product_name"].values
            product_name = pname[0] if len(pname) > 0 else None

            results.append({
                "product_id": int(pid),
                "product_name": product_name,
                "samples": int(len(group)),
                "avg_price": round(mean_price, 2),
                "avg_quantity": round(mean_qty, 2),
                "price_std": round(std_price, 2),
                "price_qty_correlation": round(float(corr), 4) if corr is not None else None,
                "p_value": round(float(p_val), 4) if p_val is not None else None,
                "price_elasticity": round(float(elasticity), 4) if elasticity is not None else None,
                "elasticity_type": (
                    "Elastic" if elasticity is not None and abs(elasticity) > 1
                    else "Inelastic" if elasticity is not None and abs(elasticity) < 1
                    else "Unit Elastic" if elasticity is not None else None
                ),
            })

        df = pd.DataFrame(results)
        logger.info(f"Price elasticity analysis: {len(df)} products with sufficient data")
        return df

    def run_all_product_analytics(
        self,
        staging_dir: Optional[Path] = None,
        processed_dir: Optional[Path] = None,
    ) -> Dict:
        from config.settings import settings
        staging = Path(staging_dir) if staging_dir else settings.STAGING_DATA_DIR
        processed = Path(processed_dir) if processed_dir else settings.PROCESSED_DATA_DIR
        processed.mkdir(parents=True, exist_ok=True)

        tables = ["products", "categories", "suppliers", "order_items", "orders",
                  "returns", "reviews", "inventory"]
        dfs = {}
        for t in tables:
            try:
                dfs[t] = self._read(t, staging)
            except FileNotFoundError:
                dfs[t] = pd.DataFrame()

        results = {}
        results["product_360"] = self.build_product_360(staging_dir=staging)

        if all(k in dfs and len(dfs[k]) > 0 for k in ["products", "categories", "suppliers",
                                                       "order_items", "orders"]):
            results["product_matrix"] = self.fe.compute_product_matrix(
                dfs["products"], dfs["order_items"], dfs["orders"], dfs["categories"]
            )

        if all(k in dfs and len(dfs[k]) > 0 for k in ["order_items", "orders", "products"]):
            results["lifecycle"] = self.product_lifecycle_analysis(
                dfs["order_items"], dfs["orders"], dfs["products"]
            )
            results["price_elasticity"] = self.price_elasticity_analysis(
                dfs["order_items"], dfs["orders"], dfs["products"]
            )

        for name, value in results.items():
            if isinstance(value, pd.DataFrame):
                out = processed / f"product_{name}.csv"
                value.to_csv(out, index=False)
                logger.info(f"  Saved product_{name}: {len(value):,} rows")

        logger.info(f"Product analytics complete: {len(results)} artifacts produced")
        return results
