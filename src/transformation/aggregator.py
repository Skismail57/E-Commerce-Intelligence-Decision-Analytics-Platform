from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from datetime import datetime

from config.logging_config import get_logger

logger = get_logger(__name__)


class DataAggregator:
    def __init__(self, data_dir: Optional[Path] = None):
        self.data_dir = Path(data_dir) if data_dir else None

    @staticmethod
    def _read_csv(table: str, data_dir: Path) -> pd.DataFrame:
        path = Path(data_dir) / f"{table}.csv"
        df = pd.read_csv(path, low_memory=False)
        for dc in [c for c in df.columns if "date" in c.lower()]:
            try:
                df[dc] = pd.to_datetime(df[dc])
            except Exception:
                pass
        return df

    @staticmethod
    def build_order_line_fact(
        orders_df: pd.DataFrame,
        order_items_df: pd.DataFrame,
        customers_df: pd.DataFrame,
        products_df: pd.DataFrame,
        categories_df: pd.DataFrame,
        suppliers_df: pd.DataFrame,
        stores_df: pd.DataFrame,
        marketing_campaigns_df: pd.DataFrame,
        payments_df: pd.DataFrame,
    ) -> pd.DataFrame:
        fact = order_items_df.merge(orders_df, on="order_id", how="left", suffixes=("", "_ord"))
        fact = fact.merge(
            customers_df[["customer_id", "first_name", "last_name", "gender", "age", "city", "state", "customer_segment"]],
            on="customer_id", how="left",
        )
        fact["customer_name"] = fact["first_name"].fillna("") + " " + fact["last_name"].fillna("")
        fact = fact.merge(
            products_df[["product_id", "product_name", "category_id", "supplier_id", "cost_price", "selling_price", "brand_name"]],
            on="product_id", how="left", suffixes=("", "_prod"),
        )
        fact = fact.merge(categories_df, on="category_id", how="left")
        fact = fact.merge(suppliers_df[["supplier_id", "supplier_name"]], on="supplier_id", how="left")
        if "store_id" in fact.columns and "store_id" in stores_df.columns:
            fact = fact.merge(
                stores_df[["store_id", "store_name", "city", "state"]],
                on="store_id", how="left", suffixes=("", "_store"),
            )
        if "campaign_id" in fact.columns and len(marketing_campaigns_df) > 0:
            fact = fact.merge(
                marketing_campaigns_df[["campaign_id", "campaign_name", "channel"]],
                on="campaign_id", how="left", suffixes=("", "_camp"),
            )
        if "payment_id" in fact.columns and len(payments_df) > 0:
            fact = fact.merge(
                payments_df[["payment_id", "payment_method", "payment_status"]],
                on="payment_id", how="left",
            )

        if "order_date" in fact.columns:
            od = pd.to_datetime(fact["order_date"])
            fact["order_day"] = od.dt.floor("D")
            fact["order_week"] = od.dt.to_period("W").dt.to_timestamp()
            fact["order_month"] = od.dt.to_period("M").dt.to_timestamp()
            fact["order_quarter"] = od.dt.to_period("Q").dt.to_timestamp()
            fact["order_year"] = od.dt.year
            fact["order_dow"] = od.dt.isoweekday()
            fact["order_hour"] = od.dt.hour

        fact["line_cogs"] = fact["quantity"] * fact["cost_price"].fillna(0)
        fact["line_gross_profit"] = fact["line_total"].fillna(0) - fact["line_cogs"]
        fact["line_margin_pct"] = np.where(
            fact["line_total"].fillna(0) > 0,
            fact["line_gross_profit"] / fact["line_total"] * 100,
            0.0,
        )

        logger.info(f"Order line fact built: {len(fact):,} rows x {len(fact.columns)} cols")
        return fact

    @staticmethod
    def aggregate_daily_sales(
        orders_df: pd.DataFrame,
        order_items_df: pd.DataFrame,
        customers_df: pd.DataFrame,
    ) -> pd.DataFrame:
        orders_valid = orders_df.copy()
        if "order_date" not in orders_valid.columns:
            raise ValueError("orders_df must contain order_date")

        orders_valid["order_day"] = pd.to_datetime(orders_valid["order_date"]).dt.floor("D")
        is_valid = orders_valid["order_status"].isin(["Delivered", "Shipped", "Processing", "Returned"])
        is_delivered = orders_valid["order_status"].isin(["Delivered", "Shipped", "Processing"])
        is_returned = orders_valid["order_status"] == "Returned"
        is_cancelled = orders_valid["order_status"] == "Cancelled"

        first_orders = orders_valid.groupby("customer_id")["order_day"].min().reset_index(name="first_order_day")
        orders_valid = orders_valid.merge(first_orders, on="customer_id", how="left")
        orders_valid["is_new_customer"] = orders_valid["order_day"] == orders_valid["first_order_day"]

        oi_daily = order_items_df.merge(
            orders_valid[["order_id", "order_day"]], on="order_id", how="left"
        ).groupby("order_day").agg(
            units_sold=("quantity", "sum"),
            unique_products_sold=("product_id", "nunique"),
        ).reset_index()

        daily = orders_valid.groupby("order_day").agg(
            gross_revenue=("order_total", lambda x: x[is_valid.loc[x.index]].sum()),
            total_discount=("discount_amount", lambda x: x[is_valid.loc[x.index]].sum()),
            total_tax=("tax_amount", lambda x: x[is_valid.loc[x.index]].sum()),
            total_shipping=("shipping_cost", lambda x: x[is_valid.loc[x.index]].sum()),
            returned_revenue=("order_total", lambda x: x[is_returned.loc[x.index]].sum()),
            cancelled_revenue=("order_total", lambda x: x[is_cancelled.loc[x.index]].sum()),
            total_orders=("order_id", "nunique"),
            delivered_orders=("order_status", lambda x: (x == "Delivered").sum()),
            shipped_orders=("order_status", lambda x: (x == "Shipped").sum()),
            processing_orders=("order_status", lambda x: (x == "Processing").sum()),
            returned_orders=("order_status", lambda x: (x == "Returned").sum()),
            cancelled_orders=("order_status", lambda x: (x == "Cancelled").sum()),
            unique_customers=("customer_id", "nunique"),
            new_customers=("is_new_customer", "sum"),
        ).reset_index()

        net_rev = daily["gross_revenue"] * orders_valid["order_status"].isin(
            ["Delivered", "Shipped", "Processing"]
        ).mean()
        daily["net_revenue"] = (
            daily["gross_revenue"]
            - daily["returned_revenue"]
            - daily["cancelled_revenue"]
        )

        daily = daily.merge(oi_daily, on="order_day", how="left")
        for c in ["units_sold", "unique_products_sold"]:
            daily[c] = daily[c].fillna(0).astype(int)

        logger.info(f"Daily sales aggregation: {len(daily)} days")
        return daily.sort_values("order_day").reset_index(drop=True)

    @staticmethod
    def aggregate_monthly_kpis(
        orders_df: pd.DataFrame,
        order_items_df: pd.DataFrame,
        customers_df: pd.DataFrame,
        products_df: pd.DataFrame,
    ) -> pd.DataFrame:
        orders_valid = orders_df.copy()
        orders_valid["order_month"] = pd.to_datetime(orders_valid["order_date"]).dt.to_period("M").dt.to_timestamp()
        valid_mask = orders_valid["order_status"].isin(["Delivered", "Shipped", "Processing", "Returned"])

        monthly = orders_valid[valid_mask].groupby("order_month").agg(
            total_revenue_inr=("order_total", "sum"),
            orders=("order_id", "nunique"),
            unique_customers=("customer_id", "nunique"),
            return_count=("order_status", lambda x: (x == "Returned").sum()),
            avg_order_value_inr=("order_total", "mean"),
        ).reset_index()

        oi_valid = order_items_df.merge(
            orders_valid[["order_id", "order_month"]], on="order_id", how="left"
        ).merge(
            products_df[["product_id", "cost_price"]], on="product_id", how="left"
        )
        oi_valid["line_profit"] = (
            oi_valid["line_total"].fillna(0)
            - oi_valid["quantity"] * oi_valid["cost_price"].fillna(0)
        )

        m_profit = oi_valid.groupby("order_month").agg(
            units_sold=("quantity", "sum"),
            gross_profit_inr=("line_profit", "sum"),
            total_revenue_oi=("line_total", "sum"),
        ).reset_index()

        monthly = monthly.merge(m_profit, on="order_month", how="left")
        monthly["gross_margin_pct"] = np.where(
            monthly["total_revenue_oi"].fillna(0) > 0,
            monthly["gross_profit_inr"].fillna(0) / monthly["total_revenue_oi"] * 100,
            0.0,
        )
        monthly["return_rate_pct"] = np.where(
            monthly["orders"] > 0,
            monthly["return_count"] / monthly["orders"] * 100,
            0.0,
        )

        for c in ["units_sold", "gross_profit_inr"]:
            monthly[c] = monthly[c].fillna(0.0)

        logger.info(f"Monthly KPIs: {len(monthly)} months")
        return monthly.sort_values("order_month").reset_index(drop=True)

    @staticmethod
    def aggregate_region_performance(
        customers_df: pd.DataFrame,
        orders_df: pd.DataFrame,
    ) -> pd.DataFrame:
        orders_valid = orders_df[orders_df["order_status"].isin(
            ["Delivered", "Shipped", "Processing", "Returned"]
        )]
        cust_orders = customers_df[["customer_id", "state", "city"]].merge(
            orders_valid[["order_id", "customer_id", "order_total", "order_status"]],
            on="customer_id", how="inner",
        )

        region = cust_orders.groupby("state").agg(
            orders=("order_id", "nunique"),
            customers=("customer_id", "nunique"),
            revenue_inr=("order_total", "sum"),
            return_count=("order_status", lambda x: (x == "Returned").sum()),
            cities_reached=("city", "nunique"),
        ).reset_index()

        region["aov_inr"] = np.where(region["orders"] > 0, region["revenue_inr"] / region["orders"], 0.0)
        region["return_rate_pct"] = np.where(
            region["orders"] > 0,
            region["return_count"] / region["orders"] * 100,
            0.0,
        )
        region = region.sort_values("revenue_inr", ascending=False).reset_index(drop=True)
        logger.info(f"Region performance: {len(region)} states")
        return region

    @staticmethod
    def aggregate_category_performance(
        products_df: pd.DataFrame,
        categories_df: pd.DataFrame,
        order_items_df: pd.DataFrame,
        orders_df: pd.DataFrame,
        returns_df: pd.DataFrame,
        reviews_df: pd.DataFrame,
    ) -> pd.DataFrame:
        orders_valid = orders_df[orders_df["order_status"].isin(
            ["Delivered", "Shipped", "Processing", "Returned"]
        )]
        oi_valid = order_items_df[order_items_df["order_id"].isin(orders_valid["order_id"])]

        cat_stats = oi_valid.merge(
            products_df[["product_id", "category_id", "cost_price"]],
            on="product_id", how="left",
        ).merge(categories_df, on="category_id", how="left")

        cat_stats["line_cogs"] = cat_stats["quantity"] * cat_stats["cost_price"].fillna(0)
        cat_stats["line_profit"] = cat_stats["line_total"].fillna(0) - cat_stats["line_cogs"]

        cat_grp = cat_stats.groupby(["category_name", "subcategory"]).agg(
            orders=("order_id", "nunique"),
            revenue_inr=("line_total", "sum"),
            gross_profit_inr=("line_profit", "sum"),
            units_sold=("quantity", "sum"),
            skus=("product_id", "nunique"),
        ).reset_index()

        cat_grp["gross_margin_pct"] = np.where(
            cat_grp["revenue_inr"] > 0,
            cat_grp["gross_profit_inr"] / cat_grp["revenue_inr"] * 100,
            0.0,
        )

        ret_counts = returns_df.merge(
            products_df[["product_id", "category_id"]], on="product_id", how="left"
        ).merge(categories_df, on="category_id", how="left")
        ret_grp = ret_counts.groupby(["category_name", "subcategory"])["return_id"].nunique().reset_index(name="returns")
        cat_grp = cat_grp.merge(ret_grp, on=["category_name", "subcategory"], how="left")
        cat_grp["returns"] = cat_grp["returns"].fillna(0).astype(int)
        cat_grp["return_rate_pct"] = np.where(
            cat_grp["units_sold"] > 0,
            cat_grp["returns"] / cat_grp["units_sold"] * 100,
            0.0,
        )

        rv = reviews_df.merge(
            products_df[["product_id", "category_id"]], on="product_id", how="left"
        ).merge(categories_df, on="category_id", how="left")
        rv_grp = rv.groupby(["category_name", "subcategory"])["rating"].mean().reset_index(name="avg_rating")
        cat_grp = cat_grp.merge(rv_grp, on=["category_name", "subcategory"], how="left")
        cat_grp["avg_rating"] = cat_grp["avg_rating"].fillna(0.0).round(2)

        cat_grp = cat_grp.sort_values("revenue_inr", ascending=False).reset_index(drop=True)
        logger.info(f"Category performance: {len(cat_grp)} category×subcategory rows")
        return cat_grp

    @staticmethod
    def aggregate_inventory_health(
        inventory_df: pd.DataFrame,
        products_df: pd.DataFrame,
        categories_df: pd.DataFrame,
        stores_df: pd.DataFrame,
        order_items_df: pd.DataFrame,
        orders_df: pd.DataFrame,
        demand_window_days: int = 30,
    ) -> pd.DataFrame:
        max_date = orders_df["order_date"].max() if len(orders_df) > 0 else pd.Timestamp.now()
        min_date = max_date - pd.Timedelta(days=demand_window_days)

        oi_window = order_items_df.merge(
            orders_df[["order_id", "order_date"]], on="order_id", how="left"
        )
        oi_window = oi_window[
            (oi_window["order_date"] >= min_date)
            & (oi_window["order_date"] <= max_date)
            & oi_window["order_status"].isin(["Delivered", "Shipped", "Processing", "Returned"])
            if "order_status" in oi_window.columns
            else (oi_window["order_date"] >= min_date) & (oi_window["order_date"] <= max_date)
        ]
        avg_demand = oi_window.groupby("product_id")["quantity"].sum().reset_index()
        avg_demand["average_daily_demand"] = avg_demand["quantity"] / demand_window_days
        avg_demand = avg_demand.drop(columns=["quantity"])

        health = inventory_df.merge(
            products_df[["product_id", "product_name", "category_id", "cost_price", "selling_price"]],
            on="product_id", how="left",
        ).merge(categories_df, on="category_id", how="left")
        health = health.merge(
            stores_df[["store_id", "store_name", "city", "state"]],
            on="store_id", how="left", suffixes=("_prod", "_store"),
        )
        health = health.merge(avg_demand, on="product_id", how="left")
        health["average_daily_demand"] = health["average_daily_demand"].fillna(0.0)

        lead_time = health["lead_time_days"].fillna(7)
        safety = health["safety_stock"].fillna(0)

        health["lead_time_demand"] = health["average_daily_demand"] * lead_time
        health["recommended_reorder_point"] = health["lead_time_demand"] + safety
        health["recommended_reorder_qty"] = (
            (health["average_daily_demand"] * lead_time * 2 + safety - health["stock_quantity"].fillna(0))
            .clip(lower=0)
            .round()
            .astype(int)
        )

        def _stock_status(row):
            sq = row["stock_quantity"] if pd.notna(row["stock_quantity"]) else 0
            ss = row["safety_stock"] if pd.notna(row["safety_stock"]) else 0
            rl = row["reorder_level"] if pd.notna(row["reorder_level"]) else 50
            if sq == 0:
                return "Out of Stock"
            if sq <= ss:
                return "Critical"
            if sq <= rl:
                return "Reorder"
            return "OK"

        health["stock_status"] = health.apply(_stock_status, axis=1)
        health["days_of_inventory"] = np.where(
            health["average_daily_demand"] > 0,
            (health["stock_quantity"].fillna(0) / health["average_daily_demand"]).round(1),
            None,
        )
        health["inventory_value_cost"] = health["stock_quantity"].fillna(0) * health["cost_price"].fillna(0)
        health["inventory_value_retail"] = health["stock_quantity"].fillna(0) * health["selling_price"].fillna(0)
        health["overstock_flag"] = np.where(
            health["stock_quantity"].fillna(0) > (health["reorder_level"].fillna(50) * 5),
            "Overstock",
            "Normal",
        )

        logger.info(f"Inventory health: {len(health):,} product×store rows")
        return health

    def run_all_aggregations(
        self,
        staging_dir: Optional[Path] = None,
        processed_dir: Optional[Path] = None,
        save: bool = True,
    ) -> Dict[str, pd.DataFrame]:
        from config.settings import settings
        staging = Path(staging_dir) if staging_dir else settings.STAGING_DATA_DIR
        processed = Path(processed_dir) if processed_dir else settings.PROCESSED_DATA_DIR
        processed.mkdir(parents=True, exist_ok=True)

        logger.info(f"Running aggregations: {staging} -> {processed}")
        tables = [
            "customers", "products", "categories", "suppliers", "stores",
            "orders", "order_items", "payments", "returns", "reviews",
            "marketing_campaigns", "inventory",
        ]
        dfs: Dict[str, pd.DataFrame] = {}
        for t in tables:
            try:
                dfs[t] = self._read_csv(t, staging)
            except FileNotFoundError:
                logger.warning(f"Table {t} not found, skipping")

        results: Dict[str, pd.DataFrame] = {}

        req_order_line = ["orders", "order_items", "customers", "products",
                          "categories", "suppliers", "stores"]
        if all(k in dfs for k in req_order_line):
            results["order_line_fact"] = self.build_order_line_fact(
                dfs["orders"], dfs["order_items"], dfs["customers"],
                dfs["products"], dfs["categories"], dfs["suppliers"],
                dfs["stores"],
                dfs.get("marketing_campaigns", pd.DataFrame()),
                dfs.get("payments", pd.DataFrame()),
            )

        if all(k in dfs for k in ["orders", "order_items", "customers"]):
            results["daily_sales"] = self.aggregate_daily_sales(
                dfs["orders"], dfs["order_items"], dfs["customers"]
            )

        if all(k in dfs for k in ["orders", "order_items", "customers", "products"]):
            results["monthly_kpis"] = self.aggregate_monthly_kpis(
                dfs["orders"], dfs["order_items"], dfs["customers"], dfs["products"]
            )

        if all(k in dfs for k in ["customers", "orders"]):
            results["region_performance"] = self.aggregate_region_performance(
                dfs["customers"], dfs["orders"]
            )

        cat_req = ["products", "categories", "order_items", "orders", "returns", "reviews"]
        if all(k in dfs for k in cat_req):
            results["category_performance"] = self.aggregate_category_performance(
                dfs["products"], dfs["categories"], dfs["order_items"],
                dfs["orders"], dfs["returns"], dfs["reviews"],
            )

        inv_req = ["inventory", "products", "categories", "stores", "order_items", "orders"]
        if all(k in dfs for k in inv_req):
            results["inventory_health"] = self.aggregate_inventory_health(
                dfs["inventory"], dfs["products"], dfs["categories"],
                dfs["stores"], dfs["order_items"], dfs["orders"],
            )

        if save:
            for name, df in results.items():
                out = processed / f"{name}.csv"
                df.to_csv(out, index=False)
                logger.info(f"  Saved {name}: {len(df):,} rows -> {out}")

        logger.info(f"Aggregations complete: {len(results)} datasets produced")
        return results
