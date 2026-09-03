from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

from config.logging_config import get_logger
from src.transformation.feature_engineering import FeatureEngineer

logger = get_logger(__name__)


class CustomerIntelligence:
    def __init__(self, data_dir: Optional[Path] = None):
        self.data_dir = Path(data_dir) if data_dir else None
        self.feature_engineer = FeatureEngineer(data_dir)

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

    def build_customer_360(
        self,
        data_dir: Optional[Path] = None,
        save: bool = False,
    ) -> pd.DataFrame:
        directory = Path(data_dir) if data_dir else self.data_dir
        if directory is None:
            raise ValueError("data_dir must be provided")

        cust = self._read("customers", directory)
        orders = self._read("orders", directory)
        order_items = self._read("order_items", directory)
        returns = self._read("returns", directory)
        reviews = self._read("reviews", directory)
        payments = self._read("payments", directory)
        sessions = self._read("website_sessions", directory)

        valid_statuses = ["Delivered", "Shipped", "Processing", "Returned"]
        valid_orders = orders[orders["order_status"].isin(valid_statuses)]

        c360 = cust.copy()

        per_order = valid_orders.groupby("customer_id").agg(
            total_orders=("order_id", "nunique"),
            cancelled_orders=("order_status", lambda x: (x == "Cancelled").sum()),
            returned_orders=("order_status", lambda x: (x == "Returned").sum()),
            total_revenue=("order_total", "sum"),
            total_discount=("discount_amount", "sum"),
            avg_order_value=("order_total", "mean"),
            first_order_date=("order_date", "min"),
            last_order_date=("order_date", "max"),
            active_months=("order_date", lambda x: x.dt.to_period("M").nunique()),
        ).reset_index()

        oi_cust = order_items.merge(
            orders[["order_id", "customer_id"]], on="order_id", how="left"
        )
        unique_prods = oi_cust.groupby("customer_id")["product_id"].nunique().reset_index(name="unique_products_bought")
        per_order = per_order.merge(unique_prods, on="customer_id", how="left")
        per_order["unique_products_bought"] = per_order["unique_products_bought"].fillna(0).astype(int)

        ret_cust = returns.groupby("customer_id")["return_id"].nunique().reset_index(name="total_returns")
        per_order = per_order.merge(ret_cust, on="customer_id", how="left")
        per_order["total_returns"] = per_order["total_returns"].fillna(0).astype(int)

        rv_cust = reviews.groupby("customer_id").agg(
            total_reviews=("review_id", "nunique"),
            avg_rating_given=("rating", "mean"),
        ).reset_index()
        per_order = per_order.merge(rv_cust, on="customer_id", how="left")
        per_order["total_reviews"] = per_order["total_reviews"].fillna(0).astype(int)
        per_order["avg_rating_given"] = per_order["avg_rating_given"].fillna(0.0)

        pay_cust = payments[payments["payment_status"] == "Success"].groupby("customer_id")["amount"].sum().reset_index(name="total_paid")
        per_order = per_order.merge(pay_cust, on="customer_id", how="left")
        per_order["total_paid"] = per_order["total_paid"].fillna(0.0)

        ws_cust = sessions.dropna(subset=["customer_id"]).groupby("customer_id")["session_id"].nunique().reset_index(name="website_sessions")
        per_order = per_order.merge(ws_cust, on="customer_id", how="left")
        per_order["website_sessions"] = per_order["website_sessions"].fillna(0).astype(int)

        c360 = c360.merge(per_order, on="customer_id", how="left")

        for c in ["total_orders", "cancelled_orders", "returned_orders"]:
            c360[c] = c360[c].fillna(0).astype(int)
        for c in ["total_revenue", "total_discount", "avg_order_value"]:
            c360[c] = c360[c].fillna(0.0)

        c360["days_since_last_order"] = np.where(
            c360["total_orders"] > 0,
            (pd.Timestamp.now() - pd.to_datetime(c360["last_order_date"])).dt.days,
            (pd.Timestamp.now() - pd.to_datetime(c360["signup_date"])).dt.days,
        )
        c360["customer_tenure_days"] = (pd.Timestamp.now() - pd.to_datetime(c360["signup_date"])).dt.days

        logger.info(f"Customer 360 built: {len(c360):,} customers x {len(c360.columns)} cols")
        return c360

    @staticmethod
    def cohort_analysis(
        customers_df: pd.DataFrame,
        orders_df: pd.DataFrame,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        valid_statuses = ["Delivered", "Shipped", "Processing", "Returned"]
        valid_orders = orders_df[orders_df["order_status"].isin(valid_statuses)]

        cust_cohort = customers_df[["customer_id", "signup_date"]].copy()
        cust_cohort["cohort_month"] = pd.to_datetime(cust_cohort["signup_date"]).dt.to_period("M")

        order_cohort = valid_orders[["customer_id", "order_id", "order_date"]].merge(
            cust_cohort[["customer_id", "cohort_month"]], on="customer_id", how="left"
        )
        order_cohort["activity_month"] = pd.to_datetime(order_cohort["order_date"]).dt.to_period("M")
        order_cohort["months_since_signup"] = (
            (order_cohort["activity_month"] - order_cohort["cohort_month"]).apply(lambda x: x.n if pd.notna(x) else None)
        )

        cohort_sizes = cust_cohort.groupby("cohort_month")["customer_id"].nunique().reset_index(name="cohort_size")

        retention = order_cohort.dropna(subset=["months_since_signup"]).groupby(
            ["cohort_month", "months_since_signup"]
        )["customer_id"].nunique().reset_index(name="active_customers")

        retention = retention.merge(cohort_sizes, on="cohort_month", how="left")
        retention["retention_pct"] = (retention["active_customers"] / retention["cohort_size"] * 100).round(2)
        retention = retention[retention["months_since_signup"].between(0, 11)]
        retention = retention.sort_values(["cohort_month", "months_since_signup"])

        cohort_pivot = retention.pivot(
            index="cohort_month", columns="months_since_signup", values="retention_pct"
        ).round(2)

        cohort_revenue = order_cohort.dropna(subset=["months_since_signup"]).merge(
            valid_orders[["order_id", "order_total"]], on="order_id", how="left"
        ).groupby(["cohort_month", "months_since_signup"])["order_total"].sum().reset_index(name="revenue_inr")
        cohort_revenue = cohort_revenue[cohort_revenue["months_since_signup"].between(0, 11)]

        logger.info(f"Cohort analysis: {len(cohort_sizes)} cohorts, {len(retention)} rows")
        return cohort_pivot, cohort_revenue

    def pareto_analysis(
        self,
        customers_df: pd.DataFrame,
        orders_df: pd.DataFrame,
        order_items_df: pd.DataFrame,
        products_df: pd.DataFrame,
    ) -> Dict:
        valid_statuses = ["Delivered", "Shipped", "Processing", "Returned"]
        valid_orders = orders_df[orders_df["order_status"].isin(valid_statuses)]

        oi_with_cost = order_items_df.merge(
            products_df[["product_id", "cost_price"]], on="product_id", how="left"
        )
        oi_profit = oi_with_cost.merge(
            valid_orders[["order_id", "customer_id"]], on="order_id", how="left"
        )
        oi_profit["line_profit"] = (
            oi_profit["line_total"].fillna(0)
            - oi_profit["quantity"] * oi_profit["cost_price"].fillna(0)
        )

        per_cust = oi_profit.groupby("customer_id").agg(
            total_revenue=("line_total", "sum"),
            total_profit=("line_profit", "sum"),
            total_orders=("order_id", "nunique"),
        ).reset_index()

        per_cust = per_cust.merge(
            customers_df[["customer_id", "first_name", "last_name", "state", "customer_segment"]],
            on="customer_id", how="left",
        )
        per_cust["customer_name"] = per_cust["first_name"].fillna("") + " " + per_cust["last_name"].fillna("")

        per_cust = per_cust.sort_values("total_profit", ascending=False).reset_index(drop=True)
        per_cust["cumulative_profit"] = per_cust["total_profit"].cumsum()
        total_profit = per_cust["total_profit"].sum()
        per_cust["cumulative_profit_pct"] = per_cust["cumulative_profit"] / total_profit * 100
        per_cust["percentile"] = np.arange(1, len(per_cust) + 1) / len(per_cust) * 100

        def _top_pct(pct):
            top_n = int(len(per_cust) * pct / 100)
            return per_cust.head(max(1, top_n))["total_profit"].sum() / total_profit * 100 if total_profit > 0 else 0

        pareto = {
            "total_customers": int(len(per_cust)),
            "total_profit_inr": float(total_profit),
            "top_10pct_profit_share_pct": round(_top_pct(10), 2),
            "top_20pct_profit_share_pct": round(_top_pct(20), 2),
            "top_50pct_profit_share_pct": round(_top_pct(50), 2),
            "top_100": per_cust.head(100).to_dict("records"),
            "full_df": per_cust,
        }

        logger.info(
            f"Pareto analysis: top 10% customers = {pareto['top_10pct_profit_share_pct']:.2f}% profit, "
            f"top 20% = {pareto['top_20pct_profit_share_pct']:.2f}%"
        )
        return pareto

    def run_all_customer_analytics(
        self,
        staging_dir: Optional[Path] = None,
        processed_dir: Optional[Path] = None,
    ) -> Dict:
        from config.settings import settings
        staging = Path(staging_dir) if staging_dir else settings.STAGING_DATA_DIR
        processed = Path(processed_dir) if processed_dir else settings.PROCESSED_DATA_DIR
        processed.mkdir(parents=True, exist_ok=True)

        dfs = {}
        for t in ["customers", "orders", "order_items", "products", "returns", "reviews", "payments", "website_sessions"]:
            try:
                dfs[t] = self._read(t, staging)
            except FileNotFoundError:
                dfs[t] = pd.DataFrame()

        results = {}

        results["customer_360"] = self.build_customer_360(staging_dir=staging)
        results["rfm"] = self.feature_engineer.compute_rfm(dfs["customers"], dfs["orders"])
        results["clv"] = self.feature_engineer.compute_clv(
            dfs["customers"], dfs["orders"], dfs["order_items"], dfs["products"]
        )

        if len(dfs["customers"]) > 0 and len(dfs["orders"]) > 0:
            results["cohort_pivot"], results["cohort_revenue"] = self.cohort_analysis(
                dfs["customers"], dfs["orders"]
            )

        if all(k in dfs and len(dfs[k]) > 0 for k in ["customers", "orders", "order_items", "products"]):
            results["pareto"] = self.pareto_analysis(
                dfs["customers"], dfs["orders"], dfs["order_items"], dfs["products"]
            )

        for name, value in results.items():
            if isinstance(value, pd.DataFrame):
                out = processed / f"customer_{name}.csv"
                value.to_csv(out, index=False)
                logger.info(f"  Saved customer_{name}: {len(value):,} rows")

        results["churn_features"] = self.feature_engineer.compute_churn_features(
            dfs["customers"], dfs["orders"], dfs.get("order_items", pd.DataFrame()),
            dfs.get("payments", pd.DataFrame()), dfs.get("reviews", pd.DataFrame()),
            dfs.get("website_sessions", pd.DataFrame()),
        )
        out = processed / "churn_features.csv"
        results["churn_features"].to_csv(out, index=False)

        logger.info(f"Customer analytics complete: {len(results)} artifacts produced")
        return results
