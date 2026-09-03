from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

from config.logging_config import get_logger

logger = get_logger(__name__)


class MarketingAnalyzer:
    def __init__(self, data_dir: Optional[Path] = None):
        self.data_dir = Path(data_dir) if data_dir else None

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

    def build_marketing_funnel(
        self,
        data_dir: Optional[Path] = None,
    ) -> pd.DataFrame:
        directory = Path(data_dir) if data_dir else self.data_dir
        if directory is None:
            raise ValueError("data_dir must be provided")

        sessions = self._read("website_sessions", directory)
        campaigns = self._read("marketing_campaigns", directory)
        spend = self._read("marketing_spend", directory)
        orders = self._read("orders", directory)
        customers = self._read("customers", directory)

        if "campaign_id" in sessions.columns and len(campaigns) > 0:
            sessions = sessions.merge(
                campaigns[["campaign_id", "campaign_name", "channel"]],
                on="campaign_id", how="left", suffixes=("", "_camp"),
            )
            if "channel_camp" in sessions.columns:
                sessions["channel"] = sessions.get("channel", pd.Series()).combine_first(
                    sessions["channel_camp"]
                )
        sessions["channel"] = sessions.get("channel", "Unknown").fillna("Unknown")

        sessions["session_day"] = pd.to_datetime(sessions["session_date"]).dt.floor("D")

        funnel = sessions.groupby(["channel", "session_day"]).agg(
            sessions=("session_id", "nunique"),
            page_views=("page_views", "sum"),
            product_views=("product_views", "sum"),
            cart_adds=("cart_adds", "sum"),
            checkouts_started=("checkout_started", "sum"),
            checkouts_completed=("checkout_completed", "sum"),
            signed_in_users=("customer_id", lambda x: x.notna().sum()),
        ).reset_index()

        first_order = customers[["customer_id", "signup_date"]].merge(
            orders.groupby("customer_id")["order_date"].min().reset_index(name="first_order_date"),
            on="customer_id", how="left",
        )

        orders_day = orders.copy()
        orders_day["order_day"] = pd.to_datetime(orders_day["order_date"]).dt.floor("D")
        valid_mask = orders_day["order_status"].isin(["Delivered", "Shipped", "Processing", "Returned"])
        orders_day["valid_revenue"] = np.where(valid_mask, orders_day["order_total"], 0.0)
        orders_day["is_new_customer"] = orders_day["customer_id"].isin(
            first_order[
                first_order["first_order_date"].dt.floor("D") == first_order["signup_date"].dt.floor("D")
            ]["customer_id"].tolist()
        ) if "first_order_date" in first_order.columns else False

        valid_orders = orders_day[valid_mask]
        cust_sessions = sessions.dropna(subset=["customer_id"])[[
            "customer_id", "session_day", "channel"
        ]].drop_duplicates(subset=["customer_id", "session_day"])

        channel_orders = valid_orders.merge(
            cust_sessions, on=["customer_id"], how="left", suffixes=("", "_sess")
        )
        channel_orders = channel_orders[
            channel_orders["order_day"] == channel_orders["session_day"].fillna(channel_orders["order_day"])
        ] if "session_day" in channel_orders.columns else valid_orders

        if "channel_sess" in channel_orders.columns:
            channel_orders["channel"] = channel_orders["channel_sess"]
        channel_orders["channel"] = channel_orders.get("channel", "Unknown").fillna("Unknown")

        ch_order_agg = channel_orders.groupby(["channel", "order_day"]).agg(
            orders=("order_id", "nunique"),
            new_customers=("is_new_customer", "sum"),
            revenue_inr=("valid_revenue", "sum"),
        ).reset_index().rename(columns={"order_day": "session_day"})

        spend["spend_day"] = pd.to_datetime(spend["spend_date"]).dt.floor("D")
        spend_agg = spend.groupby(["channel", "spend_day"]).agg(
            impressions=("impressions", "sum"),
            clicks=("clicks", "sum"),
            spend_inr=("spend_amount", "sum"),
        ).reset_index().rename(columns={"spend_day": "session_day"})

        result = funnel.merge(ch_order_agg, on=["channel", "session_day"], how="left")
        result = result.merge(spend_agg, on=["channel", "session_day"], how="left")

        for c in ["impressions", "clicks", "spend_inr", "orders", "new_customers", "revenue_inr"]:
            result[c] = result[c].fillna(0.0)

        result["ctr_pct"] = np.where(
            result["impressions"] > 0,
            result["clicks"] / result["impressions"] * 100, 0.0
        ).round(4)
        result["cpc"] = np.where(
            result["clicks"] > 0,
            result["spend_inr"] / result["clicks"], 0.0
        ).round(4)
        result["conversion_rate_pct"] = np.where(
            result["sessions"] > 0,
            result["checkouts_completed"] / result["sessions"] * 100, 0.0
        ).round(2)
        result["cac"] = np.where(
            result["new_customers"] > 0,
            result["spend_inr"] / result["new_customers"], 0.0
        ).round(2)
        result["roas"] = np.where(
            result["spend_inr"] > 0,
            result["revenue_inr"] / result["spend_inr"], 0.0
        ).round(2)
        result["session_to_cart_rate_pct"] = np.where(
            result["sessions"] > 0,
            result["cart_adds"] / result["sessions"] * 100, 0.0
        ).round(2)
        result["cart_to_checkout_rate_pct"] = np.where(
            result["cart_adds"] > 0,
            result["checkouts_started"] / result["cart_adds"] * 100, 0.0
        ).round(2)
        result["checkout_completion_rate_pct"] = np.where(
            result["checkouts_started"] > 0,
            result["checkouts_completed"] / result["checkouts_started"] * 100, 0.0
        ).round(2)

        logger.info(f"Marketing funnel built: {len(result):,} channel×day rows")
        return result

    def build_campaign_performance(
        self,
        data_dir: Optional[Path] = None,
    ) -> pd.DataFrame:
        directory = Path(data_dir) if data_dir else self.data_dir
        if directory is None:
            raise ValueError("data_dir must be provided")

        campaigns = self._read("marketing_campaigns", directory)
        spend = self._read("marketing_spend", directory)
        sessions = self._read("website_sessions", directory)
        orders = self._read("orders", directory)

        spend_agg = spend.groupby("campaign_id").agg(
            total_impressions=("impressions", "sum"),
            total_clicks=("clicks", "sum"),
            total_spend=("spend_amount", "sum"),
            days_active=("spend_date", "nunique"),
        ).reset_index()

        sessions_agg = sessions.dropna(subset=["campaign_id"]).groupby("campaign_id").agg(
            total_sessions=("session_id", "nunique"),
            total_page_views=("page_views", "sum"),
            total_cart_adds=("cart_adds", "sum"),
            total_checkouts=("checkout_completed", "sum"),
            customers_reached=("customer_id", "nunique"),
        ).reset_index()

        valid_statuses = ["Delivered", "Shipped", "Processing", "Returned"]
        valid_orders = orders[
            (orders["order_status"].isin(valid_statuses))
            & (orders.get("campaign_id", pd.Series(dtype="Int64")).notna())
        ] if "campaign_id" in orders.columns else pd.DataFrame()

        if len(valid_orders) > 0:
            order_agg = valid_orders.groupby("campaign_id").agg(
                total_orders=("order_id", "nunique"),
                total_revenue=("order_total", "sum"),
                unique_customers=("customer_id", "nunique"),
            ).reset_index()
        else:
            order_agg = pd.DataFrame(columns=["campaign_id", "total_orders", "total_revenue", "unique_customers"])

        perf = campaigns.merge(spend_agg, on="campaign_id", how="left")
        perf = perf.merge(sessions_agg, on="campaign_id", how="left")
        perf = perf.merge(order_agg, on="campaign_id", how="left")

        for c in ["total_impressions", "total_clicks", "total_spend",
                  "total_sessions", "total_page_views", "total_cart_adds",
                  "total_checkouts", "customers_reached",
                  "total_orders", "total_revenue", "unique_customers"]:
            perf[c] = perf[c].fillna(0.0)

        perf["ctr_pct"] = np.where(
            perf["total_impressions"] > 0,
            perf["total_clicks"] / perf["total_impressions"] * 100, 0.0
        ).round(4)
        perf["cpc"] = np.where(
            perf["total_clicks"] > 0,
            perf["total_spend"] / perf["total_clicks"], 0.0
        ).round(4)
        perf["cac"] = np.where(
            perf["unique_customers"] > 0,
            perf["total_spend"] / perf["unique_customers"], 0.0
        ).round(2)
        perf["roas"] = np.where(
            perf["total_spend"] > 0,
            perf["total_revenue"] / perf["total_spend"], 0.0
        ).round(2)
        perf["conversion_rate_pct"] = np.where(
            perf["total_sessions"] > 0,
            perf["total_checkouts"] / perf["total_sessions"] * 100, 0.0
        ).round(2)
        perf["budget_utilization_pct"] = np.where(
            perf["total_budget"] > 0,
            perf["total_spend"] / perf["total_budget"] * 100, 0.0
        ).round(2)
        perf["daily_spend_avg"] = np.where(
            perf["days_active"] > 0,
            perf["total_spend"] / perf["days_active"], 0.0
        ).round(2)

        logger.info(f"Campaign performance: {len(perf)} campaigns")
        return perf

    @staticmethod
    def overall_conversion_funnel(
        website_sessions_df: pd.DataFrame,
        orders_df: pd.DataFrame,
    ) -> Dict:
        sessions = website_sessions_df.copy()
        total_sessions = len(sessions)
        total_page_views = float(sessions["page_views"].sum())
        total_product_views = float(sessions["product_views"].sum())
        total_cart_adds = float(sessions["cart_adds"].sum())
        total_checkouts_started = float(sessions["checkout_started"].sum())
        total_checkouts_completed = float(sessions["checkout_completed"].sum())

        valid_statuses = ["Delivered", "Shipped", "Processing", "Returned"]
        valid_orders = orders_df[orders_df["order_status"].isin(valid_statuses)]
        total_orders = int(len(valid_orders))
        total_customers = int(valid_orders["customer_id"].nunique()) if len(valid_orders) > 0 else 0

        funnel = {
            "total_sessions": total_sessions,
            "total_page_views": total_page_views,
            "total_product_views": total_product_views,
            "total_cart_adds": int(total_cart_adds),
            "total_checkouts_started": int(total_checkouts_started),
            "total_checkouts_completed": int(total_checkouts_completed),
            "total_orders": total_orders,
            "total_unique_customers": total_customers,
            "session_to_pageview": round(total_page_views / total_sessions, 2) if total_sessions > 0 else 0,
            "session_to_productview_pct": round(total_product_views / total_sessions * 100, 2) if total_sessions > 0 else 0,
            "session_to_cart_pct": round(total_cart_adds / total_sessions * 100, 2) if total_sessions > 0 else 0,
            "cart_to_checkout_pct": round(total_checkouts_started / total_cart_adds * 100, 2) if total_cart_adds > 0 else 0,
            "checkout_completion_pct": round(total_checkouts_completed / total_checkouts_started * 100, 2) if total_checkouts_started > 0 else 0,
            "session_to_order_pct": round(total_orders / total_sessions * 100, 4) if total_sessions > 0 else 0,
            "checkout_to_order_pct": round(total_orders / total_checkouts_completed * 100, 2) if total_checkouts_completed > 0 else 0,
        }

        logger.info(
            f"Conversion funnel: sessions→cart={funnel['session_to_cart_pct']}%, "
            f"cart→checkout={funnel['cart_to_checkout_pct']}%, "
            f"session→order={funnel['session_to_order_pct']}%"
        )
        return funnel

    def run_all_marketing_analytics(
        self,
        staging_dir: Optional[Path] = None,
        processed_dir: Optional[Path] = None,
    ) -> Dict:
        from config.settings import settings
        staging = Path(staging_dir) if staging_dir else settings.STAGING_DATA_DIR
        processed = Path(processed_dir) if processed_dir else settings.PROCESSED_DATA_DIR
        processed.mkdir(parents=True, exist_ok=True)

        tables = ["marketing_campaigns", "marketing_spend", "website_sessions",
                  "orders", "customers"]
        dfs = {}
        for t in tables:
            try:
                dfs[t] = self._read(t, staging)
            except FileNotFoundError:
                dfs[t] = pd.DataFrame()

        results = {}
        if all(k in dfs for k in tables):
            results["channel_funnel"] = self.build_marketing_funnel(staging_dir=staging)
            results["campaign_performance"] = self.build_campaign_performance(staging_dir=staging)
            results["overall_funnel"] = self.overall_conversion_funnel(
                dfs["website_sessions"], dfs["orders"]
            )

        for name, value in results.items():
            if isinstance(value, pd.DataFrame):
                out = processed / f"marketing_{name}.csv"
                value.to_csv(out, index=False)
                logger.info(f"  Saved marketing_{name}: {len(value):,} rows")

        logger.info(f"Marketing analytics complete: {len(results)} artifacts produced")
        return results
