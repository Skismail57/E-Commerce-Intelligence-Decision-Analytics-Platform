from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

from config.logging_config import get_logger

logger = get_logger(__name__)


class FeatureEngineer:
    def __init__(self, data_dir: Optional[Path] = None):
        self.data_dir = Path(data_dir) if data_dir else None
        self._dfs: Dict[str, pd.DataFrame] = {}

    def _read_csv(self, table: str, data_dir: Optional[Path] = None) -> pd.DataFrame:
        directory = data_dir or self.data_dir
        if directory is None:
            raise ValueError("data_dir must be provided")
        path = Path(directory) / f"{table}.csv"
        df = pd.read_csv(path, low_memory=False)
        for dc in [c for c in df.columns if "date" in c.lower()]:
            try:
                df[dc] = pd.to_datetime(df[dc])
            except Exception:
                pass
        return df

    def load_all(self, data_dir: Optional[Path] = None) -> Dict[str, pd.DataFrame]:
        directory = data_dir or self.data_dir
        tables = [
            "customers", "products", "categories", "orders", "order_items",
            "payments", "returns", "reviews", "inventory", "suppliers",
            "stores", "employees", "marketing_campaigns", "marketing_spend",
            "website_sessions",
        ]
        for t in tables:
            try:
                self._dfs[t] = self._read_csv(t, directory)
            except FileNotFoundError:
                logger.warning(f"Table {t} not found, skipping")
        logger.info(f"Loaded {len(self._dfs)} tables")
        return self._dfs

    @staticmethod
    def compute_rfm(
        customers_df: pd.DataFrame,
        orders_df: pd.DataFrame,
        reference_date: Optional[datetime] = None,
    ) -> pd.DataFrame:
        orders_valid = orders_df[orders_df["order_status"].isin(
            ["Delivered", "Shipped", "Processing", "Returned"]
        )].copy()

        ref = reference_date or orders_valid["order_date"].max()
        if isinstance(ref, pd.Timestamp):
            ref = ref.to_pydatetime()

        customer_orders = orders_valid.groupby("customer_id").agg(
            last_order_date=("order_date", "max"),
            frequency=("order_id", "nunique"),
            monetary_value=("order_total", "sum"),
        ).reset_index()

        rfm = customers_df[["customer_id", "signup_date", "customer_segment"]].merge(
            customer_orders, on="customer_id", how="left"
        )

        rfm["last_order_date"] = rfm["last_order_date"].fillna(rfm["signup_date"])
        rfm["recency_days"] = (ref - pd.to_datetime(rfm["last_order_date"])).dt.days
        rfm["frequency"] = rfm["frequency"].fillna(0).astype(int)
        rfm["monetary_value"] = rfm["monetary_value"].fillna(0.0)

        def _ntile_with_zero(series: pd.Series, n_buckets: int = 5, reverse: bool = False) -> pd.Series:
            mask_zero = series == 0
            nz = series[~mask_zero]
            if len(nz) == 0:
                return pd.Series([1] * len(series), index=series.index)
            if reverse:
                nz_scores = pd.qcut(nz.rank(method="first"), n_buckets, labels=False) + 1
                scores_map = dict(zip(nz.index, n_buckets + 1 - nz_scores))
            else:
                nz_scores = pd.qcut(nz.rank(method="first"), n_buckets, labels=False) + 1
                scores_map = dict(zip(nz.index, nz_scores))
            result = series.index.map(lambda i: scores_map.get(i, 1))
            return result

        rfm["r_score"] = _ntile_with_zero(rfm["recency_days"], 5, reverse=True).astype(int)
        rfm["f_score"] = _ntile_with_zero(rfm["frequency"], 5).astype(int)
        rfm["m_score"] = _ntile_with_zero(rfm["monetary_value"], 5).astype(int)
        rfm["rfm_total"] = rfm["r_score"] + rfm["f_score"] + rfm["m_score"]
        rfm["rfm_cell"] = (rfm["r_score"] * 100 + rfm["f_score"] * 10 + rfm["m_score"]).astype(int)

        def _segment(row) -> str:
            r, f, m = row["r_score"], row["f_score"], row["m_score"]
            if r == 5 and f >= 4 and m >= 4:
                return "Champions"
            if r >= 4 and f >= 3 and m >= 3:
                return "Loyal Customers"
            if r >= 4 and 1 <= f <= 3 and 2 <= m <= 3:
                return "Potential Loyalists"
            if r == 5 and f <= 2 and m <= 2:
                return "New Customers"
            if r == 2 and f >= 2 and m >= 2:
                return "At Risk"
            if r == 1 and f >= 4 and m >= 4:
                return "Can't Lose Them"
            if r <= 2 and f <= 2 and m <= 2:
                return "Lost Customers"
            return "Other"

        rfm["rfm_segment"] = rfm.apply(_segment, axis=1)
        logger.info(f"RFM segmentation complete: {len(rfm):,} customers, "
                    f"{rfm['rfm_segment'].nunique()} segments")
        return rfm

    @staticmethod
    def compute_clv(
        customers_df: pd.DataFrame,
        orders_df: pd.DataFrame,
        order_items_df: pd.DataFrame,
        products_df: pd.DataFrame,
        gross_margin_pct: Optional[float] = None,
        lifespan_months: float = 36.0,
    ) -> pd.DataFrame:
        orders_valid = orders_df[orders_df["order_status"].isin(
            ["Delivered", "Shipped", "Processing", "Returned"]
        )].copy()

        oi_with_cost = order_items_df.merge(
            products_df[["product_id", "cost_price"]], on="product_id", how="left"
        )
        oi_agg = oi_with_cost.groupby("order_id").agg(
            line_revenue=("line_total", "sum"),
            line_cogs=("quantity", lambda x: (x * oi_with_cost.loc[x.index, "cost_price"]).sum()),
        ).reset_index()
        oi_agg["line_profit"] = oi_agg["line_revenue"] - oi_agg["line_cogs"].fillna(0)

        orders_with_profit = orders_valid.merge(oi_agg, on="order_id", how="left")

        per_cust = orders_with_profit.groupby("customer_id").agg(
            total_orders=("order_id", "nunique"),
            total_revenue=("order_total", "sum"),
            total_profit=("line_profit", lambda x: x.fillna(0).sum()),
            first_order=("order_date", "min"),
            last_order=("order_date", "max"),
        ).reset_index()

        cust_base = customers_df[["customer_id", "signup_date"]].merge(
            per_cust, on="customer_id", how="left"
        )

        cust_base["total_orders"] = cust_base["total_orders"].fillna(0).astype(int)
        cust_base["total_revenue"] = cust_base["total_revenue"].fillna(0.0)
        cust_base["total_profit"] = cust_base["total_profit"].fillna(0.0)

        cust_base["avg_order_value"] = np.where(
            cust_base["total_orders"] > 0,
            cust_base["total_revenue"] / cust_base["total_orders"],
            0.0,
        )

        cust_base["lifespan_days"] = np.where(
            cust_base["total_orders"] > 0,
            (pd.to_datetime(cust_base["last_order"]) - pd.to_datetime(cust_base["first_order"])).dt.days,
            0.0,
        )
        cust_base["purchase_frequency"] = np.where(
            cust_base["lifespan_days"] > 0,
            cust_base["total_orders"] / (cust_base["lifespan_days"] / 30.0),
            cust_base["total_orders"] / lifespan_months,
        )

        if gross_margin_pct is None:
            total_rev = cust_base["total_revenue"].sum()
            total_prof = cust_base["total_profit"].sum()
            gm = (total_prof / total_rev) if total_rev > 0 else 0.30
        else:
            gm = gross_margin_pct / 100.0

        cust_base["clv"] = (
            cust_base["avg_order_value"]
            * cust_base["purchase_frequency"]
            * lifespan_months
            * gm
        )
        cust_base["clv"] = cust_base["clv"].round(2)

        cust_base["customer_value_tier"] = pd.qcut(
            cust_base["clv"].rank(method="first"),
            5,
            labels=["Low", "Medium-Low", "Medium", "Medium-High", "High"],
        )

        logger.info(f"CLV calculation complete: {len(cust_base):,} customers, "
                    f"avg CLV = ₹{cust_base['clv'].mean():,.2f}")
        return cust_base

    @staticmethod
    def compute_churn_features(
        customers_df: pd.DataFrame,
        orders_df: pd.DataFrame,
        order_items_df: pd.DataFrame,
        payments_df: pd.DataFrame,
        reviews_df: pd.DataFrame,
        website_sessions_df: pd.DataFrame,
        reference_date: Optional[datetime] = None,
        churn_days: int = 90,
    ) -> pd.DataFrame:
        """
        Compute churn features with proper temporal split to avoid target leakage.
        
        Observation window: Jan 2024 - Sep 2024 (features based on this period)
        Prediction window: Oct 2024 - Dec 2024 (target based on this period)
        
        This ensures the model predicts future churn, not just infers from recency.
        """
        # Set observation end date (Sep 30, 2024)
        observation_end = datetime(2024, 9, 30)
        # Set prediction window (Oct 1, 2024 - Dec 31, 2024)
        prediction_start = datetime(2024, 10, 1)
        prediction_end = datetime(2024, 12, 31)
        
        # Filter orders for observation window (feature calculation)
        obs_orders = orders_df[
            (orders_df["order_date"] <= observation_end)
        ].copy()
        
        # Filter orders for prediction window (target calculation)
        pred_orders = orders_df[
            (orders_df["order_date"] >= prediction_start) &
            (orders_df["order_date"] <= prediction_end)
        ].copy()
        
        # Calculate features from observation window
        customer_orders = obs_orders.groupby("customer_id").agg(
            first_order_date=("order_date", "min"),
            last_order_date=("order_date", "max"),
            total_orders=("order_id", "nunique"),
            avg_order_value=("order_total", "mean"),
            total_spend=("order_total", "sum"),
            avg_discount_amount=("discount_amount", "mean"),
            returns_count=("order_status", lambda x: (x == "Returned").sum()),
        ).reset_index()
        customer_orders["return_rate"] = np.where(
            customer_orders["total_orders"] > 0,
            customer_orders["returns_count"] / customer_orders["total_orders"],
            0.0,
        )
        
        # Calculate discount_usage_pct from avg_discount_amount and avg_order_value
        customer_orders["discount_usage_pct"] = np.where(
            customer_orders["avg_order_value"] > 0,
            (customer_orders["avg_discount_amount"] / customer_orders["avg_order_value"]) * 100,
            0.0,
        )
        
        # Calculate total_units_bought from order_items
        if len(order_items_df) > 0:
            obs_order_items = order_items_df.merge(
                obs_orders[["order_id", "customer_id"]], on="order_id", how="inner"
            )
            customer_units = obs_order_items.groupby("customer_id").agg(
                total_units_bought=("quantity", "sum")
            ).reset_index()
            customer_orders = customer_orders.merge(customer_units, on="customer_id", how="left")
            customer_orders["total_units_bought"] = customer_orders["total_units_bought"].fillna(0).astype(int)
        else:
            customer_orders["total_units_bought"] = 0

        # Session features from observation window
        if len(website_sessions_df) > 0 and "customer_id" in website_sessions_df.columns:
            ws = website_sessions_df.dropna(subset=["customer_id"])
            ws = ws[ws["session_date"] <= observation_end]
            if len(ws) > 0:
                customer_sessions = ws.groupby("customer_id").agg(
                    sessions_count=("session_id", "nunique"),
                    avg_page_views=("page_views", "mean"),
                    avg_session_sec=("session_duration_sec", "mean"),
                    checkout_rate=("checkout_completed", "mean"),
                ).reset_index()
            else:
                customer_sessions = pd.DataFrame(columns=[
                    "customer_id", "sessions_count", "avg_page_views",
                    "avg_session_sec", "checkout_rate",
                ])
        else:
            customer_sessions = pd.DataFrame(columns=[
                "customer_id", "sessions_count", "avg_page_views",
                "avg_session_sec", "checkout_rate",
            ])

        # Review features from observation window
        if len(reviews_df) > 0:
            rv = reviews_df[reviews_df["review_date"] <= observation_end]
            cust_reviews = rv.groupby("customer_id").agg(
                avg_review_rating=("rating", "mean"),
                review_count=("review_id", "nunique"),
            ).reset_index()
        else:
            cust_reviews = pd.DataFrame(columns=["customer_id", "avg_review_rating", "review_count"])

        # Build features dataframe
        features = customers_df[["customer_id", "city", "state", "customer_segment"]].copy()
        if "age" in customers_df.columns:
            features["age"] = customers_df["age"]
        if "gender" in customers_df.columns:
            features["gender"] = customers_df["gender"]

        features = features.merge(customer_orders, on="customer_id", how="left")
        features = features.merge(customer_sessions, on="customer_id", how="left")
        features = features.merge(cust_reviews, on="customer_id", how="left")

        # Fill missing values
        for col in ["total_orders", "review_count", "sessions_count"]:
            if col in features.columns:
                features[col] = features[col].fillna(0).astype(int)

        num_fill_cols = [
            "avg_order_value", "total_spend", "return_rate", "avg_discount_amount",
            "avg_page_views", "avg_session_sec", "checkout_rate",
            "avg_review_rating",
        ]
        for col in num_fill_cols:
            if col in features.columns:
                features[col] = features[col].fillna(0.0)

        # Customer tenure (as of observation end)
        features["customer_tenure_days"] = (
            observation_end - pd.to_datetime(customers_df["signup_date"])
        ).dt.days

        # Days since last order (as of observation end) - this is now a feature, not the target
        features["days_since_last_order"] = np.where(
            features["total_orders"] > 0,
            (observation_end - pd.to_datetime(features["last_order_date"])).dt.days,
            features["customer_tenure_days"],
        )

        # Calculate target: Did customer place ANY order in prediction window?
        pred_customer_orders = pred_orders.groupby("customer_id").size().reset_index(name="pred_window_orders")
        features = features.merge(pred_customer_orders, on="customer_id", how="left")
        features["pred_window_orders"] = features["pred_window_orders"].fillna(0).astype(int)
        
        # Churn label: No orders in prediction window AND had at least 1 order in observation window
        features["churn_label_90d"] = (
            (features["pred_window_orders"] == 0) & 
            (features["total_orders"] >= 1)
        ).astype(int)

        # Drop the prediction window orders column (not needed for training)
        features = features.drop(columns=["pred_window_orders"])

        logger.info(f"Churn features complete: {len(features):,} customers, "
                    f"churn rate = {features['churn_label_90d'].mean():.2%}")
        return features

    @staticmethod
    def compute_product_matrix(
        products_df: pd.DataFrame,
        order_items_df: pd.DataFrame,
        orders_df: pd.DataFrame,
        categories_df: pd.DataFrame,
    ) -> pd.DataFrame:
        orders_valid = orders_df[orders_df["order_status"].isin(
            ["Delivered", "Shipped", "Processing", "Returned"]
        )]
        oi_valid = order_items_df[order_items_df["order_id"].isin(orders_valid["order_id"])]

        prod_stats = oi_valid.merge(
            products_df[["product_id", "cost_price"]], on="product_id", how="left"
        ).merge(
            products_df[["product_id", "selling_price", "category_id", "product_name"]],
            on="product_id", how="left", suffixes=("", "_y")
        )
        if "selling_price_y" in prod_stats.columns:
            prod_stats = prod_stats.drop(columns=["selling_price_y", "category_id_y", "product_name_y"], errors="ignore")

        per_product = prod_stats.groupby("product_id").agg(
            units_sold=("quantity", "sum"),
            revenue_inr=("line_total", "sum"),
        ).reset_index()

        per_product2 = oi_valid.merge(
            products_df[["product_id", "cost_price"]], on="product_id", how="left"
        )
        per_product2["cogs"] = per_product2["quantity"] * per_product2["cost_price"].fillna(0)
        profit = per_product2.groupby("product_id")["cogs"].sum().reset_index(name="cogs")
        per_product = per_product.merge(profit, on="product_id", how="left")
        per_product["profit_inr"] = per_product["revenue_inr"] - per_product["cogs"].fillna(0)
        per_product["margin_ratio"] = np.where(
            per_product["revenue_inr"] > 0,
            per_product["profit_inr"] / per_product["revenue_inr"],
            0.0,
        )

        product_info = products_df[[
            "product_id", "product_name", "category_id", "supplier_id",
            "selling_price", "launch_date", "product_status",
        ]].merge(categories_df, on="category_id", how="left")

        result = product_info.merge(per_product, on="product_id", how="left")
        for c in ["units_sold", "revenue_inr", "cogs", "profit_inr"]:
            result[c] = result[c].fillna(0.0)
        result["margin_ratio"] = result["margin_ratio"].fillna(0.0)

        median_revenue = result["revenue_inr"].median()
        median_margin = result["margin_ratio"].median()

        def _quadrant(row) -> str:
            r_ok = row["revenue_inr"] >= median_revenue
            m_ok = row["margin_ratio"] >= median_margin
            if r_ok and m_ok:
                return "Stars"
            if r_ok and not m_ok:
                return "Volume"
            if not r_ok and not m_ok:
                return "Remove"
            return "Premium"

        result["quadrant"] = result.apply(_quadrant, axis=1)
        logger.info(f"Product matrix complete: {len(result):,} products, "
                    f"quadrants = {result['quadrant'].value_counts().to_dict()}")
        return result

    def run_all_transformations(
        self,
        staging_dir: Optional[Path] = None,
        processed_dir: Optional[Path] = None,
        save: bool = True,
    ) -> Dict[str, pd.DataFrame]:
        from config.settings import settings
        staging = Path(staging_dir) if staging_dir else settings.STAGING_DATA_DIR
        processed = Path(processed_dir) if processed_dir else settings.PROCESSED_DATA_DIR
        processed.mkdir(parents=True, exist_ok=True)

        logger.info(f"Running transformations: {staging} -> {processed}")
        dfs = self.load_all(staging)

        results: Dict[str, pd.DataFrame] = {}

        if "customers" in dfs and "orders" in dfs:
            results["rfm_segments"] = self.compute_rfm(dfs["customers"], dfs["orders"])

        if all(k in dfs for k in ["customers", "orders", "order_items", "products"]):
            results["clv"] = self.compute_clv(
                dfs["customers"], dfs["orders"], dfs["order_items"], dfs["products"]
            )

        churn_tables = ["customers", "orders", "order_items", "payments", "reviews", "website_sessions"]
        if all(k in dfs for k in churn_tables):
            results["churn_features"] = self.compute_churn_features(
                dfs["customers"], dfs["orders"], dfs["order_items"],
                dfs["payments"], dfs["reviews"], dfs["website_sessions"],
            )

        if all(k in dfs for k in ["products", "order_items", "orders", "categories"]):
            results["product_matrix"] = self.compute_product_matrix(
                dfs["products"], dfs["order_items"], dfs["orders"], dfs["categories"]
            )

        if save:
            for name, df in results.items():
                out = processed / f"{name}.csv"
                df.to_csv(out, index=False)
                logger.info(f"  Saved {name}: {len(df):,} rows -> {out}")

        logger.info(f"Transformations complete: {len(results)} datasets produced")
        return results
