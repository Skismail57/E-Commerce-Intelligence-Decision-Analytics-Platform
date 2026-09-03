from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from config.logging_config import get_logger
from config.settings import settings
from src.transformation.aggregator import DataAggregator

logger = get_logger(__name__)


class DemandForecaster:
    """Demand forecasting with multiple model comparison + ML regression ensemble."""

    def __init__(self, data_dir: Optional[Path] = None, processed_dir: Optional[Path] = None):
        self.data_dir = Path(data_dir) if data_dir else settings.STAGING_DATA_DIR
        self.processed_dir = Path(processed_dir) if processed_dir else settings.PROCESSED_DATA_DIR
        settings.ensure_dirs()
        self.model_results: Dict[str, Dict] = {}

    @staticmethod
    def _build_ts_from_orders(orders_df: pd.DataFrame, order_items_df: pd.DataFrame,
                               products_df: Optional[pd.DataFrame] = None,
                               granularity: str = "D",
                               product_id: Optional[int] = None,
                               category_id: Optional[int] = None) -> pd.DataFrame:
        """Build a time series with zero-filled gaps for consistent forecasting."""
        orders_cols = {c.lower(): c for c in orders_df.columns}
        items_cols = {c.lower(): c for c in order_items_df.columns}

        o = orders_df.copy()
        # Find order date column
        date_col = None
        for dc in [orders_cols.get("order_date"), orders_cols.get("order date")]:
            if dc and dc in o.columns:
                date_col = dc
                break
        if date_col is None:
            # Try to find any column containing 'date'
            for c in orders_cols.values():
                if c in o.columns and "date" in c.lower():
                    date_col = c
                    break
        if date_col is None:
            raise ValueError("No order date column found")
        o[date_col] = pd.to_datetime(o[date_col], errors="coerce")

        oi = order_items_df.copy()
        key_order = [c for c in [orders_cols.get("order_id"), "order_id"] if c in orders_df.columns][0]
        key_prod = [c for c in [items_cols.get("product_id"), "product_id"] if c in oi.columns][0]
        key_qty = [c for c in [items_cols.get("quantity"), "quantity", "qty"] if c in oi.columns][0]
        key_price = [c for c in [items_cols.get("unit_price"), "unit_price", "price"] if c in oi.columns][0]

        merged = oi.merge(o[[key_order, date_col]], on=key_order, how="inner")

        if product_id is not None and key_prod in merged.columns:
            merged = merged[merged[key_prod] == product_id]
        if category_id is not None and products_df is not None:
            cat_col = [c for c in ["category_id"] if c in products_df.columns][0]
            prods_in_cat = products_df[products_df[cat_col] == category_id][key_prod]
            merged = merged[merged[key_prod].isin(prods_in_cat)]

        merged = merged.dropna(subset=[date_col])
        merged["_date"] = merged[date_col].dt.floor(granularity)
        ts = (merged.groupby("_date")
                     .agg(units_sold=(key_qty, "sum"),
                          revenue_inr=(key_qty, lambda s: (s * merged.loc[s.index, key_price]).sum()))
                     .reset_index()
                     .rename(columns={"_date": "date"}))
        ts["date"] = pd.to_datetime(ts["date"])
        idx = pd.date_range(ts["date"].min(), ts["date"].max(), freq=granularity)
        ts = ts.set_index("date").reindex(idx).fillna(0).rename_axis("date").reset_index()
        return ts

    @staticmethod
    def _split(ts: pd.DataFrame, horizon: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
        train = ts.iloc[:-horizon] if len(ts) > horizon else ts.copy()
        test = ts.iloc[-horizon:] if len(ts) > horizon else ts.tail(0)
        return train, test

    @staticmethod
    def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
        return float(np.mean(np.abs(y_true - y_pred)))

    @staticmethod
    def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
        return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))

    @classmethod
    def mape(cls, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        y_true, y_pred = np.asarray(y_true, dtype=float), np.asarray(y_pred, dtype=float)
        mask = y_true != 0
        if not mask.any():
            return 0.0
        return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)

    def forecast_moving_average(self, ts: pd.DataFrame, horizon: int, windows: Tuple[int, ...] = (7, 14, 28)) -> Dict:
        units = ts["units_sold"].values
        preds_ensemble = np.zeros(horizon, dtype=float)
        counts = np.zeros(horizon, dtype=float)
        all_results = []
        for w in windows:
            if len(units) >= w:
                roll = pd.Series(units).rolling(w, min_periods=1).mean().values
            else:
                roll = np.full(len(units), float(np.mean(units)) if len(units) else 0.0)
            base = float(roll[-1])
            pred = np.full(horizon, base)
            preds_ensemble += pred
            counts += 1
            all_results.append({f"ma_{w}": pred.tolist()})
        ensemble = preds_ensemble / np.where(counts == 0, 1, counts)
        return {"forecast": ensemble, "components": all_results}

    def forecast_exponential_smoothing(self, ts: pd.DataFrame, horizon: int, alphas: Tuple[float, ...] = (0.2, 0.5, 0.8)) -> Dict:
        series = ts["units_sold"].astype(float).values
        best = None
        best_alpha = None
        best_error = float("inf")
        for alpha in alphas:
            level = series[0] if len(series) else 0.0
            fitted = np.zeros(len(series))
            for i in range(len(series)):
                fitted[i] = level
                level = alpha * series[i] + (1 - alpha) * level
            if len(series) > 2:
                err = self.mape(series[1:], fitted[1:])
                if err < best_error:
                    best_error = err
                    best = level
                    best_alpha = alpha
            else:
                best = level
                best_alpha = alpha
        forecast = np.full(horizon, float(best))
        return {"forecast": forecast, "best_alpha": best_alpha, "best_in_sample_mape": best_error}

    def forecast_snaive(self, ts: pd.DataFrame, horizon: int, seasonality: int = 7) -> Dict:
        series = ts["units_sold"].astype(float).values
        if len(series) < seasonality:
            seasonality = max(1, len(series))
        idx = (np.arange(horizon) % seasonality) - seasonality
        forecast = series[idx] if len(series) else np.zeros(horizon)
        return {"forecast": forecast, "seasonality": seasonality}

    def forecast_arima(self, ts: pd.DataFrame, horizon: int, order: Tuple[int, int, int] = (1, 1, 1)) -> Dict:
        series = ts["units_sold"].astype(float).values
        try:
            from statsmodels.tsa.arima.model import ARIMA
            model = ARIMA(series, order=order)
            fit = model.fit()
            fcast = fit.forecast(steps=horizon)
            return {"forecast": np.asarray(fcast), "order": order, "aic": float(getattr(fit, "aic", 0.0))}
        except Exception as e:
            logger.info(f"ARIMA unavailable, using ES fallback: {e}")
            return self.forecast_exponential_smoothing(ts, horizon)

    def forecast_prophet(self, ts: pd.DataFrame, horizon: int) -> Dict:
        df = ts.rename(columns={"date": "ds", "units_sold": "y"})[["ds", "y"]]
        try:
            from prophet import Prophet
            m = Prophet(daily_seasonality=False, weekly_seasonality=True, yearly_seasonality=True)
            m.fit(df)
            future = m.make_future_dataframe(periods=horizon)
            f = m.predict(future)
            return {"forecast": f["yhat"].tail(horizon).values}
        except Exception as e:
            logger.info(f"Prophet unavailable, using ES fallback: {e}")
            return self.forecast_exponential_smoothing(ts, horizon)

    @staticmethod
    def _build_ml_features(ts: pd.DataFrame, target: str = "units_sold") -> pd.DataFrame:
        df = ts.set_index("date").copy()
        df["t"] = np.arange(len(df))
        df["dow"] = df.index.dayofweek
        df["dom"] = df.index.day
        df["week"] = df.index.isocalendar().week.astype(int)
        df["month"] = df.index.month
        df["quarter"] = df.index.quarter
        df["year"] = df.index.year
        df["is_weekend"] = df["dow"].isin([5, 6]).astype(int)
        for w in [3, 7, 14, 28]:
            df[f"lag_{w}"] = df[target].shift(w)
            df[f"ma_{w}"] = df[target].shift(1).rolling(w, min_periods=1).mean()
        df = df.bfill().ffill().fillna(0)
        return df.reset_index(drop=False)

    def time_series_cross_validation(
        self,
        ts: pd.DataFrame,
        model_method,
        n_splits: int = 5,
        horizon: int = 30
    ) -> Dict[str, List[float]]:
        """
        Perform time series cross-validation for a forecasting model.
        
        Args:
            ts: Time series DataFrame with 'date' and 'units_sold' columns
            model_method: Forecasting method function
            n_splits: Number of CV splits
            horizon: Forecast horizon for each split
        
        Returns:
            Dictionary with MAE, RMSE, MAPE values for each split
        """
        if len(ts) < n_splits * horizon + horizon:
            logger.warning(f"Insufficient data for {n_splits}-fold CV with horizon {horizon}")
            n_splits = max(1, len(ts) // (horizon * 2))
        
        cv_metrics = {"mae": [], "rmse": [], "mape_pct": []}
        
        for i in range(n_splits):
            # Calculate split indices
            test_start_idx = len(ts) - (n_splits - i) * horizon
            train_end_idx = test_start_idx
            
            if train_end_idx < horizon:
                break
            
            train = ts.iloc[:train_end_idx]
            test = ts.iloc[test_start_idx:test_start_idx + horizon]
            
            if len(test) == 0:
                continue
            
            # Fit and predict
            result = model_method(train, horizon)
            y_pred = result["forecast"][:len(test)]
            y_true = test["units_sold"].values
            
            # Calculate metrics
            cv_metrics["mae"].append(self.mae(y_true, y_pred))
            cv_metrics["rmse"].append(self.rmse(y_true, y_pred))
            cv_metrics["mape_pct"].append(self.mape(y_true, y_pred))
        
        # Calculate mean and std of metrics
        cv_results = {
            "mae_mean": np.mean(cv_metrics["mae"]) if cv_metrics["mae"] else None,
            "mae_std": np.std(cv_metrics["mae"]) if cv_metrics["mae"] else None,
            "rmse_mean": np.mean(cv_metrics["rmse"]) if cv_metrics["rmse"] else None,
            "rmse_std": np.std(cv_metrics["rmse"]) if cv_metrics["rmse"] else None,
            "mape_mean": np.mean(cv_metrics["mape_pct"]) if cv_metrics["mape_pct"] else None,
            "mape_std": np.std(cv_metrics["mape_pct"]) if cv_metrics["mape_pct"] else None,
            "n_splits": len(cv_metrics["mae"])
        }
        
        return cv_results

    def forecast_ml_ensemble(self, ts: pd.DataFrame, horizon: int) -> Dict:
        feature_df = self._build_ml_features(ts, target="units_sold")
        feature_cols = [c for c in feature_df.columns if c not in ("date", "units_sold", "revenue_inr")]
        train = feature_df.iloc[: len(ts)]
        y_train = ts["units_sold"].astype(float).values[: len(train)]
        X_train = train[feature_cols].values

        last_row = feature_df.iloc[[-1]].copy()
        future_rows = []
        for h in range(1, horizon + 1):
            nr = last_row.copy()
            nr["t"] += h
            date_h = (pd.Timestamp(ts["date"].max()) + pd.Timedelta(days=h))
            nr["dow"] = date_h.dayofweek
            nr["dom"] = date_h.day
            nr["week"] = int(date_h.isocalendar()[1])
            nr["month"] = date_h.month
            nr["quarter"] = (date_h.month - 1) // 3 + 1
            nr["year"] = date_h.year
            nr["is_weekend"] = int(date_h.dayofweek in (5, 6))
            future_rows.append(nr.iloc[0])
        future_df = pd.DataFrame(future_rows)
        future_df = future_df[feature_cols].fillna(0)

        try:
            from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
            rf = RandomForestRegressor(n_estimators=100, max_depth=8, random_state=42)
            gb = GradientBoostingRegressor(n_estimators=100, max_depth=4, random_state=42)
            rf.fit(X_train, y_train)
            gb.fit(X_train, y_train)
            pred = 0.6 * gb.predict(future_df.values) + 0.4 * rf.predict(future_df.values)
            return {"forecast": np.clip(pred, 0, None), "models_used": ["RF", "GradientBoosting"]}
        except Exception as e:
            logger.info(f"ML forecast skipped, using MA fallback: {e}")
            return self.forecast_moving_average(ts, horizon)

    def run_all(
        self,
        orders_df: Optional[pd.DataFrame] = None,
        order_items_df: Optional[pd.DataFrame] = None,
        products_df: Optional[pd.DataFrame] = None,
        staging_dir: Optional[Path] = None,
        processed_dir: Optional[Path] = None,
        horizon: int = 30,
        top_n_products: int = 20,
        save: bool = True,
    ) -> Dict:
        """Forecast overall + top-N products + category totals, compare all models, write CSVs."""
        self.data_dir = Path(staging_dir) if staging_dir else self.data_dir
        self.processed_dir = Path(processed_dir) if processed_dir else self.processed_dir
        settings.ensure_dirs()

        if orders_df is None or order_items_df is None:
            fe = type("X", (), {})()
            from src.transformation.feature_engineering import FeatureEngineer
            f = FeatureEngineer(self.data_dir)
            dfs = f.load_all()
            orders_df = dfs.get("orders")
            order_items_df = dfs.get("order_items")
            products_df = dfs.get("products")
            if orders_df is None or order_items_df is None:
                raise FileNotFoundError("orders and order_items CSVs must exist in data dir")

        overall_ts = self._build_ts_from_orders(orders_df, order_items_df, granularity="D")
        train, test = self._split(overall_ts, min(horizon, max(horizon // 3, 7)))
        test_horizon = len(test)

        model_methods = {
            "Moving Average": lambda ts_, h_: self.forecast_moving_average(ts_, h_),
            "Exponential Smoothing": lambda ts_, h_: self.forecast_exponential_smoothing(ts_, h_),
            "Seasonal Naive": lambda ts_, h_: self.forecast_snaive(ts_, h_),
            "ARIMA": lambda ts_, h_: self.forecast_arima(ts_, h_),
            "ML Ensemble": lambda ts_, h_: self.forecast_ml_ensemble(ts_, h_),
        }
        model_compare_rows: List[Dict] = []
        overall_forecast_rows: List[Dict] = []
        model_forecasts: Dict[str, np.ndarray] = {}
        
        # Use time series cross-validation for model comparison
        cv_n_splits = min(5, len(overall_ts) // (horizon * 2))
        if cv_n_splits >= 2:
            logger.info(f"Using time series cross-validation with {cv_n_splits} splits")
        else:
            logger.info("Using simple train/test split (insufficient data for CV)")
        
        for name, method in model_methods.items():
            # Use CV if enough data, otherwise use simple split
            if cv_n_splits >= 2:
                cv_results = self.time_series_cross_validation(overall_ts, method, n_splits=cv_n_splits, horizon=horizon)
                metrics = {
                    "model": name,
                    "mae": cv_results["mae_mean"],
                    "rmse": cv_results["rmse_mean"],
                    "mape_pct": cv_results["mape_mean"],
                    "mae_std": cv_results["mae_std"],
                    "rmse_std": cv_results["rmse_std"],
                    "mape_std": cv_results["mape_std"],
                    "cv_splits": cv_results["n_splits"]
                }
            else:
                # Fallback to simple split
                res = method(train, test_horizon or 1)
                fv = res["forecast"]
                model_forecasts[name] = fv
                if test_horizon > 0:
                    y_true = test["units_sold"].values[:test_horizon]
                    y_pred = fv[:test_horizon]
                    metrics = {
                        "model": name,
                        "mae": self.mae(y_true, y_pred),
                        "rmse": self.rmse(y_true, y_pred),
                        "mape_pct": self.mape(y_true, y_pred),
                        "mae_std": None,
                        "rmse_std": None,
                        "mape_std": None,
                        "cv_splits": 1
                    }
                else:
                    metrics = {
                        "model": name,
                        "mae": None,
                        "rmse": None,
                        "mape_pct": None,
                        "mae_std": None,
                        "rmse_std": None,
                        "mape_std": None,
                        "cv_splits": 1
                    }
            model_compare_rows.append(metrics)
            
            # Generate future forecast
            future_res = method(overall_ts, horizon)
            future_fv = future_res["forecast"]
            future_dates = pd.date_range(overall_ts["date"].max() + pd.Timedelta(days=1),
                                         periods=horizon, freq="D")
            for d, v in zip(future_dates, future_fv):
                overall_forecast_rows.append({"date": d, "model": name, "forecast_units": float(max(0, v))})

        compare_df = pd.DataFrame(model_compare_rows)
        overall_future_df = pd.DataFrame(overall_forecast_rows)

        best_model = compare_df.loc[compare_df["mape_pct"].idxmin(), "model"] if len(compare_df) else "Moving Average"
        best_model_forecast = overall_future_df[overall_future_df["model"] == best_model].copy()

        key_prod = [c for c in ["product_id"] if c in order_items_df.columns][0]
        top_products = (order_items_df.groupby(key_prod)["quantity"].sum()
                                         .sort_values(ascending=False)
                                         .head(top_n_products)
                                         .index.tolist())
        product_forecast_rows: List[Dict] = []
        for pid in top_products:
            try:
                p_ts = self._build_ts_from_orders(orders_df, order_items_df, granularity="D", product_id=pid)
            except Exception:
                continue
            if len(p_ts) < 14:
                continue
            best_res = model_methods[best_model](p_ts, horizon)
            fv = best_res["forecast"]
            future_dates = pd.date_range(p_ts["date"].max() + pd.Timedelta(days=1), periods=horizon, freq="D")
            for d, v in zip(future_dates, fv):
                product_forecast_rows.append({"product_id": pid, "date": d, "forecast_units": float(max(0, v))})

        product_future_df = pd.DataFrame(product_forecast_rows)
        if products_df is not None and len(product_future_df) and key_prod in products_df.columns:
            product_future_df = product_future_df.merge(
                products_df[[key_prod, "product_name", "category_id"]], on=key_prod, how="left"
            )

        outputs: Dict[str, pd.DataFrame] = {
            "forecast_overall_history": overall_ts,
            "forecast_model_comparison": compare_df,
            "forecast_overall_future": overall_future_df,
            "forecast_overall_best": best_model_forecast,
            "forecast_top_products_future": product_future_df,
        }
        if save:
            for name, df in outputs.items():
                if df is not None and len(df) > 0:
                    out = self.processed_dir / f"{name}.csv"
                    df.to_csv(out, index=False)
                    logger.info(f"Saved {name} -> {out}")

        return {
            "best_model": best_model,
            "model_comparison": compare_df.to_dict(orient="records") if len(compare_df) else [],
            "horizon_days": horizon,
            "top_products_count": len(top_products),
            "outputs": {k: v.shape for k in outputs.keys()} if outputs else {},
            "rows": {k: len(v) for k, v in outputs.items() if v is not None},
        }
