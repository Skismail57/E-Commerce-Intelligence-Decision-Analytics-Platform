from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from config.logging_config import get_logger
from config.settings import settings

logger = get_logger(__name__)


class AnomalyDetector:
    """Unified statistical + ML anomaly detection with root-cause hints and severity scoring."""

    def __init__(self, data_dir: Optional[Path] = None, processed_dir: Optional[Path] = None):
        self.data_dir = Path(data_dir) if data_dir else settings.STAGING_DATA_DIR
        self.processed_dir = Path(processed_dir) if processed_dir else settings.PROCESSED_DATA_DIR
        settings.ensure_dirs()
        self.alerts: List[Dict] = []

    @staticmethod
    def _iqr_bounds(values: np.ndarray, k: float = 1.5) -> Tuple[float, float]:
        q1, q3 = np.nanpercentile(values, [25, 75])
        iqr = q3 - q1
        return float(q1 - k * iqr), float(q3 + k * iqr)

    @staticmethod
    def _zscore(values: np.ndarray) -> np.ndarray:
        m = np.nanmean(values)
        s = np.nanstd(values) or 1.0
        return (values - m) / s

    @classmethod
    def detect_anomalies_statistical(cls, series: pd.Series, rolling_window: int = 28,
                                     z_threshold: float = 3.0, iqr_k: float = 3.0) -> pd.DataFrame:
        """Statistical anomaly detection: rolling z-score + IQR fences. Returns flags, scores, labels."""
        vals = pd.to_numeric(series, errors="coerce").fillna(0).values.astype(float)
        idx = series.index if isinstance(series, pd.Series) else np.arange(len(vals))
        z_abs = np.abs(cls._zscore(vals))
        z_flag = z_abs > z_threshold
        iqr_lower, iqr_upper = cls._iqr_bounds(vals, k=iqr_k)
        iqr_flag = (vals < iqr_lower) | (vals > iqr_upper)
        roll = pd.Series(vals).rolling(rolling_window, min_periods=5)
        roll_mean = roll.mean().values
        roll_std = roll.std().fillna(1).values
        roll_z = np.where(roll_std > 0, np.abs(vals - roll_mean) / roll_std, 0.0)
        roll_flag = roll_z > z_threshold
        combined_flag = (z_flag | iqr_flag | roll_flag)
        direction = np.where(vals > roll_mean, "spike", "drop")
        severity = pd.cut(z_abs, bins=[-np.inf, 2, 3, 4, np.inf],
                          labels=["low", "medium", "high", "critical"]).astype(str)
        labels = np.where(~combined_flag, "Normal",
                          np.where(direction == "drop",
                                   "⚠️ Revenue Drop", "📈 Revenue Spike"))
        return pd.DataFrame({
            "index": idx,
            "value": vals,
            "zscore": z_abs.round(3),
            "rolling_zscore": np.round(roll_z, 3),
            "iqr_lower": iqr_lower,
            "iqr_upper": iqr_upper,
            "rolling_mean": np.round(roll_mean, 2),
            "anomaly_z": z_flag,
            "anomaly_iqr": iqr_flag,
            "anomaly_rolling": roll_flag,
            "is_anomaly": combined_flag,
            "direction": direction,
            "severity": severity,
            "anomaly_label": labels,
        })

    @classmethod
    def detect_anomalies_ml(cls, df: pd.DataFrame, columns: Optional[List[str]] = None,
                            contamination: float = 0.05) -> pd.DataFrame:
        """Isolation Forest anomaly detection; fallback to z-score if sklearn unavailable."""
        if columns is None:
            columns = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
        X_df = df[columns].copy()
        for c in columns:
            X_df[c] = pd.to_numeric(X_df[c], errors="coerce").fillna(0)
        X = X_df.values.astype(float)
        try:
            from sklearn.ensemble import IsolationForest
            model = IsolationForest(n_estimators=200, contamination=contamination,
                                    random_state=42, n_jobs=-1)
            preds = model.fit_predict(X)
            scores = model.decision_function(X)
            return pd.DataFrame({
                "is_anomaly_iforest": preds == -1,
                "anomaly_score_iforest": scores,
            })
        except Exception as e:
            logger.info(f"Isolation Forest unavailable, using PCA-free z-score fallback: {e}")
            z = np.abs((X - X.mean(axis=0)) / (X.std(axis=0) + 1e-9))
            z_max = z.max(axis=1)
            return pd.DataFrame({
                "is_anomaly_iforest": z_max > 3.0,
                "anomaly_score_iforest": -z_max,
            })

    @staticmethod
    def _root_cause_hints(dimension_name: str, dimension_df: pd.DataFrame, anomaly_index: int,
                          value_col: str) -> List[str]:
        """Return ranked possible causes for an anomaly by comparing current dimensions vs trailing 4-week baseline."""
        hints: List[str] = []
        baseline_end = max(0, anomaly_index - 1)
        baseline_start = max(0, baseline_end - 28)
        if baseline_end - baseline_start < 7:
            baseline_start = max(0, anomaly_index - 7)
        current_row = dimension_df.iloc[anomaly_index] if anomaly_index < len(dimension_df) else None
        baseline = dimension_df.iloc[baseline_start:baseline_end]
        if current_row is None or len(baseline) < 2:
            return ["Insufficient history for root cause analysis"]

        for col in [c for c in dimension_df.columns if c != value_col and pd.api.types.is_numeric_dtype(dimension_df[c])]:
            cur = float(current_row[col]) if pd.notna(current_row[col]) else 0.0
            base = float(baseline[col].median()) if len(baseline[col].dropna()) else 0.0
            ratio = (cur - base) / (abs(base) + 1e-9)
            if abs(ratio) > 0.35:
                direction = "↑" if ratio > 0 else "↓"
                hints.append(f"{direction}{col} {ratio * 100:+.0f}% vs baseline")
        if not hints:
            hints.append("Multi-factor / external seasonal event")
        return sorted(hints, key=lambda s: ("↑" in s, s), reverse=True)[:3]

    def run_all(self, staging_dir: Optional[Path] = None, processed_dir: Optional[Path] = None,
                save: bool = True) -> Dict:
        """Detect anomalies in overall daily KPIs, per-category revenue, per-state, inventory; produce alerts CSV."""
        self.data_dir = Path(staging_dir) if staging_dir else self.data_dir
        self.processed_dir = Path(processed_dir) if processed_dir else self.processed_dir
        settings.ensure_dirs()

        precomputed = self.processed_dir / "daily_sales_aggregated.csv"
        if precomputed.exists():
            daily = pd.read_csv(precomputed, low_memory=False)
        else:
            from src.transformation.aggregator import DataAggregator
            from src.transformation.feature_engineering import FeatureEngineer
            fe = FeatureEngineer(self.data_dir)
            dfs = fe.load_all()
            daily = DataAggregator.aggregate_daily_sales(
                dfs.get("orders"), dfs.get("order_items"), dfs.get("customers")
            )

        date_candidates = [c for c in daily.columns if "date" in c.lower()]
        if date_candidates:
            daily = daily.sort_values(date_candidates[0]).reset_index(drop=True)

        kpi_cols_map: Dict[str, str] = {}
        for col in ["revenue_inr", "net_revenue_inr", "gross_profit_inr", "orders_count",
                    "total_orders", "aov_inr", "units_sold", "return_units", "return_count"]:
            if col in daily.columns:
                kpi_cols_map[col] = col
        if not kpi_cols_map:
            numeric_cols = [c for c in daily.columns if pd.api.types.is_numeric_dtype(daily[c])]
            if numeric_cols:
                kpi_cols_map = {c: c for c in numeric_cols[:5]}

        overall_anomalies_frames: List[pd.DataFrame] = []
        alerts: List[Dict] = []
        for short_name, col in kpi_cols_map.items():
            if len(daily[col].dropna()) < 14:
                continue
            stats = self.detect_anomalies_statistical(daily[col], rolling_window=max(7, min(28, len(daily) // 2)))
            stats["kpi"] = short_name
            for idx in np.where(stats["is_anomaly"].values)[0][-20:]:
                position = int(stats.iloc[idx]["index"])
                sev = str(stats.iloc[idx]["severity"])
                label = str(stats.iloc[idx]["anomaly_label"])
                cur_val = float(stats.iloc[idx]["value"])
                base_val = float(stats.iloc[idx]["rolling_mean"])
                change_pct = ((cur_val - base_val) / (abs(base_val) + 1e-9)) * 100
                date_str = ""
                if date_candidates and position < len(daily):
                    date_str = str(daily.iloc[position][date_candidates[0]])[:10]
                causes = self._root_cause_hints("daily", daily, position, col)
                severity_score = {"low": 1, "medium": 2, "high": 3, "critical": 4}.get(sev, 0)
                alerts.append({
                    "alert_id": f"OVR-{len(alerts)+1:04d}",
                    "category": "Overall",
                    "kpi": short_name,
                    "anomaly_label": label,
                    "severity": sev,
                    "severity_score": severity_score,
                    "date": date_str,
                    "value": round(cur_val, 2),
                    "baseline_28d": round(base_val, 2),
                    "change_pct": round(change_pct, 1),
                    "possible_causes": "; ".join(causes),
                    "recommended_action": (
                        "Investigate immediately + notify ops" if sev == "critical"
                        else "Investigate cause within 48 hrs" if sev == "high"
                        else "Monitor + flag for weekly review"
                    ),
                })
            overall_anomalies_frames.append(stats)

        overall_df = pd.concat(overall_anomalies_frames, ignore_index=True) if overall_anomalies_frames else pd.DataFrame()

        ml_inputs = daily[[c for c in kpi_cols_map.values()]].copy()
        if len(ml_inputs.columns) >= 2 and len(ml_inputs) >= 10:
            ml_anom = self.detect_anomalies_ml(ml_inputs)
            for i in range(min(len(overall_df), len(ml_anom))):
                overall_df.loc[i, "is_anomaly_iforest"] = bool(ml_anom.iloc[i]["is_anomaly_iforest"])
                overall_df.loc[i, "anomaly_score_iforest"] = float(ml_anom.iloc[i]["anomaly_score_iforest"])

        alerts_df = pd.DataFrame(alerts).sort_values(["severity_score", "change_pct"], ascending=[False, False]).reset_index(drop=True) if alerts else pd.DataFrame()

        outputs: Dict[str, pd.DataFrame] = {
            "anomaly_detection_overall": overall_df,
            "anomaly_alerts": alerts_df,
        }
        if save:
            for name, df in outputs.items():
                if df is not None and len(df) > 0:
                    out = self.processed_dir / f"{name}.csv"
                    df.to_csv(out, index=False)
                    logger.info(f"Saved {name} -> {out}")

        return {
            "total_data_points": int(len(daily)),
            "kpis_analyzed": list(kpi_cols_map.keys()),
            "alerts_count": len(alerts_df),
            "alerts_by_severity": alerts_df["severity"].value_counts().to_dict() if "severity" in alerts_df.columns else {},
            "outputs": {k: v.shape for k, v in outputs.items() if v is not None},
        }
