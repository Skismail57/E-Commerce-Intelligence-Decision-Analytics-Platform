from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from config.logging_config import get_logger
from config.settings import settings
from src.transformation.feature_engineering import FeatureEngineer

logger = get_logger(__name__)


class ChurnPredictor:
    """Customer churn prediction: trains an ensemble classifier, outputs probability & risk tiers."""

    NUMERIC_FEATURES: List[str] = [
        "days_since_last_order", "total_orders", "avg_order_value",
        "total_spend", "return_rate", "discount_usage_pct",
        "sessions_count", "customer_tenure_days", "review_count", "total_units_bought",
    ]
    CATEGORICAL_FEATURES: List[str] = [
        "customer_segment", "gender", "state",
    ]
    LABEL_COL: str = "churn_label_90d"
    RISK_BINS: List[float] = [0.0, 0.2, 0.5, 1.0]
    RISK_LABELS: List[str] = ["Low", "Medium", "High"]

    def __init__(self, data_dir: Optional[Path] = None, processed_dir: Optional[Path] = None):
        self.data_dir = Path(data_dir) if data_dir else settings.STAGING_DATA_DIR
        self.processed_dir = Path(processed_dir) if processed_dir else settings.PROCESSED_DATA_DIR
        settings.ensure_dirs()
        self.model = None
        self.feature_importance: Optional[pd.DataFrame] = None
        self._metrics: Optional[Dict[str, float]] = None

    def _load_churn_features(self) -> pd.DataFrame:
        path = self.processed_dir / "customer_churn_features.csv"
        if path.exists():
            df = pd.read_csv(path, low_memory=False)
            if "date" in df.columns:
                pass
            logger.info(f"Loaded precomputed churn features from {path}")
            return df
        fe = FeatureEngineer(self.data_dir)
        fe.load_all()
        df = fe.compute_churn_features(
            fe._dfs.get("customers"),
            fe._dfs.get("orders"),
            fe._dfs.get("order_items"),
            fe._dfs.get("payments"),
            fe._dfs.get("reviews"),
            fe._dfs.get("website_sessions"),
        )
        return df

    @staticmethod
    def _prepare_features(churn_df: pd.DataFrame) -> Tuple[pd.DataFrame, np.ndarray, List[str]]:
        df = churn_df.copy()
        target_cols = [c for c in ["churn_label_90d", "churned", "is_churned"] if c in df.columns]
        label_col = target_cols[0] if target_cols else None

        use_cols = [c for c in ChurnPredictor.NUMERIC_FEATURES if c in df.columns]
        for col in use_cols:
            df[col] = pd.to_numeric(df[col], errors="coerce")
            df[col] = df[col].fillna(df[col].median() if df[col].notna().any() else 0)

        cat_cols = [c for c in ChurnPredictor.CATEGORICAL_FEATURES if c in df.columns]
        for col in cat_cols:
            df[col] = df[col].astype(str).fillna("UNKNOWN")

        feature_df = df[use_cols].copy()
        for col in cat_cols:
            feature_df = pd.concat([feature_df, pd.get_dummies(df[col], prefix=col, drop_first=True)], axis=1)

        feature_df = feature_df.replace([np.inf, -np.inf], np.nan).fillna(0)
        feature_names = list(feature_df.columns)

        y = np.array([])
        if label_col is not None:
            y = pd.to_numeric(df[label_col], errors="coerce").fillna(0).astype(int).values
        return feature_df, y, feature_names

    def train_model(
        self,
        churn_df: Optional[pd.DataFrame] = None,
        test_size: float = 0.25,
        random_state: int = 42,
        use_temporal_split: bool = True,
    ) -> Dict:
        """Train a RandomForest + LogReg soft-voting ensemble, or best-effort single classifier.
        
        Args:
            use_temporal_split: If True, uses time-based split (train on earlier data, test on later data).
                              If False, uses random stratified split.
        """
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                                     f1_score, roc_auc_score, classification_report)

        df = churn_df if churn_df is not None else self._load_churn_features()
        feature_df, y, feature_names = self._prepare_features(df)
        if len(y) == 0 or len(set(y)) < 2:
            logger.warning("Not enough label classes to train churn model; using heuristic scoring.")
            self.model = None
            self._compute_heuristic_importance(df, feature_names)
            self._metrics = {"accuracy": None, "auc_roc": None, "status": "heuristic_fallback"}
            return {"metrics": self._metrics, "n_samples": len(df), "status": "heuristic_fallback"}

        # Use temporal split if requested and we have date information
        if use_temporal_split and "customer_tenure_days" in df.columns:
            # Sort by customer_tenure_days (proxy for signup date)
            sorted_indices = df["customer_tenure_days"].argsort()
            split_idx = int(len(sorted_indices) * (1 - test_size))
            train_indices = sorted_indices[:split_idx]
            test_indices = sorted_indices[split_idx:]
            
            X_train = feature_df.values[train_indices]
            X_test = feature_df.values[test_indices]
            y_train = y[train_indices]
            y_test = y[test_indices]
            logger.info(f"Using temporal train/test split: {len(X_train)} train, {len(X_test)} test")
        else:
            # Fall back to random stratified split
            X_train, X_test, y_train, y_test = train_test_split(
                feature_df.values, y, test_size=test_size,
                random_state=random_state, stratify=y,
            )
            logger.info(f"Using random stratified train/test split")

        rf = RandomForestClassifier(n_estimators=150, max_depth=10, min_samples_leaf=10,
                                    class_weight="balanced", random_state=random_state, n_jobs=-1)
        lr = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=random_state)

        rf.fit(X_train, y_train)
        try:
            lr.fit(X_train, y_train)
        except Exception:
            lr = None

        rf_proba = rf.predict_proba(X_test)[:, 1]
        if lr is not None:
            lr_proba = lr.predict_proba(X_test)[:, 1]
            blend_proba = 0.7 * rf_proba + 0.3 * lr_proba
        else:
            blend_proba = rf_proba

        y_pred = (blend_proba >= 0.5).astype(int)
        self._metrics = {
            "accuracy": float(accuracy_score(y_test, y_pred)),
            "precision": float(precision_score(y_test, y_pred, zero_division=0)),
            "recall": float(recall_score(y_test, y_pred, zero_division=0)),
            "f1": float(f1_score(y_test, y_pred, zero_division=0)),
            "auc_roc": float(roc_auc_score(y_test, blend_proba)),
            "n_train": int(len(X_train)),
            "n_test": int(len(X_test)),
        }

        rf_imp = pd.DataFrame({"feature": feature_names, "importance": rf.feature_importances_})
        rf_imp = rf_imp.sort_values("importance", ascending=False).reset_index(drop=True)
        self.feature_importance = rf_imp
        self.model = {"rf": rf, "lr": lr, "feature_names": feature_names}
        return {
            "metrics": self._metrics,
            "feature_importance_top": rf_imp.head(10).to_dict(orient="records"),
            "n_samples": len(df),
            "status": "trained_ensemble",
        }

    def _compute_heuristic_importance(self, churn_df: pd.DataFrame, feature_names: List[str]) -> None:
        numeric = [c for c in feature_names if c in churn_df.columns]
        if not numeric:
            numeric = [c for c in self.NUMERIC_FEATURES if c in churn_df.columns]
        self.feature_importance = pd.DataFrame({
            "feature": numeric,
            "importance": np.linspace(1.0, 0.1, max(len(numeric), 1)) / max(len(numeric), 1),
        }).sort_values("importance", ascending=False).reset_index(drop=True)

    def predict(self, churn_df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """Backward-compatible alias for predict_proba."""
        return self.predict_proba(churn_df)

    def predict_proba(self, churn_df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """Return churn probabilities + risk tier + top-3 risk driver columns for each customer."""
        df = churn_df if churn_df is not None else self._load_churn_features()
        feature_df, y, feature_names = self._prepare_features(df)
        X = feature_df.values

        if self.model is None or self.model.get("rf") is None:
            proba = self._heuristic_score(df)
        else:
            rf = self.model["rf"]
            lr = self.model.get("lr")
            rf_p = rf.predict_proba(X)[:, 1]
            if lr is not None:
                try:
                    lr_p = lr.predict_proba(X)[:, 1]
                    proba = 0.7 * rf_p + 0.3 * lr_p
                except Exception:
                    proba = rf_p
            else:
                proba = rf_p

        result = df.copy()
        result["churn_probability"] = np.clip(proba, 0.0, 1.0)
        result["risk_tier"] = pd.cut(
            result["churn_probability"], bins=self.RISK_BINS, labels=self.RISK_LABELS, include_lowest=True
        )
        result["risk_score"] = (result["churn_probability"] * 1000).round().astype(int)
        result[["churn_probability", "risk_score", "risk_tier"]] = (
            result[["churn_probability", "risk_score", "risk_tier"]]
        )
        result = self._add_risk_drivers(result, feature_df)
        return result

    def _heuristic_score(self, df: pd.DataFrame) -> np.ndarray:
        """Heuristic churn score (no model needed): recency + frequency + returns + discount dependency."""
        cols_available = set(df.columns)
        components = []
        for col, positive_in_churn in [
            ("days_since_last_order", True),
            ("return_rate", True),
            ("discount_usage_pct", True),
        ]:
            if col in cols_available:
                series = pd.to_numeric(df[col], errors="coerce").fillna(0)
                norm = (series - series.min()) / (series.max() - series.min() + 1e-9)
                components.append(norm if positive_in_churn else 1 - norm)
        for col in ["total_orders", "total_spend", "average_order_value"]:
            if col in cols_available:
                series = pd.to_numeric(df[col], errors="coerce").fillna(0)
                norm = (series - series.min()) / (series.max() - series.min() + 1e-9)
                components.append(1 - norm)

        if not components:
            return np.full(len(df), 0.3)
        score = np.mean(np.column_stack(components), axis=1)
        return np.clip(score, 0.0, 1.0)

    def _add_risk_drivers(self, result: pd.DataFrame, feature_df: pd.DataFrame) -> pd.DataFrame:
        """For each customer, compute a per-feature risk delta vs low-risk reference (percentile 25)."""
        use = self.feature_importance
        if use is None or len(use) == 0:
            result["top_risk_drivers"] = ""
            return result
        top_feats = [f for f in use["feature"].head(6).tolist() if f in feature_df.columns]
        if not top_feats:
            result["top_risk_drivers"] = ""
            return result

        low_mask = result["risk_tier"] == "Low"
        if low_mask.sum() < 5:
            low_mask = result["churn_probability"] <= result["churn_probability"].quantile(0.3)
        ref = feature_df.loc[low_mask.values, top_feats].median()
        rows: List[str] = []
        for i in range(len(result)):
            deltas = ((feature_df.iloc[i][top_feats] - ref) / (ref.abs() + 1e-9)).fillna(0)
            sorted_d = deltas.sort_values(ascending=False)
            top = [(f, float(v)) for f, v in sorted_d.head(3).items() if v > 0.15]
            if not top:
                rows.append("No unusual drivers")
            else:
                rows.append("; ".join(f"{f} +{int(100*v)}%" for f, v in top))
        result["top_risk_drivers"] = rows
        return result

    def recommended_actions(self, churn_predictions: pd.DataFrame) -> pd.DataFrame:
        """Suggest a retention action per customer based on risk tier + CLV tier."""
        actions: List[str] = []
        for _, row in churn_predictions.iterrows():
            tier = row.get("risk_tier", "Low")
            clv = str(row.get("clv_tier", "Medium-Low")) if "clv_tier" in churn_predictions.columns else "Medium"
            if tier == "High":
                if clv in ("High", "Medium-High"):
                    actions.append("Priority: 1:1 outreach + exclusive loyalty discount (15%+ 3mo)")
                else:
                    actions.append("Send targeted win-back coupon (10% off) + personalization campaign")
            elif tier == "Medium":
                actions.append("Nurture: personalized product recommendations + limited-time free shipping")
            else:
                actions.append("Maintain: periodic newsletter with curated deals")
        churn_predictions = churn_predictions.copy()
        churn_predictions["recommended_action"] = actions
        return churn_predictions

    @property
    def metrics(self) -> Optional[Dict[str, float]]:
        return self._metrics

    def run_all(self, staging_dir: Optional[Path] = None, processed_dir: Optional[Path] = None, save: bool = True) -> Dict:
        self.data_dir = Path(staging_dir) if staging_dir else self.data_dir
        self.processed_dir = Path(processed_dir) if processed_dir else self.processed_dir
        settings.ensure_dirs()

        churn_df = self._load_churn_features()
        train_result = self.train_model(churn_df)
        preds = self.predict_proba(churn_df)
        preds_with_actions = self.recommended_actions(preds)

        outputs: Dict[str, pd.DataFrame] = {
            "churn_predictions": preds_with_actions,
            "churn_feature_importance": self.feature_importance.reset_index(drop=True) if self.feature_importance is not None else pd.DataFrame(),
        }
        if save:
            for name, df in outputs.items():
                if df is not None and len(df) > 0:
                    out = self.processed_dir / f"{name}.csv"
                    df.to_csv(out, index=False)
                    logger.info(f"Saved {name} -> {out}")

        return {
            "training": train_result,
            "outputs": {k: v.shape for k, v in outputs.items() if v is not None},
            "risk_distribution": preds_with_actions["risk_tier"].value_counts(dropna=False).to_dict() if "risk_tier" in preds_with_actions.columns else {},
        }
