from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import (
    norm, skew, kurtosis, zscore, pearsonr,
    f_oneway, chi2_contingency, mannwhitneyu, kruskal,
)
from datetime import datetime

from config.logging_config import get_logger

logger = get_logger(__name__)


class StatisticalAnalyzer:
    def __init__(self, significance_level: float = 0.05):
        self.alpha = significance_level
        self._reports: Dict = {}

    @staticmethod
    def descriptive_statistics(series: pd.Series) -> Dict:
        s = series.dropna()
        if len(s) == 0:
            return {"count": 0}

        numeric = pd.api.types.is_numeric_dtype(s)
        if not numeric:
            vc = s.value_counts()
            return {
                "count": int(len(s)),
                "unique": int(s.nunique()),
                "top": vc.index[0] if len(vc) > 0 else None,
                "top_freq": int(vc.iloc[0]) if len(vc) > 0 else 0,
            }

        return {
            "count": int(len(s)),
            "mean": float(s.mean()),
            "median": float(s.median()),
            "std": float(s.std()) if len(s) > 1 else 0.0,
            "var": float(s.var()) if len(s) > 1 else 0.0,
            "min": float(s.min()),
            "max": float(s.max()),
            "q25": float(s.quantile(0.25)),
            "q75": float(s.quantile(0.75)),
            "iqr": float(s.quantile(0.75) - s.quantile(0.25)),
            "skewness": float(skew(s)) if len(s) > 2 else 0.0,
            "kurtosis": float(kurtosis(s)) if len(s) > 3 else 0.0,
            "cv_pct": float(s.std() / s.mean() * 100) if s.mean() != 0 and len(s) > 1 else None,
        }

    @staticmethod
    def detect_outliers(
        series: pd.Series,
        method: str = "iqr",
        threshold: Optional[float] = None,
    ) -> Tuple[pd.Series, pd.Series]:
        s = series.dropna()
        if len(s) < 4:
            return pd.Series(dtype=bool), s

        if method == "iqr":
            q1 = s.quantile(0.25)
            q3 = s.quantile(0.75)
            iqr = q3 - q1
            k = threshold or 1.5
            lower = q1 - k * iqr
            upper = q3 + k * iqr
            mask = ~((s < lower) | (s > upper))
            outlier_mask = (s < lower) | (s > upper)
        elif method == "zscore":
            z = np.abs(zscore(s))
            t = threshold or 3.0
            outlier_mask = pd.Series(z > t, index=s.index, dtype=bool)
        elif method == "mad":
            median = s.median()
            mad = np.median(np.abs(s - median))
            if mad == 0:
                mad = 1e-9
            mod_z = 0.6745 * (s - median) / mad
            t = threshold or 3.5
            outlier_mask = np.abs(mod_z) > t
        else:
            raise ValueError(f"Unknown method: {method}")

        return outlier_mask, s

    @staticmethod
    def distribution_test(series: pd.Series, test: str = "normaltest") -> Dict:
        s = series.dropna()
        if len(s) < 8:
            return {"p_value": None, "statistic": None, "is_normal": None, "n": len(s)}

        if test == "normaltest":
            stat, p = stats.normaltest(s)
        elif test == "shapiro":
            n = min(len(s), 5000)
            stat, p = stats.shapiro(s.sample(n=n, random_state=42))
        elif test == "ks_normal":
            stat, p = stats.kstest(s, "norm", args=(s.mean(), s.std()))
        else:
            raise ValueError(f"Unknown test: {test}")

        return {
            "test": test,
            "statistic": float(stat),
            "p_value": float(p),
            "is_normal": bool(p > 0.05),
            "n": int(len(s)),
        }

    @staticmethod
    def correlation_analysis(
        df: pd.DataFrame,
        method: str = "pearson",
        numeric_only: bool = True,
    ) -> pd.DataFrame:
        if numeric_only:
            df_num = df.select_dtypes(include=[np.number])
        else:
            df_num = df

        if method in ["pearson", "spearman", "kendall"]:
            corr = df_num.corr(method=method)
        else:
            raise ValueError(f"Unknown correlation method: {method}")
        return corr

    @staticmethod
    def test_group_differences(
        values: pd.Series,
        groups: pd.Series,
        test: str = "auto",
    ) -> Dict:
        df = pd.DataFrame({"value": values, "group": groups}).dropna()
        group_names = df["group"].unique()

        if len(group_names) < 2:
            return {"test": test, "p_value": None, "significant": None, "groups": len(group_names)}

        group_data = [df[df["group"] == g]["value"].values for g in group_names]

        if test == "auto":
            all_normal = all(
                len(d) >= 8 and stats.normaltest(d).pvalue > 0.05
                for d in group_data if len(d) >= 8
            )
            if len(group_names) == 2:
                test = "t_test" if all_normal else "mannwhitneyu"
            else:
                test = "anova" if all_normal else "kruskal"

        try:
            if test == "t_test":
                stat, p = stats.ttest_ind(*group_data[:2], equal_var=False)
            elif test == "mannwhitneyu":
                stat, p = mannwhitneyu(*group_data[:2], alternative="two-sided")
            elif test == "anova":
                stat, p = f_oneway(*group_data)
            elif test == "kruskal":
                stat, p = kruskal(*group_data)
            else:
                raise ValueError(f"Unknown test: {test}")
        except Exception as e:
            logger.warning(f"Test {test} failed: {e}")
            return {"test": test, "p_value": None, "statistic": None, "significant": None}

        return {
            "test": test,
            "statistic": float(stat),
            "p_value": float(p),
            "significant": bool(p < 0.05),
            "groups": int(len(group_names)),
        }

    @staticmethod
    def chi_square_test(series1: pd.Series, series2: pd.Series) -> Dict:
        df = pd.DataFrame({"s1": series1, "s2": series2}).dropna()
        if len(df) == 0:
            return {"p_value": None, "significant": None, "n": 0}
        ct = pd.crosstab(df["s1"], df["s2"])
        if ct.shape[0] < 2 or ct.shape[1] < 2:
            return {"p_value": None, "significant": None, "n": len(df)}
        chi2, p, dof, expected = chi2_contingency(ct)
        return {
            "chi2": float(chi2),
            "p_value": float(p),
            "dof": int(dof),
            "significant": bool(p < 0.05),
            "n": int(len(df)),
        }

    @staticmethod
    def seasonality_decompose(
        time_series: pd.Series,
        period: Optional[int] = None,
        model: str = "additive",
    ) -> Optional[Dict[str, pd.Series]]:
        try:
            from statsmodels.tsa.seasonal import seasonal_decompose
        except ImportError:
            logger.warning("statsmodels not available for seasonal decomposition")
            return None

        if not isinstance(time_series.index, pd.DatetimeIndex):
            return None

        s = time_series.dropna().sort_index()
        if len(s) < 2 * (period or 7):
            return None

        p = period or 7
        try:
            result = seasonal_decompose(s, model=model, period=p, extrapolate_trend="freq")
            return {
                "trend": result.trend,
                "seasonal": result.seasonal,
                "residual": result.resid,
                "observed": result.observed,
            }
        except Exception as e:
            logger.warning(f"Seasonal decompose failed: {e}")
            return None

    def analyze_dataset(self, df: pd.DataFrame, name: str = "dataset") -> Dict:
        logger.info(f"Running statistical analysis on {name}: {len(df):,} rows x {len(df.columns)} cols")

        report = {"name": name, "shape": {"rows": int(len(df)), "cols": int(len(df.columns))}}

        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        desc_stats = {}
        for col in numeric_cols:
            desc_stats[col] = self.descriptive_statistics(df[col])
        report["descriptive"] = desc_stats

        outlier_report = {}
        for col in numeric_cols:
            mask, _ = self.detect_outliers(df[col], method="iqr")
            outlier_report[col] = {
                "count": int(mask.sum()),
                "pct": round(float(mask.sum() / len(mask) * 100), 2) if len(mask) > 0 else 0.0,
            }
        report["outliers"] = outlier_report

        if len(numeric_cols) >= 2:
            report["correlation_pearson"] = self.correlation_analysis(
                df, method="pearson"
            ).round(4).to_dict()

        if len(numeric_cols) >= 1:
            dist_tests = {}
            for col in numeric_cols[: min(len(numeric_cols), 10)]:
                dist_tests[col] = self.distribution_test(df[col], test="normaltest")
            report["distribution_tests"] = dist_tests

        self._reports[name] = report
        logger.info(f"Statistical analysis complete for {name}")
        return report
