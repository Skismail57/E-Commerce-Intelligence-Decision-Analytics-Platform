"""
Customer Lifetime Value (CLV) Prediction
Implements BG/NBD and Gamma-Gamma models for probabilistic CLV estimation.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

from config.logging_config import get_logger

logger = get_logger(__name__)


class CLVPredictor:
    """
    Customer Lifetime Value predictor using BG/NBD and Gamma-Gamma models.
    
    BG/NBD (Beta-Geometric/Negative Binomial Distribution):
    - Models repeat purchase behavior
    - Predicts transaction rate and dropout probability
    
    Gamma-Gamma:
    - Models monetary value per transaction
    - Assumes transaction value varies around customer mean
    """
    
    def __init__(self, data_dir: Optional[Path] = None):
        self.data_dir = Path(data_dir) if data_dir else None
        self._dfs: Dict[str, pd.DataFrame] = {}
        self.bg_nbd_params: Optional[Dict[str, float]] = None
        self.gamma_gamma_params: Optional[Dict[str, float]] = None
        self.clv_predictions: Optional[pd.DataFrame] = None
    
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
        """Load all required tables."""
        directory = data_dir or self.data_dir
        tables = [
            "customers", "orders", "order_items",
        ]
        for t in tables:
            try:
                self._dfs[t] = self._read_csv(t, directory)
            except FileNotFoundError:
                logger.warning(f"Table {t} not found, skipping")
        logger.info(f"Loaded {len(self._dfs)} tables")
        return self._dfs
    
    def compute_rfm_data(
        self,
        observation_end: Optional[datetime] = None
    ) -> pd.DataFrame:
        """
        Compute RFM (Recency, Frequency, Monetary) data for BG/NBD model.
        
        Args:
            observation_end: End date for observation period
        
        Returns:
            DataFrame with columns: customer_id, frequency, recency, T, monetary_value
        """
        if "orders" not in self._dfs:
            raise ValueError("Orders data required")
        
        orders_df = self._dfs["orders"]
        valid_orders = orders_df[orders_df["order_status"].isin(['Delivered', 'Shipped', 'Processing', 'Returned'])]
        
        observation_end = observation_end or datetime(2024, 12, 31)
        
        # Calculate RFM metrics
        customer_orders = valid_orders.groupby("customer_id").agg({
            "order_date": ["min", "max", "count"],
            "order_total": "sum"
        }).reset_index()
        customer_orders.columns = ["customer_id", "first_purchase", "last_purchase", "frequency", "monetary_value"]
        
        # Calculate recency (days since last purchase)
        customer_orders["recency"] = (
            (observation_end - pd.to_datetime(customer_orders["last_purchase"])).dt.days
        )
        
        # Calculate T (days since first purchase)
        customer_orders["T"] = (
            (observation_end - pd.to_datetime(customer_orders["first_purchase"])).dt.days
        )
        
        # Filter customers with at least one purchase
        customer_orders = customer_orders[customer_orders["frequency"] > 0]
        
        # Calculate average monetary value per transaction
        customer_orders["monetary_value"] = customer_orders["monetary_value"] / customer_orders["frequency"]
        
        return customer_orders[["customer_id", "frequency", "recency", "T", "monetary_value"]]
    
    def fit_bg_nbd(
        self,
        rfm_data: pd.DataFrame,
        max_iter: int = 1000,
        tol: float = 1e-6
    ) -> Dict[str, float]:
        """
        Fit BG/NBD model using maximum likelihood estimation.
        
        Args:
            rfm_data: DataFrame with frequency, recency, T columns
            max_iter: Maximum iterations for optimization
            tol: Convergence tolerance
        
        Returns:
            Dictionary with fitted parameters (r, alpha, a, b)
        """
        from scipy.optimize import minimize
        
        def negative_log_likelihood(params, frequency, recency, T):
            r, alpha, a, b = params
            
            # Ensure parameters are positive
            if np.any(params <= 0):
                return np.inf
            
            # Log-likelihood for BG/NBD model
            # Simplified implementation based on Fader et al. (2005)
            
            # A1 calculation
            A1 = np.log(r + frequency) - r * np.log(alpha) - frequency * np.log(alpha + T)
            
            # A2 calculation
            A2 = np.log(a) + np.log(b) + (r + frequency) * np.log(alpha + T) - (r + frequency) * np.log(alpha + recency + T)
            A2 -= np.log(a + b + frequency) + (r + frequency) * np.log(alpha + recency + T)
            
            # Log-likelihood
            ll = A1 + A2
            
            return -np.sum(ll)
        
        frequency = rfm_data["frequency"].values
        recency = rfm_data["recency"].values
        T = rfm_data["T"].values
        
        # Initial parameter guesses
        initial_params = [0.5, 10.0, 0.5, 10.0]
        
        # Optimize
        result = minimize(
            negative_log_likelihood,
            initial_params,
            args=(frequency, recency, T),
            bounds=[(1e-6, None), (1e-6, None), (1e-6, None), (1e-6, None)],
            options={'maxiter': max_iter, 'disp': False}
        )
        
        if result.success:
            self.bg_nbd_params = {
                'r': result.x[0],
                'alpha': result.x[1],
                'a': result.x[2],
                'b': result.x[3]
            }
            logger.info(f"BG/NBD parameters fitted: {self.bg_nbd_params}")
        else:
            logger.warning(f"BG/NBD optimization failed: {result.message}")
            # Use default parameters
            self.bg_nbd_params = {'r': 0.5, 'alpha': 10.0, 'a': 0.5, 'b': 10.0}
        
        return self.bg_nbd_params
    
    def fit_gamma_gamma(
        self,
        rfm_data: pd.DataFrame,
        max_iter: int = 1000,
        tol: float = 1e-6
    ) -> Dict[str, float]:
        """
        Fit Gamma-Gamma model for monetary value.
        
        Args:
            rfm_data: DataFrame with frequency and monetary_value columns
            max_iter: Maximum iterations for optimization
            tol: Convergence tolerance
        
        Returns:
            Dictionary with fitted parameters (p, q, gamma)
        """
        from scipy.optimize import minimize
        
        def negative_log_likelihood(params, frequency, monetary_value):
            p, q, gamma = params
            
            # Ensure parameters are positive
            if np.any(params <= 0):
                return np.inf
            
            # Log-likelihood for Gamma-Gamma model
            # Simplified implementation
            
            # Calculate expected monetary value
            x = monetary_value if isinstance(monetary_value, np.ndarray) else monetary_value.values
            n = frequency if isinstance(frequency, np.ndarray) else frequency.values
            
            # Log-likelihood components
            ll = (p * np.log(q) + (p + n) * np.log(gamma + n * x) - 
                  (p + n) * np.log(q + n * x) - 
                  np.log(gamma) - np.sum(np.log(x)))
            
            return -np.sum(ll)
        
        frequency = rfm_data["frequency"].values
        monetary_value = rfm_data["monetary_value"].values
        
        # Initial parameter guesses
        initial_params = [2.0, 3.0, 100.0]
        
        # Optimize
        result = minimize(
            negative_log_likelihood,
            initial_params,
            args=(frequency, monetary_value),
            bounds=[(1e-6, None), (1e-6, None), (1e-6, None)],
            options={'maxiter': max_iter, 'disp': False}
        )
        
        if result.success:
            self.gamma_gamma_params = {
                'p': result.x[0],
                'q': result.x[1],
                'gamma': result.x[2]
            }
            logger.info(f"Gamma-Gamma parameters fitted: {self.gamma_gamma_params}")
        else:
            logger.warning(f"Gamma-Gamma optimization failed: {result.message}")
            # Use default parameters
            self.gamma_gamma_params = {'p': 2.0, 'q': 3.0, 'gamma': 100.0}
        
        return self.gamma_gamma_params
    
    def predict_p_alive(
        self,
        rfm_data: pd.DataFrame,
        prediction_period: int = 90
    ) -> pd.Series:
        """
        Predict probability that a customer is still alive.
        
        Args:
            rfm_data: DataFrame with frequency, recency, T columns
            prediction_period: Number of days to predict into future
        
        Returns:
            Series with p_alive values
        """
        if self.bg_nbd_params is None:
            raise ValueError("BG/NBD model not fitted")
        
        r = self.bg_nbd_params['r']
        alpha = self.bg_nbd_params['alpha']
        a = self.bg_nbd_params['a']
        b = self.bg_nbd_params['b']
        
        frequency = rfm_data["frequency"].values
        recency = rfm_data["recency"].values
        T = rfm_data["T"].values
        
        # Calculate p_alive using BG/NBD formula
        # P(alive) = 1 / (1 + (a / (b + frequency)) * ((alpha + T) / (alpha + recency))^(r + frequency))
        
        numerator = (alpha + T) ** (r + frequency)
        denominator = (alpha + recency) ** (r + frequency)
        ratio = numerator / denominator
        
        p_alive = 1 / (1 + (a / (b + frequency)) * ratio)
        
        return pd.Series(p_alive, index=rfm_data["customer_id"])
    
    def predict_expected_transactions(
        self,
        rfm_data: pd.DataFrame,
        prediction_period: int = 90
    ) -> pd.Series:
        """
        Predict expected number of transactions in prediction period.
        
        Args:
            rfm_data: DataFrame with frequency, recency, T columns
            prediction_period: Number of days to predict into future
        
        Returns:
            Series with expected transaction counts
        """
        if self.bg_nbd_params is None:
            raise ValueError("BG/NBD model not fitted")
        
        r = self.bg_nbd_params['r']
        alpha = self.bg_nbd_params['alpha']
        a = self.bg_nbd_params['a']
        b = self.bg_nbd_params['b']
        
        frequency = rfm_data["frequency"].values
        recency = rfm_data["recency"].values
        T = rfm_data["T"].values
        
        # Calculate expected transactions using BG/NBD formula
        # E(Y(t)) = (a + b) / (a - 1) * (1 - ((alpha + T) / (alpha + T + t))^(r + frequency) * (b + frequency) / (a + b + frequency))
        
        t = prediction_period
        term1 = (a + b) / (a - 1) if a > 1 else 0
        
        term2_numerator = (alpha + T) ** (r + frequency)
        term2_denominator = (alpha + T + t) ** (r + frequency)
        term2 = term2_numerator / term2_denominator
        
        term3 = (b + frequency) / (a + b + frequency)
        
        expected_transactions = term1 * (1 - term2 * term3)
        
        return pd.Series(expected_transactions, index=rfm_data["customer_id"])
    
    def predict_clv(
        self,
        rfm_data: pd.DataFrame,
        prediction_period: int = 90,
        discount_rate: float = 0.01
    ) -> pd.DataFrame:
        """
        Predict Customer Lifetime Value.
        
        Args:
            rfm_data: DataFrame with frequency, recency, T, monetary_value columns
            prediction_period: Number of days to predict into future
            discount_rate: Monthly discount rate for present value calculation
        
        Returns:
            DataFrame with CLV predictions
        """
        if self.bg_nbd_params is None:
            self.fit_bg_nbd(rfm_data)
        if self.gamma_gamma_params is None:
            self.fit_gamma_gamma(rfm_data)
        
        # Predict expected transactions
        expected_transactions = self.predict_expected_transactions(rfm_data, prediction_period)
        
        # Predict expected monetary value using Gamma-Gamma
        p = self.gamma_gamma_params['p']
        q = self.gamma_gamma_params['q']
        gamma = self.gamma_gamma_params['gamma']
        
        frequency = rfm_data["frequency"].values
        monetary_value = rfm_data["monetary_value"].values
        
        # Expected monetary value
        expected_monetary = (p * gamma + q * frequency * monetary_value) / (p + frequency - 1)
        
        # Calculate CLV
        clv = expected_transactions.values * expected_monetary
        
        # Apply discounting
        months = prediction_period / 30
        discount_factor = 1 / (1 + discount_rate) ** months
        clv_discounted = clv * discount_factor
        
        # Create result DataFrame
        result = rfm_data[["customer_id"]].copy()
        result["expected_transactions"] = expected_transactions.values
        result["expected_monetary_value"] = expected_monetary
        result["clv"] = clv
        result["clv_discounted"] = clv_discounted
        
        # Create CLV tiers
        clv_percentiles = result["clv"].quantile([0, 0.25, 0.5, 0.75, 1.0]).values
        # Handle duplicate bin edges
        if len(np.unique(clv_percentiles)) < 5:
            # Fallback to simple tiering based on value ranges
            result["clv_tier"] = pd.cut(
                result["clv"],
                bins=[-np.inf, 1000, 5000, 10000, 50000, np.inf],
                labels=["Low", "Medium-Low", "Medium", "Medium-High", "High"],
                include_lowest=True
            )
        else:
            result["clv_tier"] = pd.cut(
                result["clv"],
                bins=clv_percentiles,
                labels=["Low", "Medium-Low", "Medium", "Medium-High", "High"],
                include_lowest=True,
                duplicates='drop'
            )
        
        self.clv_predictions = result
        return result
    
    def run_all(
        self,
        data_dir: Optional[Path] = None,
        processed_dir: Optional[Path] = None,
        save: bool = True
    ) -> Dict:
        """
        Run the full CLV prediction pipeline.
        
        Args:
            data_dir: Directory containing staging data
            processed_dir: Directory to save processed outputs
            save: Whether to save outputs
        
        Returns:
            Dictionary with pipeline results
        """
        self.data_dir = Path(data_dir) if data_dir else self.data_dir
        self.load_all(self.data_dir)
        
        # Compute RFM data
        rfm_data = self.compute_rfm_data()
        
        # Fit models and predict CLV
        clv_predictions = self.predict_clv(rfm_data)
        
        outputs = {
            "rfm_data": rfm_data,
            "clv_predictions": clv_predictions
        }
        
        if save:
            output_dir = Path(processed_dir) if processed_dir else self.data_dir.parent / "processed"
            output_dir.mkdir(exist_ok=True)
            
            for name, df in outputs.items():
                if df is not None and len(df) > 0:
                    out_path = output_dir / f"clv_{name}.csv"
                    df.to_csv(out_path, index=False)
                    logger.info(f"Saved {name} -> {out_path}")
        
        return {
            "status": "success",
            "bg_nbd_params": self.bg_nbd_params,
            "gamma_gamma_params": self.gamma_gamma_params,
            "outputs": {k: v.shape if v is not None else None for k, v in outputs.items()},
            "clv_distribution": clv_predictions["clv_tier"].value_counts().to_dict() if clv_predictions is not None else {}
        }
