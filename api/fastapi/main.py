"""
FastAPI Application for E-Commerce Intelligence & Decision Analytics Platform
Provides REST API endpoints for analytics, predictions, and business insights.
"""

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from config.settings import settings
from config.logging_config import get_logger

logger = get_logger(__name__)

app = FastAPI(
    title="E-Commerce Intelligence API",
    description="REST API for e-commerce analytics, predictions, and decision support",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Pydantic models for request/response
class CustomerChurnRequest(BaseModel):
    customer_id: int


class ForecastRequest(BaseModel):
    product_id: Optional[int] = None
    horizon_days: int = 30
    model_type: str = "moving_average"


class RecommendationRequest(BaseModel):
    customer_id: int
    num_recommendations: int = 5


class KPIResponse(BaseModel):
    total_revenue: float
    net_revenue: float
    total_orders: int
    unique_customers: int
    aov: float
    return_rate: float
    gross_margin_pct: float
    period_start: str
    period_end: str


class ChurnPredictionResponse(BaseModel):
    customer_id: int
    churn_probability: float
    risk_tier: str
    risk_score: int
    top_risk_drivers: str


class ForecastResponse(BaseModel):
    dates: List[str]
    forecast_values: List[float]
    lower_bound: List[float]
    upper_bound: List[float]
    model_type: str
    mae: Optional[float] = None
    rmse: Optional[float] = None
    mape_pct: Optional[float] = None


class RecommendationResponse(BaseModel):
    customer_id: int
    recommendations: List[Dict[str, Any]]
    strategy: str


# Helper functions to load data
def load_processed_data(filename: str) -> Optional[pd.DataFrame]:
    """Load processed data from the processed directory."""
    path = settings.PROCESSED_DATA_DIR / filename
    if path.exists():
        return pd.read_csv(path, low_memory=False)
    return None


def load_staging_data(filename: str) -> Optional[pd.DataFrame]:
    """Load staging data from the staging directory."""
    path = settings.STAGING_DATA_DIR / filename
    if path.exists():
        return pd.read_csv(path, low_memory=False)
    return None


# API Endpoints

@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "name": "E-Commerce Intelligence API",
        "version": "1.0.0",
        "status": "operational",
        "endpoints": {
            "kpi": "/api/v1/kpi",
            "churn_prediction": "/api/v1/churn/predict",
            "forecast": "/api/v1/forecast",
            "recommendations": "/api/v1/recommendations",
            "health": "/health"
        }
    }


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "data_directories": {
            "staging": str(settings.STAGING_DATA_DIR),
            "processed": str(settings.PROCESSED_DATA_DIR),
            "staging_exists": settings.STAGING_DATA_DIR.exists(),
            "processed_exists": settings.PROCESSED_DATA_DIR.exists()
        }
    }


@app.get("/api/v1/kpi", response_model=Dict[str, Any])
async def get_executive_kpis(
    start_date: Optional[str] = Query(None, description="Start date (YYYY-MM-DD)"),
    end_date: Optional[str] = Query(None, description="End date (YYYY-MM-DD)")
):
    """
    Get executive KPIs for a specified date range.
    
    Returns total revenue, net revenue, orders, customers, AOV, return rate, and gross margin.
    """
    try:
        orders_df = load_staging_data("orders.csv")
        if orders_df is None:
            raise HTTPException(status_code=404, detail="Orders data not found")
        
        orders_df["order_date"] = pd.to_datetime(orders_df["order_date"])
        
        # Set default date range if not provided
        if start_date is None:
            start_date = "2024-01-01"
        if end_date is None:
            end_date = "2024-12-31"
        
        start_dt = pd.to_datetime(start_date)
        end_dt = pd.to_datetime(end_date)
        
        # Filter orders by date range
        filtered_orders = orders_df[
            (orders_df["order_date"] >= start_dt) & 
            (orders_df["order_date"] <= end_dt)
        ]
        
        valid_orders = filtered_orders[filtered_orders["order_status"].isin(['Delivered', 'Shipped', 'Processing', 'Returned'])]
        
        # Calculate KPIs
        total_revenue = float(valid_orders["order_total"].sum())
        cancelled_revenue = float(filtered_orders[filtered_orders["order_status"] == "Cancelled"]["order_total"].sum())
        returned_revenue = float(filtered_orders[filtered_orders["order_status"] == "Returned"]["order_total"].sum())
        net_revenue = total_revenue - cancelled_revenue - returned_revenue
        
        total_orders = len(valid_orders)
        unique_customers = int(valid_orders["customer_id"].nunique())
        aov = total_revenue / total_orders if total_orders > 0 else 0.0
        return_rate = (len(filtered_orders[filtered_orders["order_status"] == "Returned"]) / total_orders * 100) if total_orders > 0 else 0.0
        gross_margin_pct = 29.0  # Assumed 29% margin
        
        return {
            "total_revenue": total_revenue,
            "net_revenue": net_revenue,
            "total_orders": total_orders,
            "unique_customers": unique_customers,
            "aov": aov,
            "return_rate": return_rate,
            "gross_margin_pct": gross_margin_pct,
            "period_start": start_date,
            "period_end": end_date,
            "currency": "INR"
        }
    except Exception as e:
        logger.error(f"Error calculating KPIs: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/v1/churn/predict", response_model=Dict[str, Any])
async def predict_churn(request: CustomerChurnRequest):
    """
    Predict churn probability for a specific customer.
    
    Returns churn probability, risk tier, and key risk drivers.
    """
    try:
        churn_predictions_df = load_processed_data("churn_predictions.csv")
        
        if churn_predictions_df is None:
            # Fallback: use churn features
            churn_features_df = load_processed_data("customer_churn_features.csv")
            if churn_features_df is None:
                raise HTTPException(status_code=404, detail="Churn prediction data not found")
            
            # Simple heuristic prediction
            customer_data = churn_features_df[churn_features_df["customer_id"] == request.customer_id]
            if len(customer_data) == 0:
                raise HTTPException(status_code=404, detail="Customer not found")
            
            churn_prob = float(customer_data["churn_label_90d"].iloc[0])
            risk_tier = "High" if churn_prob >= 0.6 else "Medium" if churn_prob >= 0.3 else "Low"
            risk_score = int(churn_prob * 1000)
            
            return {
                "customer_id": request.customer_id,
                "churn_probability": churn_prob,
                "risk_tier": risk_tier,
                "risk_score": risk_score,
                "top_risk_drivers": "Data not available",
                "prediction_method": "heuristic"
            }
        
        # Use actual predictions
        customer_data = churn_predictions_df[churn_predictions_df["customer_id"] == request.customer_id]
        if len(customer_data) == 0:
            raise HTTPException(status_code=404, detail="Customer not found")
        
        row = customer_data.iloc[0]
        return {
            "customer_id": request.customer_id,
            "churn_probability": float(row["churn_probability"]),
            "risk_tier": str(row["risk_tier"]),
            "risk_score": int(row["risk_score"]),
            "top_risk_drivers": str(row.get("top_risk_drivers", "Not available")),
            "prediction_method": "ml_model"
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error predicting churn: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/v1/forecast", response_model=Dict[str, Any])
async def get_forecast(request: ForecastRequest):
    """
    Get demand forecast for specified horizon.
    
    Returns forecast values with confidence intervals.
    """
    try:
        forecast_history_df = load_processed_data("forecast_overall_history.csv")
        forecast_future_df = load_processed_data("forecast_overall_best.csv")
        
        if forecast_history_df is None or forecast_future_df is None:
            raise HTTPException(status_code=404, detail="Forecast data not found")
        
        # Get forecast for specified horizon
        forecast_data = forecast_future_df.head(request.horizon_days)
        
        if len(forecast_data) == 0:
            raise HTTPException(status_code=400, detail="Insufficient forecast data")
        
        dates = pd.to_datetime(forecast_data["date"]).dt.strftime("%Y-%m-%d").tolist()
        forecast_values = forecast_data["forecast_units"].tolist()
        
        # Calculate confidence bounds (simplified)
        forecast_array = np.array(forecast_values)
        lower_bound = (forecast_array * 0.9).tolist()
        upper_bound = (forecast_array * 1.1).tolist()
        
        # Get model metrics if available
        model_comparison = load_processed_data("forecast_model_comparison.csv")
        mae = None
        rmse = None
        mape_pct = None
        
        if model_comparison is not None and len(model_comparison) > 0:
            best_model_row = model_comparison.loc[model_comparison["mape_pct"].idxmin()]
            mae = float(best_model_row.get("mae", 0))
            rmse = float(best_model_row.get("rmse", 0))
            mape_pct = float(best_model_row.get("mape_pct", 0))
        
        return {
            "dates": dates,
            "forecast_values": forecast_values,
            "lower_bound": lower_bound,
            "upper_bound": upper_bound,
            "model_type": request.model_type,
            "mae": mae,
            "rmse": rmse,
            "mape_pct": mape_pct,
            "horizon_days": request.horizon_days
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error generating forecast: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/v1/recommendations", response_model=Dict[str, Any])
async def get_recommendations(request: RecommendationRequest):
    """
    Get product recommendations for a customer.
    
    Returns personalized product recommendations based on purchase history and collaborative filtering.
    """
    try:
        # Load customer and order data
        customers_df = load_staging_data("customers.csv")
        orders_df = load_staging_data("orders.csv")
        order_items_df = load_staging_data("order_items.csv")
        products_df = load_staging_data("products.csv")
        
        if any(df is None for df in [customers_df, orders_df, order_items_df, products_df]):
            raise HTTPException(status_code=404, detail="Required data not found")
        
        # Check if customer exists
        customer = customers_df[customers_df["customer_id"] == request.customer_id]
        if len(customer) == 0:
            raise HTTPException(status_code=404, detail="Customer not found")
        
        # Get customer's purchase history
        customer_orders = orders_df[orders_df["customer_id"] == request.customer_id]
        if len(customer_orders) == 0:
            # New customer - recommend popular products
            popular_products = order_items_df.groupby("product_id")["quantity"].sum().nlargest(request.num_recommendations)
            recommended_product_ids = popular_products.index.tolist()
        else:
            # Existing customer - recommend based on category preferences
            customer_order_items = order_items_df[order_items_df["order_id"].isin(customer_orders["order_id"])]
            category_counts = customer_order_items.merge(
                products_df[["product_id", "category_id"]], on="product_id", how="left"
            )["category_id"].value_counts()
            
            # Get top category
            top_category = category_counts.index[0] if len(category_counts) > 0 else None
            
            if top_category:
                # Recommend products from same category
                category_products = products_df[products_df["category_id"] == top_category]["product_id"]
                purchased_product_ids = customer_order_items["product_id"].unique()
                recommended_product_ids = category_products[~category_products.isin(purchased_product_ids)].head(request.num_recommendations).tolist()
            else:
                # Fallback to popular products
                popular_products = order_items_df.groupby("product_id")["quantity"].sum().nlargest(request.num_recommendations)
                recommended_product_ids = popular_products.index.tolist()
        
        # Get product details
        recommended_products = products_df[products_df["product_id"].isin(recommended_product_ids)]
        
        recommendations = []
        for _, product in recommended_products.iterrows():
            recommendations.append({
                "product_id": int(product["product_id"]),
                "product_name": str(product["product_name"]),
                "category_id": int(product["category_id"]),
                "selling_price": float(product["selling_price"]),
                "brand_name": str(product.get("brand_name", "Unknown"))
            })
        
        return {
            "customer_id": request.customer_id,
            "recommendations": recommendations,
            "strategy": "category_based" if len(customer_orders) > 0 else "popular_products",
            "num_recommendations": len(recommendations)
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error generating recommendations: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/customers/{customer_id}", response_model=Dict[str, Any])
async def get_customer_profile(customer_id: int):
    """
    Get comprehensive profile for a specific customer.
    
    Returns customer demographics, purchase history, and analytics metrics.
    """
    try:
        customers_df = load_staging_data("customers.csv")
        orders_df = load_staging_data("orders.csv")
        churn_predictions_df = load_processed_data("churn_predictions.csv")
        
        if customers_df is None:
            raise HTTPException(status_code=404, detail="Customer data not found")
        
        customer = customers_df[customers_df["customer_id"] == customer_id]
        if len(customer) == 0:
            raise HTTPException(status_code=404, detail="Customer not found")
        
        customer_info = customer.iloc[0].to_dict()
        
        # Get order statistics
        if orders_df is not None:
            customer_orders = orders_df[orders_df["customer_id"] == customer_id]
            order_stats = {
                "total_orders": len(customer_orders),
                "total_spend": float(customer_orders["order_total"].sum()),
                "avg_order_value": float(customer_orders["order_total"].mean()),
                "last_order_date": str(customer_orders["order_date"].max()) if len(customer_orders) > 0 else None
            }
        else:
            order_stats = {}
        
        # Get churn prediction
        churn_info = {}
        if churn_predictions_df is not None:
            churn_data = churn_predictions_df[churn_predictions_df["customer_id"] == customer_id]
            if len(churn_data) > 0:
                churn_info = {
                    "churn_probability": float(churn_data.iloc[0]["churn_probability"]),
                    "risk_tier": str(churn_data.iloc[0]["risk_tier"]),
                    "risk_score": int(churn_data.iloc[0]["risk_score"])
                }
        
        return {
            "customer_id": customer_id,
            "profile": customer_info,
            "order_statistics": order_stats,
            "churn_prediction": churn_info
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting customer profile: {e}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
