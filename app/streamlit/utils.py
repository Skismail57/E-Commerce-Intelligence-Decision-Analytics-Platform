"""
Streamlit Utilities Module
Contains utility functions for the Streamlit application.
"""

import streamlit as st
import pandas as pd
import numpy as np
from config.settings import settings
from config.logging_config import get_logger

logger = get_logger(__name__)


def format_currency(value):
    """Format value as Indian Rupees"""
    if pd.isna(value):
        return "₹0"
    if abs(value) >= 1e7:
        return f"₹{value/1e7:.2f} Cr"
    elif abs(value) >= 1e5:
        return f"₹{value/1e5:.2f} L"
    else:
        return f"₹{value:,.0f}"


def format_number(value):
    """Format large numbers"""
    if pd.isna(value):
        return "0"
    if abs(value) >= 1e7:
        return f"{value/1e7:.2f} Cr"
    elif abs(value) >= 1e5:
        return f"{value/1e5:.2f} L"
    elif abs(value) >= 1e3:
        return f"{value/1e3:.2f} K"
    else:
        return f"{value:,.0f}"


def format_percentage(value):
    """Format as percentage"""
    if pd.isna(value):
        return "0%"
    return f"{value:.2f}%"


@st.cache_data(ttl=3600)
def load_data():
    """Load all data from processed directory"""
    data_dir = settings.PROCESSED_DATA_DIR
    data = {}
    
    try:
        # Core tables
        core_files = [
            "customers.csv", "products.csv", "categories.csv",
            "orders.csv", "order_items.csv", "payments.csv",
            "returns.csv", "reviews.csv"
        ]
        
        for file in core_files:
            file_path = data_dir / file
            if file_path.exists():
                table_name = file.replace(".csv", "")
                data[table_name] = pd.read_csv(file_path)
                logger.info(f"Loaded {table_name} from {file_path}")
        
        # Analytics tables
        transform_files = [
            "rfm_segments.csv", "clv.csv", "churn_features.csv", "product_matrix.csv",
            "orders.csv", "order_items.csv",
            "forecast_model_comparison.csv", "forecast_overall_best.csv",
            "forecast_overall_future.csv", "forecast_overall_history.csv",
            "forecast_top_products_future.csv"
        ]
        
        for file in transform_files:
            file_path = data_dir / file
            if file_path.exists():
                table_name = file.replace(".csv", "")
                data[table_name] = pd.read_csv(file_path)
                logger.info(f"Loaded {table_name} from {file_path}")
        
        logger.info(f"Loaded {len(data)} tables")
        
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        st.error(f"Error loading data: {e}")
    
    return data


def filter_data_by_date(df, date_column, start_date, end_date):
    """Filter dataframe by date range"""
    df[date_column] = pd.to_datetime(df[date_column])
    return df[
        (df[date_column] >= pd.to_datetime(start_date)) &
        (df[date_column] <= pd.to_datetime(end_date))
    ]


def calculate_kpis(orders_df, customers_df):
    """Calculate executive KPIs"""
    if orders_df is None or len(orders_df) == 0:
        return None
    
    total_revenue = orders_df['order_total'].sum()
    total_orders_count = len(orders_df)
    customers_count = customers_df['customer_id'].nunique() if customers_df is not None else 0
    
    # Calculate net revenue (excluding returns)
    returned_orders = orders_df[orders_df['order_status'] == 'Returned']
    return_amount = returned_orders['order_total'].sum() if len(returned_orders) > 0 else 0
    net_revenue = total_revenue - return_amount
    
    # Calculate profit (assuming 30% margin)
    gross_margin = 0.30
    gross_profit = net_revenue * gross_margin
    
    # Calculate AOV
    aov = total_revenue / total_orders_count if total_orders_count > 0 else 0
    
    # Calculate return rate
    return_rate = (len(returned_orders) / total_orders_count * 100) if total_orders_count > 0 else 0
    
    return {
        'total_revenue': total_revenue,
        'net_revenue': net_revenue,
        'gross_profit': gross_profit,
        'gross_margin': gross_margin * 100,
        'total_orders': total_orders_count,
        'customers': customers_count,
        'aov': aov,
        'return_rate': return_rate
    }
