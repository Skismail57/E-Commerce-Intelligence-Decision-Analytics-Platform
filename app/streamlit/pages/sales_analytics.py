"""
Sales Analytics Page Module
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from app.streamlit.utils import format_currency
from config.logging_config import get_logger

logger = get_logger(__name__)


def render(data):
    """Render sales analytics page"""
    st.markdown('<h1 class="main-header float">📊 Sales Analytics</h1>', unsafe_allow_html=True)
    
    if "orders" not in data:
        st.warning("Order data not available. Please run data transformations first.")
        return
    
    orders_df = data["orders"].copy()
    orders_df["order_date"] = pd.to_datetime(orders_df["order_date"])
    
    # Date filter
    col1, col2 = st.columns(2)
    with col1:
        start_date = st.date_input("Start Date", value=pd.to_datetime("2024-01-01").date())
    with col2:
        end_date = st.date_input("End Date", value=pd.to_datetime("2024-12-31").date())
    
    # Filter orders by date range
    filtered_orders = orders_df[
        (orders_df["order_date"] >= pd.to_datetime(start_date)) &
        (orders_df["order_date"] <= pd.to_datetime(end_date))
    ]
    
    # Daily Sales Trend
    st.subheader("Daily Sales Trend")
    daily_sales = filtered_orders.groupby(filtered_orders["order_date"].dt.date).agg({
        "order_total": "sum",
        "order_id": "count"
    }).reset_index()
    daily_sales.columns = ["date", "revenue", "orders"]
    
    if len(daily_sales) > 0:
        fig = make_subplots(specs=[[{"secondary_y": False}]])
        
        fig.add_trace(
            go.Scatter(
                x=daily_sales["date"],
                y=daily_sales["revenue"],
                mode='lines',
                name='Revenue',
                line=dict(color='#00ffff', width=2)
            ),
            secondary_y=False
        )
        
        fig.update_layout(
            title='Daily Sales Trend',
            xaxis_title='Date',
            yaxis_title='Revenue (₹)',
            template='plotly_dark',
            plot_bgcolor='rgba(26, 26, 46, 0.8)',
            paper_bgcolor='rgba(26, 26, 46, 0.8)',
            font=dict(color='#ffffff'),
            margin=dict(l=60, r=40, t=80, b=60),
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No sales data for selected date range")
    
    # Order Status Distribution
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Order Status Distribution")
        status_counts = filtered_orders["order_status"].value_counts()
        
        fig = go.Figure(data=[go.Bar(
            x=status_counts.index,
            y=status_counts.values,
            marker=dict(color=['#00ff00', '#00ffff', '#ff00ff', '#ff6600', '#ff0066']),
            text=status_counts.values,
            textposition='outside'
        )])
        
        fig.update_layout(
            title='Order Status Distribution',
            xaxis_title='Status',
            yaxis_title='Count',
            template='plotly_dark',
            plot_bgcolor='rgba(26, 26, 46, 0.8)',
            paper_bgcolor='rgba(26, 26, 46, 0.8)',
            font=dict(color='#ffffff'),
            margin=dict(l=60, r=40, t=80, b=60)
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("Day of Week Analysis")
        filtered_orders["day_of_week"] = filtered_orders["order_date"].dt.day_name()
        dow_revenue = filtered_orders.groupby("day_of_week")["order_total"].sum()
        day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        dow_revenue = dow_revenue.reindex(day_order, fill_value=0)
        
        fig = go.Figure(data=[go.Bar(
            x=dow_revenue.index,
            y=dow_revenue.values,
            marker=dict(color='#ff00ff'),
            text=[f"₹{x/1e5:.1f}L" for x in dow_revenue.values],
            textposition='outside'
        )])
        
        fig.update_layout(
            title='Revenue by Day of Week',
            xaxis_title='Day',
            yaxis_title='Revenue (₹)',
            template='plotly_dark',
            plot_bgcolor='rgba(26, 26, 46, 0.8)',
            paper_bgcolor='rgba(26, 26, 46, 0.8)',
            font=dict(color='#ffffff'),
            margin=dict(l=60, r=40, t=80, b=60)
        )
        st.plotly_chart(fig, use_container_width=True)
