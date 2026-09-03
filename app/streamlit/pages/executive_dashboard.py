"""
Executive Dashboard Page Module
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from app.streamlit.utils import format_currency, format_number, calculate_kpis
from config.logging_config import get_logger

logger = get_logger(__name__)


def render(data):
    """Render executive dashboard page"""
    st.markdown('<h1 class="main-header float">🏠 Executive Dashboard</h1>', unsafe_allow_html=True)
    
    # Advanced Interactive Controls
    with st.expander("🎛️ Advanced Filters & Options", expanded=True):
        col1, col2, col3 = st.columns(3)
        with col1:
            date_range = st.date_input(
                "Date Range",
                value=(pd.to_datetime("2024-01-01"), pd.to_datetime("2024-12-31")),
                key="exec_date_range"
            )
        with col2:
            status_filter = st.multiselect(
                "Order Status",
                options=['Delivered', 'Shipped', 'Processing', 'Returned', 'Cancelled'],
                default=['Delivered', 'Shipped', 'Processing', 'Returned'],
                key="exec_status_filter"
            )
        with col3:
            view_mode = st.selectbox(
                "View Mode",
                ["Overview", "Detailed", "Comparative"],
                key="exec_view_mode"
            )
    
    # Calculate KPIs
    if "orders" in data and "customers" in data:
        orders_df = data["orders"].copy()
        customers_df = data["customers"].copy()
        
        # Filter by date range
        if len(date_range) == 2:
            orders_df = orders_df[
                (pd.to_datetime(orders_df["order_date"]) >= pd.to_datetime(date_range[0])) &
                (pd.to_datetime(orders_df["order_date"]) <= pd.to_datetime(date_range[1]))
            ]
        
        # Filter by status
        if status_filter:
            orders_df = orders_df[orders_df["order_status"].isin(status_filter)]
        
        valid_orders = orders_df[orders_df["order_status"].isin(['Delivered', 'Shipped', 'Processing', 'Returned'])]
        
        kpis = calculate_kpis(valid_orders, customers_df)
        
        if kpis:
            total_revenue = kpis['total_revenue']
            net_revenue = kpis['net_revenue']
            gross_profit = kpis['gross_profit']
            gross_margin = kpis['gross_margin']
            total_orders_count = kpis['total_orders']
            customers_count = kpis['customers']
            aov = kpis['aov']
            return_rate = kpis['return_rate']
        else:
            st.warning("Unable to calculate KPIs")
            return
    else:
        st.warning("Order data not available. Please run data transformations first.")
        return
    
    # KPI Cards
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Revenue", f"₹{total_revenue/1e7:.2f} Cr")
    with col2:
        st.metric("Net Revenue", f"₹{net_revenue/1e7:.2f} Cr")
    with col3:
        st.metric("Gross Profit", f"₹{gross_profit/1e7:.2f} Cr")
    with col4:
        st.metric("Gross Margin", f"{gross_margin:.1f}%")
    
    col5, col6, col7, col8 = st.columns(4)
    
    with col5:
        st.metric("Total Orders", f"{total_orders_count:,}")
    with col6:
        st.metric("Customers", f"{customers_count:,}")
    with col7:
        st.metric("AOV", f"₹{aov:,.0f}")
    with col8:
        st.metric("Return Rate", f"{return_rate:.1f}%")
    
    st.markdown("---")
    
    # Charts
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Revenue Trend")
        if "orders" in data:
            valid_orders["order_month"] = pd.to_datetime(valid_orders["order_date"]).dt.to_period('M')
            monthly_revenue = valid_orders.groupby("order_month")["order_total"].sum().reset_index()
            monthly_revenue["order_month"] = monthly_revenue["order_month"].astype(str)
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=monthly_revenue["order_month"],
                y=monthly_revenue["order_total"],
                mode='lines+markers',
                name='Revenue',
                line=dict(color='#00ffff', width=3),
                marker=dict(size=8, color='#00ffff')
            ))
            
            fig.update_layout(
                title='Monthly Revenue Trend',
                xaxis_title='Month',
                yaxis_title='Revenue (₹)',
                template='plotly_dark',
                plot_bgcolor='rgba(26, 26, 46, 0.8)',
                paper_bgcolor='rgba(26, 26, 46, 0.8)',
                font=dict(color='#ffffff'),
                margin=dict(l=60, r=40, t=80, b=60)
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Revenue trend data not available")
    
    with col2:
        st.subheader("Category Performance")
        if "order_items" in data and "products" in data:
            order_items_df = data["order_items"]
            products_df = data["products"]
            
            revenue_col = 'line_total' if 'line_total' in order_items_df.columns else 'quantity'
            category_revenue = order_items_df.merge(
                products_df[["product_id", "category_id"]], 
                on="product_id", 
                how="left"
            ).groupby("category_id")[revenue_col].sum().reset_index()
            
            if len(category_revenue) > 0:
                fig = go.Figure(data=[go.Bar(
                    x=category_revenue["category_id"],
                    y=category_revenue[revenue_col],
                    marker=dict(color='#ff00ff'),
                    text=[f"₹{x/1e5:.1f}L" if revenue_col == 'line_total' else f"{x:,.0f}" for x in category_revenue[revenue_col]],
                    textposition='outside'
                )])
                
                fig.update_layout(
                    title='Revenue by Category',
                    xaxis_title='Category',
                    yaxis_title='Revenue (₹)',
                    template='plotly_dark',
                    plot_bgcolor='rgba(26, 26, 46, 0.8)',
                    paper_bgcolor='rgba(26, 26, 46, 0.8)',
                    font=dict(color='#ffffff'),
                    margin=dict(l=40, r=40, t=80, b=40)
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Category data not available")
        else:
            st.info("Category performance data not available")
    
    # Regional Performance
    st.subheader("Regional Performance")
    if "orders" in data and "customers" in data:
        orders_customers = orders_df.merge(
            data["customers"][["customer_id", "state"]], on="customer_id", how="left"
        )
        
        state_revenue = orders_customers.groupby("state")["order_total"].sum().reset_index()
        
        if len(state_revenue) > 0:
            fig = go.Figure(data=go.Choropleth(
                locations=state_revenue["state"],
                locationmode='country names',
                z=state_revenue["order_total"],
                text=state_revenue["state"],
                colorscale='Cividis',
                colorbar_title='Revenue (₹)'
            ))
            
            fig.update_layout(
                title='Revenue by State',
                geo=dict(
                    showframe=False,
                    showcoastlines=True,
                    projection_type='equirectangular'
                ),
                template='plotly_dark',
                plot_bgcolor='rgba(26, 26, 46, 0.8)',
                paper_bgcolor='rgba(26, 26, 46, 0.8)',
                font=dict(color='#ffffff'),
                margin=dict(l=0, r=0, t=80, b=0),
                height=400
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Regional performance data not available")
