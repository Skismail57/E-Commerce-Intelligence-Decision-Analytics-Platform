"""
Product Analytics Page Module
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from app.streamlit.utils import format_currency
from config.logging_config import get_logger

logger = get_logger(__name__)


def render(data):
    """Render product analytics page"""
    st.markdown('<h1 class="main-header float">📦 Product Analytics</h1>', unsafe_allow_html=True)
    
    # Product Matrix
    st.subheader("Product Matrix")
    
    if "product_matrix" in data:
        product_matrix = data["product_matrix"]
        
        if len(product_matrix) > 0:
            fig = go.Figure()
            
            quadrants = {
                'Stars': {'color': '#00ff00', 'x': 'high', 'y': 'high'},
                'Volume': {'color': '#00ffff', 'x': 'low', 'y': 'high'},
                'Premium': {'color': '#ff00ff', 'x': 'high', 'y': 'low'},
                'Remove': {'color': '#ff0066', 'x': 'low', 'y': 'low'}
            }
            
            for quadrant, config in quadrants.items():
                quadrant_data = product_matrix[product_matrix['quadrant'] == quadrant]
                if len(quadrant_data) > 0:
                    fig.add_trace(go.Scatter(
                        x=quadrant_data['margin_ratio'] * 100,
                        y=quadrant_data['revenue_inr'],
                        mode='markers',
                        name=quadrant,
                        marker=dict(
                            size=10,
                            color=config['color'],
                            opacity=0.7
                        ),
                        text=quadrant_data['product_name'],
                        hovertemplate='<b>%{text}</b><br>Margin: %{x:.2f}%<br>Revenue: ₹%{y:,.0f}<extra></extra>'
                    ))
            
            fig.update_layout(
                title='Product Matrix (Margin vs Revenue)',
                xaxis_title='Margin (%)',
                yaxis_title='Revenue (₹)',
                template='plotly_dark',
                plot_bgcolor='rgba(26, 26, 46, 0.8)',
                paper_bgcolor='rgba(26, 26, 46, 0.8)',
                font=dict(color='#ffffff'),
                margin=dict(l=60, r=40, t=80, b=60),
                height=500
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Quadrant summary
            quadrant_summary = product_matrix.groupby('quadrant').agg({
                'product_id': 'count',
                'revenue_inr': 'sum',
                'margin_ratio': 'mean'
            }).reset_index()
            quadrant_summary.columns = ['Quadrant', 'Products', 'Total Revenue', 'Avg Margin']
            quadrant_summary['Avg Margin'] = quadrant_summary['Avg Margin'] * 100
            
            st.dataframe(quadrant_summary, use_container_width=True)
        else:
            st.info("Product matrix data not available")
    else:
        st.info("Product matrix not available. Please run product analytics first.")
    
    # Top Products
    st.subheader("Top Products by Revenue")
    
    if "order_items" in data and "products" in data:
        order_items_df = data["order_items"]
        products_df = data["products"]
        
        revenue_col = 'line_total' if 'line_total' in order_items_df.columns else 'quantity'
        product_revenue = order_items_df.groupby('product_id')[revenue_col].sum().reset_index()
        product_revenue = product_revenue.merge(
            products_df[['product_id', 'product_name', 'category_id']],
            on='product_id',
            how='left'
        )
        product_revenue = product_revenue.sort_values(revenue_col, ascending=False).head(20)
        
        if len(product_revenue) > 0:
            fig = go.Figure(data=[go.Bar(
                x=product_revenue[revenue_col],
                y=product_revenue['product_name'],
                orientation='h',
                marker=dict(color='#00ffff'),
                text=[f"₹{x/1e5:.1f}L" if revenue_col == 'line_total' else f"{x:,.0f}" for x in product_revenue[revenue_col]],
                textposition='outside'
            )])
            
            fig.update_layout(
                title='Top 20 Products by Revenue',
                xaxis_title='Revenue (₹)',
                yaxis_title='Product',
                template='plotly_dark',
                plot_bgcolor='rgba(26, 26, 46, 0.8)',
                paper_bgcolor='rgba(26, 26, 46, 0.8)',
                font=dict(color='#ffffff'),
                margin=dict(l=200, r=40, t=80, b=60),
                height=600
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Product revenue data not available")
    else:
        st.info("Product data not available")
