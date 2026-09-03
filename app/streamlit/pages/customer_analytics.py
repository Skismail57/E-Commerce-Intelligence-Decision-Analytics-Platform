"""
Customer Analytics Page Module
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from app.streamlit.utils import format_number
from config.logging_config import get_logger

logger = get_logger(__name__)


def render(data):
    """Render customer analytics page"""
    st.markdown('<h1 class="main-header float">👥 Customer Analytics</h1>', unsafe_allow_html=True)
    
    # RFM Segmentation
    st.subheader("RFM Segmentation")
    
    if "rfm_segments" in data:
        rfm_df = data["rfm_segments"]
        
        if len(rfm_df) > 0:
            segment_col = 'rfm_segment' if 'rfm_segment' in rfm_df.columns else 'segment' if 'segment' in rfm_df.columns else 'customer_segment'
            segment_counts = rfm_df[segment_col].value_counts()
            
            fig = go.Figure(data=[go.Pie(
                labels=segment_counts.index,
                values=segment_counts.values,
                hole=0.4,
                marker=dict(colors=['#00ffff', '#ff00ff', '#00ff00', '#ff6600', '#ff0066', '#9900ff', '#0066ff'])
            )])
            
            fig.update_layout(
                title='Customer Segments Distribution',
                template='plotly_dark',
                plot_bgcolor='rgba(26, 26, 46, 0.8)',
                paper_bgcolor='rgba(26, 26, 46, 0.8)',
                font=dict(color='#ffffff'),
                margin=dict(l=20, r=20, t=80, b=20)
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Segment details
            st.subheader("Segment Details")
            segment_stats = rfm_df.groupby(segment_col).agg({
                "customer_id": "count",
                "recency_days" if "recency_days" in rfm_df.columns else "recency": "mean",
                "frequency": "mean",
                "monetary_value" if "monetary_value" in rfm_df.columns else "monetary": "mean"
            }).reset_index()
            segment_stats.columns = ["Segment", "Customers", "Avg Recency", "Avg Frequency", "Avg Monetary"]
            
            st.dataframe(segment_stats, use_container_width=True)
        else:
            st.info("RFM segment data not available")
    else:
        st.info("RFM segments not available. Please run customer analytics first.")
    
    # CLV Analysis
    st.subheader("Customer Lifetime Value")
    
    if "clv" in data:
        clv_df = data["clv"]
        
        if len(clv_df) > 0:
            # CLV distribution
            clv_col = 'predicted_clv' if 'predicted_clv' in clv_df.columns else 'clv' if 'clv' in clv_df.columns else clv_df.columns[0]
            fig = go.Figure(data=[go.Histogram(
                x=clv_df[clv_col],
                nbinsx=30,
                marker=dict(color='#00ffff')
            )])
            
            fig.update_layout(
                title='CLV Distribution',
                xaxis_title='Predicted CLV (₹)',
                yaxis_title='Count',
                template='plotly_dark',
                plot_bgcolor='rgba(26, 26, 46, 0.8)',
                paper_bgcolor='rgba(26, 26, 46, 0.8)',
                font=dict(color='#ffffff'),
                margin=dict(l=60, r=40, t=80, b=60)
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # CLV by segment
            segment_col = 'segment' if 'segment' in clv_df.columns else 'customer_segment' if 'customer_segment' in clv_df.columns else 'rfm_segment' if 'rfm_segment' in clv_df.columns else 'customer_value_tier' if 'customer_value_tier' in clv_df.columns else None
            if segment_col:
                clv_by_segment = clv_df.groupby(segment_col)[clv_col].agg(['mean', 'sum']).reset_index()
                
                fig = go.Figure(data=[go.Bar(
                    x=clv_by_segment[segment_col],
                    y=clv_by_segment["mean"],
                    marker=dict(color='#ff00ff'),
                    text=[f"₹{x/1e5:.1f}L" for x in clv_by_segment["mean"]],
                    textposition='outside'
                )])
                
                fig.update_layout(
                    title='Average CLV by Segment',
                    xaxis_title='Segment',
                    yaxis_title='Average CLV (₹)',
                    template='plotly_dark',
                    plot_bgcolor='rgba(26, 26, 46, 0.8)',
                    paper_bgcolor='rgba(26, 26, 46, 0.8)',
                    font=dict(color='#ffffff'),
                    margin=dict(l=60, r=40, t=80, b=60)
                )
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("CLV data not available")
    else:
        st.info("CLV predictions not available. Please run CLV analysis first.")
