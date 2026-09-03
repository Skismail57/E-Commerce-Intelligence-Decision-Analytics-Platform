"""
Churn Prediction Page Module
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from app.streamlit.utils import format_percentage
from config.logging_config import get_logger

logger = get_logger(__name__)


def render(data):
    """Render churn prediction page"""
    st.markdown('<h1 class="main-header float">🎯 Customer Churn Prediction</h1>', unsafe_allow_html=True)
    
    if "churn_features" in data:
        churn_df = data["churn_features"]
        
        if len(churn_df) > 0:
            # Churn risk distribution
            st.subheader("Churn Risk Distribution")
            
            if "churn_probability" in churn_df.columns:
                churn_df['risk_segment'] = pd.cut(
                    churn_df['churn_probability'],
                    bins=[0, 0.3, 0.6, 1.0],
                    labels=['Low Risk', 'Medium Risk', 'High Risk']
                )
                
                risk_counts = churn_df['risk_segment'].value_counts()
                
                fig = go.Figure(data=[go.Pie(
                    labels=risk_counts.index,
                    values=risk_counts.values,
                    hole=0.4,
                    marker=dict(colors=['#00ff00', '#ff6600', '#ff0066'])
                )])
                
                fig.update_layout(
                    title='Customer Risk Segments',
                    template='plotly_dark',
                    plot_bgcolor='rgba(26, 26, 46, 0.8)',
                    paper_bgcolor='rgba(26, 26, 46, 0.8)',
                    font=dict(color='#ffffff'),
                    margin=dict(l=20, r=20, t=80, b=20)
                )
                st.plotly_chart(fig, use_container_width=True)
            
            # Churn by segment
            st.subheader("Churn by Customer Segment")
            
            if "segment" in churn_df.columns and "churn_label_90d" in churn_df.columns:
                churn_by_segment = churn_df.groupby('segment')['churn_label_90d'].mean().reset_index()
                churn_by_segment.columns = ['Segment', 'Churn Rate']
            elif "customer_segment" in churn_df.columns and "churn_label_90d" in churn_df.columns:
                churn_by_segment = churn_df.groupby('customer_segment')['churn_label_90d'].mean().reset_index()
                churn_by_segment.columns = ['Segment', 'Churn Rate']
            elif "rfm_segment" in churn_df.columns and "churn_label_90d" in churn_df.columns:
                churn_by_segment = churn_df.groupby('rfm_segment')['churn_label_90d'].mean().reset_index()
                churn_by_segment.columns = ['Segment', 'Churn Rate']
            else:
                st.info("Segment data not available for churn analysis")
                return
                
            fig = go.Figure(data=[go.Bar(
                x=churn_by_segment['Segment'],
                y=churn_by_segment['Churn Rate'],
                marker=dict(color='#ff00ff'),
                text=[f"{x:.1%}" for x in churn_by_segment['Churn Rate']],
                textposition='outside'
            )])
            
            fig.update_layout(
                title='Churn Rate by Segment',
                xaxis_title='Segment',
                yaxis_title='Churn Rate',
                template='plotly_dark',
                plot_bgcolor='rgba(26, 26, 46, 0.8)',
                paper_bgcolor='rgba(26, 26, 46, 0.8)',
                font=dict(color='#ffffff'),
                margin=dict(l=60, r=40, t=80, b=60)
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Feature importance
            st.subheader("Churn Risk Factors")
            
            # Display top at-risk customers
            st.subheader("Top At-Risk Customers")
            
            if "customer_id" in churn_df.columns and "churn_probability" in churn_df.columns:
                segment_col = 'segment' if 'segment' in churn_df.columns else 'customer_segment' if 'customer_segment' in churn_df.columns else 'rfm_segment' if 'rfm_segment' in churn_df.columns else None
                cols_to_show = ['customer_id', 'churn_probability']
                if segment_col:
                    cols_to_show.append(segment_col)
                top_risk = churn_df.nlargest(20, 'churn_probability')[cols_to_show]
                top_risk.columns = ['Customer ID', 'Churn Probability'] + (['Segment'] if segment_col else [])
                top_risk['Churn Probability'] = top_risk['Churn Probability'].apply(lambda x: f"{x:.1%}")
                
                st.dataframe(top_risk, use_container_width=True)
        else:
            st.info("Churn prediction data not available. Please run churn modeling first.")
    else:
        st.info("Churn features not available. Please run churn feature engineering first.")
