"""
Forecasting Page Module
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from app.streamlit.utils import format_currency
from config.logging_config import get_logger

logger = get_logger(__name__)


def render(data):
    """Render forecasting page"""
    st.markdown('<h1 class="main-header float">📈 Demand Forecasting</h1>', unsafe_allow_html=True)
    
    # Overall Forecast
    st.subheader("Overall Demand Forecast")
    
    if "forecast_overall_future" in data:
        forecast_df = data["forecast_overall_future"]
        
        if len(forecast_df) > 0:
            # Plot forecast
            fig = go.Figure()
            
            date_col = 'date' if 'date' in forecast_df.columns else 'ds' if 'ds' in forecast_df.columns else forecast_df.columns[0]
            forecast_col = 'forecast' if 'forecast' in forecast_df.columns else 'yhat' if 'yhat' in forecast_df.columns else forecast_df.columns[1] if len(forecast_df.columns) > 1 else forecast_df.columns[0]
            
            fig.add_trace(go.Scatter(
                x=forecast_df[date_col],
                y=forecast_df[forecast_col],
                mode='lines+markers',
                name='Forecast',
                line=dict(color='#00ffff', width=2),
                marker=dict(size=6)
            ))
            
            lower_col = 'lower_bound' if 'lower_bound' in forecast_df.columns else 'yhat_lower' if 'yhat_lower' in forecast_df.columns else None
            upper_col = 'upper_bound' if 'upper_bound' in forecast_df.columns else 'yhat_upper' if 'yhat_upper' in forecast_df.columns else None
            
            if lower_col and upper_col:
                fig.add_trace(go.Scatter(
                    x=forecast_df[date_col],
                    y=forecast_df[upper_col],
                    mode='lines',
                    name='Upper Bound',
                    line=dict(color='rgba(0, 255, 255, 0.3)', width=0),
                    showlegend=False
                ))
                fig.add_trace(go.Scatter(
                    x=forecast_df[date_col],
                    y=forecast_df[lower_col],
                    mode='lines',
                    name='Lower Bound',
                    line=dict(color='rgba(0, 255, 255, 0.3)', width=0),
                    fill='tonexty',
                    fillcolor='rgba(0, 255, 255, 0.1)',
                    showlegend=False
                ))
                
                fig.update_layout(
                    title='Demand Forecast',
                    xaxis_title='Date',
                    yaxis_title='Demand',
                    template='plotly_dark',
                    plot_bgcolor='rgba(26, 26, 46, 0.8)',
                    paper_bgcolor='rgba(26, 26, 46, 0.8)',
                    font=dict(color='#ffffff'),
                    margin=dict(l=60, r=40, t=80, b=60),
                    height=400
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Forecast summary
                total_forecast = forecast_df[forecast_col].sum()
                st.metric("Total Forecasted Demand", f"{total_forecast:,.0f}")
        else:
            st.info("Forecast data not available")
    else:
        st.info("Forecast not available. Please run demand forecasting first.")
    
    # Model Comparison
    st.subheader("Forecast Model Comparison")
    
    if "forecast_model_comparison" in data:
        model_comparison = data["forecast_model_comparison"]
        
        if len(model_comparison) > 0:
            model_col = 'model' if 'model' in model_comparison.columns else 'Model' if 'Model' in model_comparison.columns else model_comparison.columns[0]
            rmse_col = 'rmse' if 'rmse' in model_comparison.columns else 'RMSE' if 'RMSE' in model_comparison.columns else model_comparison.columns[1] if len(model_comparison.columns) > 1 else model_comparison.columns[0]
            
            fig = go.Figure(data=[go.Bar(
                x=model_comparison[model_col],
                y=model_comparison[rmse_col],
                marker=dict(color='#ff00ff'),
                text=[f"{x:.2f}" for x in model_comparison[rmse_col]],
                textposition='outside'
            )])
            
            fig.update_layout(
                title='Model Performance (RMSE)',
                xaxis_title='Model',
                yaxis_title='RMSE',
                template='plotly_dark',
                plot_bgcolor='rgba(26, 26, 46, 0.8)',
                paper_bgcolor='rgba(26, 26, 46, 0.8)',
                font=dict(color='#ffffff'),
                margin=dict(l=60, r=40, t=80, b=60)
            )
            st.plotly_chart(fig, use_container_width=True)
            
            st.dataframe(model_comparison, use_container_width=True)
        else:
            st.info("Model comparison data not available")
