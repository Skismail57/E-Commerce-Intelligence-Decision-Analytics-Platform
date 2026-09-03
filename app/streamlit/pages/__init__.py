"""
Streamlit Pages Module
"""

from app.streamlit.pages.executive_dashboard import render as render_executive_dashboard
from app.streamlit.pages.sales_analytics import render as render_sales_analytics
from app.streamlit.pages.customer_analytics import render as render_customer_analytics
from app.streamlit.pages.product_analytics import render as render_product_analytics
from app.streamlit.pages.churn_prediction import render as render_churn_prediction
from app.streamlit.pages.forecasting import render as render_forecasting

__all__ = [
    'render_executive_dashboard',
    'render_sales_analytics',
    'render_customer_analytics',
    'render_product_analytics',
    'render_churn_prediction',
    'render_forecasting'
]
