"""
E-Commerce Intelligence & Decision Analytics Platform
Streamlit Application - Main Dashboard (Refactored)
"""

import streamlit as st
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from config.settings import settings
from config.logging_config import get_logger
from app.streamlit.styles import apply_styles
from app.streamlit.utils import load_data
from app.streamlit.pages import (
    render_executive_dashboard,
    render_sales_analytics,
    render_customer_analytics,
    render_product_analytics,
    render_churn_prediction,
    render_forecasting
)

logger = get_logger(__name__)

# Page configuration
st.set_page_config(
    page_title="E-Commerce Intelligence Platform",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Apply custom styles
apply_styles()

# Sidebar navigation
st.sidebar.title("🚀 Navigation")
st.sidebar.markdown("---")

page = st.sidebar.radio(
    "Select Page",
    [
        "🏠 Executive Dashboard",
        "📊 Sales Analytics",
        "👥 Customer Analytics",
        "📦 Product Analytics",
        "🎯 Churn Prediction",
        "📈 Demand Forecasting"
    ]
)

st.sidebar.markdown("---")
st.sidebar.markdown("### About")
st.sidebar.markdown("""
This platform provides comprehensive e-commerce analytics including:
- Executive KPIs
- Customer Intelligence (RFM, CLV)
- Product Analytics
- Churn Prediction
- Demand Forecasting
""")

# Load data
data = load_data()

# Render selected page
if page == "🏠 Executive Dashboard":
    render_executive_dashboard(data)
elif page == "📊 Sales Analytics":
    render_sales_analytics(data)
elif page == "👥 Customer Analytics":
    render_customer_analytics(data)
elif page == "📦 Product Analytics":
    render_product_analytics(data)
elif page == "🎯 Churn Prediction":
    render_churn_prediction(data)
elif page == "📈 Demand Forecasting":
    render_forecasting(data)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #b0b0b0; padding: 2rem;'>
    <p>E-Commerce Intelligence & Decision Analytics Platform</p>
    <p style='font-size: 0.8rem;'>Built with Streamlit, Plotly, and Python</p>
</div>
""", unsafe_allow_html=True)
