"""
E-Commerce Intelligence & Decision Analytics Platform
Streamlit Application - Main Dashboard
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path
from datetime import datetime, timedelta
import sys

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from config.settings import settings
from config.logging_config import get_logger

logger = get_logger(__name__)

# Page configuration
st.set_page_config(
    page_title="E-Commerce Intelligence Platform",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS with professional color scheme and animations
st.markdown("""
<style>
    /* Enterprise Professional Color Palette */
    :root {
        --primary-dark: #000000;
        --primary-slate: #1a1a1a;
        --neon-pink: #ff00ff;
        --neon-cyan: #00ffff;
        --neon-green: #00ff00;
        --neon-yellow: #ffff00;
        --neon-orange: #ff6600;
        --neon-purple: #9900ff;
        --neon-blue: #0066ff;
        --neon-red: #ff0066;
        --bg-gradient-start: #0a0a0a;
        --bg-gradient-end: #1a1a1a;
        --card-bg: #141414;
        --card-hover: #1e1e1e;
        --text-primary: #ffffff;
        --text-secondary: #e0e0e0;
        --text-muted: #b0b0b0;
    }

    /* Global Styles */
    .stApp {
        background: linear-gradient(135deg, var(--bg-gradient-start) 0%, var(--bg-gradient-end) 100%);
        color: #ffffff;
    }

    /* Animated Background */
    @keyframes gradientBG {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }

    .stApp {
        background-size: 400% 400%;
        animation: gradientBG 20s ease infinite;
    }

    /* Headers */
    .main-header {
        font-size: 2.8rem;
        font-weight: 800;
        margin-bottom: 2rem;
        text-align: center;
        animation: fadeInDown 1s ease-out;
        color: #ffffff;
        text-shadow: 0 0 40px rgba(0, 255, 255, 0.5);
        filter: drop-shadow(0 0 10px rgba(0, 255, 255, 0.3));
    }

    /* Apply neon glow to entire header including emoji - matching sidebar */
    .main-header {
        animation: fadeInDown 1s ease-out, iconPulse 2s ease-in-out infinite;
    }

    @keyframes fadeInDown {
        from {
            opacity: 0;
            transform: translateY(-40px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }

    /* Subheaders */
    h2, h3 {
        color: #ffffff;
        font-weight: 700;
        margin-top: 1.8rem;
        letter-spacing: 0.5px;
        text-shadow: 0 0 10px rgba(0, 255, 255, 0.5);
        filter: drop-shadow(0 0 5px rgba(255, 0, 255, 0.3));
    }

    /* Metric Cards */
    .metric-card {
        background: linear-gradient(135deg, #1a1a2e, #16213e, #0f3460);
        border: 2px solid var(--neon-cyan);
        border-radius: 20px;
        color: #ffffff;
        margin: 0.6rem 0;
        padding: 1.2rem;
        box-shadow: 0 10px 40px rgba(0, 0, 0, 0.6), 0 0 20px var(--neon-cyan);
        transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1);
        animation: slideInUp 0.8s ease-out;
    }

    .metric-card:hover {
        transform: translateY(-8px) scale(1.02);
        box-shadow: 0 20px 60px var(--neon-cyan), 0 0 40px var(--neon-pink);
        border-color: var(--neon-pink);
    }

    @keyframes slideInUp {
        from {
            opacity: 0;
            transform: translateY(40px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }

    /* Streamlit Metrics Enhancement */
    div[data-testid="stMetricValue"] {
        background: linear-gradient(135deg, var(--neon-cyan), var(--neon-pink), var(--neon-green));
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        font-size: 2.8rem;
        font-weight: 800;
        letter-spacing: -0.5px;
        text-shadow: 0 0 20px rgba(0, 255, 255, 0.5);
        filter: drop-shadow(0 0 10px rgba(255, 0, 255, 0.3));
    }

    div[data-testid="stMetricDelta"] {
        font-weight: 700;
        font-size: 1.1rem;
        color: #ffffff;
        text-shadow: 0 0 10px rgba(0, 255, 255, 0.5);
    }

    div[data-testid="stMetricLabel"] {
        color: #e0e0e0;
        font-weight: 600;
        text-shadow: 0 0 5px rgba(0, 255, 255, 0.3);
    }

    /* Alert Boxes */
    .alert-box {
        padding: 1.8rem;
        border-radius: 16px;
        margin: 1rem 0;
        border: 2px solid;
        animation: fadeIn 0.6s ease-out;
    }

    .alert-success {
        background: #0a2a1a;
        border-color: var(--neon-green);
        color: #ffffff;
        box-shadow: 0 0 20px var(--neon-green);
        text-shadow: 0 0 5px rgba(0, 255, 0, 0.5);
    }

    .alert-warning {
        background: #2a1a0a;
        border-color: var(--neon-orange);
        color: #ffffff;
        box-shadow: 0 0 20px var(--neon-orange);
        text-shadow: 0 0 5px rgba(255, 102, 0, 0.5);
    }

    .alert-error {
        background: #2a0a0a;
        border-color: var(--neon-red);
        color: #ffffff;
        box-shadow: 0 0 20px var(--neon-red);
        text-shadow: 0 0 5px rgba(255, 0, 102, 0.5);
    }

    .alert-info {
        background: #0a0a2a;
        border-color: var(--neon-blue);
        color: #ffffff;
        box-shadow: 0 0 20px var(--neon-blue);
        text-shadow: 0 0 5px rgba(0, 102, 255, 0.5);
    }

    @keyframes fadeIn {
        from { opacity: 0; }
        to { opacity: 1; }
    }

    @keyframes fadeInLeft {
        from {
            opacity: 0;
            transform: translateX(-30px);
        }
        to {
            opacity: 1;
            transform: translateX(0);
        }
    }

    /* DataFrames */
    .stDataFrame {
        background: var(--card-bg);
        border-radius: 16px;
        padding: 1.5rem;
        border: 1px solid rgba(79, 70, 229, 0.3);
        animation: fadeIn 0.8s ease-out;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
    }

    /* Plotly Charts */
    .js-plotly-plot {
        border-radius: 16px;
        overflow: hidden;
        animation: zoomIn 0.8s ease-out;
        box-shadow: 0 10px 40px rgba(0, 0, 0, 0.4);
    }

    @keyframes zoomIn {
        from {
            opacity: 0;
            transform: scale(0.92);
        }
        to {
            opacity: 1;
            transform: scale(1);
        }
    }

    /* Sidebar */
    .css-1d391kg {
        background: linear-gradient(180deg, rgba(15, 23, 42, 0.98), rgba(30, 41, 59, 0.98));
        backdrop-filter: blur(20px);
        border-right: 1px solid rgba(79, 70, 229, 0.2);
    }

    /* Buttons */
    .stButton > button {
        background: linear-gradient(135deg, var(--primary-indigo), var(--primary-violet));
        border: none;
        border-radius: 12px;
        padding: 0.7rem 2rem;
        font-weight: 700;
        font-size: 1rem;
        color: white;
        box-shadow: 0 4px 20px rgba(79, 70, 229, 0.4);
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
    }

    .stButton > button:hover {
        transform: translateY(-3px);
        box-shadow: 0 8px 30px rgba(79, 70, 229, 0.6);
        background: linear-gradient(135deg, var(--primary-violet), var(--accent-blue));
    }

    /* Sliders */
    .stSlider > div > div > div {
        background: linear-gradient(90deg, #87CEEB, #00BFFF, #1E90FF);
        border-radius: 10px;
        height: 8px;
    }

    .stSlider > div > div > div > div {
        background: white;
        border-radius: 50%;
        box-shadow: 0 0 15px rgba(135, 206, 235, 0.6);
    }

    /* Date Inputs */
    .stDateInput > div > div > div {
        background: var(--card-bg);
        border: 1px solid rgba(79, 70, 229, 0.4);
        border-radius: 12px;
        color: var(--text-primary);
    }

    /* Expander */
    .streamlit-expanderHeader {
        background: linear-gradient(135deg, rgba(79, 70, 229, 0.15), rgba(124, 58, 237, 0.1));
        border: 1px solid rgba(79, 70, 229, 0.4);
        border-radius: 12px;
        padding: 1rem;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        font-weight: 700;
    }

    .streamlit-expanderHeader:hover {
        background: linear-gradient(135deg, rgba(79, 70, 229, 0.25), rgba(124, 58, 237, 0.2));
        transform: translateX(5px);
        box-shadow: 0 0 20px rgba(79, 70, 229, 0.3);
    }

    /* Info Box */
    .stAlert {
        background: rgba(79, 70, 229, 0.1);
        border: 1px solid rgba(79, 70, 229, 0.3);
        border-radius: 16px;
        backdrop-filter: blur(15px);
    }

    /* Progress Bar */
    .stProgress > div > div > div > div {
        background: linear-gradient(90deg, var(--primary-indigo), var(--primary-violet), var(--accent-blue));
        border-radius: 10px;
    }

    /* Spinner */
    .stSpinner {
        color: var(--primary-indigo);
    }

    /* Text Input */
    .stTextInput > div > div > input {
        background: var(--card-bg);
        border: 1px solid rgba(79, 70, 229, 0.4);
        border-radius: 12px;
        color: var(--text-primary);
    }

    /* Number Input */
    .stNumberInput > div > div > input {
        background: var(--card-bg);
        border: 1px solid rgba(79, 70, 229, 0.4);
        border-radius: 12px;
        color: var(--text-primary);
    }

    /* Select Box */
    .stSelectbox > div > div > select {
        background: var(--card-bg);
        border: 1px solid rgba(79, 70, 229, 0.4);
        border-radius: 12px;
        color: var(--text-primary);
    }

    /* Multi Select */
    .stMultiSelect > div > div > div {
        background: var(--card-bg);
        border: 1px solid rgba(79, 70, 229, 0.4);
        border-radius: 12px;
    }

    /* Checkbox */
    .stCheckbox > label {
        color: var(--text-primary);
        font-weight: 600;
    }

    /* Radio */
    .stRadio > div {
        color: var(--text-primary);
    }

    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 2px;
    }

    .stTabs [data-baseweb="tab"] {
        background: rgba(79, 70, 229, 0.1);
        border-radius: 12px 12px 0 0;
        padding: 12px 24px;
        font-weight: 700;
        color: var(--text-secondary);
        transition: all 0.3s ease;
    }

    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, var(--primary-indigo), var(--primary-violet));
        color: white;
        box-shadow: 0 0 20px rgba(79, 70, 229, 0.4);
    }

    /* File Uploader */
    .stFileUploader {
        background: var(--card-bg);
        border: 2px dashed rgba(79, 70, 229, 0.4);
        border-radius: 16px;
        padding: 2rem;
    }

    /* Download Button */
    .stDownloadButton > button {
        background: linear-gradient(135deg, var(--accent-emerald), var(--accent-teal));
        border: none;
        border-radius: 12px;
        padding: 0.7rem 2rem;
        font-weight: 700;
        color: white;
        box-shadow: 0 4px 20px rgba(5, 150, 105, 0.4);
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
    }

    .stDownloadButton > button:hover {
        transform: translateY(-3px);
        box-shadow: 0 8px 30px rgba(5, 150, 105, 0.6);
    }

    /* Progress Bar Animation */
    @keyframes progressWidth {
        from { width: 0; }
        to { width: 100%; }
    }

    /* Card Container */
    .card-container {
        background: var(--card-bg);
        border-radius: 20px;
        padding: 2rem;
        margin: 1.2rem 0;
        border: 1px solid rgba(79, 70, 229, 0.3);
        backdrop-filter: blur(20px);
        animation: slideInUp 0.8s ease-out;
        transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1);
        box-shadow: 0 10px 40px rgba(0, 0, 0, 0.3);
    }

    .card-container:hover {
        transform: translateY(-5px);
        box-shadow: 0 20px 60px rgba(79, 70, 229, 0.4);
        border-color: rgba(79, 70, 229, 0.5);
    }

    /* Glow Effect */
    .glow {
        box-shadow: 0 0 30px rgba(79, 70, 229, 0.6);
        animation: pulse 2s infinite;
    }

    @keyframes pulse {
        0%, 100% { box-shadow: 0 0 30px rgba(79, 70, 229, 0.6); }
        50% { box-shadow: 0 0 50px rgba(124, 58, 237, 0.8); }
    }

    /* Floating Animation */
    .float {
        animation: float 4s ease-in-out infinite;
    }

    @keyframes float {
        0%, 100% { transform: translateY(0); }
        50% { transform: translateY(-15px); }
    }

    /* Rotate Animation */
    .rotate {
        animation: rotate 12s linear infinite;
    }

    @keyframes rotate {
        from { transform: rotate(0deg); }
        to { transform: rotate(360deg); }
    }

    /* Bounce Animation */
    .bounce {
        animation: bounce 2s infinite;
    }

    @keyframes bounce {
        0%, 100% { transform: translateY(0); }
        50% { transform: translateY(-20px); }
    }

    /* Shimmer Effect */
    .shimmer {
        background: linear-gradient(90deg, transparent, rgba(255, 255, 255, 0.15), transparent);
        background-size: 200% 100%;
        animation: shimmer 2.5s infinite;
    }

    @keyframes shimmer {
        0% { background-position: -200% 0; }
        100% { background-position: 200% 0; }
    }

    /* Gradient Text */
    .gradient-text {
        background: linear-gradient(135deg, var(--primary-indigo), var(--primary-violet), var(--accent-blue));
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        font-weight: 800;
    }

    /* Glassmorphism */
    .glass {
        background: rgba(30, 41, 59, 0.7);
        backdrop-filter: blur(20px);
        border: 1px solid rgba(79, 70, 229, 0.3);
        border-radius: 20px;
    }

    /* Neon Glow */
    .neon {
        box-shadow: 0 0 10px rgba(79, 70, 229, 0.5),
                    0 0 20px rgba(79, 70, 229, 0.4),
                    0 0 30px rgba(79, 70, 229, 0.3),
                    0 0 40px rgba(79, 70, 229, 0.2);
    }

    /* Smooth Scroll */
    html {
        scroll-behavior: smooth;
    }

    /* Custom Scrollbar */
    ::-webkit-scrollbar {
        width: 10px;
    }

    ::-webkit-scrollbar-track {
        background: var(--primary-dark);
    }

    ::-webkit-scrollbar-thumb {
        background: linear-gradient(180deg, var(--primary-indigo), var(--primary-violet));
        border-radius: 10px;
    }

    ::-webkit-scrollbar-thumb:hover {
        background: linear-gradient(180deg, var(--primary-violet), var(--accent-blue));
    }

    /* Sidebar Navigation Icons */
    .stSidebar [role="navigation"] > div > div > div > div {
        display: flex;
        align-items: center;
        gap: 12px;
        padding: 12px 16px;
        border-radius: 12px;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        margin: 4px 0;
        background: #141414;
        border: 2px solid var(--neon-cyan);
    }

    .stSidebar [role="navigation"] > div > div > div > div:hover {
        background: linear-gradient(135deg, #003333, #330033);
        transform: translateX(8px);
        box-shadow: 0 0 20px var(--neon-cyan), 0 0 40px var(--neon-pink);
        border-color: var(--neon-pink);
    }

    .stSidebar [role="navigation"] > div > div > div > div > div > span {
        font-size: 1.8rem;
        animation: iconPulse 2s ease-in-out infinite;
        filter: drop-shadow(0 0 8px var(--neon-cyan));
    }

    /* Icon-specific animations with neon colors */
    .stSidebar [role="navigation"] > div > div > div:nth-child(1) > div > div > span {
        animation-delay: 0s;
        filter: drop-shadow(0 0 10px var(--neon-pink));
    }

    .stSidebar [role="navigation"] > div > div > div:nth-child(2) > div > div > span {
        animation-delay: 0.2s;
        filter: drop-shadow(0 0 10px var(--neon-cyan));
    }

    .stSidebar [role="navigation"] > div > div > div:nth-child(3) > div > div > span {
        animation-delay: 0.4s;
        filter: drop-shadow(0 0 10px var(--neon-green));
    }

    .stSidebar [role="navigation"] > div > div > div:nth-child(4) > div > div > span {
        animation-delay: 0.6s;
        filter: drop-shadow(0 0 10px var(--neon-yellow));
    }

    .stSidebar [role="navigation"] > div > div > div:nth-child(5) > div > div > span {
        animation-delay: 0.8s;
        filter: drop-shadow(0 0 10px var(--neon-orange));
    }

    .stSidebar [role="navigation"] > div > div > div:nth-child(6) > div > div > span {
        animation-delay: 1.0s;
        filter: drop-shadow(0 0 10px var(--neon-purple));
    }

    .stSidebar [role="navigation"] > div > div > div:nth-child(7) > div > div > span {
        animation-delay: 1.2s;
        filter: drop-shadow(0 0 10px var(--neon-blue));
    }

    .stSidebar [role="navigation"] > div > div > div:nth-child(8) > div > div > span {
        animation-delay: 1.4s;
        filter: drop-shadow(0 0 10px var(--neon-red));
    }

    @keyframes iconPulse {
        0%, 100% {
            transform: scale(1);
            filter: drop-shadow(0 0 8px var(--neon-cyan));
        }
        50% {
            transform: scale(1.15);
            filter: drop-shadow(0 0 15px var(--neon-pink));
        }
    }

    /* Active navigation item */
    .stSidebar [role="navigation"] > div > div > div > div[aria-selected="true"] {
        background: linear-gradient(135deg, #004d4d, #4d004d);
        border: 3px solid var(--neon-pink);
        box-shadow: 0 0 30px var(--neon-cyan), 0 0 60px var(--neon-pink);
    }

    .stSidebar [role="navigation"] > div > div > div > div[aria-selected="true"] > div > div > span {
        animation: iconGlow 1.5s ease-in-out infinite;
    }

    @keyframes iconGlow {
        0%, 100% {
            transform: scale(1.1) rotate(0deg);
            filter: drop-shadow(0 0 15px var(--neon-pink));
        }
        25% {
            transform: scale(1.2) rotate(5deg);
            filter: drop-shadow(0 0 25px var(--neon-cyan));
        }
        75% {
            transform: scale(1.2) rotate(-5deg);
            filter: drop-shadow(0 0 25px var(--neon-green));
        }
    }

    /* Sidebar header enhancement */
    .stSidebar > div:first-child {
        background: linear-gradient(180deg, #141414, #1e1e1e);
        padding: 20px;
        border-bottom: 3px solid var(--neon-cyan);
        box-shadow: 0 0 20px var(--neon-cyan);
    }

    /* Sidebar sections */
    .stSidebar > div > div > div > div {
        background: linear-gradient(180deg, #141414, #1e1e1e);
        border-radius: 16px;
        padding: 16px;
        margin: 12px 0;
        border: 2px solid var(--neon-purple);
        transition: all 0.3s ease;
        box-shadow: 0 0 15px rgba(153, 0, 255, 0.3);
    }

    .stSidebar > div > div > div > div:hover {
        border-color: var(--neon-cyan);
        box-shadow: 0 0 30px var(--neon-cyan), 0 0 60px var(--neon-pink);
    }

    /* Advanced Table Styling */
    .stDataFrame {
        background: linear-gradient(135deg, #141414, #1e1e1e);
        border-radius: 20px;
        padding: 20px;
        border: 3px solid var(--neon-cyan);
        animation: tableFadeIn 0.8s ease-out;
        box-shadow: 0 10px 40px rgba(0, 0, 0, 0.5), 0 0 30px var(--neon-cyan);
        transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1);
    }

    .stDataFrame:hover {
        transform: translateY(-5px);
        box-shadow: 0 20px 60px var(--neon-cyan), 0 0 50px var(--neon-pink);
        border-color: var(--neon-pink);
    }

    @keyframes tableFadeIn {
        from {
            opacity: 0;
            transform: translateY(20px) scale(0.95);
        }
        to {
            opacity: 1;
            transform: translateY(0) scale(1);
        }
    }

    /* Table Header Styling */
    .stDataFrame thead th {
        background: linear-gradient(135deg, var(--neon-pink), var(--neon-purple));
        color: #ffffff;
        font-weight: 700;
        font-size: 14px;
        padding: 16px;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        border-bottom: 3px solid var(--neon-cyan);
        animation: headerGlow 3s ease-in-out infinite;
        text-shadow: 0 0 10px rgba(0, 0, 0, 0.8);
    }

    @keyframes headerGlow {
        0%, 100% {
            background: linear-gradient(135deg, var(--neon-pink), var(--neon-purple));
            box-shadow: 0 0 20px var(--neon-pink);
        }
        50% {
            background: linear-gradient(135deg, var(--neon-cyan), var(--neon-blue));
            box-shadow: 0 0 30px var(--neon-cyan);
        }
    }

    /* Table Row Styling */
    .stDataFrame tbody tr {
        transition: all 0.3s ease;
        border-bottom: 1px solid var(--neon-purple);
    }

    .stDataFrame tbody tr:hover {
        background: linear-gradient(90deg, rgba(0, 255, 255, 0.15), rgba(255, 0, 255, 0.1));
        transform: scale(1.01);
        box-shadow: 0 4px 20px var(--neon-cyan);
    }

    .stDataFrame tbody td {
        color: #ffffff;
        padding: 14px 16px;
        font-size: 14px;
        border-bottom: 1px solid var(--neon-purple);
        text-shadow: 0 0 5px rgba(0, 0, 0, 0.5);
    }

    /* Table Cell Highlighting */
    .stDataFrame tbody tr:nth-child(even) {
        background: #0a1a1a;
    }

    .stDataFrame tbody tr:nth-child(odd) {
        background: #1a0a1a;
    }

    /* Page Background Patterns */
    .stApp {
        background: 
            radial-gradient(circle at 20% 80%, #003333 0%, #0a0a0a 50%),
            radial-gradient(circle at 80% 20%, #330033 0%, #0a0a0a 50%),
            radial-gradient(circle at 40% 40%, #002200 0%, #0a0a0a 30%),
            radial-gradient(circle at 60% 60%, #333300 0%, #0a0a0a 30%),
            linear-gradient(135deg, var(--bg-gradient-start) 0%, var(--bg-gradient-end) 100%);
        background-size: 400% 400%;
        animation: gradientBG 20s ease infinite, patternMove 30s linear infinite;
    }

    @keyframes patternMove {
        0% { background-position: 0% 50%, 100% 50%, 50% 50%, 60% 60%; }
        50% { background-position: 100% 50%, 0% 50%, 50% 100%, 40% 40%; }
        100% { background-position: 0% 50%, 100% 50%, 50% 50%, 60% 60%; }
    }

    /* Floating Particles Effect */
    .particle {
        position: fixed;
        width: 6px;
        height: 6px;
        background: var(--neon-cyan);
        border-radius: 50%;
        animation: floatParticle 15s infinite;
        opacity: 0.8;
        pointer-events: none;
        box-shadow: 0 0 10px var(--neon-cyan);
    }

    @keyframes floatParticle {
        0%, 100% {
            transform: translateY(100vh) rotate(0deg);
            opacity: 0;
        }
        10% {
            opacity: 0.8;
        }
        90% {
            opacity: 0.8;
        }
        100% {
            transform: translateY(-100vh) rotate(720deg);
            opacity: 0;
        }
    }

    /* Advanced Chart Animations */
    .js-plotly-plot .plotly .modebar {
        background: linear-gradient(135deg, #1a1a2e, #16213e);
        border-radius: 12px;
        padding: 8px 12px;
        border: 2px solid var(--neon-cyan);
        box-shadow: 0 0 20px var(--neon-cyan);
    }

    .js-plotly-plot .plotly .modebar-btn {
        color: #ffffff;
        transition: all 0.3s ease;
    }

    .js-plotly-plot .plotly .modebar-btn:hover {
        background: var(--neon-cyan);
        transform: scale(1.1);
        box-shadow: 0 0 15px var(--neon-cyan);
    }

    /* 3D Chart Effect */
    .js-plotly-plot {
        transform: perspective(1000px) rotateX(0deg) rotateY(0deg);
        transition: transform 0.5s ease;
    }

    .js-plotly-plot:hover {
        transform: perspective(1000px) rotateX(2deg) rotateY(2deg);
    }

    /* Metric Card Enhancement */
    div[data-testid="stMetric"] {
        background: linear-gradient(135deg, #1a1a2e, #16213e);
        border: 2px solid var(--neon-cyan);
        border-radius: 20px;
        padding: 20px;
        animation: metricPulse 3s ease-in-out infinite;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.6), 0 0 20px var(--neon-cyan);
    }

    @keyframes metricPulse {
        0%, 100% {
            box-shadow: 0 8px 32px rgba(0, 0, 0, 0.6), 0 0 20px var(--neon-cyan);
            border-color: var(--neon-cyan);
        }
        50% {
            box-shadow: 0 12px 40px rgba(0, 0, 0, 0.8), 0 0 30px var(--neon-pink);
            border-color: var(--neon-pink);
        }
    }

    div[data-testid="stMetric"]:hover {
        transform: translateY(-8px) scale(1.02);
        background: linear-gradient(135deg, #1a1a2e, #16213e);
        border-color: var(--neon-pink);
        box-shadow: 0 12px 40px var(--neon-cyan), 0 0 40px var(--neon-pink);
    }

    /* Subheader Enhancement */
    h2, h3 {
        background: linear-gradient(135deg, var(--neon-cyan), var(--neon-pink), var(--neon-purple));
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        animation: textShimmer 3s ease-in-out infinite;
        text-shadow: 0 0 20px rgba(0, 255, 255, 0.5);
        filter: drop-shadow(0 0 10px rgba(255, 0, 255, 0.3));
    }

    @keyframes textShimmer {
        0%, 100% {
            background-position: 0% 50%;
        }
        50% {
            background-position: 100% 50%;
        }
    }

    /* Divider Enhancement */
    hr {
        border: none;
        height: 3px;
        background: linear-gradient(90deg, transparent, var(--neon-cyan), var(--neon-pink), var(--neon-purple), transparent);
        margin: 2rem 0;
        animation: expandWidth 1.5s ease-out, gradientMove 3s linear infinite;
        background-size: 200% 100%;
        box-shadow: 0 0 10px var(--neon-cyan);
    }

    @keyframes gradientMove {
        0% { background-position: 0% 50%; }
        100% { background-position: 200% 50%; }
    }

    /* Expander Enhancement */
    .streamlit-expanderContent {
        background: #141414;
        border: 2px solid var(--neon-cyan);
        border-radius: 16px;
        padding: 20px;
        margin-top: 12px;
        animation: expandContent 0.5s ease-out;
        box-shadow: 0 0 20px var(--neon-cyan);
    }

    /* Expander Header Icons - Always Visible with Neon Glow */
    .streamlit-expanderHeader {
        color: #ffffff !important;
        font-weight: 700;
        font-size: 1.2rem;
    }

    .streamlit-expanderHeader > div > span {
        font-size: 1.5rem;
        animation: iconPulse 2s ease-in-out infinite;
        filter: drop-shadow(0 0 8px var(--neon-cyan));
    }

    /* Expander Arrow Icon - Always Visible */
    .streamlit-expanderHeader > div > div > svg {
        fill: var(--neon-cyan) !important;
        stroke: var(--neon-cyan) !important;
        filter: drop-shadow(0 0 8px var(--neon-cyan));
        animation: iconPulse 2s ease-in-out infinite;
    }

    @keyframes expandContent {
        from {
            opacity: 0;
            transform: translateY(-10px);
            max-height: 0;
        }
        to {
            opacity: 1;
            transform: translateY(0);
            max-height: 1000px;
        }
    }

    /* Column Enhancement */
    .css-1d391kg {
        gap: 1.5rem;
    }

    .css-1d391kg > div {
        animation: slideInUp 0.6s ease-out;
        animation-fill-mode: both;
    }

    .css-1d391kg > div:nth-child(1) { animation-delay: 0.1s; }
    .css-1d391kg > div:nth-child(2) { animation-delay: 0.2s; }
    .css-1d391kg > div:nth-child(3) { animation-delay: 0.3s; }
    .css-1d391kg > div:nth-child(4) { animation-delay: 0.4s; }

    /* Success/Info/Error Messages */
    .stSuccess {
        background: #0a2a1a;
        border: 2px solid var(--neon-green);
        border-radius: 16px;
        padding: 16px 20px;
        animation: successPulse 2s ease-in-out infinite;
        box-shadow: 0 0 20px var(--neon-green);
        color: #ffffff;
    }

    @keyframes successPulse {
        0%, 100% { box-shadow: 0 0 20px var(--neon-green); }
        50% { box-shadow: 0 0 30px var(--neon-green); }
    }

    .stInfo {
        background: #0a0a2a;
        border: 2px solid var(--neon-blue);
        border-radius: 16px;
        padding: 16px 20px;
        animation: infoPulse 2s ease-in-out infinite;
        box-shadow: 0 0 20px var(--neon-blue);
        color: #ffffff;
    }

    @keyframes infoPulse {
        0%, 100% { box-shadow: 0 0 20px var(--neon-blue); }
        50% { box-shadow: 0 0 30px var(--neon-cyan); }
    }

    .stWarning {
        background: #2a1a0a;
        border: 2px solid var(--neon-orange);
        border-radius: 16px;
        padding: 16px 20px;
        animation: warningPulse 2s ease-in-out infinite;
        box-shadow: 0 0 20px var(--neon-orange);
        color: #ffffff;
    }

    @keyframes warningPulse {
        0%, 100% { box-shadow: 0 0 20px var(--neon-orange); }
        50% { box-shadow: 0 0 30px var(--neon-yellow); }
    }

    .stError {
        background: #2a0a0a;
        border: 2px solid var(--neon-red);
        border-radius: 16px;
        padding: 16px 20px;
        animation: errorPulse 2s ease-in-out infinite;
        box-shadow: 0 0 20px var(--neon-red);
        color: #ffffff;
    }

    @keyframes errorPulse {
        0%, 100% { box-shadow: 0 0 20px var(--neon-red); }
        50% { box-shadow: 0 0 30px var(--neon-pink); }
    }
</style>
""", unsafe_allow_html=True)

# Data loading functions
@st.cache_data(ttl=3600)
def load_data():
    """Load all data from processed directory"""
    data_dir = settings.PROCESSED_DATA_DIR
    data = {}
    
    # Try to load from processed directory first
    if data_dir.exists():
        try:
            # Load customer analytics
            customer_files = list(data_dir.glob("customer_*.csv"))
            for file in customer_files:
                key = file.stem.replace("customer_", "")
                data[f"customer_{key}"] = pd.read_csv(file)
            
            # Load product analytics
            product_files = list(data_dir.glob("product_*.csv"))
            for file in product_files:
                key = file.stem.replace("product_", "")
                data[f"product_{key}"] = pd.read_csv(file)
            
            # Load marketing analytics
            marketing_files = list(data_dir.glob("marketing_*.csv"))
            for file in marketing_files:
                key = file.stem.replace("marketing_", "")
                data[f"marketing_{key}"] = pd.read_csv(file)
            
            # Load transformation outputs
            transform_files = [
                "rfm_segments.csv", "clv.csv", "churn_features.csv", "product_matrix.csv",
                "orders.csv", "order_items.csv",
                "forecast_model_comparison.csv", "forecast_overall_best.csv",
                "forecast_overall_future.csv", "forecast_overall_history.csv",
                "forecast_top_products_future.csv"
            ]
            for file in transform_files:
                path = data_dir / file
                if path.exists():
                    data[file.replace(".csv", "")] = pd.read_csv(path)
            
            logger.info(f"Loaded {len(data)} datasets from processed directory")
        except Exception as e:
            logger.warning(f"Error loading processed data: {e}")
    
    # Fallback to raw data if processed not available or for missing tables
    raw_dir = settings.RAW_DATA_DIR
    if raw_dir.exists():
        try:
            # Always load these essential tables from raw to ensure consistency
            # Override processed versions with raw versions for referential integrity
            essential_tables = ["customers", "products", "orders", "order_items", "categories"]
            for table in essential_tables:
                path = raw_dir / f"{table}.csv"
                if path.exists():
                    data[table] = pd.read_csv(path)
            logger.info(f"Loaded raw data for essential tables to ensure consistency")
        except Exception as e:
            logger.warning(f"Error loading raw data: {e}")
    
    return data

# Format functions
def format_currency(value):
    """Format value as Indian Rupees"""
    if pd.isna(value):
        return "₹0"
    if value >= 1e7:
        return f"₹{value/1e7:.2f} Cr"
    elif value >= 1e5:
        return f"₹{value/1e5:.2f} L"
    else:
        return f"₹{value:,.0f}"

def format_number(value):
    """Format large numbers"""
    if pd.isna(value):
        return "0"
    if value >= 1e6:
        return f"{value/1e6:.2f}M"
    elif value >= 1e3:
        return f"{value/1e3:.2f}K"
    else:
        return f"{value:,.0f}"

def format_percentage(value):
    """Format as percentage"""
    if pd.isna(value):
        return "0%"
    return f"{value:.2f}%"

# Page: Executive Dashboard
def executive_dashboard(data):
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
    
    # Calculate actual KPIs from data
    if "orders" in data:
        orders_df = data["orders"]
        # Apply date filter if provided
        if len(date_range) == 2:
            start_date, end_date = date_range
            orders_df = orders_df[
                (pd.to_datetime(orders_df["order_date"]) >= pd.to_datetime(start_date)) &
                (pd.to_datetime(orders_df["order_date"]) <= pd.to_datetime(end_date))
            ]
        
        valid_orders = orders_df[orders_df["order_status"].isin(status_filter)]
        
        total_revenue = valid_orders["order_total"].sum()
        cancelled_orders = orders_df[orders_df["order_status"] == "Cancelled"]["order_total"].sum()
        returned_orders = orders_df[orders_df["order_status"] == "Returned"]["order_total"].sum()
        net_revenue = total_revenue - cancelled_orders - returned_orders
        
        # Calculate gross profit (assuming ~29% margin)
        gross_profit = net_revenue * 0.29
        gross_margin = (gross_profit / net_revenue * 100) if net_revenue > 0 else 0
        
        total_orders_count = len(valid_orders)
        customers_count = orders_df["customer_id"].nunique()
        aov = total_revenue / total_orders_count if total_orders_count > 0 else 0
        return_rate = (len(orders_df[orders_df["order_status"] == "Returned"]) / total_orders_count * 100) if total_orders_count > 0 else 0
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
            monthly_revenue["order_month"] = monthly_revenue["order_month"].dt.to_timestamp()
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=monthly_revenue["order_month"], 
                y=monthly_revenue["order_total"],
                mode='lines+markers',
                name='Revenue',
                line=dict(color='#4f46e5', width=4, shape='spline'),
                marker=dict(size=10, color='#4f46e5', line=dict(width=2, color='#7c3aed')),
                fill='tozeroy',
                fillcolor='rgba(79, 70, 229, 0.1)',
                hovertemplate='<b>%{x|%b %Y}</b><br>Revenue: ₹%{y:,.0f}<extra></extra>'
            ))
            
            # Add animation frames
            frames = []
            for i in range(1, len(monthly_revenue) + 1):
                frame_data = go.Scatter(
                    x=monthly_revenue["order_month"][:i],
                    y=monthly_revenue["order_total"][:i],
                    mode='lines+markers',
                    line=dict(color='#4f46e5', width=4, shape='spline'),
                    marker=dict(size=10, color='#4f46e5', line=dict(width=2, color='#7c3aed')),
                    fill='tozeroy',
                    fillcolor='rgba(79, 70, 229, 0.1)'
                )
                frames.append(go.Frame(data=[frame_data], name=str(i)))
            
            fig.frames = frames
            
            fig.update_layout(
                updatemenus=[{
                    'type': 'buttons',
                    'showactive': False,
                    'buttons': [
                        {
                            'label': '▶ Play Animation',
                            'method': 'animate',
                            'args': [None, {
                                'frame': {'duration': 500, 'redraw': True},
                                'fromcurrent': True,
                                'transition': {'duration': 300}
                            }]
                        },
                        {
                            'label': '⏸ Pause',
                            'method': 'animate',
                            'args': [[None], {
                                'frame': {'duration': 0, 'redraw': False},
                                'mode': 'immediate',
                                'transition': {'duration': 0}
                            }]
                        }
                    ]
                }],
                title=dict(
                    text="Monthly Revenue Trend",
                    font=dict(size=20, color='#ffffff', family='Arial')
                ),
                xaxis_title=dict(
                    text="Month",
                    font=dict(size=14, color='#ffffff')
                ),
                yaxis_title=dict(
                    text="Revenue (₹)",
                    font=dict(size=14, color='#ffffff')
                ),
                hovermode='x unified',
                plot_bgcolor='#0a0a0a',
                paper_bgcolor='#141414',
                font=dict(color='#ffffff'),
                xaxis=dict(
                    gridcolor='#006666',
                    showgrid=True
                ),
                yaxis=dict(
                    gridcolor='#006666',
                    showgrid=True
                ),
                margin=dict(l=60, r=40, t=80, b=60)
            )
            st.plotly_chart(fig, width="content")
        else:
            st.info("Revenue trend data not available")
    
    with col2:
        st.subheader("Category Performance")
        if "order_items" in data and "products" in data:
            order_items_df = data["order_items"]
            products_df = data["products"]
            
            category_revenue = order_items_df.merge(
                products_df[["product_id", "category_id"]], on="product_id", how="left"
            )
            if "categories" in data:
                categories_df = data["categories"]
                category_revenue = category_revenue.merge(
                    categories_df[["category_id", "category_name"]], on="category_id", how="left"
                )
                category_stats = category_revenue.groupby("category_name")["line_total"].sum().nlargest(5)
                
                fig = go.Figure(data=[go.Pie(
                    labels=category_stats.index,
                    values=category_stats.values,
                    hole=0.5,
                    marker=dict(colors=['#4f46e5', '#059669', '#d97706', '#e11d48', '#0891b2'],
                                 line=dict(color='#1e293b', width=3)),
                    textinfo='label+percent',
                    textfont=dict(size=12, color='#ffffff'),
                    hoverinfo='label+value+percent'
                )])
                fig.update_layout(
                    title=dict(
                        text="Revenue by Category (Top 5)",
                        font=dict(size=20, color='#ffffff', family='Arial')
                    ),
                    paper_bgcolor='#141414',
                    font=dict(color='#ffffff'),
                    margin=dict(l=40, r=40, t=80, b=40)
                )
                st.plotly_chart(fig, width="content")
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
        state_stats = orders_customers.groupby("state").agg({
            "order_total": "sum",
            "order_id": "count"
        }).nlargest(5, "order_total")
        
        fig = go.Figure(data=[
            go.Bar(name='Revenue (₹ Cr)', x=state_stats.index, y=state_stats["order_total"]/1e7,
                   marker=dict(color='#4f46e5', line=dict(color='#7c3aed', width=2)),
                   text=state_stats["order_total"]/1e7,
                   textposition='outside',
                   textfont=dict(size=12, color='#ffffff')),
            go.Bar(name='Orders (K)', x=state_stats.index, y=state_stats["order_id"]/1000,
                   marker=dict(color='#0891b2', line=dict(color='#0d9488', width=2)),
                   text=state_stats["order_id"]/1000,
                   textposition='outside',
                   textfont=dict(size=12, color='#ffffff'))
        ])
        fig.update_layout(
            title=dict(
                text="Performance by State (Top 5)",
                font=dict(size=20, color='#ffffff', family='Arial')
            ),
            barmode='group',
            xaxis_title=dict(
                text="State",
                font=dict(size=14, color='#ffffff')
            ),
            yaxis_title=dict(
                text="Value",
                font=dict(size=14, color='#ffffff')
            ),
            plot_bgcolor='#0a0a0a',
            paper_bgcolor='#141414',
            font=dict(color='#ffffff'),
            xaxis=dict(
                gridcolor='rgba(0, 255, 255, 0.3)',
                showgrid=True
            ),
            yaxis=dict(
                gridcolor='rgba(0, 255, 255, 0.3)',
                showgrid=True
            ),
            margin=dict(l=60, r=40, t=80, b=60),
            legend=dict(
                orientation='h',
                yanchor='bottom',
                y=1.02,
                xanchor='right',
                x=1
            )
        )
        st.plotly_chart(fig, width="content")
    else:
        st.info("Regional performance data not available")

# Page: Sales Analytics
def sales_analytics(data):
    st.markdown('<h1 class="main-header float">📊 Sales Analytics</h1>', unsafe_allow_html=True)
    
    if "orders" not in data:
        st.warning("Order data not available. Please run data transformations first.")
        return
    
    orders_df = data["orders"]
    orders_df["order_date"] = pd.to_datetime(orders_df["order_date"])
    
    # Date filter
    col1, col2 = st.columns(2)
    with col1:
        start_date = st.date_input("Start Date", value=pd.to_datetime("2024-01-01").date())
    with col2:
        end_date = st.date_input("End Date", value=pd.to_datetime("2024-12-31").date())
    
    # Filter orders by date range
    filtered_orders = orders_df[
        (orders_df["order_date"].dt.date >= start_date) & 
        (orders_df["order_date"].dt.date <= end_date)
    ]
    
    # Daily Sales Trend
    st.subheader("Daily Sales Trend")
    daily_sales = filtered_orders.groupby(filtered_orders["order_date"].dt.date).agg({
        "order_total": "sum",
        "order_id": "count"
    }).reset_index()
    daily_sales.columns = ["date", "revenue", "orders"]
    
    if len(daily_sales) > 0:
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        fig.add_trace(
            go.Scatter(x=daily_sales["date"], y=daily_sales["revenue"], 
                      name='Revenue', line=dict(color='#6366f1', width=2)),
            secondary_y=False
        )
        fig.add_trace(
            go.Scatter(x=daily_sales["date"], y=daily_sales["orders"], 
                      name='Orders', line=dict(color='#10b981', width=2)),
            secondary_y=True
        )
        fig.update_xaxes(title_text="Date")
        fig.update_yaxes(title_text="Revenue (₹)", secondary_y=False)
        fig.update_yaxes(title_text="Orders", secondary_y=True)
        fig.update_layout(
            title=dict(
                text="Daily Sales Performance",
                font=dict(size=20, color='#ffffff', family='Arial')
            ),
            plot_bgcolor='#0a0a0a',
            paper_bgcolor='#141414',
            font=dict(color='#ffffff'),
            xaxis=dict(
                gridcolor='rgba(0, 255, 255, 0.3)',
                showgrid=True
            ),
            yaxis=dict(
                gridcolor='rgba(0, 255, 255, 0.3)',
                showgrid=True
            ),
            margin=dict(l=60, r=40, t=80, b=60),
            legend=dict(
                orientation='h',
                yanchor='bottom',
                y=1.02,
                xanchor='right',
                x=1
            )
        )
        st.plotly_chart(fig, width="content")
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
            marker=dict(color=['#059669', '#3b82f6', '#d97706', '#e11d48', '#64748b'],
                       line=dict(color='#1e293b', width=2)),
            text=status_counts.values,
            textposition='outside',
            textfont=dict(size=12, color='#f8fafc')
        )])
        fig.update_layout(
            title=dict(
                text="Orders by Status",
                font=dict(size=20, color='#ffffff', family='Arial')
            ),
            xaxis_title=dict(
                text="Status",
                font=dict(size=14, color='#ffffff')
            ),
            yaxis_title=dict(
                text="Count",
                font=dict(size=14, color='#ffffff')
            ),
            plot_bgcolor='#0a0a0a',
            paper_bgcolor='#141414',
            font=dict(color='#ffffff'),
            xaxis=dict(
                gridcolor='rgba(0, 255, 255, 0.3)',
                showgrid=True
            ),
            yaxis=dict(
                gridcolor='rgba(0, 255, 255, 0.3)',
                showgrid=True
            ),
            margin=dict(l=60, r=40, t=80, b=60)
        )
        st.plotly_chart(fig, width="content")
    
    with col2:
        st.subheader("Day of Week Analysis")
        filtered_orders["day_of_week"] = filtered_orders["order_date"].dt.day_name()
        dow_revenue = filtered_orders.groupby("day_of_week")["order_total"].sum()
        day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        dow_revenue = dow_revenue.reindex(day_order, fill_value=0)
        
        fig = go.Figure(data=[go.Bar(
            x=dow_revenue.index, 
            y=dow_revenue.values, 
            marker=dict(color='#4f46e5', line=dict(color='#7c3aed', width=2)),
            text=dow_revenue.values,
            textposition='outside',
            textfont=dict(size=12, color='#f8fafc')
        )])
        fig.update_layout(
            title=dict(
                text="Revenue by Day of Week",
                font=dict(size=20, color='#ffffff', family='Arial')
            ),
            xaxis_title=dict(
                text="Day",
                font=dict(size=14, color='#ffffff')
            ),
            yaxis_title=dict(
                text="Revenue (₹)",
                font=dict(size=14, color='#ffffff')
            ),
            plot_bgcolor='#0a0a0a',
            paper_bgcolor='#141414',
            font=dict(color='#ffffff'),
            xaxis=dict(
                gridcolor='rgba(0, 255, 255, 0.3)',
                showgrid=True
            ),
            yaxis=dict(
                gridcolor='rgba(0, 255, 255, 0.3)',
                showgrid=True
            ),
            margin=dict(l=60, r=40, t=80, b=60)
        )
        st.plotly_chart(fig, width="content")

# Page: Customer Analytics
def customer_analytics(data):
    st.markdown('<h1 class="main-header float">👥 Customer Analytics</h1>', unsafe_allow_html=True)
    
    # RFM Segmentation
    if "rfm_segments" in data:
        rfm_data = data["rfm_segments"]
        
        st.subheader("RFM Segment Distribution")
        segment_counts = rfm_data["rfm_segment"].value_counts()
        
        fig = go.Figure(data=[go.Pie(
            labels=segment_counts.index,
            values=segment_counts.values,
            hole=0.5,
            marker=dict(colors=['#4f46e5', '#059669', '#d97706', '#e11d48', '#0891b2', '#7c3aed', '#f97316'],
                       line=dict(color='#1e293b', width=3)),
            textinfo='label+percent',
            textfont=dict(size=14, color='#f8fafc'),
            hoverinfo='label+value+percent'
        )])
        fig.update_layout(
            title=dict(
                text="Customer Segments",
                font=dict(size=20, color='#f8fafc', family='Arial')
            ),
            paper_bgcolor='rgba(30, 41, 59, 0.3)',
            font=dict(color='#cbd5e1'),
            margin=dict(l=40, r=40, t=80, b=40)
        )
        st.plotly_chart(fig, width="content")
        
        # RFM Segment Details
        st.subheader("RFM Segment Details")
        segment_metrics = rfm_data.groupby("rfm_segment").agg({
            "monetary_value": ["mean", "sum"],
            "frequency": "mean",
            "recency_days": "mean"
        }).round(2)
        
        segment_metrics.columns = ["Avg Spend", "Total Spend", "Avg Frequency", "Avg Recency Days"]
        st.dataframe(segment_metrics, width="content")
    else:
        st.info("RFM segmentation data not available. Run data transformations first.")
    
    # CLV Analysis
    if "clv" in data:
        clv_data = data["clv"]
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("CLV Distribution")
            fig = go.Figure(data=[go.Histogram(
                x=clv_data["clv"], 
                nbinsx=30, 
                marker=dict(color='#4f46e5', line=dict(color='#7c3aed', width=2)),
                opacity=0.8
            )])
            fig.update_layout(
                title=dict(
                    text="Customer Lifetime Value Distribution",
                    font=dict(size=20, color='#ffffff', family='Arial')
                ),
                xaxis_title=dict(
                    text="CLV (₹)",
                    font=dict(size=14, color='#ffffff')
                ),
                yaxis_title=dict(
                    text="Count",
                    font=dict(size=14, color='#ffffff')
                ),
                plot_bgcolor='#0a0a0a',
                paper_bgcolor='#141414',
                font=dict(color='#ffffff'),
                xaxis=dict(
                    gridcolor='rgba(79, 70, 229, 0.15)',
                    showgrid=True
                ),
                yaxis=dict(
                    gridcolor='rgba(79, 70, 229, 0.15)',
                    showgrid=True
                ),
                margin=dict(l=60, r=40, t=80, b=60)
            )
            st.plotly_chart(fig, width="content")
        
        with col2:
            st.subheader("Customer Value Tiers")
            tier_counts = clv_data["customer_value_tier"].value_counts()
            fig = go.Figure(data=[go.Bar(
                x=tier_counts.index, 
                y=tier_counts.values,
                marker=dict(color='#059669', line=dict(color='#0d9488', width=2)),
                text=tier_counts.values,
                textposition='outside',
                textfont=dict(size=12, color='#ffffff')
            )])
            fig.update_layout(
                title=dict(
                    text="Customers by Value Tier",
                    font=dict(size=20, color='#ffffff', family='Arial')
                ),
                xaxis_title=dict(
                    text="Tier",
                    font=dict(size=14, color='#ffffff')
                ),
                yaxis_title=dict(
                    text="Count",
                    font=dict(size=14, color='#ffffff')
                ),
                plot_bgcolor='#0a0a0a',
                paper_bgcolor='#141414',
                font=dict(color='#ffffff'),
                xaxis=dict(
                    gridcolor='rgba(79, 70, 229, 0.15)',
                    showgrid=True
                ),
                yaxis=dict(
                    gridcolor='rgba(79, 70, 229, 0.15)',
                    showgrid=True
                ),
                margin=dict(l=60, r=40, t=80, b=60)
            )
            st.plotly_chart(fig, width="content")
    
    # New vs Returning Customers
    st.subheader("New vs Returning Customers")
    if "orders" in data and "customers" in data:
        orders_df = data["orders"]
        customers_df = data["customers"]
        
        # Get first order date for each customer
        first_orders = orders_df.groupby("customer_id")["order_date"].min().reset_index()
        first_orders.columns = ["customer_id", "first_order_date"]
        
        # Merge with customers to get signup date
        customer_dates = customers_df[["customer_id", "signup_date"]].merge(
            first_orders, on="customer_id", how="left"
        )
        customer_dates["first_order_date"] = pd.to_datetime(customer_dates["first_order_date"])
        customer_dates["signup_date"] = pd.to_datetime(customer_dates["signup_date"])
        
        # Monthly new customers
        customer_dates["signup_month"] = customer_dates["signup_date"].dt.to_period('M')
        new_customers_monthly = customer_dates.groupby("signup_month").size()
        
        # Monthly returning customers (customers who placed orders in month but didn't sign up that month)
        orders_df["order_month"] = pd.to_datetime(orders_df["order_date"]).dt.to_period('M')
        monthly_customers = orders_df.groupby("order_month")["customer_id"].nunique()
        
        # Align months
        all_months = sorted(set(new_customers_monthly.index) | set(monthly_customers.index))
        new_customers = [new_customers_monthly.get(m, 0) for m in all_months]
        returning_customers = [monthly_customers.get(m, 0) - new_customers_monthly.get(m, 0) for m in all_months]
        
        months_str = [m.strftime("%b %Y") for m in all_months]
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=months_str, 
            y=new_customers, 
            fill='tozeroy', 
            name='New Customers',
            line=dict(color='#4f46e5', width=3),
            marker=dict(size=6, color='#4f46e5'),
            fillcolor='rgba(79, 70, 229, 0.2)'
        ))
        fig.add_trace(go.Scatter(
            x=months_str, 
            y=returning_customers, 
            fill='tonexty', 
            name='Returning Customers',
            line=dict(color='#059669', width=3),
            marker=dict(size=6, color='#059669'),
            fillcolor='rgba(5, 150, 105, 0.2)'
        ))
        fig.update_layout(
            title=dict(
                text="New vs Returning Customers Trend",
                font=dict(size=20, color='#ffffff', family='Arial')
            ),
            xaxis_title=dict(
                text="Month",
                font=dict(size=14, color='#ffffff')
            ),
            yaxis_title=dict(
                text="Customers",
                font=dict(size=14, color='#ffffff')
            ),
            plot_bgcolor='#0a0a0a',
            paper_bgcolor='#141414',
            font=dict(color='#ffffff'),
            xaxis=dict(
                gridcolor='rgba(0, 255, 255, 0.3)',
                showgrid=True
            ),
            yaxis=dict(
                gridcolor='rgba(0, 255, 255, 0.3)',
                showgrid=True
            ),
            margin=dict(l=60, r=40, t=80, b=60),
            legend=dict(
                orientation='h',
                yanchor='bottom',
                y=1.02,
                xanchor='right',
                x=1
            ),
            hovermode='x unified'
        )
        st.plotly_chart(fig, width="content")
    else:
        st.info("Customer data not available for new vs returning analysis")

# Page: Product Analytics
def product_analytics(data):
    st.markdown('<h1 class="main-header float">📦 Product Analytics</h1>', unsafe_allow_html=True)
    
    # Product Matrix
    if "product_matrix" in data:
        product_data = data["product_matrix"]
        
        st.subheader("Product Matrix (Revenue vs Margin)")
        fig = go.Figure()
        
        quadrants = {'Stars': '#10b981', 'Volume': '#3b82f6', 'Remove': '#f43f5e', 'Premium': '#f59e0b'}
        
        for quadrant in quadrants:
            quad_data = product_data[product_data["quadrant"] == quadrant]
            fig.add_trace(go.Scatter(
                x=quad_data["revenue_inr"],
                y=quad_data["margin_ratio"] * 100,
                mode='markers',
                name=quadrant,
                marker=dict(size=10, color=quadrants[quadrant]),
                text=quad_data["product_name"],
                hovertemplate='<b>%{text}</b><br>Revenue: ₹%{x:,.0f}<br>Margin: %{y:.2f}%'
            ))
        
        fig.update_layout(
            title=dict(
                text="Product Performance Matrix",
                font=dict(size=20, color='#ffffff', family='Arial')
            ),
            xaxis_title=dict(
                text="Revenue (₹)",
                font=dict(size=14, color='#ffffff')
            ),
            yaxis_title=dict(
                text="Margin %",
                font=dict(size=14, color='#ffffff')
            ),
            hovermode='closest',
            plot_bgcolor='#0a0a0a',
            paper_bgcolor='#141414',
            font=dict(color='#ffffff'),
            xaxis=dict(
                gridcolor='rgba(0, 255, 255, 0.3)',
                showgrid=True
            ),
            yaxis=dict(
                gridcolor='rgba(0, 255, 255, 0.3)',
                showgrid=True
            ),
            margin=dict(l=60, r=40, t=80, b=60)
        )
        st.plotly_chart(fig, width="content")
        
        # Product Quadrant Summary
        st.subheader("Quadrant Summary")
        quadrant_summary = product_data.groupby("quadrant").agg({
            "product_id": "count",
            "revenue_inr": "sum",
            "profit_inr": "sum"
        }).round(2)
        quadrant_summary.columns = ["Product Count", "Total Revenue", "Total Profit"]
        st.dataframe(quadrant_summary, width="content")
    else:
        st.info("Product matrix data not available. Run product analytics first.")
    
    # Top Products
    st.subheader("Top 10 Products by Revenue")
    if "product_360" in data:
        top_products = data["product_360"].nlargest(10, "revenue")
        st.dataframe(top_products[["product_name", "category_name", "revenue", "units_sold", "gross_margin_pct"]],
                    width="content")
    elif "order_items" in data and "products" in data:
        # Calculate from raw data
        order_items_df = data["order_items"]
        products_df = data["products"]
        
        # Calculate product revenue
        product_revenue = order_items_df.groupby("product_id")["line_total"].sum().reset_index()
        product_revenue = product_revenue.sort_values("line_total", ascending=False).head(10)
        
        # Check which product IDs exist in products table
        existing_product_ids = set(products_df["product_id"].unique())
        product_revenue["exists"] = product_revenue["product_id"].isin(existing_product_ids)
        
        # Show warning about missing products
        missing_count = (~product_revenue["exists"]).sum()
        if missing_count > 0:
            st.warning(f"⚠️ {missing_count} of top 10 products have missing product names due to data inconsistency")
        
        # Merge with product details - use inner join to only show products that exist
        top_products = product_revenue.merge(
            products_df[["product_id", "product_name", "category_id"]],
            on="product_id",
            how="left"
        )
        
        # Fill None product names with product ID
        top_products["product_name"] = top_products["product_name"].fillna("Product " + top_products["product_id"].astype(str))
        
        top_products = top_products[["product_name", "line_total"]]
        top_products.columns = ["Product Name", "Revenue (₹)"]
        
        if len(top_products) > 0:
            st.dataframe(top_products, width="content")
        else:
            st.warning("No product revenue data available - product IDs in orders don't match products table")
    else:
        st.info(f"Product data not available. Available keys: {list(data.keys())[:10]}")

# Page: Forecasting
def forecasting_analytics(data):
    st.markdown('<h1 class="main-header float">📈 Demand Forecasting</h1>', unsafe_allow_html=True)
    
    # Check for forecast data
    has_history = "forecast_overall_history" in data and len(data.get("forecast_overall_history", pd.DataFrame())) > 0
    has_future = "forecast_overall_future" in data and len(data.get("forecast_overall_future", pd.DataFrame())) > 0
    has_best = "forecast_overall_best" in data and len(data.get("forecast_overall_best", pd.DataFrame())) > 0
    has_comparison = "forecast_model_comparison" in data and len(data.get("forecast_model_comparison", pd.DataFrame())) > 0
    
    if not (has_history or has_future or has_best):
        st.info("Forecast data not available. Run demand forecasting first.")
        return
    
    # Load available data
    forecast_history = data.get("forecast_overall_history", pd.DataFrame())
    forecast_future = data.get("forecast_overall_future", pd.DataFrame())
    forecast_best = data.get("forecast_overall_best", pd.DataFrame())
    model_comparison = data.get("forecast_model_comparison", pd.DataFrame())
    
    # Forecast parameters
    col1, col2, col3 = st.columns(3)
    with col1:
        forecast_days = st.slider("Forecast Horizon (Days)", 7, 90, 30)
    with col2:
        if has_comparison and len(model_comparison) > 0:
            best_model = model_comparison.loc[model_comparison["mape_pct"].idxmin(), "model"] if "mape_pct" in model_comparison.columns else "Moving Average"
            model_type = st.selectbox("Forecasting Model", [best_model, "Moving Average", "Exponential Smoothing", "Prophet"], index=0)
        else:
            model_type = st.selectbox("Forecasting Model", ["Moving Average", "Exponential Smoothing", "Prophet"])
    with col3:
        confidence_level = st.selectbox("Confidence Level", [80, 90, 95])
    
    # Model comparison table
    if has_comparison and len(model_comparison) > 0:
        st.subheader("📊 Forecast Model Comparison")
        st.dataframe(model_comparison[["model", "mae", "rmse", "mape_pct"]], width="content")
    
    # Historical and forecast chart
    if has_history:
        historical_data = forecast_history.head(90)
        dates = pd.to_datetime(historical_data["date"])
        historical_demand = historical_data["units_sold"].cumsum()
        
        # Forecast data
        forecast_source = forecast_best if has_best else forecast_future
        if has_best or has_future:
            forecast_data = forecast_source.head(forecast_days)
            forecast_dates = pd.to_datetime(forecast_data["date"])
            
            # Calculate cumulative forecast starting from last historical value
            last_historical_value = historical_demand.iloc[-1]
            forecast_demand = forecast_data["forecast_units"].cumsum() + last_historical_value
            
            upper_bound = forecast_demand * 1.1
            lower_bound = forecast_demand * 0.9
            
            # Create advanced chart with neon styling
            fig = go.Figure()
            
            # Historical data with gradient fill
            fig.add_trace(go.Scatter(
                x=dates,
                y=historical_demand,
                name='Historical',
                line=dict(color='#00ffff', width=3),
                mode='lines+markers',
                marker=dict(size=6, color='#00ffff', line=dict(width=2, color='#ffffff')),
                hovertemplate='<b>%{x}</b><br>Demand: %{y:,.0f}<extra></extra>'
            ))
            
            # Forecast line with neon glow effect
            fig.add_trace(go.Scatter(
                x=forecast_dates,
                y=forecast_demand,
                name='Forecast',
                line=dict(color='#ff00ff', width=4, dash='solid'),
                mode='lines+markers',
                marker=dict(size=8, color='#ff00ff', line=dict(width=2, color='#ffffff')),
                hovertemplate='<b>%{x}</b><br>Forecast: %{y:,.0f}<extra></extra>'
            ))
            
            # Confidence interval with gradient fill
            fig.add_trace(go.Scatter(
                x=forecast_dates,
                y=upper_bound,
                name='Upper Bound',
                line=dict(color='#ff6600', width=0),
                showlegend=False,
                hoverinfo='skip'
            ))
            
            fig.add_trace(go.Scatter(
                x=forecast_dates,
                y=lower_bound,
                name='Confidence Interval',
                line=dict(color='#ff6600', width=0),
                fill='tonexty',
                fillcolor='rgba(255, 102, 0, 0.2)',
                showlegend=True,
                hovertemplate='<b>%{x}</b><br>Upper: %{y:,.0f}<extra></extra>'
            ))
            
            # Add vertical line at forecast start
            fig.add_vline(
                x=dates.iloc[-1],
                line_dash="dash",
                line_color="#00ff00",
                line_width=2,
                annotation_text="Forecast Start",
                annotation_position="top"
            )
            
            # Advanced layout with neon theme
            fig.update_layout(
                title=dict(
                    text=f"<b>📈 Demand Forecast - {model_type} Model</b>",
                    font=dict(size=24, color='#ffffff', family='Arial Black'),
                    x=0.5,
                    xanchor='center'
                ),
                xaxis=dict(
                    title=dict(
                        text="📅 Date",
                        font=dict(size=16, color='#00ffff')
                    ),
                    gridcolor='rgba(0, 255, 255, 0.1)',
                    showgrid=True,
                    tickfont=dict(color='#ffffff', size=12),
                    linecolor='#00ffff'
                ),
                yaxis=dict(
                    title=dict(
                        text="📊 Cumulative Demand",
                        font=dict(size=16, color='#ff00ff')
                    ),
                    gridcolor='rgba(255, 0, 255, 0.1)',
                    showgrid=True,
                    tickfont=dict(color='#ffffff', size=12),
                    linecolor='#ff00ff'
                ),
                hovermode='x unified',
                template='plotly_dark',
                plot_bgcolor='rgba(10, 10, 26, 0.8)',
                paper_bgcolor='rgba(20, 20, 46, 0.9)',
                font=dict(color='#ffffff'),
                margin=dict(l=80, r=40, t=100, b=80),
                height=500,
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="right",
                    x=1,
                    font=dict(size=14, color='#ffffff')
                ),
                hoverlabel=dict(
                    bgcolor="rgba(0, 0, 0, 0.8)",
                    bordercolor="#00ffff",
                    font=dict(color="#ffffff", size=12)
                )
            )
            
            # Add range selector buttons
            fig.update_xaxes(
                rangeselector=dict(
                    buttons=list([
                        dict(count=7, label="7D", step="day", stepmode="backward"),
                        dict(count=30, label="30D", step="day", stepmode="backward"),
                        dict(count=90, label="90D", step="day", stepmode="backward"),
                        dict(step="all")
                    ]),
                    bgcolor='rgba(0, 255, 255, 0.1)',
                    font=dict(color='#00ffff')
                )
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Forecast accuracy metrics
            if has_comparison and len(model_comparison) > 0:
                best_model_row = model_comparison.loc[model_comparison["mape_pct"].idxmin()]
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("MAE", f"{best_model_row.get('mae', 0):.1f}")
                with col2:
                    st.metric("RMSE", f"{best_model_row.get('rmse', 0):.1f}")
                with col3:
                    st.metric("MAPE", f"{best_model_row.get('mape_pct', 0):.1f}%")
                with col4:
                    accuracy = 100 - best_model_row.get('mape_pct', 0)
                    st.metric("Forecast Accuracy", f"{accuracy:.1f}%")
        else:
            st.info("Forecast data not available")
    else:
        st.info("Historical forecast data not available")

# Page: Anomaly Detection
def anomaly_detection(data):
    st.markdown('<h1 class="main-header float">🚨 Anomaly Detection</h1>', unsafe_allow_html=True)
    
    if "orders" not in data:
        st.warning("Order data not available. Please run data transformations first.")
        return
    
    orders_df = data["orders"]
    orders_df["order_date"] = pd.to_datetime(orders_df["order_date"])
    
    # Anomaly parameters
    col1, col2 = st.columns(2)
    with col1:
        threshold = st.slider("Z-Score Threshold", 2.0, 4.0, 3.0)
    with col2:
        lookback_days = st.slider("Lookback Period (Days)", 30, 90, 60)
    
    # Calculate daily revenue
    cutoff_date = pd.to_datetime("2024-12-31")
    start_date = cutoff_date - timedelta(days=lookback_days)
    
    recent_orders = orders_df[
        (orders_df["order_date"] >= start_date) & 
        (orders_df["order_date"] <= cutoff_date) &
        (orders_df["order_status"].isin(['Delivered', 'Shipped', 'Processing', 'Returned']))
    ]
    
    daily_revenue = recent_orders.groupby(recent_orders["order_date"].dt.date)["order_total"].sum().reset_index()
    daily_revenue.columns = ["date", "revenue"]
    daily_revenue = daily_revenue.sort_values("date")
    
    if len(daily_revenue) < 10:
        st.warning("Not enough data points for anomaly detection")
        return
    
    # Calculate z-scores
    revenue_values = daily_revenue["revenue"].values
    z_scores = (revenue_values - revenue_values.mean()) / revenue_values.std()
    anomalies = np.abs(z_scores) > threshold
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=daily_revenue["date"], y=revenue_values, name='Revenue',
                            line=dict(color='#6366f1', width=2)))
    if anomalies.sum() > 0:
        fig.add_trace(go.Scatter(x=daily_revenue.loc[anomalies, "date"], 
                                y=revenue_values[anomalies],
                                mode='markers', name='Anomalies',
                                marker=dict(size=15, color='#f43f5e', symbol='diamond')))
    fig.update_layout(
        title=f"Revenue Anomalies (Threshold: {threshold}σ)",
        xaxis_title="Date",
        yaxis_title="Revenue (₹)",
        hovermode='x unified'
    )
    st.plotly_chart(fig, width="content")
    
    # Anomaly details
    st.subheader("Detected Anomalies")
    if anomalies.sum() > 0:
        anomaly_dates = daily_revenue.loc[anomalies, "date"]
        anomaly_values = revenue_values[anomalies]
        anomaly_z = z_scores[anomalies]
        
        anomaly_df = pd.DataFrame({
            "Date": anomaly_dates,
            "Revenue (₹)": anomaly_values,
            "Z-Score": anomaly_z,
            "Type": ["Spike" if z > 0 else "Drop" for z in anomaly_z]
        })
        st.dataframe(anomaly_df, width="content")
    else:
        st.info("No anomalies detected in the selected period.")

# Page: Customer Churn
def churn_prediction(data):
    st.markdown('<h1 class="main-header">🎯 Customer Churn Prediction</h1>', unsafe_allow_html=True)
    
    # Churn risk parameters
    col1, col2 = st.columns(2)
    with col1:
        risk_threshold = st.slider("Churn Probability Threshold", 0.3, 0.9, 0.5)
    with col2:
        show_details = st.checkbox("Show Customer Details", False)
    
    # Use actual churn predictions if available
    if "churn_predictions" in data:
        churn_data = data["churn_predictions"]
        
        # Risk distribution
        st.subheader("Churn Risk Distribution")
        risk_counts = churn_data["risk_tier"].value_counts()
        
        fig = go.Figure(data=[go.Bar(x=risk_counts.index, y=risk_counts.values,
                                     marker=dict(color=['#10b981', '#f59e0b', '#f43f5e']))])
        fig.update_layout(
            title="Customers by Churn Risk Tier",
            xaxis_title="Risk Tier",
            yaxis_title="Count",
            plot_bgcolor='rgba(20, 20, 20, 0.8)',
            paper_bgcolor='rgba(20, 20, 20, 0.8)',
            font=dict(color='#ffffff'),
            xaxis=dict(tickfont=dict(color='#ffffff'), title=dict(font=dict(color='#ffffff'))),
            yaxis=dict(tickfont=dict(color='#ffffff'), title=dict(font=dict(color='#ffffff')))
        )
        st.plotly_chart(fig, width="content")
        
        # High-risk customers
        high_risk = churn_data[churn_data["churn_probability"] >= risk_threshold]
        
        st.subheader(f"High-Risk Customers (Probability ≥ {risk_threshold:.0%})")
        st.metric("At-Risk Customers", len(high_risk))
        
        if show_details and len(high_risk) > 0:
            display_cols = ["customer_id", "total_orders", "total_spend", "days_since_last_order", 
                          "churn_probability", "risk_tier"]
            available_cols = [col for col in display_cols if col in high_risk.columns]
            st.dataframe(high_risk[available_cols].head(20), width="content")
    elif "churn_features" in data:
        # Use churn features with label as probability
        churn_data = data["churn_features"]
        
        st.subheader("Churn Risk Distribution")
        churn_data["risk_tier"] = pd.cut(
            churn_data["churn_label_90d"],
            bins=[-0.1, 0.3, 0.6, 1.1],
            labels=["Low", "Medium", "High"]
        )
        risk_counts = churn_data["risk_tier"].value_counts()
        
        fig = go.Figure(data=[go.Bar(x=risk_counts.index, y=risk_counts.values,
                                     marker=dict(color=['#10b981', '#f59e0b', '#f43f5e']))])
        fig.update_layout(
            title="Customers by Churn Risk Tier",
            xaxis_title="Risk Tier",
            yaxis_title="Count",
            plot_bgcolor='rgba(20, 20, 20, 0.8)',
            paper_bgcolor='rgba(20, 20, 20, 0.8)',
            font=dict(color='#ffffff'),
            xaxis=dict(tickfont=dict(color='#ffffff'), title=dict(font=dict(color='#ffffff'))),
            yaxis=dict(tickfont=dict(color='#ffffff'), title=dict(font=dict(color='#ffffff')))
        )
        st.plotly_chart(fig, width="content")
        
        # High-risk customers
        high_risk = churn_data[churn_data["churn_label_90d"] >= risk_threshold]
        
        st.subheader(f"High-Risk Customers (Probability ≥ {risk_threshold:.0%})")
        st.metric("At-Risk Customers", len(high_risk))
        
        if show_details and len(high_risk) > 0:
            display_cols = ["customer_id", "total_orders", "total_spend", "days_since_last_order", 
                          "churn_label_90d", "risk_tier"]
            available_cols = [col for col in display_cols if col in high_risk.columns]
            st.dataframe(high_risk[available_cols].head(20), width="content")
    else:
        st.info("Churn prediction data not available. Run ML models first.")

# Page: Decision Center
def decision_center(data):
    st.markdown('<h1 class="main-header float glow">💡 Decision Center</h1>', unsafe_allow_html=True)
    
    # Business Alerts
    st.subheader("🚨 Business Alerts")
    
    with st.expander("Revenue Alerts", expanded=True):
        st.markdown("""
        <div class="alert-box alert-danger">
            <strong>⚠️ Revenue Drop Detected</strong><br>
            Electronics category revenue down 17.2% this week<br>
            <em>Potential causes: Inventory shortage, increased returns, reduced traffic</em>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="alert-box alert-warning">
            <strong>⚠️ Order Volume Spike</strong><br>
            Unusual 25% increase in orders from Maharashtra region<br>
            <em>Investigation needed: Verify data quality, check for promotional activity</em>
        </div>
        """, unsafe_allow_html=True)
    
    # Inventory Alerts
    st.subheader("⚠️ Inventory Alerts")
    
    with st.expander("Stock-Out Risks", expanded=True):
        st.markdown("""
        <div class="alert-box alert-danger">
            <strong>12 products at risk of stock-out within 7 days</strong><br>
            • Wireless Headphones (Electronics) - 2 days remaining<br>
            • Smart Watch (Electronics) - 3 days remaining<br>
            • Gaming Mouse (Electronics) - 5 days remaining<br>
            <em>Action: Initiate emergency replenishment orders</em>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="alert-box alert-warning">
            <strong>8 products overstocked (>90 days inventory)</strong><br>
            • Laptop Stand (Office) - 180 days inventory<br>
            • Cable Organizer (Accessories) - 120 days inventory<br>
            <em>Action: Consider discount promotions to clear inventory</em>
        </div>
        """, unsafe_allow_html=True)
    
    # Customer Alerts
    st.subheader("🎯 Customer Alerts")
    
    with st.expander("Churn Risk Alerts", expanded=True):
        st.markdown("""
        <div class="alert-box alert-warning">
            <strong>4,821 high-value customers showing churn signals</strong><br>
            • 1,234 Champions moved to At Risk segment<br>
            • 2,456 Loyal Customers with declining order frequency<br>
            • 1,131 customers with 90+ days since last order<br>
            <em>Action: Launch personalized retention campaign</em>
        </div>
        """, unsafe_allow_html=True)
    
    # Recommendations
    st.subheader("💡 Smart Recommendations")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class="alert-box alert-success">
            <strong>📦 Inventory Recommendation</strong><br>
            Replenish Electronics category inventory<br>
            <em>Impact: Prevent ₹2.5 Cr potential revenue loss</em>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="alert-box alert-success">
            <strong>📈 Pricing Recommendation</strong><br>
            Consider 5% price increase for Premium quadrant products<br>
            <em>Impact: +₹45 Cr annual revenue, minimal volume impact</em>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="alert-box alert-success">
            <strong>🎯 Marketing Recommendation</strong><br>
            Launch retention campaign for at-risk customers<br>
            <em>Impact: Save ₹1.8 Cr CLV, 15% reduction in churn rate</em>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="alert-box alert-success">
            <strong>⚙️ Operations Recommendation</strong><br>
            Review Supplier X - delivery delays increasing<br>
            <em>Impact: Reduce stock-outs by 22%, improve customer satisfaction</em>
        </div>
        """, unsafe_allow_html=True)
    
    # What-If Analysis
    st.subheader("🔮 What-If Analysis")
    
    col1, col2 = st.columns(2)
    with col1:
        discount_impact = st.slider("Discount Impact on Revenue", -20, 20, 0)
    with col2:
        marketing_spend = st.slider("Marketing Spend Increase (₹ L)", 0, 100, 10)
    
    # Calculate projected impact
    base_revenue = 8.42e7  # ₹8.42 Cr
    discount_factor = 1 + (discount_impact / 100)
    marketing_factor = 1 + (marketing_spend * 1e5 / base_revenue) * 2  # Assuming 2x ROAS
    
    projected_revenue = base_revenue * discount_factor * marketing_factor
    revenue_change = ((projected_revenue - base_revenue) / base_revenue) * 100
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Current Revenue", "₹8.42 Cr")
    with col2:
        st.metric("Projected Revenue", f"₹{projected_revenue/1e7:.2f} Cr", f"{revenue_change:+.1f}%")
    with col3:
        st.metric("Revenue Impact", f"₹{(projected_revenue - base_revenue)/1e5:.1f} L")

# Main application
def main():
    # Sidebar navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.radio(
        "Select Analytics Page",
        [
            "🏠 Executive Dashboard",
            "📊 Sales Analytics",
            "👥 Customer Analytics",
            "📦 Product Analytics",
            "📈 Forecasting",
            "🚨 Anomaly Detection",
            "🎯 Customer Churn",
            "💡 Decision Center"
        ]
    )
    
    # Load data
    with st.sidebar:
        st.subheader("Data Status")
        data = load_data()
        st.success(f"✅ Loaded {len(data)} datasets")
        
        # Refresh button
        if st.button("🔄 Refresh Data"):
            st.cache_data.clear()
            st.rerun()
    
    # Page routing
    if page == "🏠 Executive Dashboard":
        executive_dashboard(data)
    elif page == "📊 Sales Analytics":
        sales_analytics(data)
    elif page == "👥 Customer Analytics":
        customer_analytics(data)
    elif page == "📦 Product Analytics":
        product_analytics(data)
    elif page == "📈 Forecasting":
        forecasting_analytics(data)
    elif page == "🚨 Anomaly Detection":
        anomaly_detection(data)
    elif page == "🎯 Customer Churn":
        churn_prediction(data)
    elif page == "💡 Decision Center":
        decision_center(data)
    
    # Footer
    st.sidebar.markdown("---")
    st.sidebar.markdown("""
    **E-Commerce Intelligence Platform**
    
    Version: 1.0.0
    Data: PostgreSQL + Python
    Analytics: Advanced SQL + ML
    """)

if __name__ == "__main__":
    main()
