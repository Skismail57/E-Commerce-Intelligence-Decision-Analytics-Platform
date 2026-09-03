"""
Streamlit Styles Module
Contains all CSS styling for the application.
"""

STYLES = """
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

    @keyframes fadeIn {
        from { opacity: 0; }
        to { opacity: 1; }
    }

    /* Sidebar Styling */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0a0a0a 0%, #1a1a1a 100%);
        border-right: 2px solid var(--neon-cyan);
        box-shadow: 0 0 20px rgba(0, 255, 255, 0.3);
    }

    [data-testid="stSidebar"] > div:first-child {
        background: transparent;
    }

    /* Sidebar Navigation */
    .sidebar-nav-item {
        padding: 0.8rem 1rem;
        margin: 0.3rem 0;
        border-radius: 10px;
        transition: all 0.3s ease;
        border: 1px solid transparent;
    }

    .sidebar-nav-item:hover {
        background: linear-gradient(90deg, rgba(0, 255, 255, 0.1), rgba(255, 0, 255, 0.1));
        border-color: var(--neon-cyan);
        transform: translateX(5px);
    }

    /* Charts */
    .js-plotly-plot {
        background: rgba(26, 26, 46, 0.8);
        border-radius: 15px;
        border: 1px solid var(--neon-cyan);
        box-shadow: 0 0 30px rgba(0, 255, 255, 0.2);
    }

    /* DataFrames */
    .stDataFrame {
        background: rgba(26, 26, 46, 0.8);
        border-radius: 15px;
        border: 1px solid var(--neon-cyan);
    }

    /* Buttons */
    .stButton > button {
        background: linear-gradient(135deg, var(--neon-cyan), var(--neon-purple));
        border: none;
        border-radius: 10px;
        color: #ffffff;
        font-weight: 700;
        transition: all 0.3s ease;
        box-shadow: 0 0 20px rgba(0, 255, 255, 0.3);
    }

    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 0 30px rgba(0, 255, 255, 0.5);
    }

    /* Selectbox and Multiselect */
    .stSelectbox > div > div,
    .stMultiSelect > div > div {
        background: rgba(26, 26, 46, 0.8);
        border: 1px solid var(--neon-cyan);
        border-radius: 10px;
    }

    /* Date Input */
    .stDateInput > div > div > input {
        background: rgba(26, 26, 46, 0.8);
        border: 1px solid var(--neon-cyan);
        border-radius: 10px;
        color: #ffffff;
    }

    /* Slider */
    .stSlider > div > div > div {
        background: var(--neon-cyan);
    }

    /* Progress Bar */
    .stProgress > div > div > div > div {
        background: linear-gradient(90deg, var(--neon-cyan), var(--neon-pink));
    }

    /* Expander */
    .streamlit-expanderHeader {
        background: rgba(26, 26, 46, 0.8);
        border: 1px solid var(--neon-cyan);
        border-radius: 10px;
        color: #ffffff;
    }

    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 2rem;
    }

    .stTabs [data-baseweb="tab"] {
        background: rgba(26, 26, 46, 0.8);
        border: 1px solid var(--neon-cyan);
        border-radius: 10px 10px 0 0;
        color: #ffffff;
    }

    /* Info and Success Messages */
    .stAlert {
        background: rgba(26, 26, 46, 0.8);
        border: 1px solid var(--neon-cyan);
        border-radius: 15px;
        color: #ffffff;
    }

    /* Icon Pulse Animation */
    @keyframes iconPulse {
        0%, 100% {
            filter: drop-shadow(0 0 5px rgba(0, 255, 255, 0.5));
        }
        50% {
            filter: drop-shadow(0 0 20px rgba(0, 255, 255, 0.8));
        }
    }
</style>
"""


def apply_styles():
    """Apply custom styles to the Streamlit app."""
    import streamlit as st
    st.markdown(STYLES, unsafe_allow_html=True)
