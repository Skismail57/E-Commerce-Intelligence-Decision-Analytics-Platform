"""
Streamlit Application Module
"""

from app.streamlit.styles import apply_styles, STYLES
from app.streamlit.utils import (
    format_currency,
    format_number,
    format_percentage,
    load_data,
    filter_data_by_date,
    calculate_kpis
)

__all__ = [
    'apply_styles',
    'STYLES',
    'format_currency',
    'format_number',
    'format_percentage',
    'load_data',
    'filter_data_by_date',
    'calculate_kpis'
]
