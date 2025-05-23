# root_finder_ui.py
import streamlit as st
import numpy as np
from sympy import symbols, sympify, lambdify, diff

from methods.graphical import graphical_ui
from methods.incremental import incremental_ui
from methods.bisection import bisection_ui
from methods.regula_falsi import regula_falsi_ui
from methods.newton_raphson import newton_raphson_ui
from methods.secant import secant_ui

st.set_page_config(page_title="Root Finder", layout="wide", page_icon="🔎")

# --- Custom Styles ---
st.markdown("""
    <style>
    @import url('https://fonts.cdnfonts.com/css/valorax');
    body, html, h1, h2, h3, p, div, span, input, label, button, select, textarea {
        font-family: 'Valorax', sans-serif !important;
    }
    body {
        background-color: #0f0f1a;
    }
    section[data-testid="stMain"] {
        background-color: #0f0f1a;
    }
    .stButton>button {
        background-color: transparent;
        color: #ff00ff;
        border: 2px solid #00fff7;
        border-radius: 15px;
        font-weight: bold;
        margin-bottom: 10px;
        width: 100%;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        background-color: #00fff7;
        color: #0f0f1a;
        box-shadow: 0 0 10px #00fff7;
        border-color: #ff00ff;
    }
    .custom-panel {
        border: 2px solid #ff00ff;
        border-radius: 20px;
        padding: 20px;
        background-color: #1a1a2e;
        box-shadow: 0 0 10px #ff00ff33;
    }
    h1, h2, h3 {
        color: magenta;
        text-shadow: 0 0 5px #ff00ff;
    }
    label, .stMultiSelect, .stTextInput, .stNumberInput {
        color: #00fff7 !important;
    }
    .stTextInput > div > div > input:focus,
    .stNumberInput > div > div > input:focus {
        outline: none;
        box-shadow: 0 0 6px #ff00ff;
    }
    .stTextInput > div > div > input:hover,
    .stNumberInput > div > div > input:hover {
        outline: none;
        box-shadow: 0 0 8px #00fff7;
    }
    .st-expander>summary {
        color: #ff00ff;
    }
    .stAlert, .element-container:has(.stAlert) {
        background-color: #1f0033;
        color: #00fff7;
        border: 1px solid #ff00ff;
        border-radius: 12px;
        box-shadow: 0 0 12px #ff00ff55;
    }
    .element-container:has(.stDataFrame) {
        background-color: #1f0033 !important;
        border: 1px solid #00fff7;
        border-radius: 12px;
        box-shadow: 0 0 12px #00fff733;
    }
    @keyframes glowHeader {
        0% { text-shadow: 0 0 5px #ff00ff, 0 0 10px #00fff7; }
        50% { text-shadow: 0 0 20px #ff00ff, 0 0 30px #00fff7; }
        100% { text-shadow: 0 0 5px #ff00ff, 0 0 10px #00fff7; }
    }
    h1, h2, h3 {
        animation: glowHeader 2s infinite;
    }
    .element-container:has(.stDataFrame) table {
        color: #00fff7 !important;
        background-color: #12122a !important;
        border-collapse: collapse;
        border: 1px solid #ff00ff;
        box-shadow: 0 0 12px #00fff755;
    }
    .element-container:has(.stDataFrame) thead tr {
        background-color: #1f0033 !important;
        color: #ff00ff !important;
    }
    .element-container:has(.stDataFrame) tbody tr:hover {
        background-color: #262626 !important;
    }
    .element-container:has(.stDataFrame) td, .element-container:has(.stDataFrame) th {
        border: 1px solid #00fff7 !important;
        padding: 8px;
    }
    </style>
""", unsafe_allow_html=True)

# --- Title ---
st.markdown("""
<div style='text-align:left; margin-bottom: 2rem;'>
    <h1>🔎 NUMERICAL</h1>
</div>
""", unsafe_allow_html=True)

# --- Layout Panel ---
col1, col2 = st.columns([1, 3])

if "selected_methods" not in st.session_state:
    st.session_state.selected_methods = []

with col1:
    if st.button("🧹 Clear Selections"):
        st.session_state.selected_methods = []

    st.markdown("<h2 style='color:magenta;'>Methods</h2>", unsafe_allow_html=True)

    method_labels = {
        "Graphical": "GRAPHICAL METHOD",
        "Incremental": "INCREMENTAL METHOD",
        "Bisection": "BISECTION METHOD",
        "False": "FALSE POSITION METHOD",
        "Newton": "NEWTON RAPHSON METHOD",
        "Secant": "SECANT METHOD"
    }

    for key, label in method_labels.items():
        if st.button(label, key=key):
            if key not in st.session_state.selected_methods:
                st.session_state.selected_methods.append(key)

with col2:
    st.markdown("""<div class='custom-panel'>""", unsafe_allow_html=True)
    st.markdown("<h2 style='color:magenta;'>⚙️ Settings</h2>", unsafe_allow_html=True)

    f_expr_input = st.text_input("Function f(x)", value="x**3 - 6*x**2 + 11*x - 6", key="function_input")
    x_start = st.number_input("Start", value=0.0, format="%.4f", key="x_start")
    x_end = st.number_input("End", value=5.0, format="%.4f", key="x_end")

    run = st.button("🚀 Run Root-Finding", key="run_button")
    st.markdown("""</div>""", unsafe_allow_html=True)

# --- Help Section ---
with st.expander("ℹ️ Direction"):
    st.markdown("""
    <div style='color: #00fff7;'>
    - Input the function in Python format, e.g. `x**2 - 4`.<br>
    - Set the interval for root-finding.<br>
    - Click one or more method buttons to prepare.<br>
    - Click "Run Root-Finding" to see individual results and root summaries.
    </div>
    """, unsafe_allow_html=True)

# --- Main Execution ---
if run:
    x = symbols('x')
    try:
        f_expr = sympify(f_expr_input)
        f = lambdify(x, f_expr, modules='numpy')
        df_expr = diff(f_expr, x)
        df = lambdify(x, df_expr, 'numpy')
    except Exception as e:
        st.error(f"❌ Invalid function: {e}")
        st.stop()

    x_range = (x_start, x_end)
    all_roots = []

    method_ui = {
        "Graphical": ("📈 Graphical Method", lambda: graphical_ui(f, x_range)),
        "Incremental": ("🔍 Incremental Search", lambda: incremental_ui(f, x_range)),
        "Bisection": ("🪓 Bisection Method", lambda: bisection_ui(f, x_range)),
        "False": ("📐 Regula Falsi Method", lambda: regula_falsi_ui(f, x_range)),
        "Newton": ("📉 Newton–Raphson Method", lambda: newton_raphson_ui(f, df, x_range)),
        "Secant": ("📏 Secant Method", lambda: secant_ui(f, x_range)),
    }

    for method in st.session_state.selected_methods:
        title, func = method_ui[method]
        st.markdown(f"<h3 style='color:#ff00ff;'>{title}</h3>", unsafe_allow_html=True)
        try:
            roots = func()
            if roots:
                st.success(f"✅ Found {len(roots)} root(s).")
                all_roots += roots
            else:
                st.warning("⚠️ No roots found.")
        except Exception as e:
            st.error(f"❌ {method} failed: {e}")

    st.markdown("---")
