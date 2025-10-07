"""
Professional Styling for LLM Router Analytics Platform
Enterprise-grade CSS styling for Streamlit dashboard
"""

import streamlit as st

def load_professional_css():
    """Load professional CSS styling for the dashboard"""
    st.markdown("""
    <style>
        /* Import Inter font */
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
        
        /* ============================================================================
           CSS VARIABLES (Design System)
           ============================================================================ */
        :root {
            --primary-900: #0f172a;
            --primary-800: #1e293b;
            --primary-700: #334155;
            --primary-600: #475569;
            --primary-100: #f1f5f9;
            --primary-50: #f8fafc;
            
            --accent-500: #3b82f6;
            --accent-400: #60a5fa;
            --accent-600: #2563eb;
            
            --success-500: #10b981;
            --warning-500: #f59e0b;
            --error-500: #ef4444;
            
            --surface-primary: #ffffff;
            --surface-secondary: #f8fafc;
            --surface-tertiary: #f1f5f9;
            
            --text-primary: #0f172a;
            --text-secondary: #475569;
            --text-tertiary: #94a3b8;
            
            --border-light: #e2e8f0;
            --border-medium: #cbd5e1;
            
            --shadow-sm: 0 1px 2px 0 rgb(0 0 0 / 0.05);
            --shadow-md: 0 4px 6px -1px rgb(0 0 0 / 0.1), 0 2px 4px -2px rgb(0 0 0 / 0.1);
            --shadow-lg: 0 10px 15px -3px rgb(0 0 0 / 0.1), 0 4px 6px -4px rgb(0 0 0 / 0.1);
            --shadow-xl: 0 20px 25px -5px rgb(0 0 0 / 0.1), 0 8px 10px -6px rgb(0 0 0 / 0.1);
            
            --radius-sm: 6px;
            --radius-md: 8px;
            --radius-lg: 12px;
            --radius-xl: 16px;
        }
        
        /* ============================================================================
           GLOBAL STYLES
           ============================================================================ */
        .stApp {
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif !important;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
            color: var(--text-primary);
            line-height: 1.5;
            font-feature-settings: 'kern';
            -webkit-font-smoothing: antialiased;
            -moz-osx-font-smoothing: grayscale;
        }
        
        /* ============================================================================
           SIDEBAR STYLING
           ============================================================================ */
        .css-1d391kg, .css-17eq0hr {
            background: var(--surface-primary) !important;
            border-right: 1px solid var(--border-light) !important;
            box-shadow: var(--shadow-lg) !important;
        }
        
        .css-1lcbmhc {
            padding: 2rem 1.5rem !important;
        }
        
        /* Sidebar radio buttons */
        .css-1lcbmhc .css-1v0mbdj {
            border-radius: var(--radius-md) !important;
            margin-bottom: 0.25rem !important;
            padding: 0.75rem 1rem !important;
            transition: all 0.2s cubic-bezier(0.4, 0, 0.2, 1) !important;
            font-weight: 500 !important;
        }
        
        .css-1lcbmhc .css-1v0mbdj:hover {
            background: var(--surface-tertiary) !important;
            color: var(--text-primary) !important;
        }
        
        .css-1lcbmhc .css-1v0mbdj[data-checked="true"] {
            background: linear-gradient(135deg, var(--accent-500) 0%, var(--accent-600) 100%) !important;
            color: white !important;
            box-shadow: var(--shadow-md) !important;
        }
        
        /* ============================================================================
           MAIN CONTENT AREA
           ============================================================================ */
        .css-12oz5g7 {
            background: var(--surface-secondary) !important;
            padding: 2rem !important;
        }
        
        .block-container {
            background: var(--surface-secondary) !important;
            padding-top: 2rem !important;
            padding-bottom: 2rem !important;
        }
        
        /* ============================================================================
           TYPOGRAPHY
           ============================================================================ */
        h1, h2, h3, h4, h5, h6 {
            font-family: 'Inter', sans-serif !important;
            color: var(--text-primary) !important;
            font-weight: 600 !important;
            letter-spacing: -0.025em !important;
        }
        
        h1 {
            font-size: 2rem !important;
            font-weight: 700 !important;
            margin-bottom: 0.5rem !important;
        }
        
        h2 {
            font-size: 1.5rem !important;
            margin-bottom: 1rem !important;
            border-bottom: 1px solid var(--border-light) !important;
            padding-bottom: 0.5rem !important;
        }
        
        h3 {
            font-size: 1.25rem !important;
            margin-bottom: 1rem !important;
        }
        
        /* ============================================================================
           METRIC CARDS
           ============================================================================ */
        [data-testid="metric-container"] {
            background: var(--surface-primary) !important;
            border: 1px solid var(--border-light) !important;
            border-radius: var(--radius-xl) !important;
            padding: 1.5rem !important;
            box-shadow: var(--shadow-md) !important;
            transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
            position: relative !important;
            overflow: hidden !important;
        }
        
        [data-testid="metric-container"]::before {
            content: '' !important;
            position: absolute !important;
            top: 0 !important;
            left: 0 !important;
            right: 0 !important;
            height: 4px !important;
            background: linear-gradient(90deg, var(--accent-500) 0%, var(--accent-400) 100%) !important;
        }
        
        [data-testid="metric-container"]:hover {
            transform: translateY(-4px) !important;
            box-shadow: var(--shadow-xl) !important;
        }
        
        [data-testid="metric-container"] [data-testid="metric-value"] {
            font-family: 'Inter', sans-serif !important;
            font-size: 2.25rem !important;
            font-weight: 700 !important;
            color: var(--text-primary) !important;
            letter-spacing: -0.025em !important;
            line-height: 1 !important;
        }
        
        [data-testid="metric-container"] [data-testid="metric-label"] {
            font-family: 'Inter', sans-serif !important;
            font-size: 0.875rem !important;
            color: var(--text-secondary) !important;
            font-weight: 500 !important;
            margin-bottom: 0.5rem !important;
        }
        
        [data-testid="metric-container"] [data-testid="metric-delta"] {
            font-family: 'Inter', sans-serif !important;
            font-size: 0.8125rem !important;
            font-weight: 600 !important;
            margin-top: 0.5rem !important;
        }
        
        /* Delta color customization */
        [data-testid="metric-container"] [data-testid="metric-delta"] svg[fill="rgb(255, 43, 43)"] {
            fill: var(--error-500) !important;
        }
        
        [data-testid="metric-container"] [data-testid="metric-delta"] svg[fill="rgb(9, 171, 59)"] {
            fill: var(--success-500) !important;
        }
        
        /* ============================================================================
           CHARTS AND PLOTLY
           ============================================================================ */
        .js-plotly-plot {
            background: var(--surface-primary) !important;
            border: 1px solid var(--border-light) !important;
            border-radius: var(--radius-xl) !important;
            padding: 1rem !important;
            box-shadow: var(--shadow-md) !important;
            transition: all 0.3s ease !important;
        }
        
        .js-plotly-plot:hover {
            box-shadow: var(--shadow-lg) !important;
        }
        
        /* ============================================================================
           DATAFRAMES AND TABLES
           ============================================================================ */
        .stDataFrame {
            background: var(--surface-primary) !important;
            border: 1px solid var(--border-light) !important;
            border-radius: var(--radius-xl) !important;
            overflow: hidden !important;
            box-shadow: var(--shadow-md) !important;
        }
        
        .stDataFrame [data-testid="stDataFrameResizable"] {
            background: var(--surface-primary) !important;
        }
        
        .stDataFrame table {
            font-family: 'Inter', sans-serif !important;
            font-size: 0.875rem !important;
        }
        
        .stDataFrame thead th {
            background: var(--surface-tertiary) !important;
            color: var(--text-secondary) !important;
            font-weight: 600 !important;
            font-size: 0.8125rem !important;
            text-transform: uppercase !important;
            letter-spacing: 0.05em !important;
            padding: 1rem 1.25rem !important;
            border-bottom: 1px solid var(--border-light) !important;
        }
        
        .stDataFrame tbody td {
            padding: 1rem 1.25rem !important;
            color: var(--text-primary) !important;
            border-bottom: 1px solid var(--border-light) !important;
        }
        
        .stDataFrame tbody tr:hover {
            background: var(--surface-secondary) !important;
        }
        
        .stDataFrame tbody tr:last-child td {
            border-bottom: none !important;
        }
        
        /* ============================================================================
           ALERTS AND NOTIFICATIONS
           ============================================================================ */
        .stAlert {
            border-radius: var(--radius-lg) !important;
            border-left: 4px solid !important;
            padding: 1rem 1.25rem !important;
            font-size: 0.875rem !important;
            font-weight: 500 !important;
            font-family: 'Inter', sans-serif !important;
            margin: 1rem 0 !important;
        }
        
        .stAlert[data-baseweb="notification"] {
            background: #f0fdf4 !important;
            border-left-color: var(--success-500) !important;
            color: #166534 !important;
        }
        
        .stAlert[data-baseweb="notification"][kind="warning"] {
            background: #fffbeb !important;
            border-left-color: var(--warning-500) !important;
            color: #9a3412 !important;
        }
        
        .stAlert[data-baseweb="notification"][kind="error"] {
            background: #fef2f2 !important;
            border-left-color: var(--error-500) !important;
            color: #991b1b !important;
        }
        
        .stAlert[data-baseweb="notification"][kind="info"] {
            background: #eff6ff !important;
            border-left-color: var(--accent-500) !important;
            color: #1e40af !important;
        }
        
        /* ============================================================================
           BUTTONS
           ============================================================================ */
        .stButton > button {
            background: var(--accent-500) !important;
            color: white !important;
            border: none !important;
            border-radius: var(--radius-md) !important;
            font-family: 'Inter', sans-serif !important;
            font-weight: 500 !important;
            padding: 0.5rem 1rem !important;
            transition: all 0.2s cubic-bezier(0.4, 0, 0.2, 1) !important;
            box-shadow: var(--shadow-sm) !important;
        }
        
        .stButton > button:hover {
            background: var(--accent-600) !important;
            transform: translateY(-1px) !important;
            box-shadow: var(--shadow-md) !important;
        }
        
        .stButton > button:active {
            transform: translateY(0px) !important;
            box-shadow: var(--shadow-sm) !important;
        }
        
        /* ============================================================================
           FORM ELEMENTS
           ============================================================================ */
        .stSelectbox > div > div {
            border-radius: var(--radius-md) !important;
            border-color: var(--border-light) !important;
            background: var(--surface-primary) !important;
            font-family: 'Inter', sans-serif !important;
        }
        
        .stSelectbox > div > div:focus-within {
            border-color: var(--accent-500) !important;
            box-shadow: 0 0 0 1px var(--accent-500) !important;
        }
        
        .stSlider > div > div > div {
            background: var(--accent-500) !important;
        }
        
        .stCheckbox > label {
            font-family: 'Inter', sans-serif !important;
            font-size: 0.875rem !important;
            color: var(--text-secondary) !important;
        }
        
        .stRadio > label {
            font-family: 'Inter', sans-serif !important;
            font-size: 0.875rem !important;
            color: var(--text-secondary) !important;
        }
        
        /* ============================================================================
           SPINNER AND LOADING
           ============================================================================ */
        .stSpinner > div {
            border-color: var(--accent-500) !important;
            border-top-color: transparent !important;
        }
        
        /* ============================================================================
           RESPONSIVE DESIGN
           ============================================================================ */
        @media (max-width: 768px) {
            .css-12oz5g7 {
                padding: 1rem !important;
            }
            
            h1 {
                font-size: 1.75rem !important;
            }
            
            [data-testid="metric-container"] {
                padding: 1rem !important;
            }
            
            [data-testid="metric-container"] [data-testid="metric-value"] {
                font-size: 1.875rem !important;
            }
        }
        
        /* ============================================================================
           HIDE STREAMLIT ELEMENTS
           ============================================================================ */
        #MainMenu {
            visibility: hidden;
        }
        
        footer {
            visibility: hidden;
        }
        
        header {
            visibility: hidden;
        }
        
        .css-15zrgzn {
            display: none;
        }
        
        .css-eczf16 {
            display: none;
        }
        
        /* ============================================================================
           CUSTOM ANIMATIONS
           ============================================================================ */
        @keyframes pulse {
            0%, 100% {
                opacity: 1;
            }
            50% {
                opacity: 0.5;
            }
        }
        
        @keyframes fadeIn {
            from {
                opacity: 0;
                transform: translateY(10px);
            }
            to {
                opacity: 1;
                transform: translateY(0);
            }
        }
        
        .fade-in {
            animation: fadeIn 0.5s ease-out;
        }
        
        /* ============================================================================
           PROFESSIONAL TOUCHES
           ============================================================================ */
        
        /* Custom scrollbar */
        ::-webkit-scrollbar {
            width: 8px;
        }
        
        ::-webkit-scrollbar-track {
            background: var(--surface-tertiary);
        }
        
        ::-webkit-scrollbar-thumb {
            background: var(--border-medium);
            border-radius: 4px;
        }
        
        ::-webkit-scrollbar-thumb:hover {
            background: var(--primary-600);
        }
        
        /* Selection color */
        ::selection {
            background: var(--accent-500);
            color: white;
        }
        
        /* Focus outlines */
        button:focus,
        select:focus,
        input:focus {
            outline: 2px solid var(--accent-500) !important;
            outline-offset: 2px !important;
        }
        
        /* Professional shadows for elevated content */
        .elevated {
            box-shadow: var(--shadow-xl) !important;
        }
        
        /* Status indicators */
        .status-dot {
            display: inline-block;
            width: 8px;
            height: 8px;
            border-radius: 50%;
            margin-right: 8px;
        }
        
        .status-online {
            background: var(--success-500);
        }
        
        .status-offline {
            background: var(--error-500);
        }
        
        .status-warning {
            background: var(--warning-500);
        }
        
        /* Professional badges */
        .badge {
            display: inline-flex;
            align-items: center;
            padding: 0.25rem 0.75rem;
            border-radius: 9999px;
            font-size: 0.75rem;
            font-weight: 500;
            font-family: 'Inter', sans-serif;
        }
        
        .badge-success {
            background: #dcfce7;
            color: #166534;
        }
        
        .badge-warning {
            background: #fef3c7;
            color: #92400e;
        }
        
        .badge-error {
            background: #fee2e2;
            color: #991b1b;
        }
        
        .badge-info {
            background: #dbeafe;
            color: #1e40af;
        }
        
        /* ============================================================================
           DARK MODE SUPPORT (Optional)
           ============================================================================ */
        @media (prefers-color-scheme: dark) {
            :root {
                --surface-primary: #1e293b;
                --surface-secondary: #0f172a;
                --surface-tertiary: #334155;
                --text-primary: #f8fafc;
                --text-secondary: #cbd5e1;
                --text-tertiary: #64748b;
                --border-light: #334155;
                --border-medium: #475569;
            }
        }
        
        /* ============================================================================
           PRINT STYLES
           ============================================================================ */
        @media print {
            .stApp {
                background: white !important;
            }
            
            .css-1d391kg {
                display: none !important;
            }
            
            [data-testid="metric-container"] {
                break-inside: avoid;
                margin-bottom: 1rem;
            }
        }
        
    </style>
    """, unsafe_allow_html=True)


def get_color_palette():
    """Get the professional color palette for charts and components"""
    return {
        'primary': '#3b82f6',
        'secondary': '#8b5cf6', 
        'success': '#10b981',
        'warning': '#f59e0b',
        'error': '#ef4444',
        'gradient': ['#667eea', '#764ba2', '#f093fb', '#f5576c', '#4facfe'],
        'neutral': ['#0f172a', '#1e293b', '#334155', '#475569', '#64748b', '#94a3b8']
    }


def apply_chart_theme():
    """Apply professional theme to Plotly charts"""
    return {
        'layout': {
            'font': {'family': 'Inter, sans-serif', 'size': 12, 'color': '#475569'},
            'plot_bgcolor': 'transparent',
            'paper_bgcolor': 'transparent',
            'colorway': get_color_palette()['gradient'],
            'margin': {'t': 60, 'r': 20, 'b': 40, 'l': 60}
        }
    }
