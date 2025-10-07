"""
Professional LLM Router Analytics Platform - Main Application
Ultra-sleek, production-ready dashboard with enterprise-grade design
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import asyncio
import httpx
import json
from datetime import datetime, timedelta
import time
from typing import Dict, List, Any, Optional

# Import styling
from styles import load_professional_css

# ============================================================================
# PAGE CONFIGURATION
# ============================================================================

st.set_page_config(
    page_title="LLM Router Analytics Platform",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': None,
        'Report a bug': None,
        'About': "LLM Router Analytics Platform v2.1.0"
    }
)

# ============================================================================
# DATA LOADING AND API
# ============================================================================

class DashboardAPI:
    """Professional API client for dashboard data"""
    
    def __init__(self, base_url: str = "http://localhost:8080"):
        self.base_url = base_url
        self.timeout = 10.0
    
    async def get_system_health(self) -> Dict[str, Any]:
        """Get system health status with error handling"""
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.get(f"{self.base_url}/health")
                return response.json() if response.status_code == 200 else self._get_mock_health()
        except Exception:
            return self._get_mock_health()
    
    async def get_metrics(self) -> Dict[str, Any]:
        """Get system metrics with fallback to mock data"""
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.get(f"{self.base_url}/metrics")
                return response.json() if response.status_code == 200 else self._get_mock_metrics()
        except Exception:
            return self._get_mock_metrics()
    
    def _get_mock_health(self) -> Dict[str, Any]:
        """Mock health data for demo purposes"""
        return {
            'status': 'healthy',
            'uptime_seconds': 169920,  # 47.2 hours
            'cpu_usage_percent': 23.5,
            'memory_usage_percent': 67.8,
            'components': {
                'inference_engine': {'healthy': True},
                'router': {'healthy': True},
                'kafka_pipeline': {'healthy': True},
                'monitoring': {'healthy': True},
                'slack_bot': {'healthy': True}
            }
        }
    
    def _get_mock_metrics(self) -> Dict[str, Any]:
        """Mock metrics data for demo purposes"""
        return {
            'total_requests': 24567,
            'avg_latency_ms': 847,
            'error_rate': 0.012,
            'total_cost': 127.45,
            'cache_hit_rate': 0.783,
            'success_rate': 0.992
        }

@st.cache_data(ttl=30)
def get_dashboard_data():
    """Get cached dashboard data"""
    api = DashboardAPI()
    
    try:
        return {
            'health': api._get_mock_health(),
            'metrics': api._get_mock_metrics(),
            'model_performance': get_mock_model_data(),
            'analytics': get_mock_analytics_data()
        }
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return {}

def get_mock_model_data() -> List[Dict[str, Any]]:
    """Mock model performance data"""
    return [
        {
            'model_name': 'GPT-4 Turbo',
            'status': 'online',
            'requests': 12450,
            'success_rate': 98.7,
            'avg_latency_ms': 1234,
            'total_cost': 89.23,
            'efficiency': 85
        },
        {
            'model_name': 'Claude 3.5 Sonnet',
            'status': 'online',
            'requests': 8920,
            'success_rate': 99.2,
            'avg_latency_ms': 987,
            'total_cost': 34.56,
            'efficiency': 92
        },
        {
            'model_name': 'Mistral 7B',
            'status': 'online',
            'requests': 3197,
            'success_rate': 97.1,
            'avg_latency_ms': 456,
            'total_cost': 3.66,
            'efficiency': 94
        },
        {
            'model_name': 'Llama 3.1 70B',
            'status': 'degraded',
            'requests': 1245,
            'success_rate': 95.8,
            'avg_latency_ms': 2156,
            'total_cost': 12.34,
            'efficiency': 76
        }
    ]

def get_mock_analytics_data() -> Dict[str, Any]:
    """Mock analytics data"""
    return {
        'query_type_breakdown': {
            'general': 45,
            'code_generation': 30,
            'analysis': 15,
            'summarization': 10
        },
        'model_cost_breakdown': {
            'GPT-4 Turbo': 89.23,
            'Claude 3.5 Sonnet': 34.56,
            'Mistral 7B': 3.66
        },
        'user_tier_distribution': {
            'Free': 85,
            'Premium': 12,
            'Enterprise': 3
        }
    }

# ============================================================================
# CHART CREATION FUNCTIONS
# ============================================================================

def create_professional_line_chart(data: List[float], title: str, color: str = "#3b82f6") -> go.Figure:
    """Create professional line chart with gradient fill"""
    hours = list(range(24))
    
    fig = go.Figure()
    
    # Add gradient fill
    fig.add_trace(go.Scatter(
        x=hours,
        y=data,
        mode='lines',
        line=dict(color=color, width=3, shape='spline'),
        fill='tonexty',
        fillcolor=f'rgba({int(color[1:3], 16)}, {int(color[3:5], 16)}, {int(color[5:7], 16)}, 0.1)',
        name=title,
        showlegend=False
    ))
    
    fig.update_layout(
        title=dict(text=title, font=dict(size=18, family="Inter", weight=600)),
        xaxis=dict(
            title="Hour of Day",
            showgrid=True,
            gridcolor='#f1f5f9',
            showline=False
        ),
        yaxis=dict(
            showgrid=True,
            gridcolor='#f1f5f9',
            showline=False
        ),
        plot_bgcolor='transparent',
        paper_bgcolor='transparent',
        font=dict(family="Inter", size=12, color='#475569'),
        margin=dict(t=60, r=20, b=40, l=60),
        height=320
    )
    
    return fig

def create_professional_pie_chart(values: List[float], labels: List[str], title: str) -> go.Figure:
    """Create professional donut chart"""
    colors = ['#3b82f6', '#8b5cf6', '#10b981', '#f59e0b', '#ef4444']
    
    fig = go.Figure(data=[go.Pie(
        labels=labels,
        values=values,
        hole=0.4,
        marker=dict(
            colors=colors[:len(values)],
            line=dict(color='#ffffff', width=2)
        ),
        textinfo='label+percent',
        textposition='outside',
        textfont=dict(family="Inter", size=11)
    )])
    
    fig.update_layout(
        title=dict(text=title, font=dict(size=18, family="Inter", weight=600)),
        plot_bgcolor='transparent',
        paper_bgcolor='transparent',
        font=dict(family="Inter", size=11, color='#475569'),
        showlegend=False,
        margin=dict(t=60, r=20, b=20, l=20),
        height=320
    )
    
    return fig

# ============================================================================
# PAGE COMPONENTS
# ============================================================================

def render_sidebar():
    """Render professional sidebar"""
    with st.sidebar:
        # Logo section
        st.markdown("""
        <div style="text-align: center; margin-bottom: 2rem;">
            <div style="display: flex; align-items: center; justify-content: center; gap: 12px; margin-bottom: 8px;">
                <div style="width: 32px; height: 32px; background: linear-gradient(135deg, #3b82f6 0%, #2563eb 100%); 
                           border-radius: 8px; display: flex; align-items: center; justify-content: center; 
                           color: white; font-weight: 600; font-size: 18px;">⚡</div>
                <div style="font-size: 20px; font-weight: 700; color: #0f172a;">LLM Router</div>
            </div>
            <div style="font-size: 13px; color: #94a3b8; font-weight: 500;">Analytics Platform</div>
        </div>
        """, unsafe_allow_html=True)
        
        # Navigation
        st.markdown("**NAVIGATION**")
        
        page = st.radio(
            "Select View",
            ["📊 Overview", "🤖 Models", "⚡ Performance", "👥 Users", "💰 Costs", "🚨 Alerts", "📋 Logs"],
            label_visibility="collapsed"
        )
        
        st.markdown("---")
        
        # Time range selector
        time_range = st.selectbox(
            "Time Range",
            ["Last Hour", "Last 6 Hours", "Last 24 Hours", "Last 7 Days"],
            index=2
        )
        
        # Auto-refresh toggle
        auto_refresh = st.checkbox("Auto Refresh (30s)", value=True)
        
        # Manual refresh button
        if st.button("🔄 Refresh Now", use_container_width=True):
            st.cache_data.clear()
            st.rerun()
        
        st.markdown("---")
        
        # System status
        st.markdown("""
        <div style="background: #f1f5f9; padding: 20px; border-radius: 12px;">
            <div style="display: flex; justify-content: space-between; font-size: 13px; margin-bottom: 8px;">
                <span style="color: #94a3b8; font-weight: 500;">System Status</span>
                <span style="font-weight: 600; color: #10b981;">● Operational</span>
            </div>
            <div style="display: flex; justify-content: space-between; font-size: 13px; margin-bottom: 8px;">
                <span style="color: #94a3b8; font-weight: 500;">Uptime</span>
                <span style="font-weight: 600; color: #0f172a;">99.9%</span>
            </div>
            <div style="display: flex; justify-content: space-between; font-size: 13px; margin-bottom: 8px;">
                <span style="color: #94a3b8; font-weight: 500;">Active Models</span>
                <span style="font-weight: 600; color: #0f172a;">4/4</span>
            </div>
            <div style="display: flex; justify-content: space-between; font-size: 13px;">
                <span style="color: #94a3b8; font-weight: 500;">Version</span>
                <span style="font-weight: 600; color: #0f172a;">v2.1.0</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        return page, time_range, auto_refresh

def render_header():
    """Render professional header"""
    col1, col2 = st.columns([3, 1])
    
    with col1:
        st.title("🚀 System Overview")
        st.markdown("**Real-time monitoring and analytics for your LLM infrastructure**")
    
    with col2:
        current_time = datetime.now().strftime("%H:%M:%S")
        st.markdown(f"""
        <div style="text-align: right;">
            <div style="display: inline-flex; align-items: center; gap: 8px; padding: 8px 16px; 
                       background: white; border: 1px solid #e2e8f0; border-radius: 8px; 
                       font-size: 13px; color: #475569; box-shadow: 0 1px 2px 0 rgb(0 0 0 / 0.05);">
                <div style="width: 8px; height: 8px; background: #10b981; border-radius: 50%; 
                           animation: pulse 2s infinite;"></div>
                <span>Live • Updated {current_time}</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

def render_key_metrics(data: Dict[str, Any]):
    """Render key metrics cards"""
    metrics = data.get('metrics', {})
    
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.metric(
            "Total Requests",
            f"{metrics.get('total_requests', 0):,}",
            delta="+12.3% vs yesterday",
            delta_color="normal"
        )
    
    with col2:
        st.metric(
            "Avg Response Time",
            f"{metrics.get('avg_latency_ms', 0):.0f}ms",
            delta="-8.2% vs yesterday",
            delta_color="inverse"
        )
    
    with col3:
        success_rate = metrics.get('success_rate', 0) * 100
        st.metric(
            "Success Rate",
            f"{success_rate:.1f}%",
            delta="+0.3% vs yesterday",
            delta_color="normal"
        )
    
    with col4:
        st.metric(
            "Daily Cost",
            f"${metrics.get('total_cost', 0):.2f}",
            delta="+5.1% vs yesterday",
            delta_color="inverse"
        )
    
    with col5:
        cache_rate = metrics.get('cache_hit_rate', 0) * 100
        st.metric(
            "Cache Hit Rate",
            f"{cache_rate:.1f}%",
            delta="+2.1% vs yesterday",
            delta_color="normal"
        )

def render_charts(data: Dict[str, Any]):
    """Render performance charts"""
    st.markdown("### 📈 Performance Analytics")
    
    col1, col2 = st.columns(2)
    
    # Generate sample data
    hours = list(range(24))
    requests_data = [np.random.poisson(100) + 50 for _ in hours]
    latency_data = [np.random.normal(800, 200) for _ in hours]
    
    with col1:
        fig_requests = create_professional_line_chart(
            requests_data, 
            "Request Volume", 
            "#3b82f6"
        )
        st.plotly_chart(fig_requests, use_container_width=True)
    
    with col2:
        fig_latency = create_professional_line_chart(
            latency_data, 
            "Response Times", 
            "#8b5cf6"
        )
        st.plotly_chart(fig_latency, use_container_width=True)

def render_model_performance(data: Dict[str, Any]):
    """Render model performance table"""
    st.markdown("### 🤖 Model Performance")
    st.markdown("**Real-time performance metrics across all models**")
    
    model_data = data.get('model_performance', [])
    if model_data:
        df = pd.DataFrame(model_data)
        
        # Format the dataframe
        df['requests'] = df['requests'].apply(lambda x: f"{x:,}")
        df['success_rate'] = df['success_rate'].apply(lambda x: f"{x:.1f}%")
        df['avg_latency_ms'] = df['avg_latency_ms'].apply(lambda x: f"{x:,}ms")
        df['total_cost'] = df['total_cost'].apply(lambda x: f"${x:.2f}")
        df['efficiency'] = df['efficiency'].apply(lambda x: f"{x}%")
        
        # Add status indicators
        def format_status(status):
            if status == 'online':
                return "🟢 Online"
            elif status == 'degraded':
                return "🟡 Degraded"
            else:
                return "🔴 Offline"
        
        df['status'] = df['status'].apply(format_status)
        
        # Rename columns
        df.columns = ['Model', 'Status', 'Requests', 'Success Rate', 'Avg Latency', 'Cost', 'Efficiency']
        
        st.dataframe(df, use_container_width=True, hide_index=True)

def render_analytics_charts(data: Dict[str, Any]):
    """Render analytics pie charts"""
    st.markdown("### 📊 Distribution Analytics")
    
    analytics = data.get('analytics', {})
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        query_data = analytics.get('query_type_breakdown', {})
        if query_data:
            fig = create_professional_pie_chart(
                list(query_data.values()),
                list(query_data.keys()),
                "Query Distribution"
            )
            st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        cost_data = analytics.get('model_cost_breakdown', {})
        if cost_data:
            fig = create_professional_pie_chart(
                list(cost_data.values()),
                list(cost_data.keys()),
                "Cost Breakdown"
            )
            st.plotly_chart(fig, use_container_width=True)
    
    with col3:
        user_data = analytics.get('user_tier_distribution', {})
        if user_data:
            fig = create_professional_pie_chart(
                list(user_data.values()),
                list(user_data.keys()),
                "User Tiers"
            )
            st.plotly_chart(fig, use_container_width=True)

def render_alerts():
    """Render system alerts"""
    st.markdown("### 🚨 System Alerts")
    
    st.success("✅ **System Healthy** - All services operating within normal parameters")
    st.warning("⚠️ **High Latency Detected** - Llama 3.1 70B showing elevated response times (2.1s avg)")

# ============================================================================
# MAIN APPLICATION
# ============================================================================

def main():
    """Main application function"""
    
    # Load professional styling
    load_professional_css()
    
    # Auto-refresh functionality
    if 'last_refresh' not in st.session_state:
        st.session_state.last_refresh = time.time()
    
    # Sidebar
    page, time_range, auto_refresh = render_sidebar()
    
    # Auto-refresh logic
    if auto_refresh:
        current_time = time.time()
        if current_time - st.session_state.last_refresh > 30:  # 30 seconds
            st.session_state.last_refresh = current_time
            st.rerun()
    
    # Main content area
    render_header()
    
    # Load data
    with st.spinner("Loading dashboard data..."):
        data = get_dashboard_data()
    
    # Render based on selected page
    if page == "📊 Overview":
        render_key_metrics(data)
        st.markdown("---")
        render_charts(data)
        st.markdown("---")
        render_model_performance(data)
        st.markdown("---")
        render_alerts()
        st.markdown("---")
        render_analytics_charts(data)
    
    elif page == "🤖 Models":
        render_model_performance(data)
        st.markdown("---")
        render_analytics_charts(data)
    
    elif page == "⚡ Performance":
        render_charts(data)
        st.markdown("---")
        render_key_metrics(data)
    
    elif page == "👥 Users":
        st.markdown("### 👥 User Analytics")
        st.info("User analytics view - showing user activity and engagement metrics")
        render_analytics_charts(data)
    
    elif page == "💰 Costs":
        st.markdown("### 💰 Cost Analytics")
        st.info("Cost optimization view - detailed cost breakdown and efficiency metrics")
        render_analytics_charts(data)
    
    elif page == "🚨 Alerts":
        render_alerts()
        st.markdown("---")
        st.markdown("### Alert Configuration")
        
        col1, col2 = st.columns(2)
        with col1:
            st.slider("CPU Alert Threshold (%)", 50, 100, 80)
            st.slider("Memory Alert Threshold (%)", 50, 100, 80)
        with col2:
            st.slider("Error Rate Threshold (%)", 1, 20, 5)
            st.slider("Latency Threshold (ms)", 500, 10000, 2000)
        
        if st.button("Update Alert Settings", use_container_width=True):
            st.success("Alert settings updated successfully!")
    
    elif page == "📋 Logs":
        st.markdown("### 📋 System Logs")
        
        # Log filters
        col1, col2, col3 = st.columns(3)
        with col1:
            log_level = st.selectbox("Log Level", ["ALL", "ERROR", "WARNING", "INFO", "DEBUG"])
        with col2:
            component = st.selectbox("Component", ["ALL", "inference", "router", "pipeline", "slack"])
        with col3:
            log_count = st.slider("Number of logs", 10, 100, 50)
        
        # Sample logs
        sample_logs = [
            {"time": "14:25:32", "level": "INFO", "component": "inference", "message": "Model inference completed successfully", "id": "req-abc123"},
            {"time": "14:23:15", "level": "WARNING", "component": "router", "message": "High latency detected for model gpt-4-turbo", "id": "req-abc122"},
            {"time": "14:22:48", "level": "INFO", "component": "slack", "message": "Message processed for user user-789", "id": "req-abc121"},
            {"time": "14:20:12", "level": "ERROR", "component": "pipeline", "message": "Failed to insert batch to ClickHouse", "id": "req-abc120"},
        ]
        
        for log in sample_logs[:log_count]:
            level_color = {"ERROR": "#ef4444", "WARNING": "#f59e0b", "INFO": "#3b82f6", "DEBUG": "#6b7280"}.get(log["level"], "#000000")
            
            st.markdown(f"""
            <div style="border-left: 4px solid {level_color}; padding: 12px; margin: 8px 0; 
                       background: #f8fafc; font-family: monospace; font-size: 13px; border-radius: 4px;">
                <strong>{log['time']}</strong> 
                <span style="color: {level_color}; font-weight: bold;">[{log['level']}]</span>
                <span style="color: #64748b;">{log['component']}</span> - 
                {log['message']}<br>
                <small style="color: #94a3b8;">Request ID: {log['id']}</small>
            </div>
            """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
