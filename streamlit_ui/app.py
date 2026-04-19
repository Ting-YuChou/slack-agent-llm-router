"""
Professional LLM Router Analytics Platform - Main Application
Ultra-sleek, production-ready dashboard with enterprise-grade design
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import os
import httpx
from datetime import datetime
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
        "Get Help": None,
        "Report a bug": None,
        "About": "LLM Router Analytics Platform v2.1.0",
    },
)

# ============================================================================
# DATA LOADING AND API
# ============================================================================

DEFAULT_API_BASE_URL = os.getenv(
    "LLM_ROUTER_DASHBOARD_API_URL", "http://localhost:8080"
)
TIME_RANGE_TO_HOURS = {
    "Last Hour": 1,
    "Last 6 Hours": 6,
    "Last 24 Hours": 24,
    "Last 7 Days": 168,
}
MAX_HISTORY_POINTS = 60


class DashboardAPI:
    """Professional API client for dashboard data"""

    def __init__(self, base_url: Optional[str] = None):
        self.base_url = (base_url or DEFAULT_API_BASE_URL).rstrip("/")
        self.timeout = 10.0

    def _get(
        self, path: str, params: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Perform a GET request and normalize failures."""
        try:
            with httpx.Client(timeout=self.timeout) as client:
                response = client.get(f"{self.base_url}{path}", params=params)
                response.raise_for_status()
                return response.json()
        except Exception as exc:
            return {"error": str(exc), "path": path}

    def get_dashboard(self, hours: int = 24) -> Dict[str, Any]:
        """Get live dashboard bundle."""
        return self._get("/dashboard", params={"hours": hours})

    def get_logs(
        self,
        limit: int = 50,
        level: Optional[str] = None,
        component: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Get recent structured logs."""
        params: Dict[str, Any] = {"limit": limit}
        if level and level != "ALL":
            params["level"] = level
        if component and component != "ALL":
            params["component"] = component
        return self._get("/dashboard/logs", params=params)


def build_empty_dashboard_payload(error: Optional[str] = None) -> Dict[str, Any]:
    """Return an empty dashboard payload when backend data is unavailable."""
    return {
        "error": error,
        "health": {"status": "unknown", "services": {}},
        "overview": {},
        "analytics": {},
        "model_performance": [],
        "alerts": [],
        "routing_features": {},
        "routing_guardrails": {},
        "routing_policy_state": {},
        "sources": {},
        "capabilities": {},
        "timestamp": time.time(),
    }


@st.cache_data(ttl=30)
def get_dashboard_data(hours: int) -> Dict[str, Any]:
    """Get cached live dashboard data."""
    api = DashboardAPI()
    payload = api.get_dashboard(hours=hours)
    if payload.get("error"):
        return build_empty_dashboard_payload(payload["error"])
    return payload


@st.cache_data(ttl=15)
def get_dashboard_logs(limit: int, level: str, component: str) -> Dict[str, Any]:
    """Get cached live dashboard logs."""
    api = DashboardAPI()
    payload = api.get_logs(limit=limit, level=level, component=component)
    if payload.get("error"):
        return {"error": payload["error"], "logs": []}
    return payload


def safe_number(value: Any, default: float = 0.0) -> float:
    """Convert a possibly missing metric to float."""
    if value is None:
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def format_optional_number(
    value: Any,
    decimals: int = 1,
    suffix: str = "",
    prefix: str = "",
) -> str:
    """Format optional numeric values for tables and cards."""
    if value is None:
        return "-"

    try:
        numeric_value = float(value)
    except (TypeError, ValueError):
        return "-"

    if decimals == 0:
        formatted = f"{numeric_value:,.0f}"
    else:
        formatted = f"{numeric_value:,.{decimals}f}"

    return f"{prefix}{formatted}{suffix}"


def update_live_history(data: Dict[str, Any]):
    """Track live dashboard snapshots across refreshes."""
    history = st.session_state.setdefault("dashboard_history", [])
    overview = data.get("overview", {})
    sample_timestamp = float(data.get("timestamp", time.time()) or time.time())
    if history and abs(history[-1]["sample_timestamp"] - sample_timestamp) < 1:
        return

    history.append(
        {
            "sample_timestamp": sample_timestamp,
            "label": datetime.fromtimestamp(sample_timestamp).strftime("%H:%M:%S"),
            "total_requests": safe_number(overview.get("total_requests")),
            "avg_latency_ms": safe_number(overview.get("avg_latency_ms")),
            "total_cost": safe_number(overview.get("total_cost")),
            "cache_hit_rate": safe_number(overview.get("cache_hit_rate")) * 100,
        }
    )
    st.session_state.dashboard_history = history[-MAX_HISTORY_POINTS:]


# ============================================================================
# CHART CREATION FUNCTIONS
# ============================================================================


def create_professional_line_chart(
    x_values: List[Any],
    y_values: List[float],
    title: str,
    color: str = "#3b82f6",
    x_title: str = "Sample",
):
    """Create professional line chart with gradient fill."""
    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=x_values,
            y=y_values,
            mode="lines+markers",
            line=dict(color=color, width=3, shape="spline"),
            fill="tozeroy",
            fillcolor=f"rgba({int(color[1:3], 16)}, {int(color[3:5], 16)}, {int(color[5:7], 16)}, 0.1)",
            marker=dict(size=7),
            name=title,
            showlegend=False,
        )
    )

    fig.update_layout(
        title=dict(text=title, font=dict(size=18, family="Inter", weight=600)),
        xaxis=dict(title=x_title, showgrid=True, gridcolor="#f1f5f9", showline=False),
        yaxis=dict(showgrid=True, gridcolor="#f1f5f9", showline=False),
        plot_bgcolor="transparent",
        paper_bgcolor="transparent",
        font=dict(family="Inter", size=12, color="#475569"),
        margin=dict(t=60, r=20, b=40, l=60),
        height=320,
    )

    return fig


def create_professional_bar_chart(
    frame: pd.DataFrame,
    x_field: str,
    y_field: str,
    title: str,
    color: str = "#3b82f6",
) -> go.Figure:
    """Create professional bar chart."""
    fig = px.bar(
        frame,
        x=x_field,
        y=y_field,
        color_discrete_sequence=[color],
    )

    fig.update_layout(
        title=dict(text=title, font=dict(size=18, family="Inter", weight=600)),
        plot_bgcolor="transparent",
        paper_bgcolor="transparent",
        font=dict(family="Inter", size=12, color="#475569"),
        showlegend=False,
        margin=dict(t=60, r=20, b=40, l=40),
        height=320,
    )
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(showgrid=True, gridcolor="#f1f5f9")
    return fig


def create_professional_pie_chart(
    values: List[float], labels: List[str], title: str
) -> go.Figure:
    """Create professional donut chart"""
    colors = ["#3b82f6", "#8b5cf6", "#10b981", "#f59e0b", "#ef4444"]

    fig = go.Figure(
        data=[
            go.Pie(
                labels=labels,
                values=values,
                hole=0.4,
                marker=dict(
                    colors=colors[: len(values)], line=dict(color="#ffffff", width=2)
                ),
                textinfo="label+percent",
                textposition="outside",
                textfont=dict(family="Inter", size=11),
            )
        ]
    )

    fig.update_layout(
        title=dict(text=title, font=dict(size=18, family="Inter", weight=600)),
        plot_bgcolor="transparent",
        paper_bgcolor="transparent",
        font=dict(family="Inter", size=11, color="#475569"),
        showlegend=False,
        margin=dict(t=60, r=20, b=20, l=20),
        height=320,
    )

    return fig


# ============================================================================
# PAGE COMPONENTS
# ============================================================================


def render_sidebar():
    """Render professional sidebar"""
    with st.sidebar:
        # Logo section
        st.markdown(
            """
        <div style="text-align: center; margin-bottom: 2rem;">
            <div style="display: flex; align-items: center; justify-content: center; gap: 12px; margin-bottom: 8px;">
                <div style="width: 32px; height: 32px; background: linear-gradient(135deg, #3b82f6 0%, #2563eb 100%); 
                           border-radius: 8px; display: flex; align-items: center; justify-content: center; 
                           color: white; font-weight: 600; font-size: 18px;">⚡</div>
                <div style="font-size: 20px; font-weight: 700; color: #0f172a;">LLM Router</div>
            </div>
            <div style="font-size: 13px; color: #94a3b8; font-weight: 500;">Analytics Platform</div>
        </div>
        """,
            unsafe_allow_html=True,
        )

        # Navigation
        st.markdown("**NAVIGATION**")

        page = st.radio(
            "Select View",
            [
                "📊 Overview",
                "🤖 Models",
                "⚡ Performance",
                "🧭 Routing",
                "👥 Users",
                "💰 Costs",
                "🚨 Alerts",
                "📋 Logs",
            ],
            label_visibility="collapsed",
        )

        st.markdown("---")

        # Time range selector
        time_range = st.selectbox(
            "Time Range",
            ["Last Hour", "Last 6 Hours", "Last 24 Hours", "Last 7 Days"],
            index=2,
        )

        st.caption(
            "Dashboard responses are cached for 30 seconds. Use Refresh Now to fetch a new snapshot."
        )

        # Manual refresh button
        if st.button("🔄 Refresh Now", use_container_width=True):
            st.cache_data.clear()
            st.rerun()

        return page, time_range


def render_sidebar_status(data: Dict[str, Any]):
    """Render live sidebar status after dashboard data loads."""
    health = data.get("health", {})
    services = health.get("services", {})
    capabilities = data.get("capabilities", {})
    sources = data.get("sources", {})
    status = health.get("status", "unknown")

    status_color = {
        "healthy": "#10b981",
        "unhealthy": "#ef4444",
        "unknown": "#f59e0b",
    }.get(status, "#f59e0b")
    status_label = {
        "healthy": "Operational",
        "unhealthy": "Degraded",
        "unknown": "Unavailable",
    }.get(status, "Unavailable")

    active_services = sum(1 for service_ok in services.values() if service_ok)
    total_services = len(services)
    active_models = len(data.get("model_performance", []))
    analytics_source = sources.get("analytics", "unavailable").replace("_", " ")
    pipeline_state = "enabled" if capabilities.get("pipeline_analytics") else "disabled"

    with st.sidebar:
        st.markdown("---")
        st.markdown(
            f"""
        <div style="background: #f1f5f9; padding: 20px; border-radius: 12px;">
            <div style="display: flex; justify-content: space-between; font-size: 13px; margin-bottom: 8px;">
                <span style="color: #94a3b8; font-weight: 500;">System Status</span>
                <span style="font-weight: 600; color: {status_color};">● {status_label}</span>
            </div>
            <div style="display: flex; justify-content: space-between; font-size: 13px; margin-bottom: 8px;">
                <span style="color: #94a3b8; font-weight: 500;">Services Healthy</span>
                <span style="font-weight: 600; color: #0f172a;">{active_services}/{total_services}</span>
            </div>
            <div style="display: flex; justify-content: space-between; font-size: 13px; margin-bottom: 8px;">
                <span style="color: #94a3b8; font-weight: 500;">Visible Models</span>
                <span style="font-weight: 600; color: #0f172a;">{active_models}</span>
            </div>
            <div style="display: flex; justify-content: space-between; font-size: 13px; margin-bottom: 8px;">
                <span style="color: #94a3b8; font-weight: 500;">Analytics Source</span>
                <span style="font-weight: 600; color: #0f172a;">{analytics_source}</span>
            </div>
            <div style="display: flex; justify-content: space-between; font-size: 13px;">
                <span style="color: #94a3b8; font-weight: 500;">Pipeline</span>
                <span style="font-weight: 600; color: #0f172a;">{pipeline_state}</span>
            </div>
        </div>
        """,
            unsafe_allow_html=True,
        )
        st.caption(f"API: `{DEFAULT_API_BASE_URL}`")


def render_header():
    """Render professional header"""
    col1, col2 = st.columns([3, 1])

    with col1:
        st.title("🚀 System Overview")
        st.markdown(
            "**Real-time monitoring and analytics for your LLM infrastructure**"
        )

    with col2:
        current_time = datetime.now().strftime("%H:%M:%S")
        st.markdown(
            f"""
        <div style="text-align: right;">
            <div style="display: inline-flex; align-items: center; gap: 8px; padding: 8px 16px; 
                       background: white; border: 1px solid #e2e8f0; border-radius: 8px; 
                       font-size: 13px; color: #475569; box-shadow: 0 1px 2px 0 rgb(0 0 0 / 0.05);">
                <div style="width: 8px; height: 8px; background: #10b981; border-radius: 50%; 
                           animation: pulse 2s infinite;"></div>
                <span>Live • Updated {current_time}</span>
            </div>
        </div>
        """,
            unsafe_allow_html=True,
        )


def render_key_metrics(data: Dict[str, Any]):
    """Render key metrics cards"""
    metrics = data.get("overview", {})

    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        st.metric(
            "Requests",
            format_optional_number(metrics.get("total_requests"), decimals=0),
        )

    with col2:
        st.metric(
            "Avg Response Time",
            format_optional_number(
                metrics.get("avg_latency_ms"), decimals=0, suffix="ms"
            ),
        )

    with col3:
        st.metric(
            "Success Rate",
            format_optional_number(
                safe_number(metrics.get("success_rate")) * 100,
                decimals=1,
                suffix="%",
            ),
        )

    with col4:
        st.metric(
            "Cost",
            format_optional_number(metrics.get("total_cost"), decimals=2, prefix="$"),
        )

    with col5:
        st.metric(
            "Cache Hit Rate",
            format_optional_number(
                safe_number(metrics.get("cache_hit_rate")) * 100,
                decimals=1,
                suffix="%",
            ),
        )


def render_charts(data: Dict[str, Any]):
    """Render performance charts"""
    st.markdown("### 📈 Live Performance Analytics")

    col1, col2 = st.columns(2)
    history = st.session_state.get("dashboard_history", [])
    model_data = data.get("model_performance", [])

    with col1:
        if len(history) >= 2:
            fig_requests = create_professional_line_chart(
                [point["label"] for point in history],
                [point["total_requests"] for point in history],
                "Requests Across Refresh Samples",
                "#3b82f6",
                x_title="Refresh Sample",
            )
            st.plotly_chart(fig_requests, use_container_width=True)
        elif model_data:
            df = pd.DataFrame(model_data).sort_values("requests", ascending=False)
            fig_requests = create_professional_bar_chart(
                df,
                "model_name",
                "requests",
                "Requests by Model",
                "#3b82f6",
            )
            st.plotly_chart(fig_requests, use_container_width=True)
        else:
            st.info("No live request volume data available yet.")

    with col2:
        if len(history) >= 2:
            fig_latency = create_professional_line_chart(
                [point["label"] for point in history],
                [point["avg_latency_ms"] for point in history],
                "Latency Across Refresh Samples",
                "#8b5cf6",
                x_title="Refresh Sample",
            )
            st.plotly_chart(fig_latency, use_container_width=True)
        elif model_data:
            latency_frame = pd.DataFrame(
                [row for row in model_data if row.get("avg_latency_ms") is not None]
            )
            if not latency_frame.empty:
                fig_latency = create_professional_bar_chart(
                    latency_frame.sort_values("avg_latency_ms", ascending=False),
                    "model_name",
                    "avg_latency_ms",
                    "Latency by Model (ms)",
                    "#8b5cf6",
                )
                st.plotly_chart(fig_latency, use_container_width=True)
            else:
                st.info(
                    "Per-model latency is available when ClickHouse analytics are enabled."
                )
        else:
            st.info("No live latency data available yet.")


def render_model_performance(data: Dict[str, Any]):
    """Render model performance table"""
    st.markdown("### 🤖 Model Performance")
    st.markdown(
        "**Live performance metrics across the models visible to this dashboard**"
    )
    st.caption(
        f"Source: `{data.get('sources', {}).get('model_performance', 'unavailable')}`"
    )

    model_data = data.get("model_performance", [])
    if model_data:
        df = pd.DataFrame(model_data)
        df["requests"] = df["requests"].apply(
            lambda value: format_optional_number(value, decimals=0)
        )
        df["success_rate"] = df["success_rate"].apply(
            lambda value: format_optional_number(value, decimals=1, suffix="%")
        )
        df["avg_latency_ms"] = df["avg_latency_ms"].apply(
            lambda value: format_optional_number(value, decimals=0, suffix="ms")
        )
        df["tokens_per_second"] = df["tokens_per_second"].apply(
            lambda value: format_optional_number(value, decimals=1)
        )
        df["error_count"] = df["error_count"].apply(
            lambda value: format_optional_number(value, decimals=0)
        )
        df["total_cost"] = df["total_cost"].apply(
            lambda value: format_optional_number(value, decimals=2, prefix="$")
        )
        df = df.rename(
            columns={
                "model_name": "Model",
                "requests": "Requests",
                "success_rate": "Success Rate",
                "avg_latency_ms": "Avg Latency",
                "tokens_per_second": "Tokens / Sec",
                "error_count": "Errors",
                "total_cost": "Cost",
                "source": "Source",
            }
        )
        st.dataframe(df, use_container_width=True, hide_index=True)
    else:
        st.info("No model performance data is currently available.")


def render_analytics_charts(data: Dict[str, Any]):
    """Render analytics pie charts"""
    st.markdown("### 📊 Distribution Analytics")
    st.caption(f"Source: `{data.get('sources', {}).get('analytics', 'unavailable')}`")

    analytics = data.get("analytics", {})

    col1, col2, col3 = st.columns(3)

    with col1:
        primary_breakdown = analytics.get("query_type_breakdown") or analytics.get(
            "model_request_distribution", {}
        )
        if primary_breakdown:
            fig = create_professional_pie_chart(
                list(primary_breakdown.values()),
                list(primary_breakdown.keys()),
                "Query Distribution"
                if analytics.get("query_type_breakdown")
                else "Model Request Mix",
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No query distribution data available.")

    with col2:
        cost_data = analytics.get("model_cost_breakdown", {})
        if cost_data:
            fig = create_professional_pie_chart(
                list(cost_data.values()), list(cost_data.keys()), "Cost Breakdown"
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No model cost data available.")

    with col3:
        user_data = analytics.get("user_tier_distribution", {})
        if user_data:
            fig = create_professional_pie_chart(
                list(user_data.values()), list(user_data.keys()), "User Tiers"
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No user tier data available.")


def render_alerts(data: Dict[str, Any]):
    """Render system alerts"""
    st.markdown("### 🚨 System Alerts")
    st.caption(f"Source: `{data.get('sources', {}).get('alerts', 'unavailable')}`")

    alerts = data.get("alerts", [])
    if not alerts:
        st.success("✅ No active alerts detected in live metrics.")
        return

    severity_rank = {"critical": 0, "warning": 1, "info": 2}
    sorted_alerts = sorted(
        alerts,
        key=lambda alert: severity_rank.get(alert.get("severity", "info"), 3),
    )

    for alert in sorted_alerts[:10]:
        severity = alert.get("severity", "info")
        timestamp = alert.get("timestamp") or "unknown time"
        message = (
            f"**{alert.get('title', 'Alert')}** - {alert.get('description', 'No details')}"
            f"\n\nSource: `{alert.get('source', 'unknown')}` • Timestamp: `{timestamp}`"
        )
        if severity == "critical":
            st.error(message)
        elif severity == "warning":
            st.warning(message)
        else:
            st.info(message)


def render_routing_features(data: Dict[str, Any]):
    """Render request-side Flink routing feature summaries."""
    st.markdown("### 🧭 Routing Features")
    st.caption(
        f"Source: `{data.get('sources', {}).get('routing_features', 'unavailable')}`"
    )

    routing_features = data.get("routing_features", {}) or {}
    request_count = int(routing_features.get("request_count", 0) or 0)
    if request_count <= 0:
        st.info("No request-side Flink routing feature data is currently available.")
        return

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Requests", request_count)
    with col2:
        st.metric(
            "Fast Lane Candidates",
            int(routing_features.get("fast_lane_count", 0) or 0),
        )
    with col3:
        st.metric(
            "High Reasoning",
            int(routing_features.get("requires_high_reasoning_count", 0) or 0),
        )

    breakdown_col1, breakdown_col2, breakdown_col3 = st.columns(3)
    breakdowns = [
        ("Query Type", routing_features.get("query_type_breakdown", {})),
        ("Complexity", routing_features.get("query_complexity_breakdown", {})),
        ("Session Hotness", routing_features.get("session_hotness_breakdown", {})),
    ]

    for column, (title, breakdown) in zip(
        [breakdown_col1, breakdown_col2, breakdown_col3], breakdowns
    ):
        with column:
            st.markdown(f"**{title}**")
            if breakdown:
                chart = pd.DataFrame(
                    {
                        "Category": list(breakdown.keys()),
                        "Count": list(breakdown.values()),
                    }
                )
                st.bar_chart(chart.set_index("Category"))
            else:
                st.info(f"No {title.lower()} data available.")

    preference_col1, preference_col2, preference_col3 = st.columns(3)
    preference_sections = [
        ("Preferred Models", routing_features.get("top_preferred_models", {})),
        ("Avoid Models", routing_features.get("top_avoid_models", {})),
        ("Avoid Providers", routing_features.get("top_avoid_providers", {})),
    ]

    for column, (title, items) in zip(
        [preference_col1, preference_col2, preference_col3], preference_sections
    ):
        with column:
            st.markdown(f"**{title}**")
            if items:
                st.dataframe(
                    pd.DataFrame(
                        {"Name": list(items.keys()), "Count": list(items.values())}
                    ),
                    use_container_width=True,
                    hide_index=True,
                )
            else:
                st.info(f"No {title.lower()} data available.")

    recent_requests = routing_features.get("recent_requests", [])
    if recent_requests:
        st.markdown("**Recent Routing Feature Events**")
        st.dataframe(
            pd.DataFrame(recent_requests), use_container_width=True, hide_index=True
        )


def render_routing_guardrails(data: Dict[str, Any]):
    """Render recent routing guardrails from Flink analytics."""
    st.markdown("### 🛡️ Routing Guardrails")
    st.caption(
        f"Source: `{data.get('sources', {}).get('routing_guardrails', 'unavailable')}`"
    )

    routing_guardrails = data.get("routing_guardrails", {}) or {}
    recent_guardrails = routing_guardrails.get("recent_guardrails", []) or []
    persisted_guardrails = routing_guardrails.get("persisted_guardrails", []) or []
    total_count = int(routing_guardrails.get("guardrail_count", 0) or 0)

    if total_count <= 0 and not persisted_guardrails:
        st.success("✅ No active routing guardrails are currently visible.")
        return

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Recent Guardrails", total_count)
    with col2:
        st.metric(
            "Scopes",
            len(routing_guardrails.get("scope_breakdown", {}) or {}),
        )
    with col3:
        st.metric(
            "Triggers",
            len(routing_guardrails.get("trigger_breakdown", {}) or {}),
        )

    breakdown_col1, breakdown_col2 = st.columns(2)
    with breakdown_col1:
        scope_breakdown = routing_guardrails.get("scope_breakdown", {}) or {}
        st.markdown("**Scope Breakdown**")
        if scope_breakdown:
            st.bar_chart(
                pd.DataFrame(
                    {
                        "Scope": list(scope_breakdown.keys()),
                        "Count": list(scope_breakdown.values()),
                    }
                ).set_index("Scope")
            )
        else:
            st.info("No scope breakdown data available.")

    with breakdown_col2:
        trigger_breakdown = routing_guardrails.get("trigger_breakdown", {}) or {}
        st.markdown("**Trigger Breakdown**")
        if trigger_breakdown:
            st.bar_chart(
                pd.DataFrame(
                    {
                        "Trigger": list(trigger_breakdown.keys()),
                        "Count": list(trigger_breakdown.values()),
                    }
                ).set_index("Trigger")
            )
        else:
            st.info("No trigger breakdown data available.")

    if recent_guardrails:
        st.markdown("**Recent Guardrails**")
        st.dataframe(
            pd.DataFrame(recent_guardrails), use_container_width=True, hide_index=True
        )

    if persisted_guardrails:
        st.markdown("**Persisted Guardrails**")
        st.dataframe(
            pd.DataFrame(persisted_guardrails),
            use_container_width=True,
            hide_index=True,
        )


def render_routing_policy_state(data: Dict[str, Any]):
    """Render rolling routing policy state summaries and persisted audit events."""
    st.markdown("### 🧩 Routing Policy State")
    st.caption(
        f"Source: `{data.get('sources', {}).get('routing_policy_state', 'unavailable')}`"
    )

    routing_policy_state = data.get("routing_policy_state", {}) or {}
    recent_states = routing_policy_state.get("recent_states", []) or []
    persisted_states = routing_policy_state.get("persisted_states", []) or []
    total_count = int(routing_policy_state.get("state_count", 0) or 0)

    if total_count <= 0 and not persisted_states:
        st.info("No rolling routing policy state is currently visible.")
        return

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("State Events", total_count)
    with col2:
        st.metric(
            "Burst Protection",
            int(routing_policy_state.get("burst_protection_count", 0) or 0),
        )
    with col3:
        st.metric(
            "Enterprise Priority",
            int(routing_policy_state.get("enterprise_priority_count", 0) or 0),
        )
    with col4:
        st.metric(
            "Fast Lane Routes",
            int(routing_policy_state.get("route_to_fast_lane_count", 0) or 0),
        )

    breakdown_col1, breakdown_col2, breakdown_col3 = st.columns(3)
    breakdowns = [
        ("Scope", routing_policy_state.get("scope_breakdown", {})),
        (
            "Complexity",
            routing_policy_state.get("query_complexity_breakdown", {}),
        ),
        (
            "Dominant Query Type",
            routing_policy_state.get("dominant_query_type_breakdown", {}),
        ),
    ]

    for column, (title, breakdown) in zip(
        [breakdown_col1, breakdown_col2, breakdown_col3], breakdowns
    ):
        with column:
            st.markdown(f"**{title}**")
            if breakdown:
                st.bar_chart(
                    pd.DataFrame(
                        {
                            "Category": list(breakdown.keys()),
                            "Count": list(breakdown.values()),
                        }
                    ).set_index("Category")
                )
            else:
                st.info(f"No {title.lower()} data available.")

    preference_col1, preference_col2, preference_col3 = st.columns(3)
    preference_sections = [
        ("Preferred Models", routing_policy_state.get("top_preferred_models", {})),
        ("Avoid Models", routing_policy_state.get("top_avoid_models", {})),
        ("Avoid Providers", routing_policy_state.get("top_avoid_providers", {})),
    ]

    for column, (title, items) in zip(
        [preference_col1, preference_col2, preference_col3], preference_sections
    ):
        with column:
            st.markdown(f"**{title}**")
            if items:
                st.dataframe(
                    pd.DataFrame(
                        {"Name": list(items.keys()), "Count": list(items.values())}
                    ),
                    use_container_width=True,
                    hide_index=True,
                )
            else:
                st.info(f"No {title.lower()} data available.")

    if recent_states:
        st.markdown("**Recent Rolling Policy State**")
        st.dataframe(
            pd.DataFrame(recent_states), use_container_width=True, hide_index=True
        )

    if persisted_states:
        st.markdown("**Persisted Policy State Audit Trail**")
        st.dataframe(
            pd.DataFrame(persisted_states),
            use_container_width=True,
            hide_index=True,
        )


def render_data_availability_notice(data: Dict[str, Any]):
    """Show data availability or backend connectivity notices."""
    if data.get("error"):
        st.error(
            "Dashboard backend is unavailable. "
            f"API: `{DEFAULT_API_BASE_URL}`\n\n{data['error']}"
        )
        return

    capabilities = data.get("capabilities", {})
    partial_reasons = []
    if not capabilities.get("pipeline_analytics"):
        partial_reasons.append(
            "detailed query and model analytics are limited because the pipeline service is not active in this backend process"
        )
    if not capabilities.get("monitoring"):
        partial_reasons.append(
            "alert history is limited to threshold-based checks because the monitoring service is disabled"
        )
    if not capabilities.get("logs"):
        partial_reasons.append(
            "the configured structured log file is not present, so the Logs page may be empty"
        )

    if partial_reasons:
        st.info("Live dashboard connected. " + "; ".join(partial_reasons) + ".")


def render_logs_page():
    """Render live structured logs page."""
    st.markdown("### 📋 System Logs")

    col1, col2, col3 = st.columns(3)
    with col1:
        log_level = st.selectbox(
            "Log Level", ["ALL", "ERROR", "WARNING", "INFO", "DEBUG"]
        )
    with col2:
        component = st.selectbox(
            "Component",
            [
                "ALL",
                "api",
                "main",
                "inference",
                "router",
                "pipeline",
                "monitoring",
                "slack",
                "bot",
            ],
        )
    with col3:
        log_count = st.slider("Number of logs", 10, 100, 50)

    log_payload = get_dashboard_logs(log_count, log_level, component)
    if log_payload.get("error"):
        st.error(
            "Unable to load structured logs from the backend.\n\n"
            f"{log_payload['error']}"
        )
        return

    logs = log_payload.get("logs", [])
    if not logs:
        st.info("No log entries matched the current filters.")
        return

    for log in logs:
        level_color = {
            "ERROR": "#ef4444",
            "WARNING": "#f59e0b",
            "INFO": "#3b82f6",
            "DEBUG": "#6b7280",
        }.get(log.get("level"), "#000000")
        timestamp = log.get("timestamp") or "-"
        request_id = log.get("request_id") or "-"

        st.markdown(
            f"""
        <div style="border-left: 4px solid {level_color}; padding: 12px; margin: 8px 0;
                   background: #f8fafc; font-family: monospace; font-size: 13px; border-radius: 4px;">
            <strong>{timestamp}</strong>
            <span style="color: {level_color}; font-weight: bold;">[{log.get('level', 'INFO')}]</span>
            <span style="color: #64748b;">{log.get('component', 'app')}</span> -
            {log.get('message', '')}<br>
            <small style="color: #94a3b8;">Request ID: {request_id}</small>
        </div>
        """,
            unsafe_allow_html=True,
        )


# ============================================================================
# MAIN APPLICATION
# ============================================================================


def main():
    """Main application function"""

    # Load professional styling
    load_professional_css()

    # Sidebar
    page, time_range = render_sidebar()
    hours = TIME_RANGE_TO_HOURS.get(time_range, 24)

    # Main content area
    render_header()

    # Load data
    with st.spinner("Loading dashboard data..."):
        data = get_dashboard_data(hours)

    render_sidebar_status(data)
    render_data_availability_notice(data)
    if not data.get("error"):
        update_live_history(data)

    # Render based on selected page
    if page == "📊 Overview":
        render_key_metrics(data)
        st.markdown("---")
        render_charts(data)
        st.markdown("---")
        render_model_performance(data)
        st.markdown("---")
        render_routing_features(data)
        st.markdown("---")
        render_routing_policy_state(data)
        st.markdown("---")
        render_routing_guardrails(data)
        st.markdown("---")
        render_alerts(data)
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

    elif page == "🧭 Routing":
        render_routing_features(data)
        st.markdown("---")
        render_routing_policy_state(data)
        st.markdown("---")
        render_routing_guardrails(data)

    elif page == "👥 Users":
        st.markdown("### 👥 User Analytics")
        st.info(
            "Live user distribution and engagement metrics from the dashboard backend."
        )
        render_analytics_charts(data)

    elif page == "💰 Costs":
        st.markdown("### 💰 Cost Analytics")
        st.info("Live cost distribution based on the models visible to this dashboard.")
        render_analytics_charts(data)

    elif page == "🚨 Alerts":
        render_alerts(data)
        st.markdown("---")
        render_routing_guardrails(data)

    elif page == "📋 Logs":
        render_logs_page()


if __name__ == "__main__":
    main()
