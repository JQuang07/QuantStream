"""
QuantStream Dashboard — Streamlit Application
===============================================
frontend/app.py

Layout
------
  Wide-page layout with two main content tabs and a persistent sidebar:

  ┌──────────────────────────────────────────┬──────────────────┐
  │  [Single Equity]  [ETF Analysis]         │  🤖 AI Chatbot   │
  │                                          │                  │
  │  Ticker input  ·  Period  ·  [Analyze]   │  chat history    │
  │  ─────── metric cards row ──────────     │  chat history    │
  │  ─────── master chart (4 panels) ──      │                  │
  │  Volatility chart  │  SMA Z-Score        │  [type here...]  │
  │  Regime heatmap    │  CVaR chart         │                  │
  └──────────────────────────────────────────┴──────────────────┘

Silent state injection
----------------------
Every chat message silently attaches a ``DashboardStateIn``-compatible
payload built from ``st.session_state`` so the LLM is grounded in the
live chart without the user having to describe it.

Streamlit version requirements
--------------------------------
Requires Streamlit ≥ 1.31 for ``st.write_stream``, ``st.chat_message``,
``st.chat_input``, and ``st.tabs``.
"""

from __future__ import annotations

import os
import uuid
from typing import Optional

import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

# ── Must be the very first Streamlit call ─────────────────────────────────
st.set_page_config(
    page_title    = "QuantStream Dashboard",
    page_icon     = "📈",
    layout        = "wide",
    initial_sidebar_state = "expanded",
    menu_items    = {
        "Get Help"    : None,
        "Report a bug": None,
        "About"       : "QuantStream — AI-powered US Equities Analytics Dashboard",
    },
)

import api_client
from api_client import BackendError, StreamError

# ---------------------------------------------------------------------------
# st.secrets → BACKEND_URL override (Streamlit Community Cloud)
# ---------------------------------------------------------------------------
try:
    _secret_url = st.secrets.get("BACKEND_URL")  # type: ignore[attr-defined]
    if _secret_url:
        api_client.BACKEND_URL = _secret_url.rstrip("/")
except Exception:
    pass  # st.secrets is unavailable outside Community Cloud; env var suffices

# ---------------------------------------------------------------------------
# Design tokens
# ---------------------------------------------------------------------------
_BG          = "#0e1117"          # Streamlit default dark background
_BG_CARD     = "rgba(255,255,255,0.03)"
_GRID        = "rgba(255,255,255,0.06)"
_TMPL        = "plotly_dark"

# Regime palette (matches LLM system-prompt labels exactly)
_REGIME_FILL: dict[str, str] = {
    "Low-Volatility Bull"   : "rgba(46,204,113,0.13)",
    "High-Volatility Bear"  : "rgba(231,76,60,0.13)",
    "Transitional / Ranging": "rgba(243,156,18,0.10)",
}
_REGIME_LINE: dict[str, str] = {
    "Low-Volatility Bull"   : "#2ECC71",
    "High-Volatility Bear"  : "#E74C3C",
    "Transitional / Ranging": "#F39C12",
}

_COLOR_UP    = "#26C281"
_COLOR_DOWN  = "#E74C3C"
_COLOR_MACD  = "#4FC3F7"
_COLOR_SIG   = "#FF8C00"
_COLOR_RSI   = "#9B59B6"
_COLOR_GK    = "#FF8C00"
_COLOR_VOL   = "#4FC3F7"

# Preset tickers
_EQUITY_PRESETS = ["AAPL", "MSFT", "NVDA", "GOOGL", "AMZN", "TSLA", "META"]
_ETF_PRESETS = {
    "Broad Market" : ["SPY", "QQQ", "IWM", "DIA", "VTI"],
    "Sectors"      : ["XLK", "XLF", "XLE", "XLV", "XLI"],
    "Fixed Income" : ["TLT", "HYG", "LQD", "SHY", "BND"],
    "Commodities"  : ["GLD", "SLV", "USO", "UNG", "PDBC"],
    "International": ["EFA", "EEM", "VWO", "FXI", "INDA"],
}
_PERIOD_OPTIONS = ["1mo", "3mo", "6mo", "1y", "2y", "5y"]

# ---------------------------------------------------------------------------
# Custom CSS
# ---------------------------------------------------------------------------
st.markdown(
    """
<style>
/* ── Page chrome ────────────────────────────── */
footer { visibility: hidden; }
#MainMenu { visibility: hidden; }
.block-container { padding-top: 1.2rem; padding-bottom: 1rem; }

/* ── Metric cards ────────────────────────────── */
[data-testid="stMetric"] {
    background: rgba(255,255,255,0.03);
    border-radius: 10px;
    padding: 14px 16px;
    border: 1px solid rgba(255,255,255,0.07);
}
[data-testid="stMetricLabel"] { font-size: 0.78rem; opacity: 0.7; }
[data-testid="stMetricValue"] { font-size: 1.4rem; font-weight: 700; }

/* ── Sidebar width + chat ─────────────────────── */
[data-testid="stSidebar"] {
    min-width: 370px !important;
    max-width: 370px !important;
}
[data-testid="stSidebar"] [data-testid="stChatMessage"] {
    font-size: 0.84rem;
}

/* ── Ticker quick-pick buttons ─────────────────── */
.stButton button {
    border-radius: 6px;
    font-size: 0.78rem;
    padding: 4px 10px;
}

/* ── Tab label ─────────────────────────────────── */
[data-testid="stTab"] { font-weight: 600; }

/* ── Section separator ─────────────────────────── */
hr { border-color: rgba(255,255,255,0.08); margin: 0.6rem 0; }
</style>
""",
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Session-state initialisation
# ---------------------------------------------------------------------------
_DEFAULTS: dict = {
    "equity_data"    : None,     # analytics payload for Single Equity tab
    "equity_ticker"  : "AAPL",
    "equity_period"  : "2y",
    "etf_data"       : None,     # analytics payload for ETF tab
    "etf_ticker"     : "SPY",
    "etf_period"     : "2y",
    "chat_history"   : [],       # [{"role": "user"|"assistant", "content": str}]
    "session_id"     : str(uuid.uuid4()),
    "active_source"  : "equity", # which tab last loaded analytics
}
for _k, _v in _DEFAULTS.items():
    if _k not in st.session_state:
        st.session_state[_k] = _v


# ===========================================================================
# Helper utilities
# ===========================================================================

def _pct(v: Optional[float], decimals: int = 1) -> str:
    if v is None:
        return "N/A"
    return f"{v * 100:.{decimals}f}%"


def _float(v: Optional[float], decimals: int = 2) -> str:
    if v is None:
        return "N/A"
    return f"{v:.{decimals}f}"


def _regime_badge(regime_name: str) -> str:
    """Return an HTML badge string coloured by regime."""
    color = _REGIME_LINE.get(regime_name, "#7F8C8D")
    return (
        f'<span style="background:{color}22;color:{color};'
        f'padding:3px 10px;border-radius:5px;'
        f'font-size:0.82em;font-weight:700;">'
        f"{regime_name}</span>"
    )


def _regime_spans(regime_labels: list[dict]) -> list[dict]:
    """Collapse consecutive same-label entries into {start, end, label} spans."""
    if not regime_labels:
        return []
    spans: list[dict] = []
    start  = regime_labels[0]["date"]
    label  = regime_labels[0]["label"]
    for rec in regime_labels[1:]:
        if rec["label"] != label:
            spans.append({"start": start, "end": rec["date"], "label": label})
            start = rec["date"]
            label = rec["label"]
    spans.append({"start": start, "end": regime_labels[-1]["date"], "label": label})
    return spans


def build_dashboard_state(analytics: dict, ticker: str) -> dict:
    """
    Extract a ``DashboardStateIn``-compatible dict from an analytics payload.
    Used to silently inject live context into every chat request.
    All missing / None values fall back to safe defaults so the backend
    never receives a validation error.
    """
    tech   = analytics.get("technicals_latest", {})
    regime = analytics.get("regime", {})
    risk   = analytics.get("risk", {})
    hist   = risk.get("historical", {})

    return {
        "ticker"          : ticker,
        "current_regime"  : regime.get("current_regime") or "Unknown",
        "regime_state_id" : int(regime.get("current_regime_id") or 0),
        "rolling_vol_21d" : float(tech.get("rolling_vol_21d") or 0.15),
        "gk_vol_21d"      : float(tech.get("gk_vol_21d")      or 0.15),
        "rsi_14"          : float(tech.get("rsi_14")           or 50.0),
        "macd_histogram"  : float(tech.get("macd_histogram")   or 0.0),
        "cvar_95"         : float(hist.get("cvar_95")          or -0.02),
        "last_close"      : float(analytics.get("last_close")  or 0.0),
        "data_as_of"      : analytics.get("data_as_of")        or "",
        "sma_zscore_200d" : tech.get("sma_zscore_200d"),  # may be None
    }


# ===========================================================================
# Chart builders — pure functions, no ``st`` calls inside
# ===========================================================================

def build_master_chart(analytics: dict) -> go.Figure:
    """
    4-panel master chart:
      Row 1  Candlestick + SMA 200 + regime colour bands
      Row 2  Volume bars (green/red)
      Row 3  MACD (histogram + lines)
      Row 4  RSI with overbought/oversold bands
    """
    ohlcv  = analytics.get("ohlcv", [])
    ts     = analytics.get("time_series", {})
    regime = analytics.get("regime", {})
    ticker = analytics.get("ticker", "")

    if not ohlcv:
        return go.Figure()

    dates   = [r["date"]           for r in ohlcv]
    opens   = [r.get("open")       for r in ohlcv]
    highs   = [r.get("high")       for r in ohlcv]
    lows    = [r.get("low")        for r in ohlcv]
    closes  = [r.get("close")      for r in ohlcv]
    volumes = [r.get("volume", 0)  for r in ohlcv]

    fig = make_subplots(
        rows=4, cols=1,
        shared_xaxes=True,
        row_heights=[0.50, 0.12, 0.22, 0.16],
        vertical_spacing=0.01,
    )

    # ── Row 1: Candlestick ─────────────────────────────────────────────────
    fig.add_trace(
        go.Candlestick(
            x=dates, open=opens, high=highs, low=lows, close=closes,
            name=ticker,
            increasing=dict(line=dict(color=_COLOR_UP),   fillcolor=_COLOR_UP),
            decreasing=dict(line=dict(color=_COLOR_DOWN), fillcolor=_COLOR_DOWN),
        ),
        row=1, col=1,
    )

    # SMA 200 overlay
    sma_pts = ts.get("sma_zscore") or []
    sma_valid = [(r["date"], r["sma"]) for r in sma_pts if r.get("sma") is not None]
    if sma_valid:
        fig.add_trace(
            go.Scatter(
                x=[p[0] for p in sma_valid],
                y=[p[1] for p in sma_valid],
                mode="lines", name="SMA 200",
                line=dict(color=_COLOR_SIG, width=1.5, dash="dot"),
                opacity=0.85,
            ),
            row=1, col=1,
        )

    # ── Row 2: Volume ──────────────────────────────────────────────────────
    vol_colors = [
        _COLOR_UP   if (c is not None and o is not None and c >= o) else _COLOR_DOWN
        for c, o in zip(closes, opens)
    ]
    fig.add_trace(
        go.Bar(
            x=dates, y=volumes, name="Volume",
            marker_color=vol_colors, opacity=0.55, showlegend=False,
        ),
        row=2, col=1,
    )

    # ── Row 3: MACD ────────────────────────────────────────────────────────
    macd_pts = ts.get("macd") or []
    if macd_pts:
        m_dates = [r["date"]                for r in macd_pts]
        m_hist  = [r.get("histogram")       for r in macd_pts]
        m_line  = [r.get("macd_line")       for r in macd_pts]
        m_sig   = [r.get("signal_line")     for r in macd_pts]

        hist_colors = [
            _COLOR_UP if (h is not None and h >= 0) else _COLOR_DOWN
            for h in m_hist
        ]
        fig.add_trace(
            go.Bar(x=m_dates, y=m_hist, name="MACD Hist",
                   marker_color=hist_colors, opacity=0.75, showlegend=False),
            row=3, col=1,
        )
        fig.add_trace(
            go.Scatter(x=m_dates, y=m_line, mode="lines", name="MACD",
                       line=dict(color=_COLOR_MACD, width=1.8)),
            row=3, col=1,
        )
        fig.add_trace(
            go.Scatter(x=m_dates, y=m_sig, mode="lines", name="Signal",
                       line=dict(color=_COLOR_SIG, width=1.8)),
            row=3, col=1,
        )

    # ── Row 4: RSI ─────────────────────────────────────────────────────────
    rsi_pts = ts.get("rsi") or []
    if rsi_pts:
        r_dates = [r["date"]      for r in rsi_pts]
        r_vals  = [r.get("value") for r in rsi_pts]
        fig.add_trace(
            go.Scatter(x=r_dates, y=r_vals, mode="lines", name="RSI (14)",
                       line=dict(color=_COLOR_RSI, width=1.8)),
            row=4, col=1,
        )
        fig.add_hline(y=70, line_dash="dash",
                      line_color=f"rgba(231,76,60,0.55)",  line_width=1, row=4, col=1)
        fig.add_hline(y=30, line_dash="dash",
                      line_color=f"rgba(38,194,129,0.55)", line_width=1, row=4, col=1)
        fig.add_hline(y=50, line_dash="dot",
                      line_color="rgba(255,255,255,0.18)", line_width=1, row=4, col=1)
        fig.add_hrect(y0=70, y1=100, fillcolor="rgba(231,76,60,0.05)",
                      line_width=0, row=4, col=1)
        fig.add_hrect(y0=0,  y1=30,  fillcolor="rgba(38,194,129,0.05)",
                      line_width=0, row=4, col=1)

    # ── Regime colour bands on row 1 ───────────────────────────────────────
    regime_labels = ts.get("regime_labels") or []
    if regime.get("available") and regime_labels:
        for span in _regime_spans(regime_labels):
            fill = _REGIME_FILL.get(span["label"], "rgba(127,140,141,0.07)")
            try:
                fig.add_vrect(
                    x0=span["start"], x1=span["end"],
                    fillcolor=fill, line_width=0,
                    row=1, col=1, layer="below",
                )
            except Exception:
                pass  # date range issues — skip gracefully

    # ── Layout ─────────────────────────────────────────────────────────────
    fig.update_layout(
        template=_TMPL, paper_bgcolor=_BG, plot_bgcolor=_BG,
        height=740,
        margin=dict(l=60, r=20, t=16, b=20),
        legend=dict(
            orientation="h", yanchor="bottom", y=1.01,
            xanchor="right",  x=1, font_size=11,
        ),
        xaxis_rangeslider_visible=False,
    )
    for row_n in range(1, 5):
        fig.update_xaxes(showgrid=True, gridcolor=_GRID, row=row_n, col=1)
        fig.update_yaxes(showgrid=True, gridcolor=_GRID, row=row_n, col=1)

    fig.update_yaxes(title_text="Price ($)",  title_font_size=10, row=1, col=1)
    fig.update_yaxes(showticklabels=False,                        row=2, col=1)
    fig.update_yaxes(title_text="MACD",       title_font_size=10, row=3, col=1)
    fig.update_yaxes(title_text="RSI",        title_font_size=10,
                     range=[0, 100],                               row=4, col=1)

    return fig


def build_volatility_chart(analytics: dict) -> go.Figure:
    """
    Overlay of 21-day annualised rolling vol (close-to-close) vs
    Garman-Klass vol — both from the time-series payload.
    """
    ts  = analytics.get("time_series", {})
    fig = go.Figure()

    rv_pts = ts.get("rolling_vol") or []
    if rv_pts:
        fig.add_trace(go.Scatter(
            x=[r["date"] for r in rv_pts],
            y=[r.get("value") for r in rv_pts],
            mode="lines", name="Close-to-Close (21d)",
            line=dict(color=_COLOR_VOL, width=2),
            fill="tozeroy", fillcolor="rgba(79,195,247,0.06)",
        ))

    gk_pts = ts.get("gk_vol") or []
    if gk_pts:
        fig.add_trace(go.Scatter(
            x=[r["date"] for r in gk_pts],
            y=[r.get("value") for r in gk_pts],
            mode="lines", name="Garman-Klass (21d)",
            line=dict(color=_COLOR_GK, width=2, dash="dot"),
        ))

    # Typical vol reference band: 15 %–25 %
    if rv_pts:
        fig.add_hrect(y0=0.15, y1=0.25,
                      fillcolor="rgba(255,255,255,0.03)", line_width=0,
                      annotation_text="Normal range",
                      annotation_font_size=10,
                      annotation_position="right")

    fig.update_layout(
        template=_TMPL, paper_bgcolor=_BG, plot_bgcolor=_BG,
        height=260,
        margin=dict(l=60, r=80, t=36, b=30),
        title=dict(text="Historical Volatility — Close-to-Close vs Garman-Klass (21d Ann.)",
                   font_size=13, x=0),
        yaxis=dict(title="Ann. Vol", tickformat=".0%",
                   showgrid=True, gridcolor=_GRID),
        xaxis=dict(showgrid=True, gridcolor=_GRID),
        legend=dict(orientation="h", y=1.1, x=0, font_size=11),
    )
    return fig


def build_zscore_chart(analytics: dict) -> go.Figure:
    """
    SMA-200 z-score time series with ±1σ and ±2σ reference lines and shaded zones.
    """
    ts  = analytics.get("time_series", {})
    pts = ts.get("sma_zscore") or []
    if not pts:
        return go.Figure()

    dates   = [r["date"]       for r in pts]
    zscores = [r.get("zscore") for r in pts]

    fig = go.Figure()

    # Zero line fill areas
    fig.add_hrect(y0=2, y1=5,   fillcolor="rgba(231,76,60,0.06)",   line_width=0)
    fig.add_hrect(y0=-5, y1=-2, fillcolor="rgba(38,194,129,0.06)",  line_width=0)

    # Z-score line
    fig.add_trace(go.Scatter(
        x=dates, y=zscores, mode="lines", name="Z-Score",
        line=dict(color=_COLOR_RSI, width=2),
        fill="tozeroy", fillcolor="rgba(155,89,182,0.05)",
    ))

    # Reference lines at ±1, ±2
    for level in (2, 1, -1, -2):
        col = "rgba(231,76,60,0.55)" if abs(level) == 2 else "rgba(243,156,18,0.45)"
        fig.add_hline(
            y=level, line_dash="dash", line_color=col, line_width=1,
            annotation_text=f"{level:+d}σ",
            annotation_position="right",
            annotation_font_size=10,
        )
    fig.add_hline(y=0, line_dash="solid",
                  line_color="rgba(255,255,255,0.2)", line_width=1)

    fig.update_layout(
        template=_TMPL, paper_bgcolor=_BG, plot_bgcolor=_BG,
        height=260,
        margin=dict(l=60, r=70, t=36, b=30),
        title=dict(text="Price Z-Score vs 200-Day SMA", font_size=13, x=0),
        yaxis=dict(title="Std. Devs", showgrid=True, gridcolor=_GRID),
        xaxis=dict(showgrid=True, gridcolor=_GRID),
        showlegend=False,
    )
    return fig


def build_regime_heatmap(analytics: dict) -> Optional[go.Figure]:
    """
    HMM state-transition probability matrix rendered as a blue-scale heatmap.
    Returns ``None`` if regime data is unavailable.
    """
    regime = analytics.get("regime", {})
    if not regime.get("available"):
        return None

    matrix = regime.get("transition_matrix") or []
    names  = regime.get("regime_names") or []
    if not matrix or not names:
        return None

    # Short display labels (strip long common prefixes)
    short = (
        n.replace("Low-Volatility ", "")
         .replace("High-Volatility ", "")
         .replace(" / Ranging", "")
        for n in names
    )
    labels = list(short)

    z        = [[round(v or 0, 4) for v in row] for row in matrix]
    text_fmt = [[f"{v:.1%}" for v in row] for row in z]

    fig = go.Figure(go.Heatmap(
        z=z, x=labels, y=labels,
        colorscale=[[0, "#0e1117"], [0.4, "#1a3a4a"],
                    [0.7, "#1a5276"], [1,   "#4FC3F7"]],
        text=text_fmt, texttemplate="%{text}",
        textfont=dict(size=12, color="white"),
        showscale=True, zmin=0, zmax=1,
        colorbar=dict(title="P", tickformat=".0%",
                      lenmode="fraction", len=0.9),
    ))
    fig.update_layout(
        template=_TMPL, paper_bgcolor=_BG, plot_bgcolor=_BG,
        height=260,
        margin=dict(l=20, r=20, t=36, b=20),
        title=dict(text="HMM Transition Probability Matrix", font_size=13, x=0),
        xaxis_title="To →", yaxis_title="From →",
        xaxis=dict(side="bottom"),
    )
    return fig


def build_risk_chart(analytics: dict) -> go.Figure:
    """
    Grouped bar chart comparing historical vs parametric VaR / CVaR at 95 %.
    """
    risk  = analytics.get("risk", {})
    hist  = risk.get("historical", {})
    param = risk.get("parametric", {})
    fig   = go.Figure()

    metrics   = ["VaR (95%)", "CVaR (95%)"]
    h_vals    = [abs(hist.get("var_95") or 0),  abs(hist.get("cvar_95") or 0)]
    p_vals    = [abs(param.get("var_95") or 0), abs(param.get("cvar_95") or 0)]

    fig.add_trace(go.Bar(name="Historical",             x=metrics, y=h_vals,
                         marker_color=_COLOR_VOL, opacity=0.85))
    fig.add_trace(go.Bar(name="Parametric (Normal)",    x=metrics, y=p_vals,
                         marker_color=_COLOR_GK, opacity=0.85))

    fig.update_layout(
        template=_TMPL, paper_bgcolor=_BG, plot_bgcolor=_BG,
        barmode="group",
        height=260,
        margin=dict(l=60, r=20, t=36, b=30),
        title=dict(text="Value-at-Risk & Expected Shortfall — 95%, 1-Day",
                   font_size=13, x=0),
        yaxis=dict(title="1-Day Loss", tickformat=".2%",
                   showgrid=True, gridcolor=_GRID),
        xaxis=dict(showgrid=False),
        legend=dict(orientation="h", y=1.12, x=0, font_size=11),
    )
    return fig


# ===========================================================================
# Shared analytics section renderer
# (called by both the Single Equity tab and the ETF tab)
# ===========================================================================

def render_analytics_section(
    data_key: str,       # session-state key for the analytics dict
    ticker_key: str,     # session-state key for the current ticker string
    period_key: str,     # session-state key for the current period string
    source_label: str,   # "equity" | "etf"  — sets st.session_state.active_source
    preset_tickers: Optional[list[str]] = None,
    preset_groups: Optional[dict] = None,
) -> None:
    """
    Render the full analytics UI for one tab.

    Parameters
    ----------
    data_key       : session-state key that holds the analytics dict.
    ticker_key     : session-state key for the current ticker string.
    period_key     : session-state key for the current period string.
    source_label   : written to ``st.session_state.active_source`` when
                     analytics are loaded, so the sidebar knows which tab's
                     state to inject into chat requests.
    preset_tickers : flat list of quick-pick tickers (Single Equity tab).
    preset_groups  : dict[category, list[ticker]] (ETF tab).
    """

    # ── Ticker quick-pick buttons ──────────────────────────────────────────
    if preset_tickers:
        cols = st.columns(len(preset_tickers))
        for i, sym in enumerate(preset_tickers):
            if cols[i].button(sym, key=f"qp_{source_label}_{sym}", use_container_width=True):
                st.session_state[ticker_key] = sym

    if preset_groups:
        for category, syms in preset_groups.items():
            with st.expander(f"📂 {category}", expanded=False):
                gc = st.columns(len(syms))
                for i, sym in enumerate(syms):
                    if gc[i].button(sym, key=f"qpg_{source_label}_{category}_{sym}",
                                    use_container_width=True):
                        st.session_state[ticker_key] = sym

    # ── Ticker + period input row ──────────────────────────────────────────
    st.markdown("---")
    input_col, period_col, btn_col, _ = st.columns([2, 1.4, 1.2, 4])

    with input_col:
        ticker_input = st.text_input(
            "Ticker Symbol",
            value=st.session_state[ticker_key],
            key=f"ticker_input_{source_label}",
            placeholder="e.g. AAPL",
            label_visibility="collapsed",
        ).upper().strip()

    with period_col:
        period_input = st.selectbox(
            "Period",
            options=_PERIOD_OPTIONS,
            index=_PERIOD_OPTIONS.index(st.session_state.get(period_key, "2y")),
            key=f"period_sel_{source_label}",
            label_visibility="collapsed",
        )

    with btn_col:
        analyze_clicked = st.button(
            "⚡ Analyze",
            key=f"analyze_btn_{source_label}",
            use_container_width=True,
            type="primary",
        )

    # ── Fetch analytics on button click ───────────────────────────────────
    ticker_changed = ticker_input != st.session_state[ticker_key]
    period_changed = period_input != st.session_state.get(period_key)
    should_fetch   = analyze_clicked or (ticker_changed and ticker_input) or period_changed

    if should_fetch and ticker_input:
        st.session_state[ticker_key] = ticker_input
        st.session_state[period_key] = period_input
        with st.spinner(f"Fetching {ticker_input} data and computing analytics…"):
            try:
                data = api_client.get_analytics(ticker_input, period_input)
                st.session_state[data_key]        = data
                st.session_state["active_source"] = source_label
                st.rerun()
            except BackendError as exc:
                st.error(f"**Backend error {exc.status_code}:** {exc.detail}")
                return

    analytics = st.session_state.get(data_key)

    # ── Empty state ────────────────────────────────────────────────────────
    if analytics is None:
        st.markdown(
            """
<div style="text-align:center;padding:60px 20px;opacity:0.55;">
<div style="font-size:3rem;">📈</div>
<p style="font-size:1.1rem;margin-top:1rem;">
Enter a ticker symbol above and click <strong>⚡ Analyze</strong> to load the dashboard.
</p>
<p style="font-size:0.85rem;">
Default period is <strong>2y</strong> — recommended for 200-day SMA and HMM regime detection.
</p>
</div>
""",
            unsafe_allow_html=True,
        )
        return

    # ── Header ─────────────────────────────────────────────────────────────
    ticker   = analytics.get("ticker", "")
    tech     = analytics.get("technicals_latest", {})
    regime   = analytics.get("regime", {})
    risk     = analytics.get("risk", {})
    hist_risk= risk.get("historical", {})

    close    = analytics.get("last_close")
    as_of    = analytics.get("data_as_of", "")
    n_days   = analytics.get("n_trading_days", 0)

    header_l, header_r = st.columns([3, 1])
    with header_l:
        st.markdown(
            f"## {ticker}   "
            f"<span style='font-size:1.4rem;font-weight:400;'>"
            f"${close:,.2f}</span>"
            f"<span style='font-size:0.85rem;color:rgba(255,255,255,0.45);'>"
            f"  ·  as of {as_of}  ·  {n_days} trading days</span>",
            unsafe_allow_html=True,
        )
    with header_r:
        if regime.get("available"):
            st.markdown(
                "<div style='text-align:right;padding-top:12px;'>"
                + _regime_badge(regime.get("current_regime", "Unknown"))
                + "</div>",
                unsafe_allow_html=True,
            )

    # ── Metric cards row ───────────────────────────────────────────────────
    m1, m2, m3, m4, m5, m6 = st.columns(6)

    rsi_val = tech.get("rsi_14")
    vol_val = tech.get("rolling_vol_21d")
    gk_val  = tech.get("gk_vol_21d")
    cvar_v  = hist_risk.get("cvar_95")
    macd_v  = tech.get("macd_histogram")
    zscore  = tech.get("sma_zscore_200d")

    with m1:
        st.metric(
            "RSI (14)",
            f"{rsi_val:.1f}" if rsi_val else "N/A",
            delta=f"{rsi_val - 50:+.1f} vs neutral" if rsi_val else None,
            delta_color="inverse" if rsi_val and rsi_val > 70 else "normal",
            help="14-period Wilder RSI.  >70 overbought · <30 oversold.",
        )
    with m2:
        st.metric(
            "Rolling Vol 21d",
            _pct(vol_val),
            delta=f"{(vol_val - 0.20) * 100:+.1f}pp vs 20%" if vol_val else None,
            delta_color="inverse",
            help="Annualised 21-day close-to-close historical volatility.",
        )
    with m3:
        st.metric(
            "GK Vol 21d",
            _pct(gk_val),
            help="Annualised Garman-Klass OHLC volatility estimator.",
        )
    with m4:
        st.metric(
            "MACD Histogram",
            f"{macd_v:+.5f}" if macd_v is not None else "N/A",
            delta_color="normal",
            help="MACD histogram (12/26/9 EMA).  Positive = bullish momentum.",
        )
    with m5:
        st.metric(
            "SMA Z-Score (200d)",
            f"{zscore:+.2f}σ" if zscore is not None else "N/A",
            delta_color="inverse" if zscore and abs(zscore) > 2 else "normal",
            help="Standard deviations of price above/below its 200-day SMA.",
        )
    with m6:
        st.metric(
            "CVaR 95% (1d)",
            _pct(cvar_v, 2) if cvar_v else "N/A",
            help="Historical Expected Shortfall: avg. loss on worst 5% of days.",
        )

    st.markdown("---")

    # ── Master chart ───────────────────────────────────────────────────────
    with st.container():
        st.plotly_chart(
            build_master_chart(analytics),
            use_container_width=True,
            config={"displayModeBar": True, "displaylogo": False,
                    "modeBarButtonsToRemove": ["autoScale2d"]},
        )

    # ── Secondary charts (2 columns × 2 rows) ─────────────────────────────
    row2_l, row2_r = st.columns(2)

    with row2_l:
        st.plotly_chart(
            build_volatility_chart(analytics),
            use_container_width=True,
            config={"displaylogo": False},
        )

    with row2_r:
        if analytics.get("time_series", {}).get("sma_zscore"):
            st.plotly_chart(
                build_zscore_chart(analytics),
                use_container_width=True,
                config={"displaylogo": False},
            )
        else:
            st.info(
                "📐 SMA Z-Score requires ≥ 200 trading days.  "
                "Try period **2y** or **5y**."
            )

    row3_l, row3_r = st.columns(2)

    with row3_l:
        heatmap_fig = build_regime_heatmap(analytics)
        if heatmap_fig:
            st.plotly_chart(heatmap_fig, use_container_width=True,
                            config={"displaylogo": False})
        else:
            reason = regime.get("reason", "Regime data unavailable.")
            st.info(f"🔬 **Regime classifier:** {reason}")

    with row3_r:
        st.plotly_chart(
            build_risk_chart(analytics),
            use_container_width=True,
            config={"displaylogo": False},
        )

    # ── Regime state statistics expander ──────────────────────────────────
    if regime.get("available"):
        with st.expander("📊 HMM State Statistics", expanded=False):
            names   = regime.get("regime_names", [])
            means   = regime.get("state_means", [])
            stds    = regime.get("state_stds", [])
            ll      = regime.get("log_likelihood")
            n_train = regime.get("n_training_samples")
            current = regime.get("current_regime", "")

            header_cols = st.columns([3, 2, 2, 2])
            header_cols[0].markdown("**State**")
            header_cols[1].markdown("**Mean Return**")
            header_cols[2].markdown("**Return Std Dev**")
            header_cols[3].markdown("**Status**")

            for i, name in enumerate(names):
                row_cols = st.columns([3, 2, 2, 2])
                mu_pct = f"{means[i] * 100:+.4f}%" if i < len(means) else "N/A"
                sd_pct = f"{stds[i]  * 100:.4f}%"  if i < len(stds)  else "N/A"
                badge  = "**← current**" if name == current else ""
                row_cols[0].markdown(_regime_badge(name), unsafe_allow_html=True)
                row_cols[1].markdown(mu_pct)
                row_cols[2].markdown(sd_pct)
                row_cols[3].markdown(badge)

            st.markdown(
                f"<small style='opacity:0.5;'>Log-likelihood: {ll:.2f} &nbsp;·&nbsp; "
                f"Training samples: {n_train}</small>",
                unsafe_allow_html=True,
            )


# ===========================================================================
# Main layout
# ===========================================================================

# ── App header ──────────────────────────────────────────────────────────────
st.markdown(
    "<h1 style='margin-bottom:0;'>📈 QuantStream Dashboard</h1>"
    "<p style='margin-top:4px;opacity:0.5;font-size:0.9rem;'>"
    "AI-powered quantitative analytics for US equities  ·  "
    "HMM regime detection  ·  Gemini strategy assistant"
    "</p>",
    unsafe_allow_html=True,
)

# ── Two main content tabs ────────────────────────────────────────────────────
tab_equity, tab_etf = st.tabs(["📊 Single Equity", "🏦 ETF Analysis"])

with tab_equity:
    render_analytics_section(
        data_key      = "equity_data",
        ticker_key    = "equity_ticker",
        period_key    = "equity_period",
        source_label  = "equity",
        preset_tickers= _EQUITY_PRESETS,
    )

with tab_etf:
    st.markdown(
        "<p style='font-size:0.88rem;opacity:0.6;margin-bottom:4px;'>"
        "Select from curated ETF categories below, or type any ETF symbol."
        "</p>",
        unsafe_allow_html=True,
    )
    render_analytics_section(
        data_key     = "etf_data",
        ticker_key   = "etf_ticker",
        period_key   = "etf_period",
        source_label = "etf",
        preset_groups= _ETF_PRESETS,
    )


# ===========================================================================
# Sidebar — AI Chatbot
# Rendered AFTER the main tabs so it reads any analytics freshly loaded
# during this same script execution.
# ===========================================================================

with st.sidebar:

    # ── Header ──────────────────────────────────────────────────────────────
    st.markdown(
        "<h2 style='margin-bottom:2px;'>🤖 QuantStream AI</h2>"
        "<p style='font-size:0.8rem;opacity:0.55;margin-top:0;'>"
        "Strategy assistant grounded in live chart data</p>",
        unsafe_allow_html=True,
    )

    # ── Backend connectivity badge ──────────────────────────────────────────
    # Cache the health check in session state; re-probe only on new session.
    if "backend_status" not in st.session_state:
        st.session_state["backend_status"] = api_client.health_check()

    _status = st.session_state["backend_status"]
    _s_label = _status.get("status", "unknown")
    _s_color = {"ok": "#2ECC71", "degraded": "#F39C12"}.get(_s_label, "#E74C3C")
    st.markdown(
        f'<span style="font-size:0.78rem;color:{_s_color};">'
        f"⬤ Backend: {_s_label}</span>",
        unsafe_allow_html=True,
    )

    # ── Context indicator ────────────────────────────────────────────────────
    _src = st.session_state.get("active_source", "equity")
    _ctx_data = st.session_state.get(
        "equity_data" if _src == "equity" else "etf_data"
    )
    _ctx_ticker = st.session_state.get(
        "equity_ticker" if _src == "equity" else "etf_ticker", ""
    )

    if _ctx_data:
        _ctx_regime = _ctx_data.get("regime", {}).get("current_regime", "Unknown")
        st.markdown(
            f"<div style='font-size:0.78rem;opacity:0.6;margin-bottom:4px;'>"
            f"Context: <strong>{_ctx_ticker}</strong> · "
            + _regime_badge(_ctx_regime)
            + "</div>",
            unsafe_allow_html=True,
        )
    else:
        st.caption("⚠️ Load analytics first to ground AI responses in live data.")

    # ── Session controls ─────────────────────────────────────────────────────
    c1, c2 = st.columns(2)
    with c1:
        if st.button("🔄 New Session", use_container_width=True,
                     help="Clear chat history and start a fresh session."):
            st.session_state["chat_history"] = []
            st.session_state["session_id"]   = str(uuid.uuid4())
            st.session_state["backend_status"] = api_client.health_check()
            st.rerun()
    with c2:
        st.markdown(
            f"<div style='font-size:0.7rem;opacity:0.4;padding-top:8px;"
            f"text-align:center;font-family:monospace;'>"
            f"{st.session_state['session_id'][:8]}…</div>",
            unsafe_allow_html=True,
        )

    st.markdown("---")

    # ── Conversation history ─────────────────────────────────────────────────
    # Render the committed history (all turns except the one currently streaming).
    # New turns are appended and rendered live via st.write_stream below.
    for msg in st.session_state["chat_history"]:
        avatar = "🧑‍💻" if msg["role"] == "user" else "🤖"
        with st.chat_message(msg["role"], avatar=avatar):
            st.markdown(msg["content"])

    # ── Input and streaming ──────────────────────────────────────────────────
    prompt = st.chat_input(
        "Ask about the chart, regime, or strategy…",
        key="sidebar_chat_input",
    )

    if prompt:
        # ── Display user message immediately ──────────────────────────────
        with st.chat_message("user", avatar="🧑‍💻"):
            st.markdown(prompt)

        # ── Build dashboard state payload ──────────────────────────────────
        if _ctx_data:
            dash_state = build_dashboard_state(_ctx_data, _ctx_ticker)
        else:
            # Graceful fallback when no analytics are loaded yet
            dash_state = {
                "ticker"          : "N/A",
                "current_regime"  : "Unknown",
                "regime_state_id" : 0,
                "rolling_vol_21d" : 0.15,
                "gk_vol_21d"      : 0.15,
                "rsi_14"          : 50.0,
                "macd_histogram"  : 0.0,
                "cvar_95"         : -0.02,
                "last_close"      : 0.0,
                "data_as_of"      : "",
                "sma_zscore_200d" : None,
            }

        # ── Stream assistant response ──────────────────────────────────────
        with st.chat_message("assistant", avatar="🤖"):
            full_response: Optional[str] = None
            try:
                token_gen = api_client.stream_chat(
                    message        = prompt,
                    dashboard_state= dash_state,
                    session_id     = st.session_state["session_id"],
                    history        = [
                        {"role": m["role"], "content": m["content"]}
                        for m in st.session_state["chat_history"]
                    ],
                )
                # st.write_stream renders tokens progressively and returns
                # the complete concatenated string when exhausted.
                full_response = st.write_stream(token_gen)

            except StreamError as exc:
                st.error(f"⚠️ **AI Error:** {exc}")

            except BackendError as exc:
                if exc.status_code == 503:
                    st.warning(
                        "⚡ **Backend offline.** "
                        "The Render free-tier service may be asleep. "
                        "Try again in ~30 seconds."
                    )
                elif exc.status_code == 504:
                    st.warning(
                        "⏱️ **Request timed out.** "
                        "The backend may still be waking up — please retry."
                    )
                else:
                    st.error(f"⚠️ **Backend error {exc.status_code}:** {exc.detail}")

            except Exception as exc:
                st.error(f"⚠️ **Unexpected error:** {exc}")

        # ── Persist to history ─────────────────────────────────────────────
        st.session_state["chat_history"].append(
            {"role": "user", "content": prompt}
        )
        if full_response:
            st.session_state["chat_history"].append(
                {"role": "assistant", "content": full_response}
            )

    # ── Usage tips (shown when chat is empty) ────────────────────────────────
    if not st.session_state["chat_history"]:
        st.markdown(
            """
<div style='font-size:0.78rem;opacity:0.42;padding:10px 4px;'>
<strong>💡 Try asking:</strong><br>
• "What does the current regime mean for momentum strategies?"<br>
• "Is the RSI signalling anything I should act on?"<br>
• "How should I size a position given this CVaR?"<br>
• "What's the MACD telling us right now?"<br>
• "Is this an overbought situation or sustained trend?"
</div>
""",
            unsafe_allow_html=True,
        )