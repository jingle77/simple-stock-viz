from __future__ import annotations

import os
from datetime import datetime, timedelta, date
from typing import Optional
import numpy as np

import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st
from dotenv import load_dotenv


# --- Streamlit page config ---
st.set_page_config(
    page_title="Stock vs S&P 500 (Earnings & Dividends)",
    layout="wide",
)


# -------------------------------
# Config & utility helpers
# -------------------------------
def load_api_key() -> str:
    """
    Load FMP_API_KEY from .env using python-dotenv.
    Stop the Streamlit app with a clear error if missing.
    """
    load_dotenv()
    api_key = os.getenv("FMP_API_KEY")

    if not api_key:
        st.error(
            "FMP_API_KEY is missing.\n\n"
            "Create a `.env` file in the same directory as this app with:\n"
            "FMP_API_KEY=your_api_key_here"
        )
        st.stop()

    return api_key


def _safe_get(url: str, params: dict) -> dict | list:
    """
    Wrapper around requests.get with basic error handling.
    Returns parsed JSON (dict or list), or an empty list on failure.
    """
    try:
        resp = requests.get(url, params=params, timeout=30)
        resp.raise_for_status()
        return resp.json()
    except requests.RequestException as exc:
        st.error(f"API request failed: {exc}")
        return []


def _extract_historical_price_payload(data: dict | list) -> list[dict]:
    """
    FMP historical endpoints can return either:
      - { "symbol": "...", "historical": [ ... ] }
      - [ { ... }, { ... } ]
    This helper normalizes to a list of dicts.
    """
    if isinstance(data, dict):
        if isinstance(data.get("historical"), list):
            return data["historical"]
        return []
    if isinstance(data, list):
        return data
    return []


def _extract_list_from_payload(data: dict | list, key: str | None = None) -> list[dict]:
    """
    Normalize FMP responses for earnings/dividends to a list.
    Many endpoints return either:
      - [ {...}, {...} ]
      - { key: [ {...}, {...} ] }
    """
    if isinstance(data, dict) and key and isinstance(data.get(key), list):
        return data[key]
    if isinstance(data, list):
        return data
    return []


def _normalize_price_df(raw: dict | list, symbol: str) -> pd.DataFrame:
    """
    Turn raw JSON from FMP historical price endpoint into a normalized DataFrame
    with at least: date (datetime), adjClose, close, price.
    """
    rows = _extract_historical_price_payload(raw)
    if not rows:
        return pd.DataFrame(columns=["date", "adjClose", "close", "price"])

    df = pd.DataFrame(rows)

    # Ensure date column exists and is datetime
    if "date" not in df.columns:
        return pd.DataFrame(columns=["date", "adjClose", "close", "price"])

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"])

    # Build adjClose
    if "adjClose" in df.columns:
        pass
    elif "adjustedClose" in df.columns:
        df["adjClose"] = df["adjustedClose"]
    elif "close" in df.columns:
        df["adjClose"] = df["close"]
    else:
        df["adjClose"] = pd.NA

    # Ensure close exists
    if "close" not in df.columns:
        df["close"] = df["adjClose"]

    # price = adjClose (main plotting series)
    df["price"] = df["adjClose"]

    # Drop rows where we still don't have a price
    df = df.dropna(subset=["price"])

    # Sort by ascending date
    df = df.sort_values("date").reset_index(drop=True)

    # Attach symbol if not present
    if "symbol" not in df.columns:
        df["symbol"] = symbol

    return df[["date", "adjClose", "close", "price", "symbol"]]


def _normalize_earnings_df(raw: dict | list) -> pd.DataFrame:
    """
    Normalize FMP earnings into columns:
    date, epsActual, epsEstimated, revenueActual, revenueEstimated
    """
    rows = _extract_list_from_payload(raw, key="earnings")
    if not rows:
        return pd.DataFrame(
            columns=["date", "epsActual", "epsEstimated", "revenueActual", "revenueEstimated"]
        )

    df = pd.DataFrame(rows)

    if "date" not in df.columns:
        return pd.DataFrame(
            columns=["date", "epsActual", "epsEstimated", "revenueActual", "revenueEstimated"]
        )

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"])

    for col in ["epsActual", "epsEstimated", "revenueActual", "revenueEstimated"]:
        if col not in df.columns:
            df[col] = pd.NA

    df = df.sort_values("date").reset_index(drop=True)
    return df[["date", "epsActual", "epsEstimated", "revenueActual", "revenueEstimated"]]


def _normalize_dividends_df(raw: dict | list) -> pd.DataFrame:
    """
    Normalize FMP dividends.

    Keep:
      - date
      - adjDividend (if present)
      - dividend (if present)
      - paymentDate (if present)
      - frequency (if present)
    """
    rows = _extract_list_from_payload(raw, key=None)
    if not rows:
        return pd.DataFrame(columns=["date", "adjDividend", "dividend", "paymentDate", "frequency"])

    df = pd.DataFrame(rows)

    if "date" not in df.columns:
        return pd.DataFrame(columns=["date", "adjDividend", "dividend", "paymentDate", "frequency"])

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"])

    # Parse paymentDate if present
    if "paymentDate" in df.columns:
        df["paymentDate"] = pd.to_datetime(df["paymentDate"], errors="coerce")

    # Ensure frequency exists
    if "frequency" not in df.columns:
        df["frequency"] = pd.NA

    df = df.sort_values("date").reset_index(drop=True)

    cols = ["date"]
    for col in ["adjDividend", "dividend", "paymentDate", "frequency"]:
        if col in df.columns:
            cols.append(col)

    return df[cols]


def filter_to_last_year(df: pd.DataFrame, reference_end_date: pd.Timestamp) -> pd.DataFrame:
    """
    Filter a time series DataFrame (with a 'date' column) to the last 365 days
    ending at reference_end_date (inclusive).
    """
    if df.empty:
        return df

    start_date = reference_end_date - pd.Timedelta(days=365)
    mask = (df["date"] >= start_date) & (df["date"] <= reference_end_date)
    return df.loc[mask].copy()


def format_billions(value) -> str:
    """
    Turn a large numeric value into a B/M/K string.
    (Kept here in case you want it later; the current chart function
    doesn't use revenue labels.)
    """
    try:
        v = float(value)
    except (TypeError, ValueError):
        return "N/A"

    abs_v = abs(v)
    if abs_v >= 1e12:
        return f"{v / 1e12:.1f}T"
    if abs_v >= 1e9:
        return f"{v / 1e9:.1f}B"
    if abs_v >= 1e6:
        return f"{v / 1e6:.1f}M"
    if abs_v >= 1e3:
        return f"{v / 1e3:.1f}K"
    return f"{v:.0f}"


# -------------------------------
# FMP fetchers (cached)
# -------------------------------
@st.cache_data(show_spinner=False)
def fetch_price_history(symbol: str, api_key: str) -> pd.DataFrame:
    """
    Fetch historical daily price data for a symbol from FMP and normalize.

    Endpoint:
      https://financialmodelingprep.com/stable/historical-price-eod/full
    """
    url = "https://financialmodelingprep.com/stable/historical-price-eod/full"
    params = {"symbol": symbol, "apikey": api_key}
    raw = _safe_get(url, params=params)
    df = _normalize_price_df(raw, symbol)

    if df.empty:
        return df

    # Fetch ~1.5 years to be safe, then app will filter to last 1 year
    today = pd.Timestamp(datetime.utcnow().date())
    start_fetch = today - pd.Timedelta(days=550)
    df = df[df["date"] >= start_fetch].reset_index(drop=True)

    return df


@st.cache_data(show_spinner=False)
def fetch_sp500_history(api_key: str) -> pd.DataFrame:
    """
    Fetch S&P 500 (^GSPC) historical daily price data.
    Uses same endpoint and normalization as fetch_price_history.
    """
    symbol = "^GSPC"
    url = "https://financialmodelingprep.com/stable/historical-price-eod/full"
    params = {"symbol": symbol, "apikey": api_key}
    raw = _safe_get(url, params=params)
    df = _normalize_price_df(raw, symbol)

    if df.empty:
        return df

    today = pd.Timestamp(datetime.utcnow().date())
    start_fetch = today - pd.Timedelta(days=550)
    df = df[df["date"] >= start_fetch].reset_index(drop=True)

    return df


@st.cache_data(show_spinner=False)
def fetch_earnings(symbol: str, api_key: str) -> pd.DataFrame:
    """
    Fetch earnings history for a symbol from FMP and normalize.

    Endpoint:
      https://financialmodelingprep.com/stable/earnings
    """
    url = "https://financialmodelingprep.com/stable/earnings"
    params = {"symbol": symbol, "apikey": api_key}
    raw = _safe_get(url, params=params)
    df = _normalize_earnings_df(raw)
    return df


@st.cache_data(show_spinner=False)
def fetch_dividends(symbol: str, api_key: str) -> pd.DataFrame:
    """
    Fetch dividend history for a symbol from FMP and normalize.

    Endpoint:
      https://financialmodelingprep.com/stable/dividends
    """
    url = "https://financialmodelingprep.com/stable/dividends"
    params = {"symbol": symbol, "apikey": api_key}
    raw = _safe_get(url, params=params)
    df = _normalize_dividends_df(raw)
    return df


# -------------------------------
# Chart builder (from your other project)
# -------------------------------
def build_price_and_events_chart(
    symbol: str,
    as_of_date: date,
    stock_prices: pd.DataFrame,
    sp500_prices: pd.DataFrame,
    *,
    company_name: Optional[str] = None,  # kept for future flexibility, not used in title
    earnings_events: Optional[pd.DataFrame] = None,
    dividend_events: Optional[pd.DataFrame] = None,
    **_: dict,
) -> go.Figure:
    """
    Dual-axis *indexed* price chart: stock vs S&P 500, but axes labeled in actual prices.

    - Both series are rebased to 100 at the first common date in the data window.
    - Left y-axis: labeled in stock PRICE (but underlying data is indexed).
    - Right y-axis: labeled in S&P 500 PRICE (underlying data also indexed).
    - Earnings markers: EPS estimate / actual / Δ near the (indexed) price line.
    - Dividend markers: '$<amount>' labels, larger, dark text.
    """
    # --- Prep price data ---
    stock = stock_prices.copy()
    stock["date"] = pd.to_datetime(stock["date"])
    stock = stock.sort_values("date")

    spx = sp500_prices.copy()
    spx["date"] = pd.to_datetime(spx["date"])
    spx = spx.sort_values("date")

    # Ensure columns exist
    if "adjClose" not in stock.columns or "adjClose" not in spx.columns:
        raise ValueError("Expected 'adjClose' column in both stock and sp500 dataframes.")

    # ------------------------------------------------------
    # Rebase both series to 100 at the first common date
    # ------------------------------------------------------
    common_dates = sorted(set(stock["date"]).intersection(set(spx["date"])))
    if not common_dates:
        raise ValueError("No overlapping dates between stock and S&P 500 series.")

    base_date = common_dates[0]

    # Restrict both series to dates >= base_date (so both start together)
    stock = stock[stock["date"] >= base_date].copy()
    spx = spx[spx["date"] >= base_date].copy()

    # Get base values on the base_date
    stock_base = float(stock.loc[stock["date"] == base_date, "adjClose"].iloc[0])
    spx_base = float(spx.loc[spx["date"] == base_date, "adjClose"].iloc[0])

    stock["indexed"] = stock["adjClose"] / stock_base * 100.0
    spx["indexed"] = spx["adjClose"] / spx_base * 100.0

    fig = go.Figure()

    # --- Stock trace (left axis, indexed) ---
    fig.add_trace(
        go.Scatter(
            x=stock["date"],
            y=stock["indexed"],
            mode="lines",
            name=symbol,
            hovertemplate=f"{symbol}: %{{y:.2f}}<extra></extra>",
            yaxis="y",  # primary
        )
    )

    # --- S&P 500 trace (right axis, indexed) ---
    fig.add_trace(
        go.Scatter(
            x=spx["date"],
            y=spx["indexed"],
            mode="lines",
            name="S&P 500",
            line=dict(dash="dot"),
            hovertemplate="S&P 500: %{y:.2f}<extra></extra>",
            yaxis="y2",  # secondary axis
        )
    )

    # --- Vertical analysis-date line as a shape (does NOT affect y-scale) ---
    as_of_ts = pd.to_datetime(as_of_date)
    fig.add_shape(
        type="line",
        x0=as_of_ts,
        x1=as_of_ts,
        y0=0,
        y1=1,
        xref="x",
        yref="paper",  # spans full height of the plotting area
        line=dict(color="gray", dash="dash", width=1),
    )

    # ------------------------------------------------------------------
    # Earnings & dividends markers (if data is provided)
    # ------------------------------------------------------------------
    # Attach events to the *indexed* stock series so labels sit on/near the line.
    stock_for_merge = stock[["date", "indexed"]].rename(columns={"indexed": "price"})
    stock_for_merge = stock_for_merge.sort_values("date")

    # Colors
    pos_color = "#006400"  # dark green for positive Δ
    neg_color = "#8B0000"  # dark red for negative Δ
    neu_color = "#222222"  # dark neutral for flat / missing
    earnings_marker_color = "#1f77b4"  # distinct blue
    dividends_marker_color = "#ff7f0e"  # distinct orange

    # --- Earnings markers ---
    if earnings_events is not None and not earnings_events.empty:
        ed = earnings_events.copy()
        ed["date"] = pd.to_datetime(ed["date"])
        ed = ed.sort_values("date")

        merged_ed = pd.merge_asof(
            ed,
            stock_for_merge,
            on="date",
            direction="backward",
        )

        # Scatter markers (visible in legend)
        fig.add_trace(
            go.Scatter(
                x=merged_ed["date"],
                y=merged_ed["price"],
                mode="markers",
                name="Earnings",
                marker=dict(symbol="diamond", size=12, color=earnings_marker_color),
                hovertemplate=(
                    "Earnings<br>"
                    "Date: %{x|%Y-%m-%d}<br>"
                    "Indexed level: %{y:.2f}<extra></extra>"
                ),
                yaxis="y",
            )
        )

        # Annotations for each earnings event: Est / Act / Δ
        for _, row in merged_ed.iterrows():
            event_date = row["date"]
            price_at_event = row["price"]

            est = row.get("epsEstimated")
            act = row.get("epsActual")

            lines = []
            if pd.notna(est):
                lines.append(f"Est {est:.2f}")
            if pd.notna(act):
                lines.append(f"Act {act:.2f}")

            # Difference if we have both
            color = neu_color
            if pd.notna(est) and pd.notna(act):
                diff = act - est
                if diff > 0:
                    color = pos_color
                elif diff < 0:
                    color = neg_color
                else:
                    color = neu_color
                lines.append(f"Δ {diff:+.2f}")
            elif not lines:
                # No usable data at all; fall back on generic label
                lines.append("Earnings")

            text = "<br>".join(lines)

            fig.add_annotation(
                x=event_date,
                y=price_at_event,
                xanchor="center",
                yanchor="bottom",
                yshift=10,  # pixels above the price point
                showarrow=False,
                text=text,
                font=dict(size=12, color=color),
                bgcolor="rgba(255,255,255,0.85)",
                bordercolor=color,
                borderwidth=0.5,
                borderpad=2,
            )

    # --- Dividend markers ---
    if dividend_events is not None and not dividend_events.empty:
        dd = dividend_events.copy()
        dd["date"] = pd.to_datetime(dd["date"])
        dd = dd.sort_values("date")

        merged_dd = pd.merge_asof(
            dd,
            stock_for_merge,
            on="date",
            direction="backward",
        )

        labels = []
        for _, row in merged_dd.iterrows():
            # Prefer adjDividend, fallback to dividend
            val = row.get("adjDividend")
            if pd.isna(val):
                val = row.get("dividend")
            if pd.notna(val):
                label = f"${val:.2f}"
            else:
                label = "$?"
            labels.append(label)

        fig.add_trace(
            go.Scatter(
                x=merged_dd["date"],
                y=merged_dd["price"],
                mode="markers+text",
                name="Dividends",
                marker=dict(symbol="triangle-down", size=12, color=dividends_marker_color),
                text=labels,
                textposition="top center",
                textfont=dict(size=12, color="#111111"),
                hovertemplate=(
                    "Dividend<br>"
                    "Date: %{x|%Y-%m-%d}<br>"
                    "Indexed level: %{y:.2f}<extra></extra>"
                ),
                yaxis="y",
            )
        )

    # ------------------------------------------------------------------
    # Layout: dual y-axes, shared indexed range, but price-based tick labels
    # ------------------------------------------------------------------
    # Compute a shared range in *indexed* space so both series line up
    all_indexed_vals = pd.concat(
        [stock["indexed"], spx["indexed"]], axis=0, ignore_index=True
    )
    ymin = float(all_indexed_vals.min())
    ymax = float(all_indexed_vals.max())
    pad = (ymax - ymin) * 0.05 if ymax > ymin else 5.0
    yrange = [ymin - pad, ymax + pad]

    # Choose some common tick positions in indexed space
    tickvals = [float(v) for v in np.linspace(yrange[0], yrange[1], 6)]

    # Map those ticks back to actual prices for each axis
    stock_ticktext = [f"{stock_base * tv / 100.0:.2f}" for tv in tickvals]
    spx_ticktext = [f"{spx_base * tv / 100.0:.2f}" for tv in tickvals]

    fig.update_layout(
        xaxis=dict(title="Date"),
        yaxis=dict(
            title=symbol,  # left axis = stock, no "indexed" suffix
            range=yrange,
            tickmode="array",
            tickvals=tickvals,
            ticktext=stock_ticktext,
        ),
        yaxis2=dict(
            title="S&P 500",
            overlaying="y",
            side="right",
            showgrid=False,
            range=yrange,
            tickmode="array",
            tickvals=tickvals,
            ticktext=spx_ticktext,
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.06,
            xanchor="center",
            x=0.5,
            font=dict(size=16),  # slightly bigger legend font
        ),
        margin=dict(l=40, r=40, t=40, b=40),
        hovermode="x unified",
    )

    # Intentionally no chart title; Streamlit heading provides the context.
    return fig

# -------------------------------
# Streamlit app layout & logic
# -------------------------------
def main() -> None:
    st.markdown(
        "<h2 style='margin-bottom:0.5rem;'>Stock vs S&amp;P 500</h2>",
        unsafe_allow_html=True,
    )
    st.caption(
        "Daily prices with earnings and dividend annotations &mdash; data via Financial Modeling Prep."
    )

    api_key = load_api_key()

    # --- Sidebar controls ---
    with st.sidebar:
        st.header("Controls")

        popular_tickers = [
            "AAPL",
            "MSFT",
            "AMZN",
            "GOOGL",
            "META",
            "NVDA",
            "TSLA",
            "HD",
            "JPM",
            "XOM",
        ]

        ticker_mode = st.radio(
            "Symbol input",
            options=["Popular tickers", "Custom symbol"],
            horizontal=False,
        )

        if ticker_mode == "Popular tickers":
            symbol = st.selectbox("Select symbol", options=popular_tickers, index=0)
        else:
            symbol = st.text_input("Enter symbol", value="AAPL").upper().strip()

        st.markdown("---")

        _ = st.selectbox(
            "Date range",
            options=["Last 1 year (rolling)"],
            index=0,
            help="For now, the app shows the last 1 year up to the latest available trading day.",
        )

        if st.button("Refresh cached data", help="Clear all cached API responses."):
            st.cache_data.clear()
            st.success("Cache cleared. Reloading with fresh data...")
            try:
                st.rerun()
            except Exception:  # older Streamlit versions
                st.experimental_rerun()  # type: ignore[attr-defined]

    if not symbol:
        st.info("Select or enter a stock symbol in the sidebar to begin.")
        return

    # --- Data fetch & preparation ---
    with st.spinner(f"Fetching data for {symbol} and S&P 500..."):
        stock_prices = fetch_price_history(symbol, api_key)
        sp500_prices = fetch_sp500_history(api_key)
        earnings = fetch_earnings(symbol, api_key)
        dividends = fetch_dividends(symbol, api_key)

    if stock_prices.empty:
        st.error(f"Could not fetch price history for {symbol}.")
        return

    if sp500_prices.empty:
        st.error("Could not fetch price history for S&P 500 (^GSPC).")
        return

    # Determine the reference end date as the latest available trading day for the stock
    end_date = stock_prices["date"].max()
    if pd.isna(end_date):
        st.error(f"No valid dates in price history for {symbol}.")
        return

    # Filter everything to the last 1 year up to end_date
    stock_last_year = filter_to_last_year(stock_prices, end_date)
    sp500_last_year = filter_to_last_year(sp500_prices, end_date)
    earnings_last_year = filter_to_last_year(earnings, end_date) if not earnings.empty else earnings
    dividends_last_year = (
        filter_to_last_year(dividends, end_date) if not dividends.empty else dividends
    )

    if stock_last_year.empty:
        st.error(f"No price data for {symbol} in the last 1 year window.")
        return

    # Warn if limited history
    first_available = stock_last_year["date"].min()
    if (end_date - first_available).days < 90:
        st.warning(
            f"{symbol.upper()} has less than 90 days of history in the selected window. "
            "The chart may be noisy."
        )

    # Build and render the figure using your chart builder
    fig = build_price_and_events_chart(
        symbol=symbol.upper(),
        as_of_date=end_date.date(),
        stock_prices=stock_last_year,
        sp500_prices=sp500_last_year,
        earnings_events=earnings_last_year if not earnings_last_year.empty else None,
        dividend_events=dividends_last_year if not dividends_last_year.empty else None,
    )

    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    # Optional small details below the chart
    col1, col2 = st.columns(2)
    with col1:
        st.caption(
            f"Showing data from **{stock_last_year['date'].min().date()}** "
            f"to **{stock_last_year['date'].max().date()}**."
        )
    with col2:
        st.caption("Markers: diamonds = earnings, triangles = dividends.")


if __name__ == "__main__":
    main()
