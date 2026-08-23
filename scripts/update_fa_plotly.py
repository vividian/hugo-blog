from __future__ import annotations

import argparse
import html
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple
from urllib.parse import urlparse

import math
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots


ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from scripts import update_fa

DEFAULT_FRAGMENT_PATH = ROOT_DIR / "generated" / "fa" / "latest_fa_fragment.html"
LEGACY_FRAGMENT_PATH = ROOT_DIR / "data" / "fa" / "latest_fa_fragment.html"

APP_VERSION = "v2.6.4"

FONT_FAMILY = "-apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Noto Sans KR', sans-serif"
CHART_COLORWAY = [
    "#3182CE",  # Blue
    "#38A169",  # Green
    "#DD6B20",  # Orange
    "#805AD5",  # Purple
    "#E53E3E",  # Red
    "#319795",  # Teal
    "#D69E2E",  # Yellow
    "#718096",  # Gray
    "#ED64A6",  # Pink
]
THEME_BG = "rgba(0,0,0,0)"
THEME_TEXT = "#64748b"
THEME_GRID = "rgba(148, 163, 184, 0.15)"

MARKET_KPI_CONFIG = [
    {"label": "S&P 500", "ticker": "^GSPC", "decimals": 2},
    {"label": "나스닥 100", "ticker": "^NDX", "decimals": 2},
    {"label": "SCHD", "ticker": "SCHD", "decimals": 2},
    {"label": "미국 7-10년 국채(IEF)", "ticker": "IEF", "decimals": 2},
    {"label": "코스피", "ticker": "^KS11", "decimals": 2},
    {"label": "코스닥", "ticker": "^KQ11", "decimals": 2},
]


def _format_korean_amount(val: float) -> str:
    """원화 금액 숫자를 한글 단위(억, 만) 표현으로 변환한다."""
    if pd.isna(val) or val == 0:
        return "0"
    sign = "-" if val < 0 else ""
    val = abs(val)

    eok = val / 100_000_000
    if eok >= 0.1:
        if eok == int(eok):
            return f"{sign}{int(eok)}억"
        else:
            formatted_eok = f"{eok:.2f}".rstrip("0").rstrip(".")
            return f"{sign}{formatted_eok}억"
    elif val >= 10_000:
        man = val / 10_000
        if man == int(man):
            return f"{sign}{int(man):,}만"
        else:
            formatted_man = f"{man:.2f}".rstrip("0").rstrip(".")
            return f"{sign}{formatted_man}만"
    else:
        return f"{sign}{val:,.0f}"


def _get_korean_y_ticks(y_max: float, y_min: float = 0, target_ticks: int = 5):
    if pd.isna(y_max) or y_max <= 0:
        return [0], ["0"]

    start = 0 if y_min >= 0 else y_min
    span = y_max - start

    raw_step = span / target_ticks
    step_candidates = [
        10_000_000,
        20_000_000,
        50_000_000,
        100_000_000,
        200_000_000,
        500_000_000,
        1_000_000_000,
        2_000_000_000,
        5_000_000_000,
    ]

    step = step_candidates[-1]
    for candidate in step_candidates:
        if candidate >= raw_step * 0.75:
            step = candidate
            break

    if raw_step * 0.75 > step_candidates[-1]:
        step = int(np.ceil(raw_step / 100_000_000)) * 100_000_000

    start_tick = int(np.floor(start / step)) * step
    end_tick = int(np.ceil(y_max / step)) * step

    tickvals = []
    curr = start_tick
    while curr <= end_tick + step * 0.01:
        tickvals.append(curr)
        curr += step

    ticktext = [_format_korean_amount(v) for v in tickvals]
    return tickvals, ticktext


def _palette_color(idx: int) -> str:
    return CHART_COLORWAY[idx % len(CHART_COLORWAY)]


@dataclass
class ReportData:
    month_end: pd.Timestamp
    records: pd.DataFrame
    fx_series_full: pd.Series
    fx_series_month: pd.Series
    account_df: pd.DataFrame
    summary_df: pd.DataFrame
    holdings_df: pd.DataFrame
    dividends_pivot: Optional[pd.DataFrame]
    yearly_dividends_pivot: Optional[pd.DataFrame]
    valid_detail_accounts: Sequence[str]
    invest_series: pd.Series
    symbol_map: Dict[str, update_fa.AssetConfig]


def _as_float(value: object) -> Optional[float]:
    try:
        if value is None or pd.isna(value):
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _fmt_krw(value: Optional[float]) -> str:
    if value is None or pd.isna(value):
        return "-"
    return f"{value:,.0f}원"


def _fmt_pct(value: Optional[float]) -> str:
    if value is None or pd.isna(value):
        return "-"
    return f"{value:+.2f}%"


def _fmt_number(value: Optional[float], decimals: int = 2, suffix: str = "") -> str:
    if value is None or pd.isna(value):
        return "-"
    if decimals <= 0:
        return f"{value:,.0f}{suffix}"
    return f"{value:,.{decimals}f}{suffix}"


def _build_change_text(
    current: Optional[float],
    previous: Optional[float],
    *,
    decimals: int = 2,
    delta_suffix: str = "",
) -> Tuple[str, str]:
    if current is None or previous is None:
        return "-", ""
    delta = current - previous
    state = "positive" if delta > 0 else "negative" if delta < 0 else ""
    pct_text = ""
    if previous != 0:
        pct = (delta / previous) * 100.0
        pct_text = f" ({pct:+.2f}%)"
    delta_text = _fmt_number(delta, decimals, delta_suffix)
    if delta > 0 and not delta_text.startswith("+"):
        delta_text = f"+{delta_text}"
    return f"증감 {delta_text}{pct_text}", state


def _fetch_market_snapshots() -> Dict[str, Tuple[Optional[float], Optional[float]]]:
    tickers = [cfg["ticker"] for cfg in MARKET_KPI_CONFIG]
    try:
        raw = update_fa.fetch_latest_prices(tickers)
    except Exception:
        raw = {}

    snapshots: Dict[str, Tuple[Optional[float], Optional[float]]] = {}
    for cfg in MARKET_KPI_CONFIG:
        ticker = cfg["ticker"]
        scale = float(cfg.get("scale", 1.0))
        price_pair = raw.get(ticker)
        if price_pair is None:
            snapshots[ticker] = (None, None)
            continue
        last_price, prev_price = price_pair
        snapshots[ticker] = (last_price * scale, prev_price * scale)
    return snapshots


def _kpi_card(label: str, value: str, sub: str = "", state: str = "") -> str:
    state_class = f" {state}" if state else ""
    return (
        f"<div class=\"fa-kpi-card{state_class}\">"
        f"<div class=\"fa-kpi-label\">{html.escape(label)}</div>"
        f"<div class=\"fa-kpi-value\">{value}</div>"
        f"<div class=\"fa-kpi-sub\">{html.escape(sub)}</div>"
        "</div>"
    )


def _build_kpi_row(data: ReportData) -> str:
    total_row = None
    if not data.summary_df.empty and "계좌" in data.summary_df.columns:
        total_match = data.summary_df[data.summary_df["계좌"] == "합계"]
        if not total_match.empty:
            total_row = total_match.iloc[-1]

    invest = _as_float(total_row["투자금"]) if total_row is not None and "투자금" in total_row else None
    valuation = _as_float(total_row["평가금"]) if total_row is not None and "평가금" in total_row else None
    profit = _as_float(total_row["수익금"]) if total_row is not None and "수익금" in total_row else None
    return_rate = _as_float(total_row["수익률"]) if total_row is not None and "수익률" in total_row else None

    if return_rate is not None and abs(return_rate) <= 1.0:
        return_rate *= 100.0

    monthly_div = None
    if data.dividends_pivot is not None and not data.dividends_pivot.empty:
        monthly_div = _as_float(data.dividends_pivot.sort_index().iloc[-1].sum())

    fx = _as_float(data.fx_series_full.iloc[-1]) if not data.fx_series_full.empty else None
    fx_prev = _as_float(data.fx_series_full.iloc[-2]) if len(data.fx_series_full) > 1 else fx
    fx_change_text, fx_state = _build_change_text(fx, fx_prev, decimals=2)
    month_label = data.month_end.strftime("%Y.%m")

    profit_str = _fmt_krw(profit)
    profit_state = "positive" if (profit or 0) > 0 else "negative" if (profit or 0) < 0 else ""
    return_str = _fmt_pct(return_rate)
    return_state = "positive" if (return_rate or 0) > 0 else "negative" if (return_rate or 0) < 0 else ""

    cards = [
        _kpi_card("총 평가금", _fmt_krw(valuation), f"{month_label} 기준"),
        _kpi_card("총 투자금", _fmt_krw(invest), f"{month_label} 누적"),
        _kpi_card("총 수익금", f"<span class='fa-num-{profit_state}'>{profit_str}</span>", "실현+평가", profit_state),
        _kpi_card("총 수익률", f"<span class='fa-num-{return_state}'>{return_str}</span>", "투자금 대비", return_state),
        _kpi_card("월 배당금", _fmt_krw(monthly_div), f"{month_label} 합계"),
        _kpi_card("USD/KRW", _fmt_number(fx, 2, "원"), fx_change_text, fx_state),
    ]
    return "<div class=\"fa-kpi-grid\">" + "".join(cards) + "</div>"


def _build_market_kpi_row() -> str:
    snapshots = _fetch_market_snapshots()
    cards: List[str] = []
    for cfg in MARKET_KPI_CONFIG:
        ticker = cfg["ticker"]
        decimals = int(cfg.get("decimals", 2))
        value_suffix = str(cfg.get("value_suffix", ""))
        delta_suffix = str(cfg.get("delta_suffix", value_suffix))
        current, previous = snapshots.get(ticker, (None, None))
        change_text, state = _build_change_text(
            current,
            previous,
            decimals=decimals,
            delta_suffix=delta_suffix,
        )
        cards.append(
            _kpi_card(
                str(cfg["label"]),
                _fmt_number(current, decimals, value_suffix),
                change_text,
                state,
            )
        )
    return "<div class=\"fa-kpi-grid fa-kpi-grid-market\">" + "".join(cards) + "</div>"


def _build_assets_trend(account_df: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    for idx, column in enumerate(account_df.columns):
        label = update_fa.account_label(column)
        fig.add_trace(
            go.Scatter(
                x=account_df.index,
                y=account_df[column],
                mode="lines",
                name=label,
                line=dict(color=_palette_color(idx), width=2.5),
                hovertemplate="%{x|%Y-%m}: %{y:,.0f}원<extra>%{fullData.name}</extra>",
            )
        )
    fig.update_layout(
        height=370,
        margin=dict(l=15, r=15, t=65, b=25),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0, font=dict(size=13.5, family=FONT_FAMILY)),
        showlegend=True,
        font=dict(family=FONT_FAMILY, size=14),
        paper_bgcolor=THEME_BG,
        plot_bgcolor=THEME_BG,
    )
    fig.update_xaxes(tickfont=dict(size=13, family=FONT_FAMILY), showgrid=False)
    y_max = account_df.max().max() if not account_df.empty else 0
    tickvals, ticktext = _get_korean_y_ticks(y_max, y_min=0)
    fig.update_yaxes(
        tickmode="array",
        tickvals=tickvals,
        ticktext=ticktext,
        rangemode="tozero",
        tickfont=dict(size=13, family=FONT_FAMILY),
        showgrid=True,
        gridcolor=THEME_GRID,
    )
    return fig


def _build_assets_investment_trend(account_df: pd.DataFrame, invest_series: pd.Series) -> go.Figure:
    fig = go.Figure()
    total_valuation = account_df.sum(axis=1)
    invest_aligned = update_fa.align_series(invest_series, account_df.index)

    fig.add_trace(
        go.Scatter(
            x=account_df.index,
            y=invest_aligned,
            mode="lines",
            name="누적 투자금",
            line=dict(color="#4A5568", width=2.5),
            hovertemplate="%{x|%Y-%m}: %{y:,.0f}원<extra>누적 투자금</extra>",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=account_df.index,
            y=total_valuation,
            mode="lines",
            name="누적 평가금",
            line=dict(color="#E53E3E", width=2.5),
            hovertemplate="%{x|%Y-%m}: %{y:,.0f}원<extra>누적 평가금</extra>",
        )
    )
    fig.update_layout(
        height=370,
        margin=dict(l=15, r=15, t=65, b=25),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0, font=dict(size=13.5, family=FONT_FAMILY)),
        showlegend=True,
        font=dict(family=FONT_FAMILY, size=14),
        paper_bgcolor=THEME_BG,
        plot_bgcolor=THEME_BG,
    )
    fig.update_xaxes(tickfont=dict(size=13, family=FONT_FAMILY), showgrid=False)
    y_max = max(np.nanmax(invest_aligned), np.nanmax(total_valuation)) if len(invest_aligned) > 0 else 0
    tickvals, ticktext = _get_korean_y_ticks(y_max, y_min=0)
    fig.update_yaxes(
        tickmode="array",
        tickvals=tickvals,
        ticktext=ticktext,
        rangemode="tozero",
        tickfont=dict(size=13, family=FONT_FAMILY),
        showgrid=True,
        gridcolor=THEME_GRID,
    )
    return fig


def _build_portfolio_allocation_charts(holdings_df: pd.DataFrame, symbol_map: Dict[str, update_fa.AssetConfig]) -> go.Figure:
    df = holdings_df.copy()
    regions = []
    asset_classes = []
    asset_groups = []

    for _, row in df.iterrows():
        symbol = row["종목"]
        config = symbol_map.get(symbol)
        region = "기타"
        asset_class = "기타"
        if config:
            region = config.region or "기타"
            asset_class = config.asset_class or "기타"

        regions.append(region)
        asset_classes.append(asset_class)

        ac_lower = asset_class.lower()
        if any(keyword in ac_lower for keyword in ["현금", "mma", "kofr", "saving", "저축"]):
            group = "현금성 자산"
        elif any(keyword in ac_lower for keyword in ["tlt", "ief", "국채", "채권", "tltw"]):
            group = "채권"
        elif any(keyword in ac_lower for keyword in ["골드", "금", "gold"]):
            group = "대안자산(금)"
        else:
            group = "주식"

        asset_groups.append(group)

    df["region"] = regions
    df["asset_class"] = asset_classes
    df["asset_group"] = asset_groups

    group_df = df.groupby("asset_group")["평가금"].sum().reset_index()
    region_df = df.groupby("region")["평가금"].sum().reset_index()
    class_df = df.groupby("asset_class")["평가금"].sum().reset_index()

    def group_minor_df(df_in: pd.DataFrame, col_name: str, threshold_pct: float = 2.5) -> pd.DataFrame:
        if len(df_in) <= 5:
            return df_in
        total = df_in["평가금"].sum()
        if total == 0:
            return df_in
        df_sorted = df_in.sort_values(by="평가금", ascending=False).copy()
        df_sorted["pct"] = df_sorted["평가금"] / total * 100

        major_df = df_sorted[df_sorted["pct"] >= threshold_pct].copy()
        minor_df = df_sorted[df_sorted["pct"] < threshold_pct].copy()

        if not minor_df.empty:
            minor_sum = minor_df["평가금"].sum()
            etc_mask = major_df[col_name] == "기타"
            if etc_mask.any():
                major_df.loc[etc_mask, "평가금"] += minor_sum
            else:
                new_row = pd.DataFrame([{col_name: "기타", "평가금": minor_sum}])
                major_df = pd.concat([major_df, new_row], ignore_index=True)
        return major_df.drop(columns=["pct"], errors="ignore").sort_values(by="평가금", ascending=False)

    group_df = group_minor_df(group_df, "asset_group")
    region_df = group_minor_df(region_df, "region")
    class_df = group_minor_df(class_df, "asset_class")

    fig = make_subplots(
        rows=1,
        cols=3,
        specs=[[{"type": "domain"}, {"type": "domain"}, {"type": "domain"}]],
        subplot_titles=["<b>자산군 비중</b>", "<b>지역 비중</b>", "<b>대표 자산 비중</b>"],
    )

    fig.add_trace(
        go.Pie(
            labels=group_df["asset_group"],
            values=group_df["평가금"],
            textinfo="percent+label",
            textposition="inside",
            insidetextorientation="horizontal",
            hole=0.5,
            name="자산군",
            showlegend=False,
            marker=dict(colors=CHART_COLORWAY),
            hovertemplate="<b>%{label}</b><br>평가금: %{value:,.0f}원<br>비중: %{percent}<extra></extra>",
        ),
        row=1,
        col=1,
    )

    fig.add_trace(
        go.Pie(
            labels=region_df["region"],
            values=region_df["평가금"],
            textinfo="percent+label",
            textposition="inside",
            insidetextorientation="horizontal",
            hole=0.5,
            name="지역",
            showlegend=False,
            marker=dict(colors=CHART_COLORWAY),
            hovertemplate="<b>%{label}</b><br>평가금: %{value:,.0f}원<br>비중: %{percent}<extra></extra>",
        ),
        row=1,
        col=2,
    )

    fig.add_trace(
        go.Pie(
            labels=class_df["asset_class"],
            values=class_df["평가금"],
            textinfo="percent+label",
            textposition="inside",
            insidetextorientation="horizontal",
            hole=0.5,
            name="대표 자산",
            showlegend=False,
            marker=dict(colors=CHART_COLORWAY),
            hovertemplate="<b>%{label}</b><br>평가금: %{value:,.0f}원<br>비중: %{percent}<extra></extra>",
        ),
        row=1,
        col=3,
    )

    fig.update_layout(
        height=380,
        margin=dict(l=10, r=10, t=45, b=10),
        font=dict(family=FONT_FAMILY, size=14),
        paper_bgcolor=THEME_BG,
        plot_bgcolor=THEME_BG,
        uniformtext_mode="hide",
        uniformtext_minsize=12,
    )

    for annotation in fig["layout"]["annotations"]:
        annotation["font"] = dict(size=14.5, color=THEME_TEXT, family=FONT_FAMILY)

    return fig


def _build_dividends_chart(pivot: pd.DataFrame) -> go.Figure:
    pivot_sorted = pivot.sort_index()
    if len(pivot_sorted) > 12:
        pivot_sorted = pivot_sorted.tail(12)
    fig = go.Figure()

    totals = pivot_sorted.sum(axis=1)

    for idx, col in enumerate(pivot.columns):
        is_last = idx == len(pivot.columns) - 1
        texts = [f"{totals.iloc[i]:,.0f}" if is_last else "" for i in range(len(pivot_sorted))]

        fig.add_trace(
            go.Bar(
                x=pivot_sorted.index,
                y=pivot_sorted[col],
                name=col,
                marker=dict(color=_palette_color(idx)),
                text=texts,
                textposition="outside" if is_last else "none",
                textfont=dict(size=13, color=THEME_TEXT, family=FONT_FAMILY),
                cliponaxis=False,
                hovertemplate=f"<b>{col}</b><br>%{{x|%Y-%m}}: %{{y:,.0f}}원<extra></extra>",
            )
        )
    fig.update_layout(
        barmode="stack",
        height=320,
        margin=dict(l=15, r=15, t=30, b=25),
        showlegend=False,
        font=dict(family=FONT_FAMILY, size=14),
        paper_bgcolor=THEME_BG,
        plot_bgcolor=THEME_BG,
    )
    fig.update_yaxes(
        tickformat=",.0f",
        dtick=500_000,
        showgrid=True,
        gridcolor=THEME_GRID,
        zeroline=False,
        tickfont=dict(size=13, family=FONT_FAMILY),
    )
    if not pivot_sorted.empty:
        min_x = pd.to_datetime(pivot_sorted.index.min()) - pd.Timedelta(days=15)
        max_x = pd.to_datetime(pivot_sorted.index.max()) + pd.Timedelta(days=15)
        fig.update_xaxes(tickformat="%y%m", range=[min_x, max_x], tickfont=dict(size=13, family=FONT_FAMILY))
    else:
        fig.update_xaxes(tickformat="%y%m", tickfont=dict(size=13, family=FONT_FAMILY))
    return fig


def _build_yearly_dividends_chart(pivot: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    pivot_sorted = pivot.sort_index()
    years = pd.to_datetime(pivot_sorted.index).strftime("%Y").tolist()

    totals = pivot_sorted.sum(axis=1)

    for idx, col in enumerate(pivot_sorted.columns):
        is_last = idx == len(pivot_sorted.columns) - 1
        texts = [f"{totals.iloc[i]:,.0f}" if is_last else "" for i in range(len(pivot_sorted))]

        fig.add_trace(
            go.Bar(
                x=years,
                y=pivot_sorted[col],
                name=col,
                marker=dict(color=_palette_color(idx)),
                text=texts,
                textposition="outside" if is_last else "none",
                textfont=dict(size=13, color=THEME_TEXT, family=FONT_FAMILY),
                cliponaxis=False,
                hovertemplate=f"<b>{col}</b><br>%{{x}}년: %{{y:,.0f}}원<extra></extra>",
            )
        )
    fig.update_layout(
        barmode="stack",
        height=320,
        margin=dict(l=15, r=15, t=30, b=25),
        showlegend=False,
        font=dict(family=FONT_FAMILY, size=14),
        paper_bgcolor=THEME_BG,
        plot_bgcolor=THEME_BG,
    )
    fig.update_yaxes(tickformat=",.0f", showgrid=True, gridcolor=THEME_GRID, zeroline=False, tickfont=dict(size=13, family=FONT_FAMILY))
    fig.update_xaxes(tickfont=dict(size=13, family=FONT_FAMILY))
    return fig


# =========================================================================
# 순수 HTML + 모바일 반응형 카드 뷰 테이블 렌더러
# =========================================================================

def _build_account_assets_html_table(summary_df: pd.DataFrame) -> str:
    """계좌 요약 데이터프레임을 PC 테이블 & 모바일 카드로 변환되는 반응형 HTML로 렌더링"""
    if summary_df.empty:
        return "<p class='fa-empty-text'>계좌 데이터가 없습니다.</p>"

    lines = [
        "<div class='fa-table-wrapper'>",
        "<table class='fa-table fa-table-responsive fa-table-account-summary'>",
        "<thead>",
        "  <tr>",
        "    <th>계좌</th>",
        "    <th class='text-right'>투자금</th>",
        "    <th class='text-right'>평가금</th>",
        "    <th class='text-right'>수익금</th>",
        "    <th class='text-right'>수익률</th>",
        "    <th class='text-right'>비중</th>",
        "    <th class='text-right'>배당금</th>",
        "  </tr>",
        "</thead>",
        "<tbody>",
    ]

    total_row = None
    for _, row in summary_df.iterrows():
        acct_name = str(row["계좌"])
        if acct_name == "합계":
            total_row = row
            continue

        label = update_fa.account_label(acct_name)
        invest = _as_float(row.get("투자금"))
        valuation = _as_float(row.get("평가금"))
        profit = _as_float(row.get("수익금"))
        return_rate = _as_float(row.get("수익률"))
        weight = _as_float(row.get("비중"))
        dividend = _as_float(row.get("배당금"))

        profit_cls = "fa-num-positive" if (profit or 0) > 0 else "fa-num-negative" if (profit or 0) < 0 else ""
        profit_badge = "fa-badge-positive" if (profit or 0) > 0 else "fa-badge-negative" if (profit or 0) < 0 else "fa-badge-neutral"
        rate_cls = "fa-num-positive" if (return_rate or 0) > 0 else "fa-num-negative" if (return_rate or 0) < 0 else ""

        invest_str = f"{invest:,.0f}원" if invest is not None else "-"
        eval_str = f"{valuation:,.0f}원" if valuation is not None else "-"
        profit_str = f"{profit:+,.0f}원" if profit is not None and profit != 0 else (f"{profit:,.0f}원" if profit is not None else "-")
        rate_str = f"{return_rate * 100:+.2f}%" if return_rate is not None else "-"
        weight_str = f"{weight * 100:.1f}%" if weight is not None else "-"
        div_str = f"{dividend:,.0f}원" if dividend is not None and dividend > 0 else "-"

        lines.append("  <tr>")
        lines.append(f"    <td data-label='계좌' class='fa-col-account'><strong>{html.escape(label)}</strong></td>")
        lines.append(f"    <td data-label='투자금' class='text-right fa-num'>{invest_str}</td>")
        lines.append(f"    <td data-label='평가금' class='text-right fa-num fa-font-bold'>{eval_str}</td>")
        lines.append(f"    <td data-label='수익금' class='text-right fa-num {profit_cls}'>{profit_str}</td>")
        lines.append(f"    <td data-label='수익률' class='text-right fa-num'><span class='fa-badge {profit_badge}'>{rate_str}</span></td>")
        lines.append(f"    <td data-label='비중' class='text-right fa-num'>{weight_str}</td>")
        lines.append(f"    <td data-label='배당금' class='text-right fa-num'>{div_str}</td>")
        lines.append("  </tr>")

    lines.append("</tbody>")

    if total_row is not None:
        invest = _as_float(total_row.get("투자금"))
        valuation = _as_float(total_row.get("평가금"))
        profit = _as_float(total_row.get("수익금"))
        return_rate = _as_float(total_row.get("수익률"))
        dividend = _as_float(total_row.get("배당금"))

        profit_cls = "fa-num-positive" if (profit or 0) > 0 else "fa-num-negative" if (profit or 0) < 0 else ""
        profit_badge = "fa-badge-positive" if (profit or 0) > 0 else "fa-badge-negative" if (profit or 0) < 0 else "fa-badge-neutral"

        invest_str = f"{invest:,.0f}원" if invest is not None else "-"
        eval_str = f"{valuation:,.0f}원" if valuation is not None else "-"
        profit_str = f"{profit:+,.0f}원" if profit is not None and profit != 0 else (f"{profit:,.0f}원" if profit is not None else "-")
        rate_str = f"{return_rate * 100:+.2f}%" if return_rate is not None else "-"
        div_str = f"{dividend:,.0f}원" if dividend is not None and dividend > 0 else "-"

        lines.append("<tfoot>")
        lines.append("  <tr class='fa-tr-total'>")
        lines.append("    <td data-label='계좌'><strong>합계</strong></td>")
        lines.append(f"    <td data-label='투자금' class='text-right fa-num'>{invest_str}</td>")
        lines.append(f"    <td data-label='평가금' class='text-right fa-num fa-font-bold'>{eval_str}</td>")
        lines.append(f"    <td data-label='수익금' class='text-right fa-num {profit_cls}'>{profit_str}</td>")
        lines.append(f"    <td data-label='수익률' class='text-right fa-num'><span class='fa-badge {profit_badge}'>{rate_str}</span></td>")
        lines.append("    <td data-label='비중' class='text-right fa-num'>100.0%</td>")
        lines.append(f"    <td data-label='배당금' class='text-right fa-num'>{div_str}</td>")
        lines.append("  </tr>")
        lines.append("</tfoot>")

    lines.append("</table>")
    lines.append("</div>")
    return "\n".join(lines)


def _build_total_holdings_html_table(holdings_df: pd.DataFrame) -> str:
    """전체 보유 종목 데이터프레임을 PC 테이블 & 모바일 카드로 변환되는 반응형 HTML로 렌더링"""
    filtered = holdings_df[holdings_df["계좌"] != "sema"].copy()
    if filtered.empty:
        return "<p class='fa-empty-text'>보유 종목 데이터가 없습니다.</p>"

    filtered["계좌라벨"] = filtered["계좌"].apply(update_fa.account_label)
    filtered = filtered.sort_values(["계좌", "평가금"], ascending=[True, False])

    lines = [
        "<div class='fa-table-wrapper'>",
        "<table class='fa-table fa-table-responsive fa-table-holdings'>",
        "<thead>",
        "  <tr>",
        "    <th>계좌</th>",
        "    <th>종목</th>",
        "    <th class='text-right'>수량</th>",
        "    <th class='text-right'>평단가</th>",
        "    <th class='text-right'>매수금액</th>",
        "    <th class='text-right'>현재가</th>",
        "    <th class='text-right'>평가금액</th>",
        "    <th class='text-right'>수익금</th>",
        "    <th class='text-right'>수익률</th>",
        "    <th class='text-right'>등락률</th>",
        "  </tr>",
        "</thead>",
        "<tbody>",
    ]

    for _, row in filtered.iterrows():
        acct_label = str(row["계좌라벨"])
        symbol = str(row["종목"])
        qty = _as_float(row.get("수량"))
        avg_price = _as_float(row.get("평단가"))
        buy_amt = _as_float(row.get("금액")) or _as_float(row.get("매수금"))
        cur_price = _as_float(row.get("현재가"))
        eval_amt = _as_float(row.get("평가금"))
        profit = _as_float(row.get("수익금"))
        return_rate = _as_float(row.get("수익률"))
        fluct_rate = _as_float(row.get("등락률"))

        profit_cls = "fa-num-positive" if (profit or 0) > 0 else "fa-num-negative" if (profit or 0) < 0 else ""
        profit_badge = "fa-badge-positive" if (profit or 0) > 0 else "fa-badge-negative" if (profit or 0) < 0 else "fa-badge-neutral"
        fluct_cls = "fa-num-positive" if (fluct_rate or 0) > 0 else "fa-num-negative" if (fluct_rate or 0) < 0 else ""

        qty_str = f"{qty:,.2f}".rstrip("0").rstrip(".") if qty is not None else "-"
        avg_str = f"{avg_price:,.0f}원" if avg_price is not None else "-"
        buy_str = f"{buy_amt:,.0f}원" if buy_amt is not None else "-"
        cur_str = f"{cur_price:,.0f}원" if cur_price is not None else "-"
        eval_str = f"{eval_amt:,.0f}원" if eval_amt is not None else "-"
        profit_str = f"{profit:+,.0f}원" if profit is not None and profit != 0 else (f"{profit:,.0f}원" if profit is not None else "-")
        rate_str = f"{return_rate * 100:+.2f}%" if return_rate is not None else "-"
        fluct_str = f"{fluct_rate * 100:+.2f}%" if fluct_rate is not None else "-"

        lines.append("  <tr>")
        lines.append(f"    <td data-label='계좌'><span class='fa-chip-account'>{html.escape(acct_label)}</span></td>")
        lines.append(f"    <td data-label='종목' class='fa-col-symbol'><strong>{html.escape(symbol)}</strong></td>")
        lines.append(f"    <td data-label='수량' class='text-right fa-num'>{qty_str}</td>")
        lines.append(f"    <td data-label='평단가' class='text-right fa-num'>{avg_str}</td>")
        lines.append(f"    <td data-label='매수금액' class='text-right fa-num'>{buy_str}</td>")
        lines.append(f"    <td data-label='현재가' class='text-right fa-num'>{cur_str}</td>")
        lines.append(f"    <td data-label='평가금액' class='text-right fa-num fa-font-bold'>{eval_str}</td>")
        lines.append(f"    <td data-label='수익금' class='text-right fa-num {profit_cls}'>{profit_str}</td>")
        lines.append(f"    <td data-label='수익률' class='text-right fa-num'><span class='fa-badge {profit_badge}'>{rate_str}</span></td>")
        lines.append(f"    <td data-label='등락률' class='text-right fa-num {fluct_cls}'>{fluct_str}</td>")
        lines.append("  </tr>")

    lines.append("</tbody>")
    lines.append("</table>")
    lines.append("</div>")
    return "\n".join(lines)


def _build_account_detail_section(
    data: update_fa.MonthlyData,
    fig_renderer: Callable[[go.Figure], str],
) -> str:
    """계좌별 상세 현황(비중 차트 + 종목별 카드 + 리밸런싱 가이드)을 탭 UI로 통합 렌더링"""
    tab_btns = []
    tab_panes = []

    # 계좌별/종목별 누적 배당금 사전 계산
    div_records = data.records[(data.records["배당"].notna()) & (pd.to_numeric(data.records["배당"], errors="coerce") > 0)].copy()
    div_by_acct_sym: Dict[Tuple[str, str], float] = {}
    if not div_records.empty:
        div_records["배당원화"] = div_records.apply(
            lambda r: update_fa.convert_to_krw(r["계좌"], float(r["배당"]), pd.Timestamp(r["일자"]), data.fx_series_month),
            axis=1,
        )
        for (acct_code, sym), grp in div_records.groupby(["계좌", "종목"]):
            div_by_acct_sym[(str(acct_code).strip(), str(sym).strip())] = float(grp["배당원화"].sum())

    accounts = list(data.valid_detail_accounts)
    for idx, account in enumerate(accounts):
        label = update_fa.account_label(account)
        tab_id = f"fa-tab-content-{account}"
        active_cls = " active" if idx == 0 else ""

        # 상단 가로 탭 버튼
        tab_btns.append(
            f"<button type='button' class='fa-tab-btn{active_cls}' data-target='{tab_id}'>{html.escape(label)}</button>"
        )

        # 계좌 요약 데이터
        status_row = data.summary_df[data.summary_df["계좌"] == account]
        status_data = status_row.iloc[0] if not status_row.empty else {}

        invest_val = _as_float(status_data.get("투자금"))
        eval_val = _as_float(status_data.get("평가금"))
        dividend_val = _as_float(status_data.get("배당금"))

        # 계좌별 보유 종목 데이터
        account_holdings = data.holdings_df[data.holdings_df["계좌"] == account].copy()
        if account == "sema" and account_holdings.empty and not status_row.empty:
            account_holdings = pd.DataFrame([{
                "계좌": "sema",
                "종목": "교직원공제회",
                "매수금": invest_val,
                "평가금": eval_val,
                "수익금": _as_float(status_data.get("수익금")),
                "수익률": (_as_float(status_data.get("수익금")) / invest_val) if (invest_val and invest_val > 0) else 0.0,
            }])

        if not account_holdings.empty:
            account_holdings = account_holdings.sort_values("평가금", ascending=False)

        raw_account_name = update_fa.ACCOUNT_RAW_NAMES.get(account)
        if not raw_account_name:
            title_val = update_fa.ACCOUNT_TITLES.get(f"title_{account}_detail", "")
            raw_account_name = title_val.replace("◉ 상세계좌: ", "").strip()

        rebal_df = update_fa.calculate_rebalancing_df(raw_account_name, account_holdings, data.symbol_map)

        buy_val = account_holdings["매수금"].sum() if not account_holdings.empty else 0.0
        if buy_val == 0.0 and account == "sema" and invest_val is not None:
            buy_val = invest_val

        # 투자금 대비 수익금 및 수익률
        invest_profit = _as_float(status_data.get("수익금"))
        if invest_profit is None and eval_val is not None and invest_val is not None:
            invest_profit = eval_val - invest_val
        invest_rate = (invest_profit / invest_val * 100.0) if (invest_val and invest_val > 0 and invest_profit is not None) else 0.0

        # 매수금 대비 수익금 및 수익률
        buy_profit = (eval_val - buy_val) if (eval_val is not None and buy_val > 0) else 0.0
        buy_rate = (buy_profit / buy_val * 100.0) if (buy_val > 0 and buy_profit is not None) else 0.0

        inv_p_cls = "fa-num-positive" if (invest_profit or 0) > 0 else "fa-num-negative" if (invest_profit or 0) < 0 else ""
        inv_p_bdg = "fa-badge-positive" if (invest_profit or 0) > 0 else "fa-badge-negative" if (invest_profit or 0) < 0 else "fa-badge-neutral"

        buy_p_cls = "fa-num-positive" if (buy_profit or 0) > 0 else "fa-num-negative" if (buy_profit or 0) < 0 else ""
        buy_p_bdg = "fa-badge-positive" if (buy_profit or 0) > 0 else "fa-badge-negative" if (buy_profit or 0) < 0 else "fa-badge-neutral"

        # 계좌 요약 미니 KPI 그리드 (평가/배당 -> 투자/투자수익 -> 매수/매수수익)
        mini_kpis = [
            f"<div class='fa-mini-kpi'><div class='fa-mini-kpi-lbl'>현재 평가금</div><div class='fa-mini-kpi-val fa-font-bold'>{eval_val:,.0f}원</div></div>" if eval_val is not None else "",
            f"<div class='fa-mini-kpi'><div class='fa-mini-kpi-lbl'>누적 배당금</div><div class='fa-mini-kpi-val' style='color: var(--fa-purple);'>{dividend_val:,.0f}원</div></div>" if dividend_val is not None and dividend_val > 0 else "",
            f"<div class='fa-mini-kpi'><div class='fa-mini-kpi-lbl'>투자금 (원금)</div><div class='fa-mini-kpi-val'>{invest_val:,.0f}원</div></div>" if invest_val is not None else "",
            f"<div class='fa-mini-kpi'><div class='fa-mini-kpi-lbl'>투자 대비 수익</div><div class='fa-mini-kpi-val {inv_p_cls}'>{invest_profit:+,.0f}원 <span class='fa-badge {inv_p_bdg}'>{invest_rate:+.2f}%</span></div></div>" if invest_profit is not None else "",
            f"<div class='fa-mini-kpi'><div class='fa-mini-kpi-lbl'>총 매수금</div><div class='fa-mini-kpi-val'>{buy_val:,.0f}원</div></div>" if buy_val > 0 else "",
            f"<div class='fa-mini-kpi'><div class='fa-mini-kpi-lbl'>매수 대비 수익</div><div class='fa-mini-kpi-val {buy_p_cls}'>{buy_profit:+,.0f}원 <span class='fa-badge {buy_p_bdg}'>{buy_rate:+.2f}%</span></div></div>" if buy_val > 0 else "",
        ]
        mini_kpi_html = f"<div class='fa-mini-kpi-grid'>{''.join(mini_kpis)}</div>"

        # 도넛 차트 (큼직한 도넛 + 내부 % + 하단 종목명 범례)
        chart_html = ""
        if not account_holdings.empty and account_holdings["평가금"].sum() > 0:
            pie_fig = go.Figure(
                data=[
                    go.Pie(
                        labels=account_holdings["종목"],
                        values=account_holdings["평가금"],
                        textinfo="percent",
                        textposition="inside",
                        insidetextfont=dict(size=14, color="#ffffff", family=FONT_FAMILY),
                        insidetextorientation="horizontal",
                        hole=0.46,
                        showlegend=True,
                        marker=dict(colors=[_palette_color(i) for i in range(len(account_holdings))]),
                        hovertemplate="<b>%{label}</b><br>평가금: %{value:,.0f}원<br>비중: %{percent}<extra></extra>",
                    )
                ]
            )
            pie_fig.update_layout(
                height=320,
                margin=dict(l=10, r=10, t=10, b=45),
                paper_bgcolor=THEME_BG,
                plot_bgcolor=THEME_BG,
                font=dict(family=FONT_FAMILY, size=13),
                legend=dict(
                    orientation="h",
                    yanchor="top",
                    y=-0.08,
                    xanchor="center",
                    x=0.5,
                    font=dict(size=13, family=FONT_FAMILY),
                ),
            )
            chart_html = fig_renderer(pie_fig)

        # 종목별 카드 그리드 HTML (2열 반응형 종목 카드)
        stock_cards = []
        total_acct_eval = account_holdings["평가금"].sum() if not account_holdings.empty else 0.0
        total_acct_buy = buy_val if buy_val > 0 else 1.0

        for _, hrow in account_holdings.iterrows():
            sym = str(hrow["종목"]).strip()
            b_amt = _as_float(hrow.get("매수금")) or 0.0
            e_amt = _as_float(hrow.get("평가금")) or 0.0
            weight_pct = (e_amt / total_acct_eval * 100.0) if total_acct_eval > 0 else 0.0

            # 종목별 매수금 대비 수익금 및 수익률
            buy_p_amt = e_amt - b_amt
            buy_r_rate = (buy_p_amt / b_amt * 100.0) if b_amt > 0 else 0.0

            # 종목별 투자금(계좌 원금 배분액) 대비 수익금 및 수익률
            alloc_inv_amt = (invest_val * (b_amt / total_acct_buy)) if (invest_val and invest_val > 0 and b_amt > 0) else b_amt
            inv_p_amt = e_amt - alloc_inv_amt
            inv_r_rate = (inv_p_amt / alloc_inv_amt * 100.0) if alloc_inv_amt > 0 else 0.0

            cum_div = div_by_acct_sym.get((account, sym), 0.0)
            if cum_div == 0.0 and account == "sema" and dividend_val is not None:
                cum_div = dividend_val
            div_str = f"{cum_div:,.0f}원"

            buy_p_cls = "fa-num-positive" if buy_p_amt > 0 else "fa-num-negative" if buy_p_amt < 0 else ""
            buy_p_bdg = "fa-badge-positive" if buy_p_amt > 0 else "fa-badge-negative" if buy_p_amt < 0 else "fa-badge-neutral"

            inv_p_cls = "fa-num-positive" if inv_p_amt > 0 else "fa-num-negative" if inv_p_amt < 0 else ""
            inv_p_bdg = "fa-badge-positive" if inv_p_amt > 0 else "fa-badge-negative" if inv_p_amt < 0 else "fa-badge-neutral"

            b_str = f"{b_amt:,.0f}원"
            e_str = f"{e_amt:,.0f}원"
            inv_str = f"{alloc_inv_amt:,.0f}원"
            buy_p_str = f"{buy_p_amt:+,.0f}원" if buy_p_amt != 0 else "0원"
            buy_r_str = f"{buy_r_rate:+.2f}%"
            inv_p_str = f"{inv_p_amt:+,.0f}원" if inv_p_amt != 0 else "0원"
            inv_r_str = f"{inv_r_rate:+.2f}%"

            stock_cards.append(
                f"<div class='fa-stock-card'>"
                f"  <div class='fa-stock-card-head'>"
                f"    <div class='fa-stock-card-title'>{html.escape(sym)}</div>"
                f"    <div class='fa-stock-card-badges'>"
                f"      <span class='fa-chip-weight'>비중 {weight_pct:.1f}%</span>"
                f"      <span class='fa-badge {buy_p_bdg}'>{buy_r_str}</span>"
                f"    </div>"
                f"  </div>"
                f"  <div class='fa-stock-card-body'>"
                f"    <div class='fa-stock-field'>"
                f"      <span class='fa-stock-lbl'>현재 평가금</span>"
                f"      <span class='fa-stock-val fa-font-bold'>{e_str}</span>"
                f"    </div>"
                f"    <div class='fa-stock-field'>"
                f"      <span class='fa-stock-lbl'>누적 배당금</span>"
                f"      <span class='fa-stock-val' style='color: var(--fa-purple);'>{div_str}</span>"
                f"    </div>"
                f"    <div class='fa-stock-field'>"
                f"      <span class='fa-stock-lbl'>투자금 (원금)</span>"
                f"      <span class='fa-stock-val'>{inv_str}</span>"
                f"    </div>"
                f"    <div class='fa-stock-field'>"
                f"      <span class='fa-stock-lbl'>투자 대비 수익</span>"
                f"      <span class='fa-stock-val {inv_p_cls}'>{inv_p_str} <span class='fa-badge {inv_p_bdg}' style='font-size:0.75rem; padding:1px 6px;'>{inv_r_str}</span></span>"
                f"    </div>"
                f"    <div class='fa-stock-field'>"
                f"      <span class='fa-stock-lbl'>총 매수금</span>"
                f"      <span class='fa-stock-val'>{b_str}</span>"
                f"    </div>"
                f"    <div class='fa-stock-field'>"
                f"      <span class='fa-stock-lbl'>매수 대비 수익</span>"
                f"      <span class='fa-stock-val {buy_p_cls}'>{buy_p_str} <span class='fa-badge {buy_p_bdg}' style='font-size:0.75rem; padding:1px 6px;'>{buy_r_str}</span></span>"
                f"    </div>"
                f"  </div>"
                f"</div>"
            )

        stock_cards_html = f"<div class='fa-stock-grid'>{''.join(stock_cards)}</div>"

        # 리밸런싱 가이드 뱃지 카드들
        rebal_cards = []
        if rebal_df is not None and not rebal_df.empty:
            for _, rrow in rebal_df.iterrows():
                diff = rrow["조정금액"]
                diff_q = rrow.get("조정주수", 0)
                asset_name = str(rrow["자산군"])
                if diff > 100:
                    qty_str = f" (+{diff_q:g}주)" if diff_q > 0 else ""
                    rebal_cards.append(
                        f"<div class='fa-rebal-item buy'>"
                        f"<div class='fa-rebal-tag buy'>매수 필요</div>"
                        f"<div class='fa-rebal-name'>{html.escape(asset_name)}</div>"
                        f"<div class='fa-rebal-val'>+{diff:,.0f}원{qty_str}</div>"
                        f"</div>"
                    )
                elif diff < -100:
                    qty_str = f" ({diff_q:g}주)" if diff_q < 0 else ""
                    rebal_cards.append(
                        f"<div class='fa-rebal-item sell'>"
                        f"<div class='fa-rebal-tag sell'>매도 필요</div>"
                        f"<div class='fa-rebal-name'>{html.escape(asset_name)}</div>"
                        f"<div class='fa-rebal-val'>-{abs(diff):,.0f}원{qty_str}</div>"
                        f"</div>"
                    )
                else:
                    rebal_cards.append(
                        f"<div class='fa-rebal-item ok'>"
                        f"<div class='fa-rebal-tag ok'>비중 적정</div>"
                        f"<div class='fa-rebal-name'>{html.escape(asset_name)}</div>"
                        f"<div class='fa-rebal-val'>0원 (목표 유지)</div>"
                        f"</div>"
                    )

        rebal_html = ""
        # 계좌별 최근 거래내역 HTML 생성
        acct_summary, acct_items = _build_trading_history(
            data.records,
            data.fx_series_month,
            data.month_end,
            filter_account=account,
            limit=8,
        )
        acct_history_html = ""
        if acct_items:
            badge_map = {
                "buy": ("매수", "fa-badge-positive", "fa-num-positive"),
                "sell": ("매도", "fa-badge-negative", "fa-num-negative"),
                "div": ("배당", "fa-badge-purple", "fa-num-purple"),
                "invest": ("투자금", "fa-badge-neutral", ""),
            }
            acct_item_rows = []
            for a_item in acct_items:
                kind = a_item.get("kind", "")
                badge_text, badge_cls, amt_cls = badge_map.get(kind, ("기타", "fa-badge-neutral", ""))
                date_str = a_item.get("date", "")
                symbol = a_item.get("symbol", "")
                amount_str = a_item.get("amount_str", "")
                sub_detail = a_item.get("sub_detail", "")

                acct_item_rows.append(
                    f"<div class='fa-history-card'>"
                    f"  <div class='fa-history-card-left'>"
                    f"    <div class='fa-history-card-header'>"
                    f"      <span class='fa-badge {badge_cls}'>{badge_text}</span>"
                    f"      <span class='fa-history-date'>{html.escape(date_str)}</span>"
                    f"    </div>"
                    f"    <div class='fa-history-symbol'>{html.escape(symbol)}</div>"
                    f"  </div>"
                    f"  <div class='fa-history-card-right'>"
                    f"    <div class='fa-history-amount {amt_cls}'>{html.escape(amount_str)}</div>"
                    f"    <div class='fa-history-subdetail'>{html.escape(sub_detail)}</div>"
                    f"  </div>"
                    f"</div>"
                )

            period_label = str(acct_summary.get("period_str", "최근 거래"))
            acct_history_html = (
                f"<div class='fa-acct-history-section'>"
                f"  <div class='fa-subcard-title'>📋 최근 계좌 거래내역 ({period_label})</div>"
                f"  <div class='fa-history-list'>{''.join(acct_item_rows)}</div>"
                f"</div>"
            )

        # 탭 패널 완성
        pane_html = (
            f"<div id='{tab_id}' class='fa-tab-pane{active_cls}'>"
            f"{mini_kpi_html}"
            f"<div class='fa-account-split-grid'>"
            f"  <div class='fa-account-chart-col'>"
            f"    <div class='fa-subcard-title'>자산 비중</div>"
            f"    <div class='fa-account-chart-card'>{chart_html}</div>"
            f"  </div>"
            f"  <div class='fa-account-table-col'>"
            f"    <div class='fa-subcard-title'>보유 종목 현황</div>"
            f"    {stock_cards_html}"
            f"  </div>"
            f"</div>"
            f"{rebal_html}"
            f"{acct_history_html}"
            f"</div>"
        )
        tab_panes.append(pane_html)

    # 탭 네비게이션 + 컨텐츠 전체를 카드에 패키징
    nav_html = f"<div class='fa-tab-nav-wrapper'><div class='fa-tab-nav' role='tablist'>{''.join(tab_btns)}</div></div>"
    content_html = f"<div class='fa-tab-panes'>{''.join(tab_panes)}</div>"

    return (
        "<section class='fa-card fa-card-wide fa-card-tabs'>"
        "<header class='fa-card-head'><h2>상세 계좌 현황 & 리밸런싱</h2></header>"
        f"<div class='fa-card-body'>{nav_html}{content_html}</div>"
        "</section>"
    )


def _build_trading_history(
    records: pd.DataFrame,
    fx_series: pd.Series,
    month_end: pd.Timestamp,
    filter_account: Optional[str] = None,
    limit: Optional[int] = None,
) -> Tuple[Dict[str, object], List[Dict[str, str]]]:
    """거래 내역을 정형화된 데이터와 리스트로 반환 (특정 계좌 필터 지원)"""
    target_records = records.copy()
    if filter_account:
        target_records = target_records[target_records["계좌"].astype(str).str.strip() == filter_account]

    period = month_end.to_period("M")
    start = period.start_time
    end = period.end_time
    month_records = target_records[(target_records["일자"] >= start) & (target_records["일자"] <= end)].copy()

    is_recent_mode = False
    # 특정 계좌 필터인데 당월 거래가 없을 경우, 해당 계좌의 가장 최근 거래 N건 가져오기
    if month_records.empty and filter_account and not target_records.empty:
        month_records = target_records.sort_values("일자", ascending=False).head(limit or 5).copy()
        is_recent_mode = True

    if month_records.empty:
        return {}, []

    buy_total = sell_total = invest_total = div_total = 0.0
    items: List[Dict[str, str]] = []

    def fmt_currency(val: float) -> str:
        return f"{val:,.0f}원"

    month_records = month_records.sort_values("일자", ascending=False)
    if limit and limit > 0:
        month_records = month_records.head(limit)

    for _, row in month_records.iterrows():
        date = pd.Timestamp(row["일자"])
        date_str = f"{date:%Y.%m.%d}"
        acct_code = str(row.get("계좌", "")).strip()
        account = update_fa.account_label(acct_code)
        symbol = str(row.get("종목", "")).strip()
        qty = row.get("수량")
        price = row.get("단가")
        dividend = row.get("배당")
        invest = row.get("투자금")

        has_qty_price = pd.notna(qty) and pd.notna(price) and qty != 0
        has_dividend = pd.notna(dividend) and dividend != 0
        has_invest = pd.notna(invest) and invest != 0

        if has_qty_price:
            trade_amt = update_fa.convert_to_krw(acct_code, float(qty) * float(price), date, fx_series)
            unit_price = update_fa.convert_to_krw(acct_code, float(price), date, fx_series)
            if qty > 0:
                buy_total += trade_amt
                items.append({
                    "kind": "buy",
                    "date": date_str,
                    "account": account,
                    "symbol": symbol,
                    "amount_str": f"+{fmt_currency(trade_amt)}",
                    "sub_detail": f"단가 {fmt_currency(unit_price)} · {abs(qty):g}주",
                })
            else:
                sell_total += abs(trade_amt)
                items.append({
                    "kind": "sell",
                    "date": date_str,
                    "account": account,
                    "symbol": symbol,
                    "amount_str": f"-{fmt_currency(abs(trade_amt))}",
                    "sub_detail": f"단가 {fmt_currency(unit_price)} · {abs(qty):g}주",
                })
        if has_dividend:
            div_amt = update_fa.convert_to_krw(acct_code, float(dividend), date, fx_series)
            div_total += div_amt
            native_str = "" if acct_code not in update_fa.USD_ACCOUNTS else f" ({dividend}달러)"
            items.append({
                "kind": "div",
                "date": date_str,
                "account": account,
                "symbol": symbol,
                "amount_str": f"+{fmt_currency(div_amt)}",
                "sub_detail": f"배당금 수령{native_str}",
            })
        if has_invest:
            invest_amt = float(str(invest).replace(",", "")) if invest else 0.0
            if acct_code in update_fa.USD_ACCOUNTS:
                invest_amt = update_fa.convert_to_krw(acct_code, invest_amt, date, fx_series)
            invest_total += invest_amt
            items.append({
                "kind": "invest",
                "date": date_str,
                "account": account,
                "symbol": "투자금 증액",
                "amount_str": f"+{fmt_currency(invest_amt)}",
                "sub_detail": "계좌 입금",
            })

    period_str = f"{period.year}년 {period.month:02d}월" if not is_recent_mode else "최근 거래"
    summary_data: Dict[str, object] = {
        "period_str": period_str,
        "invest_total": invest_total,
        "buy_total": buy_total,
        "sell_total": sell_total,
        "div_total": div_total,
    }
    return summary_data, items


def _render_history_html(summary_data: Dict[str, object], items: List[Dict[str, str]]) -> str:
    if not summary_data and not items:
        return "<p class='fa-empty-text'>해당 월의 거래 내역이 없습니다.</p>"

    period_str = str(summary_data.get("period_str", "-"))
    invest_amt = _as_float(summary_data.get("invest_total")) or 0.0
    buy_amt = _as_float(summary_data.get("buy_total")) or 0.0
    sell_amt = _as_float(summary_data.get("sell_total")) or 0.0
    div_amt = _as_float(summary_data.get("div_total")) or 0.0

    cards = [
        f"<div class='fa-kpi-card'>"
        f"  <div class='fa-kpi-label'>집계 기간</div>"
        f"  <div class='fa-kpi-value'>{html.escape(period_str)}</div>"
        f"  <div class='fa-kpi-sub'>당월 거래</div>"
        f"</div>",
        f"<div class='fa-kpi-card'>"
        f"  <div class='fa-kpi-label'>투자금 증액</div>"
        f"  <div class='fa-kpi-value'>{invest_amt:,.0f}원</div>"
        f"  <div class='fa-kpi-sub'>원금 입금</div>"
        f"</div>",
        f"<div class='fa-kpi-card'>"
        f"  <div class='fa-kpi-label'>총 매수금</div>"
        f"  <div class='fa-kpi-value fa-num-positive'>{buy_amt:,.0f}원</div>"
        f"  <div class='fa-kpi-sub'>매수 체결</div>"
        f"</div>",
        f"<div class='fa-kpi-card'>"
        f"  <div class='fa-kpi-label'>총 매도금</div>"
        f"  <div class='fa-kpi-value fa-num-negative'>{sell_amt:,.0f}원</div>"
        f"  <div class='fa-kpi-sub'>매도 체결</div>"
        f"</div>",
        f"<div class='fa-kpi-card'>"
        f"  <div class='fa-kpi-label'>총 배당금</div>"
        f"  <div class='fa-kpi-value' style='color: var(--fa-purple);'>{div_amt:,.0f}원</div>"
        f"  <div class='fa-kpi-sub'>배당 수령</div>"
        f"</div>",
    ]

    summary_grid = f"<div class='fa-kpi-grid fa-history-kpi-grid'>{''.join(cards)}</div>"
    item_rows = []

    badge_map = {
        "buy": ("매수", "fa-badge-positive", "fa-num-positive"),
        "sell": ("매도", "fa-badge-negative", "fa-num-negative"),
        "div": ("배당", "fa-badge-purple", "fa-num-purple"),
        "invest": ("투자금", "fa-badge-neutral", ""),
    }

    for item in items:
        kind = item.get("kind", "")
        badge_text, badge_cls, amt_cls = badge_map.get(kind, ("기타", "fa-badge-neutral", ""))
        date_str = item.get("date", "")
        account = item.get("account", "")
        symbol = item.get("symbol", "")
        amount_str = item.get("amount_str", "")
        sub_detail = item.get("sub_detail", "")

        item_rows.append(
            f"<div class='fa-history-card'>"
            f"  <div class='fa-history-card-left'>"
            f"    <div class='fa-history-card-header'>"
            f"      <span class='fa-badge {badge_cls}'>{badge_text}</span>"
            f"      <span class='fa-history-account'>{html.escape(account)}</span>"
            f"      <span class='fa-history-date'>{html.escape(date_str)}</span>"
            f"    </div>"
            f"    <div class='fa-history-symbol'>{html.escape(symbol)}</div>"
            f"  </div>"
            f"  <div class='fa-history-card-right'>"
            f"    <div class='fa-history-amount {amt_cls}'>{html.escape(amount_str)}</div>"
            f"    <div class='fa-history-subdetail'>{html.escape(sub_detail)}</div>"
            f"  </div>"
            f"</div>"
        )

    return f"{summary_grid}<div class='fa-history-list'>{''.join(item_rows)}</div>"


def _render_figure_html(fig: go.Figure) -> str:
    fig.update_xaxes(fixedrange=True)
    fig.update_yaxes(fixedrange=True)
    fig.update_layout(dragmode=False)
    return pio.to_html(
        fig,
        full_html=False,
        include_plotlyjs=False,
        config={
            "displayModeBar": False,
            "responsive": True,
            "scrollZoom": False,
            "doubleClick": False,
        },
    )


def _dashboard_card(title: str, body_html: str, extra_class: str = "") -> str:
    klass = f"fa-card {extra_class}".strip()
    return (
        f"<section class=\"{klass}\">"
        f"<header class=\"fa-card-head\"><h2>{html.escape(title)}</h2></header>"
        f"<div class=\"fa-card-body\">{body_html}</div>"
        "</section>"
    )


def _build_report_data(records: pd.DataFrame) -> ReportData:
    latest_date = records["일자"].dropna().max()
    if pd.isna(latest_date):
        raise ValueError("No valid dates in trading records.")

    fx_series_full = update_fa.build_fx_series(records, latest_date)
    period = latest_date.to_period("M")
    month_end = period.to_timestamp(how="end")
    records_upto = records[records["일자"] <= month_end].copy()
    fx_series_month = fx_series_full.loc[:month_end]

    account_df = update_fa.build_account_valuation_df(records_upto, fx_series_month, month_end)
    summary_df = update_fa.build_account_assets(records_upto, account_df, fx_series_month)
    holdings_df = update_fa.build_holdings_df(records_upto, fx_series_full)

    try:
        dividends_pivot = update_fa.load_dividend_pivot(records_upto, fx_series_month, month_end)
    except ValueError:
        dividends_pivot = None

    try:
        yearly_dividends_pivot = update_fa.load_yearly_dividend_pivot(records_upto, fx_series_month, month_end)
    except ValueError:
        yearly_dividends_pivot = None

    valid_detail_accounts = []
    if not summary_df.empty and "평가금" in summary_df.columns:
        valid_detail_accounts = summary_df.loc[
            (summary_df["계좌"] != "합계") & (summary_df["평가금"] > 0),
            "계좌",
        ].astype(str).tolist()

    symbol_map = update_fa.load_symbol_map()
    invest_series = update_fa._build_investment_series(records_upto, fx_series_month)

    return ReportData(
        month_end=month_end,
        records=records_upto,
        fx_series_full=fx_series_full,
        fx_series_month=fx_series_month,
        account_df=account_df,
        summary_df=summary_df,
        holdings_df=holdings_df,
        dividends_pivot=dividends_pivot,
        yearly_dividends_pivot=yearly_dividends_pivot,
        valid_detail_accounts=valid_detail_accounts,
        invest_series=invest_series,
        symbol_map=symbol_map,
    )


def _build_dashboard_fragment(data: ReportData) -> str:
    def fig_html(fig: go.Figure) -> str:
        return _render_figure_html(fig)

    assets_investment_fig = _build_assets_investment_trend(data.account_df, data.invest_series)
    assets_fig = _build_assets_trend(data.account_df)
    portfolio_alloc_fig = _build_portfolio_allocation_charts(data.holdings_df, data.symbol_map)
    account_summary_html = _build_account_assets_html_table(data.summary_df)
    holdings_html = _build_total_holdings_html_table(data.holdings_df)

    monthly_div_fig = (
        _build_dividends_chart(data.dividends_pivot)
        if data.dividends_pivot is not None and not data.dividends_pivot.empty
        else None
    )
    yearly_div_fig = (
        _build_yearly_dividends_chart(data.yearly_dividends_pivot)
        if data.yearly_dividends_pivot is not None and not data.yearly_dividends_pivot.empty
        else None
    )
    trading_summary, trading_items = _build_trading_history(data.records, data.fx_series_month, data.month_end)

    account_detail_section_html = _build_account_detail_section(data, fig_html)

    blocks: List[str] = [
        "<section class=\"fa-hero\">"
        "<div class=\"fa-hero-header\">"
        "  <div>"
        f"    <div class=\"fa-hero-title\">{html.escape(data.month_end.strftime('%Y년 %m월 자산 대시보드'))} <span class=\"fa-badge fa-badge-neutral\" style=\"font-size:0.75rem; vertical-align:middle; margin-left:6px;\">{APP_VERSION}</span></div>"
        f"    <div class=\"fa-hero-meta\">최종 업데이트: {html.escape(data.month_end.strftime('%Y-%m-%d'))} · Engine {APP_VERSION}</div>"
        "  </div>"
        "  <a href=\"https://fa-admin.vividian.net\" class=\"fa-btn-admin\" target=\"_blank\" rel=\"noopener noreferrer\">"
        "    <span>⚙️ 거래내역 관리</span>"
        "  </a>"
        "</div>"
        "</section>",
        _build_kpi_row(data),
        _build_market_kpi_row(),
        _dashboard_card(update_fa.ACCOUNT_TITLES.get("title_assets_investment_trend", "누적 투자금 vs 평가금 추세"), fig_html(assets_investment_fig), extra_class="fa-card-wide"),
        _dashboard_card(update_fa.ACCOUNT_TITLES.get("title_assets_trend", "전체 금융자산 추이"), fig_html(assets_fig), extra_class="fa-card-wide"),
        _dashboard_card("전체 포트폴리오 비중", fig_html(portfolio_alloc_fig), extra_class="fa-card-wide"),
        _dashboard_card(update_fa.ACCOUNT_TITLES.get("title_account_assets", "계좌 요약"), account_summary_html, extra_class="fa-card-wide"),
        _dashboard_card(update_fa.ACCOUNT_TITLES.get("title_total_holdings", "전체 보유 종목"), holdings_html, extra_class="fa-card-wide"),
    ]

    if monthly_div_fig is not None:
        blocks.append(
            _dashboard_card(
                update_fa.ACCOUNT_TITLES.get("title_monthly_dividends", "월별 배당금 추이"),
                fig_html(monthly_div_fig),
                extra_class="fa-card-wide",
            ),
        )
    if yearly_div_fig is not None:
        blocks.append(
            _dashboard_card(
                update_fa.ACCOUNT_TITLES.get("title_yearly_dividends", "연별 배당금 현황"),
                fig_html(yearly_div_fig),
                extra_class="fa-card-wide",
            ),
        )

    if account_detail_section_html:
        blocks.append(account_detail_section_html)

    if trading_summary or trading_items:
        blocks.append(
            _dashboard_card(
                update_fa.ACCOUNT_TITLES.get("title_trading_history", "최근 거래 내역"),
                _render_history_html(trading_summary, trading_items),
                extra_class="fa-card-wide",
            )
        )

    styles = """
<style>
/* =========================================================
   FA Modern Fintech Dashboard Design System
   ========================================================= */
.fa-dashboard {
  --fa-bg: transparent;
  --fa-card-bg: #ffffff;
  --fa-card-border: #e2e8f0;
  --fa-card-shadow: 0 4px 16px -2px rgba(0, 0, 0, 0.06), 0 2px 4px -1px rgba(0, 0, 0, 0.04);
  --fa-text-main: #0f172a;
  --fa-text-muted: #64748b;
  --fa-text-sub: #94a3b8;
  --fa-kpi-bg: #f8fafc;
  --fa-table-header-bg: #f1f5f9;
  --fa-table-stripe: #f8fafc;
  --fa-table-hover: #f1f5f9;
  --fa-border: #e2e8f0;
  
  --fa-gain: #e53e3e;
  --fa-gain-bg: #fff5f5;
  --fa-loss: #3182ce;
  --fa-loss-bg: #ebf8ff;
  --fa-accent: #4f46e5;
  --fa-accent-bg: #eef2ff;
  --fa-purple: #805ad5;
  --fa-purple-bg: #faf5ff;
  --fa-ok: #38a169;
  --fa-ok-bg: #f0fff4;

  font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Noto Sans KR", sans-serif;
  color: var(--fa-text-main);
  line-height: 1.5;
}

/* =========================================================
   Sleek Dark Mode (블로그 다크모드와 완벽하게 어우러지는 세련된 테마)
   ========================================================= */
body[data-theme="dark"] .fa-dashboard,
html[data-theme="dark"] .fa-dashboard,
html.dark .fa-dashboard,
.dark .fa-dashboard {
  --fa-bg: transparent;
  --fa-card-bg: #1e293b;
  --fa-card-border: #334155;
  --fa-card-shadow: 0 4px 20px -2px rgba(0, 0, 0, 0.4), 0 2px 6px -1px rgba(0, 0, 0, 0.2);
  --fa-text-main: #f8fafc;
  --fa-text-muted: #94a3b8;
  --fa-text-sub: #64748b;
  --fa-kpi-bg: #0f172a;
  --fa-table-header-bg: #0f172a;
  --fa-table-stripe: #182234;
  --fa-table-hover: #283548;
  --fa-border: #334155;
  
  --fa-gain: #f87171;
  --fa-gain-bg: rgba(239, 68, 68, 0.15);
  --fa-loss: #60a5fa;
  --fa-loss-bg: rgba(59, 130, 246, 0.15);
  --fa-accent: #818cf8;
  --fa-accent-bg: rgba(99, 102, 241, 0.15);
  --fa-purple: #c084fc;
  --fa-purple-bg: rgba(168, 85, 247, 0.15);
  --fa-ok: #4ade80;
  --fa-ok-bg: rgba(34, 197, 94, 0.15);
}

.fa-kpi-card,
.fa-card,
.fa-stock-card,
.fa-trade-card,
.fa-rebal-item {
  background: var(--fa-card-bg);
  border: 1px solid var(--fa-card-border);
  color: var(--fa-text-main);
  box-shadow: var(--fa-card-shadow);
}

.fa-card-head {
  background: var(--fa-card-bg);
  border-bottom: 1px solid var(--fa-card-border);
  color: var(--fa-text-main);
}

/* 숫자 서식 및 색상 유틸리티 */
.fa-num { font-variant-numeric: tabular-nums; letter-spacing: -0.01em; }
.fa-num-positive { color: var(--fa-gain) !important; font-weight: 600; }
.fa-num-negative { color: var(--fa-loss) !important; font-weight: 600; }
.fa-font-bold { font-weight: 700; }
.text-right { text-align: right; }
.text-center { text-align: center; }

/* 뱃지 */
.fa-badge {
  display: inline-block;
  padding: 3px 8px;
  border-radius: 6px;
  font-size: 0.8rem;
  font-weight: 600;
  line-height: 1.2;
}
.fa-badge-positive { background: var(--fa-gain-bg); color: var(--fa-gain); }
.fa-badge-negative { background: var(--fa-loss-bg); color: var(--fa-loss); }
.fa-badge-neutral { background: var(--fa-table-header-bg); color: var(--fa-text-muted); }
.fa-badge-purple { background: var(--fa-purple-bg); color: var(--fa-purple); }

.fa-chip-account {
  display: inline-block;
  padding: 2px 8px;
  background: var(--fa-accent-bg);
  color: var(--fa-accent);
  border-radius: 9999px;
  font-size: 0.75rem;
  font-weight: 600;
}

/* Hero Section */
.fa-hero { margin: 8px 0 16px; }
.fa-hero-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  flex-wrap: wrap;
  gap: 12px;
}
.fa-hero-title { font-size: 1.55rem; font-weight: 800; letter-spacing: -0.02em; }
.fa-hero-meta { margin-top: 4px; color: var(--fa-text-muted); font-size: 0.88rem; }
.fa-btn-admin {
  display: inline-flex;
  align-items: center;
  gap: 6px;
  background: #4f46e5 !important;
  color: #ffffff !important;
  border: 1px solid #4338ca !important;
  padding: 8px 16px;
  border-radius: 8px;
  font-size: 0.85rem;
  font-weight: 700;
  text-decoration: none;
  box-shadow: 0 2px 8px rgba(79, 70, 229, 0.3) !important;
  transition: all 0.2s ease;
}
.fa-btn-admin:hover {
  background: #4338ca !important;
  color: #ffffff !important;
  transform: translateY(-1px);
  box-shadow: 0 4px 12px rgba(79, 70, 229, 0.4) !important;
}

/* KPI Grid */
.fa-kpi-grid {
  display: grid;
  grid-template-columns: repeat(6, minmax(0, 1fr));
  gap: 12px;
  margin-bottom: 20px;
}
@media (max-width: 1100px) { .fa-kpi-grid { grid-template-columns: repeat(3, minmax(0, 1fr)); } }
@media (max-width: 640px) { .fa-kpi-grid { grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 8px; } }

.fa-kpi-card {
  background: var(--fa-card-bg);
  border: 1px solid var(--fa-card-border);
  border-radius: 14px;
  padding: 14px 16px;
  box-shadow: var(--fa-card-shadow);
  display: flex;
  flex-direction: column;
  justify-content: space-between;
}
.fa-kpi-label { font-size: 0.82rem; color: var(--fa-text-muted); font-weight: 500; }
.fa-kpi-value { font-size: 1.18rem; font-weight: 800; margin: 6px 0 2px; }
.fa-kpi-sub { font-size: 0.76rem; color: var(--fa-text-sub); }

/* Card Wrapper */
.fa-card {
  background: var(--fa-card-bg);
  border: 1px solid var(--fa-card-border);
  border-radius: 16px;
  box-shadow: var(--fa-card-shadow);
  margin-bottom: 24px;
  overflow: hidden;
}
.fa-card-head {
  padding: 16px 20px;
  border-bottom: 1px solid var(--fa-card-border);
  background: var(--fa-card-bg);
}
.fa-card-head h2 { margin: 0; font-size: 1.12rem; font-weight: 700; }
.fa-card-body { padding: 16px 20px; }
.fa-subcard-title { font-size: 0.95rem; font-weight: 700; color: var(--fa-text-muted); margin-bottom: 10px; }

/* Table System (PC Standard) */
.fa-table-wrapper {
  width: 100%;
  overflow-x: auto;
  -webkit-overflow-scrolling: touch;
}
.fa-table {
  width: 100%;
  border-collapse: collapse;
  font-size: 0.9rem;
  text-align: left;
}
.fa-table th {
  background: var(--fa-table-header-bg);
  color: var(--fa-text-muted);
  font-weight: 600;
  padding: 10px 14px;
  border-bottom: 1px solid var(--fa-border);
  white-space: nowrap;
}
.fa-table td {
  padding: 12px 14px;
  border-bottom: 1px solid var(--fa-border);
  color: var(--fa-text-main);
  vertical-align: middle;
}
.fa-table tbody tr:nth-child(even) { background: var(--fa-table-stripe); }
.fa-table tbody tr:hover { background: var(--fa-table-hover); transition: background 0.15s ease; }
.fa-tr-total {
  background: var(--fa-table-header-bg) !important;
  font-weight: 700;
  border-top: 2px solid var(--fa-border);
}

/* =========================================================
   Mobile Responsive Transformation (Table -> Card View)
   ========================================================= */
@media (max-width: 768px) {
  .fa-card-body { padding: 12px; }
  .fa-card-head { padding: 12px 16px; }
  
  .fa-table-responsive {
    display: block;
    width: 100%;
  }
  .fa-table-responsive thead {
    display: none; /* 모바일 헤더 숨김 */
  }
  .fa-table-responsive tbody {
    display: flex;
    flex-direction: column;
    gap: 12px;
  }
  .fa-table-responsive tr {
    display: block;
    background: var(--fa-card-bg) !important;
    border: 1px solid var(--fa-card-border);
    border-radius: 12px;
    padding: 12px 14px;
    box-shadow: 0 2px 8px rgba(0, 0, 0, 0.04);
  }
  .fa-table-responsive tfoot tr {
    display: block;
    background: var(--fa-table-header-bg) !important;
    border: 2px solid var(--fa-accent);
    border-radius: 12px;
    padding: 12px 14px;
    margin-top: 12px;
  }
  .fa-table-responsive td {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 6px 0;
    border-bottom: 1px dashed var(--fa-border);
    font-size: 0.88rem;
  }
  .fa-table-responsive td:last-child {
    border-bottom: none;
  }
  .fa-table-responsive td::before {
    content: attr(data-label);
    font-weight: 600;
    color: var(--fa-text-muted);
    font-size: 0.82rem;
    margin-right: 12px;
  }
  
  /* 모바일 종목명/계좌명 카드 최상단 강조 */
  .fa-table-responsive .fa-col-symbol,
  .fa-table-responsive .fa-col-account {
    font-size: 1rem;
    padding-bottom: 8px;
    margin-bottom: 4px;
    border-bottom: 1px solid var(--fa-border);
  }
}

/* =========================================================
   Interactive Account Tab Navigation
   ========================================================= */
.fa-tab-nav-wrapper {
  overflow-x: auto;
  -webkit-overflow-scrolling: touch;
  margin-bottom: 18px;
  padding-bottom: 4px;
}
.fa-tab-nav {
  display: flex;
  gap: 8px;
  min-width: max-content;
}
.fa-tab-btn {
  background: var(--fa-table-header-bg);
  border: 1px solid var(--fa-card-border);
  color: var(--fa-text-muted);
  padding: 8px 16px;
  border-radius: 9999px;
  font-size: 0.88rem;
  font-weight: 600;
  cursor: pointer;
  transition: all 0.2s ease;
  white-space: nowrap;
}
.fa-tab-btn:hover {
  background: var(--fa-border);
  color: var(--fa-text-main);
}
.fa-tab-btn.active {
  background: var(--fa-accent);
  color: #ffffff;
  border-color: var(--fa-accent);
  box-shadow: 0 2px 8px rgba(79, 70, 229, 0.3);
}

.fa-tab-pane {
  display: none;
  animation: faFadeIn 0.25s ease-in-out;
}
.fa-tab-pane.active {
  display: block;
}
@keyframes faFadeIn {
  from { opacity: 0; transform: translateY(4px); }
  to { opacity: 1; transform: translateY(0); }
}

/* Account Mini KPI Grid */
.fa-mini-kpi-grid {
  display: grid;
  grid-template-columns: repeat(6, minmax(0, 1fr));
  gap: 10px;
  margin-bottom: 18px;
}
@media (max-width: 1100px) { .fa-mini-kpi-grid { grid-template-columns: repeat(3, minmax(0, 1fr)); } }
@media (max-width: 640px) { .fa-mini-kpi-grid { grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 8px; } }

.fa-mini-kpi {
  background: var(--fa-kpi-bg);
  border: 1px solid var(--fa-border);
  border-radius: 10px;
  padding: 10px 12px;
}
.fa-mini-kpi-lbl { font-size: 0.76rem; color: var(--fa-text-muted); font-weight: 500; }
.fa-mini-kpi-val { font-size: 0.98rem; font-weight: 700; margin-top: 3px; }

/* Account Detail Transaction History Section */
.fa-acct-history-section {
  margin-top: 24px;
  padding-top: 18px;
  border-top: 1px dashed var(--fa-border);
}
.fa-acct-history-section .fa-history-list {
  display: grid;
  grid-template-columns: repeat(2, minmax(0, 1fr));
  gap: 10px;
}
@media (max-width: 768px) {
  .fa-acct-history-section .fa-history-list {
    grid-template-columns: 1fr;
  }
}

/* Account Split Grid (Chart + Table) */
.fa-account-split-grid {
  display: grid;
  grid-template-columns: 320px 1fr;
  gap: 20px;
  align-items: start;
  margin-bottom: 18px;
}
@media (max-width: 900px) {
  .fa-account-split-grid {
    grid-template-columns: 1fr;
    gap: 16px;
  }
}
.fa-account-chart-col {
  background: transparent;
}
.fa-account-chart-card {
  background: var(--fa-kpi-bg);
  border: 1px solid var(--fa-border);
  border-radius: 12px;
  padding: 10px;
}
.fa-account-table-col {
  background: transparent;
}

/* Stock Cards Grid (2-Column) */
.fa-stock-grid {
  display: grid;
  grid-template-columns: repeat(2, minmax(0, 1fr));
  gap: 12px;
}
@media (max-width: 700px) {
  .fa-stock-grid {
    grid-template-columns: 1fr;
    gap: 10px;
  }
}
.fa-stock-card {
  background: var(--fa-card-bg);
  border: 1px solid var(--fa-card-border);
  border-radius: 12px;
  padding: 12px 16px;
  box-shadow: 0 1px 3px rgba(0, 0, 0, 0.03);
  display: flex;
  flex-direction: column;
  gap: 8px;
  transition: transform 0.15s ease, box-shadow 0.15s ease;
}
.fa-stock-card:hover {
  transform: translateY(-2px);
  box-shadow: 0 4px 12px rgba(0, 0, 0, 0.06);
}
.fa-stock-card-head {
  display: flex;
  justify-content: space-between;
  align-items: center;
  border-bottom: 1px solid var(--fa-border);
  padding-bottom: 8px;
}
.fa-stock-card-title {
  font-size: 0.98rem;
  font-weight: 700;
  color: var(--fa-text-main);
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
}
.fa-stock-card-badges {
  display: flex;
  align-items: center;
  gap: 5px;
  flex-shrink: 0;
}
.fa-chip-weight {
  font-size: 0.72rem;
  font-weight: 600;
  color: var(--fa-text-muted);
  background: var(--fa-table-header-bg);
  padding: 2px 6px;
  border-radius: 4px;
}
.fa-stock-card-body {
  display: grid;
  grid-template-columns: repeat(2, 1fr);
  gap: 8px 12px;
}
.fa-stock-field {
  display: flex;
  flex-direction: column;
  gap: 2px;
}
.fa-stock-lbl {
  font-size: 0.74rem;
  color: var(--fa-text-muted);
}
.fa-stock-val {
  font-size: 0.88rem;
  font-weight: 600;
  font-variant-numeric: tabular-nums;
}

/* Rebalancing Guide */
.fa-rebal-box {
  background: var(--fa-kpi-bg);
  border: 1px solid var(--fa-border);
  border-radius: 12px;
  padding: 14px 16px;
  margin-top: 16px;
}
.fa-rebal-title {
  font-size: 0.92rem;
  font-weight: 700;
  margin-bottom: 10px;
  color: var(--fa-text-main);
}
.fa-rebal-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
  gap: 10px;
}
.fa-rebal-item {
  background: var(--fa-card-bg);
  border: 1px solid var(--fa-card-border);
  border-radius: 10px;
  padding: 10px 12px;
  display: flex;
  flex-direction: column;
  gap: 4px;
}
.fa-rebal-item.buy { border-left: 4px solid var(--fa-gain); }
.fa-rebal-item.sell { border-left: 4px solid var(--fa-loss); }
.fa-rebal-item.ok { border-left: 4px solid var(--fa-ok); }

.fa-rebal-tag {
  font-size: 0.72rem;
  font-weight: 700;
  text-transform: uppercase;
}
.fa-rebal-tag.buy { color: var(--fa-gain); }
.fa-rebal-tag.sell { color: var(--fa-loss); }
.fa-rebal-tag.ok { color: var(--fa-ok); }
.fa-rebal-name { font-size: 0.88rem; font-weight: 600; }
.fa-rebal-val { font-size: 0.82rem; font-weight: 700; color: var(--fa-text-muted); }

/* Trading History Modern Timeline & KPI Grid */
.fa-history-kpi-grid {
  display: grid;
  grid-template-columns: repeat(5, minmax(0, 1fr));
  gap: 10px;
  margin-bottom: 16px;
}
@media (max-width: 990px) {
  .fa-history-kpi-grid {
    grid-template-columns: repeat(3, minmax(0, 1fr));
  }
}
@media (max-width: 600px) {
  .fa-history-kpi-grid {
    grid-template-columns: repeat(2, minmax(0, 1fr));
    gap: 8px;
  }
}
.fa-history-list {
  display: grid;
  grid-template-columns: repeat(2, minmax(0, 1fr));
  gap: 12px;
}
@media (max-width: 768px) {
  .fa-history-list {
    grid-template-columns: 1fr;
    gap: 10px;
  }
}
.fa-history-card {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 12px 16px;
  background: var(--fa-kpi-bg);
  border: 1px solid var(--fa-card-border);
  border-radius: 12px;
  box-shadow: 0 1px 3px rgba(0, 0, 0, 0.03);
  transition: transform 0.15s ease, box-shadow 0.15s ease;
}
.fa-history-card:hover {
  transform: translateY(-2px);
  box-shadow: 0 4px 12px rgba(0, 0, 0, 0.06);
}
.fa-history-card-left {
  display: flex;
  flex-direction: column;
  gap: 4px;
  min-width: 0;
}
.fa-history-card-header {
  display: flex;
  align-items: center;
  gap: 6px;
}
.fa-history-account {
  font-size: 0.76rem;
  font-weight: 600;
  color: var(--fa-text-muted);
  background: var(--fa-card-bg);
  border: 1px solid var(--fa-border);
  padding: 1px 6px;
  border-radius: 4px;
}
.fa-history-date {
  font-size: 0.78rem;
  color: var(--fa-text-sub);
}
.fa-history-symbol {
  font-size: 0.95rem;
  font-weight: 700;
  color: var(--fa-text-main);
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
}
.fa-history-card-right {
  display: flex;
  flex-direction: column;
  align-items: flex-end;
  gap: 2px;
  flex-shrink: 0;
  margin-left: 14px;
}
.fa-history-amount {
  font-size: 1.02rem;
  font-weight: 700;
  font-variant-numeric: tabular-nums;
}
.fa-num-purple {
  color: var(--fa-purple) !important;
}
.fa-history-subdetail {
  font-size: 0.78rem;
  color: var(--fa-text-muted);
  white-space: nowrap;
}

.fa-empty-text { color: var(--fa-text-muted); font-size: 0.9rem; margin: 8px 0; }
</style>

<script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
<script>
document.addEventListener("DOMContentLoaded", function () {
  // 계좌 탭 전환 이벤트 리스너
  const tabBtns = document.querySelectorAll(".fa-tab-btn");
  tabBtns.forEach(btn => {
    btn.addEventListener("click", function () {
      const targetId = this.getAttribute("data-target");
      const container = this.closest(".fa-card-tabs");
      if (!container) return;

      // 버튼 활성화 토글
      container.querySelectorAll(".fa-tab-btn").forEach(b => b.classList.remove("active"));
      this.classList.add("active");

      // 패널 활성화 토글
      container.querySelectorAll(".fa-tab-pane").forEach(p => p.classList.remove("active"));
      const targetPane = document.getElementById(targetId);
      if (targetPane) {
        targetPane.classList.add("active");
        // 해당 패널 내부의 Plotly 차트 리사이즈
        const chartDiv = targetPane.querySelector(".plotly-graph-div");
        if (chartDiv && window.Plotly) {
          window.Plotly.Plots.resize(chartDiv);
        }
      }

      setTimeout(() => {
        window.dispatchEvent(new Event("resize"));
      }, 50);
    });
  });
});
</script>
"""
    return "<div class=\"fa-dashboard\">" + styles + "".join(blocks) + "</div>"


def _wrap_standalone_html(content_html: str, title: str) -> str:
    return "\n".join(
        [
            "<!doctype html>",
            "<html lang=\"ko\">",
            "<head>",
            "  <meta charset=\"utf-8\">",
            "  <meta name=\"viewport\" content=\"width=device-width, initial-scale=1.0, user-scalable=yes, maximum-scale=5.0\">",
            f"  <title>{html.escape(title)}</title>",
            "  <script src=\"https://cdn.plot.ly/plotly-2.35.2.min.js\"></script>",
            "  <style>",
            "    :root { color-scheme: light; }",
            "    body { margin: 0; background: #f8fafc; font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Noto Sans KR', sans-serif; }",
            "    .fa-standalone-wrap { max-width: 1200px; margin: 0 auto; padding: 20px 16px 40px; }",
            "  </style>",
            "</head>",
            "<body>",
            "  <div class=\"fa-standalone-wrap\">",
            content_html,
            "  </div>",
            "</body>",
            "</html>",
            "",
        ]
    )


def _remove_legacy_fragment(current_fragment_path: Path) -> None:
    if LEGACY_FRAGMENT_PATH == current_fragment_path:
        return
    if not LEGACY_FRAGMENT_PATH.exists():
        return
    LEGACY_FRAGMENT_PATH.unlink()
    print(f"Removed legacy fragment: {LEGACY_FRAGMENT_PATH}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build FA dashboard HTML outputs")
    parser.add_argument(
        "--index",
        type=Path,
        help="Deprecated. Kept for backward compatibility.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Standalone HTML output path (default: <static_dir>/latest_fa.html)",
    )
    parser.add_argument(
        "--fragment-output",
        type=Path,
        help="Fragment HTML path for Hugo shortcode (default: <root>/generated/fa/latest_fa_fragment.html)",
    )
    parser.add_argument(
        "--no-standalone",
        action="store_true",
        help="Only write fragment output and skip standalone HTML output",
    )
    parser.add_argument(
        "--title",
        default="FA 대시보드 - 자세히 보기",
        help="HTML title text",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    update_fa.update_titles_from_fa_yaml()
    static_dir = update_fa.PATHS.get("static_dir", ROOT_DIR / "content/fa")
    output_path = args.output or (static_dir / "latest_fa.html")
    fragment_path = args.fragment_output or DEFAULT_FRAGMENT_PATH

    records = update_fa.read_trading_records()
    if records.empty:
        raise ValueError("Trading records are empty.")
    data = _build_report_data(records)

    dashboard_fragment = _build_dashboard_fragment(data)
    fragment_path.parent.mkdir(parents=True, exist_ok=True)
    fragment_path.write_text(dashboard_fragment, encoding="utf-8")
    print(f"Dashboard fragment saved: {fragment_path}")
    _remove_legacy_fragment(fragment_path)

    if not args.no_standalone:
        standalone_html = _wrap_standalone_html(dashboard_fragment, args.title)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(standalone_html, encoding="utf-8")
        print(f"HTML saved: {output_path}")


if __name__ == "__main__":
    main()
