from __future__ import annotations

import argparse
import html
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple
from urllib.parse import urlparse

import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots


ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from scripts import update_fa


TABLE_WIDTH = 560
TABLE_HEADER_HEIGHT = 30
TABLE_ROW_HEIGHT = 28
TABLE_LINE_WIDTH = 1
TABLE_PADDING_HEIGHT = 8
FONT_FAMILY = "NanumSquareRound, 'Nanum Square', 'NanumSquareRound', sans-serif"
CHART_COLORWAY = [
    "#4E79A7",
    "#59A14F",
    "#F28E2B",
    "#B07AA1",
    "#E15759",
    "#76B7B2",
    "#EDC948",
    "#9C755F",
]
THEME_BG = "#F7FAFC"
THEME_BG_ALT = "#ECF2F7"
THEME_BG_EMPH = "#DEE8F0"
THEME_TEXT = "#1F2D3D"
THEME_HEADER_BG = "#324A5F"
THEME_BORDER = "#CAD5E0"
THEME_GRID = "#D8E1EA"
COLOR_GAIN = "#C0392B"
COLOR_LOSS = "#2A6F97"
DETAIL_TABLE_COLUMNWIDTH = [1.3, 1, 1, 0.9, 0.8]
TABLE_HEADER_ALIGN = "center"
EXCHANGE_RATE_TABLE_ALIGN = "center"
ACCOUNT_ASSETS_TABLE_ALIGN = ["left", "right", "right", "right", "right", "right", "right"]
TOTAL_HOLDINGS_TABLE_ALIGN = ["left", "left", "right", "right", "right", "right", "right", "right", "right"]
DETAIL_TABLE_ALIGN = ["left", "right", "right", "right", "right"]


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
    yearly_returns_df: Optional[pd.DataFrame]
    valid_detail_accounts: Sequence[str]


def _extract_image_keys(markdown_text: str) -> List[str]:
    cleaned = re.sub(r"<!--.*?-->", "", markdown_text, flags=re.DOTALL)
    token_re = re.compile(r"<img\b[^>]*>|!\[[^\]]*\]\([^\)]+\)", re.IGNORECASE)
    img_tag_re = re.compile(r"<img\b[^>]*>", re.IGNORECASE)
    attr_re = re.compile(r"(\w+)\s*=\s*([\"'])(.*?)\2")
    md_img_re = re.compile(r"!\[(?P<alt>[^\]]*)\]\((?P<src>[^\s\)]+)\)")

    srcs: List[str] = []
    for match in token_re.finditer(cleaned):
        token = match.group(0)
        if img_tag_re.match(token):
            attrs = {m.group(1).lower(): m.group(3) for m in attr_re.finditer(token)}
            src = attrs.get("src")
            if src:
                srcs.append(src)
            continue
        md_match = md_img_re.match(token)
        if md_match:
            srcs.append(md_match.group("src"))

    keys: List[str] = []
    for src in srcs:
        name = Path(urlparse(src).path).name
        if not (name.startswith("latest_") and name.endswith(".webp")):
            continue
        keys.append(name[len("latest_") : -len(".webp")])
    return keys


def _table_height(row_count: int, *, min_height: int = 180) -> int:
    height = (
        TABLE_HEADER_HEIGHT
        + (row_count * (TABLE_ROW_HEIGHT + TABLE_LINE_WIDTH))
        + TABLE_PADDING_HEIGHT
    )
    return max(min_height, height)


def _build_exchange_rate_table(fx_series: pd.Series) -> go.Figure:
    series = fx_series.sort_index()
    if series.empty:
        raise ValueError("FX series is empty.")
    current_rate = float(series.iloc[-1])
    prev_rate = float(series.iloc[-2]) if len(series) > 1 else current_rate
    change = current_rate - prev_rate
    change_pct = (change / prev_rate * 100) if prev_rate else 0.0

    window_end = pd.to_datetime(series.index.max())
    window_start = window_end - pd.DateOffset(years=3)
    recent_series = series.loc[series.index >= window_start]
    avg_3y = float(recent_series.mean()) if not recent_series.empty else float(series.mean())

    headers = ["환율(원/USD)", "증감", "직전 3년 평균 환율"]
    values = [
        f"{current_rate:,.2f}",
        f"{change:+.2f} ({change_pct:+.2f}%)",
        f"{avg_3y:,.2f}",
    ]

    gain_color = COLOR_GAIN
    loss_color = COLOR_LOSS
    change_color = gain_color if change > 0 else loss_color if change < 0 else THEME_TEXT

    fig = go.Figure(
        data=[
            go.Table(
                header=dict(
                    values=headers,
                    fill_color=THEME_HEADER_BG,
                    font=dict(color="white", size=14, family=FONT_FAMILY),
                    align=TABLE_HEADER_ALIGN,
                    height=TABLE_HEADER_HEIGHT,
                    line_color=THEME_BORDER,
                    line_width=TABLE_LINE_WIDTH,
                ),
                cells=dict(
                    values=[[values[0]], [values[1]], [values[2]]],
                    fill_color=[[THEME_BG], [THEME_BG], [THEME_BG_ALT]],
                    font=dict(color=[THEME_TEXT, change_color, THEME_TEXT], size=14, family=FONT_FAMILY),
                    align=EXCHANGE_RATE_TABLE_ALIGN,
                    height=TABLE_ROW_HEIGHT,
                    line_color=THEME_BORDER,
                    line_width=TABLE_LINE_WIDTH,
                ),
            )
        ]
    )
    fig.update_layout(
        height=_table_height(1, min_height=120),
        width=TABLE_WIDTH,
        margin=dict(l=0, r=0, t=0, b=0),
        font=dict(family=FONT_FAMILY),
        paper_bgcolor=THEME_BG,
    )
    return fig


def _build_assets_trend(account_df: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    for idx, column in enumerate(account_df.columns):
        label = update_fa.account_label(column)
        fig.add_trace(
            go.Scatter(
                x=account_df.index,
                y=account_df[column],
                mode="lines+markers",
                name=label,
                line=dict(color=_palette_color(idx)),
                marker=dict(color=_palette_color(idx)),
                hovertemplate="%{x|%Y-%m}: %{y:,.0f}<extra>%{fullData.name}</extra>",
            )
        )
    fig.update_layout(
        height=340,
        margin=dict(l=28, r=12, t=12, b=26),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        showlegend=False,
        font=dict(family=FONT_FAMILY),
        paper_bgcolor=THEME_BG,
        plot_bgcolor=THEME_BG,
    )
    fig.update_yaxes(tickformat=",.0f")
    return fig


def _build_account_assets_table(summary_df: pd.DataFrame) -> go.Figure:
    display_df = update_fa.format_summary_table(summary_df)
    values = [display_df[col].tolist() for col in display_df.columns]
    aligns = ACCOUNT_ASSETS_TABLE_ALIGN
    row_colors = []
    for idx, account in enumerate(display_df["계좌"].tolist()):
        if account == "합계":
            row_colors.append(THEME_BG_EMPH)
        else:
            row_colors.append(THEME_BG if idx % 2 == 0 else THEME_BG_ALT)
    fill_colors = [row_colors] * len(display_df.columns)
    fig = go.Figure(
        data=[
            go.Table(
                header=dict(
                    values=list(display_df.columns),
                    fill_color=THEME_HEADER_BG,
                    font=dict(color="white", size=14, family=FONT_FAMILY),
                    align=TABLE_HEADER_ALIGN,
                    height=TABLE_HEADER_HEIGHT,
                    line_color=THEME_BORDER,
                    line_width=TABLE_LINE_WIDTH,
                ),
                cells=dict(
                    values=values,
                    fill_color=fill_colors,
                    font=dict(color=THEME_TEXT, size=13, family=FONT_FAMILY),
                    align=aligns,
                    height=TABLE_ROW_HEIGHT,
                    line_color=THEME_BORDER,
                    line_width=TABLE_LINE_WIDTH,
                ),
            )
        ]
    )
    fig.update_layout(
        height=_table_height(len(display_df), min_height=220),
        width=TABLE_WIDTH,
        margin=dict(l=0, r=0, t=0, b=0),
        font=dict(family=FONT_FAMILY),
        paper_bgcolor=THEME_BG,
    )
    return fig


def _format_total_holdings(holdings_df: pd.DataFrame) -> pd.DataFrame:
    filtered = holdings_df[holdings_df["계좌"] != "sema"].copy()
    if filtered.empty:
        return pd.DataFrame()
    filtered["계좌"] = filtered["계좌"].apply(update_fa.account_label)
    filtered = filtered.sort_values(["계좌", "평가금"], ascending=[True, False])

    def fmt_qty(val: float) -> str:
        return f"{val:,.2f}".rstrip("0").rstrip(".")

    def fmt_currency(val: Optional[float]) -> str:
        if val is None or pd.isna(val):
            return "-"
        return f"{val:,.0f}"

    def fmt_rate(val: Optional[float]) -> str:
        if val is None or pd.isna(val):
            return "-"
        sign = "+" if val > 0 else ""
        return f"{sign}{val * 100:.2f}%"

    return pd.DataFrame(
        {
            "계좌": filtered["계좌"],
            "종목": filtered["종목"],
            "수량": filtered["수량"].apply(fmt_qty),
            "평단가": filtered["평단가"].apply(fmt_currency),
            "금액": filtered["금액"].apply(fmt_currency),
            "현재가": filtered["현재가"].apply(fmt_currency),
            "수익금": filtered["수익금"].apply(fmt_currency),
            "수익률": filtered["수익률"].apply(fmt_rate),
            "등락률": filtered["등락률"].apply(fmt_rate),
        }
    )


def _build_total_holdings_table(holdings_df: pd.DataFrame) -> Optional[go.Figure]:
    display_df = _format_total_holdings(holdings_df)
    if display_df.empty:
        return None
    values = [display_df[col].tolist() for col in display_df.columns]
    aligns = TOTAL_HOLDINGS_TABLE_ALIGN
    row_colors = [THEME_BG if idx % 2 == 0 else THEME_BG_ALT for idx in range(len(display_df))]
    fill_colors = [row_colors] * len(display_df.columns)
    fig = go.Figure(
        data=[
            go.Table(
                columnwidth=[1, 1.5, 0.7, 0.9, 1.2, 0.9, 1.1, 0.9, 0.8],
                header=dict(
                    values=list(display_df.columns),
                    fill_color=THEME_HEADER_BG,
                    font=dict(color="white", size=13, family=FONT_FAMILY),
                    align=TABLE_HEADER_ALIGN,
                    height=TABLE_HEADER_HEIGHT,
                    line_color=THEME_BORDER,
                    line_width=TABLE_LINE_WIDTH,
                ),
                cells=dict(
                    values=values,
                    fill_color=fill_colors,
                    font=dict(color=THEME_TEXT, size=12, family=FONT_FAMILY),
                    align=aligns,
                    height=TABLE_ROW_HEIGHT,
                    line_color=THEME_BORDER,
                    line_width=TABLE_LINE_WIDTH,
                ),
            )
        ]
    )
    fig.update_layout(
        height=_table_height(len(display_df), min_height=260),
        width=TABLE_WIDTH,
        margin=dict(l=0, r=0, t=0, b=0),
        font=dict(family=FONT_FAMILY),
        paper_bgcolor=THEME_BG,
    )
    return fig


def _build_dividends_chart(pivot: pd.DataFrame) -> go.Figure:
    chart_height = 300
    pivot_sorted = pivot.sort_index()
    if len(pivot_sorted) > 12:
        pivot_sorted = pivot_sorted.tail(12)
    fig = go.Figure()
    for idx, col in enumerate(pivot.columns):
        fig.add_trace(
            go.Bar(
                x=pivot_sorted.index,
                y=pivot_sorted[col],
                name=col,
                marker=dict(color=_palette_color(idx)),
            )
        )
    fig.update_layout(
        barmode="stack",
        height=chart_height,
        margin=dict(l=28, r=12, t=12, b=22),
        showlegend=False,
        font=dict(family=FONT_FAMILY),
        paper_bgcolor=THEME_BG,
        plot_bgcolor=THEME_BG,
    )
    fig.update_yaxes(
        tickformat=",.0f",
        dtick=500_000,
        showgrid=True,
        gridcolor=THEME_GRID,
        zeroline=False,
    )
    if not pivot_sorted.empty:
        min_x = pd.to_datetime(pivot_sorted.index.min()) - pd.Timedelta(days=15)
        max_x = pd.to_datetime(pivot_sorted.index.max()) + pd.Timedelta(days=15)
        fig.update_xaxes(tickformat="%y%m", range=[min_x, max_x])
    else:
        fig.update_xaxes(tickformat="%y%m")
    return fig


def _build_yearly_dividends_chart(pivot: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    pivot_sorted = pivot.sort_index()
    years = pd.to_datetime(pivot_sorted.index).strftime("%Y").tolist()
    for idx, col in enumerate(pivot_sorted.columns):
        fig.add_trace(
            go.Bar(
                x=years,
                y=pivot_sorted[col],
                name=col,
                marker=dict(color=_palette_color(idx)),
            )
        )
    fig.update_layout(
        barmode="stack",
        height=300,
        margin=dict(l=28, r=12, t=12, b=22),
        showlegend=False,
        font=dict(family=FONT_FAMILY),
        paper_bgcolor=THEME_BG,
        plot_bgcolor=THEME_BG,
    )
    fig.update_yaxes(tickformat=",.0f", showgrid=True, gridcolor=THEME_GRID, zeroline=False)
    return fig


def _build_yearly_return_chart(returns_df: pd.DataFrame, column: str) -> go.Figure:
    data = returns_df.dropna(subset=[column]).copy()
    years = data["연도"].astype(str).tolist()
    values = (data[column] * 100).tolist()
    colors = [COLOR_GAIN if val >= 0 else COLOR_LOSS for val in values]

    fig = go.Figure(
        data=[
            go.Bar(
                x=years,
                y=values,
                marker=dict(color=colors),
            )
        ]
    )
    fig.update_layout(
        height=270,
        margin=dict(l=28, r=12, t=12, b=22),
        showlegend=False,
        font=dict(family=FONT_FAMILY),
        paper_bgcolor=THEME_BG,
        plot_bgcolor=THEME_BG,
    )
    fig.update_yaxes(tickformat=".1f", ticksuffix="%", showgrid=True, gridcolor=THEME_GRID, zeroline=True)
    return fig


def _build_account_detail(account: str, holdings_df: pd.DataFrame) -> Optional[go.Figure]:
    account_holdings = holdings_df[holdings_df["계좌"] == account].copy()
    if account_holdings.empty:
        return None
    account_holdings = account_holdings.sort_values("평가금", ascending=False)

    def fmt_currency(val: Optional[float]) -> str:
        if val is None or pd.isna(val):
            return "-"
        return f"{val:,.0f}"

    def fmt_rate(val: Optional[float]) -> str:
        if val is None or pd.isna(val):
            return "-"
        sign = "+" if val > 0 else ""
        return f"{sign}{val * 100:.2f}%"

    table_df = pd.DataFrame(
        {
            "종목": account_holdings["종목"],
            "매수금": account_holdings["매수금"].apply(fmt_currency),
            "평가금": account_holdings["평가금"].apply(fmt_currency),
            "수익금": account_holdings["수익금"].apply(fmt_currency),
            "수익률": account_holdings["수익률"].apply(fmt_rate),
        }
    )
    row_colors = [_palette_color(idx) for idx in range(len(table_df))]
    fill_colors = [row_colors] + [[THEME_BG] * len(table_df) for _ in range(len(table_df.columns) - 1)]
    fig = make_subplots(
        rows=1,
        cols=2,
        specs=[[{"type": "domain"}, {"type": "table"}]],
        column_widths=[0.3, 0.7],
    )
    fig.add_trace(
        go.Pie(
            labels=account_holdings["종목"],
            values=account_holdings["평가금"],
            textinfo="percent",
            hole=0.35,
            showlegend=False,
            marker=dict(colors=row_colors),
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Table(
            columnwidth=DETAIL_TABLE_COLUMNWIDTH,
            header=dict(
                values=list(table_df.columns),
                fill_color=THEME_HEADER_BG,
                font=dict(color="white", size=13, family=FONT_FAMILY),
                align=TABLE_HEADER_ALIGN,
                height=TABLE_HEADER_HEIGHT,
                line_color=THEME_BORDER,
                line_width=TABLE_LINE_WIDTH,
            ),
            cells=dict(
                values=[table_df[col].tolist() for col in table_df.columns],
                fill_color=fill_colors,
                font=dict(color=["white"] + [THEME_TEXT] * (len(table_df.columns) - 1), size=12, family=FONT_FAMILY),
                align=DETAIL_TABLE_ALIGN,
                height=TABLE_ROW_HEIGHT,
                line_color=THEME_BORDER,
                line_width=TABLE_LINE_WIDTH,
            ),
        ),
        row=1,
        col=2,
    )
    fig.update_layout(
        height=_table_height(len(table_df), min_height=260),
        margin=dict(l=10, r=10, t=10, b=10),
        paper_bgcolor=THEME_BG,
        showlegend=False,
        font=dict(family=FONT_FAMILY),
    )
    return fig


def _build_trading_history(records: pd.DataFrame,
                           fx_series: pd.Series,
                           month_end: pd.Timestamp) -> Tuple[str, List[Tuple[str, str]]]:
    period = month_end.to_period("M")
    start = period.start_time
    end = period.end_time
    month_records = records[(records["일자"] >= start) & (records["일자"] <= end)].copy()
    if month_records.empty:
        return "", []

    buy_total = sell_total = invest_total = div_total = 0.0
    lines: List[Tuple[str, str]] = []

    def fmt_currency(val: float) -> str:
        return f"{val:,.0f}"

    month_records = month_records.sort_values("일자", ascending=False)
    for _, row in month_records.iterrows():
        date = pd.Timestamp(row["일자"])
        date_str = f"{date:%Y년 %m월 %d일}"
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
                lines.append((
                    "buy",
                    f"{date_str} - (매수) {account}: {symbol} {fmt_currency(trade_amt)}원 매수 (단가 {fmt_currency(unit_price)}원, {abs(qty):g}주)",
                ))
            else:
                sell_total += abs(trade_amt)
                lines.append((
                    "sell",
                    f"{date_str} - (매도) {account}: {symbol} {fmt_currency(abs(trade_amt))}원 매도 (단가 {fmt_currency(unit_price)}원, {abs(qty):g}주)",
                ))
        if has_dividend:
            div_amt = update_fa.convert_to_krw(acct_code, float(dividend), date, fx_series)
            div_total += div_amt
            native_str = "" if acct_code not in update_fa.USD_ACCOUNTS else f" ({dividend}달러)"
            lines.append((
                "div",
                f"{date_str} - (배당금) {account}: {symbol} 배당 {fmt_currency(div_amt)}원 수령{native_str}",
            ))
        if has_invest:
            invest_amt = float(str(invest).replace(",", "")) if invest else 0.0
            if acct_code in update_fa.USD_ACCOUNTS:
                invest_amt = update_fa.convert_to_krw(acct_code, invest_amt, date, fx_series)
            invest_total += invest_amt
            lines.append((
                "invest",
                f"{date_str} - (투자금) {account}: 투자금 {fmt_currency(invest_amt)}원 증액",
            ))

    summary = (
        f"{period.year}년 {period.month:02d}월 투자금: {fmt_currency(invest_total)}원, "
        f"매수: {fmt_currency(buy_total)}원, 매도: {fmt_currency(sell_total)}원, "
        f"배당금: {fmt_currency(div_total)}원"
    )
    return summary, lines


def _render_history_html(summary: str, lines: List[Tuple[str, str]]) -> str:
    if not summary and not lines:
        return "<div class=\"history-empty\">No trading history.</div>"
    parts = [f"<div class=\"history-summary\">{html.escape(summary)}</div>"]
    for kind, text in lines:
        parts.append(f"<div class=\"history-line {kind}\">{html.escape(text)}</div>")
    return "\n".join(parts)


def _render_figure_html(fig: go.Figure, *, include_js: bool) -> str:
    return pio.to_html(
        fig,
        full_html=False,
        include_plotlyjs="inline" if include_js else False,
        config={
            "displayModeBar": False,
            "responsive": True,
            "scrollZoom": False,
            "doubleClick": False,
        },
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

    try:
        yearly_returns_df = update_fa.build_yearly_returns(records_upto, fx_series_month, month_end)
    except ValueError:
        yearly_returns_df = None

    valid_detail_accounts = []
    if not summary_df.empty and "평가금" in summary_df.columns:
        valid_detail_accounts = summary_df.loc[
            (summary_df["계좌"] != "합계") & (summary_df["평가금"] > 0),
            "계좌",
        ].astype(str).tolist()

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
        yearly_returns_df=yearly_returns_df,
        valid_detail_accounts=valid_detail_accounts,
    )


def _render_title(key: str) -> str:
    title = update_fa.ACCOUNT_TITLES.get(key, key)
    return f"<h2 class=\"section-title\">{html.escape(title)}</h2>"


def _default_keys() -> List[str]:
    return [
        "title_exchange_rate",
        "exchange_rate",
        "title_assets_trend",
        "assets_trend",
        "title_account_assets",
        "account_assets",
        "title_total_holdings",
        "total_holdings",
        "title_trading_history",
        "trading_history",
        "title_monthly_dividends",
        "monthly_dividends",
        "title_yearly_dividends",
        "yearly_dividends",
        "title_yearly_return_investment",
        "yearly_return_investment",
        "title_yearly_return_valuation",
        "yearly_return_valuation",
        "title_usa_detail",
        "usa_detail",
        "title_sema_detail",
        "sema_detail",
        "title_irp_detail",
        "irp_detail",
        "title_psf1_detail",
        "psf1_detail",
        "title_psf2_detail",
        "psf2_detail",
        "title_isa2_detail",
        "isa2_detail",
    ]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a single Plotly HTML report")
    parser.add_argument(
        "--index",
        type=Path,
        help="Path to fa index.md (default: <static_dir>/index.md)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Output HTML path (default: <static_dir>/latest_fa.html)",
    )
    parser.add_argument(
        "--title",
        default="FA Snapshot",
        help="HTML title text",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    static_dir = update_fa.PATHS.get("static_dir", ROOT_DIR / "content/fa")
    index_path = args.index or (static_dir / "index.md")
    output_path = args.output or (static_dir / "latest_fa.html")

    records = update_fa.read_trading_records()
    if records.empty:
        raise ValueError("Trading records are empty.")
    data = _build_report_data(records)

    keys = []
    if index_path.exists():
        keys = _extract_image_keys(index_path.read_text(encoding="utf-8"))
    if not keys:
        keys = _default_keys()

    sections: List[str] = []
    plotly_included = False

    for key in keys:
        if key.startswith("title_"):
            sections.append(_render_title(key))
            continue

        fig: Optional[go.Figure] = None
        custom_html: Optional[str] = None

        if key == "exchange_rate":
            fig = _build_exchange_rate_table(data.fx_series_full)
        elif key == "assets_trend":
            fig = _build_assets_trend(data.account_df)
        elif key == "account_assets":
            fig = _build_account_assets_table(data.summary_df)
        elif key == "total_holdings":
            fig = _build_total_holdings_table(data.holdings_df)
        elif key == "monthly_dividends":
            if data.dividends_pivot is not None and not data.dividends_pivot.empty:
                fig = _build_dividends_chart(data.dividends_pivot)
        elif key == "yearly_dividends":
            if data.yearly_dividends_pivot is not None and not data.yearly_dividends_pivot.empty:
                fig = _build_yearly_dividends_chart(data.yearly_dividends_pivot)
        elif key == "yearly_return_investment":
            if data.yearly_returns_df is not None and not data.yearly_returns_df.empty:
                fig = _build_yearly_return_chart(data.yearly_returns_df, "투자금기준수익률")
        elif key == "yearly_return_valuation":
            if data.yearly_returns_df is not None and not data.yearly_returns_df.empty:
                fig = _build_yearly_return_chart(data.yearly_returns_df, "평가금기준수익률")
        elif key == "trading_history":
            summary, lines = _build_trading_history(data.records, data.fx_series_month, data.month_end)
            custom_html = _render_history_html(summary, lines)
        elif key.endswith("_detail"):
            account = key.replace("_detail", "")
            if account in data.valid_detail_accounts:
                fig = _build_account_detail(account, data.holdings_df)

        if fig is not None:
            is_table_section = key in {"exchange_rate", "account_assets", "total_holdings"}
            section_class = "section-chart table-chart" if is_table_section else "section-chart"
            sections.append(
                f"<div class=\"{section_class}\">{_render_figure_html(fig, include_js=not plotly_included)}</div>"
            )
            plotly_included = True
        elif custom_html:
            sections.append(f"<div class=\"section-text\">{custom_html}</div>")

    html_output = [
        "<!doctype html>",
        "<html lang=\"ko\">",
        "<head>",
        "  <meta charset=\"utf-8\">",
        "  <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">",
        f"  <title>{html.escape(args.title)}</title>",
        "  <style>",
        "    :root { color-scheme: light; }",
        "    @font-face {",
        "      font-family: \"NanumSquareRound\";",
        "      src: url(\"/fonts/NanumSquareRoundR.ttf\") format(\"truetype\");",
        "      font-weight: 400;",
        "      font-style: normal;",
        "      font-display: swap;",
        "    }",
        "    @font-face {",
        "      font-family: \"NanumSquareRound\";",
        "      src: url(\"/fonts/NanumSquareRoundEB.ttf\") format(\"truetype\");",
        "      font-weight: 700;",
        "      font-style: normal;",
        "      font-display: swap;",
        "    }",
        f"    body {{ margin: 0; background: {THEME_BG}; font-family: \"NanumSquareRound\", \"Nanum Square\", sans-serif; color: {THEME_TEXT}; }}",
        "    .container { max-width: 560px; margin: 0 auto; padding: 8px 10px 24px; }",
        "    .section-title { font-size: 17px; font-weight: 700; margin: 14px 0 8px; }",
        "    .section-chart { margin: 4px 0 12px; display: flex; justify-content: flex-start; align-items: flex-start; }",
        "    .table-chart { display: flex; justify-content: flex-start; }",
        "    .section-chart .plotly-graph-div { margin: 0 !important; }",
        "    .plotly-graph-div { overflow: hidden !important; }",
        "    .plotly-graph-div .svg-container { overflow: hidden !important; }",
        "    .plotly-graph-div .table text { dominant-baseline: middle; alignment-baseline: central; }",
        "    .section-text { margin: 4px 0 12px; font-size: 13px; line-height: 1.4; }",
        "    .history-summary { font-weight: 700; margin-bottom: 6px; }",
        "    .history-line { margin: 3px 0; }",
        f"    .history-line.buy {{ color: {COLOR_GAIN}; }}",
        f"    .history-line.sell {{ color: {COLOR_LOSS}; }}",
        "  </style>",
        "</head>",
        "<body>",
        "  <div class=\"container\">",
        "\n".join(sections),
        "  </div>",
        "  <script>",
        "    (function () {",
        "      function docHeight() {",
        "        var body = document.body;",
        "        var html = document.documentElement;",
        "        return Math.max(",
        "          body ? body.scrollHeight : 0,",
        "          body ? body.offsetHeight : 0,",
        "          html ? html.clientHeight : 0,",
        "          html ? html.scrollHeight : 0,",
        "          html ? html.offsetHeight : 0",
        "        );",
        "      }",
        "      function resizeIframe() {",
        "        var frame = window.frameElement;",
        "        if (!frame) return;",
        "        frame.style.overflow = \"hidden\";",
        "        frame.setAttribute(\"scrolling\", \"no\");",
        "        frame.style.height = String(docHeight() + 16) + \"px\";",
        "      }",
        "      window.addEventListener(\"load\", resizeIframe);",
        "      window.addEventListener(\"resize\", resizeIframe);",
        "      if (typeof ResizeObserver !== \"undefined\") {",
        "        var observer = new ResizeObserver(function () { resizeIframe(); });",
        "        observer.observe(document.body);",
        "      }",
        "      setTimeout(resizeIframe, 50);",
        "      setTimeout(resizeIframe, 300);",
        "      setTimeout(resizeIframe, 1000);",
        "    })();",
        "  </script>",
        "</body>",
        "</html>",
        "",
    ]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(html_output), encoding="utf-8")
    print(f"HTML saved: {output_path}")


if __name__ == "__main__":
    main()
