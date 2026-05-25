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

DEFAULT_FRAGMENT_PATH = ROOT_DIR / "generated" / "fa" / "latest_fa_fragment.html"
LEGACY_FRAGMENT_PATH = ROOT_DIR / "data" / "fa" / "latest_fa_fragment.html"


TABLE_HEADER_HEIGHT = 30
TABLE_ROW_HEIGHT = 28
TABLE_LINE_WIDTH = 1
TABLE_PADDING_HEIGHT = 8
FONT_FAMILY = "Roboto, sans-serif"
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
MARKET_KPI_CONFIG = [
    {"label": "S&P500", "ticker": "^GSPC", "decimals": 2},
    {"label": "나스닥100", "ticker": "^NDX", "decimals": 2},
    {"label": "SCHD", "ticker": "SCHD", "decimals": 2},
    {"label": "IEF", "ticker": "IEF", "decimals": 2},
    {"label": "코스피", "ticker": "^KS11", "decimals": 2},
    {"label": "코스닥", "ticker": "^KQ11", "decimals": 2},
]


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



def _build_font_colors(values_matrix: list, default_col_colors: list) -> list:
    """Plotly 테이블 셀별로 텍스트 색상을 계산한다 (+는 빨간색, -는 파란색)."""
    import pandas as pd
    import re
    color_matrix = []
    
    for c_idx, col_values in enumerate(values_matrix):
        default_color = default_col_colors[c_idx] if c_idx < len(default_col_colors) else default_col_colors[-1]
        col_colors = []
        for val in col_values:
            if pd.isna(val) and not isinstance(val, str):
                col_colors.append(default_color)
                continue
            s_val = str(val).strip()
            # 숫자로 구성되어 있고(문자 포함 x), + 또는 - 가 포함된 경우 색상 적용 (날짜 제외)
            has_letters = bool(re.search(r'[a-zA-Z가-힣]', s_val))
            is_date = bool(re.search(r'^\d{4}-\d{2}-\d{2}', s_val))
            
            if not has_letters and not is_date:
                if '+' in s_val:
                    col_colors.append('#b42318')
                elif '-' in s_val:
                    col_colors.append('#1d4ed8')
                else:
                    col_colors.append(default_color)
            else:
                col_colors.append(default_color)
        color_matrix.append(col_colors)
    return color_matrix

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
                mode="lines",
                name=label,
                line=dict(color=_palette_color(idx), width=2),
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
    fig.update_yaxes(tickformat=",.0f", rangemode="tozero")
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
            line=dict(color="#2c3e50", width=2),
            hovertemplate="%{x|%Y-%m}: %{y:,.0f}<extra>누적 투자금</extra>",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=account_df.index,
            y=total_valuation,
            mode="lines",
            name="누적 평가금",
            line=dict(color="#d63031", width=2),
            hovertemplate="%{x|%Y-%m}: %{y:,.0f}<extra>누적 평가금</extra>",
        )
    )
    fig.update_layout(
        height=340,
        margin=dict(l=28, r=12, t=12, b=26),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        showlegend=True,
        font=dict(family=FONT_FAMILY),
        paper_bgcolor=THEME_BG,
        plot_bgcolor=THEME_BG,
    )
    fig.update_yaxes(tickformat=",.0f", rangemode="tozero")
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
                    font=dict(color=_build_font_colors(values, [THEME_TEXT] * len(values)), size=13, family=FONT_FAMILY),
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
                    font=dict(color=_build_font_colors(values, [THEME_TEXT] * len(values)), size=12, family=FONT_FAMILY),
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
    
    # 각 인덱스(월별)마다 총합을 미리 계산합니다.
    totals = pivot_sorted.sum(axis=1)

    for idx, col in enumerate(pivot.columns):
        # 마지막 항목에서만 막대 상단에 총합 텍스트를 출력하도록 설정
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
                textfont=dict(size=11, color=THEME_TEXT),
                cliponaxis=False
            )
        )
    fig.update_layout(
        barmode="stack",
        height=chart_height,
        margin=dict(l=28, r=12, t=30, b=22),
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
    
    # 각 연도별로 총 배당금 합계 계산
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
                textfont=dict(size=11, color=THEME_TEXT),
                cliponaxis=False
            )
        )
    fig.update_layout(
        barmode="stack",
        height=300,
        margin=dict(l=28, r=12, t=30, b=22),
        showlegend=False,
        font=dict(family=FONT_FAMILY),
        paper_bgcolor=THEME_BG,
        plot_bgcolor=THEME_BG,
    )
    fig.update_yaxes(tickformat=",.0f", showgrid=True, gridcolor=THEME_GRID, zeroline=False)
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
                font=dict(color=_build_font_colors([table_df[col].tolist() for col in table_df.columns], ["white"] + [THEME_TEXT] * (len(table_df.columns) - 1)), size=12, family=FONT_FAMILY),
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
    # 임의의 드래그/핀치 줌 및 확대 조작을 강제로 차단하고 고정(Lock)시킵니다.
    fig.update_xaxes(fixedrange=True)
    fig.update_yaxes(fixedrange=True)
    return pio.to_html(
        fig,
        full_html=False,
        # Hugo canonifyURLs가 inline Plotly 번들 내부의 "/<a href=/" 정규식을
        # 절대 URL로 치환해 스크립트를 깨뜨릴 수 있어 CDN 로드 방식으로 고정합니다.
        include_plotlyjs="cdn" if include_js else False,
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

    valid_detail_accounts = []
    if not summary_df.empty and "평가금" in summary_df.columns:
        valid_detail_accounts = summary_df.loc[
            (summary_df["계좌"] != "합계") & (summary_df["평가금"] > 0),
            "계좌",
        ].astype(str).tolist()

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
        default="FA Snapshot",
        help="HTML title text",
    )
    return parser.parse_args()


def _as_float(value: object) -> Optional[float]:
    try:
        if value is None or pd.isna(value):
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _fmt_krw(value: Optional[float]) -> str:
    if value is None:
        return "-"
    return f"{value:,.0f}원"


def _fmt_pct(value: Optional[float]) -> str:
    if value is None:
        return "-"
    return f"{value:+.2f}%"


def _fmt_number(value: Optional[float], decimals: int = 2, suffix: str = "") -> str:
    if value is None:
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


import re

def _kpi_card(label: str, value: str, sub: str = "", state: str = "") -> str:
    state_class = f" {state}" if state else ""
    
    def _colorize(text: str) -> str:
        t = html.escape(text)
        t = re.sub(r'(\+[0-9.,]+%?p?)', r'<span class="fa-text-positive">\1</span>', t)
        t = re.sub(r'(-[0-9.,]+%?p?)', r'<span class="fa-text-negative">\1</span>', t)
        return t

    val_html = _colorize(value)
    sub_html = _colorize(sub)

    return (
        f"<div class=\"fa-kpi-card{state_class}\">"
        f"<div class=\"fa-kpi-label\">{html.escape(label)}</div>"
        f"<div class=\"fa-kpi-value\">{val_html}</div>"
        f"<div class=\"fa-kpi-sub\">{sub_html}</div>"
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

    cards = [
        _kpi_card("총 평가금", _fmt_krw(valuation), f"{month_label} 기준"),
        _kpi_card("총 투자금", _fmt_krw(invest), f"{month_label} 누적"),
        _kpi_card(
            "총 수익금",
            _fmt_krw(profit),
            "실현+평가",
            "positive" if (profit or 0) > 0 else "negative" if (profit or 0) < 0 else "",
        ),
        _kpi_card(
            "총 수익률",
            _fmt_pct(return_rate),
            "투자금 대비",
            "positive" if (return_rate or 0) > 0 else "negative" if (return_rate or 0) < 0 else "",
        ),
        _kpi_card("월 배당금", _fmt_krw(monthly_div), f"{month_label} 합계"),
        _kpi_card("USD/KRW", _fmt_number(fx, 2), fx_change_text, fx_state),
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


def _dashboard_card(title: str, body_html: str, extra_class: str = "") -> str:
    klass = f"fa-card {extra_class}".strip()
    return (
        f"<section class=\"{klass}\">"
        f"<header class=\"fa-card-head\"><h2>{html.escape(title)}</h2></header>"
        f"<div class=\"fa-card-body\">{body_html}</div>"
        "</section>"
    )


def _build_dashboard_fragment(data: ReportData) -> str:
    plotly_included = False

    def fig_html(fig: go.Figure) -> str:
        nonlocal plotly_included
        rendered = _render_figure_html(fig, include_js=not plotly_included)
        plotly_included = True
        return rendered

    assets_investment_fig = _build_assets_investment_trend(data.account_df, data.invest_series)
    assets_fig = _build_assets_trend(data.account_df)
    account_fig = _build_account_assets_table(data.summary_df)
    holdings_fig = _build_total_holdings_table(data.holdings_df)
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
    trading_summary, trading_lines = _build_trading_history(data.records, data.fx_series_month, data.month_end)

    blocks: List[str] = [
        "<section class=\"fa-hero\">"
        f"<div class=\"fa-hero-title\">{html.escape(data.month_end.strftime('%Y년 %m월 자산 대시보드'))}</div>"
        f"<div class=\"fa-hero-meta\">업데이트: {html.escape(data.month_end.strftime('%Y-%m-%d'))}</div>"
        "</section>",
        _build_kpi_row(data),
        _build_market_kpi_row(),
        _dashboard_card(update_fa.ACCOUNT_TITLES.get("title_assets_investment_trend", "누적 투자금 vs 평가금 추세"), fig_html(assets_investment_fig), extra_class="fa-card-wide"),
        _dashboard_card(update_fa.ACCOUNT_TITLES.get("title_assets_trend", "자산 추이"), fig_html(assets_fig), extra_class="fa-card-wide"),
        _dashboard_card(update_fa.ACCOUNT_TITLES.get("title_account_assets", "계좌 요약"), fig_html(account_fig)),
    ]

    if holdings_fig is not None:
        blocks.append(
            _dashboard_card(
                update_fa.ACCOUNT_TITLES.get("title_total_holdings", "보유 종목"),
                fig_html(holdings_fig),
                extra_class="fa-card-wide",
            )
        )

    if monthly_div_fig is not None:
        blocks.append(
            _dashboard_card(
                update_fa.ACCOUNT_TITLES.get("title_monthly_dividends", "월별 배당"),
                fig_html(monthly_div_fig),
                extra_class="fa-card-wide",
            ),
        )
    if yearly_div_fig is not None:
        blocks.append(
            _dashboard_card(
                update_fa.ACCOUNT_TITLES.get("title_yearly_dividends", "연별 배당"),
                fig_html(yearly_div_fig),
                extra_class="fa-card-wide",
            ),
        )

    if trading_summary or trading_lines:
        blocks.append(
            _dashboard_card(
                update_fa.ACCOUNT_TITLES.get("title_trading_history", "거래 내역"),
                f"<div class=\"section-text\">{_render_history_html(trading_summary, trading_lines)}</div>",
                extra_class="fa-card-wide",
            )
        )

    for account in data.valid_detail_accounts:
        fig = _build_account_detail(account, data.holdings_df)
        if fig is None:
            continue
        detail_title = update_fa.ACCOUNT_TITLES.get(
            f"title_{account}_detail", f"상세계좌: {update_fa.account_label(account)}"
        )
        blocks.append(
            _dashboard_card(
                detail_title,
                fig_html(fig),
                extra_class="fa-card-wide",
            )
        )

    styles = """
<style>
.fa-dashboard {
  --fa-card-bg: #ffffff;
  --fa-card-border: #d8dee8;
  --fa-text: #1f2d3d;
  --fa-muted: #6b7280;
  --fa-kpi-bg: #f7fafc;
  --fa-shadow: 0 1px 2px rgba(16, 24, 40, 0.08);
  color: var(--fa-text);
}
html.dark .fa-dashboard {
  --fa-card-bg: #171b22;
  --fa-card-border: #2a3342;
  --fa-text: #e5e7eb;
  --fa-muted: #9ca3af;
  --fa-kpi-bg: #12161d;
  --fa-shadow: none;
}
.fa-dashboard .plotly-graph-div .svg-container { overflow: visible !important; }
.fa-dashboard .plotly-graph-div .main-svg text {
  line-height: 1 !important;
}
.fa-dashboard .plotly-graph-div .table text {
  dominant-baseline: middle !important;
  alignment-baseline: middle !important;
}
.fa-dashboard .plotly-graph-div .table .cells text { fill: #1f2d3d !important; }
.fa-dashboard .plotly-graph-div .table .header text { fill: #ffffff !important; }
.fa-hero { margin: 6px 0 12px; }
.fa-hero-title { font-size: 1.4rem; font-weight: 700; }
.fa-hero-meta { margin-top: 4px; color: var(--fa-muted); font-size: 0.92rem; }
.fa-kpi-grid {
  display: grid;
  grid-template-columns: repeat(6, minmax(0, 1fr));
  gap: 10px;
  margin: 10px 0 14px;
}
@media (max-width: 1200px) { .fa-kpi-grid { grid-template-columns: repeat(3, minmax(0, 1fr)); } }
@media (max-width: 700px) { .fa-kpi-grid { grid-template-columns: repeat(2, minmax(0, 1fr)); } }
.fa-kpi-card {
  border: 1px solid var(--fa-card-border);
  border-radius: 10px;
  padding: 10px 12px;
  background: var(--fa-kpi-bg);
  box-shadow: var(--fa-shadow);
}
.fa-kpi-label { color: var(--fa-muted); font-size: 0.82rem; }
.fa-kpi-value { margin-top: 4px; font-weight: 700; font-size: 1.08rem; }
.fa-kpi-sub { margin-top: 2px; color: var(--fa-muted); font-size: 0.78rem; }
.fa-text-positive { color: #b42318; }
.fa-text-negative { color: #1d4ed8; }
html.dark .fa-text-positive { color: #f87171; }
html.dark .fa-text-negative { color: #60a5fa; }
.fa-grid { display: grid; gap: 12px; margin: 12px 0; }
.fa-grid-2 { grid-template-columns: repeat(2, minmax(0, 1fr)); }
@media (max-width: 1000px) { .fa-grid-2 { grid-template-columns: 1fr; } }
.fa-card {
  border: 1px solid var(--fa-card-border);
  border-radius: 12px;
  background: var(--fa-card-bg);
  box-shadow: var(--fa-shadow);
  overflow: hidden;
}
.fa-card-head { padding: 12px 14px; border-bottom: 1px solid var(--fa-card-border); }
.fa-card-head h2 { margin: 0; font-size: 1.06rem; }
.fa-card-body { padding: 10px 12px 12px; }
.fa-card-wide { margin: 12px 0; }
.section-text { margin: 0; font-size: 13px; line-height: 1.45; }
.history-summary { font-weight: 700; margin-bottom: 6px; }
.history-line { margin: 3px 0; color: var(--fa-text); }
.history-line.buy { color: #b42318; }
.history-line.sell { color: #1d4ed8; }
html.dark .history-line.buy { color: #f87171; }
html.dark .history-line.sell { color: #60a5fa; }
.fa-detail-list { display: grid; gap: 10px; }
.fa-detail-item {
  border: 1px solid var(--fa-card-border);
  border-radius: 10px;
  background: transparent;
}
.fa-detail-item > summary {
  list-style: none;
  cursor: pointer;
  padding: 10px 12px;
  font-weight: 600;
  border-bottom: 1px solid transparent;
}
.fa-detail-item[open] > summary { border-bottom-color: var(--fa-card-border); }
.fa-detail-item > summary::-webkit-details-marker { display: none; }
.fa-detail-chart { padding: 8px 10px 10px; }
</style>
"""
    return "<div class=\"fa-dashboard\">" + styles + "".join(blocks) + "</div>"


def _wrap_standalone_html(content_html: str, title: str) -> str:
    return "\n".join(
        [
            "<!doctype html>",
            "<html lang=\"ko\">",
            "<head>",
            "  <meta charset=\"utf-8\">",
            "  <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">",
            f"  <title>{html.escape(title)}</title>",
            "  <style>",
            "    :root { color-scheme: light dark; }",
            "    body { margin: 0; font-family: \"Roboto\", sans-serif; }",
            "    .fa-standalone-wrap { max-width: 900px; margin: 0 auto; padding: 18px 16px 32px; }",
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


def main() -> None:
    args = parse_args()
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
