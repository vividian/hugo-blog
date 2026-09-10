from __future__ import annotations

import argparse
import html
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple
from urllib.parse import urlparse

import math
from datetime import datetime
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

APP_VERSION = "v2.7.66"

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
    return f"{value:,.0f}"


def _fmt_pct(value: Optional[float]) -> str:
    if value is None or pd.isna(value):
        return "-"
    return f"{value:+.2f}"


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
        pct_text = f" ({pct:+.2f})"
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


def _kpi_card(label: str, value: str, sub: str = "", state: str = "", sub_state: str = "", sub_onclick: str = "", sub_title: str = "") -> str:
    state_class = f" {state}" if state else ""
    actual_sub_state = sub_state if sub_state else state
    sub_cls = f" fa-num-{actual_sub_state}" if actual_sub_state in ("positive", "negative") else ""
    tooltip_title = sub_title or "클릭하여 종목별 변동 상세 보기"
    onclick_attr = f" onclick=\"{sub_onclick}\" style=\"cursor:pointer; text-decoration:underline; text-underline-offset:3px;\" title=\"{html.escape(tooltip_title)}\"" if sub_onclick else ""
    return (
        f"<div class=\"fa-kpi-card{state_class}\">"
        f"<div class=\"fa-kpi-label\">{html.escape(label)}</div>"
        f"<div class=\"fa-kpi-value\">{value}</div>"
        f"<div class=\"fa-kpi-sub{sub_cls}\"{onclick_attr}>{html.escape(sub)}</div>"
        "</div>"
    )


def _build_eval_change_modal(data: ReportData, cur_eval_val: float, prev_eval_val: float, eval_month_change: float) -> str:
    """총 평가금의 최근 12개월 전월대비 변동 추세를 보여주는 Plotly 막대 그래프 팝업 모달"""
    monthly_data_list = []

    if data.account_df is not None and not data.account_df.empty:
        total_eval_s = data.account_df.sum(axis=1)
        monthly_eval = total_eval_s.groupby(total_eval_s.index.to_period("M")).last()

        # 최근 12개월 period 추출
        periods = monthly_eval.index[-12:]
        for p in periods:
            c_val = float(monthly_eval.loc[p])
            p_prev = p - 1
            if p_prev in monthly_eval.index:
                prev_v = float(monthly_eval.loc[p_prev])
                diff_v = c_val - prev_v
            else:
                prev_v = c_val
                diff_v = 0.0

            rate_v = (diff_v / prev_v * 100.0) if prev_v > 0 else 0.0
            monthly_data_list.append({
                "period": p,
                "label": p.strftime("%Y.%m"),
                "short_label": p.strftime("%y.%m"),
                "eval": c_val,
                "diff": diff_v,
                "rate": rate_v
            })

    # 최신 월 수치가 실시간 KPI 수치(cur_eval_val, eval_month_change)와 일치하도록 보정
    if monthly_data_list:
        monthly_data_list[-1]["eval"] = cur_eval_val
        monthly_data_list[-1]["diff"] = eval_month_change
        if prev_eval_val > 0:
            monthly_data_list[-1]["rate"] = (eval_month_change / prev_eval_val) * 100.0

    if not monthly_data_list:
        return ""

    diffs = [m["diff"] for m in monthly_data_list]
    total_12m_diff = sum(diffs)
    best_item = max(monthly_data_list, key=lambda x: x["diff"])
    worst_item = min(monthly_data_list, key=lambda x: x["diff"])

    x_labels = [m["label"] for m in monthly_data_list]
    y_values = [m["diff"] for m in monthly_data_list]
    bar_colors = ["#e53e3e" if v >= 0 else "#3182ce" for v in y_values]

    def _fmt_bar_krw(v: float) -> str:
        sign = "+" if v > 0 else ""
        if abs(v) >= 100000000:
            return f"{sign}{v/100000000:.1f}억"
        elif abs(v) >= 10000:
            return f"{sign}{v/10000:,.0f}만"
        else:
            return f"{sign}{v:,.0f}"

    bar_texts = [_fmt_bar_krw(v) for v in y_values]

    custom_data = [
        [
            f"{m['diff']:+,.0f}원",
            f"{m['rate']:+.2f}%",
            f"{m['eval']:,.0f}원"
        ]
        for m in monthly_data_list
    ]

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=x_labels,
            y=y_values,
            text=bar_texts,
            textposition="outside",
            textfont=dict(family=FONT_FAMILY, size=11, color="#334155"),
            marker=dict(color=bar_colors, line=dict(width=0)),
            customdata=custom_data,
            hovertemplate="<b>%{x}</b><br>전월대비: <b>%{customdata[0]}</b> (%{customdata[1]})<br>총 평가금: %{customdata[2]}<extra></extra>",
            cliponaxis=False,
        )
    )

    max_abs_y = max([abs(v) for v in y_values]) if y_values else 1000000
    y_pad = max_abs_y * 0.22
    y_min = min(y_values) - y_pad
    y_max = max(y_values) + y_pad

    fig.update_layout(
        height=320,
        margin=dict(l=10, r=10, t=35, b=20),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        showlegend=False,
        font=dict(family=FONT_FAMILY, size=12),
        bargap=0.35,
    )
    fig.update_xaxes(
        tickfont=dict(family=FONT_FAMILY, size=11, color="#64748b"),
        showgrid=False,
        fixedrange=True,
    )
    fig.update_yaxes(
        range=[y_min, y_max],
        showgrid=True,
        gridcolor="#f1f5f9",
        zeroline=True,
        zerolinecolor="#94a3b8",
        zerolinewidth=1.5,
        tickfont=dict(family=FONT_FAMILY, size=11, color="#94a3b8"),
        fixedrange=True,
    )

    chart_html = fig.to_html(include_plotlyjs=False, full_html=False, config={"displayModeBar": False, "responsive": True})

    table_rows = []
    for m in reversed(monthly_data_list):
        net_cls = "fa-num-positive" if m["diff"] > 0 else "fa-num-negative" if m["diff"] < 0 else ""
        diff_str = f"{m['diff']:+,.0f}원" if m["diff"] != 0 else "0원"
        rate_str = f"{m['rate']:+.2f}%" if m["rate"] != 0 else "0.00%"
        table_rows.append(
            f"<tr>"
            f"  <td class='fa-num fa-font-bold'>{m['label']}</td>"
            f"  <td class='text-right fa-num'>{m['eval']:,.0f}원</td>"
            f"  <td class='text-right fa-num fa-font-bold {net_cls}'>{diff_str}</td>"
            f"  <td class='text-right fa-num {net_cls}'>{rate_str}</td>"
            f"</tr>"
        )

    net_12m_cls = "fa-num-positive" if total_12m_diff > 0 else "fa-num-negative" if total_12m_diff < 0 else ""
    cur_cls = "fa-num-positive" if eval_month_change > 0 else "fa-num-negative" if eval_month_change < 0 else ""

    return f"""
<div id="fa-eval-change-modal" class="fa-modal-overlay" onclick="if(event.target===this)closeEvalChangeModal()">
  <div class="fa-modal-card" style="max-width:780px; width:95%;">
    <div class="fa-modal-header">
      <h3 class="fa-modal-title">📈 최근 12개월 전월대비 자산 평가금 변동 추세</h3>
      <button type="button" class="fa-modal-close" onclick="closeEvalChangeModal()" aria-label="닫기">✕</button>
    </div>
    <div class="fa-modal-body">
      <div class="fa-modal-summary-grid" style="grid-template-columns: repeat(auto-fit, minmax(130px, 1fr));">
        <div class="fa-modal-stat-box">
          <div class="fa-modal-stat-lbl">당월 전월대비 변동액</div>
          <div class="fa-modal-stat-val {cur_cls}">{eval_month_change:+,.0f}원</div>
        </div>
        <div class="fa-modal-stat-box">
          <div class="fa-modal-stat-lbl">최근 12개월 누적 순변동</div>
          <div class="fa-modal-stat-val {net_12m_cls}">{total_12m_diff:+,.0f}원</div>
        </div>
        <div class="fa-modal-stat-box">
          <div class="fa-modal-stat-lbl">최대 상승 월</div>
          <div class="fa-modal-stat-val fa-num-positive">{best_item['label']} ({_fmt_bar_krw(best_item['diff'])})</div>
        </div>
        <div class="fa-modal-stat-box">
          <div class="fa-modal-stat-lbl">최대 하락 월</div>
          <div class="fa-modal-stat-val fa-num-negative">{worst_item['label']} ({_fmt_bar_krw(worst_item['diff'])})</div>
        </div>
      </div>

      <div style="background:var(--fa-card-bg); border:1px solid var(--fa-card-border); border-radius:12px; padding:12px 8px 4px; margin:16px 0;">
        <div style="font-size:0.85rem; font-weight:700; color:var(--fa-text-main); margin:0 8px 4px;">📊 최근 12개월 월별 전월대비 손익 막대 그래프</div>
        {chart_html}
      </div>

      <div class="fa-table-wrapper" style="max-height:240px; overflow-y:auto;">
        <table class="fa-table fa-table-modal-detail">
          <thead>
            <tr>
              <th>기준월</th>
              <th class="text-right">총 평가금</th>
              <th class="text-right">전월대비 변동액</th>
              <th class="text-right">전월대비 증감률</th>
            </tr>
          </thead>
          <tbody>
            {''.join(table_rows)}
          </tbody>
        </table>
      </div>
    </div>
  </div>
</div>
"""


def _build_day_change_modal(data: ReportData, eval_day_change: float) -> str:
    """총 평가금/수익금의 전일대비 증감액을 종목별로 상세히 분해하여 보여주는 팝업 모달"""
    items = []
    total_gain = 0.0
    total_loss = 0.0
    gain_count = 0
    loss_count = 0

    if not data.holdings_df.empty:
        for _, hrow in data.holdings_df.iterrows():
            sym = str(hrow.get("종목", "")).strip()
            acct = str(hrow.get("계좌", "")).strip()
            acct_label = update_fa.account_label(acct)
            qty = _as_float(hrow.get("수량")) or 0.0
            cur_price = _as_float(hrow.get("현재가")) or 0.0
            r = _as_float(hrow.get("등락률"))
            val = _as_float(hrow.get("평가금")) or 0.0

            change_amt = 0.0
            if r is not None and (1.0 + r) != 0:
                change_amt = (val * r / (1.0 + r))

            prev_price = (cur_price / (1.0 + r)) if (r is not None and (1.0 + r) != 0) else cur_price

            if change_amt > 0:
                total_gain += change_amt
                gain_count += 1
            elif change_amt < 0:
                total_loss += change_amt
                loss_count += 1

            items.append({
                "account": acct_label,
                "symbol": sym,
                "qty": qty,
                "cur_price": cur_price,
                "prev_price": prev_price,
                "rate": r,
                "change_amt": change_amt,
            })

    # 변동액 오름차순(손실이 큰 순서대로 먼저 표시)
    items.sort(key=lambda x: x["change_amt"])

    table_rows = []
    for item in items:
        amt = item["change_amt"]
        r = item["rate"]
        amt_cls = "fa-num-positive" if amt > 0 else "fa-num-negative" if amt < 0 else ""
        badge_cls = "fa-badge-positive" if (r or 0) > 0 else "fa-badge-negative" if (r or 0) < 0 else "fa-badge-neutral"
        rate_str = f"{r * 100:+.2f}%" if r is not None else "-"
        amt_str = f"{amt:+,.0f}원" if amt != 0 else "0원"

        table_rows.append(
            f"<tr>"
            f"  <td><span class='fa-chip-account'>{html.escape(item['account'])}</span></td>"
            f"  <td><strong>{html.escape(item['symbol'])}</strong></td>"
            f"  <td class='text-right fa-num'>{item['qty']:,.0f}</td>"
            f"  <td class='text-right fa-num'>{item['cur_price']:,.0f}</td>"
            f"  <td class='text-right fa-num'><span class='fa-badge {badge_cls}'>{rate_str}</span></td>"
            f"  <td class='text-right fa-num {amt_cls} fa-font-bold'>{amt_str}</td>"
            f"</tr>"
        )

    net_cls = "fa-num-positive" if eval_day_change > 0 else "fa-num-negative" if eval_day_change < 0 else ""

    return f"""
<div id="fa-day-change-modal" class="fa-modal-overlay" onclick="if(event.target===this)closeDayChangeModal()">
  <div class="fa-modal-card">
    <div class="fa-modal-header">
      <h3 class="fa-modal-title">📊 전일대비 종목별 손익 변동 상세</h3>
      <button type="button" class="fa-modal-close" onclick="closeDayChangeModal()" aria-label="닫기">✕</button>
    </div>
    <div class="fa-modal-body">
      <div class="fa-modal-summary-grid">
        <div class="fa-modal-stat-box">
          <div class="fa-modal-stat-lbl">당일 순 변동 합계</div>
          <div class="fa-modal-stat-val {net_cls}">{eval_day_change:+,.0f}원</div>
        </div>
        <div class="fa-modal-stat-box">
          <div class="fa-modal-stat-lbl">하락 종목 ({loss_count}개)</div>
          <div class="fa-modal-stat-val fa-num-negative">{total_loss:+,.0f}원</div>
        </div>
        <div class="fa-modal-stat-box">
          <div class="fa-modal-stat-lbl">상승 종목 ({gain_count}개)</div>
          <div class="fa-modal-stat-val fa-num-positive">{total_gain:+,.0f}원</div>
        </div>
      </div>
      <div class="fa-table-wrapper">
        <table class="fa-table fa-table-modal-detail">
          <thead>
            <tr>
              <th>계좌</th>
              <th>종목</th>
              <th class="text-right">보유수량</th>
              <th class="text-right">현재가</th>
              <th class="text-right">등락률</th>
              <th class="text-right">전일대비 변동액</th>
            </tr>
          </thead>
          <tbody>
            {''.join(table_rows)}
          </tbody>
        </table>
      </div>
    </div>
  </div>
</div>
"""


def _build_invest_change_modal(data: ReportData, cur_inv: float, prev_inv: float, inv_month_change: float) -> str:
    """총 투자금의 전월대비 증감액(당월 신규 입금 내역)을 보여주는 팝업 모달"""
    month_prefix = data.month_end.strftime("%Y-%m")
    month_label = data.month_end.strftime("%Y.%m")

    month_rec = data.records[data.records["일자"].dt.strftime("%Y-%m") == month_prefix].copy() if "일자" in data.records.columns else pd.DataFrame()
    invest_rec = month_rec[month_rec["투자금"].notna() & (month_rec["투자금"] > 0)].copy() if not month_rec.empty else pd.DataFrame()

    table_rows = []
    if not invest_rec.empty:
        invest_rec = invest_rec.sort_values(by=["일자", "투자금"], ascending=[True, False])
        for _, r in invest_rec.iterrows():
            d_str = r["일자"].strftime("%Y.%m.%d") if pd.notna(r["일자"]) else "-"
            acct_lbl = update_fa.account_label(str(r["계좌"]))
            sym_or_kind = str(r["종목"] or r["구분"] or "입금").strip()
            dep_val = _as_float(r["투자금"]) or 0.0
            memo = str(r["비고"] or "").strip()

            table_rows.append(
                f"<tr>"
                f"  <td class='fa-num'>{d_str}</td>"
                f"  <td><span class='fa-chip-account'>{html.escape(acct_lbl)}</span></td>"
                f"  <td><strong>{html.escape(sym_or_kind)}</strong></td>"
                f"  <td class='text-right fa-num fa-font-bold fa-num-positive'>+{dep_val:,.0f}원</td>"
                f"  <td style='color:var(--fa-text-muted); font-size:0.82rem;'>{html.escape(memo)}</td>"
                f"</tr>"
            )

    if not table_rows:
        table_rows.append("<tr><td colspan='5' class='text-center fa-empty-text' style='padding:20px;'>당월 신규 입금(투자) 내역이 없습니다.</td></tr>")

    net_cls = "fa-num-positive" if inv_month_change > 0 else "fa-num-negative" if inv_month_change < 0 else ""

    return f"""
<div id="fa-invest-change-modal" class="fa-modal-overlay" onclick="if(event.target===this)closeInvestChangeModal()">
  <div class="fa-modal-card">
    <div class="fa-modal-header">
      <h3 class="fa-modal-title">💰 {month_label}월 투자금(신규 입금) 변동 상세</h3>
      <button type="button" class="fa-modal-close" onclick="closeInvestChangeModal()" aria-label="닫기">✕</button>
    </div>
    <div class="fa-modal-body">
      <div class="fa-modal-summary-grid">
        <div class="fa-modal-stat-box">
          <div class="fa-modal-stat-lbl">당월 누적 총 투자금</div>
          <div class="fa-modal-stat-val">{cur_inv:,.0f}원</div>
        </div>
        <div class="fa-modal-stat-box">
          <div class="fa-modal-stat-lbl">전월 말 누적 투자금</div>
          <div class="fa-modal-stat-val" style="color:var(--fa-text-muted);">{prev_inv:,.0f}원</div>
        </div>
        <div class="fa-modal-stat-box">
          <div class="fa-modal-stat-lbl">전월대비 순 증감액</div>
          <div class="fa-modal-stat-val {net_cls}">{inv_month_change:+,.0f}원</div>
        </div>
      </div>
      <div class="fa-table-wrapper">
        <table class="fa-table fa-table-modal-detail">
          <thead>
            <tr>
              <th>일자</th>
              <th>계좌</th>
              <th>구분 / 종목</th>
              <th class="text-right">입금액 (원금)</th>
              <th>메모</th>
            </tr>
          </thead>
          <tbody>
            {''.join(table_rows)}
          </tbody>
        </table>
      </div>
    </div>
  </div>
</div>
"""


def _build_dividend_change_modal(data: ReportData, cur_div: float, prev_div: float, div_month_change: float) -> str:
    """월 배당금의 전월대비 증감액(당월 배당 수령 내역)을 보여주는 팝업 모달"""
    month_prefix = data.month_end.strftime("%Y-%m")
    month_label = data.month_end.strftime("%Y.%m")

    month_rec = data.records[data.records["일자"].dt.strftime("%Y-%m") == month_prefix].copy() if "일자" in data.records.columns else pd.DataFrame()
    div_rec = month_rec[month_rec["배당"].notna() & (month_rec["배당"] > 0)].copy() if not month_rec.empty else pd.DataFrame()

    table_rows = []
    if not div_rec.empty:
        div_rec = div_rec.sort_values(by=["일자", "배당"], ascending=[True, False])
        for _, r in div_rec.iterrows():
            d_str = r["일자"].strftime("%Y.%m.%d") if pd.notna(r["일자"]) else "-"
            acct_lbl = update_fa.account_label(str(r["계좌"]))
            sym = str(r["종목"] or "").strip()
            raw_div = _as_float(r["배당"]) or 0.0

            fx_rate = _as_float(r.get("환율"))
            if not fx_rate or fx_rate <= 0:
                fx_rate = float(data.fx_series_full.loc[r["일자"]]) if r["일자"] in data.fx_series_full.index else 1.0

            div_krw = raw_div * fx_rate if r["계좌"] == "usa" else raw_div
            memo = str(r["비고"] or "").strip()

            foreign_str = f"${raw_div:,.2f}" if r["계좌"] == "usa" else "-"

            table_rows.append(
                f"<tr>"
                f"  <td class='fa-num'>{d_str}</td>"
                f"  <td><span class='fa-chip-account'>{html.escape(acct_lbl)}</span></td>"
                f"  <td><strong>{html.escape(sym)}</strong></td>"
                f"  <td class='text-right fa-num fa-font-bold' style='color:var(--fa-purple);'>+{div_krw:,.0f}원</td>"
                f"  <td class='text-right fa-num fa-hide-mobile' style='color:var(--fa-text-muted); font-size:0.82rem;'>{foreign_str}</td>"
                f"  <td style='color:var(--fa-text-muted); font-size:0.82rem;'>{html.escape(memo)}</td>"
                f"</tr>"
            )

    if not table_rows:
        table_rows.append("<tr><td colspan='6' class='text-center fa-empty-text' style='padding:20px;'>당월 수령한 배당금 내역이 없습니다.</td></tr>")

    net_cls = "fa-num-positive" if div_month_change > 0 else "fa-num-negative" if div_month_change < 0 else ""

    return f"""
<div id="fa-dividend-change-modal" class="fa-modal-overlay" onclick="if(event.target===this)closeDividendChangeModal()">
  <div class="fa-modal-card">
    <div class="fa-modal-header">
      <h3 class="fa-modal-title">🎁 {month_label}월 배당금 수령 내역 및 변동 상세</h3>
      <button type="button" class="fa-modal-close" onclick="closeDividendChangeModal()" aria-label="닫기">✕</button>
    </div>
    <div class="fa-modal-body">
      <div class="fa-modal-summary-grid">
        <div class="fa-modal-stat-box">
          <div class="fa-modal-stat-lbl">당월 배당금 합계</div>
          <div class="fa-modal-stat-val" style="color:var(--fa-purple);">{cur_div:,.0f}원</div>
        </div>
        <div class="fa-modal-stat-box">
          <div class="fa-modal-stat-lbl">전월 배당금 합계</div>
          <div class="fa-modal-stat-val" style="color:var(--fa-text-muted);">{prev_div:,.0f}원</div>
        </div>
        <div class="fa-modal-stat-box">
          <div class="fa-modal-stat-lbl">전월대비 배당 증감</div>
          <div class="fa-modal-stat-val {net_cls}">{div_month_change:+,.0f}원</div>
        </div>
      </div>
      <div class="fa-table-wrapper">
        <table class="fa-table fa-table-modal-detail">
          <thead>
            <tr>
              <th>일자</th>
              <th>계좌</th>
              <th>종목</th>
              <th class="text-right">배당금 (원화)</th>
              <th class="text-right fa-hide-mobile">외화금액</th>
              <th>메모</th>
            </tr>
          </thead>
          <tbody>
            {''.join(table_rows)}
          </tbody>
        </table>
      </div>
    </div>
  </div>
</div>
"""


def _build_refresh_modal() -> str:
    """대시보드 실시간 갱신 진행 상태를 전면에 보여주는 팝업 모달"""
    return """
<div id="fa-refresh-modal" class="fa-modal-overlay fa-refresh-overlay" style="display:none;" onclick="if(event.target===this&&window._canCloseRefreshModal)closeRefreshModal()">
  <div class="fa-modal-card fa-refresh-modal-card">
    <div class="fa-refresh-modal-header">
      <div class="fa-refresh-header-left">
        <div class="fa-refresh-pulse-icon">
          <span class="fa-refresh-icon-spin">🔄</span>
        </div>
        <div>
          <h3 class="fa-refresh-modal-title">자산 대시보드 실시간 갱신</h3>
          <p class="fa-refresh-modal-desc">최신 시장 데이터와 거래내역을 동기화하여 대시보드를 생성합니다.</p>
        </div>
      </div>
      <button type="button" id="fa-refresh-modal-close" class="fa-modal-close" onclick="closeRefreshModal()" style="display:none;" aria-label="닫기">✕</button>
    </div>

    <div class="fa-refresh-modal-body">
      <!-- 진행률 바 -->
      <div class="fa-refresh-progress-section">
        <div class="fa-refresh-progress-meta">
          <span class="fa-refresh-badge" id="fa-refresh-badge">1 / 3 단계 진행 중</span>
          <span class="fa-refresh-percent" id="fa-refresh-percent">15%</span>
        </div>
        <div class="fa-refresh-progress-track">
          <div class="fa-refresh-progress-fill" id="fa-refresh-progress-fill" style="width: 15%;"></div>
        </div>
      </div>

      <!-- 작업 단계별 리스트 -->
      <div class="fa-refresh-steps-list">
        <div class="fa-refresh-step-item active" id="fa-step-1">
          <div class="fa-step-indicator">
            <span class="fa-step-dot"></span>
            <span class="fa-step-spinner"></span>
            <span class="fa-step-check">✓</span>
          </div>
          <div class="fa-step-info">
            <div class="fa-step-name">1단계: 실시간 주가 및 환율 시세 수집 & 차트 생성</div>
            <div class="fa-step-sub" id="fa-step-sub-1">야후 파이낸스 실시간 시세 조회 및 대시보드 그래프 렌더링 (약 10~15초 소요)</div>
          </div>
        </div>

        <div class="fa-refresh-step-item" id="fa-step-2">
          <div class="fa-step-indicator">
            <span class="fa-step-dot"></span>
            <span class="fa-step-spinner"></span>
            <span class="fa-step-check">✓</span>
          </div>
          <div class="fa-step-info">
            <div class="fa-step-name">2단계: Hugo 블로그 템플릿 통합 빌드</div>
            <div class="fa-step-sub" id="fa-step-sub-2">헤더, 내비게이션, 댓글 시스템 및 사이트 레이아웃 결합</div>
          </div>
        </div>

        <div class="fa-refresh-step-item" id="fa-step-3">
          <div class="fa-step-indicator">
            <span class="fa-step-dot"></span>
            <span class="fa-step-spinner"></span>
            <span class="fa-step-check">✓</span>
          </div>
          <div class="fa-step-info">
            <div class="fa-step-name">3단계: 웹 서비스 배포 및 정적 캐시 동기화</div>
            <div class="fa-step-sub" id="fa-step-sub-3">도커 Nginx 웹 서빙 경로 반영 및 브라우저 최신화 완료</div>
          </div>
        </div>
      </div>

      <!-- 실시간 안내 메시지 및 소요 시간 -->
      <div class="fa-refresh-status-bar">
        <div class="fa-refresh-current-msg">
          <span class="fa-spin" id="fa-refresh-msg-spinner">⏳</span>
          <span id="fa-refresh-msg-text">대시보드 갱신 작업을 시작했습니다...</span>
        </div>
        <div class="fa-refresh-timer" id="fa-refresh-timer">⏱️ 소요 시간: <b id="fa-refresh-timer-sec">0초</b></div>
      </div>

      <!-- 완료 메시지 박스 -->
      <div id="fa-refresh-success-banner" class="fa-refresh-banner fa-refresh-success" style="display:none;">
        <div class="fa-banner-icon">🎉</div>
        <div>
          <div class="fa-banner-title">대시보드 갱신이 완료되었습니다!</div>
          <div class="fa-banner-desc">최신 데이터가 적용된 대시보드를 불러오기 위해 페이지를 새로고침합니다...</div>
        </div>
      </div>

      <!-- 에러 메시지 박스 -->
      <div id="fa-refresh-error-banner" class="fa-refresh-banner fa-refresh-error" style="display:none;">
        <div class="fa-banner-icon">⚠️</div>
        <div>
          <div class="fa-banner-title">갱신 작업 중 문제가 발생했습니다.</div>
          <div class="fa-banner-desc" id="fa-refresh-error-desc">서버 로그를 확인하거나 잠시 후 다시 시도해 주세요.</div>
        </div>
      </div>
    </div>
  </div>
</div>
"""


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

    profit_str = _fmt_krw(profit)
    profit_state = "positive" if (profit or 0) > 0 else "negative" if (profit or 0) < 0 else ""
    return_str = _fmt_pct(return_rate)
    return_state = "positive" if (return_rate or 0) > 0 else "negative" if (return_rate or 0) < 0 else ""

    # 1. 총 평가금 전월대비 증감액
    eval_month_change = 0.0
    cur_eval_val = valuation or 0.0
    prev_eval_val = cur_eval_val
    if data.account_df is not None and not data.account_df.empty:
        total_eval_s = data.account_df.sum(axis=1)
        monthly_eval = total_eval_s.groupby(total_eval_s.index.to_period("M")).last()
        if len(monthly_eval) >= 2:
            cur_eval_val = valuation if valuation is not None else float(monthly_eval.iloc[-1])
            prev_eval_val = float(monthly_eval.iloc[-2])
            eval_month_change = cur_eval_val - prev_eval_val
        elif len(monthly_eval) == 1:
            cur_eval_val = valuation if valuation is not None else float(monthly_eval.iloc[-1])
            prev_eval_val = 0.0
            eval_month_change = cur_eval_val
    eval_state = "positive" if eval_month_change > 0 else "negative" if eval_month_change < 0 else ""
    eval_sub = f"전월대비 {eval_month_change:+,.0f}" if eval_month_change != 0 else "전월대비 0"

    # 전일대비 변동액 (총 수익금/수익률 및 일별 모달용)
    eval_day_change = 0.0
    if not data.holdings_df.empty:
        for _, hrow in data.holdings_df.iterrows():
            r = _as_float(hrow.get("등락률"))
            val = _as_float(hrow.get("평가금")) or 0.0
            if r is not None and (1.0 + r) != 0:
                eval_day_change += (val * r / (1.0 + r))

    # 2. 총 투자액 전월대비 증감액
    inv_month_change = 0.0
    cur_inv_val = invest or 0.0
    prev_inv_val = cur_inv_val
    if data.invest_series is not None and not data.invest_series.empty:
        inv_s = data.invest_series.dropna()
        monthly_inv = inv_s.groupby(inv_s.index.to_period("M")).last()
        if len(monthly_inv) >= 2:
            cur_inv_val = float(monthly_inv.iloc[-1])
            prev_inv_val = float(monthly_inv.iloc[-2])
            inv_month_change = cur_inv_val - prev_inv_val
        elif len(monthly_inv) == 1:
            cur_inv_val = float(monthly_inv.iloc[-1])
            prev_inv_val = 0.0
            inv_month_change = cur_inv_val
    inv_state = "positive" if inv_month_change > 0 else "negative" if inv_month_change < 0 else ""
    inv_sub = f"전월대비 {inv_month_change:+,.0f}" if inv_month_change != 0 else "전월대비 0"

    # 3. 총 수익금 전일대비 증감액
    profit_day_change = eval_day_change
    profit_day_state = "positive" if profit_day_change > 0 else "negative" if profit_day_change < 0 else ""
    profit_sub = f"전일대비 {profit_day_change:+,.0f}" if profit_day_change != 0 else "전일대비 0"

    # 4. 총 수익률 전일대비 증감률 (%p)
    rate_day_change_p = 0.0
    if invest and invest > 0:
        rate_day_change_p = (profit_day_change / invest) * 100.0
    rate_day_state = "positive" if rate_day_change_p > 0 else "negative" if rate_day_change_p < 0 else ""
    rate_sub = f"전일대비 {rate_day_change_p:+.2f}%p" if rate_day_change_p != 0 else "전일대비 0.00%p"

    # 5. 월 배당금 전월대비 증감액
    div_month_change = 0.0
    cur_div_val = monthly_div or 0.0
    prev_div_val = 0.0
    if data.dividends_pivot is not None and not data.dividends_pivot.empty:
        monthly_div_s = data.dividends_pivot.sort_index().sum(axis=1)
        if len(monthly_div_s) >= 2:
            cur_div_val = float(monthly_div_s.iloc[-1])
            prev_div_val = float(monthly_div_s.iloc[-2])
            div_month_change = cur_div_val - prev_div_val
        elif len(monthly_div_s) == 1:
            cur_div_val = float(monthly_div_s.iloc[-1])
            prev_div_val = 0.0
            div_month_change = cur_div_val
    div_state = "positive" if div_month_change > 0 else "negative" if div_month_change < 0 else ""
    div_sub = f"전월대비 {div_month_change:+,.0f}" if div_month_change != 0 else "전월대비 0"

    eval_modal_html = _build_eval_change_modal(data, cur_eval_val, prev_eval_val, eval_month_change)
    day_modal_html = _build_day_change_modal(data, eval_day_change)
    invest_modal_html = _build_invest_change_modal(data, cur_inv_val, prev_inv_val, inv_month_change)
    dividend_modal_html = _build_dividend_change_modal(data, cur_div_val, prev_div_val, div_month_change)
    refresh_modal_html = _build_refresh_modal()

    cards = [
        _kpi_card("총 평가금", _fmt_krw(valuation), eval_sub, sub_state=eval_state, sub_onclick="openEvalChangeModal()", sub_title="클릭하여 최근 12개월 전월대비 변동 추세 보기"),
        _kpi_card("총 투자금", _fmt_krw(invest), inv_sub, sub_state=inv_state, sub_onclick="openInvestChangeModal()"),
        _kpi_card("총 수익금", f"<span class='fa-num-{profit_state}'>{profit_str}</span>", profit_sub, state=profit_state, sub_state=profit_day_state, sub_onclick="openDayChangeModal()"),
        _kpi_card("총 수익률", f"<span class='fa-num-{return_state}'>{return_str}</span>", rate_sub, state=return_state, sub_state=rate_day_state, sub_onclick="openDayChangeModal()"),
        _kpi_card("월 배당금", _fmt_krw(monthly_div), div_sub, sub_state=div_state, sub_onclick="openDividendChangeModal()"),
        _kpi_card("USD/KRW", _fmt_number(fx, 2, ""), fx_change_text, fx_state, sub_onclick="openMarketModal('USDKRW=X', 'USD/KRW 원/달러 환율')"),
    ]
    return "<div class=\"fa-kpi-grid\">" + "".join(cards) + "</div>" + eval_modal_html + day_modal_html + invest_modal_html + dividend_modal_html + refresh_modal_html


def _fetch_all_market_history() -> Dict[str, Any]:
    """7대 시장 지표(환율/지수)의 10년 일별 종가 시계열 데이터를 스마트 수집"""
    import urllib.request
    import urllib.parse
    from datetime import datetime

    symbols = ["USDKRW=X", "^GSPC", "^NDX", "SCHD", "IEF", "^KS11", "^KQ11"]
    result_data = {}

    for sym in symbols:
        url = f"https://query1.finance.yahoo.com/v8/finance/chart/{urllib.parse.quote(sym)}?range=10y&interval=1d"
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"})
        try:
            with urllib.request.urlopen(req, timeout=8) as res:
                data = json.loads(res.read().decode("utf-8"))
                res_obj = data["chart"]["result"][0]
                ts = res_obj.get("timestamp", [])
                closes = res_obj["indicators"]["quote"][0].get("close", [])

                dates = []
                vals = []
                for t, c in zip(ts, closes):
                    if c is not None:
                        dates.append(datetime.fromtimestamp(t).strftime("%Y-%m-%d"))
                        vals.append(round(float(c), 2))
                result_data[sym] = {"dates": dates, "closes": vals}
        except Exception as e:
            print(f"(참고) {sym} 10년 시계열 수집 건너뜀: {e}")

    return result_data


def _build_market_chart_modal(market_history_data: Dict[str, Any]) -> str:
    """환율/지수 10년 시계열 인터랙티브 차트 모달"""
    json_str = json.dumps(market_history_data, ensure_ascii=False)
    return f"""
<div id="marketChartModal" class="fa-modal-overlay" onclick="if(event.target===this)closeMarketModal()">
  <div class="fa-modal-card" style="max-width: 860px;">
    <div class="fa-modal-header">
      <h3 id="modalChartTitle" class="fa-modal-title">📈 시장 지수 / 환율 추세</h3>
      <button type="button" class="fa-modal-close" onclick="closeMarketModal()" aria-label="닫기">✕</button>
    </div>
    <div class="fa-modal-body">
      <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 10px; margin-bottom: 14px;">
        <div class="period-tabs" style="display: flex; gap: 6px; flex-wrap: wrap; margin-bottom: 0;">
          <button type="button" class="tab-btn" onclick="changeMarketPeriod('1M')">1개월</button>
          <button type="button" class="tab-btn" onclick="changeMarketPeriod('3M')">3개월</button>
          <button type="button" class="tab-btn active" onclick="changeMarketPeriod('6M')">6개월</button>
          <button type="button" class="tab-btn" onclick="changeMarketPeriod('1Y')">1년</button>
          <button type="button" class="tab-btn" onclick="changeMarketPeriod('5Y')">5년</button>
          <button type="button" class="tab-btn" onclick="changeMarketPeriod('10Y')">10년</button>
        </div>
        <div id="modalPeriodStats" style="display: flex; gap: 12px; font-size: 0.84rem; color: var(--fa-text-muted);">
        </div>
      </div>
      <div id="plotlyMarketChart" style="width: 100%; height: 420px; min-height: 420px;"></div>
    </div>
  </div>
</div>
<script>
window.MARKET_HISTORY_DATA = {json_str};
</script>
"""


def _build_market_kpi_row() -> str:
    snapshots = _fetch_market_snapshots()
    market_history_data = _fetch_all_market_history()
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
                sub_onclick=f"openMarketModal('{ticker}', '{cfg['label']}')",
            )
        )
    return "<div class=\"fa-kpi-grid fa-kpi-grid-market\">" + "".join(cards) + "</div>" + _build_market_chart_modal(market_history_data)


def _build_assets_trend(account_df: pd.DataFrame, period: str = "1Y") -> go.Figure:
    fig = go.Figure()
    df = account_df.iloc[-12:] if period == "1Y" and len(account_df) > 12 else account_df.copy()

    for idx, column in enumerate(df.columns):
        label = update_fa.account_label(column)
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df[column],
                mode="lines",
                name=label,
                line=dict(color=_palette_color(idx), width=2.5),
                hovertemplate="%{x|%Y-%m}: %{y:,.0f}<extra>%{fullData.name}</extra>",
            )
        )

    # 전체 합산 최고점 표시
    if not df.empty:
        total_eval = df.sum(axis=1)
        max_val = total_eval.max()
        max_idx = total_eval.idxmax()
        max_str = f"합계최고 {max_val/100000000:.2f}억" if max_val >= 100000000 else f"합계최고 {max_val:,.0f}"

        # 최고점 점선 수직선 또는 뱃지 마커
        max_acct_col = df.loc[max_idx].idxmax()
        max_acct_top_val = df.loc[max_idx, max_acct_col]

        fig.add_trace(
            go.Scatter(
                x=[max_idx],
                y=[max_acct_top_val],
                mode="markers+text",
                name="합계 최고점",
                marker=dict(size=8, color="#E53E3E", line=dict(color="#ffffff", width=2)),
                text=[f"🏆 {max_str}"],
                textposition="top center",
                textfont=dict(size=12, color="#E53E3E", family=FONT_FAMILY),
                hoverinfo="none",
                showlegend=False,
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
    y_max = df.max().max() if not df.empty else 0
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


def _build_assets_investment_trend(account_df: pd.DataFrame, invest_series: pd.Series, period: str = "1Y") -> go.Figure:
    fig = go.Figure()
    df = account_df.iloc[-12:] if period == "1Y" and len(account_df) > 12 else account_df.copy()

    total_valuation = df.sum(axis=1)
    invest_aligned = update_fa.align_series(invest_series, df.index)

    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=invest_aligned,
            mode="lines",
            name="누적 투자금",
            line=dict(color="#4A5568", width=2.5),
            hovertemplate="%{x|%Y-%m}: %{y:,.0f}<extra>누적 투자금</extra>",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=total_valuation,
            mode="lines",
            name="누적 평가금",
            line=dict(color="#E53E3E", width=2.5),
            hovertemplate="%{x|%Y-%m}: %{y:,.0f}<extra>누적 평가금</extra>",
        )
    )

    # 최고 평가금 포인트 마커 표시
    if len(total_valuation) > 0 and not total_valuation.isna().all():
        max_val = total_valuation.max()
        max_idx = total_valuation.idxmax()
        max_str = f"최고 {max_val/100000000:.2f}억" if max_val >= 100000000 else f"최고 {max_val:,.0f}"

        fig.add_trace(
            go.Scatter(
                x=[max_idx],
                y=[max_val],
                mode="markers+text",
                name="최고 평가금",
                marker=dict(size=8, color="#E53E3E", line=dict(color="#ffffff", width=2)),
                text=[f"🏆 {max_str}"],
                textposition="top center",
                textfont=dict(size=12, color="#E53E3E", family=FONT_FAMILY),
                hoverinfo="none",
                showlegend=False,
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


def _render_trend_tab_card(title: str, fig_1y: go.Figure, fig_all: go.Figure, card_id: str) -> str:
    """직전 1년 / 전체 기간 탭이 포함된 차트 카드 HTML (전체 탭 기본 활성화)"""
    return f"""
<section class="fa-card fa-card-wide fa-card-tabs" id="{card_id}">
  <header class="fa-card-head" style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 8px;">
    <h2>{title}</h2>
    <div class="fa-holdings-tab-nav" style="display: flex; margin-bottom: 0;">
      <button type="button" class="fa-tab-btn" data-target="{card_id}-1y">직전 1년</button>
      <button type="button" class="fa-tab-btn active" data-target="{card_id}-all">전체</button>
    </div>
  </header>
  <div class="fa-card-body">
    <div id="{card_id}-1y" class="fa-tab-pane">
      {_render_figure_html(fig_1y)}
    </div>
    <div id="{card_id}-all" class="fa-tab-pane active">
      {_render_figure_html(fig_all)}
    </div>
  </div>
</section>
"""


def _collect_rebalancing_alerts(data: ReportData, threshold_pct: float = 3.0) -> List[Dict[str, Any]]:
    """모든 계좌에서 목표 비중과 threshold_pct(기본 3.0%p) 이상 격차가 발생한 리밸런싱 필요 항목을 수집합니다."""
    alerts = []
    for account in update_fa.ACCOUNT_ORDER:
        account_holdings = data.holdings_df[data.holdings_df["계좌"] == account].copy()
        if account == "sema" and account_holdings.empty and not data.summary_df.empty:
            status_row = data.summary_df[data.summary_df["계좌"] == account]
            if not status_row.empty:
                invest_val = _as_float(status_row.iloc[0].get("투자금"))
                eval_val = _as_float(status_row.iloc[0].get("평가금"))
                account_holdings = pd.DataFrame([{
                    "계좌": "sema",
                    "종목": "교직원공제회",
                    "매수금": invest_val,
                    "평가금": eval_val,
                    "수익금": _as_float(status_row.iloc[0].get("수익금")),
                    "수익률": 0.0,
                }])

        raw_account_name = update_fa.ACCOUNT_RAW_NAMES.get(account)
        if not raw_account_name:
            title_val = update_fa.ACCOUNT_TITLES.get(f"title_{account}_detail", "")
            raw_account_name = title_val.replace("◉ 상세계좌: ", "").strip()

        rebal_df = update_fa.calculate_rebalancing_df(raw_account_name, account_holdings, data.symbol_map)
        if rebal_df is None or rebal_df.empty:
            continue

        acct_label = update_fa.account_label(account)

        for _, row in rebal_df.iterrows():
            curr_pct = float(row["현재비중"])
            target_pct = float(row["목표비중"])
            diff_pct = curr_pct - target_pct
            diff_amt = float(row["조정금액"])
            asset_name = str(row["자산군"])

            if abs(diff_pct) >= threshold_pct:
                is_buy = (diff_amt > 0)
                alerts.append({
                    "account": account,
                    "account_label": acct_label,
                    "asset_name": asset_name,
                    "curr_pct": curr_pct,
                    "target_pct": target_pct,
                    "diff_pct": diff_pct,
                    "diff_amt": diff_amt,
                    "is_buy": is_buy,
                })

    # 비중 격차 절대값 큰 순으로 정렬
    alerts.sort(key=lambda x: abs(x["diff_pct"]), reverse=True)
    return alerts


def _build_rebalancing_alert_banner(data: ReportData, threshold_pct: float = 3.0) -> Optional[str]:
    """목표 비중 대비 3%p 이상 격차가 발생한 항목들을 상단 알림 배너로 렌더링"""
    alerts = _collect_rebalancing_alerts(data, threshold_pct)
    if not alerts:
        return None

    cards_html = []
    for item in alerts:
        is_buy = item["is_buy"]
        badge_cls = "fa-badge-positive" if is_buy else "fa-badge-negative"
        action_text = "매수 필요" if is_buy else "매도 필요"
        color_cls = "fa-num-positive" if is_buy else "fa-num-negative"
        sign_str = "+" if item["diff_amt"] > 0 else ""
        diff_pct_str = f"{item['diff_pct']:+.1f}%p"

        cards_html.append(
            f"<div class='fa-rebal-alert-item {'buy' if is_buy else 'sell'}'>"
            f"  <div class='fa-rebal-alert-head'>"
            f"    <div class='fa-rebal-alert-sym-wrap'>"
            f"      <span class='fa-chip-account'>{html.escape(item['account_label'])}</span>"
            f"      <strong class='fa-rebal-alert-name'>{html.escape(item['asset_name'])}</strong>"
            f"    </div>"
            f"    <span class='fa-badge {badge_cls}'>{action_text}</span>"
            f"  </div>"
            f"  <div class='fa-rebal-alert-body'>"
            f"    <div class='fa-rebal-alert-stat'>"
            f"      <span class='fa-rebal-alert-lbl'>현재 ➡️ 목표 비중</span>"
            f"      <span class='fa-rebal-alert-val'>{item['curr_pct']:.1f}% ➡️ {item['target_pct']:.1f}%</span>"
            f"    </div>"
            f"    <div class='fa-rebal-alert-stat'>"
            f"      <span class='fa-rebal-alert-lbl'>조정 필요금액</span>"
            f"      <span class='fa-rebal-alert-val fa-font-bold {color_cls}'>{sign_str}{item['diff_amt']:,.0f}</span>"
            f"    </div>"
            f"  </div>"
            f"</div>"
        )

    return (
        "<section class='fa-card fa-card-wide fa-rebal-alert-card'>"
        "  <div class='fa-rebal-alert-top-bar'>"
        "    <div class='fa-rebal-alert-title-wrap'>"
        "      <span class='fa-rebal-alert-icon'>⚖️</span>"
        "      <div>"
        f"        <div class='fa-rebal-alert-main-title'>포트폴리오 리밸런싱 알림 <span class='fa-badge fa-badge-negative' style='font-size:0.75rem; vertical-align:middle; margin-left:6px;'>{len(alerts)}건 조정 필요</span></div>"
        "        <div class='fa-rebal-alert-sub-title'>계좌별 목표 비중 대비 ±3%p 이상 벗어난 자산군 현황입니다.</div>"
        "      </div>"
        "    </div>"
        "  </div>"
        f"  <div class='fa-rebal-alert-grid'>{''.join(cards_html)}</div>"
        "</section>"
    )


def _build_single_allocation_pie(df_in: pd.DataFrame, label_col: str, title_text: str) -> go.Figure:
    fig = go.Figure(
        data=[
            go.Pie(
                labels=df_in[label_col],
                values=df_in["평가금"],
                textinfo="percent",
                textposition="inside",
                insidetextfont=dict(size=14, color="#ffffff", family=FONT_FAMILY),
                insidetextorientation="horizontal",
                hole=0.46,
                showlegend=True,
                marker=dict(colors=[_palette_color(i) for i in range(len(df_in))]),
                hovertemplate="<b>%{label}</b><br>평가금: %{value:,.0f}<br>비중: %{percent}<extra></extra>",
            )
        ]
    )
    fig.update_layout(
        height=340,
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
    return fig


def _build_portfolio_allocation_section(
    holdings_df: pd.DataFrame,
    symbol_map: Dict[str, update_fa.AssetConfig],
    fig_renderer: Callable[[go.Figure], str],
) -> str:
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

    fig_group = _build_single_allocation_pie(group_df, "asset_group", "자산군 비중")
    fig_region = _build_single_allocation_pie(region_df, "region", "지역 비중")
    fig_class = _build_single_allocation_pie(class_df, "asset_class", "대표 자산 비중")

    html_parts = [
        '<div class="fa-card fa-card-tabs fa-card-wide fa-alloc-card">',
        '  <div class="fa-card-head">',
        '    <h2 style="margin-bottom:12px;">전체 포트폴리오 비중</h2>',
        '    <div class="fa-tab-nav-wrapper fa-alloc-tab-nav" style="margin-bottom:4px;">',
        '      <div class="fa-tab-nav">',
        '        <button type="button" class="fa-tab-btn active" data-target="alloc-tab-group">자산군 비중</button>',
        '        <button type="button" class="fa-tab-btn" data-target="alloc-tab-region">지역 비중</button>',
        '        <button type="button" class="fa-tab-btn" data-target="alloc-tab-class">대표 자산 비중</button>',
        '      </div>',
        '    </div>',
        '  </div>',
        '  <div class="fa-card-body fa-alloc-body">',
        f'    <div id="alloc-tab-group" class="fa-tab-pane active">{fig_renderer(fig_group)}</div>',
        f'    <div id="alloc-tab-region" class="fa-tab-pane">{fig_renderer(fig_region)}</div>',
        f'    <div id="alloc-tab-class" class="fa-tab-pane">{fig_renderer(fig_class)}</div>',
        '  </div>',
        '</div>',
    ]
    return "\n".join(html_parts)


def _extract_dividend_data(records: pd.DataFrame, fx_series: pd.Series):
    """배당 레코드(원화 환산)를 추출하여 연도별, 분기별, 월별 시리즈, 연도별 종목별 상세 DataFrame 및 월별 배당 상세 딕셔너리를 반환합니다."""
    df = records.copy()
    if df.empty or "배당" not in df.columns:
        return pd.Series(dtype=float), pd.DataFrame(), pd.Series(dtype=float), pd.DataFrame(), {}

    divs = df[(df["배당"].notna()) & (df["배당"] > 0)].copy()
    if divs.empty:
        return pd.Series(dtype=float), pd.DataFrame(), pd.Series(dtype=float), pd.DataFrame(), {}

    divs["일자"] = pd.to_datetime(divs["일자"])
    divs["배당원화"] = divs.apply(
        lambda r: update_fa.convert_to_krw(r["계좌"], float(r["배당"]), r["일자"], fx_series),
        axis=1,
    )

    divs["연도"] = divs["일자"].dt.year
    divs["분기key"] = divs["일자"].dt.to_period("Q")
    divs["분기명"] = divs["일자"].apply(lambda d: f"{d.strftime('%y')}.{((d.month - 1) // 3) + 1}Q")
    divs["월"] = divs["일자"].dt.to_period("M").dt.to_timestamp()

    # 1. 연도별 시리즈
    yearly_series = divs.groupby("연도")["배당원화"].sum().sort_index()

    # 2. 분기별 집계 (분기key로 정렬 후 분기명 라벨 사용)
    quarterly_agg = divs.groupby(["분기key", "분기명"])["배당원화"].sum().reset_index()
    quarterly_agg = quarterly_agg.sort_values("분기key")

    # 3. 월별 시리즈
    monthly_series = divs.groupby("월")["배당원화"].sum().sort_index()
    if len(monthly_series) > 24:
        monthly_series = monthly_series.tail(24)

    # 4. 연도별 종목별 상세
    yearly_detail_df = divs.groupby(["연도", "종목"])["배당원화"].sum().reset_index()

    # 5. 월별 배당 상세 딕셔너리 (팝업 모달용)
    monthly_details = {}
    for m_per, grp in divs.groupby(divs["일자"].dt.to_period("M")):
        m_key = m_per.strftime("%Y-%m")
        m_label = m_per.strftime("%Y년 %m월")
        grp_sorted = grp.sort_values(by=["일자", "배당원화"], ascending=[True, False])
        m_total = float(grp["배당원화"].sum())
        items = []
        for _, r in grp_sorted.iterrows():
            d_str = r["일자"].strftime("%Y.%m.%d") if pd.notna(r["일자"]) else "-"
            acct_code = str(r["계좌"])
            acct_lbl = update_fa.account_label(acct_code)
            sym = str(r.get("종목", "") or "").strip()
            raw_div = _as_float(r["배당"]) or 0.0
            div_krw = float(r["배당원화"])
            rate_val = (div_krw / m_total * 100.0) if m_total > 0 else 0.0
            memo = str(r.get("비고", "") or "").strip()
            foreign_str = f"${raw_div:,.2f}" if acct_code == "usa" else "-"

            items.append({
                "date": d_str,
                "account": acct_lbl,
                "symbol": sym,
                "krw": round(div_krw),
                "rate": round(rate_val, 1),
                "foreign": foreign_str,
                "memo": memo
            })

        sym_totals = grp.groupby("종목")["배당원화"].sum()
        top_sym = str(sym_totals.idxmax()) if not sym_totals.empty else ""
        top_krw = float(sym_totals.max()) if not sym_totals.empty else 0.0

        monthly_details[m_key] = {
            "month_label": m_label,
            "total_krw": round(m_total),
            "top_symbol": top_sym,
            "top_krw": round(top_krw),
            "items": items
        }

    return yearly_series, quarterly_agg, monthly_series, yearly_detail_df, monthly_details


def _fmt_krw_bar_abbr(v: float) -> str:
    """막대 상단용 원화 축약 표기 (억/만)"""
    if v >= 100_000_000:
        return f"{v/100_000_000:.1f}억"
    elif v >= 10_000:
        return f"{v/10_000:,.0f}만"
    elif v > 0:
        return f"{v:,.0f}"
    return ""


def _build_yearly_dividend_bar_chart(yearly_series: pd.Series) -> go.Figure:
    fig = go.Figure()
    if yearly_series.empty:
        return fig

    x_labels = [f"'{str(y)[-2:]}" for y in yearly_series.index]
    hover_labels = [f"{y}년" for y in yearly_series.index]
    y_raw = yearly_series.values
    y_max = max(y_raw) if len(y_raw) > 0 else 1.0
    y_range = [0, y_max * 1.18]

    customdata = np.stack((hover_labels, y_raw), axis=-1)
    bar_texts = [_fmt_krw_bar_abbr(v) for v in y_raw]

    fig.add_trace(
        go.Bar(
            x=list(range(len(x_labels))),
            y=y_raw,
            text=bar_texts,
            textposition="outside",
            textfont=dict(size=12, family=FONT_FAMILY, color="#475569"),
            marker=dict(color="#4F46E5", opacity=0.9),
            customdata=customdata,
            hovertemplate="<b>%{customdata[0]}</b><br>총 배당금: <b>%{y:,.0f}원</b><extra></extra>",
            cliponaxis=False,
        )
    )
    fig.update_layout(
        height=320,
        margin=dict(l=15, r=15, t=30, b=25),
        showlegend=False,
        font=dict(family=FONT_FAMILY, size=14),
        paper_bgcolor=THEME_BG,
        plot_bgcolor=THEME_BG,
        bargap=0.35,
    )
    fig.update_yaxes(
        tickformat=",.0f",
        range=y_range,
        showgrid=True,
        gridcolor=THEME_GRID,
        zeroline=False,
        tickfont=dict(size=12, family=FONT_FAMILY),
    )
    fig.update_xaxes(
        tickmode="array",
        tickvals=list(range(len(x_labels))),
        ticktext=x_labels,
        tickfont=dict(size=13, family=FONT_FAMILY),
        showgrid=False,
    )
    return fig


def _build_quarterly_dividend_bar_chart(quarterly_agg: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    if quarterly_agg.empty:
        return fig

    x_labels = []
    hover_labels = []
    year_indices: Dict[int, List[int]] = {}
    seen_years = set()

    for idx, (_, r) in enumerate(quarterly_agg.iterrows()):
        q_per = r["분기key"]
        yr = q_per.year
        q_num = q_per.quarter
        yr_short = str(yr)[-2:]

        if yr not in year_indices:
            year_indices[yr] = []
        year_indices[yr].append(idx)

        if q_num == 1 or yr not in seen_years:
            seen_years.add(yr)
            x_labels.append(f"'{yr_short}.1Q" if q_num == 1 else f"'{yr_short}.{q_num}Q")
        else:
            x_labels.append("")
        hover_labels.append(f"'{yr_short}년 {q_num}분기")

    y_raw = quarterly_agg["배당원화"].values
    y_max = max(y_raw) if len(y_raw) > 0 else 1.0
    y_range = [0, y_max * 1.15]

    customdata = np.stack((hover_labels, y_raw), axis=-1)

    # 연도별 배경 음영 밴드 적용
    for yr_idx, (yr, idxs) in enumerate(year_indices.items()):
        if yr_idx % 2 == 1:
            fig.add_vrect(
                x0=min(idxs) - 0.5,
                x1=max(idxs) + 0.5,
                fillcolor="rgba(148, 163, 184, 0.08)",
                layer="below",
                line_width=0,
            )

    fig.add_trace(
        go.Bar(
            x=list(range(len(x_labels))),
            y=y_raw,
            marker=dict(color="#06B6D4", opacity=0.9),
            customdata=customdata,
            hovertemplate="<b>%{customdata[0]}</b><br>총 배당금: <b>%{y:,.0f}원</b><extra></extra>",
            cliponaxis=False,
        )
    )
    fig.update_layout(
        height=320,
        margin=dict(l=15, r=15, t=30, b=25),
        showlegend=False,
        font=dict(family=FONT_FAMILY, size=14),
        paper_bgcolor=THEME_BG,
        plot_bgcolor=THEME_BG,
        bargap=0.25,
    )
    fig.update_yaxes(
        tickformat=",.0f",
        range=y_range,
        showgrid=True,
        gridcolor=THEME_GRID,
        zeroline=False,
        tickfont=dict(size=12, family=FONT_FAMILY),
    )
    fig.update_xaxes(
        tickmode="array",
        tickvals=list(range(len(x_labels))),
        ticktext=x_labels,
        tickfont=dict(size=12, family=FONT_FAMILY),
        showgrid=False,
    )
    return fig


def _build_monthly_dividend_bar_chart(monthly_series: pd.Series) -> go.Figure:
    fig = go.Figure()
    if monthly_series.empty:
        return fig

    x_labels = []
    hover_labels = []
    month_keys = []
    year_indices: Dict[int, List[int]] = {}

    for idx, d in enumerate(monthly_series.index):
        yr = d.year
        yr_short = d.strftime("%y")
        m = d.month

        if yr not in year_indices:
            year_indices[yr] = []
        year_indices[yr].append(idx)

        if m == 1:
            x_labels.append(f"'{yr_short}.1")
        elif m == 7:
            x_labels.append("7")
        else:
            x_labels.append("")
        hover_labels.append(d.strftime("'%y년 %m월"))
        month_keys.append(d.strftime("%Y-%m"))

    y_raw = monthly_series.values
    y_max = max(y_raw) if len(y_raw) > 0 else 1.0
    y_range = [0, y_max * 1.15]

    customdata = np.stack((hover_labels, month_keys), axis=-1)

    # 연도별 배경 음영 밴드 적용
    for yr_idx, (yr, idxs) in enumerate(year_indices.items()):
        if yr_idx % 2 == 1:
            fig.add_vrect(
                x0=min(idxs) - 0.5,
                x1=max(idxs) + 0.5,
                fillcolor="rgba(148, 163, 184, 0.08)",
                layer="below",
                line_width=0,
            )

    fig.add_trace(
        go.Bar(
            x=list(range(len(x_labels))),
            y=y_raw,
            marker=dict(color="#10B981", opacity=0.9),
            customdata=customdata,
            hovertemplate="<b>%{customdata[0]}</b><br>총 배당금: <b>%{y:,.0f}원</b><br><span style='font-size:11px; color:#cbd5e1;'>👆 클릭하여 배당 재원 보기</span><extra></extra>",
            cliponaxis=False,
        )
    )
    fig.update_layout(
        height=320,
        margin=dict(l=15, r=15, t=30, b=25),
        showlegend=False,
        font=dict(family=FONT_FAMILY, size=14),
        paper_bgcolor=THEME_BG,
        plot_bgcolor=THEME_BG,
        bargap=0.15,
    )
    fig.update_yaxes(
        tickformat=",.0f",
        range=y_range,
        showgrid=True,
        gridcolor=THEME_GRID,
        zeroline=False,
        tickfont=dict(size=12, family=FONT_FAMILY),
    )
    fig.update_xaxes(
        tickmode="array",
        tickvals=list(range(len(x_labels))),
        ticktext=x_labels,
        tickfont=dict(size=12, family=FONT_FAMILY),
        showgrid=False,
    )
    return fig


def _build_last12m_dividend_bar_chart(monthly_series: pd.Series) -> go.Figure:
    """최근 12개월의 월별 배당금을 보여주는 막대 그래프"""
    fig = go.Figure()
    if monthly_series.empty:
        return fig

    last12m_s = monthly_series.iloc[-12:] if len(monthly_series) >= 12 else monthly_series
    x_labels = [d.strftime("'%y.%m") for d in last12m_s.index]
    hover_labels = [d.strftime("%Y년 %m월") for d in last12m_s.index]
    month_keys = [d.strftime("%Y-%m") for d in last12m_s.index]
    y_raw = last12m_s.values
    y_max = max(y_raw) if len(y_raw) > 0 else 1.0
    y_range = [0, y_max * 1.25]

    bar_texts = [_fmt_krw_bar_abbr(v) for v in y_raw]
    customdata = np.stack((hover_labels, month_keys), axis=-1)

    fig.add_trace(
        go.Bar(
            x=list(range(len(x_labels))),
            y=y_raw,
            text=bar_texts,
            textposition="outside",
            textfont=dict(size=11, family=FONT_FAMILY, color="#475569"),
            marker=dict(color="#8B5CF6", opacity=0.9),
            customdata=customdata,
            hovertemplate="<b>%{customdata[0]}</b><br>월 배당금: <b>%{y:,.0f}원</b><br><span style='font-size:11px; color:#cbd5e1;'>👆 클릭하여 배당 재원 보기</span><extra></extra>",
            cliponaxis=False,
        )
    )
    fig.update_layout(
        height=320,
        margin=dict(l=15, r=15, t=30, b=25),
        showlegend=False,
        font=dict(family=FONT_FAMILY, size=14),
        paper_bgcolor=THEME_BG,
        plot_bgcolor=THEME_BG,
        bargap=0.35,
    )
    fig.update_yaxes(
        tickformat=",.0f",
        range=y_range,
        showgrid=True,
        gridcolor=THEME_GRID,
        zeroline=False,
        tickfont=dict(size=12, family=FONT_FAMILY),
    )
    fig.update_xaxes(
        tickmode="array",
        tickvals=list(range(len(x_labels))),
        ticktext=x_labels,
        tickfont=dict(size=12, family=FONT_FAMILY),
        showgrid=False,
    )
    return fig


def _build_yearly_detail_bar_chart(year_detail_df: pd.DataFrame) -> go.Figure:
    """선택된 연도의 종목별 배당금 및 비중을 나타내는 수평 막대 그래프"""
    fig = go.Figure()
    if year_detail_df.empty:
        return fig

    sorted_df = year_detail_df.sort_values("배당원화", ascending=True)
    total_val = sorted_df["배당원화"].sum()

    symbols = sorted_df["종목"].tolist()
    vals = sorted_df["배당원화"].tolist()

    texts = []
    for v in vals:
        pct = (v / total_val * 100.0) if total_val > 0 else 0.0
        texts.append(f"{v:,.0f} ({pct:.1f})")

    fig.add_trace(
        go.Bar(
            y=symbols,
            x=vals,
            orientation="h",
            marker=dict(color=[_palette_color(i) for i in range(len(symbols))]),
            text=texts,
            textposition="outside",
            textfont=dict(size=13, color=THEME_TEXT, family=FONT_FAMILY),
            cliponaxis=False,
            hovertemplate="<b>%{y}</b><br>배당금: %{x:,.0f}<extra></extra>",
        )
    )
    max_val = max(vals) if len(vals) > 0 else 100
    fig.update_layout(
        height=max(260, len(symbols) * 44 + 60),
        margin=dict(l=20, r=120, t=20, b=20),
        showlegend=False,
        font=dict(family=FONT_FAMILY, size=13),
        paper_bgcolor=THEME_BG,
        plot_bgcolor=THEME_BG,
    )
    fig.update_xaxes(
        range=[0, max_val * 1.35],
        tickformat=",.0f",
        showgrid=True,
        gridcolor=THEME_GRID,
        zeroline=False,
        tickfont=dict(size=12, family=FONT_FAMILY),
    )
    fig.update_yaxes(
        ticksuffix="   ",
        tickfont=dict(size=13, family=FONT_FAMILY),
        showgrid=False,
    )
    return fig


def _build_month_dividend_modal(monthly_details: Dict[str, Any]) -> str:
    """월별 배당 재원 및 상세 내역 팝업 모달 HTML 및 인라인 데이터 주입"""
    json_str = json.dumps(monthly_details, ensure_ascii=False)
    return f"""
<div id="fa-month-dividend-modal" class="fa-modal-overlay" onclick="if(event.target===this)closeMonthDividendModal()">
  <div class="fa-modal-card" style="max-width:780px; width:95%;">
    <div class="fa-modal-header">
      <h3 id="fa-month-div-modal-title" class="fa-modal-title">🎁 배당 재원 및 수령 상세</h3>
      <button type="button" class="fa-modal-close" onclick="closeMonthDividendModal()" aria-label="닫기">✕</button>
    </div>
    <div class="fa-modal-body">
      <div class="fa-modal-summary-grid" style="grid-template-columns: repeat(auto-fit, minmax(130px, 1fr));">
        <div class="fa-modal-stat-box">
          <div class="fa-modal-stat-lbl">해당 월 총 배당금</div>
          <div id="fa-month-div-total" class="fa-modal-stat-val" style="color:var(--fa-purple);">-</div>
        </div>
        <div class="fa-modal-stat-box">
          <div class="fa-modal-stat-lbl">배당 지급 종목 수</div>
          <div id="fa-month-div-count" class="fa-modal-stat-val">-</div>
        </div>
        <div class="fa-modal-stat-box">
          <div class="fa-modal-stat-lbl">최대 배당 종목</div>
          <div id="fa-month-div-top-sym" class="fa-modal-stat-val fa-num-positive">-</div>
        </div>
      </div>

      <div class="fa-table-wrapper" style="max-height:360px; overflow-y:auto; margin-top:16px;">
        <table class="fa-table fa-table-modal-detail">
          <thead>
            <tr>
              <th>일자</th>
              <th>계좌</th>
              <th>종목</th>
              <th class="text-right">배당금 (원화)</th>
              <th class="text-right">비중</th>
              <th class="text-right fa-hide-mobile">외화금액</th>
              <th>메모</th>
            </tr>
          </thead>
          <tbody id="fa-month-div-tbody">
          </tbody>
        </table>
      </div>
    </div>
  </div>
</div>
<script>
window.DIVIDEND_MONTH_DETAILS = {json_str};
</script>
"""


def _build_dividends_tabbed_section(
    records: pd.DataFrame,
    fx_series: pd.Series,
    fig_renderer: Callable[[go.Figure], str],
) -> Optional[str]:
    """배당금 현황을 5개 탭(최근 12개월, 연도별, 분기별, 월별, 상세)으로 렌더링하는 통합 컴포넌트"""
    yearly_series, quarterly_agg, monthly_series, yearly_detail_df, monthly_details = _extract_dividend_data(records, fx_series)
    if yearly_series.empty:
        return None

    fig_last12m = _build_last12m_dividend_bar_chart(monthly_series)
    fig_yearly = _build_yearly_dividend_bar_chart(yearly_series)
    fig_quarterly = _build_quarterly_dividend_bar_chart(quarterly_agg)
    fig_monthly = _build_monthly_dividend_bar_chart(monthly_series)
    month_modal_html = _build_month_dividend_modal(monthly_details)

    # 상세 탭 연도별 목록 (내림차순)
    available_years = sorted(yearly_detail_df["연도"].unique().tolist(), reverse=True) if not yearly_detail_df.empty else []

    detail_panels = []
    year_options = []

    for idx, yr in enumerate(available_years):
        is_first = (idx == 0)
        active_cls = " active" if is_first else ""
        year_options.append(f"<option value='{yr}'>{yr}년</option>")

        ydf = yearly_detail_df[yearly_detail_df["연도"] == yr]
        yr_chart = _build_yearly_detail_bar_chart(ydf)
        yr_total = ydf["배당원화"].sum()

        detail_panels.append(
            f"<div id='fa-div-year-pane-{yr}' class='fa-div-year-pane{active_cls}'>"
            f"  <div class='fa-div-year-summary'>"
            f"    <span class='fa-div-summary-tag'>{yr}년 총 배당금</span>"
            f"    <span class='fa-div-summary-val'>{yr_total:,.0f}</span>"
            f"  </div>"
            f"  <div class='fa-div-chart-box'>{fig_renderer(yr_chart)}</div>"
            f"</div>"
        )

    detail_html = (
        f"<div class='fa-div-detail-wrap'>"
        f"  <div class='fa-div-ctrl-bar'>"
        f"    <label for='fa-div-year-select' class='fa-div-select-lbl'>연도 선택</label>"
        f"    <select id='fa-div-year-select' class='fa-select'>{''.join(year_options)}</select>"
        f"  </div>"
        f"  <div class='fa-div-panes-wrap'>{''.join(detail_panels)}</div>"
        f"</div>"
    )

    tabs_html = [
        "<section class='fa-card fa-card-wide'>",
        "  <header class='fa-card-head'>",
        "    <h2>배당금 및 분배금 현황 <span class='fa-unit-badge' style='font-size:0.78rem; color:var(--fa-text-muted); font-weight:normal; margin-left:6px;'>(단위: 원)</span></h2>",
        "  </header>",
        "  <div class='fa-card-body'>",
        "    <div class='fa-card-tabs'>",
        "      <div class='fa-tab-nav fa-tab-nav-sub' style='margin-bottom:16px;'>",
        "        <button class='fa-tab-btn active' data-target='fa-div-tab-last12m'>최근 12개월</button>",
        "        <button class='fa-tab-btn' data-target='fa-div-tab-yearly'>연도별</button>",
        "        <button class='fa-tab-btn' data-target='fa-div-tab-quarterly'>분기별</button>",
        "        <button class='fa-tab-btn' data-target='fa-div-tab-monthly'>월별</button>",
        "        <button class='fa-tab-btn' data-target='fa-div-tab-detail'>상세</button>",
        "      </div>",
        "      <div class='fa-tab-content'>",
        f"        <div id='fa-div-tab-last12m' class='fa-tab-pane active'>{fig_renderer(fig_last12m)}</div>",
        f"        <div id='fa-div-tab-yearly' class='fa-tab-pane'>{fig_renderer(fig_yearly)}</div>",
        f"        <div id='fa-div-tab-quarterly' class='fa-tab-pane'>{fig_renderer(fig_quarterly)}</div>",
        f"        <div id='fa-div-tab-monthly' class='fa-tab-pane'>{fig_renderer(fig_monthly)}</div>",
        f"        <div id='fa-div-tab-detail' class='fa-tab-pane'>{detail_html}</div>",
        "      </div>",
        "    </div>",
        "  </div>",
        month_modal_html,
        "</section>",
    ]
    return "\n".join(tabs_html)


# =========================================================================
# 순수 HTML + 모바일 반응형 카드 뷰 테이블 렌더러
# =========================================================================

def _fmt_man(val: Optional[float]) -> str:
    """원화 금액을 만원 단위(예: 6,200 만원, 12,500 만원) 문자열로 변환합니다."""
    if val is None or pd.isna(val):
        return "-"
    man = val / 10_000.0
    return f"{man:,.0f} 만원"


def _fmt_profit_man(profit: Optional[float], rate: Optional[float]) -> str:
    """수익금(만원 단위) 및 수익률을 조합한 문자열을 반환합니다."""
    if profit is None or pd.isna(profit):
        return "-"
    man_val = profit / 10_000.0
    sign = "+" if man_val > 0 else ""
    man_str = f"{sign}{man_val:,.0f} 만원"
    rate_str = f"{rate * 100:+.1f}" if (rate is not None and not pd.isna(rate)) else ""
    return f"{man_str} ({rate_str})" if rate_str else man_str


def _build_summary_man_table(summary_df: pd.DataFrame) -> str:
    """전체 계좌 요약 현황을 1개의 카드 안에서 계좌명 | 투자금 | 평가금 | 수익금 | 비중 | 배당금 순으로 렌더링 (원 단위)"""
    if summary_df.empty:
        return "<p class='fa-empty-text'>계좌 데이터가 없습니다.</p>"

    lines = [
        "<div class='fa-single-card-box'>",
        "<div class='fa-table-wrapper'>",
        "<table class='fa-table fa-table-eok-summary'>",
        "<thead>",
        "  <tr>",
        "    <th>계좌명</th>",
        "    <th class='text-right'>투자금</th>",
        "    <th class='text-right'>평가금</th>",
        "    <th class='text-right'>수익금</th>",
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

        invest_str = f"{invest:,.0f}" if invest is not None and not pd.isna(invest) else "-"
        eval_str = f"{valuation:,.0f}" if valuation is not None and not pd.isna(valuation) else "-"
        sign = "+" if (profit or 0) > 0 else ""
        profit_val_str = f"{sign}{profit:,.0f}" if profit is not None and not pd.isna(profit) else "-"
        rate_str = f"{return_rate * 100:+.2f}%" if (return_rate is not None and not pd.isna(return_rate)) else ""
        profit_str = f"{profit_val_str} ({rate_str})" if rate_str else profit_val_str
        weight_str = f"{weight * 100:.1f}" if weight is not None else "-"
        div_str = f"{dividend:,.0f}" if dividend is not None and dividend > 0 else "-"

        lines.append("  <tr>")
        lines.append(f"    <td class='fa-col-account'><strong>{html.escape(label)}</strong></td>")
        lines.append(f"    <td class='text-right fa-num'>{invest_str}</td>")
        lines.append(f"    <td class='text-right fa-num fa-font-bold'>{eval_str}</td>")
        lines.append(f"    <td class='text-right fa-num {profit_cls}'><span class='fa-badge {profit_badge}'>{profit_str}</span></td>")
        lines.append(f"    <td class='text-right fa-num'>{weight_str}</td>")
        lines.append(f"    <td class='text-right fa-num' style='color:var(--fa-purple);'>{div_str}</td>")
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

        invest_str = f"{invest:,.0f}" if invest is not None and not pd.isna(invest) else "-"
        eval_str = f"{valuation:,.0f}" if valuation is not None and not pd.isna(valuation) else "-"
        sign = "+" if (profit or 0) > 0 else ""
        profit_val_str = f"{sign}{profit:,.0f}" if profit is not None and not pd.isna(profit) else "-"
        rate_str = f"{return_rate * 100:+.2f}%" if (return_rate is not None and not pd.isna(return_rate)) else ""
        profit_str = f"{profit_val_str} ({rate_str})" if rate_str else profit_val_str
        div_str = f"{dividend:,.0f}" if dividend is not None and dividend > 0 else "-"

        lines.append("<tfoot>")
        lines.append("  <tr class='fa-tr-total'>")
        lines.append("    <td><strong>합계</strong></td>")
        lines.append(f"    <td class='text-right fa-num'>{invest_str}</td>")
        lines.append(f"    <td class='text-right fa-num fa-font-bold'>{eval_str}</td>")
        lines.append(f"    <td class='text-right fa-num {profit_cls}'><span class='fa-badge {profit_badge}'>{profit_str}</span></td>")
        lines.append("    <td class='text-right fa-num'>100.0</td>")
        lines.append(f"    <td class='text-right fa-num' style='color:var(--fa-purple);'>{div_str}</td>")
        lines.append("  </tr>")
        lines.append("</tfoot>")

    lines.append("</table>")
    lines.append("</div>")
    lines.append("</div>")
    return "\n".join(lines)


def _build_single_account_card(row: pd.Series) -> str:
    """개별 계좌 1개의 지표를 1개의 통합 카드로 깔끔하게 렌더링 (투자금 -> 평가금 -> 수익금 -> 비중 -> 배당금, 원 단위)"""
    if row is None:
        return ""
    acct_name = str(row["계좌"])
    label = update_fa.account_label(acct_name) if acct_name != "합계" else "전체 계좌 합산"
    invest = _as_float(row.get("투자금"))
    valuation = _as_float(row.get("평가금"))
    profit = _as_float(row.get("수익금"))
    return_rate = _as_float(row.get("수익률"))
    weight = _as_float(row.get("비중"))
    dividend = _as_float(row.get("배당금"))

    profit_cls = "fa-num-positive" if (profit or 0) > 0 else "fa-num-negative" if (profit or 0) < 0 else ""
    profit_badge = "fa-badge-positive" if (profit or 0) > 0 else "fa-badge-negative" if (profit or 0) < 0 else "fa-badge-neutral"

    invest_str = f"{invest:,.0f}원" if invest is not None else "-"
    eval_str = f"{valuation:,.0f}원" if valuation is not None else "-"
    profit_str = f"{profit:+,.0f}원" if profit is not None and profit != 0 else (f"{profit:,.0f}원" if profit is not None else "-")
    rate_str = f"{return_rate * 100:+.2f}%" if return_rate is not None else ""
    profit_disp = f"{profit_str} ({rate_str})" if rate_str else profit_str
    weight_str = f"{weight * 100:.1f}%" if weight is not None else "-"
    div_str = f"{dividend:,.0f}원" if dividend is not None and dividend > 0 else None

    div_html = f"<div class='fa-stat-line'><span class='fa-stat-lbl'>누적 배당금</span><span class='fa-stat-val' style='color:var(--fa-purple);'>{div_str}</span></div>" if div_str else ""

    return f"""
    <div class="fa-single-card-box">
      <div class="fa-single-card-header">
        <span class="fa-single-card-title">{html.escape(label)}</span>
        <span class="fa-chip-weight">비중 {weight_str}</span>
      </div>
      <div class="fa-stat-line-group">
        <div class="fa-stat-line"><span class="fa-stat-lbl">투자금</span><span class="fa-stat-val">{invest_str}</span></div>
        <div class="fa-stat-line"><span class="fa-stat-lbl">평가금</span><span class="fa-stat-val fa-font-bold">{eval_str}</span></div>
        <div class="fa-stat-line"><span class="fa-stat-lbl">수익금</span><span class="fa-stat-val {profit_cls}"><span class="fa-badge {profit_badge}">{profit_disp}</span></span></div>
        <div class="fa-stat-line"><span class="fa-stat-lbl">비중</span><span class="fa-stat-val">{weight_str}</span></div>
        {div_html}
      </div>
    </div>
    """


def _build_account_assets_html_table(summary_df: pd.DataFrame) -> str:
    """계좌별 자산 현황을 [전체(만원단위)] / [계좌별] 탭 카드 UI로 렌더링"""
    if summary_df.empty:
        return "<p class='fa-empty-text'>계좌 데이터가 없습니다.</p>"

    tab_btns = [
        "<button type='button' class='fa-tab-btn active' data-target='acct-sum-tab-all'>전체</button>",
    ]

    tab_panes = [
        f"<div id='acct-sum-tab-all' class='fa-tab-pane active'>{_build_summary_man_table(summary_df)}</div>",
    ]

    for _, row in summary_df.iterrows():
        acct_name = str(row["계좌"])
        if acct_name == "합계":
            continue
        label = update_fa.account_label(acct_name)
        tab_id = f"acct-sum-tab-{acct_name}"
        tab_btns.append(
            f"<button type='button' class='fa-tab-btn' data-target='{tab_id}'>{html.escape(label)}</button>"
        )
        tab_panes.append(
            f"<div id='{tab_id}' class='fa-tab-pane'>{_build_single_account_card(row)}</div>"
        )

    nav_html = f"<div class='fa-tab-nav-wrapper fa-acct-summary-tab-nav' style='margin-bottom:8px;'><div class='fa-tab-nav' role='tablist'>{''.join(tab_btns)}</div></div>"
    content_html = f"<div class='fa-tab-panes'>{''.join(tab_panes)}</div>"

    return (
        "<section class='fa-card fa-card-wide fa-card-tabs fa-acct-summary-card'>"
        "<header class='fa-card-head'><h2 style='margin-bottom:12px;'>계좌별 자산 현황</h2></header>"
        f"<div class='fa-card-body'>{nav_html}{content_html}</div>"
        "</section>"
    )


def _build_total_holdings_section(holdings_df: pd.DataFrame) -> str:
    """전체 보유 종목 섹션: PC는 상세 테이블 바로 노출 / 모바일은 [요약](4열 간결 테이블) 및 [상세](풀 테이블) 탭 분기"""
    filtered = holdings_df[holdings_df["계좌"] != "sema"].copy()
    if filtered.empty:
        return (
            "<section class='fa-card fa-card-wide'>"
            "<header class='fa-card-head'><h2>실시간 보유종목 현황</h2></header>"
            "<div class='fa-card-body'><p class='fa-empty-text'>보유 종목 데이터가 없습니다.</p></div>"
            "</section>"
        )

    filtered["계좌라벨"] = filtered["계좌"].apply(update_fa.account_label)
    filtered = filtered.sort_values(["계좌", "평가금"], ascending=[True, False])

    summary_rows = []
    detail_rows = []

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
        fluct_badge = "fa-badge-positive" if (fluct_rate or 0) > 0 else "fa-badge-negative" if (fluct_rate or 0) < 0 else "fa-badge-neutral"

        qty_str = f"{qty:,.2f}".rstrip("0").rstrip(".") if qty is not None else "-"
        avg_str = f"{avg_price:,.0f}" if avg_price is not None else "-"
        buy_str = f"{buy_amt:,.0f}" if buy_amt is not None else "-"
        cur_str = f"{cur_price:,.0f}" if cur_price is not None else "-"
        eval_str = f"{eval_amt:,.0f}" if eval_amt is not None else "-"
        profit_str = f"{profit:+,.0f}" if profit is not None and profit != 0 else (f"{profit:,.0f}" if profit is not None else "-")
        rate_str = f"{return_rate * 100:+.2f}" if return_rate is not None else "-"
        fluct_str = f"{fluct_rate * 100:+.2f}" if fluct_rate is not None else "-"

        rate_disp = f"수익 {rate_str}%" if rate_str != "-" else "수익 -"
        fluct_disp = f"등락 {fluct_str}%" if fluct_str != "-" else "등락 -"

        # 1. 요약 테이블 행 (계좌, 종목, 수익금, 수익률, 등락률)
        summary_rows.append("  <tr>")
        summary_rows.append(f"    <td class='fa-col-account'><span class='fa-chip-account'>{html.escape(acct_label)}</span></td>")
        summary_rows.append(f"    <td class='fa-col-sticky-symbol'><strong>{html.escape(symbol)}</strong></td>")
        summary_rows.append(f"    <td class='text-right fa-num {profit_cls}'>{profit_str}</td>")
        summary_rows.append(f"    <td class='text-right fa-num'><span class='fa-badge {profit_badge}'>{rate_str}%</span></td>")
        summary_rows.append(f"    <td class='text-right fa-num'><span class='fa-badge {fluct_badge}'>{fluct_str}%</span></td>")
        summary_rows.append("  </tr>")

        # 2. 상세 테이블 행 (풀 컬럼)
        detail_rows.append("  <tr>")
        detail_rows.append(f"    <td data-label='계좌'><span class='fa-chip-account'>{html.escape(acct_label)}</span></td>")
        detail_rows.append(
            f"    <td data-label='종목' class='fa-col-symbol'>"
            f"<div class='fa-stock-title-wrap'><span class='fa-chip-account fa-mobile-inline'>{html.escape(acct_label)}</span><strong>{html.escape(symbol)}</strong></div>"
            f"<div class='fa-stock-badges-wrap fa-mobile-inline'>"
            f"<span class='fa-badge {profit_badge} fa-stock-rate-badge'>{rate_disp}</span>"
            f"<span class='fa-badge {fluct_badge} fa-stock-fluct-badge'>{fluct_disp}</span>"
            f"</div>"
            f"</td>"
        )
        detail_rows.append(f"    <td data-label='수익률' class='text-right fa-num fa-hide-mobile'><span class='fa-badge {profit_badge}'>{rate_str}</span></td>")
        detail_rows.append(f"    <td data-label='수량' class='text-right fa-num'>{qty_str}</td>")
        detail_rows.append(f"    <td data-label='평단가' class='text-right fa-num'>{avg_str}</td>")
        detail_rows.append(f"    <td data-label='현재가' class='text-right fa-num'>{cur_str}</td>")
        detail_rows.append(f"    <td data-label='매수금' class='text-right fa-num'>{buy_str}</td>")
        detail_rows.append(f"    <td data-label='평가금' class='text-right fa-num fa-font-bold'>{eval_str}</td>")
        detail_rows.append(f"    <td data-label='수익금' class='text-right fa-num {profit_cls}'>{profit_str}</td>")
        detail_rows.append(f"    <td data-label='등락률' class='text-right fa-num fa-hide-mobile {fluct_cls}'>{fluct_str}</td>")
        detail_rows.append("  </tr>")

    # 요약 테이블 HTML (5열 구성)
    summary_table_html = "\n".join([
        "<div class='fa-table-wrapper fa-table-wrapper-sticky'>",
        "<table class='fa-table fa-table-holdings-summary'>",
        "<thead>",
        "  <tr>",
        "    <th class='fa-th-account'>계좌</th>",
        "    <th class='fa-th-symbol'>종목</th>",
        "    <th class='text-right'>수익금</th>",
        "    <th class='text-right'>수익률</th>",
        "    <th class='text-right'>등락률</th>",
        "  </tr>",
        "</thead>",
        "<tbody>",
        *summary_rows,
        "</tbody>",
        "</table>",
        "</div>"
    ])

    # 상세 테이블 HTML
    detail_table_html = "\n".join([
        "<div class='fa-table-wrapper'>",
        "<table class='fa-table fa-table-responsive fa-table-holdings'>",
        "<thead>",
        "  <tr>",
        "    <th>계좌</th>",
        "    <th>종목</th>",
        "    <th class='text-right'>수익률</th>",
        "    <th class='text-right'>수량</th>",
        "    <th class='text-right'>평단가</th>",
        "    <th class='text-right'>현재가</th>",
        "    <th class='text-right'>매수금</th>",
        "    <th class='text-right'>평가금</th>",
        "    <th class='text-right'>수익금</th>",
        "    <th class='text-right'>등락률</th>",
        "  </tr>",
        "</thead>",
        "<tbody>",
        *detail_rows,
        "</tbody>",
        "</table>",
        "</div>"
    ])

    title = update_fa.ACCOUNT_TITLES.get("title_total_holdings", "실시간 보유종목 현황")

    return (
        f"<section class='fa-card fa-card-wide fa-card-tabs fa-holdings-card'>"
        f"  <header class='fa-card-head'>"
        f"    <div style='display:flex; justify-content:space-between; align-items:center; width:100%; flex-wrap:wrap; gap:8px;'>"
        f"      <h2 style='margin-bottom:0;'>{html.escape(title)}</h2>"
        f"      <div class='fa-tab-nav fa-holdings-tab-nav'>"
        f"        <button type='button' class='fa-tab-btn active' data-target='fa-holdings-tab-summary'>요약</button>"
        f"        <button type='button' class='fa-tab-btn' data-target='fa-holdings-tab-detail'>상세</button>"
        f"      </div>"
        f"    </div>"
        f"  </header>"
        f"  <div class='fa-card-body'>"
        f"    <div id='fa-holdings-tab-summary' class='fa-tab-pane active'>{summary_table_html}</div>"
        f"    <div id='fa-holdings-tab-detail' class='fa-tab-pane'>{detail_table_html}</div>"
        f"  </div>"
        f"</section>"
    )


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

    # 계좌별/종목별 누적 투자금(사용자가 직접 입력한 deposit 합) 사전 계산
    inv_records = data.records[(data.records["투자금"].notna()) & (pd.to_numeric(data.records["투자금"], errors="coerce") > 0)].copy()
    inv_by_acct_sym: Dict[Tuple[str, str], float] = {}
    if not inv_records.empty:
        for (acct_code, sym), grp in inv_records.groupby(["계좌", "종목"]):
            sym_clean = str(sym).strip() if pd.notna(sym) else ""
            if sym_clean:
                inv_by_acct_sym[(str(acct_code).strip(), sym_clean)] = float(grp["투자금"].sum())

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
            f"<div class='fa-mini-kpi'><div class='fa-mini-kpi-lbl'>현재 평가금</div><div class='fa-mini-kpi-val fa-font-bold'>{eval_val:,.0f}</div></div>" if eval_val is not None else "",
            f"<div class='fa-mini-kpi'><div class='fa-mini-kpi-lbl'>누적 배당금</div><div class='fa-mini-kpi-val' style='color: var(--fa-purple);'>{dividend_val:,.0f}</div></div>" if dividend_val is not None and dividend_val > 0 else "",
            f"<div class='fa-mini-kpi'><div class='fa-mini-kpi-lbl'>투자금 (원금)</div><div class='fa-mini-kpi-val'>{invest_val:,.0f}</div></div>" if invest_val is not None else "",
            f"<div class='fa-mini-kpi'><div class='fa-mini-kpi-lbl'>투자 대비 수익</div><div class='fa-mini-kpi-val {inv_p_cls}'>{invest_profit:+,.0f} <span class='fa-badge {inv_p_bdg}'>{invest_rate:+.2f}</span></div></div>" if invest_profit is not None else "",
            f"<div class='fa-mini-kpi'><div class='fa-mini-kpi-lbl'>총 매수금</div><div class='fa-mini-kpi-val'>{buy_val:,.0f}</div></div>" if buy_val > 0 else "",
            f"<div class='fa-mini-kpi'><div class='fa-mini-kpi-lbl'>매수 대비 수익</div><div class='fa-mini-kpi-val {buy_p_cls}'>{buy_profit:+,.0f} <span class='fa-badge {buy_p_bdg}'>{buy_rate:+.2f}</span></div></div>" if buy_val > 0 else "",
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
                        hovertemplate="<b>%{label}</b><br>평가금: %{value:,.0f}<br>비중: %{percent}<extra></extra>",
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

            # 종목별 투자금 (사용자가 직접 입력한 투자금 deposit 합계)
            user_inv_amt = inv_by_acct_sym.get((account, sym), 0.0)
            if user_inv_amt == 0.0 and account == "sema" and invest_val is not None:
                user_inv_amt = invest_val

            if user_inv_amt > 0:
                inv_p_amt = e_amt - user_inv_amt
                inv_r_rate = (inv_p_amt / user_inv_amt * 100.0)
                inv_p_cls = "fa-num-positive" if inv_p_amt > 0 else "fa-num-negative" if inv_p_amt < 0 else ""
                inv_p_bdg = "fa-badge-positive" if inv_p_amt > 0 else "fa-badge-negative" if inv_p_amt < 0 else "fa-badge-neutral"
                inv_str = f"{user_inv_amt:,.0f}"
                inv_p_str = f"{inv_p_amt:+,.0f}" if inv_p_amt != 0 else "0"
                inv_r_str = f"{inv_r_rate:+.2f}"
                inv_badge_html = f"<span class='fa-badge {inv_p_bdg}'>투 {inv_r_str}</span>"
            else:
                inv_p_amt = None
                inv_r_rate = None
                inv_p_cls = ""
                inv_p_bdg = "fa-badge-neutral"
                inv_str = "-"
                inv_p_str = "-"
                inv_r_str = "-"
                inv_badge_html = f"<span class='fa-badge fa-badge-neutral'>투 -</span>"

            cum_div = div_by_acct_sym.get((account, sym), 0.0)
            if cum_div == 0.0 and account == "sema" and dividend_val is not None:
                cum_div = dividend_val
            div_str = f"{cum_div:,.0f}"

            buy_p_cls = "fa-num-positive" if buy_p_amt > 0 else "fa-num-negative" if buy_p_amt < 0 else ""
            buy_p_bdg = "fa-badge-positive" if buy_p_amt > 0 else "fa-badge-negative" if buy_p_amt < 0 else "fa-badge-neutral"

            b_str = f"{b_amt:,.0f}"
            e_str = f"{e_amt:,.0f}"
            buy_p_str = f"{buy_p_amt:+,.0f}" if buy_p_amt != 0 else "0"
            buy_r_str = f"{buy_r_rate:+.2f}"

            stock_cards.append(
                f"<div class='fa-stock-card'>"
                f"  <div class='fa-stock-card-head'>"
                f"    <div class='fa-stock-card-title'>{html.escape(sym)}</div>"
                f"    <div class='fa-stock-card-badges'>"
                f"      {inv_badge_html}"
                f"      <span class='fa-rate-divider'>/</span>"
                f"      <span class='fa-badge {buy_p_bdg}'>매 {buy_r_str}</span>"
                f"    </div>"
                f"  </div>"
                f"  <div class='fa-stock-card-body'>"
                f"    <div class='fa-stock-field'>"
                f"      <span class='fa-stock-lbl'>평가금</span>"
                f"      <span class='fa-stock-val fa-font-bold'>{e_str}</span>"
                f"    </div>"
                f"    <div class='fa-stock-field'>"
                f"      <span class='fa-stock-lbl'>배당금</span>"
                f"      <span class='fa-stock-val' style='color: var(--fa-purple);'>{div_str}</span>"
                f"    </div>"
                f"    <div class='fa-stock-field'>"
                f"      <span class='fa-stock-lbl'>투자금</span>"
                f"      <span class='fa-stock-val'>{inv_str}</span>"
                f"    </div>"
                f"    <div class='fa-stock-field'>"
                f"      <span class='fa-stock-lbl'>투자 수익금</span>"
                f"      <span class='fa-stock-val {inv_p_cls}'>{inv_p_str}</span>"
                f"    </div>"
                f"    <div class='fa-stock-field'>"
                f"      <span class='fa-stock-lbl'>매수금</span>"
                f"      <span class='fa-stock-val'>{b_str}</span>"
                f"    </div>"
                f"    <div class='fa-stock-field'>"
                f"      <span class='fa-stock-lbl'>매수 수익금</span>"
                f"      <span class='fa-stock-val {buy_p_cls}'>{buy_p_str}</span>"
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
                        f"<div class='fa-rebal-val'>+{diff:,.0f}{qty_str}</div>"
                        f"</div>"
                    )
                elif diff < -100:
                    qty_str = f" ({diff_q:g}주)" if diff_q < 0 else ""
                    rebal_cards.append(
                        f"<div class='fa-rebal-item sell'>"
                        f"<div class='fa-rebal-tag sell'>매도 필요</div>"
                        f"<div class='fa-rebal-name'>{html.escape(asset_name)}</div>"
                        f"<div class='fa-rebal-val'>-{abs(diff):,.0f}{qty_str}</div>"
                        f"</div>"
                    )
                else:
                    rebal_cards.append(
                        f"<div class='fa-rebal-item ok'>"
                        f"<div class='fa-rebal-tag ok'>비중 적정</div>"
                        f"<div class='fa-rebal-name'>{html.escape(asset_name)}</div>"
                        f"<div class='fa-rebal-val'>0 (목표 유지)</div>"
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
        return f"{val:,.0f}"

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
        f"  <div class='fa-kpi-value'>{invest_amt:,.0f}</div>"
        f"  <div class='fa-kpi-sub'>원금 입금</div>"
        f"</div>",
        f"<div class='fa-kpi-card'>"
        f"  <div class='fa-kpi-label'>총 매수금</div>"
        f"  <div class='fa-kpi-value fa-num-positive'>{buy_amt:,.0f}</div>"
        f"  <div class='fa-kpi-sub'>매수 체결</div>"
        f"</div>",
        f"<div class='fa-kpi-card'>"
        f"  <div class='fa-kpi-label'>총 매도금</div>"
        f"  <div class='fa-kpi-value fa-num-negative'>{sell_amt:,.0f}</div>"
        f"  <div class='fa-kpi-sub'>매도 체결</div>"
        f"</div>",
        f"<div class='fa-kpi-card'>"
        f"  <div class='fa-kpi-label'>총 배당금</div>"
        f"  <div class='fa-kpi-value' style='color: var(--fa-purple);'>{div_amt:,.0f}</div>"
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

    fig_invest_1y = _build_assets_investment_trend(data.account_df, data.invest_series, period="1Y")
    fig_invest_all = _build_assets_investment_trend(data.account_df, data.invest_series, period="ALL")
    invest_trend_card_html = _render_trend_tab_card(
        update_fa.ACCOUNT_TITLES.get("title_assets_investment_trend", "누적 투자금 vs 평가금 추세"),
        fig_invest_1y,
        fig_invest_all,
        "fa-invest-trend-tabs",
    )

    fig_assets_1y = _build_assets_trend(data.account_df, period="1Y")
    fig_assets_all = _build_assets_trend(data.account_df, period="ALL")
    assets_trend_card_html = _render_trend_tab_card(
        update_fa.ACCOUNT_TITLES.get("title_assets_trend", "전체 금융자산 추이"),
        fig_assets_1y,
        fig_assets_all,
        "fa-assets-trend-tabs",
    )

    portfolio_alloc_html = _build_portfolio_allocation_section(data.holdings_df, data.symbol_map, fig_html)
    account_summary_html = _build_account_assets_html_table(data.summary_df)
    holdings_section_html = _build_total_holdings_section(data.holdings_df)

    rebal_alert_html = _build_rebalancing_alert_banner(data, threshold_pct=3.0)
    dividends_section_html = _build_dividends_tabbed_section(data.records, data.fx_series_full, fig_html)
    trading_summary, trading_items = _build_trading_history(data.records, data.fx_series_month, data.month_end)
    account_detail_section_html = _build_account_detail_section(data, fig_html)

    now_str = datetime.now().strftime("%Y-%m-%d %H:%M")
    blocks: List[str] = [
        "<section class=\"fa-hero\">"
        "<div class=\"fa-hero-header\">"
        "  <div>"
        f"    <div class=\"fa-hero-title\">{html.escape(data.month_end.strftime('%Y년 %m월 자산 대시보드'))} <span class=\"fa-badge fa-badge-neutral\" style=\"font-size:0.75rem; vertical-align:middle; margin-left:6px;\">{APP_VERSION}</span></div>"
        f"    <div class=\"fa-hero-meta\">최종 업데이트: {now_str}</div>"
        "  </div>"
        "  <div class=\"fa-hero-actions\">"
        "    <button type=\"button\" class=\"fa-btn-hero fa-btn-hero-secondary\" onclick=\"triggerDashboardRefresh(this)\">"
        "      <span class=\"fa-refresh-icon\">🔄</span>"
        "      <span class=\"fa-refresh-text\">대시보드 갱신</span>"
        "    </button>"
        "    <a href=\"https://fa-admin.vividian.net\" class=\"fa-btn-hero fa-btn-hero-primary\" target=\"_blank\" rel=\"noopener noreferrer\">"
        "      <span>⚙️ 거래내역 관리</span>"
        "      <span class=\"fa-btn-arrow\">↗</span>"
        "    </a>"
        "  </div>"
        "</div>"
        "</section>",
    ]

    if rebal_alert_html:
        blocks.append(rebal_alert_html)

    blocks.extend([
        _build_kpi_row(data),
        _build_market_kpi_row(),
        invest_trend_card_html,
        assets_trend_card_html,
        portfolio_alloc_html,
        account_summary_html,
        holdings_section_html,
    ])

    if dividends_section_html:
        blocks.append(dividends_section_html)

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
  max-width: 1000px;
  width: 100%;
  margin: 0 auto;
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
.fa-hero-actions {
  display: flex;
  align-items: center;
  gap: 10px;
  flex-wrap: wrap;
}
.fa-btn-hero {
  display: inline-flex !important;
  align-items: center !important;
  justify-content: center !important;
  gap: 6px !important;
  height: 38px !important;
  padding: 0 16px !important;
  border-radius: 8px !important;
  font-size: 0.85rem !important;
  font-weight: 700 !important;
  line-height: 1 !important;
  text-decoration: none !important;
  transition: all 0.2s ease !important;
  cursor: pointer !important;
  box-sizing: border-box !important;
  font-family: inherit !important;
  outline: none !important;
}
.fa-btn-hero-primary {
  background: #4f46e5 !important;
  color: #ffffff !important;
  border: 1px solid #4338ca !important;
  box-shadow: 0 2px 8px rgba(79, 70, 229, 0.3) !important;
}
.fa-btn-hero-primary:hover {
  background: #4338ca !important;
  color: #ffffff !important;
  transform: translateY(-1px);
  box-shadow: 0 4px 12px rgba(79, 70, 229, 0.45) !important;
}
.fa-btn-hero-secondary {
  background: #ffffff !important;
  color: #4f46e5 !important;
  border: 1px solid #c7d2fe !important;
  box-shadow: 0 2px 6px rgba(0, 0, 0, 0.05) !important;
}
.fa-btn-hero-secondary:hover {
  background: #eef2ff !important;
  border-color: #a5b4fc !important;
  color: #4338ca !important;
  transform: translateY(-1px);
  box-shadow: 0 4px 10px rgba(79, 70, 229, 0.15) !important;
}
.fa-btn-hero.loading {
  opacity: 0.8 !important;
  cursor: wait !important;
  transform: none !important;
}
.fa-btn-arrow {
  font-size: 0.85rem;
  font-weight: 800;
  margin-left: 1px;
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

.fa-mobile-only,
.fa-mobile-inline {
  display: none;
}

/* =========================================================
   Mobile Responsive Transformation (Table -> Card View)
   ========================================================= */
@media (max-width: 768px) {
  .fa-mobile-only {
    display: inline-block !important;
  }
  .fa-mobile-inline {
    display: inline-flex !important;
  }
  .fa-hide-mobile {
    display: none !important;
  }

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

  /* 보유 종목 카드 내부 4행 컴팩트 2열 레이아웃 */
  .fa-table-holdings tr {
    display: grid !important;
    grid-template-columns: 1fr 1fr !important;
    gap: 6px 14px !important;
    padding: 12px 14px !important;
    align-items: center !important;
  }
  .fa-table-holdings td[data-label='계좌'] {
    display: none !important;
  }
  .fa-table-holdings td[data-label='종목'] {
    grid-column: 1 / -1 !important;
    grid-row: 1 / 2;
    display: flex !important;
    justify-content: space-between !important;
    align-items: center !important;
    width: 100% !important;
    padding: 0 !important;
    border-bottom: 1px dashed var(--fa-border) !important;
    padding-bottom: 8px !important;
    margin-bottom: 2px !important;
  }
  .fa-table-holdings td[data-label='종목'] .fa-stock-title-wrap {
    display: flex;
    align-items: center;
    gap: 6px;
    font-size: 0.98rem;
    font-weight: 700;
  }
  .fa-table-holdings td[data-label='종목'] .fa-stock-badges-wrap {
    display: flex !important;
    align-items: center;
    gap: 4px;
    flex-wrap: wrap;
  }
  .fa-table-holdings td[data-label='종목'] .fa-stock-rate-badge,
  .fa-table-holdings td[data-label='종목'] .fa-stock-fluct-badge {
    font-size: 0.76rem;
    padding: 2px 6px;
    font-weight: 600;
    white-space: nowrap;
    border-radius: 6px;
  }
  .fa-table-holdings td[data-label='종목']::before {
    display: none !important;
  }

  .fa-table-holdings td[data-label='수익률'] {
    display: none !important;
  }

  .fa-table-holdings td[data-label='수량'] {
    grid-column: 1 / 2;
    grid-row: 2 / 3;
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 3px 0 !important;
    border-bottom: none !important;
    font-size: 0.86rem;
  }

  .fa-table-holdings td[data-label='평단가'] {
    grid-column: 2 / 3;
    grid-row: 2 / 3;
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 3px 0 !important;
    border-bottom: none !important;
    font-size: 0.86rem;
  }

  .fa-table-holdings td[data-label='현재가'] {
    grid-column: 1 / 2;
    grid-row: 3 / 4;
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 3px 0 !important;
    border-bottom: none !important;
    font-size: 0.86rem;
  }

  .fa-table-holdings td[data-label='매수금'] {
    grid-column: 2 / 3;
    grid-row: 3 / 4;
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 3px 0 !important;
    border-bottom: none !important;
    font-size: 0.86rem;
  }

  .fa-table-holdings td[data-label='평가금'] {
    grid-column: 1 / 2;
    grid-row: 4 / 5;
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 3px 0 !important;
    border-bottom: none !important;
    font-size: 0.86rem;
  }

  .fa-table-holdings td[data-label='수익금'] {
    grid-column: 2 / 3;
    grid-row: 4 / 5;
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 3px 0 !important;
    border-bottom: none !important;
    font-size: 0.86rem;
  }
}

/* Single Card Box (계좌별 1개 통합 카드 및 요약 카드 박스) */
.fa-single-card-box {
  background: var(--fa-kpi-bg);
  border: 1px solid var(--fa-card-border);
  border-radius: 12px;
  padding: 16px 20px;
  box-shadow: 0 1px 3px rgba(0, 0, 0, 0.02);
}
.fa-single-card-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 14px;
  padding-bottom: 10px;
  border-bottom: 1px solid var(--fa-border);
}
.fa-single-card-title {
  font-size: 1.05rem;
  font-weight: 700;
  color: var(--fa-text-main);
}
.fa-stat-line-group {
  display: flex;
  flex-direction: column;
  gap: 10px;
}
.fa-stat-line {
  display: flex;
  justify-content: space-between;
  align-items: center;
  font-size: 0.92rem;
}
.fa-stat-lbl {
  color: var(--fa-text-muted);
  font-weight: 500;
}
.fa-stat-val {
  font-weight: 600;
  font-variant-numeric: tabular-nums;
  color: var(--fa-text-main);
}

.fa-table-eok-summary {
  width: 100%;
  border-collapse: collapse;
  font-size: 0.9rem;
}
.fa-table-eok-summary th {
  background: var(--fa-table-header-bg);
  color: var(--fa-text-muted);
  font-weight: 600;
  padding: 10px 14px;
  border-bottom: 1px solid var(--fa-border);
  white-space: nowrap;
}
.fa-table-eok-summary td {
  padding: 12px 14px;
  border-bottom: 1px solid var(--fa-border);
  white-space: nowrap;
}
.fa-table-eok-summary tfoot tr {
  background: var(--fa-kpi-bg);
  font-weight: 700;
}
.fa-table-eok-summary th:first-child,
.fa-table-eok-summary td:first-child {
  position: sticky;
  left: 0;
  background: var(--fa-kpi-bg);
  z-index: 2;
  box-shadow: 2px 0 6px rgba(0, 0, 0, 0.05);
}
.fa-table-eok-summary th:first-child {
  background: var(--fa-table-header-bg);
  z-index: 3;
}
.fa-table-eok-summary tfoot td:first-child {
  background: var(--fa-kpi-bg);
  z-index: 2;
}

/* =========================================================
   Holdings Summary Card Grid (요약 탭 카드 뷰)
   ========================================================= */
.fa-holdings-summary-grid {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(290px, 1fr));
  gap: 12px;
}
@media (max-width: 640px) {
  .fa-holdings-summary-grid {
    grid-template-columns: 1fr;
    gap: 10px;
  }
}
.fa-holding-card {
  background: var(--fa-kpi-bg);
  border: 1px solid var(--fa-border);
  border-radius: 12px;
  padding: 12px 14px;
  display: flex;
  flex-direction: column;
  gap: 10px;
  transition: transform 0.2s ease, box-shadow 0.2s ease;
}
.fa-holding-card:hover {
  transform: translateY(-2px);
  box-shadow: var(--fa-card-shadow);
}
.fa-holding-card-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  gap: 8px;
  border-bottom: 1px dashed var(--fa-border);
  padding-bottom: 8px;
}
.fa-holding-card-title {
  display: flex;
  align-items: center;
  gap: 6px;
  min-width: 0;
}
.fa-holding-title-text {
  font-size: 0.95rem;
  font-weight: 700;
  color: var(--fa-text-main);
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
}
.fa-holding-card-badges {
  display: flex;
  align-items: center;
  gap: 4px;
  flex-shrink: 0;
}
.fa-holding-card-badges .fa-badge {
  font-size: 0.74rem;
  padding: 2px 6px;
  font-weight: 600;
  white-space: nowrap;
  border-radius: 6px;
}
.fa-holding-card-metrics {
  display: grid;
  grid-template-columns: repeat(3, minmax(0, 1fr));
  gap: 6px;
}
.fa-holding-metric {
  display: flex;
  flex-direction: column;
  gap: 2px;
}
.fa-holding-metric-lbl {
  font-size: 0.72rem;
  color: var(--fa-text-muted);
  font-weight: 500;
}
.fa-holding-metric-val {
  font-size: 0.88rem;
  font-weight: 600;
  font-variant-numeric: tabular-nums;
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
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
  width: 100%;
  max-width: 100%;
  overflow: hidden;
}
.fa-account-chart-card {
  background: var(--fa-kpi-bg);
  border: 1px solid var(--fa-border);
  border-radius: 12px;
  padding: 10px;
  width: 100%;
  box-sizing: border-box;
  overflow: hidden;
}
.fa-account-chart-card .plotly-graph-div {
  width: 100% !important;
  margin: 0 auto;
}
.fa-account-table-col {
  background: transparent;
  width: 100%;
  max-width: 100%;
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
  gap: 4px;
  flex-shrink: 0;
}
.fa-rate-divider {
  font-size: 0.76rem;
  color: var(--fa-text-muted);
  font-weight: 600;
  margin: 0 1px;
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
  gap: 6px 16px;
}
.fa-stock-field {
  display: flex;
  justify-content: space-between;
  align-items: center;
  gap: 6px;
  padding: 3px 0;
  border-bottom: 1px dashed rgba(148, 163, 184, 0.15);
}
.fa-stock-field:nth-last-child(-n+2) {
  border-bottom: none;
}
.fa-stock-lbl {
  font-size: 0.76rem;
  color: var(--fa-text-muted);
  white-space: nowrap;
}
.fa-stock-val {
  font-size: 0.88rem;
  font-weight: 600;
  font-variant-numeric: tabular-nums;
  text-align: right;
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

/* Dividend Detail Tab Controls */
.fa-div-detail-wrap {
  width: 100%;
}
.fa-div-ctrl-bar {
  display: flex;
  align-items: center;
  gap: 10px;
  margin-bottom: 14px;
  background: var(--fa-kpi-bg);
  border: 1px solid var(--fa-border);
  border-radius: 10px;
  padding: 8px 14px;
}
.fa-div-select-lbl {
  font-size: 0.84rem;
  font-weight: 600;
  color: var(--fa-text-muted);
}
.fa-select {
  padding: 5px 12px;
  border-radius: 8px;
  border: 1px solid var(--fa-border);
  background: var(--fa-card-bg);
  color: var(--fa-text-main);
  font-size: 0.88rem;
  font-weight: 600;
  cursor: pointer;
  outline: none;
}
.fa-div-year-pane {
  display: none;
  animation: faFadeIn 0.25s ease-in-out;
}
.fa-div-year-pane.active {
  display: block;
}
.fa-div-year-summary {
  display: flex;
  align-items: center;
  justify-content: space-between;
  background: var(--fa-kpi-bg);
  border: 1px solid var(--fa-border);
  border-radius: 10px;
  padding: 10px 14px;
  margin-bottom: 12px;
}
.fa-div-summary-tag {
  font-size: 0.84rem;
  font-weight: 600;
  color: var(--fa-text-muted);
}
.fa-div-summary-val {
  font-size: 1.05rem;
  font-weight: 700;
  color: var(--fa-purple);
  font-variant-numeric: tabular-nums;
}
.fa-div-chart-box {
  overflow: hidden;
  border-radius: 12px;
  background: var(--fa-kpi-bg);
  border: 1px solid var(--fa-border);
  padding: 10px;
}
.fa-div-chart-box .plotly-graph-div {
  width: 100% !important;
  margin: 0 auto;
}

/* =========================================================
   Rebalancing Alert Banner Styles
   ========================================================= */
.fa-rebal-alert-card {
  background: linear-gradient(135deg, #ffffff 0%, #fffcf5 100%);
  border: 1px solid #fed7aa !important;
  box-shadow: 0 4px 14px rgba(249, 115, 22, 0.08) !important;
  margin-bottom: 16px;
}
.fa-rebal-alert-top-bar {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 12px;
}
.fa-rebal-alert-title-wrap {
  display: flex;
  align-items: center;
  gap: 12px;
}
.fa-rebal-alert-icon {
  font-size: 1.5rem;
  background: #ffedd5;
  border-radius: 10px;
  padding: 6px 10px;
  display: inline-flex;
  align-items: center;
  justify-content: center;
  flex-shrink: 0;
}
.fa-rebal-alert-main-title {
  font-size: 1.05rem;
  font-weight: 800;
  color: #9a3412;
  display: flex;
  align-items: center;
  flex-wrap: wrap;
  gap: 6px;
}
.fa-rebal-alert-sub-title {
  font-size: 0.82rem;
  color: var(--fa-text-muted);
  margin-top: 2px;
}
.fa-rebal-alert-grid {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(260px, 1fr));
  gap: 10px;
}
@media (max-width: 600px) {
  .fa-rebal-alert-grid {
    grid-template-columns: 1fr;
    gap: 8px;
  }
}
.fa-rebal-alert-item {
  background: #ffffff;
  border: 1px solid var(--fa-card-border);
  border-radius: 10px;
  padding: 10px 12px;
  display: flex;
  flex-direction: column;
  gap: 8px;
  box-shadow: 0 1px 3px rgba(0,0,0,0.03);
}
.fa-rebal-alert-item.buy {
  border-left: 4px solid var(--fa-gain);
}
.fa-rebal-alert-item.sell {
  border-left: 4px solid var(--fa-loss);
}
.fa-rebal-alert-head {
  display: flex;
  justify-content: space-between;
  align-items: center;
}
.fa-rebal-alert-sym-wrap {
  display: flex;
  align-items: center;
  gap: 6px;
  min-width: 0;
}
.fa-rebal-alert-name {
  font-size: 0.92rem;
  font-weight: 700;
  color: var(--fa-text-main);
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
}
.fa-rebal-alert-body {
  display: flex;
  flex-direction: column;
  gap: 5px;
  background: var(--fa-kpi-bg);
  padding: 8px 10px;
  border-radius: 8px;
  font-size: 0.82rem;
}
.fa-rebal-alert-stat {
  display: flex;
  justify-content: space-between;
  align-items: center;
}
.fa-rebal-alert-lbl {
  color: var(--fa-text-muted);
  white-space: nowrap;
  flex-shrink: 0;
}
.fa-rebal-alert-val {
  font-variant-numeric: tabular-nums;
  font-weight: 600;
  white-space: nowrap;
}

/* =========================================================
   Responsive Overrides for Desktop vs Mobile
   ========================================================= */
/* 1. 전체 포트폴리오 비중: PC 3열 나란히 표시 (탭 제거), 모바일 탭 분기 */
@media (min-width: 769px) {
  .fa-alloc-card .fa-alloc-tab-nav {
    display: none !important;
  }
  .fa-alloc-card .fa-alloc-body {
    display: grid !important;
    grid-template-columns: repeat(3, minmax(0, 1fr)) !important;
    gap: 16px !important;
  }
  .fa-alloc-card .fa-tab-pane {
    display: block !important;
  }
}
@media (max-width: 768px) {
  .fa-alloc-card .fa-tab-pane {
    display: none;
  }
  .fa-alloc-card .fa-tab-pane.active {
    display: block;
  }
}

/* 2. 계좌별 자산 현황: PC 전체 요약 테이블 표시 (탭 제거), 모바일 탭 분기 */
@media (min-width: 769px) {
  .fa-acct-summary-card .fa-acct-summary-tab-nav {
    display: none !important;
  }
  .fa-acct-summary-card #acct-sum-tab-all {
    display: block !important;
  }
  .fa-acct-summary-card .fa-tab-pane:not(#acct-sum-tab-all) {
    display: none !important;
  }
}
@media (max-width: 768px) {
  .fa-acct-summary-card .fa-acct-summary-tab-nav {
    display: block !important;
  }
  .fa-acct-summary-card .fa-tab-pane {
    display: none;
  }
  .fa-acct-summary-card .fa-tab-pane.active {
    display: block;
  }
}

/* 3. 실시간 보유종목 현황: PC 상세 테이블 고정(탭 제거), 모바일 탭 분기(기본: 요약) */
@media (min-width: 769px) {
  .fa-holdings-card .fa-holdings-tab-nav {
    display: none !important;
  }
  .fa-holdings-card #fa-holdings-tab-summary {
    display: none !important;
  }
  .fa-holdings-card #fa-holdings-tab-detail {
    display: block !important;
  }
}
@media (max-width: 768px) {
  .fa-holdings-card .fa-holdings-tab-nav {
    display: flex !important;
  }
  .fa-holdings-card .fa-tab-pane {
    display: none;
  }
  .fa-holdings-card .fa-tab-pane.active {
    display: block;
  }
}

/* 요약 테이블 스타일 및 가로 스크롤 시 종목명 고정 (Sticky Column) */
.fa-table-wrapper-sticky {
  overflow-x: auto;
  -webkit-overflow-scrolling: touch;
  position: relative;
}
.fa-table-holdings-summary {
  min-width: 440px;
  border-collapse: separate;
  border-spacing: 0;
}
.fa-table-holdings-summary th,
.fa-table-holdings-summary td {
  padding: 10px 12px;
  font-size: 0.88rem;
  white-space: nowrap;
  border-bottom: 1px solid var(--fa-border);
}
.fa-table-holdings-summary th.fa-th-account,
.fa-table-holdings-summary td.fa-col-account {
  position: sticky;
  left: 0;
  z-index: 2;
  background: var(--fa-card-bg);
  width: 60px;
}
.fa-table-holdings-summary th.fa-th-account {
  background: var(--fa-table-header-bg);
  z-index: 3;
}
.fa-table-holdings-summary th.fa-th-symbol,
.fa-table-holdings-summary td.fa-col-sticky-symbol {
  position: sticky;
  left: 60px;
  z-index: 2;
  background: var(--fa-card-bg);
  box-shadow: 4px 0 6px -2px rgba(0, 0, 0, 0.08);
}
.fa-table-holdings-summary th.fa-th-symbol {
  background: var(--fa-table-header-bg);
  z-index: 3;
  box-shadow: 4px 0 6px -2px rgba(0, 0, 0, 0.08);
}

.fa-spin {
  display: inline-block;
  animation: faRotate 0.8s linear infinite;
}
@keyframes faRotate {
  from { transform: rotate(0deg); }
  to { transform: rotate(360deg); }
}

/* 전일대비 변동 상세 모달 */
.fa-modal-overlay {
  position: fixed;
  top: 0;
  left: 0;
  right: 0;
  bottom: 0;
  background: rgba(15, 23, 42, 0.65);
  backdrop-filter: blur(4px);
  display: none;
  align-items: center;
  justify-content: center;
  z-index: 99999;
  padding: 16px;
}
.fa-modal-overlay.active {
  display: flex;
}
.fa-modal-card {
  background: var(--fa-card-bg);
  border: 1px solid var(--fa-card-border);
  border-radius: 16px;
  width: 100%;
  max-width: 680px;
  max-height: 85vh;
  display: flex;
  flex-direction: column;
  box-shadow: 0 20px 40px -8px rgba(0, 0, 0, 0.35);
  overflow: hidden;
  animation: faModalPop 0.2s cubic-bezier(0.16, 1, 0.3, 1);
}
@keyframes faModalPop {
  from { opacity: 0; transform: scale(0.96) translateY(10px); }
  to { opacity: 1; transform: scale(1) translateY(0); }
}
.fa-modal-header {
  padding: 16px 20px;
  border-bottom: 1px solid var(--fa-border);
  display: flex;
  justify-content: space-between;
  align-items: center;
  background: var(--fa-card-bg);
}
.fa-modal-title {
  font-size: 1.05rem;
  font-weight: 700;
  margin: 0;
  color: var(--fa-text-main);
}
.fa-modal-close {
  background: transparent;
  border: none;
  font-size: 1.25rem;
  color: var(--fa-text-muted);
  cursor: pointer;
  padding: 4px 8px;
  border-radius: 6px;
  transition: all 0.15s;
}
.fa-modal-close:hover {
  background: var(--fa-table-hover);
  color: var(--fa-text-main);
}
.fa-modal-body {
  padding: 16px 20px;
  overflow-y: auto;
  -webkit-overflow-scrolling: touch;
}
.fa-modal-summary-grid {
  display: grid;
  grid-template-columns: repeat(3, 1fr);
  gap: 10px;
  margin-bottom: 16px;
}
@media (max-width: 540px) {
  .fa-modal-summary-grid { grid-template-columns: 1fr; }
}
.fa-modal-stat-box {
  background: var(--fa-table-header-bg);
  border: 1px solid var(--fa-border);
  border-radius: 10px;
  padding: 10px 14px;
}
.fa-modal-stat-lbl {
  font-size: 0.76rem;
  color: var(--fa-text-muted);
  font-weight: 600;
}
.fa-modal-stat-val {
  font-size: 1.05rem;
  font-weight: 800;
  margin-top: 4px;
}
.fa-table-modal-detail th,
.fa-table-modal-detail td {
  padding: 8px 10px;
  font-size: 0.86rem;
  white-space: nowrap;
}

.period-tabs { display: flex; gap: 8px; margin-bottom: 14px; flex-wrap: wrap; }
.period-tabs .tab-btn {
  background: var(--fa-table-header-bg); border: 1px solid var(--fa-border); color: var(--fa-text-muted);
  padding: 6px 14px; border-radius: 6px; cursor: pointer; font-size: 0.85rem; font-weight: 600; transition: all 0.15s;
}
.period-tabs .tab-btn:hover { background: var(--fa-table-hover); color: var(--fa-text-main); }
.period-tabs .tab-btn.active { background: #3182ce; border-color: #3182ce; color: #ffffff; }

.fa-empty-text { color: var(--fa-text-muted); font-size: 0.9rem; margin: 8px 0; }

/* =========================================================
   대시보드 갱신 전면 모달 스타일
   ========================================================= */
.fa-refresh-overlay {
  background: rgba(15, 23, 42, 0.72) !important;
  backdrop-filter: blur(6px) !important;
}
.fa-refresh-modal-card {
  max-width: 520px;
  background: var(--fa-card-bg);
  border: 1px solid var(--fa-card-border);
  border-radius: 18px;
  box-shadow: 0 25px 50px -12px rgba(0, 0, 0, 0.4), 0 0 0 1px rgba(255, 255, 255, 0.05);
  overflow: hidden;
  padding: 0;
}
.fa-refresh-modal-header {
  padding: 20px 24px 18px;
  background: linear-gradient(135deg, rgba(79, 70, 229, 0.08) 0%, rgba(59, 130, 246, 0.04) 100%);
  border-bottom: 1px solid var(--fa-border);
  display: flex;
  justify-content: space-between;
  align-items: flex-start;
}
.fa-refresh-header-left {
  display: flex;
  align-items: center;
  gap: 14px;
}
.fa-refresh-pulse-icon {
  width: 44px;
  height: 44px;
  border-radius: 12px;
  background: #4f46e5;
  color: #ffffff;
  display: flex;
  align-items: center;
  justify-content: center;
  font-size: 1.3rem;
  box-shadow: 0 4px 14px rgba(79, 70, 229, 0.35);
  flex-shrink: 0;
}
.fa-refresh-icon-spin {
  display: inline-block;
  animation: faSpinPulse 2s linear infinite;
}
@keyframes faSpinPulse {
  0% { transform: rotate(0deg); }
  100% { transform: rotate(360deg); }
}
.fa-refresh-modal-title {
  font-size: 1.15rem;
  font-weight: 700;
  color: var(--fa-text-main);
  margin: 0 0 4px 0;
  line-height: 1.3;
}
.fa-refresh-modal-desc {
  font-size: 0.82rem;
  color: var(--fa-text-muted);
  margin: 0;
}
.fa-refresh-modal-body {
  padding: 22px 24px 24px;
}

/* 프로그레스 바 */
.fa-refresh-progress-section {
  margin-bottom: 20px;
}
.fa-refresh-progress-meta {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 8px;
}
.fa-refresh-badge {
  font-size: 0.78rem;
  font-weight: 700;
  padding: 3px 10px;
  border-radius: 9999px;
  background: var(--fa-accent-bg, #eef2ff);
  color: var(--fa-accent, #4f46e5);
}
.fa-refresh-percent {
  font-size: 0.95rem;
  font-weight: 800;
  color: var(--fa-text-main);
  font-feature-settings: "tnum";
}
.fa-refresh-progress-track {
  width: 100%;
  height: 8px;
  background: var(--fa-table-header-bg, #f1f5f9);
  border-radius: 9999px;
  overflow: hidden;
}
.fa-refresh-progress-fill {
  height: 100%;
  background: linear-gradient(90deg, #4f46e5 0%, #3b82f6 50%, #06b6d4 100%);
  border-radius: 9999px;
  transition: width 0.4s cubic-bezier(0.4, 0, 0.2, 1);
  box-shadow: 0 0 10px rgba(59, 130, 246, 0.4);
}

/* 스텝 리스트 */
.fa-refresh-steps-list {
  display: flex;
  flex-direction: column;
  gap: 12px;
  margin-bottom: 20px;
}
.fa-refresh-step-item {
  display: flex;
  align-items: flex-start;
  gap: 14px;
  padding: 12px 14px;
  border-radius: 12px;
  background: var(--fa-kpi-bg, #f8fafc);
  border: 1px solid var(--fa-border, #e2e8f0);
  transition: all 0.25s ease;
  opacity: 0.55;
}
.fa-refresh-step-item.active {
  opacity: 1;
  background: var(--fa-card-bg);
  border-color: #4f46e5;
  box-shadow: 0 4px 12px rgba(79, 70, 229, 0.08);
}
.fa-refresh-step-item.done {
  opacity: 0.9;
  border-color: #10b981;
  background: rgba(16, 185, 129, 0.04);
}
.fa-step-indicator {
  width: 24px;
  height: 24px;
  border-radius: 50%;
  display: flex;
  align-items: center;
  justify-content: center;
  flex-shrink: 0;
  margin-top: 1px;
  position: relative;
}
.fa-step-dot {
  width: 8px;
  height: 8px;
  border-radius: 50%;
  background: var(--fa-text-sub, #94a3b8);
  display: block;
}
.fa-step-spinner {
  display: none;
  width: 16px;
  height: 16px;
  border: 2px solid #e0e7ff;
  border-top-color: #4f46e5;
  border-radius: 50%;
  animation: faSpinPulse 0.8s linear infinite;
}
.fa-step-check {
  display: none;
  font-weight: 900;
  font-size: 0.85rem;
  color: #10b981;
}
.fa-refresh-step-item.active .fa-step-dot { display: none; }
.fa-refresh-step-item.active .fa-step-spinner { display: block; }
.fa-refresh-step-item.done .fa-step-dot { display: none; }
.fa-refresh-step-item.done .fa-step-spinner { display: none; }
.fa-refresh-step-item.done .fa-step-check { display: block; }

.fa-step-info {
  flex: 1;
  min-width: 0;
}
.fa-step-name {
  font-size: 0.88rem;
  font-weight: 700;
  color: var(--fa-text-main);
  margin-bottom: 2px;
}
.fa-refresh-step-item.active .fa-step-name {
  color: #4f46e5;
}
.fa-refresh-step-item.done .fa-step-name {
  color: var(--fa-text-main);
}
.fa-step-sub {
  font-size: 0.77rem;
  color: var(--fa-text-muted);
  line-height: 1.35;
}

/* 상태 바 및 타이머 */
.fa-refresh-status-bar {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 10px 14px;
  background: var(--fa-table-header-bg, #f1f5f9);
  border-radius: 10px;
  font-size: 0.82rem;
  color: var(--fa-text-muted);
  gap: 10px;
}
.fa-refresh-current-msg {
  display: flex;
  align-items: center;
  gap: 6px;
  font-weight: 600;
  color: var(--fa-text-main);
  flex: 1;
  min-width: 0;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}
.fa-refresh-timer {
  font-feature-settings: "tnum";
  white-space: nowrap;
  font-size: 0.8rem;
}
.fa-refresh-timer b {
  color: #4f46e5;
}

/* 완료 / 에러 배너 */
.fa-refresh-banner {
  margin-top: 14px;
  padding: 14px 16px;
  border-radius: 12px;
  display: flex;
  align-items: center;
  gap: 12px;
  animation: faModalPop 0.25s ease-out;
}
.fa-refresh-success {
  background: #f0fdf4;
  border: 1px solid #bbf7d0;
  color: #166534;
}
.fa-refresh-error {
  background: #fef2f2;
  border: 1px solid #fecaca;
  color: #991b1b;
}
.fa-banner-icon {
  font-size: 1.5rem;
  line-height: 1;
}
.fa-banner-title {
  font-size: 0.92rem;
  font-weight: 800;
  margin-bottom: 2px;
}
.fa-banner-desc {
  font-size: 0.8rem;
  opacity: 0.9;
}

#fa-div-tab-last12m .plotly-graph-div .bars .point,
#fa-div-tab-monthly .plotly-graph-div .bars .point {
  cursor: pointer !important;
}
</style>

<script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
<script>
window.openModal = function(id) {
  const modal = document.getElementById(id);
  if (modal) {
    modal.classList.add("active");
    document.body.style.overflow = "hidden";
  }
};
window.closeModal = function(id) {
  const modal = document.getElementById(id);
  if (modal) {
    modal.classList.remove("active");
    document.body.style.overflow = "";
  }
};
window.closeAllModals = function() {
  document.querySelectorAll(".fa-modal-overlay.active").forEach(m => {
    if (m.id === "fa-refresh-modal" && !window._canCloseRefreshModal) return;
    m.classList.remove("active");
  });
  if (!document.querySelector(".fa-modal-overlay.active")) {
    document.body.style.overflow = "";
  }
};

window.openEvalChangeModal = function() {
  window.openModal("fa-eval-change-modal");
  setTimeout(() => {
    const modal = document.getElementById("fa-eval-change-modal");
    if (modal) {
      const plots = modal.querySelectorAll(".js-plotly-plot");
      plots.forEach(p => { if (window.Plotly) window.Plotly.Plots.resize(p); });
    }
  }, 60);
};
window.closeEvalChangeModal = function() { window.closeModal("fa-eval-change-modal"); };

window.openDayChangeModal = function() { window.openModal("fa-day-change-modal"); };
window.closeDayChangeModal = function() { window.closeModal("fa-day-change-modal"); };

window.openInvestChangeModal = function() { window.openModal("fa-invest-change-modal"); };
window.closeInvestChangeModal = function() { window.closeModal("fa-invest-change-modal"); };

window.openDividendChangeModal = function() { window.openModal("fa-dividend-change-modal"); };
window.closeDividendChangeModal = function() { window.closeModal("fa-dividend-change-modal"); };

window.openMonthDividendModal = function(monthKey) {
  const details = window.DIVIDEND_MONTH_DETAILS || {};
  const data = details[monthKey];
  if (!data) {
    alert("해당 월의 상세 배당 내역이 없습니다.");
    return;
  }

  const titleEl = document.getElementById("fa-month-div-modal-title");
  if (titleEl) titleEl.innerText = "🎁 " + data.month_label + " 배당 재원 및 수령 상세";

  const totalEl = document.getElementById("fa-month-div-total");
  if (totalEl) totalEl.innerText = Number(data.total_krw).toLocaleString() + "원";

  const countEl = document.getElementById("fa-month-div-count");
  if (countEl) countEl.innerText = data.items.length + "개 종목";

  const topSymEl = document.getElementById("fa-month-div-top-sym");
  if (topSymEl) {
    if (data.top_symbol) {
      topSymEl.innerText = data.top_symbol + " (" + Number(data.top_krw).toLocaleString() + "원)";
    } else {
      topSymEl.innerText = "-";
    }
  }

  const tbody = document.getElementById("fa-month-div-tbody");
  if (tbody) {
    if (data.items && data.items.length > 0) {
      tbody.innerHTML = data.items.map(item => `
        <tr>
          <td class="fa-num">${item.date}</td>
          <td><span class="fa-chip-account">${item.account}</span></td>
          <td><strong>${item.symbol}</strong></td>
          <td class="text-right fa-num fa-font-bold" style="color:var(--fa-purple);">+${Number(item.krw).toLocaleString()}원</td>
          <td class="text-right fa-num">${item.rate.toFixed(1)}%</td>
          <td class="text-right fa-num fa-hide-mobile" style="color:var(--fa-text-muted); font-size:0.82rem;">${item.foreign}</td>
          <td style="color:var(--fa-text-muted); font-size:0.82rem;">${item.memo || "-"}</td>
        </tr>
      `).join("");
    } else {
      tbody.innerHTML = "<tr><td colspan='7' class='text-center fa-empty-text' style='padding:20px;'>배당 수령 내역이 없습니다.</td></tr>";
    }
  }

  window.openModal("fa-month-dividend-modal");
};
window.closeMonthDividendModal = function() { window.closeModal("fa-month-dividend-modal"); };

let currentMarketData = null;
const BACKEND_API_URL = "https://fa-admin.vividian.net";

window.openMarketModal = async function(symbol, title) {
  const titleEl = document.getElementById('modalChartTitle');
  if (titleEl) titleEl.innerText = "📈 " + title + " 10년 추세";
  window.openModal('marketChartModal');

  // 1. 인라인 캐시 데이터가 있으면 즉시 렌더링 (지연시간 0초)
  if (window.MARKET_HISTORY_DATA && window.MARKET_HISTORY_DATA[symbol] && window.MARKET_HISTORY_DATA[symbol].dates.length > 0) {
    currentMarketData = window.MARKET_HISTORY_DATA[symbol];
    document.querySelectorAll('#marketChartModal .period-tabs .tab-btn').forEach(btn => {
      btn.classList.toggle('active', btn.innerText.trim() === '6개월');
    });
    renderMarketPlotlyChart('6M');
    return;
  }

  // 2. 인라인 데이터 부재 시 백엔드 API에서 조회
  const container = document.getElementById('plotlyMarketChart');
  if (container) {
    container.innerHTML = "<div style='text-align:center; padding:140px 0; color:var(--fa-text-muted); font-size:0.95rem;'><span class='fa-spin' style='font-size:1.5rem;'>⏳</span><br><br>과거 시계열 데이터 불러오는 중...</div>";
  }

  document.querySelectorAll('#marketChartModal .period-tabs .tab-btn').forEach(btn => {
    btn.classList.toggle('active', btn.innerText.trim() === '6개월');
  });

  try {
    const res = await fetch(`${BACKEND_API_URL}/api/market/history?symbol=${encodeURIComponent(symbol)}`);
    if (!res.ok) throw new Error("HTTP " + res.status);
    currentMarketData = await res.json();
    renderMarketPlotlyChart('6M');
  } catch (err) {
    if (container) {
      container.innerHTML = "<div style='text-align:center; padding:140px 0; color:var(--fa-loss); font-size:0.95rem;'>⚠️ 데이터를 불러오지 못했습니다.<br><small style='color:var(--fa-text-muted);'>(" + err.message + ")</small></div>";
    }
  }
};

window.changeMarketPeriod = function(period) {
  const map = {'1M':'1개월', '3M':'3개월', '6M':'6개월', '1Y':'1년', '5Y':'5년', '10Y':'10년'};
  document.querySelectorAll('#marketChartModal .period-tabs .tab-btn').forEach(btn => {
    btn.classList.toggle('active', btn.innerText.trim() === map[period]);
  });
  renderMarketPlotlyChart(period);
};

window.renderMarketPlotlyChart = function(period) {
  if (!currentMarketData || !currentMarketData.dates || !currentMarketData.dates.length) return;
  const total = currentMarketData.dates.length;
  let sliceCount = total;
  if (period === '1M') sliceCount = 22;
  else if (period === '3M') sliceCount = 66;
  else if (period === '6M') sliceCount = 130;
  else if (period === '1Y') sliceCount = 252;
  else if (period === '5Y') sliceCount = 252 * 5;
  else if (period === '10Y') sliceCount = total;

  const dates = currentMarketData.dates.slice(-sliceCount);
  const closes = currentMarketData.closes.slice(-sliceCount);
  if (!closes.length) return;

  let maxVal = -Infinity, minVal = Infinity;
  let maxIdx = 0, minIdx = 0;
  for (let i = 0; i < closes.length; i++) {
    const v = closes[i];
    if (v > maxVal) { maxVal = v; maxIdx = i; }
    if (v < minVal) { minVal = v; minIdx = i; }
  }
  const maxDate = dates[maxIdx];
  const minDate = dates[minIdx];

  const firstVal = closes[0] || 0;
  const lastVal = closes[closes.length - 1] || 0;
  const periodReturn = firstVal > 0 ? ((lastVal - firstVal) / firstVal) * 100 : 0;
  const isUp = lastVal >= firstVal;
  const lineColor = isUp ? '#e53e3e' : '#3182ce';

  // 상단 요약 통계 칩 업데이트
  const statsEl = document.getElementById('modalPeriodStats');
  if (statsEl) {
    const retCls = periodReturn >= 0 ? 'fa-num-positive' : 'fa-num-negative';
    statsEl.innerHTML = `
      <span>🔴 최고: <strong class="fa-num-positive">${maxVal.toLocaleString(undefined, {minimumFractionDigits: 2, maximumFractionDigits: 2})}</strong></span>
      <span>🔵 최저: <strong class="fa-num-negative">${minVal.toLocaleString(undefined, {minimumFractionDigits: 2, maximumFractionDigits: 2})}</strong></span>
      <span>기간등락: <strong class="${retCls}">${periodReturn >= 0 ? '+' : ''}${periodReturn.toFixed(2)}%</strong></span>
    `;
  }

  // y축 범위 (최솟값~최댓값 오토스케일 + 12% 여백)
  let ySpan = maxVal - minVal;
  if (ySpan === 0) ySpan = maxVal * 0.05 || 1;
  const yPad = ySpan * 0.12;
  const yMin = minVal - yPad;
  const yMax = maxVal + yPad;

  const mainTrace = {
    name: '종가',
    x: dates,
    y: closes,
    type: 'scatter',
    mode: 'lines',
    line: { color: lineColor, width: 2.4 },
    hovertemplate: '%{x}: %{y:,.2f}<extra></extra>'
  };

  const pointTrace = {
    name: '최고/최저',
    x: [maxDate, minDate],
    y: [maxVal, minVal],
    type: 'scatter',
    mode: 'markers+text',
    marker: {
      size: [8, 8],
      color: ['#e53e3e', '#3182ce'],
      line: { color: '#ffffff', width: 2 }
    },
    text: [
      `최고 ${maxVal.toLocaleString(undefined, {minimumFractionDigits: 2, maximumFractionDigits: 2})}`,
      `최저 ${minVal.toLocaleString(undefined, {minimumFractionDigits: 2, maximumFractionDigits: 2})}`
    ],
    textposition: ['top center', 'bottom center'],
    textfont: { size: 12, color: ['#e53e3e', '#3182ce'], family: 'inherit' },
    hoverinfo: 'none'
  };

  const layout = {
    margin: { t: 30, r: 25, l: 55, b: 35 },
    paper_bgcolor: 'transparent',
    plot_bgcolor: 'transparent',
    showlegend: false,
    xaxis: { color: '#94a3b8', gridcolor: 'rgba(148, 163, 184, 0.15)', fixedrange: true },
    yaxis: {
      color: '#94a3b8',
      gridcolor: 'rgba(148, 163, 184, 0.15)',
      range: [yMin, yMax],
      fixedrange: true,
      tickformat: ',.2f'
    },
    hovermode: 'x'
  };

  Plotly.newPlot('plotlyMarketChart', [mainTrace, pointTrace], layout, { responsive: true, displayModeBar: false });
};

window.closeMarketModal = function() {
  window.closeModal('marketChartModal');
};

window._refreshPollTimer = null;
window._refreshClockTimer = null;
window._canCloseRefreshModal = false;

window.openRefreshModal = function() {
  const modal = document.getElementById("fa-refresh-modal");
  if (!modal) return;
  modal.style.display = "flex";
  modal.classList.add("active");
  document.body.style.overflow = "hidden";
  window._canCloseRefreshModal = false;
  const closeBtn = document.getElementById("fa-refresh-modal-close");
  if (closeBtn) closeBtn.style.display = "none";
};

window.closeRefreshModal = function() {
  const modal = document.getElementById("fa-refresh-modal");
  if (!modal) return;
  modal.classList.remove("active");
  modal.style.display = "none";
  document.body.style.overflow = "";
  if (window._refreshPollTimer) {
    clearInterval(window._refreshPollTimer);
    window._refreshPollTimer = null;
  }
  if (window._refreshClockTimer) {
    clearInterval(window._refreshClockTimer);
    window._refreshClockTimer = null;
  }
  const btn = document.querySelector(".fa-btn-hero-secondary.loading");
  if (btn) btn.classList.remove("loading");
};

document.addEventListener("keydown", function(e) {
  if (e.key === "Escape") {
    window.closeAllModals();
  }
});

window.triggerDashboardRefresh = async function(btn) {
  if (btn && btn.classList.contains("loading")) return;
  if (btn) btn.classList.add("loading");

  window.openRefreshModal();

  const badgeEl = document.getElementById("fa-refresh-badge");
  const percentEl = document.getElementById("fa-refresh-percent");
  const fillEl = document.getElementById("fa-refresh-progress-fill");
  const msgTextEl = document.getElementById("fa-refresh-msg-text");
  const timerSecEl = document.getElementById("fa-refresh-timer-sec");
  const successBanner = document.getElementById("fa-refresh-success-banner");
  const errorBanner = document.getElementById("fa-refresh-error-banner");
  const closeBtn = document.getElementById("fa-refresh-modal-close");

  const step1 = document.getElementById("fa-step-1");
  const step2 = document.getElementById("fa-step-2");
  const step3 = document.getElementById("fa-step-3");

  if (successBanner) successBanner.style.display = "none";
  if (errorBanner) errorBanner.style.display = "none";
  if (badgeEl) badgeEl.innerText = "1 / 3 단계 진행 중";
  if (percentEl) percentEl.innerText = "15%";
  if (fillEl) fillEl.style.width = "15%";
  if (msgTextEl) msgTextEl.innerText = "야후 파이낸스 실시간 시세 및 환율 수집 중...";
  if (step1) { step1.className = "fa-refresh-step-item active"; }
  if (step2) { step2.className = "fa-refresh-step-item"; }
  if (step3) { step3.className = "fa-refresh-step-item"; }

  let elapsedSeconds = 0;
  if (window._refreshClockTimer) clearInterval(window._refreshClockTimer);
  window._refreshClockTimer = setInterval(() => {
    elapsedSeconds++;
    if (timerSecEl) timerSecEl.innerText = elapsedSeconds + "초";
    if (elapsedSeconds >= 40 && closeBtn) {
      window._canCloseRefreshModal = true;
      closeBtn.style.display = "inline-block";
    }
  }, 1000);

  const adminBase = "https://fa-admin.vividian.net";

  // 1. 대시보드 빌드 트리거 요청
  try {
    await fetch(`${adminBase}/api/build-dashboard?t=` + Date.now(), {
      method: "POST",
      mode: "cors"
    });
  } catch (err) {
    console.warn("Build dashboard trigger fetch error:", err);
  }

  // 2. 상태 실시간 폴링
  let isDone = false;
  let failCount = 0;

  const pollStatus = async () => {
    if (isDone) return;
    try {
      const res = await fetch(`${adminBase}/api/dashboard-status?t=` + Date.now(), {
        method: "GET",
        mode: "cors"
      });
      if (!res.ok) throw new Error("HTTP " + res.status);
      const data = await res.json();
      failCount = 0;

      const step = data.step || 1;
      const progress = Math.max(15, data.progress || 15);
      const msg = data.message || "대시보드 생성 작업 진행 중...";

      if (msgTextEl) msgTextEl.innerText = msg;
      if (percentEl) percentEl.innerText = progress + "%";
      if (fillEl) fillEl.style.width = progress + "%";

      if (step === 1) {
        if (badgeEl) badgeEl.innerText = "1 / 3 단계 (시세 수집)";
        if (step1) step1.className = "fa-refresh-step-item active";
        if (step2) step2.className = "fa-refresh-step-item";
        if (step3) step3.className = "fa-refresh-step-item";
      } else if (step === 2) {
        if (badgeEl) badgeEl.innerText = "2 / 3 단계 (Hugo 빌드)";
        if (step1) step1.className = "fa-refresh-step-item done";
        if (step2) step2.className = "fa-refresh-step-item active";
        if (step3) step3.className = "fa-refresh-step-item";
      } else if (step === 3) {
        if (badgeEl) badgeEl.innerText = "3 / 3 단계 (배포 동기화)";
        if (step1) step1.className = "fa-refresh-step-item done";
        if (step2) step2.className = "fa-refresh-step-item done";
        if (step3) step3.className = "fa-refresh-step-item active";
      }

      if (data.status === "completed" || step >= 4 || progress >= 100) {
        isDone = true;
        clearInterval(window._refreshPollTimer);
        clearInterval(window._refreshClockTimer);

        if (step1) step1.className = "fa-refresh-step-item done";
        if (step2) step2.className = "fa-refresh-step-item done";
        if (step3) step3.className = "fa-refresh-step-item done";
        if (badgeEl) badgeEl.innerText = "갱신 완료! ✨";
        if (percentEl) percentEl.innerText = "100%";
        if (fillEl) fillEl.style.width = "100%";
        if (msgTextEl) msgTextEl.innerText = "대시보드가 최신으로 갱신되었습니다!";
        const spinner = document.getElementById("fa-refresh-msg-spinner");
        if (spinner) spinner.innerText = "✅";

        if (successBanner) successBanner.style.display = "flex";

        setTimeout(() => {
          window.location.reload(true);
        }, 1200);
        return;
      }

      if (data.status === "error") {
        isDone = true;
        clearInterval(window._refreshPollTimer);
        clearInterval(window._refreshClockTimer);
        window._canCloseRefreshModal = true;
        if (closeBtn) closeBtn.style.display = "inline-block";
        if (errorBanner) {
          const errDesc = document.getElementById("fa-refresh-error-desc");
          if (errDesc && data.error) errDesc.innerText = data.error;
          errorBanner.style.display = "flex";
        }
        if (btn) btn.classList.remove("loading");
        return;
      }
    } catch (e) {
      failCount++;
      if (failCount > 15) {
        isDone = true;
        clearInterval(window._refreshPollTimer);
        clearInterval(window._refreshClockTimer);
        window._canCloseRefreshModal = true;
        if (closeBtn) closeBtn.style.display = "inline-block";
        if (errorBanner) {
          const errDesc = document.getElementById("fa-refresh-error-desc");
          if (errDesc) errDesc.innerText = "서버 통신 연결이 원활하지 않습니다. 잠시 후 페이지를 새로고침해 주세요.";
          errorBanner.style.display = "flex";
        }
        if (btn) btn.classList.remove("loading");
      }
    }
  };

  window._refreshPollTimer = setInterval(pollStatus, 800);
  setTimeout(pollStatus, 300);
};

document.addEventListener("DOMContentLoaded", function () {
  function initDividendBarClickEvents() {
    const panes = ["fa-div-tab-last12m", "fa-div-tab-monthly"];
    panes.forEach(paneId => {
      const pane = document.getElementById(paneId);
      if (!pane) return;
      const plotDiv = pane.querySelector(".plotly-graph-div");
      if (plotDiv && typeof plotDiv.on === "function" && !plotDiv._divClickBound) {
        plotDiv._divClickBound = true;
        plotDiv.on("plotly_click", function(evtData) {
          if (evtData && evtData.points && evtData.points.length > 0) {
            const pt = evtData.points[0];
            if (pt.customdata && pt.customdata[1]) {
              window.openMonthDividendModal(pt.customdata[1]);
            }
          }
        });
      }
    });
  }

  // 탭 전환 이벤트 리스너
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
        // 만약 활성화된 패널이 상세 탭이면 내부의 활성 연도 차트도 리사이즈
        const activeYearPane = targetPane.querySelector(".fa-div-year-pane.active");
        if (activeYearPane) {
          const yearChartDiv = activeYearPane.querySelector(".plotly-graph-div");
          if (yearChartDiv && window.Plotly) {
            window.Plotly.Plots.resize(yearChartDiv);
          }
        }
        initDividendBarClickEvents();
      }

      setTimeout(() => {
        window.dispatchEvent(new Event("resize"));
        initDividendBarClickEvents();
      }, 50);
    });
  });

  // 배당금 상세 탭 연도 드롭다운 변경 리스너
  const yearSelect = document.getElementById("fa-div-year-select");
  if (yearSelect) {
    yearSelect.addEventListener("change", function () {
      const selectedYear = this.value;
      const wrap = this.closest(".fa-div-detail-wrap");
      if (!wrap) return;
      wrap.querySelectorAll(".fa-div-year-pane").forEach(p => p.classList.remove("active"));
      const targetPane = document.getElementById("fa-div-year-pane-" + selectedYear);
      if (targetPane) {
        targetPane.classList.add("active");
        const chartDiv = targetPane.querySelector(".plotly-graph-div");
        if (chartDiv && window.Plotly) {
          window.Plotly.Plots.resize(chartDiv);
        }
      }
    });
  }

  initDividendBarClickEvents();
  setTimeout(initDividendBarClickEvents, 200);
  setTimeout(initDividendBarClickEvents, 600);
  setTimeout(initDividendBarClickEvents, 1200);
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
            "  <script async src=\"https://pagead2.googlesyndication.com/pagead/js/adsbygoogle.js?client=ca-pub-9349922731061739\" crossorigin=\"anonymous\"></script>",
            "  <script src=\"https://cdn.plot.ly/plotly-2.35.2.min.js\"></script>",
            "  <style>",
            "    :root { color-scheme: light; }",
            "    body { margin: 0; background: #f8fafc; font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Noto Sans KR', sans-serif; }",
            "    .fa-standalone-wrap { max-width: 1000px; margin: 0 auto; padding: 20px 16px 40px; }",
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
    log_file = ROOT_DIR / "logs" / "plotly_exec.log"
    try:
        log_file.write_text(f"Started at {datetime.now()}\n", encoding="utf-8")
        _run_main()
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(f"Completed successfully at {datetime.now()}\n")
    except Exception as exc:
        import traceback
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(f"ERROR at {datetime.now()}:\n{traceback.format_exc()}\n")
        raise


def _log(msg: str) -> None:
    with open(ROOT_DIR / "logs" / "plotly_exec.log", "a", encoding="utf-8") as f:
        f.write(f"[{datetime.now()}] {msg}\n")
        f.flush()


def _run_main() -> None:
    args = parse_args()
    _log("Step 1: update_titles_from_fa_yaml")
    update_fa.update_titles_from_fa_yaml()
    static_dir = update_fa.PATHS.get("static_dir", ROOT_DIR / "content/fa")
    output_path = args.output or (static_dir / "latest_fa.html")
    fragment_path = args.fragment_output or DEFAULT_FRAGMENT_PATH

    _log("Step 2: read_trading_records")
    records = update_fa.read_trading_records()
    if records.empty:
        raise ValueError("Trading records are empty.")
    _log(f"Step 3: _build_report_data (records count: {len(records)})")
    data = _build_report_data(records)

    _log("Step 4: _build_dashboard_fragment")
    dashboard_fragment = _build_dashboard_fragment(data)
    fragment_path.parent.mkdir(parents=True, exist_ok=True)
    fragment_path.write_text(dashboard_fragment, encoding="utf-8")
    _remove_legacy_fragment(fragment_path)

    _log("Step 5: save standalone and sync to public")

    if not args.no_standalone:
        standalone_html = _wrap_standalone_html(dashboard_fragment, args.title)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(standalone_html, encoding="utf-8")
        print(f"HTML saved: {output_path}")

        # Nginx 도커 서빙 경로(public/fa)로도 직접 동시 반영
        public_fa = ROOT_DIR / "public" / "fa"
        public_fa.mkdir(parents=True, exist_ok=True)
        for pub_target in [public_fa / "latest_fa.html", public_fa / "index.html"]:
            try:
                pub_target.write_text(standalone_html, encoding="utf-8")
                import os
                try:
                    os.chmod(pub_target, 0o666)
                except Exception:
                    pass
                print(f"HTML synced to web public: {pub_target}")
            except Exception as e:
                print(f"(참고) 웹 경로 동기화 건너뜀: {e}")


if __name__ == "__main__":
    main()
