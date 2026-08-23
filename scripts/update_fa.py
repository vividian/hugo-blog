# 필요한 패키지: pip install pandas yfinance matplotlib pillow
#
# 이 스크립트는 Obsidian vault의 금융 자산 데이터를 처리하고 시각화하기 위한 것입니다.
# 거래 기록(trading_records.csv)을 읽어들여, yfinance를 통해 최신 주가 및 환율 정보를 가져옵니다.
# 이를 바탕으로 계좌별 자산 현황, 월별 배당금, 보유 종목 상세 내역 등 다양한 보고서를 생성합니다.
# 생성된 차트와 데이터는 Hugo 블로그의 정적 파일로 출력됩니다.
from __future__ import annotations

import argparse
import json
import re
import shutil
import textwrap
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from cycler import cycler
from datetime import datetime
import matplotlib.dates as mdates
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import matplotlib.ticker as mticker
import numpy as np
from PIL import Image
import pandas as pd
import yaml
import yfinance as yf


ROOT_DIR = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT_DIR / "config" / "config.yaml"


def _load_fa_paths() -> Dict[str, Path]:
    config = {}
    if CONFIG_PATH.exists():
        with CONFIG_PATH.open("r", encoding="utf-8") as fp:
            config = yaml.safe_load(fp) or {}
    fa_paths = (config.get("financial_assets") or {}).get("paths") or {}

    def resolve(key: str, default: str) -> Path:
        return (ROOT_DIR / fa_paths.get(key, default)).resolve()

    return {
        "trading_records": resolve("trading_records", "config/trading_records.csv"),
        "static_dir": resolve("static_dir", "content/fa"),
        "build_info": resolve("build_info", "data/fa.json"),
        "yaml": resolve("yaml", "config/fa.yaml"),
    }


PATHS = _load_fa_paths()
TRADING_RECORDS_PATH = PATHS["trading_records"]
STATIC_FINANCIALASSETS_DIR = PATHS["static_dir"]
BUILD_INFO_PATH = PATHS["build_info"]
FINANCIALASSETS_YAML_PATH = PATHS["yaml"]


# 경로 상수: 스크립트, 콘텐츠, 정적 파일, 설정 파일 위치
FX_TICKER = "USDKRW=X"  # 환율 조회를 위한 티커
START_MONTH = pd.Timestamp("2022-02-28")  # 데이터 분석 시작 월
USD_ACCOUNTS = {"usa"}  # 달러 기반 계좌
ACCOUNT_ORDER = ["usa", "kor1", "kor2", "sema", "irp", "psf1", "isa1", "psf2", "isa2"]  # 계좌 표시 순서
DETAIL_ACCOUNTS = ["usa", "kor1", "kor2", "sema", "irp", "psf1", "isa1", "psf2", "isa2"]  # 상세 내역을 표시할 계좌
ACCOUNT_LABELS = {
    "usa": "미국 주식",
    "kor1": "국내 주식1",
    "sema": "공제회",
    "irp": "IRP",
    "psf1": "연금저축1",
    "kor2": "국내 주식2",    
    "isa1": "ISA1",
    "psf2": "연금저축2",
    "isa2": "ISA2",
}
# 기본 캔버스 및 레이아웃 설정 (fa.yaml에 미정의 시 적용되는 fallback 기본값)
DEFAULT_CANVAS_LAYOUT = {
    "common": {
        "dpi": 150,
        "canvas_bg_color": "#fffdf5",
        "title_color": "#2c3e50",
        "title_font_size": 16,
        "title_row_height": 0.3,
        "default_canvas_width": 12.0,
    },
    "account_detail": {
        "canvas_width": 12.5,
        "min_canvas_height": 4.8,
        "pie_table_ratio": [0.65, 1.35],
        "pad_top": 0.25,
        "pad_bottom": 0.25,
        "pie_pad_top": 0.0,
        "pie_radius": 1.75,
        "pie_mom_fontsize": 11.5,
        "pie_mom_gap": 0.20,
        "subtitle_height": 0.28,
        "subtitle_margin_top": 0.10,
        "subtitle_margin_bottom": 0.10,
        "subtitle_fontsize": 12.5,
        "row_height": 0.38,
        "table_fontsize": 12.0,
        "main_table_fontsize": 12.0,
        "summary_table_fontsize": 12.0,
        "section_gap": 0.25,
        "rebal_line_height": 0.35,
        "rebal_fontsize": 11.5,
        "pie_pct_fontsize": 13.0,
    },
    "total_holdings": {
        "canvas_width": 12.0,
        "pad_top": 0.25,
        "pad_bottom": 0.25,
        "row_height": 0.34,
        "table_fontsize": 11.0,
        "min_canvas_height": 4.0,
    },
    "account_assets": {
        "canvas_width": 12.0,
        "pad_top": 0.25,
        "pad_bottom": 0.25,
        "row_height": 0.38,
        "table_fontsize": 12.0,
        "min_canvas_height": 4.0,
    },
    "exchange_rate": {
        "canvas_width": 12.0,
        "pad_top": 0.10,
        "pad_bottom": 0.15,
        "row_height": 0.40,
        "table_fontsize": 13.0,
    },
    "trading_history": {
        "canvas_width": 12.0,
        "pad_top": 0.3,
        "pad_bottom": 0.3,
        "summary_line_height": 0.38,
        "summary_fontsize": 14.0,
        "text_line_height": 0.32,
        "text_fontsize": 13.0,
        "min_canvas_height": 3.5,
    },
    "portfolio_allocation": {
        "canvas_width": 22.5,
        "canvas_height": 9.75,
        "title_fontsize": 20.0,
        "pie_pct_fontsize": 16.0,
        "legend_fontsize": 15.0,
    },
    "assets_trend": {
        "canvas_width": 12.0,
        "canvas_height": 6.0,
        "legend_fontsize": 12.0,
        "linewidth": 2.0,
    },
    "dividends": {
        "canvas_width": 12.0,
        "canvas_height": 6.0,
        "bar_label_fontsize": 12.0,
        "tick_fontsize": 12.0,
        "legend_fontsize": 10.5,
        "legend_ncol": 7,
    },
    "market_indices": {
        "canvas_width": 12.0,
        "pad_top": 0.10,
        "pad_bottom": 0.15,
        "row_height": 0.40,
        "table_fontsize": 13.0,
    },
}


def _load_canvas_layout() -> Dict[str, Dict[str, Any]]:
    """config/fa.yaml에서 canvas_layout 설정을 로드하고 기본값과 병합한다."""
    layout = {}
    if FINANCIALASSETS_YAML_PATH.exists():
        try:
            with FINANCIALASSETS_YAML_PATH.open("r", encoding="utf-8") as fp:
                data = yaml.safe_load(fp) or {}
                layout = data.get("canvas_layout") or {}
        except Exception as e:
            print(f"(경고) fa.yaml canvas_layout 로드 실패: {e}")

    merged: Dict[str, Dict[str, Any]] = {}
    for section, defaults in DEFAULT_CANVAS_LAYOUT.items():
        user_section = layout.get(section) or {}
        merged[section] = {**defaults, **user_section}
    return merged


LAYOUT = _load_canvas_layout()

TITLE_FONT_SIZE = float(LAYOUT["common"]["title_font_size"])
FIG_DPI = int(LAYOUT["common"]["dpi"])
CANVAS_BG_COLOR = str(LAYOUT["common"]["canvas_bg_color"])
TITLE_COLOR = str(LAYOUT["common"]["title_color"])
TITLE_POS = (0.0, 0.5)  # 제목 축 내부 좌표 (좌측, 중앙)
TITLE_ROW_HEIGHT = float(LAYOUT["common"]["title_row_height"])
FIG_LEFT = 0.05
FIG_RIGHT = 0.98
FIG_TOP = 0.98
FIG_BOTTOM = 0.06
ACCOUNT_TITLES = {
    "title_exchange_rate": "◉ 환율 (USD/KRW) 추세",
    "title_market_indices": "◉ 주요 시장 지표 (인덱스)",
    "title_assets_trend": "◉ 계좌별 자산 추세",
    "title_assets_investment_trend": "◉ 누적 투자금 vs 평가금 추세",
    "title_portfolio_allocation": "◉ 전체 포트폴리오 비중",
    "title_account_assets": "◉ 전체 계좌별 자산 현황 (투자금, 평가금, 수익금 등)",
    "title_total_holdings": "◉ 실시간 보유종목 현황",
    "title_trading_history": "◉ 보유종목 거래내역",
    "title_monthly_dividends": "◉ 월별 배당금 및 분배금 현황 (최근 12개월)",
    "title_yearly_dividends": "◉ 연별 배당금 및 분배금 현황",
    "title_usa_detail": "◉ 상세계좌: 미국주식 (SPYM:SGOV = 9:1)",
    "title_kor1_detail": "◉ 상세계좌: 국내주식1",
    "title_kor2_detail": "◉ 상세계좌: 국내주식2",
    "title_sema_detail": "◉ 상세계좌: SEMA (S&P500-FD:GOLD-FD:SAVING = 6:1:3)",
    "title_irp_detail": "◉ 상세계좌: IRP (S&P500:KOFR = 7:3)",
    "title_psf1_detail": "◉ 상세계좌: 연금저축1 (SCHD:QQQ:MMA = 6:3:1)",
    "title_isa1_detail": "◉ 상세계좌: ISA1",
    "title_psf2_detail": "◉ 상세계좌: 연금저축2 (SCHD:QQQ:MMA = 6:3:1)",
    "title_isa2_detail": "◉ 상세계좌: ISA2",
}
CONTENT_TITLE_KEYS = {
    "assets_trend": "title_assets_trend",
    "assets_investment_trend": "title_assets_investment_trend",
    "portfolio_allocation": "title_portfolio_allocation",
    "account_assets": "title_account_assets",
    "total_holdings": "title_total_holdings",
    "trading_history": "title_trading_history",
    "monthly_dividends": "title_monthly_dividends",
    "exchange_rate": "title_exchange_rate",
    "market_indices": "title_market_indices",
    "yearly_dividends": "title_yearly_dividends",
}
ACCOUNT_RAW_NAMES: Dict[str, str] = {}

@dataclass
class AssetConfig:
    """자산 설정 정보를 담는 데이터 클래스"""
    name: str
    abbrev: str
    ticker: str
    region: str = "기타"
    asset_class: str = "기타"


def parse_target_allocation(account_name: str) -> Optional[Dict[str, float]]:
    """
    계좌 제목 문자열에서 목표 비중 비율을 추출한다.
    예: '연금저축1 (SCHD:QQQ:MMA = 5:4:1)' -> {'SCHD': 0.5, 'QQQ': 0.4, 'MMA': 0.1}
    예: '미국 (SPYM:SGOV = 9:1)' -> {'SPYM': 0.9, 'SGOV': 0.1}
    예: '공제회 (S&P500-FD:GOLD-FD:SAVING = 6:1:3)' -> {'S&P500-FD': 0.6, 'GOLD-FD': 0.1, 'SAVING': 0.3}
    예: 'IRP (S&P500:SCHD-IEF = 7:3)' -> {'S&P500': 0.7, 'SCHD-IEF': 0.3}
    """
    if not account_name:
        return None
    match = re.search(r"\((.*?)=(.*?)\)", account_name)
    if not match:
        return None
    
    keys_part = match.group(1).strip()
    ratios_part = match.group(2).strip()
    
    keys = [k.strip() for k in keys_part.split(":") if k.strip()]
    try:
        ratios = [float(r.strip()) for r in ratios_part.split(":") if r.strip()]
    except ValueError:
        return None
        
    if len(keys) != len(ratios) or len(keys) == 0:
        return None
        
    total_ratio = sum(ratios)
    if total_ratio <= 0:
        return None
        
    return {k: r / total_ratio for k, r in zip(keys, ratios)}


def match_target_key(symbol: str, abbrev: str, asset_class: str, target_keys: List[str]) -> str:
    """
    종목의 정보(symbol, abbrev, asset_class)를 바탕으로 target_keys 중 가장 적절한 키에 매핑한다.
    """
    symbol_str = str(symbol or "").upper()
    abbrev_str = str(abbrev or "").upper()
    ac_str = str(asset_class or "").upper()
    combined = f"{symbol_str} {abbrev_str} {ac_str}"
    
    alias_dict = {
        "SPYM": ["SPYM", "S&P500", "SP500"],
        "SGOV": ["SGOV", "현금", "CASH"],
        "SCHD": ["SCHD"],
        "QQQ": ["QQQ", "나스닥"],
        "MMA": ["MMA", "KOFR", "현금", "SAVING", "저축"],
        "S&P500": ["S&P500", "SP500", "S&P 500"],
        "SCHD-IEF": ["SCHD:IEF", "SCHD-IEF", "IEF", "TLT", "TLTW", "국채", "채권"],
        "SCHD:IEF": ["SCHD:IEF", "SCHD-IEF", "IEF", "TLT", "TLTW", "국채", "채권"],
        "S&P500-FD": ["S&P500", "SP500"],
        "GOLD-FD": ["GOLD", "골드", "금"],
        "SAVING": ["SAVING", "저축", "현금", "MMA"],
    }
    
    # 1. 정확히 대소문자 무관하게 타겟 키명이 포함되어 있는지 검사
    for key in target_keys:
        key_upper = key.upper()
        if key_upper in symbol_str or key_upper in abbrev_str or key_upper in ac_str:
            return key
            
    # 2. 별칭 맵을 통한 키워드 매칭 검사
    for key in target_keys:
        key_upper = key.upper()
        aliases = alias_dict.get(key_upper, [key_upper])
        for alias in aliases:
            if alias.upper() in combined:
                return key
                
    return target_keys[0] if target_keys else "기타"


def calculate_rebalancing_df(account_name: str, account_holdings: pd.DataFrame, symbol_map: Optional[Dict[str, AssetConfig]] = None) -> Optional[pd.DataFrame]:
    """
    계좌별 보유종목 현황을 기반으로 목표 비중에 맞추기 위한 매도/매수 필요 금액을 계산한다.
    """
    target_alloc = parse_target_allocation(account_name)
    if not target_alloc or account_holdings is None or account_holdings.empty:
        return None
        
    total_val = account_holdings["평가금"].sum()
    if total_val <= 0:
        return None
        
    holdings = account_holdings.copy()
    keys = list(target_alloc.keys())
    
    assigned_keys = []
    for _, row in holdings.iterrows():
        sym = row["종목"]
        config = symbol_map.get(sym) if symbol_map else None
        abbrev = config.abbrev if config else sym
        ac = config.asset_class if config else ""
        
        assigned_key = match_target_key(sym, abbrev, ac, keys)
        assigned_keys.append(assigned_key)
        
    holdings["target_key"] = assigned_keys
    
    curr_by_key = holdings.groupby("target_key")["평가금"].sum().to_dict()
    
    rows = []
    for key, target_ratio in target_alloc.items():
        curr_amt = curr_by_key.get(key, 0.0)
        curr_pct = (curr_amt / total_val) * 100.0
        target_pct = target_ratio * 100.0
        target_amt = total_val * target_ratio
        diff_amt = target_amt - curr_amt  # 양수: 매수 필요, 음수: 매도 필요
        
        rows.append({
            "자산군": key,
            "현재평가금": curr_amt,
            "현재비중": curr_pct,
            "목표비중": target_pct,
            "목표평가금": target_amt,
            "조정금액": diff_amt,
        })
        
    return pd.DataFrame(rows)



@dataclass
class Position:
    """보유 종목 정보를 담는 데이터 클래스"""
    account: str
    symbol: str
    ticker: str
    quantity: float
    cost: float


def ensure_static_dir() -> None:
    """결과물을 저장할 정적 폴더가 없다면 생성한다."""
    STATIC_FINANCIALASSETS_DIR.mkdir(parents=True, exist_ok=True)


def account_label(account: str) -> str:
    """계좌 코드에 해당하는 표시용 이름을 반환한다."""
    return ACCOUNT_LABELS.get(account, account)


def _clean_numeric(series: Iterable) -> pd.Series:
    """쉼표가 포함된 문자열 숫자를 float로 변환한다."""
    ser = pd.Series(series, dtype="string").str.replace(",", "", regex=False)
    return pd.to_numeric(ser, errors="coerce")


def read_trading_records() -> pd.DataFrame:
    """fa_records.db (SQLite) 또는 trading_records.csv 파일을 읽어 DataFrame으로 반환한다."""
    db_path = ROOT_DIR / "db" / "fa_records.db"
    if db_path.exists():
        import sqlite3
        try:
            conn = sqlite3.connect(db_path)
            query = """
            SELECT date AS 일자, account AS 계좌, symbol AS 종목, kind AS 구분,
                   unit_price AS 단가, quantity AS 수량, amount AS 금액,
                   dividend AS 배당, deposit AS 투자금, evaluation AS 평가금,
                   exchange_rate AS 환율, memo AS 비고
            FROM trading_records
            ORDER BY date ASC, id ASC;
            """
            df = pd.read_sql_query(query, conn)
            conn.close()
            if not df.empty:
                df["일자"] = pd.to_datetime(df["일자"].astype(str).str.replace(".", "-", regex=False), format="mixed", errors="coerce")
                numeric_cols = ["단가", "수량", "배당", "투자금", "평가금", "환율", "금액"]
                for col in numeric_cols:
                    if col in df.columns:
                        df[col] = pd.to_numeric(df[col], errors="coerce")
                return df
        except Exception as e:
            print(f"(경고) SQLite DB 로드 실패, CSV 폴백 시도: {e}")

    if not TRADING_RECORDS_PATH.exists():
        raise FileNotFoundError(f"trading_records.csv 파일을 찾을 수 없습니다: {TRADING_RECORDS_PATH}")

    df = pd.read_csv(TRADING_RECORDS_PATH, encoding="utf-8-sig")
    # 2022.01.31, 2022-01-31 등 다양한 날짜 입력을 공통 포맷으로 정규화한다.
    df["일자"] = (
        pd.Series(df["일자"], dtype="string")
        .str.replace(".", "-", regex=False)
        .str.replace("/", "-", regex=False)
    )
    df["일자"] = pd.to_datetime(df["일자"], errors="coerce")

    numeric_cols = ["단가", "수량", "배당", "투자금", "환율", "평가금"]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = _clean_numeric(df[col])

    return df


def latest_month_code(records: pd.DataFrame) -> str:
    """거래 기록에서 가장 최근 월 코드를 'YYMM' 형식으로 반환한다."""
    latest_date = records["일자"].dropna().max()
    if pd.isna(latest_date):
        latest_date = pd.Timestamp.today()
    return latest_date.strftime("%y%m")


def load_symbol_map() -> Dict[str, AssetConfig]:
    """fa.yaml에 정의된 계좌/종목 구성 정보를 AssetConfig 형태로 적재한다."""
    with FINANCIALASSETS_YAML_PATH.open("r", encoding="utf-8") as f:
        portfolio = yaml.safe_load(f)

    symbol_map: Dict[str, AssetConfig] = {}
    for account in portfolio.get("accounts", []):
        for item in account.get("items", []):
            if len(item) < 3:
                continue
            name, abbrev, ticker = item[0], item[1], item[2]
            region = item[3] if len(item) > 3 else "기타"
            asset_class = item[4] if len(item) > 4 else "기타"
            y_ticker = ticker or ""
            if ticker and ticker.startswith("KRX:"):
                y_ticker = ticker.replace("KRX:", "") + ".KS"
            config = AssetConfig(name=name, abbrev=abbrev, ticker=y_ticker, region=region, asset_class=asset_class)
            for key in {name, abbrev}:
                if not key:
                    continue
                existing = symbol_map.get(key)
                if existing and existing.ticker != config.ticker:
                    raise ValueError(f"중복 키 '{key}'에 서로 다른 티커가 매핑되어 있습니다.")
                symbol_map[key] = config
    return symbol_map


def build_fx_series(records: pd.DataFrame, end: Optional[pd.Timestamp] = None) -> pd.Series:
    """거래 데이터 범위를 기준으로 환율(USD/KRW) 시계열 데이터를 생성한다."""
    fx_cache_file = ROOT_DIR / "data" / "fx_cache.csv"

    if records.empty:
        start = pd.Timestamp.today() - pd.Timedelta(days=60)
    else:
        earliest = records["일자"].dropna().min()
        if pd.isna(earliest):
            start = pd.Timestamp.today() - pd.Timedelta(days=60)
        else:
            start = earliest - pd.Timedelta(days=7)

    today_buffer = pd.Timestamp.today().normalize() + pd.Timedelta(days=2)
    if end is None:
        end = today_buffer
    else:
        end = max(end + pd.Timedelta(days=2), today_buffer)

    data = pd.DataFrame()
    try:
        data = yf.download(
            FX_TICKER,
            start=start,
            end=end,
            progress=False,
            auto_adjust=False,
        )
    except Exception as e:
        print(f"⚠️ [환율 다운로드 경고] yfinance 환율 조회 실패: {e}")

    if not data.empty:
        series = data["Adj Close"] if "Adj Close" in data else data
        if isinstance(series, pd.DataFrame):
            series = series.iloc[:, 0]
        series = pd.Series(series).ffill()
        series.index = pd.to_datetime(series.index).tz_localize(None)

        # 로컬 캐시에 저장
        try:
            fx_cache_file.parent.mkdir(parents=True, exist_ok=True)
            series.to_csv(fx_cache_file, header=True)
        except Exception:
            pass
        return series

    # 다운로드 실패 시 로컬 캐시 로드
    if fx_cache_file.exists():
        try:
            cached_df = pd.read_csv(fx_cache_file, index_col=0, parse_dates=True)
            cached_series = cached_df.iloc[:, 0].ffill()
            cached_series.index = pd.to_datetime(cached_series.index).tz_localize(None)
            print(f"ℹ️ [환율 캐시 사용] {fx_cache_file} 캐시된 환율 데이터를 사용합니다.")
            return cached_series
        except Exception as e:
            print(f"⚠️ [환율 캐시 로드 실패] {e}")

    # 최종 fallback: 기본 환율 시리즈 생성
    date_range = pd.date_range(start=start, end=end, freq="D")
    fallback_series = pd.Series(1380.0, index=date_range)
    print("ℹ️ [환율 기본값 사용] 1,380원 기본 환율 시계열을 생성합니다.")
    return fallback_series


def fx_rate_on(date: pd.Timestamp, fx_series: pd.Series) -> float:
    """특정 날짜의 환율을 조회한다. 해당 날짜에 데이터가 없으면 가장 가까운 과거의 데이터를 사용한다."""
    date = pd.Timestamp(date).tz_localize(None)
    available = fx_series.loc[:date]
    if available.empty:
        return float(fx_series.iloc[0])
    return float(available.iloc[-1])


def convert_to_krw(account: str,
                   amount: Optional[float],
                   date: Optional[pd.Timestamp],
                   fx_series: pd.Series,
                   *,
                   use_latest: bool = False,) -> float:
    """달러 금액을 원화로 변환한다. 달러 계좌가 아니면 그대로 반환한다."""
    if amount is None or pd.isna(amount):
        return 0.0
    value = float(amount)
    if account in USD_ACCOUNTS:
        if use_latest or date is None:
            rate = float(fx_series.iloc[-1])
        else:
            rate = fx_rate_on(date, fx_series)
        return value * rate
    return value


def get_monthly_prices(end_date: pd.Timestamp, records: Optional[pd.DataFrame] = None) -> Optional[pd.DataFrame]:
    """월말 종가 데이터를 가져온다.

    records가 주어지면 실제 거래내역(수량/단가 존재, non-sema)에 사용된 종목만 조회한다.
    """
    fa_data = load_symbol_map()
    tickers_map: Dict[str, str] = {}

    if records is not None and not records.empty:
        traded = records[
            (records["수량"].notna())
            & (records["단가"].notna())
            & (records["수량"] != 0)
            & (records["계좌"] != "sema")
        ]
        for symbol in traded["종목"].dropna().astype(str):
            config = fa_data.get(symbol)
            if not config or not config.ticker:
                continue
            tickers_map[config.ticker] = config.abbrev or config.name

    # 거래내역 기반 대상이 비어 있으면 기존 방식으로 전체 종목을 조회한다.
    if not tickers_map:
        for config in fa_data.values():
            if not config.ticker:
                continue
            tickers_map[config.ticker] = config.abbrev or config.name

    if not tickers_map:
        print("fa.yaml에 종목 정보가 없습니다.")
        return None

    tickers = list(tickers_map.keys())
    start_date = "2022-02-01"
    end_date_str = end_date.strftime("%Y-%m-%d")

    print(f"데이터 조회 중: {', '.join(tickers)}...")
    data = yf.download(
        tickers,
        start=start_date,
        end=end_date_str,
        progress=False,
        auto_adjust=False,
    )
    if data.empty:
        print("데이터 다운로드에 실패했습니다.")
        return None

    adj_close = data["Adj Close"]
    if isinstance(adj_close, pd.Series):
        adj_close = adj_close.to_frame(name=tickers[0])

    monthly_prices = adj_close.resample("BME").last()
    monthly_prices.index = monthly_prices.index.strftime("%Y-%m")
    monthly_prices.index.name = "월"
    monthly_prices = monthly_prices.rename(columns=tickers_map)
    display_order = [tickers_map[t] for t in tickers]
    available_order = [name for name in display_order if name in monthly_prices.columns]
    missing = [name for name in display_order if name not in monthly_prices.columns]
    if missing:
        print(f"(경고) 다음 종목의 가격 데이터를 찾을 수 없습니다: {', '.join(missing)}")
    if not available_order:
        print("다운로드된 가격 데이터가 없습니다.")
        return None
    monthly_prices = monthly_prices[available_order]
    table = monthly_prices.reset_index()

    return table


def build_evaluation_index(start: pd.Timestamp, end: pd.Timestamp) -> pd.DatetimeIndex:
    """평가 기준이 될 날짜 인덱스(매월 말일 및 최종일)를 생성한다."""
    end = pd.Timestamp(end).normalize()
    month_ends = pd.date_range(start=start, end=end, freq="ME")
    if not month_ends.empty and month_ends[-1] == end:
        eval_index = month_ends
    else:
        eval_index = month_ends.append(pd.DatetimeIndex([end]))
    if eval_index.empty:
        eval_index = pd.DatetimeIndex([end])
    return eval_index


def is_krw_ticker(ticker: str) -> bool:
    """티커가 원화(한국) 주식인지 여부를 확인한다."""
    return ticker.endswith((".KS", ".KQ"))


def download_adj_close(tickers: Iterable[str], 
                       start: pd.Timestamp, 
                       end: pd.Timestamp) -> pd.DataFrame:
    """지정된 기간 동안 여러 티커의 수정 종가를 다운로드한다."""
    tickers = sorted(set(tickers))
    if not tickers:
        return pd.DataFrame()
    data = yf.download(
        tickers,
        start=start - pd.Timedelta(days=5),
        end=end + pd.Timedelta(days=2),
        progress=False,
        auto_adjust=False,
    )
    if data.empty:
        raise RuntimeError("가격 데이터를 내려받지 못했습니다.")
    adj_close = data["Adj Close"] if "Adj Close" in data else data
    if isinstance(adj_close, pd.Series):
        adj_close = adj_close.to_frame(name=tickers[0])
    return adj_close.sort_index().ffill()


def build_quantity_series(trades: pd.DataFrame,
                          symbol_map: Dict[str, AssetConfig],
                          eval_index: pd.DatetimeIndex,) -> Tuple[Dict[Tuple[str, str], pd.Series], List[str]]:
    """거래 기록을 바탕으로 평가일별 보유 수량 시계열 데이터를 생성한다."""
    qty_map: Dict[Tuple[str, str], pd.Series] = {}
    accounts: List[str] = []

    filtered = trades[trades["수량"].notna() & trades["단가"].notna() & (trades["수량"] != 0)].copy()
    filtered["symbol_key"] = filtered["종목"]
    filtered["ticker"] = filtered["symbol_key"].map(lambda k: symbol_map.get(k).ticker if symbol_map.get(k) else None)
    filtered = filtered[filtered["ticker"].notna()]

    for (account, ticker), group in filtered.groupby(["계좌", "ticker"]):
        daily_qty = (
            group.sort_values("일자")
            .groupby("일자")["수량"]
            .sum()
            .sort_index()
        )
        cumulative = daily_qty.cumsum()
        aligned = cumulative.reindex(cumulative.index.union(eval_index)).sort_index().ffill()
        qty_series = aligned.reindex(eval_index).fillna(0.0)
        qty_map[(account, ticker)] = qty_series
        if account not in accounts:
            accounts.append(account)

    return qty_map, accounts


def align_series(series: pd.Series, 
                 target_index: pd.DatetimeIndex) -> pd.Series:
    """시계열 데이터를 목표 인덱스에 맞춰 정렬하고 누락된 값을 채운다."""
    combined_index = series.index.union(target_index)
    return series.reindex(combined_index).sort_index().ffill().reindex(target_index).fillna(0.0)


def compute_account_values(qty_map: Dict[Tuple[str, str], pd.Series],
                           price_df: pd.DataFrame,
                           fx_series: pd.Series,
                           eval_index: pd.DatetimeIndex,) -> Dict[str, pd.Series]:
    """계좌별 평가금액 시계열 데이터를 계산한다."""
    account_values: Dict[str, pd.Series] = {}
    for (account, ticker), qty_series in qty_map.items():
        if ticker not in price_df.columns:
            continue
        price_series = align_series(price_df[ticker], eval_index)
        value_series = qty_series * price_series
        if not is_krw_ticker(ticker):
            fx_aligned = align_series(fx_series, eval_index)
            value_series = value_series * fx_aligned
        account_values.setdefault(account, pd.Series(0.0, index=eval_index))
        account_values[account] = account_values[account] + value_series
    return account_values


def build_gongje_account_series(records: pd.DataFrame, 
                                eval_index: pd.DatetimeIndex) -> pd.Series:
    """'sema' 계좌의 평가금액 시계열을 별도로 계산한다."""
    gong = records[records["계좌"] == "sema"].copy()
    if gong.empty:
        return pd.Series(0.0, index=eval_index)

    gong["투자금"] = pd.to_numeric(gong["투자금"], errors="coerce")
    gong["평가금"] = pd.to_numeric(gong.get("평가금"), errors="coerce")
    gong_price = pd.to_numeric(gong.get("단가"), errors="coerce")
    gong_qty = pd.to_numeric(gong.get("수량"), errors="coerce")
    gong_amount = (gong_price.fillna(0) * gong_qty.fillna(0)).rename("거래금액")

    if gong["평가금"].notna().any():
        eval_rows = gong.dropna(subset=["평가금"])
        if not eval_rows.empty:
            daily_eval = eval_rows.groupby("일자")["평가금"].sum().sort_index()
            return align_series(daily_eval, eval_index)

    mask = (gong_amount > 0) & (gong["투자금"].isna() | (gong_amount == gong["투자금"]))
    gong_valid = gong.loc[mask].copy()
    if gong_valid.empty:
        return pd.Series(0.0, index=eval_index)

    gong_valid["거래금액"] = gong_amount.loc[mask]
    daily = gong_valid.groupby("일자")["거래금액"].sum().sort_index()
    cumulative = daily.cumsum()

    return align_series(cumulative, eval_index)


def build_account_valuation_df(records: pd.DataFrame, 
                               fx_series: pd.Series, 
                               end_date: pd.Timestamp) -> pd.DataFrame:
    """모든 계좌의 평가금액 시계열 데이터프레임을 생성한다."""
    symbol_map = load_symbol_map()
    eval_index = build_evaluation_index(START_MONTH, end_date)

    trades = records[records["계좌"] != "sema"].copy()
    qty_map, _ = build_quantity_series(trades, symbol_map, eval_index)

    tickers = {ticker for _, ticker in qty_map.keys()}
    account_df = pd.DataFrame(index=eval_index)

    if tickers:
        price_df = download_adj_close(
            tickers,
            start=START_MONTH - pd.DateOffset(months=1),
            end=end_date,
        )

        fx_subset = fx_series.loc[:eval_index[-1]]
        account_values = compute_account_values(qty_map, price_df, fx_subset, eval_index)
        traded_accounts_df = pd.DataFrame(account_values).reindex(eval_index).fillna(0.0)
        account_df = account_df.join(traded_accounts_df, how="left")

    gong_series = build_gongje_account_series(records, eval_index)
    account_df["sema"] = gong_series
    account_df = account_df.sort_index()
    ordered_cols = [col for col in ACCOUNT_ORDER if col in account_df.columns]
    ordered_cols += [col for col in account_df.columns if col not in ordered_cols]
    account_df = account_df[ordered_cols]

    return account_df


def _load_blog_font() -> Optional[str]:
    """블로그에 사용된 나눔스퀘어라운드 폰트를 로드한다."""
    font_candidates = [
        ROOT_DIR / "static" / "fonts" / "NanumSquareRoundEB.ttf",
        ROOT_DIR / "themes" / "hugo-blog-awesome" / "static" / "fonts" / "Roboto" / "roboto-v30-latin-regular.ttf",
    ]
    for path in font_candidates:
        if path.exists():
            try:
                fm.fontManager.addfont(str(path))
                return fm.FontProperties(fname=str(path)).get_name()
            except Exception:
                continue
    return None


def _configure_matplotlib() -> None:
    """Matplotlib 차트의 한글 폰트 및 스타일을 설정한다."""
    blog_font = _load_blog_font()
    base_order = [
        blog_font,
        "NanumSquareRoundEB.ttf",
        "AppleGothic",
        "NanumGothic",
        "Malgun Gothic",
        "Roboto",
        "DejaVu Sans",
    ]
    font_stack: List[str] = []
    available = {font.name for font in fm.fontManager.ttflist}
    for name in base_order:
        if name and name in available and name not in font_stack:
            font_stack.append(name)
    if not font_stack:
        font_stack = ["DejaVu Sans"]
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = font_stack
    plt.rcParams["axes.prop_cycle"] = cycler(color=plt.get_cmap("tab20c").colors)
    plt.rcParams["axes.unicode_minus"] = False


def plot_title_image(title_key: str, output_path: Path) -> Path:
    """제목 전용 이미지를 생성한다."""
    _configure_matplotlib()
    title = ACCOUNT_TITLES.get(title_key, title_key)
    fig, ax = plt.subplots(figsize=(12, TITLE_ROW_HEIGHT), dpi=FIG_DPI)
    fig.patch.set_facecolor(CANVAS_BG_COLOR)
    ax.axis("off")
    ax.text(
        0.0,
        0.5,
        title,
        ha="left",
        va="center",
        fontsize=TITLE_FONT_SIZE,
        fontweight="bold",
        color=TITLE_COLOR,
        transform=ax.transAxes,
    )
    fig.savefig(
        output_path,
        format=output_path.suffix.lstrip(".") or "png",
        bbox_inches="tight",
        pad_inches=0.2,
        facecolor=fig.get_facecolor(),
        edgecolor="none",
    )
    plt.close(fig)
    print(f"제목 이미지 저장 완료: {output_path}")
    return output_path


def _save_title(prefix: str,
                output_dir: Path,
                title_key: Optional[str],
                outputs: Dict[str, Path],
                graph_path: Optional[Path] = None) -> Optional[Path]:
    """주어진 제목 키를 해당하는 그래프 이미지 상단에 병합한다. (독립 파일 미생성)"""
    if not title_key or not graph_path or not graph_path.exists():
        return None
    
    # 임시 타이틀 파일 경로
    temp_title_path = graph_path.with_name(f"temp_title_{title_key}_{prefix}.webp")
    plot_title_image(title_key, temp_title_path)
    
    try:
        with Image.open(temp_title_path) as img_title, Image.open(graph_path) as img_graph:
            w_graph, h_graph = img_graph.size
            w_title, h_title = img_title.size
            
            # 타이틀 이미지 비율을 유지하면서 그래프 가로폭에 맞춰 리사이즈
            new_h_title = int(h_title * (w_graph / w_title))
            
            if hasattr(Image, "Resampling"):
                resample_filter = Image.Resampling.LANCZOS
            else:
                resample_filter = Image.ANTIALIAS  # 구 버전 PIL 호환
                
            img_title_resized = img_title.resize((w_graph, new_h_title), resample_filter)
            
            # 수직 결합된 새 이미지 생성 (배경색 CANVAS_BG_COLOR)
            merged_img = Image.new("RGBA", (w_graph, new_h_title + h_graph), color=CANVAS_BG_COLOR)
            merged_img.paste(img_title_resized, (0, 0))
            merged_img.paste(img_graph, (0, new_h_title))
            
            # 기존 그래프 파일에 덮어쓰기
            if graph_path.suffix.lower() == ".webp":
                merged_img.convert("RGB").save(graph_path, "WEBP", quality=90)
            else:
                merged_img.convert("RGB").save(graph_path, quality=90)
            print(f"타이틀 병합 완료: {graph_path} (제목: {ACCOUNT_TITLES.get(title_key, title_key)})")
    except Exception as exc:
        print(f"(경고) 타이틀 병합 실패: {graph_path} ({exc})")
    finally:
        if temp_title_path.exists():
            temp_title_path.unlink()
            
    return graph_path


def _save_canvas(fig: plt.Figure,
                 output_path: Path,
                 message: str,
                 *,
                 pad_inches: float = 0.5,
                 bbox: Optional[str] = "tight",) -> Path:
    """생성된 Matplotlib 차트를 파일로 저장한다."""
    fig.savefig(
        output_path,
        format=output_path.suffix.lstrip(".") or "png",
        bbox_inches=bbox,
        pad_inches=pad_inches,
        facecolor=fig.get_facecolor(),
        edgecolor="none",
    )
    plt.close(fig)
    print(message)
    return output_path


def _crop_top_inches(image_path: Path, inches: float, dpi: int = FIG_DPI) -> None:
    """저장된 이미지를 불러 상단 특정 인치만큼 잘라낸다."""
    if inches <= 0:
        return
    try:
        with Image.open(image_path) as img:
            width, height = img.size
            pixels = int(round(inches * dpi))
            if pixels <= 0 or pixels >= height:
                return
            cropped = img.crop((0, pixels, width, height))
            cropped.save(image_path)
    except Exception as exc:
        print(f"(경고) 이미지 상단 자르기 실패: {image_path} ({exc})")


def format_korean_amount(val: float) -> str:
    """원화 금액 숫자를 한글 단위(억, 만) 표현으로 변환한다. (예: 100,000,000 -> 1억, 50,000,000 -> 0.5억)"""
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


def plot_exchange_rate_table(fx_series: pd.Series,
                             reference_date: pd.Timestamp,
                             output_path: Path) -> Path:
    """최근 환율과 전일 증감, 직전 3년 평균 환율을 2행 3열 테이블로 그려 저장한다."""
    _configure_matplotlib()
    if fx_series.empty:
        raise ValueError("환율 데이터가 없습니다.")

    cfg = LAYOUT.get("exchange_rate", {})
    canvas_w = float(cfg.get("canvas_width", 12.0))
    pad_top = float(cfg.get("pad_top", 0.10))
    pad_bottom = float(cfg.get("pad_bottom", 0.15))
    row_h = float(cfg.get("row_height", 0.40))
    table_fontsize = float(cfg.get("table_fontsize", 13.0))

    total_rows = 2  # 헤더 1행 + 데이터 1행
    table_h = total_rows * row_h
    fig_height = pad_top + table_h + pad_bottom

    table_y = pad_bottom / fig_height
    table_h_ratio = table_h / fig_height

    fx_series = fx_series.sort_index()
    latest_date = pd.to_datetime(fx_series.index.max())
    ref_date = pd.Timestamp(reference_date) if reference_date is not None else latest_date
    window_end = min(ref_date, latest_date)
    window_start = window_end - pd.DateOffset(years=3)
    window_series = fx_series.loc[:window_end]
    recent_series = window_series.loc[window_series.index >= window_start]
    avg_3y = float(recent_series.mean()) if not recent_series.empty else float(window_series.mean())
    current_rate = float(fx_series.iloc[-1])
    prev_rate = float(fx_series.iloc[-2]) if len(fx_series) > 1 else current_rate
    change = current_rate - prev_rate
    change_pct = (change / prev_rate * 100) if prev_rate else 0.0

    headers = ["환율 (원/USD)", "증감 (전일 대비)", "직전 3년 평균 환율"]
    values = [
        f"{current_rate:,.2f}원",
        f"{change:+.2f}원 ({change_pct:+.2f}%)",
        f"{avg_3y:,.2f}원",
    ]

    fig, ax = plt.subplots(figsize=(canvas_w, fig_height), dpi=FIG_DPI)
    fig.patch.set_facecolor(CANVAS_BG_COLOR)
    ax.axis("off")

    table = ax.table(
        cellText=[values],
        colLabels=headers,
        cellLoc="center",
        loc="center",
        bbox=[0.0, table_y, 1.0, table_h_ratio],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(table_fontsize)

    header_color = "#2d3436"
    even_color = "#fffdf5"
    odd_color = "#f6f0e6"
    gain_color = "#b42318"
    loss_color = "#1d4ed8"

    uniform_h = 1.0 / total_rows

    for (row, col), cell in table.get_celld().items():
        cell.set_height(uniform_h)
        cell.set_y((total_rows - 1 - row) * uniform_h)
        cell.get_text().set_va("center")
        cell.get_text().set_y(0.5)
        cell.set_edgecolor("#dddddd")
        if row == 0:
            cell.set_facecolor(header_color)
            cell.set_text_props(color="white", weight="bold")
        else:
            shade = even_color if col % 2 == 0 else odd_color
            cell.set_facecolor(shade)
            cell.set_text_props(color="#1f2933", weight="bold")
            if col == 1:
                color = gain_color if change > 0 else loss_color if change < 0 else "#1f2933"
                cell.set_text_props(color=color, weight="bold")

    _save_canvas(fig, output_path, f"환율 테이블 저장 완료: {output_path}", pad_inches=0.15, bbox="tight")
    return True


DEFAULT_MARKET_KPI_CONFIG = [
    {"label": "S&P500", "ticker": "^GSPC", "decimals": 2},
    {"label": "나스닥100", "ticker": "^NDX", "decimals": 2},
    {"label": "SCHD", "ticker": "SCHD", "decimals": 2},
    {"label": "IEF", "ticker": "IEF", "decimals": 2},
    {"label": "코스피", "ticker": "^KS11", "decimals": 2},
    {"label": "코스닥", "ticker": "^KQ11", "decimals": 2},
]


def load_market_indices_config() -> List[Dict[str, Any]]:
    """fa.yaml에서 market_indices_config를 로드하거나 기본값을 반환한다."""
    if FINANCIALASSETS_YAML_PATH.exists():
        try:
            with FINANCIALASSETS_YAML_PATH.open("r", encoding="utf-8") as fp:
                data = yaml.safe_load(fp) or {}
                cfg_list = data.get("market_indices_config")
                if cfg_list and isinstance(cfg_list, list):
                    return cfg_list
        except Exception:
            pass
    return DEFAULT_MARKET_KPI_CONFIG


def plot_market_indices_table(output_path: Path, target_date: Optional[pd.Timestamp] = None) -> Path:
    """주요 시장 인덱스(S&P500, 나스닥100, SCHD, IEF, 코스피, 코스닥 등)의 현재가와 증감수치, 증감률을 4행 테이블로 그려 저장한다."""
    _configure_matplotlib()
    cfg_layout = LAYOUT.get("market_indices", {})
    canvas_w = float(cfg_layout.get("canvas_width", 12.0))
    pad_top = float(cfg_layout.get("pad_top", 0.10))
    pad_bottom = float(cfg_layout.get("pad_bottom", 0.15))
    row_h = float(cfg_layout.get("row_height", 0.40))
    table_fontsize = float(cfg_layout.get("table_fontsize", 13.0))

    total_rows = 4  # colLabels 1행 + 현재가 1행 + 증감수치 1행 + 증감률 1행
    table_h = total_rows * row_h
    fig_height = pad_top + table_h + pad_bottom

    table_y = pad_bottom / fig_height
    table_h_ratio = table_h / fig_height

    index_configs = load_market_indices_config()
    tickers = [c["ticker"] for c in index_configs]

    price_map: Dict[str, Tuple[Optional[float], Optional[float]]] = {}
    try:
        price_map = fetch_latest_prices(tickers)
    except Exception as e:
        print(f"(경고) 시장 지표 가격 조회 실패: {e}")

    headers = [str(c["label"]) for c in index_configs]
    price_row = []
    diff_row = []
    pct_row = []

    for c in index_configs:
        ticker = c["ticker"]
        decimals = int(c.get("decimals", 2))
        scale = float(c.get("scale", 1.0))
        pair = price_map.get(ticker)
        if pair and pair[0] is not None:
            cur_p = pair[0] * scale
            prev_p = (pair[1] * scale) if pair[1] is not None else cur_p
            diff = cur_p - prev_p
            diff_pct = (diff / prev_p * 100.0) if prev_p > 0 else 0.0

            price_row.append(f"{cur_p:,.{decimals}f}")
            sign = "+" if diff > 0 else ""
            diff_row.append(f"{sign}{diff:,.{decimals}f}")
            pct_row.append(f"{sign}{diff_pct:.2f}%")
        else:
            price_row.append("-")
            diff_row.append("-")
            pct_row.append("-")

    fig, ax = plt.subplots(figsize=(canvas_w, fig_height), dpi=FIG_DPI)
    fig.patch.set_facecolor(CANVAS_BG_COLOR)
    ax.axis("off")

    table_data = [price_row, diff_row, pct_row]
    table = ax.table(
        cellText=table_data,
        colLabels=headers,
        cellLoc="center",
        loc="center",
        bbox=[0.0, table_y, 1.0, table_h_ratio],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(table_fontsize)

    uniform_h = 1.0 / total_rows

    for (row, col), cell in table.get_celld().items():
        cell.set_height(uniform_h)
        cell.set_y((total_rows - 1 - row) * uniform_h)
        cell.get_text().set_va("center")
        cell.get_text().set_y(0.5)
        cell.set_edgecolor("#dddddd")

        if row == 0:
            cell.set_facecolor("#2d3436")
            cell.set_text_props(color="white", weight="bold")
        elif row == 1:
            cell.set_facecolor("#fffdf5")
            cell.set_text_props(color="#1f2933", weight="bold")
        elif row in (2, 3):
            cell.set_facecolor("#f9fafb" if col % 2 == 0 else "#fffdf5")
            val_text = diff_row[col] if col < len(diff_row) else ""
            if val_text.startswith("+"):
                cell.set_text_props(color="#b42318", weight="bold")
            elif val_text.startswith("-"):
                cell.set_text_props(color="#1d4ed8", weight="bold")
            else:
                cell.set_text_props(color="#1f2933", weight="bold")

    _save_canvas(fig, output_path, f"시장 지표 테이블 저장 완료: {output_path}", pad_inches=0.15, bbox="tight")
    return output_path


def plot_assets_trend(account_df: pd.DataFrame, output_path: Path) -> Path:
    """계좌별 자산 흐름을 꺾은선 그래프로 그려 저장한다."""
    _configure_matplotlib()
    cfg = LAYOUT.get("assets_trend", {})
    canvas_w = float(cfg.get("canvas_width", 12.0))
    canvas_h = float(cfg.get("canvas_height", 6.0))
    legend_fontsize = float(cfg.get("legend_fontsize", 12.0))
    line_width = float(cfg.get("linewidth", 2.0))

    fig, ax = plt.subplots(figsize=(canvas_w, canvas_h), dpi=FIG_DPI)
    fig.patch.set_facecolor(CANVAS_BG_COLOR)
    ax.set_facecolor(CANVAS_BG_COLOR)

    columns = account_df.columns.tolist()
    labels = [account_label(col) for col in columns]
    color_map = plt.colormaps["tab10"](np.linspace(0, 1, max(len(columns), 1)))

    for idx, (column, label) in enumerate(zip(columns, labels)):
        ax.plot(
            account_df.index,
            account_df[column],
            label=label,
            color=color_map[idx],
            linestyle="-",
            linewidth=line_width,
        )

    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: format_korean_amount(x)))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%y%m"))
    ax.legend(loc="upper left", frameon=False, fontsize=legend_fontsize)
    ax.grid(True, axis="y", linestyle="--", alpha=0.7)
    for spine in ax.spines.values():
        spine.set_color("#dddddd")

    y_min = np.nanmin(account_df.values)
    y_max = np.nanmax(account_df.values)
    pad = (y_max - y_min) * 0.1 if y_max > y_min else max(abs(y_max), 1.0) * 0.1
    ax.set_ylim(max(0, y_min - pad), y_max + pad)
    fig.autofmt_xdate(rotation=30)

    _save_canvas(
        fig,
        output_path,
        f"계좌 추세 그래프 저장 완료: {output_path}",
        pad_inches=0.65,
        bbox="tight",
    )
    _crop_top_inches(output_path, inches=0.5)

    return True


def plot_assets_investment_trend(account_df: pd.DataFrame, invest_series: pd.Series, output_path: Path) -> Path:
    """누적 투자금과 누적 평가금 추세를 선 그래프로 그려 저장한다."""
    _configure_matplotlib()
    cfg = LAYOUT.get("assets_trend", {})
    canvas_w = float(cfg.get("canvas_width", 12.0))
    canvas_h = float(cfg.get("canvas_height", 6.0))
    legend_fontsize = float(cfg.get("legend_fontsize", 12.0))
    line_width = float(cfg.get("linewidth", 2.0))

    fig, ax = plt.subplots(figsize=(canvas_w, canvas_h), dpi=FIG_DPI)
    fig.patch.set_facecolor(CANVAS_BG_COLOR)
    ax.set_facecolor(CANVAS_BG_COLOR)

    # 1. 평가금 누적합 시계열 구하기
    total_valuation = account_df.sum(axis=1)
    
    # 2. 투자금 누적합 시계열 정렬
    invest_aligned = align_series(invest_series, account_df.index)

    # 3. 그래프 그리기
    ax.plot(
        account_df.index,
        invest_aligned,
        label="누적 투자금",
        color="#2c3e50",
        linestyle="-",
        linewidth=line_width,
    )
    ax.plot(
        account_df.index,
        total_valuation,
        label="누적 평가금",
        color="#d63031",
        linestyle="-",
        linewidth=line_width,
    )

    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: format_korean_amount(x)))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%y%m"))
    ax.legend(loc="upper left", frameon=False, fontsize=legend_fontsize)
    ax.grid(True, axis="y", linestyle="--", alpha=0.7)
    for spine in ax.spines.values():
        spine.set_color("#dddddd")

    # y축 한도 조정
    y_min = min(np.nanmin(invest_aligned), np.nanmin(total_valuation))
    y_max = max(np.nanmax(invest_aligned), np.nanmax(total_valuation))
    pad = (y_max - y_min) * 0.1 if y_max > y_min else max(abs(y_max), 1.0) * 0.1
    ax.set_ylim(max(0, y_min - pad), y_max + pad)
    fig.autofmt_xdate(rotation=30)

    _save_canvas(
        fig,
        output_path,
        f"투자금 vs 평가금 추세 그래프 저장 완료: {output_path}",
        pad_inches=0.65,
        bbox="tight",
    )
    _crop_top_inches(output_path, inches=0.5)

    return True


def extract_trades(records: pd.DataFrame) -> pd.DataFrame:
    """거래 기록에서 실제 매매(수량, 단가 존재) 데이터만 추출한다."""
    trades = records[
        (records["수량"].notna())
        & (records["단가"].notna())
        & (records["수량"] != 0)
        & (records["계좌"] != "sema")
    ].copy()
    return trades[["계좌", "일자", "종목", "단가", "수량"]]


def compute_positions(trades: pd.DataFrame, symbol_map: Dict[str, AssetConfig], fx_series: pd.Series) -> List[Position]:
    """거래 기록을 바탕으로 현재 보유 종목(수량, 총 투자 원금)을 계산한다."""
    positions: List[Position] = []
    for (account, symbol), group in trades.groupby(["계좌", "종목"]):
        config = symbol_map.get(symbol)
        if not config:
            continue
        total_qty = 0.0
        total_cost = 0.0

        for _, row in group.sort_values("일자").iterrows():
            qty = float(row["수량"])
            price = float(row["단가"])
            trade_date = pd.Timestamp(row["일자"])
            native_amount = price * qty
            krw_flow = convert_to_krw(account, native_amount, trade_date, fx_series)

            prev_qty = total_qty
            total_qty += qty
            if qty > 0:
                total_cost += krw_flow
            else:
                if prev_qty > 0:
                    avg_cost = total_cost / prev_qty if prev_qty else 0.0
                    total_cost -= avg_cost * abs(qty)

            if total_qty <= 0:
                total_qty = 0.0
                total_cost = 0.0

        if total_qty <= 0:
            continue

        positions.append(
            Position(
                account=account,
                symbol=symbol,
                ticker=config.ticker,
                quantity=total_qty,
                cost=total_cost,
            )
        )
    return positions


def fetch_latest_prices(tickers: List[str]) -> Dict[str, Tuple[float, float]]:
    """여러 티커의 가장 최근 가격과 전일 가격을 조회한다."""
    if not tickers:
        return {}
    result: Dict[str, Tuple[float, float]] = {}
    try:
        data = yf.download(
            tickers,
            period="5d",
            interval="1d",
            progress=False,
            auto_adjust=False,
        )
        if not data.empty:
            adj_close = data["Adj Close"] if "Adj Close" in data.columns else data
            if adj_close.index.tz is not None:
                adj_close.index = adj_close.index.tz_localize(None)
            adj_close.index = adj_close.index.normalize()
            
            if isinstance(adj_close, pd.Series):
                series = adj_close[~adj_close.index.duplicated(keep="last")]
                series = series.ffill().dropna()
                if not series.empty:
                    last_price = float(series.iloc[-1])
                    prev_price = float(series.iloc[-2]) if len(series) > 1 else last_price
                    result[tickers[0]] = (last_price, prev_price)
            else:
                adj_close = adj_close[~adj_close.index.duplicated(keep="last")]
                for col in adj_close.columns:
                    series = adj_close[col].dropna()
                    if series.empty:
                        continue
                    last_price = float(series.iloc[-1])
                    prev_price = float(series.iloc[-2]) if len(series) > 1 else last_price
                    result[col] = (last_price, prev_price)
    except Exception:
        pass

    # 일괄 다운로드에서 누락된 티커 개별 재시도
    missing = [t for t in tickers if t not in result]
    for t in missing:
        try:
            hist = yf.Ticker(t).history(period="5d")
            if not hist.empty and "Close" in hist.columns:
                series = hist["Close"].dropna()
                if not series.empty:
                    last_price = float(series.iloc[-1])
                    prev_price = float(series.iloc[-2]) if len(series) > 1 else last_price
                    result[t] = (last_price, prev_price)
        except Exception:
            pass

    return result


def build_holdings_df(records: pd.DataFrame, fx_series: pd.Series) -> pd.DataFrame:
    """현재 보유 자산 현황(평가금, 매수금, 수익금) 데이터프레임을 생성한다."""
    symbol_map = load_symbol_map()
    trades = extract_trades(records)
    positions = compute_positions(trades, symbol_map, fx_series)

    prices = fetch_latest_prices(sorted({p.ticker for p in positions if p.ticker}))
    for pos in positions:
        if not pos.ticker:
            prices[pos.ticker] = (1.0, 1.0)

    rows: List[Dict[str, float]] = []
    for pos in positions:
        price_info = prices.get(pos.ticker)
        if price_info is None:
            # 주가 조회 실패 시 평단가로 fallback하여 보유 종목 누락 방지
            latest_price_native = pos.avg_price_native or 0.0
            prev_price_native = latest_price_native
        else:
            latest_price_native, prev_price_native = price_info

        valuation_native = pos.quantity * latest_price_native
        valuation = convert_to_krw(pos.account, valuation_native, None, fx_series, use_latest=True)
        avg_price = pos.cost / pos.quantity if pos.quantity else 0.0
        current_price = convert_to_krw(pos.account, latest_price_native, None, fx_series, use_latest=True)
        prev_price = convert_to_krw(pos.account, prev_price_native, None, fx_series, use_latest=True)
        rows.append(
            {
                "계좌": pos.account,
                "종목": pos.symbol,
                "평가금": valuation,
                "매수금": pos.cost,
                "수익금": valuation - pos.cost,
                "수량": pos.quantity,
                "평단가": avg_price,
                "금액": pos.cost,
                "현재가": current_price,
                "등락률": None if prev_price == 0 else (current_price - prev_price) / prev_price,
            }
        )

    gongje_all = records[records["계좌"] == "sema"].copy()
    gongje_eval = gongje_all[gongje_all["평가금"].notna()].copy()
    if not gongje_eval.empty:
        invested_by_symbol = (
            gongje_all.groupby("종목")["투자금"]
            .apply(lambda s: s.fillna(0).sum())
            .to_dict()
        )
        latest_records = (
            gongje_eval.sort_values("일자")
            .groupby("종목", as_index=False)
            .last()
        )
        for _, row in latest_records.iterrows():
            valuation = float(row["평가금"])
            if valuation <= 0:
                continue
            symbol = row["종목"]
            invested = float(invested_by_symbol.get(symbol, 0.0))
            rows.append(
                {
                    "계좌": "sema",
                    "종목": symbol,
                    "평가금": valuation,
                    "매수금": invested,
                    "수익금": valuation - invested,
                }
            )

    if not rows:
        raise ValueError("평가금을 계산할 수 있는 종목이 없습니다.")

    df = pd.DataFrame(rows)
    df = df[df["평가금"] > 0].copy()
    if df.empty:
        raise ValueError("평가금이 0보다 큰 종목이 없습니다.")

    df["수익률"] = df.apply(
        lambda row: None if row["매수금"] == 0 else row["수익금"] / row["매수금"],
        axis=1,
    )

    return df


def calculate_rebalancing_df(account_name: str, account_holdings: pd.DataFrame, symbol_map: Optional[Dict[str, AssetConfig]] = None) -> Optional[pd.DataFrame]:
    """
    계좌별 보유종목 현황을 기반으로 목표 비중에 맞추기 위한 매도/매수 필요 금액 및 수량을 계산한다.
    """
    target_alloc = parse_target_allocation(account_name)
    if not target_alloc or account_holdings is None or account_holdings.empty:
        return None
        
    total_val = account_holdings["평가금"].sum()
    if total_val <= 0:
        return None
        
    holdings = account_holdings.copy()
    keys = list(target_alloc.keys())
    
    assigned_keys = []
    for _, row in holdings.iterrows():
        sym = row["종목"]
        config = symbol_map.get(sym) if symbol_map else None
        abbrev = config.abbrev if config else sym
        ac = config.asset_class if config else ""
        
        assigned_key = match_target_key(sym, abbrev, ac, keys)
        assigned_keys.append(assigned_key)
        
    holdings["target_key"] = assigned_keys
    
    curr_by_key = holdings.groupby("target_key")["평가금"].sum().to_dict()
    
    # 대표 단가 계산
    key_qty_price = {}
    for key in keys:
        sub_df = holdings[holdings["target_key"] == key]
        if not sub_df.empty:
            if "수량" in sub_df.columns and "평가금" in sub_df.columns:
                total_q = sub_df["수량"].fillna(0).sum()
                total_e = sub_df["평가금"].sum()
                if total_q > 0 and total_e > 0:
                    key_qty_price[key] = total_e / total_q
            elif "현재가" in sub_df.columns:
                key_qty_price[key] = sub_df["현재가"].iloc[0]

    rows = []
    for key, target_ratio in target_alloc.items():
        curr_amt = curr_by_key.get(key, 0.0)
        curr_pct = (curr_amt / total_val) * 100.0
        target_pct = target_ratio * 100.0
        target_amt = total_val * target_ratio
        diff_amt = target_amt - curr_amt  # 양수: 매수 필요, 음수: 매도 필요
        
        avg_p = key_qty_price.get(key, 0.0)
        diff_qty = 0
        if avg_p > 0:
            diff_qty = int(round(diff_amt / avg_p))
        
        rows.append({
            "자산군": key,
            "현재평가금": curr_amt,
            "현재비중": curr_pct,
            "목표비중": target_pct,
            "목표평가금": target_amt,
            "조정금액": diff_amt,
            "단가": avg_p,
            "조정주수": diff_qty,
        })
        
    return pd.DataFrame(rows)


@dataclass
class Position:
    """보유 종목 정보를 담는 데이터 클래스"""
    account: str
    symbol: str
    ticker: str
    quantity: float
    cost: float


def ensure_static_dir() -> None:
    """결과물을 저장할 정적 폴더가 없다면 생성한다."""
    STATIC_FINANCIALASSETS_DIR.mkdir(parents=True, exist_ok=True)


def account_label(account: str) -> str:
    """계좌 코드에 해당하는 표시용 이름을 반환한다."""
    return ACCOUNT_LABELS.get(account, account)


def _clean_numeric(series: Iterable) -> pd.Series:
    """쉼표가 포함된 문자열 숫자를 float로 변환한다."""
    ser = pd.Series(series, dtype="string").str.replace(",", "", regex=False)
    return pd.to_numeric(ser, errors="coerce")


def read_trading_records() -> pd.DataFrame:
    """fa_records.db (SQLite) 또는 trading_records.csv 파일을 읽어 DataFrame으로 반환한다."""
    db_path = ROOT_DIR / "db" / "fa_records.db"
    if db_path.exists():
        import sqlite3
        try:
            conn = sqlite3.connect(db_path)
            query = """
            SELECT date AS 일자, account AS 계좌, symbol AS 종목, kind AS 구분,
                   unit_price AS 단가, quantity AS 수량, amount AS 금액,
                   dividend AS 배당, deposit AS 투자금, evaluation AS 평가금,
                   exchange_rate AS 환율, memo AS 비고
            FROM trading_records
            ORDER BY date ASC, id ASC;
            """
            df = pd.read_sql_query(query, conn)
            conn.close()
            if not df.empty:
                df["일자"] = pd.to_datetime(df["일자"].astype(str).str.replace(".", "-", regex=False), format="mixed", errors="coerce")
                numeric_cols = ["단가", "수량", "배당", "투자금", "평가금", "환율", "금액"]
                for col in numeric_cols:
                    if col in df.columns:
                        df[col] = pd.to_numeric(df[col], errors="coerce")
                return df
        except Exception as e:
            print(f"(경고) SQLite DB 로드 실패, CSV 폴백 시도: {e}")

    if not TRADING_RECORDS_PATH.exists():
        raise FileNotFoundError(f"trading_records.csv 파일을 찾을 수 없습니다: {TRADING_RECORDS_PATH}")

    df = pd.read_csv(TRADING_RECORDS_PATH, encoding="utf-8-sig")
    # 2022.01.31, 2022-01-31 등 다양한 날짜 입력을 공통 포맷으로 정규화한다.
    df["일자"] = (
        pd.Series(df["일자"], dtype="string")
        .str.replace(".", "-", regex=False)
        .str.replace("/", "-", regex=False)
    )
    df["일자"] = pd.to_datetime(df["일자"], errors="coerce")

    numeric_cols = ["단가", "수량", "배당", "투자금", "환율", "평가금"]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = _clean_numeric(df[col])

    return df


def latest_month_code(records: pd.DataFrame) -> str:
    """거래 기록에서 가장 최근 월 코드를 'YYMM' 형식으로 반환한다."""
    latest_date = records["일자"].dropna().max()
    if pd.isna(latest_date):
        return "2201"
    return f"{latest_date:%y%m}"


def _save_canvas(fig: plt.Figure, output_path: Path, log_message: str, pad_inches: float = 0.2, bbox: str = "tight") -> None:
    """Figure 객체를 저장하고 로그 메시지를 출력한다."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=FIG_DPI, bbox_inches=bbox, pad_inches=pad_inches)
    plt.close(fig)
    print(log_message)


def _crop_top_inches(image_path: Path, inches: float = 0.4) -> None:
    """이미지 상단의 불필요한 여백을 잘라낸다."""
    if not image_path.exists():
        return

    with Image.open(image_path) as img:
        width, height = img.size
        crop_pixels = int(inches * FIG_DPI)
        if crop_pixels >= height:
            return

        cropped = img.crop((0, crop_pixels, width, height))
        cropped.save(image_path)


def plot_account_detail(account: str, 
                        holdings_df: pd.DataFrame, 
                        status_df: pd.DataFrame, 
                        output_path: Path,
                        account_df: Optional[pd.DataFrame] = None) -> bool:
    """개별 계좌의 상세 내역(파이 차트, 보유 종목 테이블, 리밸런싱 텍스트 가이드)을 그려 저장한다."""
    # 계좌 종목
    account_holdings = holdings_df[holdings_df["계좌"] == account].copy()
    if account_holdings.empty:
        return False

    # 계좌 현황
    status_row = status_df[status_df["계좌"] == account]
    if status_row.empty:
        return False
    status_row = status_row.iloc[0]

    symbol_map = load_symbol_map()
    raw_account_name = ACCOUNT_RAW_NAMES.get(account)
    if not raw_account_name:
        title_val = ACCOUNT_TITLES.get(f"title_{account}_detail", "")
        raw_account_name = title_val.replace("◉ 상세계좌: ", "").strip()
        
    rebal_df = calculate_rebalancing_df(raw_account_name, account_holdings, symbol_map)
    has_rebal = rebal_df is not None and not rebal_df.empty

    # 전월 대비 증감액 및 증감율 계산
    mom_diff = None
    mom_pct = None
    if account_df is not None and not account_df.empty and account in account_df.columns:
        s = account_df[account].dropna()
        if len(s) >= 2:
            cur_val = float(s.iloc[-1])
            prev_val = float(s.iloc[-2])
            mom_diff = cur_val - prev_val
            if prev_val > 0:
                mom_pct = (mom_diff / prev_val) * 100.0
        elif len(s) == 1:
            mom_diff = 0.0
            mom_pct = 0.0

    account_holdings = account_holdings.sort_values("평가금", ascending=False)
    colors = plt.colormaps["tab10"](np.linspace(0, 1, max(len(account_holdings), 1)))

    # 레이아웃 설정 로드
    cfg = LAYOUT.get("account_detail", {})
    canvas_w = float(cfg.get("canvas_width", 12.5))
    min_canvas_h = float(cfg.get("min_canvas_height", 4.8))
    pie_table_ratio = cfg.get("pie_table_ratio", [0.65, 1.35])
    pad_top = float(cfg.get("pad_top", 0.25))
    pad_bottom = float(cfg.get("pad_bottom", 0.25))
    pie_pad_top = float(cfg.get("pie_pad_top", 0.0))
    pie_radius = float(cfg.get("pie_radius", 1.75))
    pie_mom_fs = float(cfg.get("pie_mom_fontsize", 11.5))
    pie_mom_gap = float(cfg.get("pie_mom_gap", 0.20))
    subtitle_h = float(cfg.get("subtitle_height", 0.28))
    subtitle_margin_top = float(cfg.get("subtitle_margin_top", 0.10))
    subtitle_margin_bottom = float(cfg.get("subtitle_margin_bottom", cfg.get("subtitle_margin", 0.10)))
    subtitle_fs = float(cfg.get("subtitle_fontsize", 13.0))
    row_h = float(cfg.get("row_height", 0.38))
    table_fs = float(cfg.get("table_fontsize", 12.0))
    main_table_fs = float(cfg.get("main_table_fontsize", table_fs))
    summary_table_fs = float(cfg.get("summary_table_fontsize", table_fs))
    section_gap = float(cfg.get("section_gap", 0.25))
    rebal_line_h = float(cfg.get("rebal_line_height", 0.35))
    rebal_fs = float(cfg.get("rebal_fontsize", 11.5))
    pie_pct_fs = float(cfg.get("pie_pct_fontsize", 13.0))

    # 종목별 데이터 테이블 가공
    table_df = account_holdings.copy()
    main_data = table_df[["종목", "매수금", "평가금", "수익금", "수익률"]].copy()
    main_data["매수금"] = main_data["매수금"].apply(lambda x: f"{x:,.0f}")
    main_data["평가금"] = main_data["평가금"].apply(lambda x: f"{x:,.0f}")
    main_data["수익금"] = main_data["수익금"].apply(lambda x: f"{x:,.0f}")
    main_data["수익률"] = main_data["수익률"].apply(lambda x: "-" if x is None or pd.isna(x) else f"{x * 100:.2f}%")

    n_main_rows = len(main_data)
    h_main_table = (n_main_rows + 1) * row_h
    h_summary_table = 2 * row_h
    n_rebal_rows = len(rebal_df) if has_rebal else 0
    h_rebal_section = (subtitle_margin_top + subtitle_h + subtitle_margin_bottom + n_rebal_rows * rebal_line_h) if has_rebal else 0.0

    # 전체 내용 높이 및 캔버스 높이 산출 (서브타이틀 마진 및 파이 높이 고려)
    h_section_summary = subtitle_margin_top + subtitle_h + subtitle_margin_bottom + h_summary_table
    h_section_main = subtitle_margin_top + subtitle_h + subtitle_margin_bottom + h_main_table
    h_pie_mom = (pie_mom_gap + 0.55) if mom_diff is not None else 0.0
    h_pie_needed = pad_top + pie_pad_top + (2 * pie_radius) + h_pie_mom + pad_bottom

    total_content_height = max(
        h_pie_needed,
        pad_top
        + h_section_summary
        + section_gap
        + h_section_main
        + (section_gap + h_rebal_section if has_rebal else 0.0)
        + pad_bottom,
    )
    fig_height = max(min_canvas_h, total_content_height)
    extra_pad = (fig_height - total_content_height) / 2.0 if fig_height > total_content_height else 0.0

    _configure_matplotlib()
    fig, axes = plt.subplots(
        1,
        2,
        figsize=(canvas_w, fig_height),
        dpi=FIG_DPI,
        gridspec_kw={"width_ratios": pie_table_ratio},
    )
    fig.patch.set_facecolor(CANVAS_BG_COLOR)
    ax_pie, ax_table = axes
    ax_pie.set_facecolor(CANVAS_BG_COLOR)

    # 파이 차트 상단 정렬 배치
    total_ratio = float(pie_table_ratio[0] + pie_table_ratio[1])
    w_pie_inch = canvas_w * (float(pie_table_ratio[0]) / total_ratio)
    target_pie_top_y = fig_height - pad_top - extra_pad - pie_pad_top
    pie_center_y = target_pie_top_y - pie_radius

    wedges, _, autotexts = ax_pie.pie(
        account_holdings["평가금"],
        labels=None,
        autopct=lambda pct: f"{pct:.1f}%" if pct >= 3 else "",
        startangle=90,
        colors=colors,
        radius=pie_radius,
        center=(0, pie_center_y),
        textprops={"fontsize": pie_pct_fs, "color": "white", "weight": "bold"},
    )
    
    # 파이 차트 하단 전월대비 증감액/증감율 뱃지
    if mom_diff is not None:
        pie_bottom_y = pie_center_y - pie_radius
        mom_y = pie_bottom_y - pie_mom_gap
        sign = "+" if mom_diff > 0 else ""
        pct_str = f"({sign}{mom_pct:.2f}%)" if mom_pct is not None else ""
        val_str = f"전월 대비  {sign}{mom_diff:,.0f}원 {pct_str}".strip()
        color_code = "#b42318" if mom_diff > 0 else ("#1d4ed8" if mom_diff < 0 else "#2c3e50")
        
        ax_pie.text(
            0,
            mom_y,
            val_str,
            ha="center",
            va="top",
            fontsize=pie_mom_fs,
            fontweight="bold",
            color=color_code,
            bbox=dict(boxstyle="round,pad=0.4,rounding_size=0.3", facecolor="#ffffff", edgecolor="#d1d5db", linewidth=1.0),
        )

    ax_pie.set_xlim(-w_pie_inch / 2.0, w_pie_inch / 2.0)
    ax_pie.set_ylim(0.0, fig_height)
    ax_pie.set_aspect("equal", adjustable="box")
    ax_pie.axis("off")

    color_map = dict(zip(account_holdings["종목"], colors))
    ax_table.axis("off")

    # Top-Down 상대 좌표 계산
    curr_y = fig_height - pad_top - extra_pad

    header_colors = ["#2d3436", "#2d3436", "#2d3436", "#2d3436", "#2d3436"]
    neutral_header = "#2d3436"

    # 1) 전체 요약 서브타이틀
    curr_y -= subtitle_margin_top
    summary_title_y = curr_y / fig_height
    ax_table.text(
        0.0,
        summary_title_y,
        "계좌 현황 (전체)",
        transform=ax_table.transAxes,
        ha="left",
        va="top",
        fontsize=subtitle_fs,
        fontweight="bold",
        color="#2c3e50",
    )
    curr_y -= subtitle_h
    curr_y -= subtitle_margin_bottom

    # 2) 계좌 현황 요약 테이블
    curr_y -= h_summary_table
    summary_table_bbox = [0.0, curr_y / fig_height, 1.0, h_summary_table / fig_height]

    summary_columns = ["투자금", "평가금", "수익금", "수익률", "배당금(누적)"]
    summary_data = pd.DataFrame(
        [
            [
                f"{status_row.get('투자금', 0):,.0f}",
                f"{status_row.get('평가금', 0):,.0f}",
                f"{status_row.get('수익금', 0):,.0f}",
                "-"
                if not status_row.get("투자금")
                else f"{(status_row.get('수익금', 0) / status_row.get('투자금')) * 100:.2f}%",
                f"{status_row.get('배당금', 0):,.0f}",
            ]
        ],
        columns=summary_columns,
    )

    summary_table = ax_table.table(
        cellText=summary_data.values,
        colLabels=summary_columns,
        cellLoc="center",
        loc="upper center",
        bbox=summary_table_bbox,
    )
    summary_table.auto_set_font_size(False)
    summary_table.set_fontsize(summary_table_fs)

    summary_uniform_height = 0.5
    for (row, col), cell in summary_table.get_celld().items():
        cell.set_height(summary_uniform_height)
        cell.set_y((1 - row) * summary_uniform_height)
        cell.get_text().set_va("center")
        cell.get_text().set_y(0.5)
        cell.set_edgecolor("#dddddd")
        if row == 0:
            cell.get_text().set_ha("center")
            color = header_colors[col] if col < len(header_colors) else neutral_header
            cell.set_facecolor(color)
            cell.set_text_props(color="white", weight="bold")
        else:
            cell.get_text().set_ha("right")
            shade = "#fffdf5" if (row % 2 == 1) else "#f6f0e6"
            cell.set_facecolor(shade)
            cell.set_text_props(color="#2c3e50", weight="bold")

    # 3) 종목별 서브타이틀
    curr_y -= section_gap
    curr_y -= subtitle_margin_top
    main_title_y = curr_y / fig_height
    ax_table.text(
        0.0,
        main_title_y,
        "계좌 현황 (종목별)",
        transform=ax_table.transAxes,
        ha="left",
        va="top",
        fontsize=subtitle_fs,
        fontweight="bold",
        color="#2c3e50",
    )
    curr_y -= subtitle_h
    curr_y -= subtitle_margin_bottom

    # 4) 메인 종목 테이블
    curr_y -= h_main_table
    main_table_bbox = [0.0, curr_y / fig_height, 1.0, h_main_table / fig_height]

    main_table = ax_table.table(
        cellText=main_data.values,
        colLabels=main_data.columns,
        cellLoc="center",
        loc="upper center",
        bbox=main_table_bbox,
    )
    main_table.auto_set_font_size(False)
    main_table.set_fontsize(main_table_fs)

    main_uniform_height = 1.0 / (n_main_rows + 1)
    for (row, col), cell in main_table.get_celld().items():
        cell.set_height(main_uniform_height)
        cell.set_y((n_main_rows - row) * main_uniform_height)
        cell.get_text().set_va("center")
        cell.get_text().set_y(0.5)
        cell.set_edgecolor("#dddddd")
        if row == 0:
            cell.get_text().set_ha("center")
            color = header_colors[col] if col < len(header_colors) else neutral_header
            cell.set_facecolor(color)
            cell.set_text_props(color="white", weight="bold")
        else:
            cell.get_text().set_ha("center" if col == 0 else "right")
            shade = "#fffdf5" if (row % 2 == 0) else "#f6f0e6"
            if col == 0:
                label = main_data.iloc[row - 1, 0]
                shade = color_map.get(label, shade)
                cell.set_text_props(color="white", weight="bold")
            else:
                col_name = main_data.columns[col]
                if col_name in ("수익금", "수익률"):
                    val_str = str(main_data.iloc[row - 1, col]).strip()
                    cleaned = re.sub(r"[^\d.-]", "", val_str)
                    try:
                        num_val = float(cleaned)
                        if num_val > 0:
                            cell.get_text().set_color("#b42318")
                        elif num_val < 0:
                            cell.get_text().set_color("#1d4ed8")
                        else:
                            cell.get_text().set_color("#2c3e50")
                    except ValueError:
                        cell.get_text().set_color("#2c3e50")
                else:
                    cell.get_text().set_color("#2c3e50")
            cell.set_facecolor(shade)



    # 5) 리밸런싱 가이드 텍스트 (목표 비중이 정의된 계좌인 경우)
    if has_rebal:
        curr_y -= section_gap
        curr_y -= subtitle_margin_top
        rebal_title_y = curr_y / fig_height
        ax_table.text(
            0.0,
            rebal_title_y,
            "리밸런싱 가이드 (목표 비중 대비)",
            transform=ax_table.transAxes,
            ha="left",
            va="top",
            fontsize=subtitle_fs,
            fontweight="bold",
            color="#2c3e50",
        )
        curr_y -= subtitle_h
        curr_y -= subtitle_margin_bottom
        
        for _, row in rebal_df.iterrows():
            line_y = curr_y / fig_height
            diff = row["조정금액"]
            diff_q = row.get("조정주수", 0)
            asset_name = str(row["자산군"])
            
            if diff > 100:
                qty_str = f" (+{diff_q:g}주 매수)" if diff_q > 0 else " (매수 필요)"
                line_str = f"-  {asset_name} : +{diff:,.0f}원{qty_str}"
                color_code = "#2980b9"
            elif diff < -100:
                qty_str = f" ({diff_q:g}주 매도)" if diff_q < 0 else " (매도 필요)"
                line_str = f"-  {asset_name} : -{abs(diff):,.0f}원{qty_str}"
                color_code = "#b42318"
            else:
                line_str = f"-  {asset_name} : 비중 적정 (0원)"
                color_code = "#27ae60"
                
            ax_table.text(
                0.02,
                line_y,
                line_str,
                transform=ax_table.transAxes,
                ha="left",
                va="top",
                fontsize=rebal_fs,
                fontweight="bold",
                color=color_code,
            )
            curr_y -= rebal_line_h

    display_name = account_label(account)
    _save_canvas(fig, output_path, f"{display_name} 상세 그래프 저장 완료: {output_path}", pad_inches=0.2, bbox="tight")
    return True


def load_dividend_pivot(records: pd.DataFrame, fx_series: pd.Series, end_date: pd.Timestamp) -> pd.DataFrame:
    """최근 12개월간의 월별, 종목별 배당금 피벗 테이블을 생성한다."""
    df = records.copy()
    df["배당"] = pd.to_numeric(df.get("배당"), errors="coerce")
    dividends = df[(df["배당"].notna()) & (df["배당"] > 0)].copy()
    if dividends.empty:
        raise ValueError("배당 데이터가 없습니다.")

    cutoff = (end_date - pd.DateOffset(months=12)).replace(day=1)
    dividends = dividends[dividends["일자"] >= cutoff]
    if dividends.empty:
        raise ValueError("최근 1년간 배당 데이터가 없습니다.")

    dividends["배당원화"] = dividends.apply(
        lambda row: convert_to_krw(row["계좌"], row["배당"], row["일자"], fx_series),
        axis=1,
    )
    dividends["월"] = dividends["일자"].dt.to_period("M").dt.to_timestamp()

    pivot = (
        dividends.groupby(["월", "종목"])["배당원화"]
        .sum()
        .unstack(fill_value=0)
        .sort_index()
    )
    pivot = pivot.loc[:, (pivot != 0).any(axis=0)]
    return pivot


def load_yearly_dividend_pivot(records: pd.DataFrame,
                               fx_series: pd.Series,
                               end_date: pd.Timestamp,
                               years: int = 5) -> pd.DataFrame:
    """최근 N년간의 연별, 종목별 배당금 피벗 테이블을 생성한다."""
    df = records.copy()
    df["배당"] = pd.to_numeric(df.get("배당"), errors="coerce")
    dividends = df[(df["배당"].notna()) & (df["배당"] > 0)].copy()
    if dividends.empty:
        raise ValueError("배당 데이터가 없습니다.")

    cutoff = (end_date - pd.DateOffset(years=years)).replace(month=1, day=1)
    dividends = dividends[dividends["일자"] >= cutoff]
    if dividends.empty:
        raise ValueError("최근 배당 데이터가 없습니다.")

    dividends["배당원화"] = dividends.apply(
        lambda row: convert_to_krw(row["계좌"], row["배당"], row["일자"], fx_series),
        axis=1,
    )
    dividends["연도"] = dividends["일자"].dt.to_period("Y").dt.to_timestamp()

    pivot = (
        dividends.groupby(["연도", "종목"])["배당원화"]
        .sum()
        .unstack(fill_value=0)
        .sort_index()
    )
    pivot = pivot.loc[:, (pivot != 0).any(axis=0)]
    return pivot


def _build_investment_series(records: pd.DataFrame, fx_series: pd.Series) -> pd.Series:
    """투자금 누적 시계열(원화)을 생성한다."""
    invest_records = records[records["투자금"].notna()].copy()
    if invest_records.empty:
        return pd.Series(dtype=float)
    invest_records = invest_records[["일자", "계좌", "투자금"]].copy()
    # 투자금은 csv 상에 이미 원화(KRW)로 기입되어 있으므로 환율을 곱하지 않고 그대로 사용합니다.
    invest_records["투자금원화"] = invest_records["투자금"]
    invest_records = invest_records.dropna(subset=["일자"])
    daily = invest_records.groupby("일자")["투자금원화"].sum().sort_index()
    return daily.cumsum()





def plot_monthly_dividends(pivot: pd.DataFrame, output_path: Path) -> Path:
    """월별 배당금 현황을 누적 막대 그래프로 그려 저장한다."""
    months = pivot.index.to_pydatetime()
    columns = pivot.columns.tolist()

    _configure_matplotlib()
    cfg = LAYOUT.get("dividends", {})
    canvas_w = float(cfg.get("canvas_width", 12.0))
    canvas_h = float(cfg.get("canvas_height", 6.0))
    bar_label_fs = float(cfg.get("bar_label_fontsize", 12.0))
    tick_fs = float(cfg.get("tick_fontsize", 12.0))
    legend_fs = float(cfg.get("legend_fontsize", 12.0))

    fig, ax = plt.subplots(figsize=(canvas_w, canvas_h), dpi=FIG_DPI)
    fig.patch.set_facecolor(CANVAS_BG_COLOR)
    ax.set_facecolor(CANVAS_BG_COLOR)
    bottoms = np.zeros(len(months))
    bar_width = 20
    colors = plt.colormaps["tab10"](np.linspace(0, 1, max(len(columns), 1)))

    for col, color in zip(columns, colors):
        values = pivot[col].values
        ax.bar(
            months,
            values,
            width=bar_width,
            bottom=bottoms,
            label=col,
            color=color,
            edgecolor="white",
        )
        bottoms += values

    totals = pivot.sum(axis=1).values
    if len(totals) > 0 and max(totals) > 0:
        ax.set_ylim(0, max(totals) * 1.08)

    for i, total in enumerate(totals):
        if total <= 0:
            continue
        ax.text(
            months[i],
            total + (total * 0.02),
            f"{total:,.0f}",
            ha="center",
            va="bottom",
            fontsize=bar_label_fs,
            color="black",
        )

    ax.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{int(x):,}"))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%y%m"))
    ax.tick_params(axis="x", rotation=45, labelsize=tick_fs)
    ax.tick_params(axis="y", labelsize=tick_fs)
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    for spine in ax.spines.values():
        spine.set_color("#dddddd")
    
    legend_ncol = int(cfg.get("legend_ncol", 7))
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.18),
        ncol=legend_ncol,
        frameon=False,
        fontsize=legend_fs,
        handlelength=1.2,
        columnspacing=1.0,
    )
    _save_canvas(
        fig,
        output_path,
        f"월별 배당 그래프 저장 완료: {output_path}",
        pad_inches=0.25,
        bbox="tight",
    )

    return True


def plot_yearly_dividends(pivot: pd.DataFrame, output_path: Path) -> Path:
    """연별 배당금 현황을 막대 그래프로 그려 저장한다. (과거 년도는 단일 색상, 당해년도만 종목별 스택 및 범례 표시)"""
    years = pivot.index.to_pydatetime()
    if len(years) == 0:
        raise ValueError("연별 배당 데이터가 없습니다.")

    _configure_matplotlib()
    cfg = LAYOUT.get("dividends", {})
    canvas_w = float(cfg.get("canvas_width", 12.0))
    canvas_h = float(cfg.get("canvas_height", 6.0))
    bar_label_fs = float(cfg.get("bar_label_fontsize", 12.0))
    tick_fs = float(cfg.get("tick_fontsize", 12.0))
    legend_fs = float(cfg.get("legend_fontsize", 10.5))

    fig, ax = plt.subplots(figsize=(canvas_w, canvas_h), dpi=FIG_DPI)
    fig.patch.set_facecolor(CANVAS_BG_COLOR)
    ax.set_facecolor(CANVAS_BG_COLOR)

    # 1. 과거 년도: 단일 색상으로 총 배당금 단일 막대 렌더링
    past_color = "#94a3b8"  # 차분한 슬레이트 그레이 단일 색상
    for i in range(len(years) - 1):
        year_total = float(pivot.iloc[i].sum())
        if year_total > 0:
            ax.bar(
                years[i],
                year_total,
                width=220,
                color=past_color,
                edgecolor="white",
            )

    # 2. 당해 년도 (현재 연도): 당해년도에 배당이 발생한 종목만 스택 막대 렌더링
    current_row = pivot.iloc[-1]
    current_symbols = [col for col in pivot.columns if current_row[col] > 0]
    colors = plt.colormaps["tab10"](np.linspace(0, 1, max(len(current_symbols), 1)))

    curr_bottom = 0.0
    for col, color in zip(current_symbols, colors):
        val = float(current_row[col])
        ax.bar(
            years[-1],
            val,
            width=220,
            bottom=curr_bottom,
            label=col,
            color=color,
            edgecolor="white",
        )
        curr_bottom += val

    # 3. 모든 연도 상단 총 배당금 텍스트 표시
    totals = pivot.sum(axis=1).values
    if len(totals) > 0 and max(totals) > 0:
        ax.set_ylim(0, max(totals) * 1.08)

    for i, total in enumerate(totals):
        if total <= 0:
            continue
        ax.text(
            years[i],
            total + (total * 0.02),
            f"{total:,.0f}",
            ha="center",
            va="bottom",
            fontsize=bar_label_fs,
            color="black",
        )

    ax.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{int(x):,}"))
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.tick_params(axis="x", rotation=0, labelsize=tick_fs)
    ax.tick_params(axis="y", labelsize=tick_fs)
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    for spine in ax.spines.values():
        spine.set_color("#dddddd")

    # 4. 당해년도 종목만 하단 범례 표시
    legend_ncol = int(cfg.get("legend_ncol", 7))
    if current_symbols:
        ax.legend(
            loc="upper center",
            bbox_to_anchor=(0.5, -0.15),
            ncol=legend_ncol,
            frameon=False,
            fontsize=legend_fs,
            handlelength=1.2,
            columnspacing=1.0,
        )

    _save_canvas(
        fig,
        output_path,
        f"연별 배당 그래프 저장 완료: {output_path}",
        pad_inches=0.25,
        bbox="tight",
    )

    return True




def plot_total_holdings(holdings_df: pd.DataFrame, output_path: Path) -> Path:
    """전체 보유 종목 현황을 표 형태로 그려 저장한다."""
    filtered = holdings_df[holdings_df["계좌"] != "sema"].copy()
    if filtered.empty:
        raise ValueError("보유 중인 종목이 없습니다.")

    cfg = LAYOUT.get("total_holdings", {})
    canvas_w = float(cfg.get("canvas_width", 12.0))
    pad_top = float(cfg.get("pad_top", 0.25))
    pad_bottom = float(cfg.get("pad_bottom", 0.25))
    row_h = float(cfg.get("row_height", 0.34))
    table_fs = float(cfg.get("table_fontsize", 11.0))
    min_canvas_h = float(cfg.get("min_canvas_height", 4.0))

    columns = ["계좌", "종목", "보유수량", "평단가", "금액", "현재가", "수익금", "수익률", "등락률"]
    formatted = filtered.copy()
    formatted["계좌"] = formatted["계좌"].apply(account_label)
    formatted["보유수량"] = formatted["수량"].apply(lambda x: f"{x:,.2f}".rstrip("0").rstrip("."))
    fmt_currency = lambda val: "-" if pd.isna(val) else f"{val:,.0f}"
    for col in ["평단가", "금액", "현재가", "수익금"]:
        formatted[col] = formatted[col].apply(fmt_currency)
    def fmt_rate(val: Optional[float]) -> str:
        if val is None or pd.isna(val):
            return "-"
        sign = "+" if val > 0 else ""
        return f"{sign}{val * 100:.2f}%"
    formatted["수익률"] = filtered["수익률"].apply(fmt_rate)
    formatted["등락률"] = filtered["등락률"].apply(fmt_rate)
    display_df = formatted[columns]

    total_rows = len(display_df) + 1
    table_h = total_rows * row_h
    fig_height = max(min_canvas_h, pad_top + table_h + pad_bottom)
    extra_pad = (fig_height - (pad_top + table_h + pad_bottom)) / 2.0 if fig_height > (pad_top + table_h + pad_bottom) else 0.0

    table_y = (pad_bottom + extra_pad) / fig_height
    table_height_ratio = table_h / fig_height

    _configure_matplotlib()
    fig, ax = plt.subplots(figsize=(canvas_w, fig_height), dpi=FIG_DPI)
    fig.patch.set_facecolor(CANVAS_BG_COLOR)
    ax.axis("off")

    table = ax.table(
        cellText=display_df.values,
        colLabels=columns,
        cellLoc="center",
        loc="center",
        bbox=[0.0, table_y, 1.0, table_height_ratio]
    )
    table.auto_set_font_size(False)
    table.set_fontsize(table_fs)
    header_color = "#2c3e50"
    even_color = "#fffdf5"
    odd_color = "#f6f0e6"
    gain_color = "#b42318"
    loss_color = "#1d4ed8"

    profit_values = filtered["수익금"].to_list()
    rate_values = filtered["수익률"].to_list()
    change_values = filtered["등락률"].to_list()

    uniform_height = 1.0 / total_rows

    for (row, col), cell in table.get_celld().items():
        cell.set_height(uniform_height)
        cell.set_y((total_rows - 1 - row) * uniform_height)
        cell.get_text().set_va("center")
        cell.get_text().set_y(0.5)
        if row == 0:
            cell.set_facecolor(header_color)
            cell.set_text_props(color="white", weight="bold")
            continue
        cell.set_facecolor(even_color if row % 2 == 0 else odd_color)
        cell.set_edgecolor("#dddddd")
        if col in (0, 1):
            cell.get_text().set_ha("left")
        else:
            cell.get_text().set_ha("right")
        if col == 6:
            profit = profit_values[row - 1]
            if pd.notna(profit):
                if profit > 0:
                    cell.get_text().set_color(gain_color)
                elif profit < 0:
                    cell.get_text().set_color(loss_color)
                else:
                    cell.get_text().set_color("#2c3e50")
        elif col in (7, 8):
            rates = rate_values if col == 7 else change_values
            value = rates[row - 1]
            if pd.notna(value):
                cell.get_text().set_color(gain_color if value > 0 else loss_color if value < 0 else "#2c3e50")

    _save_canvas(
        fig,
        output_path,
        f"보유 종목 현황 저장 완료: {output_path}",
        pad_inches=0.15,
        bbox="tight",
    )

    return True


def plot_portfolio_allocation(holdings_df: pd.DataFrame, symbol_map: Dict[str, AssetConfig], output_path: Path) -> Path:
    """전체 계좌 포트폴리오 비중(자산군, 지역, 대표자산) 도넛 차트를 생성하여 저장한다."""
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
    
    group_df = df.groupby("asset_group")["평가금"].sum()
    region_df = df.groupby("region")["평가금"].sum()
    class_df = df.groupby("asset_class")["평가금"].sum()
    
    group_df = group_df[group_df > 0].sort_values(ascending=False)
    region_df = region_df[region_df > 0].sort_values(ascending=False)
    class_df = class_df[class_df > 0].sort_values(ascending=False)

    # 2% 미만인 미세 항목들을 '기타'로 합산하는 헬퍼 함수
    def group_minor_categories(series: pd.Series, threshold_pct: float = 2.0) -> pd.Series:
        if len(series) <= 5:
            return series
        total = series.sum()
        if total == 0:
            return series
        pcts = series / total * 100
        major_mask = pcts >= threshold_pct
        major_data = series[major_mask].copy()
        minor_data = series[~major_mask].copy()
        if not minor_data.empty:
            minor_sum = minor_data.sum()
            if "기타" in major_data.index:
                major_data["기타"] += minor_sum
            else:
                major_data = pd.concat([major_data, pd.Series({"기타": minor_sum})])
        return major_data.sort_values(ascending=False)

    group_df = group_minor_categories(group_df)
    region_df = group_minor_categories(region_df)
    class_df = group_minor_categories(class_df)

    _configure_matplotlib()
    
    cfg = LAYOUT.get("portfolio_allocation", {})
    canvas_w = float(cfg.get("canvas_width", 22.5))
    canvas_h = float(cfg.get("canvas_height", 9.75))
    title_fs = float(cfg.get("title_fontsize", 20.0))
    pie_pct_fs = float(cfg.get("pie_pct_fontsize", 16.0))
    legend_fs = float(cfg.get("legend_fontsize", 15.0))

    fig, axes = plt.subplots(1, 3, figsize=(canvas_w, canvas_h), dpi=FIG_DPI)
    fig.patch.set_facecolor(CANVAS_BG_COLOR)
    
    color_palette = [
        "#4E79A7",
        "#59A14F",
        "#F28E2B",
        "#B07AA1",
        "#E15759",
        "#76B7B2",
        "#EDC948",
        "#9C755F",
    ]
    
    def draw_donut(ax, data, title):
        import matplotlib.patheffects as path_effects
        
        ax.set_facecolor(CANVAS_BG_COLOR)
        if data.empty:
            ax.text(0.5, 0.5, "데이터 없음", ha="center", va="center", fontsize=18, color="#2c3e50")
            ax.axis("off")
            return
            
        wedges, texts, autotexts = ax.pie(
            data,
            labels=None,
            autopct=lambda pct: f"{pct:.1f}%" if pct >= 5 else "",
            pctdistance=0.75,
            startangle=90,
            colors=color_palette[:len(data)] if len(data) <= len(color_palette) else plt.colormaps["tab10"](np.linspace(0, 1, len(data))),
            textprops={"fontsize": pie_pct_fs, "color": "white", "weight": "bold"},
            wedgeprops=dict(width=0.5, edgecolor="#ffffff", linewidth=2.0)
        )
        
        for at in autotexts:
            at.set_color("white")
            at.set_fontsize(pie_pct_fs)
            at.set_weight("bold")
            at.set_path_effects([path_effects.withStroke(linewidth=3, foreground="#2c3e50")])
            
        ax.set_title(title, fontsize=title_fs, fontweight="bold", color="#2c3e50", pad=20)
        ax.axis("equal")

        total_val = data.sum()
        legend_labels = [f"{label} ({val / total_val * 100:.1f}%)" for label, val in data.items()]
        
        if len(data) <= 4:
            ncol = 1
        elif len(data) <= 8:
            ncol = 2
        else:
            ncol = 3

        ax.legend(
            wedges,
            legend_labels,
            loc="upper center",
            bbox_to_anchor=(0.5, -0.05),
            ncol=ncol,
            frameon=False,
            prop={"size": legend_fs, "weight": "bold"}
        )

    draw_donut(axes[0], group_df, "자산군 비중")
    draw_donut(axes[1], region_df, "지역 비중")
    draw_donut(axes[2], class_df, "대표 자산 비중")
    
    plt.tight_layout()
    _save_canvas(fig, output_path, f"포트폴리오 비중 그래프 저장 완료: {output_path}", pad_inches=0.4, bbox="tight")
    _crop_top_inches(output_path, inches=0.4)
    
    return output_path


def _wrap_history_text(text: str, width: int = 90) -> List[str]:
    """줄바꿈 없이 긴 거래 내역 문장을 적절한 길이로 감싼다."""
    text = (text or "").strip()
    if not text:
        return [""]
    return textwrap.wrap(text, width=width, break_long_words=False, expand_tabs=False)


def plot_monthly_trading_history(records: pd.DataFrame,
                          fx_series: pd.Series,
                          month_end: pd.Timestamp,
                          output_path: Path) -> Path:
    """해당 월 거래 내역을 텍스트 형태로 정리해 저장한다."""
    period = month_end.to_period("M")
    start = period.start_time
    end = period.end_time
    month_records = records[(records["일자"] >= start) & (records["일자"] <= end)].copy()
    if month_records.empty:
        raise ValueError("해당 월 거래 내역이 없습니다.")

    cfg = LAYOUT.get("trading_history", {})
    canvas_w = float(cfg.get("canvas_width", 12.0))
    pad_top = float(cfg.get("pad_top", 0.3))
    pad_bottom = float(cfg.get("pad_bottom", 0.3))
    summary_line_h = float(cfg.get("summary_line_height", 0.38))
    summary_fs = float(cfg.get("summary_fontsize", 14.0))
    text_line_h = float(cfg.get("text_line_height", 0.32))
    text_fs = float(cfg.get("text_fontsize", 13.0))
    min_canvas_h = float(cfg.get("min_canvas_height", 3.5))

    buy_total = sell_total = invest_total = div_total = 0.0
    lines: List[Tuple[str, str]] = []

    def fmt_currency(val: float) -> str:
        return f"{val:,.0f}"

    month_records = month_records.sort_values("일자", ascending=False)
    for _, row in month_records.iterrows():
        date = pd.Timestamp(row["일자"])
        date_str = f"{date:%Y년 %m월 %d일}"
        acct_code = str(row.get("계좌", "")).strip()
        account = account_label(acct_code)
        symbol = str(row.get("종목", "")).strip()
        qty = row.get("수량")
        price = row.get("단가")
        dividend = row.get("배당")
        invest = row.get("투자금")

        has_qty_price = pd.notna(qty) and pd.notna(price) and qty != 0
        has_dividend = pd.notna(dividend) and dividend != 0
        has_invest = pd.notna(invest) and invest != 0

        if has_qty_price:
            trade_amt = convert_to_krw(acct_code, float(qty) * float(price), date, fx_series)
            unit_price = convert_to_krw(acct_code, float(price), date, fx_series)
            if qty > 0:
                buy_total += trade_amt
                lines.append((
                    "buy",
                    f"{date_str} - (매수) {account}: {symbol} {fmt_currency(trade_amt)}원 매수 (단가 {fmt_currency(unit_price)}원, {abs(qty):g}주)"
                ))
            else:
                sell_total += abs(trade_amt)
                lines.append((
                    "sell",
                    f"{date_str} - (매도) {account}: {symbol} {fmt_currency(abs(trade_amt))}원 매도 (단가 {fmt_currency(unit_price)}원, {abs(qty):g}주)"
                ))
        if has_dividend:
            div_amt = convert_to_krw(acct_code, float(dividend), date, fx_series)
            div_total += div_amt
            native_str = "" if acct_code not in USD_ACCOUNTS else f" ({dividend}달러)"
            lines.append((
                "div",
                f"{date_str} - (배당금) {account}: {symbol} 배당 {fmt_currency(div_amt)}원 수령{native_str}"
            ))
        if has_invest:
            invest_amt = float(str(invest).replace(",", "")) if invest else 0.0
            # 투자금은 csv 상에 이미 원화(KRW)로 기입되어 있으므로 환율을 곱하지 않고 그대로 사용합니다.
            invest_total += invest_amt
            lines.append((
                "invest",
                f"{date_str} - (투자금) {account}: 투자금 {fmt_currency(invest_amt)}원 증액"
            ))

    summary = f"{period.year}년 {period.month:02d}월 투자금: {fmt_currency(invest_total)}원, 매수: {fmt_currency(buy_total)}원, 매도: {fmt_currency(sell_total)}원, 배당금: {fmt_currency(div_total)}원"

    _configure_matplotlib()

    wrapped_lines: List[Tuple[str, List[str]]] = []
    for kind, text in lines:
        wrapped_lines.append((kind, _wrap_history_text(text)))

    total_wrapped = sum(len(parts) for _, parts in wrapped_lines)
    content_height = pad_top + summary_line_h + (total_wrapped * text_line_h) + pad_bottom
    fig_height = max(min_canvas_h, content_height)

    fig, ax = plt.subplots(figsize=(canvas_w, fig_height), dpi=FIG_DPI)
    fig.patch.set_facecolor(CANVAS_BG_COLOR)
    ax.axis("off")

    colors = {
        "buy": "#d63031",
        "sell": "#0984e3",
        "invest": "#2c3e50",
        "div": "#2c3e50",
    }

    curr_y = fig_height - pad_top
    summary_y = curr_y / fig_height
    ax.text(0.0, summary_y, summary, transform=ax.transAxes, ha="left", va="top", fontsize=summary_fs, fontweight="bold", color="#2c3e50")

    curr_y -= summary_line_h
    for kind, text_lines in wrapped_lines:
        for line in text_lines:
            line_y = curr_y / fig_height
            ax.text(0.0, line_y, line, transform=ax.transAxes, ha="left", va="top", fontsize=text_fs, color=colors.get(kind, "#2c3e50"))
            curr_y -= text_line_h

    _save_canvas(fig, output_path, f"월별 거래 내역 저장 완료: {output_path}", pad_inches=0.15, bbox="tight")

    return True


def generate_month_reports(prefix: str,
                           output_dir: Path,
                           records: pd.DataFrame,
                           fx_series: pd.Series,
                           month_end: pd.Timestamp,
                           monthly_prices_df: Optional[pd.DataFrame] = None,
                           fx_series_for_exchange: Optional[pd.Series] = None,) -> Dict[str, Path]:
    """특정 월의 모든 리포트(차트, 데이터)를 생성하고 파일 경로를 반환한다."""
    outputs: Dict[str, Path] = {}

    def save_title(key_name: Optional[str], graph_path: Optional[Path] = None) -> Optional[Path]:
        return _save_title(prefix, output_dir, key_name, outputs, graph_path)

    price_path = output_dir / f"{prefix}_monthly_prices.csv"
    if monthly_prices_df is not None and not monthly_prices_df.empty:
        month_key = month_end.strftime("%Y-%m")
        subset = monthly_prices_df[monthly_prices_df["월"] <= month_key]
        if not subset.empty:
            price_path.parent.mkdir(parents=True, exist_ok=True)
            subset.to_csv(price_path, index=False, encoding="utf-8")
            outputs["monthly_prices"] = price_path

    account_df = build_account_valuation_df(records, fx_series, month_end)
    valuation_path = output_dir / f"{prefix}_financialassets_trend.webp"
    plot_assets_trend(account_df, valuation_path)
    outputs["assets_trend"] = valuation_path
    save_title(CONTENT_TITLE_KEYS.get("assets_trend"), valuation_path)

    invest_series = _build_investment_series(records, fx_series)
    invest_trend_path = output_dir / f"{prefix}_assets_investment_trend.webp"
    plot_assets_investment_trend(account_df, invest_series, invest_trend_path)
    outputs["assets_investment_trend"] = invest_trend_path
    save_title(CONTENT_TITLE_KEYS.get("assets_investment_trend"), invest_trend_path)

    summary_df = build_account_assets(records, account_df, fx_series)
    summary_path = output_dir / f"{prefix}_account_assets.webp"
    display_df = format_summary_table(summary_df)
    plot_account_assets(display_df, summary_path)
    outputs["account_assets"] = summary_path
    save_title(CONTENT_TITLE_KEYS.get("account_assets"), summary_path)

    try:
        exchange_rate_path = output_dir / f"{prefix}_exchange_rate.webp"
        fx_for_exchange = fx_series_for_exchange if fx_series_for_exchange is not None else fx_series
        if fx_for_exchange is None or fx_for_exchange.empty:
            raise ValueError("환율 데이터가 없습니다.")
        reference_date = pd.to_datetime(fx_for_exchange.index.max())
        plot_exchange_rate_table(fx_for_exchange, reference_date, exchange_rate_path)
        outputs["exchange_rate"] = exchange_rate_path
        save_title(CONTENT_TITLE_KEYS.get("exchange_rate"), exchange_rate_path)
    except ValueError as exc:
        print(f"(경고) {prefix} 환율 테이블 생성 실패: {exc}")

    try:
        market_indices_path = output_dir / f"{prefix}_market_indices.webp"
        plot_market_indices_table(market_indices_path, month_end)
        outputs["market_indices"] = market_indices_path
        save_title(CONTENT_TITLE_KEYS.get("market_indices"), market_indices_path)
    except Exception as exc:
        print(f"(경고) {prefix} 주요 시장 지표 생성 실패: {exc}")

    try:
        holdings_df = build_holdings_df(records, fx_series)
        holdings_path = output_dir / f"{prefix}_total_holdings.webp"
        plot_total_holdings(holdings_df, holdings_path)
        outputs["total_holdings"] = holdings_path
        save_title(CONTENT_TITLE_KEYS.get("total_holdings"), holdings_path)

        portfolio_allocation_path = output_dir / f"{prefix}_portfolio_allocation.webp"
        symbol_map = load_symbol_map()
        plot_portfolio_allocation(holdings_df, symbol_map, portfolio_allocation_path)
        outputs["portfolio_allocation"] = portfolio_allocation_path
        save_title(CONTENT_TITLE_KEYS.get("portfolio_allocation"), portfolio_allocation_path)

        trading_history_path = output_dir / f"{prefix}_trading_history.webp"
        plot_monthly_trading_history(records, fx_series, month_end, trading_history_path)
        outputs["trading_history"] = trading_history_path
        save_title(CONTENT_TITLE_KEYS.get("trading_history"), trading_history_path)

        valid_detail_accounts: List[str] = []
        if not summary_df.empty and "평가금" in summary_df.columns:
            valid_detail_accounts = summary_df.loc[summary_df["평가금"] > 0, "계좌"].astype(str).tolist()
        valid_detail_set = set(valid_detail_accounts)

        for account in DETAIL_ACCOUNTS:
            if account not in valid_detail_set:
                continue
            detail_path = output_dir / f"{prefix}_{account}_detail.webp"
            if plot_account_detail(account, holdings_df, summary_df, detail_path, account_df=account_df):
                outputs[f"{account}_detail"] = detail_path
                detail_title_key = f"title_{account}_detail"
                if detail_title_key in ACCOUNT_TITLES:
                    save_title(detail_title_key, detail_path)
    except ValueError as exc:
        print(f"(경고) {prefix} 보유 종목 그래프 생성 실패: {exc}")

    try:
        pivot = load_dividend_pivot(records, fx_series, month_end)
        dividends_path = output_dir / f"{prefix}_monthly_dividends.webp"
        plot_monthly_dividends(pivot, dividends_path)
        outputs["monthly_dividends"] = dividends_path
        save_title(CONTENT_TITLE_KEYS.get("monthly_dividends"), dividends_path)
    except ValueError as exc:
        print(f"(경고) {prefix} 배당 그래프 생성 실패: {exc}")

    try:
        yearly_pivot = load_yearly_dividend_pivot(records, fx_series, month_end)
        yearly_dividends_path = output_dir / f"{prefix}_yearly_dividends.webp"
        plot_yearly_dividends(yearly_pivot, yearly_dividends_path)
        outputs["yearly_dividends"] = yearly_dividends_path
        save_title(CONTENT_TITLE_KEYS.get("yearly_dividends"), yearly_dividends_path)
    except ValueError as exc:
        print(f"(경고) {prefix} 연별 배당 그래프 생성 실패: {exc}")

    return outputs


def write_build_info(latest_period: Optional[pd.Period]) -> None:
    """빌드 시간과 최신 월 정보를 JSON 파일로 저장한다."""
    timestamp = pd.Timestamp.now(tz="Asia/Seoul")
    data = {"built_at": timestamp.isoformat()}
    if latest_period is not None:
        data["latest_month"] = latest_period.strftime("%Y-%m")
    BUILD_INFO_PATH.parent.mkdir(parents=True, exist_ok=True)
    with BUILD_INFO_PATH.open("w", encoding="utf-8") as fp:
        json.dump(data, fp, ensure_ascii=False, indent=2)


def build_account_assets(records: pd.DataFrame, account_df: pd.DataFrame, fx_series: pd.Series) -> pd.DataFrame:
    """계좌별 요약 정보(투자금, 평가금, 수익률 등)를 계산하여 데이터프레임으로 반환한다."""
    investment_series = (
        records[records["투자금"].notna()]
        .groupby("계좌")["투자금"]
        .sum()
    )

    dividends_df = records[(records["배당"].notna()) & (records["배당"] > 0)].copy()
    if not dividends_df.empty:
        dividends_df["배당"] = dividends_df.apply(
            lambda row: convert_to_krw(row["계좌"], row["배당"], row["일자"], fx_series),
            axis=1,
        )
        dividends = dividends_df.groupby("계좌")["배당"].sum()
    else:
        dividends = pd.Series(dtype=float)

    latest_values = account_df.iloc[-1].fillna(0.0)
    total_valuation = latest_values.sum()
    rows = []
    for account, valuation in latest_values.items():
        if valuation == 0:
            continue
        invested = float(investment_series.get(account, 0.0))
        profit = valuation - invested
        profit_rate = profit / invested if invested else None
        weight = valuation / total_valuation if total_valuation else 0.0
        dividend = dividends.get(account, 0.0)
        rows.append(
            {
                "계좌": account,
                "투자금": invested,
                "평가금": valuation,
                "수익금": profit,
                "수익률": profit_rate,
                "비중": weight,
                "배당금": dividend,
            }
        )

    summary_df = pd.DataFrame(rows)
    ordered_accounts = [acct for acct in ACCOUNT_ORDER if acct in summary_df["계좌"].values]
    ordered_accounts += [acct for acct in summary_df["계좌"] if acct not in ordered_accounts]
    summary_df["계좌"] = pd.Categorical(summary_df["계좌"], categories=ordered_accounts, ordered=True)
    summary_df = summary_df.sort_values("계좌").reset_index(drop=True)
    total_row = {
        "계좌": "합계",
        "투자금": summary_df["투자금"].sum(),
        "평가금": summary_df["평가금"].sum(),
        "수익금": summary_df["수익금"].sum(),
        "수익률": None if summary_df["투자금"].sum() == 0 else summary_df["수익금"].sum() / summary_df["투자금"].sum(),
        "비중": 1.0,
        "배당금": summary_df["배당금"].sum(),
    }
    summary_df = pd.concat([summary_df, pd.DataFrame([total_row])], ignore_index=True)
    return summary_df


def format_summary_table(summary_df: pd.DataFrame) -> pd.DataFrame:
    """요약 테이블의 숫자들을 포맷팅하여 표시용 데이터프레임을 생성한다."""
    formatted = summary_df.copy()
    formatted["계좌"] = formatted["계좌"].apply(lambda acc: account_label(acc) if acc != "합계" else acc)
    for col in ["투자금", "평가금", "수익금", "배당금"]:
        formatted[col] = formatted[col].apply(lambda x: "-" if pd.isna(x) else f"{x:,.0f}")
    formatted["수익률"] = formatted["수익률"].apply(lambda x: "-" if (x is None or pd.isna(x)) else f"{x * 100:.2f}%")
    formatted["비중"] = formatted["비중"].apply(lambda x: "-" if pd.isna(x) else f"{x * 100:.2f}%")
    return formatted[["계좌", "투자금", "평가금", "수익금", "수익률", "비중", "배당금"]]


def plot_account_assets(display_df: pd.DataFrame, output_path: Path) -> Path:
    """전체 계좌 요약 정보를 테이블 형태의 이미지로 저장한다."""
    cfg = LAYOUT.get("account_assets", {})
    canvas_w = float(cfg.get("canvas_width", 12.0))
    pad_top = float(cfg.get("pad_top", 0.25))
    pad_bottom = float(cfg.get("pad_bottom", 0.25))
    row_h = float(cfg.get("row_height", 0.38))
    table_fs = float(cfg.get("table_fontsize", 12.0))
    min_canvas_h = float(cfg.get("min_canvas_height", 4.0))

    num_rows = len(display_df)
    total_rows = num_rows + 1
    table_h = total_rows * row_h
    fig_height = max(min_canvas_h, pad_top + table_h + pad_bottom)
    extra_pad = (fig_height - (pad_top + table_h + pad_bottom)) / 2.0 if fig_height > (pad_top + table_h + pad_bottom) else 0.0

    table_y = (pad_bottom + extra_pad) / fig_height
    table_height_ratio = table_h / fig_height

    _configure_matplotlib()
    fig, ax = plt.subplots(figsize=(canvas_w, fig_height), dpi=FIG_DPI)
    fig.patch.set_facecolor(CANVAS_BG_COLOR)
    ax.axis("off")

    accounts = display_df["계좌"].tolist()
    colored_accounts = [acct for acct in accounts if acct != "합계"]
    cmap = plt.colormaps["tab10"]
    account_colors = cmap(np.linspace(0, 1, max(len(colored_accounts), 1)))
    color_map = {acct: account_colors[idx] for idx, acct in enumerate(colored_accounts)}

    table = ax.table(
        cellText=display_df.values,
        colLabels=display_df.columns,
        cellLoc="center",
        loc="center",
        colColours=["#f6f6f6"] * len(display_df.columns),
        bbox=[0.0, table_y, 1.0, table_height_ratio],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(table_fs)

    uniform_height = 1.0 / total_rows

    for (row, col), cell in table.get_celld().items():
        cell.set_height(uniform_height)
        cell.set_y((total_rows - 1 - row) * uniform_height)
        cell.get_text().set_va("center")
        cell.get_text().set_y(0.5)
        cell.set_edgecolor("#dddddd")
        if row == 0:
            cell.set_facecolor("#2d3436")
            cell.set_text_props(weight="bold", color="white")
            continue

        data_idx = row - 1
        if data_idx < 0 or data_idx >= len(display_df):
            continue
        account = accounts[data_idx]
        is_total = account == "합계"

        if col == 0:
            cell.get_text().set_ha("left")
            if is_total:
                cell.set_facecolor("#4b5563")
                cell.set_text_props(weight="bold", color="white")
            else:
                color = color_map.get(account, "#cccccc")
                cell.set_facecolor(color)
                cell.set_text_props(weight="bold", color="white")
        else:
            cell.get_text().set_ha("right")
            col_name = display_df.columns[col]
            if is_total:
                cell.set_facecolor("#e5e7eb")
                cell.get_text().set_weight("bold")
            else:
                shade = "#fffdf5" if (row % 2 == 1) else "#f6f0e6"
                cell.set_facecolor(shade)

            if col_name in ("수익금", "수익률"):
                val_str = str(display_df.iloc[data_idx, col]).strip()
                cleaned = re.sub(r"[^\d.-]", "", val_str)
                try:
                    num_val = float(cleaned)
                    if num_val > 0:
                        cell.get_text().set_color("#b42318")
                    elif num_val < 0:
                        cell.get_text().set_color("#1d4ed8")
                    else:
                        cell.get_text().set_color("#1f2933")
                except ValueError:
                    cell.get_text().set_color("#1f2933")
            else:
                cell.get_text().set_color("#1f2933")

    plt.tight_layout()
    _save_canvas(fig, output_path, f"계좌 요약 표 저장 완료: {output_path}", pad_inches=0.15, bbox="tight")

    return True


def copy_to_latest(src: Path, latest_name: str) -> None:
    """생성된 최신 리포트 파일을 'latest_...' 이름으로 복사한다."""
    dst = STATIC_FINANCIALASSETS_DIR / latest_name
    shutil.copy2(src, dst)


def parse_args() -> argparse.Namespace:
    """커맨드라인 인자를 파싱한다. --full 옵션을 통해 전체 기간 리포트 생성 여부를 결정한다."""
    parser = argparse.ArgumentParser(description="Generate financial assets reports")
    parser.add_argument(
        "--full",
        action="store_true",
        help="Generate reports for every month since 2022-02 (default: only latest month)",
    )
    return parser.parse_args()


def update_titles_from_fa_yaml() -> None:
    """fa.yaml의 계좌 이름을 읽어 ACCOUNT_TITLES를 동적으로 업데이트한다."""
    if not FINANCIALASSETS_YAML_PATH.exists():
        return

    try:
        with FINANCIALASSETS_YAML_PATH.open("r", encoding="utf-8") as f:
            portfolio = yaml.safe_load(f) or {}

        map_name_to_key = {
            "미국": "usa",
            "국내1": "kor1",
            "국내2": "kor2",
            "공제회": "sema",
            "IRP": "irp",
            "연금저축1": "psf1",
            "ISA1": "isa1",
            "연금저축2": "psf2",
            "ISA2": "isa2",
        }

        for account in portfolio.get("accounts", []):
            name = account.get("name")
            if not name:
                continue
            # 괄호 앞부분의 깨끗한 이름을 추출하여 매핑 키 탐색
            # 예: "연금저축1 (SCHD:QQQ:MMA = 6:3:1)" -> "연금저축1"
            clean_name = re.sub(r"\s*\(.*\)", "", name).strip()
            key = map_name_to_key.get(clean_name)
            if key:
                ACCOUNT_RAW_NAMES[key] = name
                ACCOUNT_RAW_NAMES[clean_name] = name
                title_key = f"title_{key}_detail"
                # "공제회"의 경우 기본 영문 표기 "SEMA"를 "공제회" 대신 사용
                display_name = name
                if clean_name == "공제회":
                    display_name = name.replace("공제회", "SEMA")
                ACCOUNT_TITLES[title_key] = f"◉ 상세계좌: {display_name}"
    except Exception as e:
        print(f"fa.yaml에서 타이틀을 업데이트하는 중 오류가 발생했습니다: {e}")


def main() -> None:
    """메인 실행 함수. 거래 내역을 읽어 월별 리포트를 생성한다."""
    args = parse_args()
    update_titles_from_fa_yaml()
    ensure_static_dir()
    records = read_trading_records()
    if records.empty:
        print("거래 내역이 없습니다.")
        return

    latest_date = records["일자"].dropna().max()
    if pd.isna(latest_date):
        print("유효한 거래 날짜가 없습니다.")
        return

    fx_series_full = build_fx_series(records, latest_date)
    months = pd.period_range(start=START_MONTH, end=latest_date, freq="M")
    if months.empty:
        months = pd.period_range(start=latest_date, end=latest_date, freq="M")
    if not args.full:
        months = months[-1:]
    latest_period = months[-1] if len(months) else None

    monthly_prices_full = get_monthly_prices(latest_date, records)

    latest_outputs: Dict[str, Path] = {}
    for idx, period in enumerate(months):
        month_end = period.to_timestamp(how="end")
        records_upto = records[records["일자"] <= month_end].copy()
        if records_upto.empty:
            continue
        fx_series = fx_series_full.loc[:month_end]
        prefix = period.strftime("%y%m")
        year_dir = STATIC_FINANCIALASSETS_DIR / f"{period.year}"
        year_dir.mkdir(parents=True, exist_ok=True)

        outputs = generate_month_reports(
            prefix,
            year_dir,
            records_upto,
            fx_series,
            month_end,
            monthly_prices_full,
            fx_series_for_exchange=fx_series_full if idx == len(months) - 1 else fx_series,
        )
        if idx == len(months) - 1:
            latest_outputs = outputs

    write_build_info(latest_period)

    latest_map = {
        "monthly_prices": "latest_monthly_prices.csv",
        "assets_trend": "latest_assets_trend.webp",
        "assets_investment_trend": "latest_assets_investment_trend.webp",
        "portfolio_allocation": "latest_portfolio_allocation.webp",
        "account_assets": "latest_account_assets.webp",
        "exchange_rate": "latest_exchange_rate.webp",
        "market_indices": "latest_market_indices.webp",
        "monthly_dividends": "latest_monthly_dividends.webp",
        "yearly_dividends": "latest_yearly_dividends.webp",
        "total_holdings": "latest_total_holdings.webp",
        "trading_history": "latest_trading_history.webp",
    }
    for account in DETAIL_ACCOUNTS:
        latest_map[f"{account}_detail"] = f"latest_{account}_detail.webp"
    for key, latest_name in latest_map.items():
        path = latest_outputs.get(key)
        if path and path.exists():
            copy_to_latest(path, latest_name)


if __name__ == "__main__":
    main()
