# 필요한 패키지: pip install pandas yfinance matplotlib pillow
#
# 이 스크립트는 Obsidian vault의 금융 자산 데이터를 처리하고 시각화하기 위한 것입니다.
# 거래 기록(trading_records.csv)을 읽어들여, yfinance를 통해 최신 주가 및 환율 정보를 가져옵니다.
# 이를 바탕으로 계좌별 자산 현황, 월별 배당금, 보유 종목 상세 내역 등 다양한 보고서를 생성합니다.
# 생성된 차트와 데이터는 Hugo 블로그의 정적 파일로 출력됩니다.
from __future__ import annotations

import argparse
import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

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
ACCOUNT_ORDER = ["usa", "kor1", "sema", "irp", "psf1", "isa1", "psf2", "isa2"]  # 계좌 표시 순서
DETAIL_ACCOUNTS = ["usa", "kor1", "sema", "irp", "psf1", "isa1", "psf2", "isa2"]  # 상세 내역을 표시할 계좌
ACCOUNT_LABELS = {
    "usa": "미국 주식",
    "kor1": "국내 주식1",
    "sema": "공제회",
    "irp": "IRP",
    "psf1": "연금저축1",
    "isa1": "ISA1",
    "psf2": "연금저축2",
    "isa2": "ISA2",
}
TITLE_FONT_SIZE = 16  # 차트 제목 폰트 크기
FIG_DPI = 150  # 차트 DPI
CANVAS_BG_COLOR = "#fffdf5"  # 차트 배경색
TITLE_COLOR = "#2c3e50"  # 차트 제목 색상
TITLE_POS = (0.0, 0.5)  # 제목 축 내부 좌표 (좌측, 중앙)
TITLE_ROW_HEIGHT = 0.3  # 제목 전용 행 높이(인치)
FIG_LEFT = 0.05
FIG_RIGHT = 0.98
FIG_TOP = 0.98
FIG_BOTTOM = 0.06
ACCOUNT_TITLES = {
    "title_assets_trend": "◉ 계좌별 자산 추세",
    "title_account_assets": "◉ 전체 계좌별 자산 현황 (투자금, 평가금, 수익금 등)",
    "title_total_holdings": "◉ 실시간 보유종목 현황",
    "title_trading_history": "◉ 보유종목 거래내역",
    "title_monthly_dividends": "◉ 월별 배당금 및 분배금 현황 (최근 12개월)",
    "title_usa_detail": "◉ 상세계좌: 미국주식 (SPYM:IEF:SGOV = 7:2:1)",
    "title_kor1_detail": "◉ 상세계좌: 국내주식1 (리츠)",
    "title_sema_detail": "◉ 상세계좌: SEMA (S&P500-FD:SAVING = 7:3)",
    "title_irp_detail": "◉ 상세계좌: IRP (S&P500:KOFR = 7:3)",
    "title_psf1_detail": "◉ 상세계좌: 연금저축1 (S&P500:IEF:MMA = 7:2:1)",
    "title_isa1_detail": "◉ 상세계좌: ISA1",
    "title_psf2_detail": "◉ 상세계좌: 연금저축2 (SCHD:QQQ:IEF:MMA = 4:3:2:1)",
    "title_isa2_detail": "◉ 상세계좌: ISA2",
}
CONTENT_TITLE_KEYS = {
    "assets_trend": "title_assets_trend",
    "account_assets": "title_account_assets",
    "total_holdings": "title_total_holdings",
    "trading_history": "title_trading_history",
    "monthly_dividends": "title_monthly_dividends",
}

@dataclass
class AssetConfig:
    """자산 설정 정보를 담는 데이터 클래스"""
    name: str
    abbrev: str
    ticker: str


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
    """trading_records.csv 파일을 읽어 DataFrame으로 반환한다."""
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
            if not ticker:
                continue
            y_ticker = ticker
            if ticker.startswith("KRX:"):
                y_ticker = ticker.replace("KRX:", "") + ".KS"
            config = AssetConfig(name=name, abbrev=abbrev, ticker=y_ticker)
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
    if records.empty:
        start = pd.Timestamp.today() - pd.Timedelta(days=60)
    else:
        earliest = records["일자"].dropna().min()
        if pd.isna(earliest):
            start = pd.Timestamp.today() - pd.Timedelta(days=60)
        else:
            start = earliest - pd.Timedelta(days=7)

    if end is None:
        end = pd.Timestamp.today() + pd.Timedelta(days=2)
    else:
        end = end + pd.Timedelta(days=2)

    data = yf.download(
        FX_TICKER,
        start=start,
        end=end,
        progress=False,
        auto_adjust=False,
    )
    if data.empty:
        raise RuntimeError("환율 데이터를 불러오지 못했습니다.")

    series = data["Adj Close"] if "Adj Close" in data else data
    if isinstance(series, pd.DataFrame):
        series = series.iloc[:, 0]
    series = pd.Series(series).ffill()
    series.index = pd.to_datetime(series.index).tz_localize(None)
    return series


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


def get_monthly_prices(end_date: pd.Timestamp) -> Optional[pd.DataFrame]:
    """fa.yaml에 정의된 모든 종목의 월말 종가 데이터를 가져온다."""
    fa_data = load_symbol_map()
    tickers_map: Dict[str, str] = {}
    for config in fa_data.values():
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
                outputs: Dict[str, Path]) -> Optional[Path]:
    """주어진 제목 키에 해당하는 제목 이미지를 저장하고 outputs에 등록한다."""
    if not title_key:
        return None
    filename = f"{prefix}_{title_key}.webp"
    title_path = output_dir / filename
    plot_title_image(title_key, title_path)
    outputs[title_key] = title_path
    return title_path


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


def plot_assets_trend(account_df: pd.DataFrame, output_path: Path) -> Path:
    """계좌별 자산 흐름을 꺾은선 그래프로 그려 저장한다."""
    _configure_matplotlib()
    fig, ax = plt.subplots(figsize=(12, 6), dpi=FIG_DPI)
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
            marker="o",
            markersize=3,
        )

    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:,.0f}"))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%y%m"))
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1), frameon=False)
    ax.grid(True, axis="y", linestyle="--", alpha=0.7)
    for spine in ax.spines.values():
        spine.set_color("#dddddd")

    y_min = np.nanmin(account_df.values)
    y_max = np.nanmax(account_df.values)
    pad = (y_max - y_min) * 0.1 if y_max > y_min else max(abs(y_max), 1.0) * 0.1
    ax.set_ylim(y_min - pad, y_max + pad)
    fig.autofmt_xdate(rotation=30)

    # fig.subplots_adjust(top=0.85, bottom=0.12, left=0.1, right=0.9)
    _save_canvas(
        fig,
        output_path,
        f"계좌 추세 그래프 저장 완료: {output_path}",
        pad_inches=0.65,
        # bbox=None,
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
    data = yf.download(
        tickers,
        period="5d",
        interval="1d",
        progress=False,
        auto_adjust=False,
    )
    if data.empty:
        return {}
    adj_close = data["Adj Close"] if "Adj Close" in data.columns else data
    if isinstance(adj_close, pd.Series):
        series = adj_close.ffill().dropna()
        if series.empty:
            return {}
        last_price = float(series.iloc[-1])
        prev_price = float(series.iloc[-2]) if len(series) > 1 else last_price
        return {tickers[0]: (last_price, prev_price)}
    filled = adj_close.ffill()
    if filled.empty:
        return {}
    last_row = filled.iloc[-1]
    prev_row = filled.iloc[-2] if len(filled) > 1 else last_row
    return {
        col: (float(last_row[col]), float(prev_row[col]))
        for col in last_row.index
    }


def build_holdings_df(records: pd.DataFrame, fx_series: pd.Series) -> pd.DataFrame:
    """현재 보유 자산 현황(평가금, 매수금, 수익금) 데이터프레임을 생성한다."""
    symbol_map = load_symbol_map()
    trades = extract_trades(records)
    positions = compute_positions(trades, symbol_map, fx_series)

    prices = fetch_latest_prices(sorted({p.ticker for p in positions}))
    rows: List[Dict[str, float]] = []
    for pos in positions:
        price_info = prices.get(pos.ticker)
        if price_info is None:
            continue
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


def plot_account_detail(account: str, 
                        holdings_df: pd.DataFrame, 
                        status_df: pd.DataFrame, 
                        output_path: Path,) -> bool:
    """개별 계좌의 상세 내역(파이 차트, 보유 종목 테이블)을 그려 저장한다."""
    # 계좌 종목
    account_holdings = holdings_df[holdings_df["계좌"] == account].copy()
    if account_holdings.empty:
        return False

    # 계좌 현황
    status_row = status_df[status_df["계좌"] == account]
    if status_row.empty:
        return False
    status_row = status_row.iloc[0]

    account_holdings = account_holdings.sort_values("평가금", ascending=False)
    colors = plt.colormaps["tab10"](np.linspace(0, 1, max(len(account_holdings), 1)))

    _configure_matplotlib()
    fig, axes = plt.subplots(
        1,
        2,
        figsize=(12, 5),
        dpi=FIG_DPI,
        gridspec_kw={"width_ratios": [0.7, 1.3]},
    )
    fig.patch.set_facecolor(CANVAS_BG_COLOR)
    ax_pie, ax_table = axes
    ax_pie.set_facecolor(CANVAS_BG_COLOR)

    # 파이 차트
    wedges, _, autotexts = ax_pie.pie(
        account_holdings["평가금"],
        labels=None,
        autopct=lambda pct: f"{pct:.1f}%"
        if pct >= 3
        else "",
        startangle=90,
        colors=colors,
        textprops={"fontsize": 14, "color": "white", "weight": "bold"},
    )
    ax_pie.set_title("")
    color_map = dict(zip(account_holdings["종목"], colors))

    # 테이블 그리기
    table_df = account_holdings.copy()

    # 종목별 데이터 테이블
    main_data = table_df[["종목", "매수금", "평가금", "수익금", "수익률"]].copy()
    main_data["매수금"] = main_data["매수금"].apply(lambda x: f"{x:,.0f}")
    main_data["평가금"] = main_data["평가금"].apply(lambda x: f"{x:,.0f}")
    main_data["수익금"] = main_data["수익금"].apply(lambda x: f"{x:,.0f}")
    main_data["수익률"] = main_data["수익률"].apply(lambda x: "-" if x is None or pd.isna(x) else f"{x * 100:.2f}%")

    main_data_rows = len(main_data)
    custom_height = {
        # 테이블 타이틀의 높이 위치, 테이블 하단 위치, 테이블 높이
        2: [0.82, 0.40, 0.36], 
        3: [0.88, 0.40, 0.42], 
        4: [0.88, 0.34, 0.48],  # 기준 값
        5: [0.88, 0.34, 0.53],
        6: [0.95, 0.32, 0.58]
    }

    ax_table.axis("off")
    ax_table.text(
        0.0,
        custom_height.get(main_data_rows, [0.88, 0.55, 0.22])[0],
        "계좌 현황 (종목별)",
        transform=ax_table.transAxes,
        ha="left",
        va="top",
        fontsize=14,
        fontweight="bold",
        color="#2c3e50",
    )

    header_colors = ["#2d3436", "#2d3436", "#2d3436", "#2d3436", "#2d3436"]
    neutral_header = "#2d3436"

    main_table = ax_table.table(
        cellText=main_data.values,
        colLabels=main_data.columns,
        cellLoc="center",
        loc="upper center",
        bbox=[0, custom_height.get(main_data_rows, [0.88, 0.55, 0.22])[1], 1, custom_height.get(main_data_rows, [0.88, 0.55, 0.22])[2]],
    )
    main_table.auto_set_font_size(False)
    main_table.set_fontsize(14)
    main_table.scale(1.05, 0.55)

    for (row, col), cell in main_table.get_celld().items():
        cell.set_edgecolor("#dddddd")
        if row == 0:
            cell.get_text().set_ha("center")
        else:
            cell.get_text().set_ha("center" if col == 0 else "right")

        if row == 0:
            color = header_colors[col] if col < len(header_colors) else neutral_header
            cell.set_facecolor(color)
            cell.set_text_props(color="white", weight="bold")
        else:
            shade = "#fffdf5" if (row % 2 == 0) else "#f6f0e6"
            if col == 0:
                label = main_data.iloc[row - 1, 0]
                shade = color_map.get(label, shade)
                cell.set_text_props(color="white", weight="bold")
            else:
                cell.set_text_props(color="#2c3e50", weight="normal")
            cell.set_facecolor(shade)

    # 계좌 현황 요약 테이블
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

    ax_table.text(
        0.0,
        0.29,
        "계좌 현황 (전체)",
        transform=ax_table.transAxes,
        ha="left",
        va="top",
        fontsize=14,
        fontweight="bold",
        color="#2c3e50",
    )

    summary_table = ax_table.table(
        cellText=summary_data.values,
        colLabels=summary_columns,
        cellLoc="center",
        loc="upper center",
        bbox=[0, 0.05, 1, 0.18],
    )
    summary_table.auto_set_font_size(False)
    summary_table.set_fontsize(14)
    summary_table.scale(1.05, 0.55)

    for (row, col), cell in summary_table.get_celld().items():
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

    plt.tight_layout()
    display_name = account_label(account)
    _save_canvas(fig, output_path, f"{display_name} 상세 그래프 저장 완료: {output_path}")
    _crop_top_inches(output_path, inches=0.7)
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


def plot_monthly_dividends(pivot: pd.DataFrame, output_path: Path) -> Path:
    """월별 배당금 현황을 누적 막대 그래프로 그려 저장한다."""
    months = pivot.index.to_pydatetime()
    columns = pivot.columns.tolist()

    _configure_matplotlib()
    fig, ax = plt.subplots(figsize=(12, 6), dpi=FIG_DPI)
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
    for i, total in enumerate(totals):
        if total <= 0:
            continue
        ax.text(
            months[i],
            total + (total * 0.02),
            f"{total:,.0f}",
            ha="center",
            va="bottom",
            fontsize=12,
            color="black",
        )

    ax.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{int(x):,}"))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%y%m"))
    ax.tick_params(axis="x", rotation=45, labelsize=12)
    ax.tick_params(axis="y", labelsize=12)
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    for spine in ax.spines.values():
        spine.set_color("#dddddd")
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1), frameon=False)
    # fig.subplots_adjust(top=0.85, bottom=0.12, left=0.1, right=0.9)
    _save_canvas(
        fig,
        output_path,
        f"월별 배당 그래프 저장 완료: {output_path}",
        pad_inches=0.65,
        bbox="tight",
    )
    _crop_top_inches(output_path, inches=0.6)

    return True


def plot_total_holdings(holdings_df: pd.DataFrame, output_path: Path) -> Path:
    """전체 보유 종목 현황을 표 형태로 그려 저장한다."""
    filtered = holdings_df[holdings_df["계좌"] != "sema"].copy()
    if filtered.empty:
        raise ValueError("보유 중인 종목이 없습니다.")

    row_count = len(filtered)
    _configure_matplotlib()
    fig, ax = plt.subplots(figsize=(12, 12), dpi=FIG_DPI)
    fig.patch.set_facecolor(CANVAS_BG_COLOR)
    ax.axis("off")

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

    table = ax.table(
        cellText=display_df.values,
        colLabels=columns,
        cellLoc="center",
        loc="center",
        bbox=[0.0, 0.05, 1.0, 0.97]
    )
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1.0, 1.15)

    header_color = "#2c3e50"
    even_color = "#fffdf5"
    odd_color = "#f6f0e6"
    gain_color = "#d63031"
    loss_color = "#0984e3"

    profit_values = filtered["수익금"].to_list()
    rate_values = filtered["수익률"].to_list()
    change_values = filtered["등락률"].to_list()

    for (row, col), cell in table.get_celld().items():
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
                    cell.get_text().set_text(f"+{display_df.iloc[row-1, col]}")
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
        pad_inches=0.55,
        bbox="tight",
    )
    _crop_top_inches(output_path, inches=0.5)

    return True


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

    buy_total = sell_total = invest_total = div_total = 0.0
    lines: List[Tuple[str, str]] = []

    def fmt_currency(val: float) -> str:
        return f"{val:,.0f}"

    month_records = month_records.sort_values("일자", ascending=False)
    for _, row in month_records.iterrows():
        date = pd.Timestamp(row["일자"])
        date_str = f"{date:%Y년 %m월 %d일}"
        account = account_label(str(row.get("계좌", "")).strip())
        symbol = str(row.get("종목", "")).strip()
        qty = row.get("수량")
        price = row.get("단가")
        dividend = row.get("배당")
        invest = row.get("투자금")

        has_qty_price = pd.notna(qty) and pd.notna(price) and qty != 0
        has_dividend = pd.notna(dividend) and dividend != 0
        has_invest = pd.notna(invest) and invest != 0

        if has_qty_price:
            trade_amt = convert_to_krw(row["계좌"], float(qty) * float(price), date, fx_series)
            unit_price = convert_to_krw(row["계좌"], float(price), date, fx_series)
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
        elif has_dividend:
            div_amt = convert_to_krw(row["계좌"], float(dividend), date, fx_series)
            div_total += div_amt
            native_str = "" if row["계좌"] not in USD_ACCOUNTS else f" ({dividend}달러)"
            lines.append((
                "div",
                f"{date_str} - (배당금) {account}: {symbol} 배당 {fmt_currency(div_amt)}원 수령{native_str}"
            ))
        elif has_invest:
            invest_amt = float(str(invest).replace(",", "")) if invest else 0.0
            if account in USD_ACCOUNTS:
                invest_amt = convert_to_krw(row["계좌"], invest_amt, date, fx_series)
            invest_total += invest_amt
            lines.append((
                "invest",
                f"{date_str} - (투자금) {account}: 투자금 {fmt_currency(invest_amt)}원 증액"
            ))

    summary = f"{period.year}년 {period.month:02d}월 투자금: {fmt_currency(invest_total)}원, 매수: {fmt_currency(buy_total)}원, 매도: {fmt_currency(sell_total)}원, 배당금: {fmt_currency(div_total)}원"

    _configure_matplotlib()
    line_count = len(lines) + 1
    fig_height = max(4.0, 1.0 + 0.35 * line_count)
    fig, ax = plt.subplots(figsize=(12, fig_height), dpi=FIG_DPI)
    fig.patch.set_facecolor(CANVAS_BG_COLOR)
    ax.axis("off")

    colors = {
        "buy": "#d63031",
        "sell": "#0984e3",
        "invest": "#2c3e50",
        "div": "#2c3e50",
    }

    y = 0.95
    ax.text(0.0, y, summary, ha="left", va="top", fontsize=14, fontweight="bold", transform=ax.transAxes, color="#2c3e50")
    y -= 0.07
    for kind, text in lines:
        ax.text(0.0, y, text, ha="left", va="top", fontsize=13, transform=ax.transAxes, color=colors.get(kind, "#2c3e50"))
        y -= 0.055

    ax.set_ylim(0, 1)
    ax.set_xlim(0, 1)
    
    _save_canvas(fig, output_path, f"월별 거래 내역 저장 완료: {output_path}", pad_inches=0.65, bbox="tight")
    _crop_top_inches(output_path, inches=0.9)

    return True


def generate_month_reports(prefix: str,
                           output_dir: Path,
                           records: pd.DataFrame,
                           fx_series: pd.Series,
                           month_end: pd.Timestamp,
                           monthly_prices_df: Optional[pd.DataFrame] = None,) -> Dict[str, Path]:
    """특정 월의 모든 리포트(차트, 데이터)를 생성하고 파일 경로를 반환한다."""
    outputs: Dict[str, Path] = {}

    def save_title(key_name: Optional[str]) -> Optional[Path]:
        return _save_title(prefix, output_dir, key_name, outputs)

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
    save_title(CONTENT_TITLE_KEYS.get("assets_trend"))

    summary_df = build_account_assets(records, account_df, fx_series)
    summary_path = output_dir / f"{prefix}_account_assets.webp"
    display_df = format_summary_table(summary_df)
    plot_account_assets(display_df, summary_path)
    outputs["account_assets"] = summary_path
    save_title(CONTENT_TITLE_KEYS.get("account_assets"))

    try:
        holdings_df = build_holdings_df(records, fx_series)
        holdings_path = output_dir / f"{prefix}_total_holdings.webp"
        plot_total_holdings(holdings_df, holdings_path)
        outputs["total_holdings"] = holdings_path
        save_title(CONTENT_TITLE_KEYS.get("total_holdings"))

        trading_history_path = output_dir / f"{prefix}_trading_history.webp"
        plot_monthly_trading_history(records, fx_series, month_end, trading_history_path)
        outputs["trading_history"] = trading_history_path
        save_title(CONTENT_TITLE_KEYS.get("trading_history"))

        for account in DETAIL_ACCOUNTS:
            detail_path = output_dir / f"{prefix}_{account}_detail.webp"
            if plot_account_detail(account, holdings_df, summary_df, detail_path):
                outputs[f"{account}_detail"] = detail_path
                detail_title_key = f"title_{account}_detail"
                if detail_title_key in ACCOUNT_TITLES:
                    save_title(detail_title_key)
    except ValueError as exc:
        print(f"(경고) {prefix} 보유 종목 그래프 생성 실패: {exc}")

    try:
        pivot = load_dividend_pivot(records, fx_series, month_end)
        dividends_path = output_dir / f"{prefix}_monthly_dividends.webp"
        plot_monthly_dividends(pivot, dividends_path)
        outputs["monthly_dividends"] = dividends_path
        save_title(CONTENT_TITLE_KEYS.get("monthly_dividends"))
    except ValueError as exc:
        print(f"(경고) {prefix} 배당 그래프 생성 실패: {exc}")

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
    row_height_factor = 1.5
    fig_height = max(5.0, (0.8 + 0.35 * len(display_df)) * row_height_factor)
    _configure_matplotlib()
    fig, ax = plt.subplots(figsize=(12, fig_height), dpi=FIG_DPI)
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
        bbox=[0.0, 0.05, 1.0, 0.85],  # [x, y, width, height]
    )
    table.auto_set_font_size(False)
    table.set_fontsize(16)
    table.scale(1, 1.3 * row_height_factor)

    num_rows = len(display_df)
    for (row, col), cell in table.get_celld().items():
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
            if is_total:
                cell.set_facecolor("#e5e7eb")
                cell.set_text_props(weight="bold", color="#111827")
            else:
                shade = "#fffdf5" if (row % 2 == 1) else "#f6f0e6"
                cell.set_facecolor(shade)
                cell.set_text_props(color="#1f2933")

    plt.tight_layout()
    _save_canvas(fig, output_path, f"계좌 요약 표 저장 완료: {output_path}")
    _crop_top_inches(output_path, inches=0.9)

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


def main() -> None:
    """메인 실행 함수. 거래 내역을 읽어 월별 리포트를 생성한다."""
    args = parse_args()
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

    monthly_prices_full = get_monthly_prices(latest_date)

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
        )
        if idx == len(months) - 1:
            latest_outputs = outputs

    write_build_info(latest_period)

    latest_map = {
        "monthly_prices": "latest_monthly_prices.csv",
        "assets_trend": "latest_assets_trend.webp",
        "account_assets": "latest_account_assets.webp",
        "monthly_dividends": "latest_monthly_dividends.webp",
        "total_holdings": "latest_total_holdings.webp",
        "trading_history": "latest_trading_history.webp",
    }
    for account in DETAIL_ACCOUNTS:
        latest_map[f"{account}_detail"] = f"latest_{account}_detail.webp"
    for title_key in CONTENT_TITLE_KEYS.values():
        latest_map[title_key] = f"latest_{title_key}.webp"
    for account in DETAIL_ACCOUNTS:
        detail_title_key = f"title_{account}_detail"
        if detail_title_key in ACCOUNT_TITLES:
            latest_map[detail_title_key] = f"latest_{detail_title_key}.webp"

    for key, latest_name in latest_map.items():
        path = latest_outputs.get(key)
        if path and path.exists():
            copy_to_latest(path, latest_name)


if __name__ == "__main__":
    main()
