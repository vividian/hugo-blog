#!/usr/bin/env python3
"""
공제회 펀드 기준가 수집 및 평가액 자동 계산 모듈 (Fund Fetcher)
- 네이버 증권 모바일 페이지 및 프론트엔드 API 기반 실시간 기준가 수집
- 캐시 관리 (data/fund_prices.json)
- 좌수 기반 평가액 산출: 좌수 x 기준가 / 1000 (소수점 이하 버림)
- SQLite DB 및 CSV 백업 동기화 유틸리티
"""

import json
import math
import os
import re
import sqlite3
import time
import urllib.request
from pathlib import Path
from typing import Any, Dict, Optional

ROOT_DIR = Path(__file__).resolve().parent.parent
CACHE_FILE = ROOT_DIR / "data" / "fund_prices.json"
DEFAULT_FUND_CODE = "K55210DT4614"  # 신한미국S&P500인덱스증권자투자신탁(UH)[주식-파생형](종류C-re)
DEFAULT_FUND_SYMBOL = "신한 SPY-FD"

# 공제회(sema)에서 취급하는 펀드 정의 (SAVING/저축은 펀드가 아니므로 제외)
SEMA_FUNDS: Dict[str, Dict[str, Any]] = {
    "K55210DT4614": {
        "code": "K55210DT4614",
        "symbol": "신한 SPY-FD",
        "name": "신한미국S&P500인덱스증권자투자신탁(UH)[주식-파생형](종류C-re)",
        "keywords": ["신한", "SPY-FD", "S&P500"],
    },
    "K55301E51782": {
        "code": "K55301E51782",
        "symbol": "미래에셋 골드",
        "name": "미래에셋골드특별자산자투자신탁(금-재간접형)종류C-P2e",
        "keywords": ["골드", "미래에셋", "GOLD"],
    },
}


def get_fund_code_by_symbol(symbol: str) -> Optional[str]:
    """
    종목명 또는 심볼을 기반으로 해당 공제회 펀드 코드를 판별합니다.
    SAVING / 저축 등은 펀드가 아니므로 None을 반환합니다.
    """
    if not symbol:
        return None
    sym_clean = symbol.strip()
    # 저축 / SAVING은 펀드 계산 대상에서 명시적 제외
    if any(k in sym_clean for k in ["SAVING", "저축", "현금", "예금"]):
        return None
    for code, info in SEMA_FUNDS.items():
        if code == sym_clean:
            return code
        if info["symbol"] == sym_clean or info["name"] == sym_clean:
            return code
        if any(kw in sym_clean for kw in info["keywords"]):
            return code
    return None


def fetch_fund_base_price(fund_code: str = DEFAULT_FUND_CODE) -> Optional[Dict[str, Any]]:
    """
    네이버 증권 모바일(m.stock.naver.com)에서 펀드의 최신 기준가 정보를 수집합니다.
    URL: https://m.stock.naver.com/domestic/fund/{fund_code}/allocation
    """
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
            "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        ),
        "Referer": f"https://m.stock.naver.com/domestic/fund/{fund_code}/allocation",
    }

    # 1. Front-API 시도
    api_url = f"https://m.stock.naver.com/front-api/fund/detail?fundCode={fund_code}"
    try:
        req = urllib.request.Request(api_url, headers=headers)
        with urllib.request.urlopen(req, timeout=8) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            if data.get("isSuccess") and "result" in data:
                res = data["result"]
                base_price = float(res.get("basePrice", 0.0))
                if base_price > 0:
                    return {
                        "fund_code": fund_code,
                        "fund_name": res.get("fundName", ""),
                        "base_price": base_price,
                        "trade_date": res.get("tradeDate", ""),
                        "change_price": float(res.get("changePrice", 0.0)),
                        "change_rate": float(res.get("returnIndex", 0.0)),
                        "nav": float(res.get("nav", 0.0)),
                        "fetched_at": time.time(),
                    }
    except Exception as e:
        print(f"⚠️ [fund_fetcher] Front-API 수집 실패 ({fund_code}): {e}")

    # 2. HTML __NEXT_DATA__ Fallback 시도
    html_url = f"https://m.stock.naver.com/domestic/fund/{fund_code}/allocation"
    try:
        req = urllib.request.Request(html_url, headers=headers)
        with urllib.request.urlopen(req, timeout=8) as resp:
            html = resp.read().decode("utf-8")
        m = re.search(r'<script id="__NEXT_DATA__" type="application/json">(.*?)</script>', html)
        if m:
            data = json.loads(m.group(1))
            queries = data.get("props", {}).get("pageProps", {}).get("dehydratedState", {}).get("queries", [])
            for q in queries:
                res = q.get("state", {}).get("data", {}).get("result")
                if isinstance(res, dict) and res.get("fundCode") == fund_code:
                    base_price = float(res.get("basePrice", 0.0))
                    if base_price > 0:
                        return {
                            "fund_code": fund_code,
                            "fund_name": res.get("fundName", ""),
                            "base_price": base_price,
                            "trade_date": res.get("tradeDate", ""),
                            "change_price": float(res.get("changePrice", 0.0)),
                            "change_rate": float(res.get("returnIndex", 0.0)),
                            "nav": float(res.get("nav", 0.0)),
                            "fetched_at": time.time(),
                        }
    except Exception as e:
        print(f"⚠️ [fund_fetcher] HTML Fallback 수집 실패 ({fund_code}): {e}")

    return None


def get_cached_or_fetch_fund_price(fund_code: str = DEFAULT_FUND_CODE, max_age_seconds: int = 300) -> Optional[Dict[str, Any]]:
    """
    캐시 파일(data/fund_prices.json)을 확인하고, 유효기간(기본 5분)이 지났으면 새로 수집합니다.
    네트워크 장애 시 캐시된 최근 값을 반환합니다.
    """
    cache_data = {}
    if CACHE_FILE.exists():
        try:
            cache_data = json.loads(CACHE_FILE.read_text(encoding="utf-8"))
        except Exception:
            cache_data = {}

    entry = cache_data.get(fund_code)
    now = time.time()
    if entry and (now - entry.get("fetched_at", 0) < max_age_seconds):
        return entry

    # 새로 수집
    latest = fetch_fund_base_price(fund_code)
    if latest:
        cache_data[fund_code] = latest
        try:
            CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
            CACHE_FILE.write_text(json.dumps(cache_data, ensure_ascii=False, indent=2), encoding="utf-8")
        except Exception as e:
            print(f"⚠️ [fund_fetcher] 캐시 저장 실패: {e}")
        return latest

    # 수집 실패 시 기존 캐시 반환
    return entry


def get_all_sema_funds_prices(max_age_seconds: int = 300) -> Dict[str, Dict[str, Any]]:
    """
    공제회 등록 모든 펀드(신한, 미래에셋골드)의 최신 기준가 정보를 딕셔너리로 반환합니다.
    """
    results = {}
    for code, info in SEMA_FUNDS.items():
        res = get_cached_or_fetch_fund_price(code, max_age_seconds=max_age_seconds)
        if res:
            res["symbol"] = info["symbol"]
            results[code] = res
    return results


def calculate_fund_valuation(quantity: float, base_price: float) -> int:
    """
    좌수 x 기준가 / 1000 계산 후 소수점 이하를 버린 정수 평가액을 반환합니다.
    공식: Math.floor(좌수 * 기준가 / 1000)
    """
    if quantity <= 0 or base_price <= 0:
        return 0
    return math.floor(quantity * base_price / 1000.0)


def sync_sema_fund_evaluations(
    db_path: Optional[Path] = None,
    fund_codes: Optional[list] = None,
) -> Dict[str, Any]:
    """
    SQLite DB에서 공제회(account='sema')의 펀드 거래를 조회하고,
    가장 최근 거래의 좌수가 등록되어 있으면 최신 기준가로 평가금(evaluation)과 단가(unit_price)를 자동 동기화합니다.
    (SAVING/저축 등 비펀드 종목은 대상에서 제외되며 기존 입력값 유지)
    반환값: 갱신 결과 딕셔너리
    """
    if db_path is None:
        db_path = ROOT_DIR / "db" / "fa_records.db"
    if not db_path.exists():
        return {"updated": False, "reason": "DB 파일이 존재하지 않습니다."}

    target_codes = fund_codes or list(SEMA_FUNDS.keys())
    results = {}
    total_updated = 0

    try:
        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row
        cur = conn.cursor()

        for code in target_codes:
            fund_def = SEMA_FUNDS.get(code)
            if not fund_def:
                continue

            fund_info = get_cached_or_fetch_fund_price(code)
            if not fund_info or fund_info.get("base_price", 0) <= 0:
                print(f"⚠️ [fund_fetcher] {fund_def['symbol']} ({code}) 기준가 조회 실패")
                results[code] = {"updated": False, "reason": "기준가 조회 실패"}
                continue

            base_price = fund_info["base_price"]
            trade_date = fund_info.get("trade_date", "")

            # SQL 패턴 생성
            where_clauses = ["symbol = ?"]
            params = [fund_def["symbol"]]
            for kw in fund_def["keywords"]:
                where_clauses.append("symbol LIKE ?")
                params.append(f"%{kw}%")

            query = f"""
                SELECT id, date, account, symbol, unit_price, quantity, deposit, evaluation
                FROM trading_records
                WHERE account = 'sema' AND ({' OR '.join(where_clauses)})
                ORDER BY date DESC, id DESC
                LIMIT 1;
            """
            cur.execute(query, params)
            row = cur.fetchone()

            if not row:
                results[code] = {"updated": False, "reason": "거래 내역 없음"}
                continue

            rec_id = row["id"]
            qty = float(row["quantity"] or 0)
            old_eval = float(row["evaluation"] or 0)
            old_unit_price = float(row["unit_price"] or 0)

            if qty <= 0:
                results[code] = {
                    "id": rec_id,
                    "symbol": row["symbol"],
                    "quantity": 0,
                    "base_price": base_price,
                    "trade_date": trade_date,
                    "updated": False,
                    "reason": "좌수(quantity)가 0입니다. fa-admin에서 좌수를 먼저 입력해주세요.",
                }
                continue

            # 새 평가금 계산: Math.floor(좌수 * 기준가 / 1000)
            new_eval = calculate_fund_valuation(qty, base_price)
            diff = new_eval - old_eval

            cur.execute("""
                UPDATE trading_records
                SET unit_price = ?, evaluation = ?
                WHERE id = ?;
            """, (base_price, new_eval, rec_id))

            print(f"✨ [fund_fetcher] 공제회 {row['symbol']} 평가액 자동 동기화 완료! "
                  f"(좌수: {qty:,.0f}좌, 기준가: {base_price:,.2f}원 -> 평가액: {new_eval:,.0f}원, 변동: {diff:+,.0f}원)")

            results[code] = {
                "id": rec_id,
                "symbol": row["symbol"],
                "quantity": qty,
                "base_price": base_price,
                "trade_date": trade_date,
                "old_eval": old_eval,
                "new_eval": new_eval,
                "diff": diff,
                "updated": True,
            }
            total_updated += 1

        conn.commit()
        conn.close()
        return {"updated": (total_updated > 0), "total_updated": total_updated, "results": results}
    except Exception as e:
        print(f"⚠️ [fund_fetcher] DB 평가금 동기화 실패: {e}")
        return {"updated": False, "error": str(e)}


if __name__ == "__main__":
    print("Testing fund_fetcher with SEMA funds...")
    prices = get_all_sema_funds_prices()
    for code, info in prices.items():
        print(f"[{info['symbol']}] ({code}) 기준가: {info['base_price']:,.2f}원 ({info['trade_date']})")
        sample_qty = 10000000
        val = calculate_fund_valuation(sample_qty, info["base_price"])
        print(f"   예시: {sample_qty:,.0f}좌 -> {val:,.0f}원")

