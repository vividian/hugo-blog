#!/usr/bin/env python3
"""
기존 config/trading_records.csv의 모든 거래 기록을 data/fa_records.db (SQLite)로 마이그레이션합니다.
"""

from pathlib import Path
import sqlite3
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parent.parent
CSV_PATH = ROOT_DIR / "config" / "trading_records.csv"
DB_PATH = ROOT_DIR / "db" / "fa_records.db"


def init_db(conn: sqlite3.Connection):
    cursor = conn.cursor()
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS trading_records (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        date TEXT NOT NULL,
        account TEXT NOT NULL,
        symbol TEXT NOT NULL,
        kind TEXT DEFAULT '',
        unit_price REAL DEFAULT 0.0,
        quantity REAL DEFAULT 0.0,
        amount REAL DEFAULT 0.0,
        dividend REAL DEFAULT 0.0,
        deposit REAL DEFAULT 0.0,
        evaluation REAL DEFAULT 0.0,
        exchange_rate REAL DEFAULT 1.0,
        memo TEXT DEFAULT '',
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );
    """)
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_records_date ON trading_records(date);")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_records_account ON trading_records(account);")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_records_symbol ON trading_records(symbol);")
    conn.commit()


def migrate():
    if not CSV_PATH.exists():
        print(f"(오류) CSV 파일을 찾을 수 없습니다: {CSV_PATH}")
        return

    df = pd.read_csv(CSV_PATH, encoding="utf-8-sig")
    print(f"총 {len(df)}건의 CSV 거래 데이터를 읽었습니다.")

    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    init_db(conn)

    # 기존 데이터 확인 및 초기화 후 적재
    cursor = conn.cursor()
    cursor.execute("DROP TABLE IF EXISTS trading_records;")
    init_db(conn)

    for _, row in df.iterrows():
        date_val = str(row.get("일자", "")).strip()
        account_val = str(row.get("계좌", "")).strip()
        symbol_val = str(row.get("종목", "")).strip() if pd.notna(row.get("종목")) else ""
        kind_val = str(row.get("구분", "")).strip() if pd.notna(row.get("구분")) else ""

        unit_price = pd.to_numeric(row.get("단가"), errors="coerce") or 0.0
        quantity = pd.to_numeric(row.get("수량"), errors="coerce") or 0.0
        amount = pd.to_numeric(row.get("금액"), errors="coerce") or 0.0
        dividend = pd.to_numeric(row.get("배당"), errors="coerce") or 0.0
        deposit = pd.to_numeric(row.get("투자금"), errors="coerce") or 0.0
        evaluation = pd.to_numeric(row.get("평가금"), errors="coerce") or 0.0
        exchange_rate = pd.to_numeric(row.get("환율"), errors="coerce") or 1.0
        memo_val = str(row.get("비고", "")).strip() if pd.notna(row.get("비고")) else ""

        cursor.execute("""
        INSERT INTO trading_records (
            date, account, symbol, kind, unit_price, quantity, amount, dividend, deposit, evaluation, exchange_rate, memo
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
        """, (
            date_val, account_val, symbol_val, kind_val,
            float(unit_price), float(quantity), float(amount),
            float(dividend), float(deposit), float(evaluation), float(exchange_rate),
            memo_val
        ))

    conn.commit()

    # 구분이 비어있는 과거 데이터를 배당/입금/출금/평가금/매도/매수로 자동 보정
    cursor.execute("UPDATE trading_records SET kind = '배당' WHERE (kind IS NULL OR kind = '') AND dividend > 0;")
    cursor.execute("UPDATE trading_records SET kind = '입금' WHERE (kind IS NULL OR kind = '') AND deposit > 0;")
    cursor.execute("UPDATE trading_records SET kind = '출금' WHERE (kind IS NULL OR kind = '') AND deposit < 0;")
    cursor.execute("UPDATE trading_records SET kind = '평가금' WHERE (kind IS NULL OR kind = '') AND evaluation > 0;")
    cursor.execute("UPDATE trading_records SET kind = '매도' WHERE (kind IS NULL OR kind = '') AND quantity < 0;")
    cursor.execute("UPDATE trading_records SET kind = '매수' WHERE (kind IS NULL OR kind = '') AND (quantity > 0 OR unit_price > 0);")
    cursor.execute("UPDATE trading_records SET date = REPLACE(date, '-', '.') WHERE date LIKE '%-%';")
    conn.commit()

    cursor.execute("SELECT COUNT(*) FROM trading_records;")
    count = cursor.fetchone()[0]
    conn.close()

    print(f"✅ SQLite DB 마이그레이션 완료: {DB_PATH} (총 {count}건 적재)")


if __name__ == "__main__":
    migrate()
