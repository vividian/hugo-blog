#!/usr/bin/env python3
"""
FA 거래내역 관리자 웹 서버 (FA Admin Server)
- SQLite DB (data/fa_records.db) 기반 거래내역 CRUD (등록/수정/삭제/조회)
- 자세히보기 대시보드와 일치하는 모던 핀테크 반응형 UI
- 거래 변경 시 config/trading_records.csv 자동 백업 동기화
- 원클릭 대시보드 생성(update_fa_plotly.py) 트리거 지원
"""

import json
import sqlite3
import subprocess
import sys
import threading
from datetime import date
from http import HTTPStatus
from http.server import HTTPServer, SimpleHTTPRequestHandler
from pathlib import Path
from typing import Any, Dict, List
from urllib.parse import parse_qs, urlparse
import yaml

ROOT_DIR = Path(__file__).resolve().parent.parent
DB_PATH = ROOT_DIR / "db" / "fa_records.db"
CSV_PATH = ROOT_DIR / "config" / "trading_records.csv"
FA_YAML_PATH = ROOT_DIR / "config" / "fa.yaml"

ACCOUNT_CHOICES = [
    {"code": "usa", "name": "미국 주식"},
    {"code": "kor1", "name": "국내 주식1"},
    {"code": "kor2", "name": "국내 주식2"},
    {"code": "sema", "name": "공제회 (SEMA)"},
    {"code": "irp", "name": "IRP"},
    {"code": "psf1", "name": "연금저축1"},
    {"code": "isa1", "name": "ISA1"},
    {"code": "psf2", "name": "연금저축2"},
    {"code": "isa2", "name": "ISA2"},
]


def add_symbol_to_fa_yaml(account_code: str, full_name: str, abbrev: str, ticker: str, region: str, asset_class: str) -> bool:
    """config/fa.yaml 파일의 해당 계좌 항목에 신규 종목 메타데이터를 추가합니다."""
    try:
        if not FA_YAML_PATH.exists():
            return False
        with open(FA_YAML_PATH, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}

        account_name_map = {
            "usa": "미국", "kor1": "국내1", "kor2": "국내2",
            "sema": "공제회", "irp": "IRP", "psf1": "연금저축1",
            "isa1": "ISA1", "psf2": "연금저축2", "isa2": "ISA2"
        }
        target_keyword = account_name_map.get(account_code, account_code)

        for acct_entry in cfg.get("accounts", []):
            acct_name = acct_entry.get("name", "")
            if target_keyword in acct_name:
                items = acct_entry.setdefault("items", [])
                # 이미 존재하는 종목인지 확인
                exists = any(it[0] == full_name or it[1] == abbrev for it in items if len(it) >= 2)
                if not exists:
                    items.append([full_name, abbrev, ticker or abbrev, region or "국내상장미국", asset_class or "주식"])
                    with open(FA_YAML_PATH, "w", encoding="utf-8") as f:
                        yaml.dump(cfg, f, allow_unicode=True, sort_keys=False)
                    print(f"✅ [fa.yaml] 신규 종목 등록 성공: [{acct_name}] {full_name} ({abbrev})")
                return True
        return False
    except Exception as e:
        print(f"⚠️ [fa.yaml] 신규 종목 등록 실패: {e}")
        return False


def get_db_connection() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def export_db_to_csv():
    """DB의 거래 데이터를 config/trading_records.csv로 자동 백업 덤프합니다."""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("""
            SELECT account AS 계좌, date AS 일자, symbol AS 종목,
                   unit_price AS 단가, quantity AS 수량, dividend AS 배당,
                   deposit AS 투자금, evaluation AS 평가금, memo AS 비고
            FROM trading_records
            ORDER BY date DESC, id DESC;
        """)
        rows = cursor.fetchall()
        conn.close()

        import pandas as pd
        df = pd.DataFrame([dict(r) for r in rows])
        if not df.empty:
            # 날짜를 YYYY.MM.DD 형식으로 보존
            df["일자"] = df["일자"].str.replace("-", ".", regex=False)
            CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(CSV_PATH, index=False, encoding="utf-8-sig")
            print(f"(백업) {CSV_PATH} 덤프 완료 ({len(df)}건)")
    except Exception as e:
        print(f"(경고) CSV 자동 백업 덤프 실패: {e}")


def run_dashboard_update():
    """백그라운드에서 update_fa.py, update_fa_plotly.py 및 nas_scheduler.py를 실행하여 웹 서비스에 즉시 100% 반영합니다."""
    def _worker():
        try:
            print("⏳ [대시보드 갱신] 자산 계산 및 대시보드 생성 시작...")
            # 1. update_fa.py (데이터 계산)
            cmd_calc = [sys.executable, str(ROOT_DIR / "scripts" / "update_fa.py")]
            subprocess.run(cmd_calc, cwd=ROOT_DIR, capture_output=True, text=True)

            # 2. update_fa_plotly.py (HTML 대시보드 생성)
            cmd_plot = [sys.executable, str(ROOT_DIR / "scripts" / "update_fa_plotly.py")]
            res = subprocess.run(cmd_plot, cwd=ROOT_DIR, capture_output=True, text=True)

            if res.returncode == 0:
                print("✅ [대시보드 갱신] HTML 생성 완료!")
                # 3. NAS 환경일 경우 nas_scheduler.py --skip-permissions 실행하여 웹 서비스 경로로 자동 동기화
                if Path("/var/services/web").exists() or Path("/volume1").exists() or Path("/volume3").exists():
                    cmd_sync = [sys.executable, str(ROOT_DIR / "scripts" / "nas_scheduler.py"), "--skip-permissions"]
                    res_sync = subprocess.run(cmd_sync, cwd=ROOT_DIR, capture_output=True, text=True)
                    if res_sync.returncode == 0:
                        print("🚀 [실시간 배포] NAS 웹 서버 동기화 성공!")
                    else:
                        print(f"⚠️ [실시간 배포] nas_scheduler 경고: {res_sync.stderr}")
            else:
                print(f"⚠️ 대시보드 갱신 에러: {res.stderr}")
        except Exception as e:
            print(f"⚠️ 대시보드 갱신 실행 실패: {e}")

    t = threading.Thread(target=_worker, daemon=True)
    t.start()


HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="ko">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0, maximum-scale=1.0, user-scalable=no">
  <title>FA 자산 거래내역 관리자</title>
  <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/flatpickr/dist/flatpickr.min.css">
  <script src="https://cdn.jsdelivr.net/npm/flatpickr"></script>
  <script src="https://cdn.jsdelivr.net/npm/flatpickr/dist/l10n/ko.js"></script>
  <style>
    :root {
      --fa-bg: #f8fafc;
      --fa-card-bg: #ffffff;
      --fa-card-border: #e2e8f0;
      --fa-card-shadow: 0 4px 20px -2px rgba(0, 0, 0, 0.05), 0 2px 6px -1px rgba(0, 0, 0, 0.03);
      --fa-text-main: #1e293b;
      --fa-text-muted: #64748b;
      --fa-text-sub: #94a3b8;
      --fa-kpi-bg: #f8fafc;
      --fa-border: #e2e8f0;
      
      --fa-gain: #e53e3e;
      --fa-gain-bg: #fff5f5;
      --fa-loss: #3182ce;
      --fa-loss-bg: #ebf8ff;
      --fa-accent: #4f46e5;
      --fa-accent-hover: #4338ca;
      --fa-accent-bg: #eef2ff;
      --fa-purple: #805ad5;
      --fa-purple-bg: #faf5ff;
      --fa-ok: #38a169;
      --fa-ok-bg: #f0fff4;
    }

    * { box-sizing: border-box; margin: 0; padding: 0; }
    body {
      background-color: var(--fa-bg);
      color: var(--fa-text-main);
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Noto Sans KR", sans-serif;
      line-height: 1.5;
      padding: 16px 12px 60px;
    }
    .fa-admin-container {
      max-width: 1050px;
      margin: 0 auto;
    }

    /* Header */
    .fa-header {
      display: flex;
      justify-content: space-between;
      align-items: center;
      margin-bottom: 20px;
      padding: 8px 4px;
      flex-wrap: wrap;
      gap: 12px;
    }
    .fa-header-title {
      font-size: 1.35rem;
      font-weight: 800;
      letter-spacing: -0.02em;
      display: flex;
      align-items: center;
      gap: 8px;
    }
    .fa-header-badge {
      font-size: 0.72rem;
      font-weight: 700;
      background: var(--fa-accent-bg);
      color: var(--fa-accent);
      padding: 3px 8px;
      border-radius: 6px;
    }
    .fa-btn-refresh {
      background: #ffffff;
      border: 1px solid var(--fa-border);
      color: var(--fa-text-main);
      padding: 8px 14px;
      border-radius: 8px;
      font-size: 0.85rem;
      font-weight: 600;
      cursor: pointer;
      display: flex;
      align-items: center;
      gap: 6px;
      transition: all 0.2s;
      box-shadow: 0 1px 2px rgba(0,0,0,0.04);
    }
    .fa-btn-refresh:hover {
      background: var(--fa-accent-bg);
      color: var(--fa-accent);
      border-color: var(--fa-accent);
    }

    /* KPI Summary Row */
    .fa-kpi-row {
      display: grid;
      grid-template-columns: repeat(4, 1fr);
      gap: 12px;
      margin-bottom: 20px;
    }
    @media (max-width: 640px) {
      .fa-kpi-row { grid-template-columns: repeat(2, 1fr); gap: 8px; }
    }
    .fa-kpi-box {
      background: var(--fa-card-bg);
      border: 1px solid var(--fa-card-border);
      border-radius: 12px;
      padding: 14px 16px;
      box-shadow: var(--fa-card-shadow);
    }
    .fa-kpi-label { font-size: 0.75rem; font-weight: 600; color: var(--fa-text-muted); margin-bottom: 4px; }
    .fa-kpi-val { font-size: 1.15rem; font-weight: 800; font-variant-numeric: tabular-nums; }

    /* Card Section */
    .fa-card {
      background: var(--fa-card-bg);
      border: 1px solid var(--fa-card-border);
      border-radius: 14px;
      box-shadow: var(--fa-card-shadow);
      padding: 20px 22px;
      margin-bottom: 24px;
    }
    .fa-card-title {
      font-size: 1.05rem;
      font-weight: 700;
      margin-bottom: 16px;
      display: flex;
      align-items: center;
      justify-content: space-between;
    }

    /* Form Styles */
    .fa-form-grid {
      display: grid;
      grid-template-columns: repeat(3, 1fr);
      gap: 14px;
    }
    @media (max-width: 768px) {
      .fa-form-grid { grid-template-columns: 1fr 1fr; }
    }
    @media (max-width: 500px) {
      .fa-form-grid { grid-template-columns: 1fr; }
    }
    .fa-field {
      display: flex;
      flex-direction: column;
      gap: 6px;
    }
    .fa-label {
      font-size: 0.8rem;
      font-weight: 700;
      color: var(--fa-text-muted);
    }
    .fa-input, .fa-select {
      background: var(--fa-bg);
      border: 1px solid var(--fa-border);
      border-radius: 8px;
      padding: 10px 12px;
      font-size: 0.92rem;
      color: var(--fa-text-main);
      font-family: inherit;
      outline: none;
      transition: border-color 0.2s, box-shadow 0.2s;
    }
    .fa-input:focus, .fa-select:focus {
      border-color: var(--fa-accent);
      box-shadow: 0 0 0 3px var(--fa-accent-bg);
      background: #ffffff;
    }

    /* Kind Segment Buttons */
    .fa-kind-group {
      display: flex;
      gap: 6px;
      flex-wrap: wrap;
    }
    .fa-kind-btn {
      flex: 1;
      min-width: 55px;
      padding: 8px 6px;
      background: var(--fa-bg);
      border: 1px solid var(--fa-border);
      border-radius: 8px;
      font-size: 0.82rem;
      font-weight: 700;
      color: var(--fa-text-muted);
      cursor: pointer;
      text-align: center;
      transition: all 0.15s;
    }
    .fa-kind-btn.active.buy { background: var(--fa-gain-bg); color: var(--fa-gain); border-color: var(--fa-gain); }
    .fa-kind-btn.active.sell { background: var(--fa-loss-bg); color: var(--fa-loss); border-color: var(--fa-loss); }
    .fa-kind-btn.active.div { background: var(--fa-purple-bg); color: var(--fa-purple); border-color: var(--fa-purple); }
    .fa-kind-btn.active.deposit { background: var(--fa-ok-bg); color: var(--fa-ok); border-color: var(--fa-ok); }
    .fa-kind-btn.active.eval { background: var(--fa-accent-bg); color: var(--fa-accent); border-color: var(--fa-accent); }

    .fa-form-actions {
      display: flex;
      justify-content: flex-end;
      gap: 10px;
      margin-top: 18px;
    }
    .fa-btn-primary {
      background: var(--fa-accent);
      color: #ffffff;
      border: none;
      padding: 10px 24px;
      border-radius: 8px;
      font-size: 0.92rem;
      font-weight: 700;
      cursor: pointer;
      transition: background 0.2s, transform 0.1s;
    }
    .fa-btn-primary:hover { background: var(--fa-accent-hover); }
    .fa-btn-primary:active { transform: scale(0.98); }

    /* Filter & Table Toolbar */
    .fa-table-toolbar {
      display: flex;
      justify-content: space-between;
      align-items: center;
      gap: 10px;
      margin-bottom: 14px;
      flex-wrap: wrap;
    }
    .fa-filter-group {
      display: flex;
      gap: 8px;
      flex-wrap: wrap;
    }

    /* Modern Table */
    .fa-table-wrap {
      overflow-x: auto;
      border: 1px solid var(--fa-border);
      border-radius: 10px;
    }
    .fa-table {
      width: 100%;
      border-collapse: collapse;
      font-size: 0.86rem;
      text-align: left;
      white-space: nowrap;
    }
    .fa-table th {
      background: var(--fa-table-header-bg);
      color: var(--fa-text-muted);
      font-weight: 700;
      padding: 10px 12px;
      border-bottom: 1px solid var(--fa-border);
    }
    .fa-table td {
      padding: 10px 12px;
      border-bottom: 1px solid var(--fa-border);
      font-variant-numeric: tabular-nums;
    }
    .fa-table tr:last-child td { border-bottom: none; }
    .fa-table tr:hover { background: rgba(0,0,0,0.015); }
    .text-right { text-align: right; }

    /* Badges */
    .badge {
      display: inline-block;
      padding: 2px 7px;
      border-radius: 5px;
      font-size: 0.75rem;
      font-weight: 700;
    }
    .badge-buy { background: var(--fa-gain-bg); color: var(--fa-gain); }
    .badge-sell { background: var(--fa-loss-bg); color: var(--fa-loss); }
    .badge-div { background: var(--fa-purple-bg); color: var(--fa-purple); }
    .badge-deposit { background: var(--fa-ok-bg); color: var(--fa-ok); }
    .badge-eval { background: var(--fa-accent-bg); color: var(--fa-accent); }
    .badge-account { background: #f1f5f9; color: #475569; }

    /* Action Buttons */
    .fa-btn-action {
      background: transparent;
      border: 1px solid var(--fa-border);
      border-radius: 6px;
      padding: 4px 8px;
      font-size: 0.76rem;
      font-weight: 600;
      cursor: pointer;
      transition: all 0.15s;
    }
    .fa-btn-edit:hover { background: var(--fa-accent-bg); color: var(--fa-accent); border-color: var(--fa-accent); }
    .fa-btn-del:hover { background: var(--fa-gain-bg); color: var(--fa-gain); border-color: var(--fa-gain); }

    /* Notification Toast */
    #toast {
      position: fixed;
      bottom: 24px;
      right: 24px;
      background: #1e293b;
      color: #ffffff;
      padding: 12px 20px;
      border-radius: 10px;
      font-size: 0.88rem;
      font-weight: 600;
      box-shadow: 0 10px 25px rgba(0,0,0,0.2);
      transform: translateY(100px);
      opacity: 0;
      transition: all 0.25s ease-out;
      z-index: 3000;
    }
    #toast.show { transform: translateY(0); opacity: 1; }

    /* Modal Styles */
    .fa-modal-overlay {
      position: fixed;
      top: 0; left: 0; right: 0; bottom: 0;
      background: rgba(15, 23, 42, 0.6);
      backdrop-filter: blur(4px);
      display: none;
      align-items: center;
      justify-content: center;
      z-index: 2000;
      padding: 16px;
    }
    .fa-modal-overlay.open { display: flex; animation: faFadeIn 0.2s ease-out; }
    .fa-modal-box {
      background: var(--fa-card-bg);
      border: 1px solid var(--fa-card-border);
      border-radius: 12px;
      width: 100%;
      max-width: 480px;
      box-shadow: 0 20px 25px -5px rgba(0, 0, 0, 0.1), 0 10px 10px -5px rgba(0, 0, 0, 0.04);
      overflow: hidden;
    }
    .fa-modal-header {
      padding: 16px 20px;
      border-bottom: 1px solid var(--fa-border);
      display: flex;
      justify-content: space-between;
      align-items: center;
      font-weight: 700;
      font-size: 1rem;
    }
    .fa-modal-body {
      padding: 20px;
      display: flex;
      flex-direction: column;
      gap: 12px;
    }
    .fa-modal-footer {
      padding: 14px 20px;
      border-top: 1px solid var(--fa-border);
      display: flex;
      justify-content: flex-end;
      gap: 8px;
      background: var(--fa-kpi-bg);
    }
  </style>
</head>
<body>
  <div class="fa-admin-container">
    <!-- Header -->
    <div class="fa-header">
      <div class="fa-header-title">
        <span>📈 FA 자산 거래내역 관리자</span>
        <span class="fa-header-badge">v2.7.20</span>
        <span class="fa-header-badge" style="background:var(--fa-border); color:var(--fa-text-muted);">SQLite DB</span>
      </div>
      <div style="display:flex; gap:8px;">
        <button class="fa-btn-refresh" onclick="triggerDashboardBuild()">
          <span>📊 대시보드 갱신</span>
        </button>
      </div>
    </div>

    <!-- KPI Row -->
    <div class="fa-kpi-row">
      <div class="fa-kpi-box">
        <div class="fa-kpi-label">총 거래 건수</div>
        <div class="fa-kpi-val" id="kpi-total-count">-</div>
      </div>
      <div class="fa-kpi-box">
        <div class="fa-kpi-label">최근 거래일</div>
        <div class="fa-kpi-val" id="kpi-latest-date">-</div>
      </div>
      <div class="fa-kpi-box">
        <div class="fa-kpi-label">당월 총 매수금</div>
        <div class="fa-kpi-val" style="color:var(--fa-gain);" id="kpi-month-buy">-</div>
      </div>
      <div class="fa-kpi-box">
        <div class="fa-kpi-label">당월 총 배당금</div>
        <div class="fa-kpi-val" style="color:var(--fa-purple);" id="kpi-month-div">-</div>
      </div>
    </div>

    <!-- Trade Input Form Card -->
    <div class="fa-card">
      <div class="fa-card-title">
        <span id="form-heading">✨ 새 거래 등록</span>
        <button type="button" id="btn-cancel-edit" class="fa-btn-action" style="display:none;" onclick="cancelEdit()">수정 취소</button>
      </div>
      <form id="trade-form" onsubmit="handleFormSubmit(event)">
        <input type="hidden" id="edit-id" value="">
        <div class="fa-form-grid">
          <!-- 일자 -->
          <div class="fa-field">
            <label class="fa-label">거래 일자</label>
            <input type="text" id="f-date" class="fa-input" required placeholder="YYYY-MM-DD">
          </div>

          <!-- 계좌 -->
          <div class="fa-field">
            <label class="fa-label">계좌 선택</label>
            <select id="f-account" class="fa-select" required onchange="handleAccountChange()">
              <option value="usa">미국 주식</option>
              <option value="kor1">국내 주식1</option>
              <option value="kor2">국내 주식2</option>
              <option value="sema">공제회 (SEMA)</option>
              <option value="irp">IRP</option>
              <option value="psf1">연금저축1</option>
              <option value="isa1">ISA1</option>
              <option value="psf2">연금저축2</option>
              <option value="isa2">ISA2</option>
            </select>
          </div>

          <!-- 거래 구분 (Kind) -->
          <div class="fa-field">
            <label class="fa-label">거래 구분</label>
            <input type="hidden" id="f-kind" value="매수">
            <div class="fa-kind-group">
              <button type="button" class="fa-kind-btn active buy" onclick="selectKind('매수', this)">매수</button>
              <button type="button" class="fa-kind-btn sell" onclick="selectKind('매도', this)">매도</button>
              <button type="button" class="fa-kind-btn div" onclick="selectKind('배당', this)">배당</button>
              <button type="button" class="fa-kind-btn deposit" onclick="selectKind('입금', this)">입금</button>
              <button type="button" class="fa-kind-btn eval" onclick="selectKind('평가금', this)">평가금</button>
            </div>
          </div>

          <!-- 종목명 선택 -->
          <div class="fa-field" id="wrap-symbol">
            <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:5px;">
              <label class="fa-label" style="margin-bottom:0;">종목명 선택</label>
              <button type="button" class="fa-btn-action" style="font-size:0.75rem; padding:2px 7px; color:var(--fa-accent); border-color:var(--fa-accent-bg);" onclick="openNewSymbolModal()">➕ 새 종목 등록</button>
            </div>
            <select id="f-symbol-select" class="fa-select" onchange="handleSymbolSelectChange()">
              <option value="">-- 종목 선택 --</option>
            </select>
            <input type="text" id="f-symbol-custom" class="fa-input" placeholder="종목명 직접 입력..." style="display:none; margin-top:6px;" oninput="handleCustomSymbolInput()">
            <input type="hidden" id="f-symbol" value="">
          </div>

          <!-- 단가 -->
          <div class="fa-field" id="wrap-unit-price">
            <label class="fa-label">체결 단가 (원/$) <span id="symbol-price-hint" style="color:var(--fa-accent); font-weight:700; font-size:0.78rem; margin-left:6px;"></span></label>
            <input type="number" step="any" id="f-unit-price" class="fa-input" placeholder="0" oninput="calcAmount()">
          </div>

          <!-- 수량 -->
          <div class="fa-field" id="wrap-quantity">
            <label class="fa-label">체결 수량 (주)</label>
            <input type="number" step="any" id="f-quantity" class="fa-input" placeholder="0" oninput="calcAmount()">
          </div>

          <!-- 정산 금액 -->
          <div class="fa-field" id="wrap-amount">
            <label class="fa-label">체결 금액 (자동계산)</label>
            <input type="number" step="any" id="f-amount" class="fa-input" placeholder="0">
          </div>

          <!-- 배당금 (구분이 배당일 때) -->
          <div class="fa-field" id="wrap-dividend" style="display:none;">
            <label class="fa-label">배당금 (원/$)</label>
            <input type="number" step="any" id="f-dividend" class="fa-input" placeholder="0">
          </div>

          <!-- 투자금 (구분이 입출금일 때) -->
          <div class="fa-field" id="wrap-deposit" style="display:none;">
            <label class="fa-label">입금액 / 투자금 (원)</label>
            <input type="number" step="any" id="f-deposit" class="fa-input" placeholder="0">
          </div>

          <!-- 평가금 (구분이 평가금일 때) -->
          <div class="fa-field" id="wrap-evaluation" style="display:none;">
            <label class="fa-label">계좌 평가금 (원)</label>
            <input type="number" step="any" id="f-evaluation" class="fa-input" placeholder="0">
          </div>

          <!-- 환율 -->
          <div class="fa-field">
            <label class="fa-label">적용 환율</label>
            <input type="number" step="any" id="f-exchange" class="fa-input" value="1.0">
          </div>

          <!-- 비고 / 메모 -->
          <div class="fa-field">
            <label class="fa-label">메모 / 비고</label>
            <input type="text" id="f-memo" class="fa-input" placeholder="메모 입력">
          </div>
        </div>

        <div class="fa-form-actions">
          <button type="submit" id="btn-submit" class="fa-btn-primary">거래 등록하기</button>
        </div>
      </form>
    </div>

    <!-- Trade History Table Card -->
    <div class="fa-card">
      <div class="fa-card-title">
        <span>📜 거래 기록 내역</span>
        <span id="record-count-badge" style="font-size:0.8rem; font-weight:600; color:var(--fa-text-muted);"></span>
      </div>

      <div class="fa-table-toolbar">
        <div class="fa-filter-group">
          <select id="filter-account" class="fa-select" onchange="loadRecords()">
            <option value="">전체 계좌</option>
            <option value="usa">미국 주식</option>
            <option value="kor1">국내 주식1</option>
            <option value="kor2">국내 주식2</option>
            <option value="sema">공제회</option>
            <option value="irp">IRP</option>
            <option value="psf1">연금저축1</option>
            <option value="isa1">ISA1</option>
            <option value="psf2">연금저축2</option>
            <option value="isa2">ISA2</option>
          </select>
          <select id="filter-kind" class="fa-select" onchange="loadRecords()">
            <option value="">전체 구분</option>
            <option value="매수">매수</option>
            <option value="매도">매도</option>
            <option value="배당">배당</option>
            <option value="입금">입금</option>
            <option value="평가금">평가금</option>
          </select>
        </div>
        <div>
          <input type="text" id="filter-search" class="fa-input" placeholder="종목명 검색..." oninput="debounceLoadRecords()">
        </div>
      </div>

      <div class="fa-table-wrap">
        <table class="fa-table">
          <thead>
            <tr>
              <th>일자</th>
              <th>계좌</th>
              <th>구분</th>
              <th>종목명</th>
              <th class="text-right">단가</th>
              <th class="text-right">수량</th>
              <th class="text-right">투자금 (입/출금)</th>
              <th class="text-right">체결/배당금액</th>
              <th>비고</th>
              <th class="text-right">관리</th>
            </tr>
          </thead>
          <tbody id="records-tbody">
            <tr><td colspan="10" style="text-align:center; padding:30px; color:var(--fa-text-muted);">데이터 불러오는 중...</td></tr>
          </tbody>
        </table>
      </div>
    </div>
  </div>

  <div id="toast"></div>

  <!-- New Symbol Modal -->
  <div id="modal-new-symbol" class="fa-modal-overlay">
    <div class="fa-modal-box">
      <div class="fa-modal-header">
        <span>✨ 계좌 신규 종목 설정 등록 (fa.yaml)</span>
        <button type="button" class="fa-btn-action" onclick="closeNewSymbolModal()">✕</button>
      </div>
      <form id="form-new-symbol" onsubmit="handleNewSymbolSubmit(event)">
        <div class="fa-modal-body">
          <div class="fa-field">
            <label class="fa-label">대상 계좌</label>
            <select id="m-account" class="fa-select" required>
              <option value="usa">미국 주식</option>
              <option value="kor1">국내 주식1</option>
              <option value="kor2">국내 주식2</option>
              <option value="sema">공제회 (SEMA)</option>
              <option value="irp">IRP</option>
              <option value="psf1">연금저축1</option>
              <option value="isa1">ISA1</option>
              <option value="psf2">연금저축2</option>
              <option value="isa2">ISA2</option>
            </select>
          </div>
          <div class="fa-field">
            <label class="fa-label">종목 정식 명칭</label>
            <input type="text" id="m-fullname" class="fa-input" required placeholder="예: ACE 미국배당다우존스">
          </div>
          <div class="fa-field">
            <label class="fa-label">종목 표시 약칭</label>
            <input type="text" id="m-abbrev" class="fa-input" required placeholder="예: ACE SCHD">
          </div>
          <div class="fa-field">
            <label class="fa-label">거래소 티커 / 코드</label>
            <input type="text" id="m-ticker" class="fa-input" placeholder="예: KRX:402970 또는 SCHD">
          </div>
          <div style="display:grid; grid-template-columns: 1fr 1fr; gap:10px;">
            <div class="fa-field">
              <label class="fa-label">지역 분류</label>
              <select id="m-region" class="fa-select">
                <option value="국내상장미국">국내상장미국</option>
                <option value="미국">미국</option>
                <option value="국내">국내</option>
              </select>
            </div>
            <div class="fa-field">
              <label class="fa-label">대표 자산군</label>
              <input type="text" id="m-asset-class" class="fa-input" placeholder="예: SCHD, QQQ, S&P500, 국채">
            </div>
          </div>
        </div>
        <div class="fa-modal-footer">
          <button type="button" class="fa-btn-action" onclick="closeNewSymbolModal()">취소</button>
          <button type="submit" class="fa-btn-primary" style="padding:7px 18px;">설정 저장하기</button>
        </div>
      </form>
    </div>
  </div>

  <script>
    const ACCOUNT_MAP = {
      usa: "미국", kor1: "국내1", kor2: "국내2", sema: "공제회",
      irp: "IRP", psf1: "연금1", isa1: "ISA1", psf2: "연금2", isa2: "ISA2"
    };

    let allRecords = [];

    document.addEventListener("DOMContentLoaded", () => {
      flatpickr("#f-date", {
        locale: "ko",
        dateFormat: "Y-m-d",
        defaultDate: new Date()
      });
      loadSymbols();
      loadRecords();
    });

    function showToast(msg) {
      const toast = document.getElementById("toast");
      toast.innerText = msg;
      toast.classList.add("show");
      setTimeout(() => toast.classList.remove("show"), 3000);
    }

    function selectKind(kind, btnEl) {
      document.querySelectorAll(".fa-kind-btn").forEach(b => b.classList.remove("active"));
      btnEl.classList.add("active");
      document.getElementById("f-kind").value = kind;

      // 필드 가시성 토글
      const isDiv = (kind === '배당');
      const isDep = (kind === '입금');
      const isEval = (kind === '평가금');

      document.getElementById("wrap-symbol").style.display = (isDep || isEval) ? 'none' : 'flex';
      document.getElementById("wrap-unit-price").style.display = (isDep || isEval) ? 'none' : 'flex';
      document.getElementById("wrap-quantity").style.display = (isDep || isEval) ? 'none' : 'flex';
      document.getElementById("wrap-amount").style.display = (isDiv || isDep || isEval) ? 'none' : 'flex';
      document.getElementById("wrap-dividend").style.display = isDiv ? 'flex' : 'none';
      document.getElementById("wrap-deposit").style.display = isDep ? 'flex' : 'none';
      document.getElementById("wrap-evaluation").style.display = isEval ? 'flex' : 'none';
    }

    function openNewSymbolModal() {
      const curAcct = document.getElementById("f-account").value;
      document.getElementById("form-new-symbol").reset();
      document.getElementById("m-account").value = curAcct;
      document.getElementById("modal-new-symbol").classList.add("open");
    }

    function closeNewSymbolModal() {
      document.getElementById("modal-new-symbol").classList.remove("open");
    }

    async function handleNewSymbolSubmit(e) {
      e.preventDefault();
      const acct = document.getElementById("m-account").value;
      const fullName = document.getElementById("m-fullname").value.trim();
      const abbrev = document.getElementById("m-abbrev").value.trim() || fullName;
      const ticker = document.getElementById("m-ticker").value.trim() || abbrev;
      const region = document.getElementById("m-region").value;
      const assetClass = document.getElementById("m-asset-class").value.trim() || "주식";

      try {
        const res = await fetch("/api/symbols/add", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            account: acct,
            full_name: fullName,
            abbrev: abbrev,
            ticker: ticker,
            region: region,
            asset_class: assetClass
          })
        });
        const data = await res.json();
        if (data.ok) {
          showToast(`신규 종목 [${abbrev}]이(가) 등록되었습니다! ✨`);
          closeNewSymbolModal();
          
          // 현재 선택된 계좌가 모달에서 추가한 계좌라면 종목 목록 즉시 갱신
          if (document.getElementById("f-account").value === acct) {
            await loadSymbols(acct);
            const sel = document.getElementById("f-symbol-select");
            // 새로 추가된 종목 선택
            for (let i = 0; i < sel.options.length; i++) {
              if (sel.options[i].value === abbrev) {
                sel.selectedIndex = i;
                break;
              }
            }
            handleSymbolSelectChange();
          }
        } else {
          alert("종목 등록에 실패했습니다.");
        }
      } catch(err) {
        alert("네트워크 통신 오류가 발생했습니다.");
      }
    }

    let currentAccountSymbols = [];

    async function loadSymbols(acct) {
      if (!acct) acct = document.getElementById("f-account").value;
      try {
        const res = await fetch("/api/symbols?account=" + encodeURIComponent(acct));
        currentAccountSymbols = await res.json();
        
        const sel = document.getElementById("f-symbol-select");
        let opts = `<option value="">-- 종목 선택 (${currentAccountSymbols.length}개) --</option>`;
        opts += currentAccountSymbols.map(s => {
          return `<option value="${s.symbol}" data-price="${s.latest_price || ''}">${s.symbol}</option>`;
        }).join("");
        opts += `<option value="__custom__">➕ [직접 입력...]</option>`;
        sel.innerHTML = opts;

        // 종목 상태 초기화
        document.getElementById("f-symbol-custom").style.display = "none";
        document.getElementById("f-symbol").value = "";
        document.getElementById("symbol-price-hint").innerText = "";
      } catch(e) {
        console.error("종목 로드 실패:", e);
      }
    }

    function handleAccountChange() {
      const acct = document.getElementById("f-account").value;
      const exField = document.getElementById("f-exchange");
      if (acct === 'usa') {
        if (parseFloat(exField.value) <= 1.0) exField.value = 1380.0;
      } else {
        exField.value = 1.0;
      }
      loadSymbols(acct);
    }

    function handleSymbolSelectChange() {
      const sel = document.getElementById("f-symbol-select");
      const customInput = document.getElementById("f-symbol-custom");
      const hiddenSymbol = document.getElementById("f-symbol");
      const hint = document.getElementById("symbol-price-hint");
      const val = sel.value;

      if (val === '__custom__') {
        customInput.style.display = "block";
        customInput.focus();
        hiddenSymbol.value = customInput.value.trim();
        hint.innerText = "";
      } else if (val) {
        customInput.style.display = "none";
        hiddenSymbol.value = val;
        
        // 최근 단가 자동 입력 및 힌트 표시
        const selectedOpt = sel.options[sel.selectedIndex];
        const latestPrice = selectedOpt.getAttribute("data-price");
        if (latestPrice && parseFloat(latestPrice) > 0) {
          const numPrice = parseFloat(latestPrice);
          document.getElementById("f-unit-price").value = numPrice;
          hint.innerText = `💡 최근 체결단가: ${numPrice.toLocaleString()}원`;
          calcAmount();
        } else {
          hint.innerText = "";
        }
      } else {
        customInput.style.display = "none";
        hiddenSymbol.value = "";
        hint.innerText = "";
      }
    }

    function handleCustomSymbolInput() {
      document.getElementById("f-symbol").value = document.getElementById("f-symbol-custom").value.trim();
    }

    function calcAmount() {
      const price = parseFloat(document.getElementById("f-unit-price").value) || 0;
      const qty = parseFloat(document.getElementById("f-quantity").value) || 0;
      if (price > 0 && qty > 0) {
        document.getElementById("f-amount").value = Math.round(price * qty);
      }
    }

    let debounceTimer = null;
    function debounceLoadRecords() {
      clearTimeout(debounceTimer);
      debounceTimer = setTimeout(loadRecords, 250);
    }

    async function loadRecords() {
      const acct = document.getElementById("filter-account").value;
      const kind = document.getElementById("filter-kind").value;
      const q = document.getElementById("filter-search").value.trim();

      const params = new URLSearchParams();
      if (acct) params.append("account", acct);
      if (kind) params.append("kind", kind);
      if (q) params.append("q", q);

      try {
        const res = await fetch("/api/records?" + params.toString());
        const data = await res.json();
        renderTable(data.records);
        renderKPIs(data.kpi);
      } catch(e) {
        console.error(e);
      }
    }

    function renderKPIs(kpi) {
      if (!kpi) return;
      document.getElementById("kpi-total-count").innerText = `${kpi.total_count.toLocaleString()}건`;
      document.getElementById("kpi-latest-date").innerText = kpi.latest_date || "-";
      document.getElementById("kpi-month-buy").innerText = `${Math.round(kpi.month_buy).toLocaleString()}원`;
      document.getElementById("kpi-month-div").innerText = `${Math.round(kpi.month_div).toLocaleString()}원`;
      document.getElementById("record-count-badge").innerText = `조회된 거래: ${kpi.filtered_count.toLocaleString()}건`;
    }

    function renderTable(records) {
      const tbody = document.getElementById("records-tbody");
      if (!records || records.length === 0) {
        tbody.innerHTML = `<tr><td colspan="10" style="text-align:center; padding:30px; color:var(--fa-text-muted);">등록된 거래 내역이 없습니다.</td></tr>`;
        return;
      }

      tbody.innerHTML = records.map(r => {
        let badgeClass = "badge-buy";
        let kindText = r.kind || "매수";
        if (kindText === "매도") badgeClass = "badge-sell";
        else if (kindText === "배당") badgeClass = "badge-div";
        else if (kindText === "입금") badgeClass = "badge-deposit";
        else if (kindText === "출금") badgeClass = "badge-sell";
        else if (kindText === "평가금") badgeClass = "badge-eval";

        let displaySymbol = r.symbol || "";
        if (!displaySymbol) {
          if (kindText === "입금") displaySymbol = "계좌 입금";
          else if (kindText === "출금") displaySymbol = "계좌 출금";
          else if (kindText === "평가금") displaySymbol = "계좌 총 평가금";
          else displaySymbol = "-";
        }

        // 1. 투자금 (입/출금) 컬럼
        let depositStr = "-";
        if (r.deposit > 0) {
          depositStr = `<span style="color:var(--fa-gain); font-weight:700;">+${r.deposit.toLocaleString()}원</span>`;
        } else if (r.deposit < 0) {
          depositStr = `<span style="color:var(--fa-loss); font-weight:700;">-${Math.abs(r.deposit).toLocaleString()}원</span>`;
        }

        // 2. 체결/배당금액 컬럼
        let tradeAmountStr = "-";
        if (r.dividend > 0) {
          tradeAmountStr = `<span style="color:var(--fa-purple); font-weight:700;">+${r.dividend.toLocaleString()}원</span>`;
        } else if (r.evaluation > 0) {
          tradeAmountStr = `<span style="font-weight:700;">${r.evaluation.toLocaleString()}원</span>`;
        } else if (r.amount > 0) {
          tradeAmountStr = (kindText === "매도" || r.quantity < 0)
            ? `<span style="color:var(--fa-loss); font-weight:700;">-${r.amount.toLocaleString()}원</span>`
            : `<span style="color:var(--fa-gain); font-weight:700;">+${r.amount.toLocaleString()}원</span>`;
        } else if (r.unit_price > 0 && r.quantity !== 0) {
          const calcAmt = Math.abs(r.unit_price * r.quantity);
          tradeAmountStr = (kindText === "매도" || r.quantity < 0)
            ? `<span style="color:var(--fa-loss); font-weight:700;">-${calcAmt.toLocaleString()}원</span>`
            : `<span style="color:var(--fa-gain); font-weight:700;">+${calcAmt.toLocaleString()}원</span>`;
        }

        const acctLabel = ACCOUNT_MAP[r.account] || r.account;

        return `
          <tr>
            <td style="font-weight:600;">${r.date}</td>
            <td><span class="badge badge-account">${acctLabel}</span></td>
            <td><span class="badge ${badgeClass}">${kindText}</span></td>
            <td style="font-weight:700; color:var(--fa-text-main);">${displaySymbol}</td>
            <td class="text-right">${r.unit_price > 0 ? r.unit_price.toLocaleString() : "-"}</td>
            <td class="text-right">${r.quantity !== 0 && r.quantity !== null && r.quantity !== undefined ? r.quantity.toLocaleString() : "-"}</td>
            <td class="text-right">${depositStr}</td>
            <td class="text-right">${tradeAmountStr}</td>
            <td style="color:var(--fa-text-muted); font-size:0.8rem;">${r.memo || ""}</td>
            <td class="text-right">
              <button class="fa-btn-action fa-btn-edit" onclick='startEdit(${JSON.stringify(r)})'>수정</button>
              <button class="fa-btn-action fa-btn-del" onclick="deleteRecord(${r.id})">삭제</button>
            </td>
          </tr>
        `;
      }).join("");
    }

    async function handleFormSubmit(e) {
      e.preventDefault();
      const editId = document.getElementById("edit-id").value;
      const kind = document.getElementById("f-kind").value;

      const payload = {
        date: document.getElementById("f-date").value,
        account: document.getElementById("f-account").value,
        kind: kind,
        symbol: document.getElementById("f-symbol").value.trim(),
        unit_price: parseFloat(String(document.getElementById("f-unit-price").value || "0").replace(/,/g, "")) || 0.0,
        quantity: parseFloat(String(document.getElementById("f-quantity").value || "0").replace(/,/g, "")) || 0.0,
        amount: parseFloat(String(document.getElementById("f-amount").value || "0").replace(/,/g, "")) || 0.0,
        dividend: parseFloat(String(document.getElementById("f-dividend").value || "0").replace(/,/g, "")) || 0.0,
        deposit: parseFloat(String(document.getElementById("f-deposit").value || "0").replace(/,/g, "")) || 0.0,
        evaluation: parseFloat(String(document.getElementById("f-evaluation").value || "0").replace(/,/g, "")) || 0.0,
        exchange_rate: parseFloat(String(document.getElementById("f-exchange").value || "1.0").replace(/,/g, "")) || 1.0,
        memo: document.getElementById("f-memo").value.trim()
      };

      if (kind === "입금") {
        if (payload.deposit <= 0) {
          alert("입금액 / 투자금을 올바르게 입력해주세요.");
          document.getElementById("f-deposit").focus();
          return;
        }
        if (!payload.symbol) payload.symbol = "계좌 입금";
      } else if (kind === "평가금") {
        if (payload.evaluation <= 0) {
          alert("평가금을 올바르게 입력해주세요.");
          document.getElementById("f-evaluation").focus();
          return;
        }
        if (!payload.symbol) payload.symbol = "계좌 평가금";
      } else if (kind === "배당") {
        if (payload.dividend <= 0) {
          alert("배당금을 올바르게 입력해주세요.");
          document.getElementById("f-dividend").focus();
          return;
        }
      }

      try {
        let res;
        if (editId) {
          payload.id = parseInt(editId);
          res = await fetch(`/api/records/${editId}`, {
            method: "PUT",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(payload)
          });
        } else {
          res = await fetch("/api/records", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(payload)
          });
        }

        const data = await res.json();
        if (data.ok) {
          showToast(editId ? "거래가 성공적으로 수정되었습니다! ✅" : "새 거래가 성공적으로 등록되었습니다! 🎉");
          cancelEdit();
          loadRecords();
          loadSymbols();
        } else {
          alert("저장 실패: " + (data.error || "오류가 발생했습니다."));
        }
      } catch(err) {
        alert("네트워크 통신 오류가 발생했습니다.");
      }
    }

    async function startEdit(r) {
      document.getElementById("edit-id").value = r.id;
      document.getElementById("f-date").value = r.date;
      document.getElementById("f-account").value = r.account;
      
      // 계좌별 종목 목록 로드 후 대기
      await loadSymbols(r.account);

      // 종목 셀렉트 매칭
      const sel = document.getElementById("f-symbol-select");
      const customInput = document.getElementById("f-symbol-custom");
      const hiddenSymbol = document.getElementById("f-symbol");
      const sym = r.symbol || "";
      hiddenSymbol.value = sym;

      let found = false;
      for (let i = 0; i < sel.options.length; i++) {
        if (sel.options[i].value === sym) {
          sel.selectedIndex = i;
          found = true;
          break;
        }
      }
      if (!found && sym) {
        sel.value = "__custom__";
        customInput.style.display = "block";
        customInput.value = sym;
      } else {
        customInput.style.display = "none";
      }

      document.getElementById("f-unit-price").value = r.unit_price || "";
      document.getElementById("f-quantity").value = r.quantity || "";
      document.getElementById("f-amount").value = r.amount || "";
      document.getElementById("f-dividend").value = r.dividend || "";
      document.getElementById("f-deposit").value = r.deposit || "";
      document.getElementById("f-evaluation").value = r.evaluation || "";
      document.getElementById("f-exchange").value = r.exchange_rate || "1.0";
      document.getElementById("f-memo").value = r.memo || "";

      // 버튼 뱃지 선택
      const kind = r.kind || "매수";
      const btn = Array.from(document.querySelectorAll(".fa-kind-btn")).find(b => b.innerText === kind);
      if (btn) selectKind(kind, btn);

      document.getElementById("form-heading").innerText = `✏️ 거래 수정 (ID #${r.id})`;
      document.getElementById("btn-submit").innerText = "수정 완료";
      document.getElementById("btn-cancel-edit").style.display = "inline-block";
      window.scrollTo({ top: 0, behavior: "smooth" });
    }

    function cancelEdit() {
      document.getElementById("edit-id").value = "";
      document.getElementById("trade-form").reset();
      document.getElementById("f-date").value = new Date().toISOString().split("T")[0];
      selectKind('매수', document.querySelector('.fa-kind-btn.buy'));
      document.getElementById("form-heading").innerText = "✨ 새 거래 등록";
      document.getElementById("btn-submit").innerText = "거래 등록하기";
      document.getElementById("btn-cancel-edit").style.display = "none";
      document.getElementById("f-symbol-custom").style.display = "none";
      document.getElementById("symbol-price-hint").innerText = "";
      loadSymbols();
    }

    async function deleteRecord(id) {
      if (!confirm("정말 이 거래 내역을 삭제하시겠습니까?")) return;
      try {
        const res = await fetch(`/api/records/${id}`, { method: "DELETE" });
        const data = await res.json();
        if (data.ok) {
          showToast("거래 내역이 삭제되었습니다.");
          loadRecords();
        }
      } catch(e) {
        alert("삭제 실패");
      }
    }

    async function triggerDashboardBuild() {
      showToast("대시보드 갱신 작업을 백그라운드에서 시작했습니다... ⏳");
      try {
        await fetch("/api/build-dashboard", { method: "POST" });
      } catch(e) {}
    }
  </script>
</body>
</html>
"""


class FAAdminRequestHandler(SimpleHTTPRequestHandler):
    def _send_json(self, data: Any, status: int = 200):
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        self.wfile.write(json.dumps(data, ensure_ascii=False).encode("utf-8"))

    def do_GET(self):
        parsed = urlparse(self.path)
        path = parsed.path

        if path in ("/", "/index.html", "/admin"):
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            self.wfile.write(HTML_TEMPLATE.encode("utf-8"))
            return

        if path == "/api/symbols":
            qs = parse_qs(parsed.query)
            acct = qs.get("account", [""])[0]

            conn = get_db_connection()
            cur = conn.cursor()
            if acct:
                cur.execute("""
                    SELECT symbol,
                           (SELECT unit_price FROM trading_records t2
                            WHERE t2.account = t1.account AND t2.symbol = t1.symbol AND t2.unit_price > 0
                            ORDER BY t2.date DESC, t2.id DESC LIMIT 1) AS latest_price,
                           MAX(date) AS latest_date,
                           SUM(COALESCE(quantity, 0)) AS net_qty
                    FROM trading_records t1
                    WHERE symbol != '' AND account = ?
                    GROUP BY symbol
                    HAVING net_qty > 0.0001 OR account = 'sema'
                    ORDER BY latest_date DESC, symbol ASC;
                """, (acct,))
            else:
                cur.execute("""
                    SELECT symbol,
                           (SELECT unit_price FROM trading_records t2
                            WHERE t2.symbol = t1.symbol AND t2.unit_price > 0
                            ORDER BY t2.date DESC, t2.id DESC LIMIT 1) AS latest_price,
                           MAX(date) AS latest_date,
                           SUM(COALESCE(quantity, 0)) AS net_qty
                    FROM trading_records t1
                    WHERE symbol != ''
                    GROUP BY symbol
                    HAVING net_qty > 0.0001 OR account = 'sema'
                    ORDER BY latest_date DESC, symbol ASC;
                """)
            rows = [dict(r) for r in cur.fetchall()]
            conn.close()
            self._send_json(rows)
            return

        if path == "/api/records":
            qs = parse_qs(parsed.query)
            acct = qs.get("account", [""])[0]
            kind = qs.get("kind", [""])[0]
            q = qs.get("q", [""])[0]

            conn = get_db_connection()
            cur = conn.cursor()

            query = "SELECT * FROM trading_records WHERE 1=1"
            params = []
            if acct:
                query += " AND account = ?"
                params.append(acct)
            if kind:
                query += " AND kind = ?"
                params.append(kind)
            if q:
                query += " AND (symbol LIKE ? OR account LIKE ? OR kind LIKE ? OR memo LIKE ?)"
                params.extend([f"%{q}%", f"%{q}%", f"%{q}%", f"%{q}%"])

            query += " ORDER BY REPLACE(date, '-', '.') DESC, id DESC"
            cur.execute(query, params)
            rows = [dict(r) for r in cur.fetchall()]

            # KPI 요약 계산
            cur.execute("SELECT COUNT(*) AS cnt, MAX(date) AS max_date FROM trading_records;")
            base_stat = cur.fetchone()
            total_count = base_stat["cnt"] if base_stat else 0
            latest_date = base_stat["max_date"] if base_stat else "-"

            # 최신 거래월 기준 당월 매수금 / 배당금 합산 (YYYY.MM 또는 YYYY-MM 지원)
            month_prefix = latest_date[:7].replace("-", ".") if latest_date and len(latest_date) >= 7 else date.today().strftime("%Y.%m")
            cur.execute("""
                SELECT
                    SUM(CASE 
                        WHEN (kind = '매수' OR (COALESCE(kind, '') = '' AND quantity > 0))
                        THEN COALESCE(amount, quantity * unit_price, 0)
                        ELSE 0 
                    END) AS month_buy,
                    SUM(COALESCE(dividend, 0)) AS month_div
                FROM trading_records
                WHERE REPLACE(date, '-', '.') LIKE ?;
            """, (f"{month_prefix}%",))
            month_stat = cur.fetchone()
            month_buy = month_stat["month_buy"] or 0.0
            month_div = month_stat["month_div"] or 0.0

            conn.close()

            self._send_json({
                "records": rows,
                "kpi": {
                    "total_count": total_count,
                    "latest_date": latest_date,
                    "month_buy": month_buy,
                    "month_div": month_div,
                    "filtered_count": len(rows),
                }
            })
            return

        self.send_error(HTTPStatus.NOT_FOUND)

    def do_POST(self):
        parsed = urlparse(self.path)
        path = parsed.path

        if path == "/api/build-dashboard":
            run_dashboard_update()
            self._send_json({"ok": True})
            return

        if path == "/api/symbols/add":
            length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(length).decode("utf-8")
            data = json.loads(body)
            acct = data.get("account", "")
            full_name = data.get("full_name", "").strip()
            abbrev = data.get("abbrev", "").strip() or full_name
            ticker = data.get("ticker", "").strip() or abbrev
            region = data.get("region", "국내상장미국")
            asset_class = data.get("asset_class", "주식")

            ok = add_symbol_to_fa_yaml(acct, full_name, abbrev, ticker, region, asset_class)
            self._send_json({"ok": ok})
            return

        if path == "/api/records":
            length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(length).decode("utf-8")
            data = json.loads(body)
            clean_date = str(data.get("date") or "").replace("-", ".").strip()

            conn = get_db_connection()
            cur = conn.cursor()
            cur.execute("""
                INSERT INTO trading_records (
                    date, account, symbol, kind, unit_price, quantity,
                    amount, dividend, deposit, evaluation, exchange_rate, memo
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
            """, (
                clean_date, data.get("account"), data.get("symbol", ""),
                data.get("kind", "매수"), float(data.get("unit_price") or 0),
                float(data.get("quantity") or 0), float(data.get("amount") or 0),
                float(data.get("dividend") or 0), float(data.get("deposit") or 0),
                float(data.get("evaluation") or 0), float(data.get("exchange_rate") or 1.0),
                data.get("memo", "")
            ))
            conn.commit()
            new_id = cur.lastrowid
            conn.close()

            export_db_to_csv()
            run_dashboard_update()
            self._send_json({"ok": True, "id": new_id})
            return

        self.send_error(HTTPStatus.NOT_FOUND)

    def do_PUT(self):
        parsed = urlparse(self.path)
        if parsed.path.startswith("/api/records/"):
            rec_id = int(parsed.path.split("/")[-1])
            length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(length).decode("utf-8")
            data = json.loads(body)
            clean_date = str(data.get("date") or "").replace("-", ".").strip()

            conn = get_db_connection()
            cur = conn.cursor()
            cur.execute("""
                UPDATE trading_records SET
                    date = ?, account = ?, symbol = ?, kind = ?, unit_price = ?,
                    quantity = ?, amount = ?, dividend = ?, deposit = ?,
                    evaluation = ?, exchange_rate = ?, memo = ?
                WHERE id = ?;
            """, (
                clean_date, data.get("account"), data.get("symbol", ""),
                data.get("kind", "매수"), float(data.get("unit_price") or 0),
                float(data.get("quantity") or 0), float(data.get("amount") or 0),
                float(data.get("dividend") or 0), float(data.get("deposit") or 0),
                float(data.get("evaluation") or 0), float(data.get("exchange_rate") or 1.0),
                data.get("memo", ""), rec_id
            ))
            conn.commit()
            conn.close()

            export_db_to_csv()
            run_dashboard_update()
            self._send_json({"ok": True})
            return

        self.send_error(HTTPStatus.NOT_FOUND)

    def do_DELETE(self):
        parsed = urlparse(self.path)
        if parsed.path.startswith("/api/records/"):
            rec_id = int(parsed.path.split("/")[-1])
            conn = get_db_connection()
            cur = conn.cursor()
            cur.execute("DELETE FROM trading_records WHERE id = ?;", (rec_id,))
            conn.commit()
            conn.close()

            export_db_to_csv()
            run_dashboard_update()
            self._send_json({"ok": True})
            return

        self.send_error(HTTPStatus.NOT_FOUND)


def ensure_db_normalized():
    """DB 시작 시 구분이 비어있는 과거 데이터를 배당/입금/출금/매도/매수로 자동 보정하고 날짜를 정규화합니다."""
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute("UPDATE trading_records SET kind = '배당' WHERE (kind IS NULL OR kind = '') AND dividend > 0;")
        cur.execute("UPDATE trading_records SET kind = '입금' WHERE (kind IS NULL OR kind = '') AND deposit > 0;")
        cur.execute("UPDATE trading_records SET kind = '출금' WHERE (kind IS NULL OR kind = '') AND deposit < 0;")
        cur.execute("UPDATE trading_records SET kind = '평가금' WHERE (kind IS NULL OR kind = '') AND evaluation > 0;")
        cur.execute("UPDATE trading_records SET kind = '매도' WHERE (kind IS NULL OR kind = '') AND quantity < 0;")
        cur.execute("UPDATE trading_records SET kind = '매수' WHERE (kind IS NULL OR kind = '') AND (quantity > 0 OR unit_price > 0);")
        cur.execute("UPDATE trading_records SET date = REPLACE(date, '-', '.') WHERE date LIKE '%-%';")
        conn.commit()
        conn.close()
    except Exception as e:
        print(f"(경고) DB 데이터 정규화 실패: {e}")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="FA Trading Records Admin Web Server")
    parser.add_argument("--port", type=int, default=8095, help="Port to listen on (default: 8095)")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host address (default: 0.0.0.0)")
    args = parser.parse_args()

    ensure_db_normalized()

    server_address = (args.host, args.port)
    httpd = HTTPServer(server_address, FAAdminRequestHandler)
    print(f"🚀 FA 거래내역 관리자 서버 시작: http://{args.host}:{args.port}")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n서버를 종료합니다.")


if __name__ == "__main__":
    main()
