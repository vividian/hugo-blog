#!/usr/bin/env python3
"""Fetch top viewed pages from Google Analytics (GA4) and store them for Hugo.

Requirements:
    pip install google-analytics-data

Output:
    Creates/overwrites blog/data/popular.json with a list of
    { "page_path": "/notes/foo/", "page_title": "Title", "views": 123 }.
"""

from __future__ import annotations

import json
import re
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

try:
    from google.analytics.data_v1beta import BetaAnalyticsDataClient
    from google.analytics.data_v1beta.types import (
        DateRange,
        Dimension,
        Metric,
        OrderBy,
        RunReportRequest,
    )
    from google.oauth2 import service_account

    GA_LIBS_AVAILABLE = True
except ModuleNotFoundError:
    GA_LIBS_AVAILABLE = False

try:
    from config_utils import get_path, get_value
except ImportError:  # When executed as module
    from .config_utils import get_path, get_value


ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = get_path("data") / "popular.json"
TITLE_SUFFIX_PATTERNS = (
    re.compile(r"\s*\|\s*Vividian Repository", flags=re.IGNORECASE),
    re.compile(r"\s*-\s*Vividian Repository", flags=re.IGNORECASE),
)


def normalize_title(title: str) -> str:
    normalized = title
    for pattern in TITLE_SUFFIX_PATTERNS:
        normalized = pattern.sub("", normalized)
    return normalized.strip()


def get_client(credentials_file: str) -> BetaAnalyticsDataClient:
    if not GA_LIBS_AVAILABLE:
        raise RuntimeError("google-analytics-data 라이브러리가 설치되어 있지 않습니다.")
    credentials = service_account.Credentials.from_service_account_file(credentials_file)
    return BetaAnalyticsDataClient(credentials=credentials)


def fetch_report(
    client: BetaAnalyticsDataClient,
    property_id: str,
    date_range_days: int,
    row_limit: int,
) -> list[dict[str, Any]]:
    end_date = datetime.utcnow().date()
    start_date = end_date - timedelta(days=date_range_days)

    request = RunReportRequest(
        property=f"properties/{property_id}",
        dimensions=[
            Dimension(name="pagePath"),
            Dimension(name="pageTitle"),
        ],
        metrics=[Metric(name="screenPageViews")],
        date_ranges=[DateRange(start_date=start_date.isoformat(), end_date=end_date.isoformat())],
        limit=row_limit * 2,  # fetch more to allow for filtering
        order_bys=[
            OrderBy(
                metric=OrderBy.MetricOrderBy(metric_name="screenPageViews"),
                desc=True,
            )
        ],
    )

    response = client.run_report(request)

    results: list[dict[str, Any]] = []
    for row in response.rows:
        page_path = row.dimension_values[0].value or ""
        page_title = row.dimension_values[1].value or ""
        page_title = normalize_title(page_title)
        views = int(row.metric_values[0].value or 0)

        if not page_path or page_path == "/" or page_path.startswith("/tags/") or page_path.startswith("/category/"):
            continue

        results.append(
            {
                "page_path": page_path,
                "page_title": page_title,
                "views": views,
            }
        )

        if len(results) >= row_limit:
            break

    return results


def main() -> int:
    property_id = get_value("google_analytics.property_id")
    credentials_rel = get_value("google_analytics.credentials_file")
    top_limit = int(get_value("google_analytics.top_limit", 5))
    date_range_days = int(get_value("google_analytics.date_range_days", 30))

    credentials_file = Path(credentials_rel or "")
    if credentials_rel and not credentials_file.is_absolute():
        credentials_file = (ROOT / credentials_rel).resolve()

    if not property_id or not credentials_rel:
        print("GA_PROPERTY_ID와 GA_CREDENTIALS_FILE 환경 변수를 설정하세요.", flush=True)
        return 1

    if not credentials_file.is_file():
        print(f"credentials 파일을 찾을 수 없습니다: {credentials_file}", flush=True)
        return 1

    if not GA_LIBS_AVAILABLE:
        print("google-analytics-data 패키지가 설치되어 있지 않습니다. 인기 글 갱신을 건너뜁니다.")
        return 0

    client = get_client(str(credentials_file))
    popular_pages = fetch_report(client, property_id, date_range_days, top_limit)

    DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    DATA_PATH.write_text(json.dumps(popular_pages, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"인기 글 데이터 갱신: {len(popular_pages)}개 -> {DATA_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
