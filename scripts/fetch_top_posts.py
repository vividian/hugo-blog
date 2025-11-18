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

def _ensure_metadata_backport() -> None:
    try:
        from importlib import metadata as stdlib_metadata  # type: ignore
        stdlib_metadata.packages_distributions  # type: ignore[attr-defined]
    except AttributeError:
        try:
            import importlib_metadata  # type: ignore[import-not-found]

            import importlib.metadata as stdlib_metadata

            stdlib_metadata.packages_distributions = importlib_metadata.packages_distributions  # type: ignore[attr-defined]
            import importlib.metadata as importlib_metadata_module  # type: ignore

            importlib_metadata_module.packages_distributions = importlib_metadata.packages_distributions  # type: ignore[attr-defined]
        except ModuleNotFoundError:
            pass


_ensure_metadata_backport()

try:
    # Google Analytics 라이브러리 임포트
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
    # 라이브러리가 설치되지 않은 경우를 대비한 처리
    GA_LIBS_AVAILABLE = False

try:
    from config_utils import get_path, get_value
except ImportError:  # When executed as module
    from .config_utils import get_path, get_value


# 스크립트의 루트 디렉토리 (blog 폴더)
ROOT = Path(__file__).resolve().parents[1]
# 인기글 데이터를 저장할 JSON 파일 경로
DATA_PATH = get_path("data") / "popular.json"
# 페이지 제목에서 제거할 접미사 패턴
TITLE_SUFFIX_PATTERNS = (
    re.compile(r"\s*\|\s*Vividian Repository", flags=re.IGNORECASE),
    re.compile(r"\s*-\s*Vividian Repository", flags=re.IGNORECASE),
)


def normalize_title(title: str) -> str:
    """페이지 제목에서 불필요한 접미사를 제거하고 공백을 정리합니다."""
    normalized = title
    for pattern in TITLE_SUFFIX_PATTERNS:
        normalized = pattern.sub("", normalized)
    return normalized.strip()


def get_client(credentials_file: str) -> BetaAnalyticsDataClient:
    """서비스 계정 인증 정보를 사용하여 Google Analytics 클라이언트를 생성합니다."""
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
    """Google Analytics에서 인기 페이지 리포트를 가져옵니다."""
    # 데이터 조회 기간 설정 (오늘부터 N일 전까지)
    end_date = datetime.utcnow().date()
    start_date = end_date - timedelta(days=date_range_days)

    # GA4 데이터 API에 보낼 요청 생성
    request = RunReportRequest(
        property=f"properties/{property_id}",
        dimensions=[
            Dimension(name="pagePath"),  # 페이지 경로
            Dimension(name="pageTitle"),  # 페이지 제목
        ],
        metrics=[Metric(name="screenPageViews")],  # 페이지 조회수
        date_ranges=[DateRange(start_date=start_date.isoformat(), end_date=end_date.isoformat())],
        limit=row_limit * 2,  # 필터링을 위해 요청 개수를 2배로 늘림
        order_bys=[
            OrderBy(
                metric=OrderBy.MetricOrderBy(metric_name="screenPageViews"),
                desc=True,  # 조회수 기준으로 내림차순 정렬
            )
        ],
    )

    # 리포트 실행
    response = client.run_report(request)

    results: list[dict[str, Any]] = []
    for row in response.rows:
        page_path = row.dimension_values[0].value or ""
        page_title = row.dimension_values[1].value or ""
        page_title = normalize_title(page_title)  # 제목 정규화
        views = int(row.metric_values[0].value or 0)

        # 유효하지 않거나, 메인 페이지, 태그/카테고리 페이지는 제외
        if not page_path or page_path == "/" or page_path.startswith("/tags/") or page_path.startswith("/category/"):
            continue

        results.append(
            {
                "page_path": page_path,
                "page_title": page_title,
                "views": views,
            }
        )

        # 원하는 개수만큼 결과가 모이면 중단
        if len(results) >= row_limit:
            break

    return results


def main() -> int:
    """스크립트의 메인 실행 함수."""
    # 설정 파일에서 Google Analytics 관련 설정값 가져오기
    property_id = get_value("google_analytics.property_id")
    credentials_rel = get_value("google_analytics.credentials_file")
    top_limit = int(get_value("google_analytics.top_limit", 5))
    date_range_days = int(get_value("google_analytics.date_range_days", 30))

    # 인증 정보 파일 경로 처리
    credentials_file = Path(credentials_rel or "")
    if credentials_rel and not credentials_file.is_absolute():
        credentials_file = (ROOT / credentials_rel).resolve()

    # 설정값 유효성 검사
    if not property_id or not credentials_rel:
        print("GA_PROPERTY_ID와 GA_CREDENTIALS_FILE 환경 변수를 설정하세요.", flush=True)
        return 1

    if not credentials_file.is_file():
        print(f"credentials 파일을 찾을 수 없습니다: {credentials_file}", flush=True)
        return 1

    # Google Analytics 라이브러리 설치 여부 확인
    if not GA_LIBS_AVAILABLE:
        print("google-analytics-data 패키지가 설치되어 있지 않습니다. 인기 글 갱신을 건너뜁니다.")
        return 0  # 오류가 아닌 정상 종료로 처리

    # GA 클라이언트 생성 및 리포트 가져오기
    client = get_client(str(credentials_file))
    popular_pages = fetch_report(client, property_id, date_range_days, top_limit)

    # 결과를 JSON 파일로 저장
    DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    DATA_PATH.write_text(json.dumps(popular_pages, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"인기 글 데이터 갱신: {len(popular_pages)}개 -> {DATA_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
