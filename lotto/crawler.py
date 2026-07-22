"""동행복권 로또 6/45 당첨결과 크롤러.

동행복권 사이트가 개편되면서 회차별 결과 페이지(`/lt645/result`)는 당첨번호를
HTML에 직접 담지 않고 자바스크립트로 그린다. 그래서 이 모듈은 두 갈래로 긁는다.

1. `/lt645/result` HTML — 회차 선택 드롭다운(`data-value`)이 서버에서 렌더링되므로
   여기서 "현재까지 추첨된 전체 회차 목록"을 얻는다.
2. `/lt645/selectPstLt645InfoNew.do` — 페이지 스크립트가 슬라이드를 채울 때 호출하는
   같은 사이트의 조회 주소. 한 번에 10회차씩 내려주므로, 회차마다 페이지를 하나씩
   여는 것보다 요청 수가 1/10로 줄고 서버 부담도 작다.

(2)는 커서 방식이라 `srchCursorLtEpsd`보다 오래된 10회차를 돌려준다. 존재하지 않는
회차를 커서로 주면 빈 목록이 오므로, 시작점은 `srchDir=center`로 잡는다.
"""

from __future__ import annotations

import logging
import random
import re
import time
from dataclasses import asdict, dataclass
from typing import Any, Iterator

import requests
from bs4 import BeautifulSoup

log = logging.getLogger(__name__)

BASE = "https://www.dhlottery.co.kr"
RESULT_PAGE_URL = f"{BASE}/lt645/result"
DRAW_DATA_URL = f"{BASE}/lt645/selectPstLt645InfoNew.do"

USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
)

BATCH_SIZE = 10  # 서버가 한 요청에 돌려주는 회차 수

# 크롤링 예의: 요청 사이 대기 시간(초) 범위. 무작위 지터를 준다.
MIN_DELAY = 0.3
MAX_DELAY = 0.7


class ParseError(RuntimeError):
    """응답 구조가 예상과 달라 파싱하지 못한 경우."""


@dataclass(frozen=True)
class DrawResult:
    """한 회차의 추첨 결과."""

    draw_no: int
    draw_date: str  # YYYY-MM-DD
    n1: int
    n2: int
    n3: int
    n4: int
    n5: int
    n6: int
    bonus: int
    first_prize_winners: int | None = None
    first_prize_amount: int | None = None  # 1등 1게임당 당첨금액(원)
    total_sales: int | None = None  # 해당 회차 판매금액(원)

    @property
    def numbers(self) -> list[int]:
        return [self.n1, self.n2, self.n3, self.n4, self.n5, self.n6]

    def to_dict(self) -> dict:
        return asdict(self)

    def __str__(self) -> str:
        nums = " ".join(f"{n:2d}" for n in self.numbers)
        return f"{self.draw_no}회 ({self.draw_date}) {nums} + 보너스 {self.bonus}"


def _normalize_date(raw: str) -> str:
    """'20250802' 또는 '2025-08-02' -> '2025-08-02'."""
    digits = re.sub(r"[^0-9]", "", str(raw))
    if len(digits) != 8:
        raise ParseError(f"추첨일 형식을 알 수 없습니다: {raw!r}")
    return f"{digits[:4]}-{digits[4:6]}-{digits[6:]}"


def parse_draw_item(item: dict[str, Any]) -> DrawResult:
    """조회 응답의 회차 한 건을 DrawResult로 변환한다.

    필드명은 사이트가 쓰는 축약형이다.
      ltEpsd=회차, tmNWnNo=N번째 당첨번호, bnsWnNo=보너스, ltRflYmd=추첨일,
      rnk1WnNope=1등 당첨자 수, rnk1WnAmt=1등 1게임당 당첨금액,
      rlvtEpsdSumNtslAmt=해당 회차 판매금액
    """
    try:
        numbers = [int(item[f"tm{i}WnNo"]) for i in range(1, 7)]
        draw_no = int(item["ltEpsd"])
        bonus = int(item["bnsWnNo"])
        draw_date = _normalize_date(item["ltRflYmd"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ParseError(f"회차 데이터 파싱 실패: {exc} — {item!r}") from exc

    if len(set(numbers)) != 6 or not all(1 <= n <= 45 for n in numbers):
        raise ParseError(f"{draw_no}회 당첨번호가 유효하지 않습니다: {numbers}")
    if not 1 <= bonus <= 45:
        raise ParseError(f"{draw_no}회 보너스번호가 유효하지 않습니다: {bonus}")

    def opt_int(key: str) -> int | None:
        value = item.get(key)
        try:
            return int(value) if value is not None else None
        except (TypeError, ValueError):
            return None

    numbers.sort()
    return DrawResult(
        draw_no=draw_no,
        draw_date=draw_date,
        **{f"n{i + 1}": n for i, n in enumerate(numbers)},
        bonus=bonus,
        first_prize_winners=opt_int("rnk1WnNope"),
        first_prize_amount=opt_int("rnk1WnAmt"),
        total_sales=opt_int("rlvtEpsdSumNtslAmt"),
    )


def parse_available_draws(html: str) -> list[int]:
    """결과 페이지 HTML의 회차 선택 드롭다운에서 전체 회차 번호를 뽑는다."""
    soup = BeautifulSoup(html, "lxml")
    draws = sorted(
        {
            int(el["data-value"])
            for el in soup.select("[data-value]")
            if str(el.get("data-value", "")).isdigit()
        }
    )
    if not draws:
        raise ParseError(
            "회차 목록을 찾지 못했습니다. 사이트 구조가 바뀌었을 수 있습니다."
        )
    return draws


class LottoCrawler:
    """세션을 재사용하며 동행복권 사이트에서 회차 결과를 긁어온다."""

    def __init__(self, timeout: float = 15.0, max_retries: int = 3) -> None:
        self.timeout = timeout
        self.max_retries = max_retries
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": USER_AGENT,
            "Referer": RESULT_PAGE_URL,
            "X-Requested-With": "XMLHttpRequest",
        })
        self._available: list[int] | None = None

    # ------------------------------------------------------------ 저수준 요청

    def _get(self, url: str, params: dict[str, str] | None = None) -> requests.Response:
        """실패 시 지수 백오프로 재시도한다."""
        last_error: Exception | None = None
        for attempt in range(1, self.max_retries + 1):
            try:
                resp = self.session.get(url, params=params, timeout=self.timeout)
                resp.raise_for_status()
                return resp
            except requests.RequestException as exc:
                last_error = exc
                wait = 2**attempt * 0.5
                log.warning("요청 실패 (%d/%d) %s: %s — %.1fs 후 재시도",
                            attempt, self.max_retries, params or url, exc, wait)
                time.sleep(wait)
        raise RuntimeError(f"요청에 실패했습니다: {url} {params or ''}") from last_error

    def _fetch_batch(self, **params: str) -> list[DrawResult]:
        """조회 주소를 호출해 회차 목록을 받아 파싱한다."""
        resp = self._get(DRAW_DATA_URL, params)
        try:
            payload = resp.json()
        except ValueError as exc:
            raise ParseError(f"JSON 응답이 아닙니다: {resp.text[:200]!r}") from exc

        items = ((payload or {}).get("data") or {}).get("list") or []
        results = []
        for item in items:
            try:
                results.append(parse_draw_item(item))
            except ParseError as exc:
                log.error("건너뜀: %s", exc)
        return results

    # ------------------------------------------------------------ 공개 API

    def available_draws(self, refresh: bool = False) -> list[int]:
        """추첨이 완료된 전체 회차 번호(오름차순). 결과 페이지 HTML에서 읽는다."""
        if self._available is None or refresh:
            self._available = parse_available_draws(self._get(RESULT_PAGE_URL).text)
        return self._available

    def latest_draw_no(self) -> int:
        return self.available_draws()[-1]

    def fetch_draw(self, draw_no: int) -> DrawResult:
        """단일 회차 결과."""
        for result in self._fetch_batch(srchDir="center", srchLtEpsd=str(draw_no)):
            if result.draw_no == draw_no:
                return result
        raise ParseError(f"{draw_no}회차 결과를 찾지 못했습니다.")

    def fetch_range(self, start: int, end: int) -> Iterator[DrawResult]:
        """start~end 회차를 최신부터 역순으로 수집한다 (양끝 포함).

        커서를 10회차씩 뒤로 옮기며 훑기 때문에 요청 수는 (회차 수 / 10)이다.
        yield 순서는 최신 -> 과거.
        """
        if start > end:
            return

        seen: set[int] = set()
        batch = self._fetch_batch(srchDir="center", srchLtEpsd=str(end))
        cursor: int | None = None

        while batch:
            in_range = [r for r in batch if start <= r.draw_no <= end and r.draw_no not in seen]
            for result in sorted(in_range, key=lambda r: r.draw_no, reverse=True):
                seen.add(result.draw_no)
                yield result

            oldest = min(r.draw_no for r in batch)
            if oldest <= start or oldest == cursor:  # 더 갈 곳이 없거나 진행이 멈춤
                break
            cursor = oldest

            time.sleep(random.uniform(MIN_DELAY, MAX_DELAY))
            batch = self._fetch_batch(srchDir="older", srchCursorLtEpsd=str(cursor))

    def close(self) -> None:
        self.session.close()

    def __enter__(self) -> "LottoCrawler":
        return self

    def __exit__(self, *exc_info) -> None:
        self.close()
