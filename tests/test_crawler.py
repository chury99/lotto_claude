"""크롤러 파싱 테스트 (네트워크 없이 고정 데이터로 검증)."""

import pytest

from lotto.crawler import (
    LottoCrawler,
    ParseError,
    _normalize_date,
    parse_available_draws,
    parse_draw_item,
)

# 실제 응답에서 가져온 한 회차 분량 (필드 구성 그대로)
SAMPLE_ITEM = {
    "gmSqNo": 5133,
    "ltEpsd": 1183,
    "tm1WnNo": 4, "tm2WnNo": 15, "tm3WnNo": 17,
    "tm4WnNo": 23, "tm5WnNo": 27, "tm6WnNo": 36,
    "bnsWnNo": 31,
    "ltRflYmd": "20250802",
    "rnk1WnNope": 13, "rnk1WnAmt": 2073966000,
    "rnk2WnNope": 92, "rnk2WnAmt": 48843403,
    "rlvtEpsdSumNtslAmt": 57802984000,
    "wholEpsdSumNtslAmt": 115605968000,
}

SAMPLE_PAGE = """
<html><body>
  <div id="ltEpsdDiv">
    <div class="option-il" data-value="1233">1233회</div>
    <div class="option-il" data-value="1232">1232회</div>
    <div class="option-il" data-value="1231">1231회</div>
    <div class="option-il">회차 선택</div>
  </div>
</body></html>
"""


@pytest.fixture
def result():
    return parse_draw_item(SAMPLE_ITEM)


def test_draw_no_and_date(result):
    assert result.draw_no == 1183
    assert result.draw_date == "2025-08-02"


def test_numbers_are_sorted(result):
    assert result.numbers == [4, 15, 17, 23, 27, 36]
    assert result.bonus == 31


def test_numbers_sorted_even_if_response_is_not():
    item = SAMPLE_ITEM | {"tm1WnNo": 36, "tm6WnNo": 4}
    assert parse_draw_item(item).numbers == [4, 15, 17, 23, 27, 36]


def test_prize_info(result):
    assert result.first_prize_winners == 13
    assert result.first_prize_amount == 2_073_966_000
    assert result.total_sales == 57_802_984_000


def test_optional_fields_missing():
    item = {k: v for k, v in SAMPLE_ITEM.items() if not k.startswith(("rnk", "rlvt"))}
    parsed = parse_draw_item(item)
    assert parsed.first_prize_winners is None
    assert parsed.total_sales is None
    assert parsed.numbers == [4, 15, 17, 23, 27, 36]  # 필수 필드는 정상


def test_missing_required_field_raises():
    item = {k: v for k, v in SAMPLE_ITEM.items() if k != "tm3WnNo"}
    with pytest.raises(ParseError):
        parse_draw_item(item)


def test_duplicate_numbers_raise():
    with pytest.raises(ParseError, match="유효하지 않"):
        parse_draw_item(SAMPLE_ITEM | {"tm2WnNo": 4})


def test_out_of_range_number_raises():
    with pytest.raises(ParseError, match="유효하지 않"):
        parse_draw_item(SAMPLE_ITEM | {"tm1WnNo": 46})


def test_out_of_range_bonus_raises():
    with pytest.raises(ParseError, match="보너스"):
        parse_draw_item(SAMPLE_ITEM | {"bnsWnNo": 0})


def test_parse_available_draws():
    assert parse_available_draws(SAMPLE_PAGE) == [1231, 1232, 1233]


def test_parse_available_draws_empty_page():
    with pytest.raises(ParseError, match="회차 목록"):
        parse_available_draws("<html><body><p>점검중</p></body></html>")


@pytest.mark.parametrize("raw,expected", [
    ("20250802", "2025-08-02"),
    ("2025-08-02", "2025-08-02"),
    (20250802, "2025-08-02"),
])
def test_normalize_date(raw, expected):
    assert _normalize_date(raw) == expected


@pytest.mark.parametrize("raw", ["2025", "", "날짜없음"])
def test_normalize_date_invalid(raw):
    with pytest.raises(ParseError):
        _normalize_date(raw)


def _item(draw_no: int) -> dict:
    return SAMPLE_ITEM | {"ltEpsd": draw_no}


def test_fetch_range_pages_backwards(monkeypatch):
    """커서를 10회차씩 뒤로 옮기며 요청 범위를 모두 훑는지 확인한다."""
    calls = []

    def fake_batch(self, **params):
        calls.append(params)
        if params["srchDir"] == "center":
            top = int(params["srchLtEpsd"])
        else:
            top = int(params["srchCursorLtEpsd"]) - 1
        draws = [n for n in range(top, top - 10, -1) if n >= 1]
        return [parse_draw_item(_item(n)) for n in draws]

    monkeypatch.setattr(LottoCrawler, "_fetch_batch", fake_batch)
    monkeypatch.setattr("lotto.crawler.time.sleep", lambda *_: None)

    with LottoCrawler() as crawler:
        results = list(crawler.fetch_range(1, 25))

    assert [r.draw_no for r in results] == list(range(25, 0, -1))
    assert calls[0] == {"srchDir": "center", "srchLtEpsd": "25"}
    assert all(c["srchDir"] == "older" for c in calls[1:])


def test_fetch_range_no_duplicates(monkeypatch):
    """배치가 겹쳐 내려와도 같은 회차를 두 번 내보내지 않는다."""
    def fake_batch(self, **params):
        return [parse_draw_item(_item(n)) for n in range(20, 10, -1)]

    monkeypatch.setattr(LottoCrawler, "_fetch_batch", fake_batch)
    monkeypatch.setattr("lotto.crawler.time.sleep", lambda *_: None)

    with LottoCrawler() as crawler:
        results = list(crawler.fetch_range(11, 20))

    assert [r.draw_no for r in results] == list(range(20, 10, -1))


def test_fetch_range_empty_when_start_after_end(monkeypatch):
    monkeypatch.setattr(LottoCrawler, "_fetch_batch",
                        lambda self, **p: pytest.fail("요청이 나가면 안 됩니다"))
    with LottoCrawler() as crawler:
        assert list(crawler.fetch_range(10, 5)) == []
