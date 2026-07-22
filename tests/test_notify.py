"""텔레그램 전송 테스트. 실제 네트워크 호출 없이 모두 모의(mock)로 검증한다."""

import pytest
import requests

from lotto import notify

PICKS = [[7, 14, 17, 20, 42, 45], [4, 10, 13, 32, 35, 40]]


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    """실제 환경변수가 테스트에 새어 들어오지 않게 한다."""
    monkeypatch.delenv(notify.TOKEN_ENV, raising=False)
    monkeypatch.delenv(notify.CHAT_ID_ENV, raising=False)


class FakeResponse:
    def __init__(self, body, status_code=200, raise_json=False):
        self._body = body
        self.status_code = status_code
        self._raise_json = raise_json

    def json(self):
        if self._raise_json:
            raise ValueError("not json")
        return self._body


# ------------------------------------------------------------------ 메시지 포맷

def test_format_message_contains_numbers():
    msg = notify.format_message(PICKS, draw_no=1234, strategy="unpopular")
    assert "1234회 추천 번호" in msg
    assert "unpopular" in msg
    assert "07 14 17 20 42 45" in msg
    assert "04 10 13 32 35 40" in msg
    assert "(합 145)" in msg  # 7+14+17+20+42+45


def test_format_message_labels_and_disclaimer():
    msg = notify.format_message(PICKS, draw_no=1, strategy="clt")
    assert "<b>A</b>" in msg and "<b>B</b>" in msg
    assert "당첨 확률을 높여주지 않습니다" in msg


def test_format_message_sorts_numbers():
    msg = notify.format_message([[45, 1, 30, 2, 20, 10]], draw_no=1, strategy="uniform")
    assert "01 02 10 20 30 45" in msg


def test_format_message_escapes_html():
    """전략명 등에 꺾쇠가 들어와도 HTML 구조를 깨뜨리지 않는다."""
    msg = notify.format_message(PICKS, 1, strategy="<b>evil</b>", note="a & b < c")
    assert "&lt;b&gt;evil&lt;/b&gt;" in msg
    assert "a &amp; b &lt; c" in msg


def test_format_message_with_note():
    msg = notify.format_message(PICKS, 1, "hot", note="메모입니다")
    assert "메모입니다" in msg


# ------------------------------------------------------------------ 설정 검증

def test_missing_env_raises():
    with pytest.raises(notify.NotifyError, match=notify.TOKEN_ENV):
        notify.TelegramNotifier()


def test_missing_chat_id_only(monkeypatch):
    monkeypatch.setenv(notify.TOKEN_ENV, "tok")
    with pytest.raises(notify.NotifyError, match=notify.CHAT_ID_ENV):
        notify.TelegramNotifier()


def test_reads_env(monkeypatch):
    monkeypatch.setenv(notify.TOKEN_ENV, "tok")
    monkeypatch.setenv(notify.CHAT_ID_ENV, "42")
    n = notify.TelegramNotifier()
    assert n.token == "tok" and n.chat_id == "42"


def test_explicit_args_override_env(monkeypatch):
    monkeypatch.setenv(notify.TOKEN_ENV, "env-tok")
    monkeypatch.setenv(notify.CHAT_ID_ENV, "env-chat")
    n = notify.TelegramNotifier(token="arg-tok", chat_id="arg-chat")
    assert n.token == "arg-tok" and n.chat_id == "arg-chat"


# ------------------------------------------------------------------ 전송 동작

def test_send_posts_expected_payload(monkeypatch):
    captured = {}

    def fake_post(url, json=None, timeout=None):
        captured["url"] = url
        captured["json"] = json
        return FakeResponse({"ok": True, "result": {"message_id": 7}})

    monkeypatch.setattr(requests, "post", fake_post)

    result = notify.TelegramNotifier(token="tok", chat_id="42").send("hello")

    assert captured["url"] == "https://api.telegram.org/bottok/sendMessage"
    assert captured["json"]["chat_id"] == "42"
    assert captured["json"]["text"] == "hello"
    assert captured["json"]["parse_mode"] == "HTML"
    assert result == {"message_id": 7}


def test_send_api_error_raises_without_leaking_token(monkeypatch):
    monkeypatch.setattr(requests, "post", lambda *a, **k: FakeResponse(
        {"ok": False, "description": "chat not found"}, status_code=400))

    with pytest.raises(notify.NotifyError) as exc:
        notify.TelegramNotifier(token="SECRET-TOKEN", chat_id="42").send("hi")

    assert "chat not found" in str(exc.value)
    assert "SECRET-TOKEN" not in str(exc.value)  # 토큰 유출 방지


def test_send_network_error_raises(monkeypatch):
    def boom(*a, **k):
        raise requests.ConnectionError("네트워크 끊김")

    monkeypatch.setattr(requests, "post", boom)
    with pytest.raises(notify.NotifyError, match="요청 실패"):
        notify.TelegramNotifier(token="tok", chat_id="42").send("hi")


def test_send_non_json_response_raises(monkeypatch):
    monkeypatch.setattr(requests, "post", lambda *a, **k: FakeResponse(
        None, status_code=502, raise_json=True))
    with pytest.raises(notify.NotifyError, match="해석하지 못했습니다"):
        notify.TelegramNotifier(token="tok", chat_id="42").send("hi")


def test_send_picks_end_to_end(monkeypatch):
    captured = {}

    def fake_post(url, json=None, timeout=None):
        captured.update(json)
        return FakeResponse({"ok": True, "result": {"message_id": 1}})

    monkeypatch.setattr(requests, "post", fake_post)
    monkeypatch.setenv(notify.TOKEN_ENV, "tok")
    monkeypatch.setenv(notify.CHAT_ID_ENV, "42")

    notify.send_picks(PICKS, draw_no=1234, strategy="unpopular")

    assert "1234회 추천 번호" in captured["text"]
    assert "07 14 17 20 42 45" in captured["text"]
