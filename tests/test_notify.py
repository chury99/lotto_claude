"""텔레그램 전송 테스트. 실제 네트워크 호출 없이 모두 모의(mock)로 검증한다."""

import json

import pytest
import requests

from lotto import notify

PICKS = [[7, 14, 17, 20, 42, 45], [4, 10, 13, 32, 35, 40]]


@pytest.fixture
def config_file(tmp_path):
    """유효한 설정 파일을 만들어 경로를 돌려준다."""
    path = tmp_path / "telegram.json"
    path.write_text(json.dumps({
        notify.TOKEN_KEY: "tok", notify.CHAT_ID_KEY: "42",
    }), encoding="utf-8")
    path.chmod(0o600)
    return path


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


# ------------------------------------------------------------------ 설정 파일

def test_load_credentials(config_file):
    assert notify.load_credentials(config_file) == ("tok", "42")


def test_missing_file_raises(tmp_path):
    with pytest.raises(notify.NotifyError, match="설정 파일이 없습니다"):
        notify.load_credentials(tmp_path / "none.json")


def test_invalid_json_raises(tmp_path):
    path = tmp_path / "bad.json"
    path.write_text("{ not json", encoding="utf-8")
    with pytest.raises(notify.NotifyError, match="올바른 JSON이 아닙니다"):
        notify.load_credentials(path)


def test_non_object_json_raises(tmp_path):
    path = tmp_path / "arr.json"
    path.write_text("[1, 2]", encoding="utf-8")
    with pytest.raises(notify.NotifyError, match="객체여야"):
        notify.load_credentials(path)


@pytest.mark.parametrize("config,missing", [
    ({notify.CHAT_ID_KEY: "42"}, notify.TOKEN_KEY),
    ({notify.TOKEN_KEY: "tok"}, notify.CHAT_ID_KEY),
    ({notify.TOKEN_KEY: "", notify.CHAT_ID_KEY: "42"}, notify.TOKEN_KEY),
    ({notify.TOKEN_KEY: "tok", notify.CHAT_ID_KEY: "   "}, notify.CHAT_ID_KEY),
])
def test_missing_values_raise(tmp_path, config, missing):
    path = tmp_path / "partial.json"
    path.write_text(json.dumps(config), encoding="utf-8")
    with pytest.raises(notify.NotifyError, match=missing):
        notify.load_credentials(path)


def test_example_placeholder_rejected(tmp_path):
    """예시 파일을 그대로 복사만 하고 값을 안 채운 경우를 잡는다."""
    path = tmp_path / "example.json"
    path.write_text(notify.EXAMPLE_CONFIG_PATH.read_text(encoding="utf-8"), encoding="utf-8")
    with pytest.raises(notify.NotifyError, match="예시 그대로"):
        notify.load_credentials(path)


def test_numeric_chat_id_accepted(tmp_path):
    """chat_id를 숫자로 적어도 문자열로 정규화한다."""
    path = tmp_path / "num.json"
    path.write_text(json.dumps({notify.TOKEN_KEY: "tok", notify.CHAT_ID_KEY: 42}),
                    encoding="utf-8")
    assert notify.load_credentials(path) == ("tok", "42")


def test_world_readable_warns(tmp_path, caplog):
    path = tmp_path / "open.json"
    path.write_text(json.dumps({notify.TOKEN_KEY: "tok", notify.CHAT_ID_KEY: "42"}),
                    encoding="utf-8")
    path.chmod(0o644)
    with caplog.at_level("WARNING"):
        notify.load_credentials(path)
    assert "chmod 600" in caplog.text


def test_config_exists(config_file, tmp_path):
    assert notify.config_exists(config_file) is True
    assert notify.config_exists(tmp_path / "none.json") is False


def test_setup_hint_mentions_example():
    hint = notify.setup_hint()
    assert "telegram.json.example" in hint
    assert "chmod 600" in hint


def test_notifier_reads_config_file(config_file):
    n = notify.TelegramNotifier(config_path=config_file)
    assert n.token == "tok" and n.chat_id == "42"


def test_explicit_args_skip_file(tmp_path):
    """토큰을 직접 주면 설정 파일이 없어도 동작한다."""
    n = notify.TelegramNotifier(token="arg-tok", chat_id="arg-chat",
                                config_path=tmp_path / "none.json")
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


def test_send_picks_end_to_end(monkeypatch, config_file):
    captured = {}

    def fake_post(url, json=None, timeout=None):
        captured.update(json)
        return FakeResponse({"ok": True, "result": {"message_id": 1}})

    monkeypatch.setattr(requests, "post", fake_post)

    notify.send_picks(PICKS, draw_no=1234, strategy="unpopular",
                      config_path=config_file)

    assert "1234회 추천 번호" in captured["text"]
    assert "07 14 17 20 42 45" in captured["text"]
