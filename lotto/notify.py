"""추천 번호를 텔레그램으로 보낸다.

봇 토큰과 채팅 ID는 설정 파일에서 읽는다(기본 `config/telegram.json`).

    {
      "bot_token": "123456:ABC-DEF...",
      "chat_id": "123456789"
    }

이 파일은 .gitignore에 있어 커밋되지 않는다. `config/telegram.json.example`을
복사해서 값을 채워 넣으면 된다.

봇 만드는 법:
  1. 텔레그램에서 @BotFather 에게 /newbot — 토큰을 받는다.
  2. 만든 봇과 대화를 시작한다(아무 메시지나 한 번 보내야 봇이 답할 수 있다).
  3. https://api.telegram.org/bot<토큰>/getUpdates 를 열어 chat.id 를 확인한다.
"""

from __future__ import annotations

import html
import json
import logging
import stat
from pathlib import Path

import requests

log = logging.getLogger(__name__)

API_BASE = "https://api.telegram.org"

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config" / "telegram.json"
EXAMPLE_CONFIG_PATH = PROJECT_ROOT / "config" / "telegram.json.example"

TOKEN_KEY = "bot_token"
CHAT_ID_KEY = "chat_id"

DEFAULT_TIMEOUT = 15.0


class NotifyError(RuntimeError):
    """전송 실패 또는 설정 누락."""


def config_exists(path: Path | str = DEFAULT_CONFIG_PATH) -> bool:
    """설정 파일이 있는지만 확인한다(내용은 읽지 않음)."""
    return Path(path).is_file()


def setup_hint(path: Path | str = DEFAULT_CONFIG_PATH) -> str:
    """설정 방법 안내 문구."""
    path = Path(path)
    return (
        f"텔레그램 설정 파일이 없습니다: {path}\n"
        f"  1) cp {EXAMPLE_CONFIG_PATH} {path}\n"
        f"  2) 파일을 열어 {TOKEN_KEY} / {CHAT_ID_KEY} 값을 채우세요\n"
        f"  3) chmod 600 {path}   (권장 — 본인만 읽도록)"
    )


def load_credentials(path: Path | str = DEFAULT_CONFIG_PATH) -> tuple[str, str]:
    """설정 파일에서 (토큰, 채팅 ID)를 읽는다."""
    path = Path(path)
    if not path.is_file():
        raise NotifyError(setup_hint(path))

    try:
        raw = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise NotifyError(f"설정 파일을 읽지 못했습니다 ({path}): {exc}") from exc

    try:
        config = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise NotifyError(
            f"설정 파일이 올바른 JSON이 아닙니다 ({path}): {exc.msg} (line {exc.lineno})"
        ) from exc

    if not isinstance(config, dict):
        raise NotifyError(f"설정 파일의 최상위는 객체여야 합니다 ({path}).")

    token = str(config.get(TOKEN_KEY, "") or "").strip()
    chat_id = str(config.get(CHAT_ID_KEY, "") or "").strip()

    missing = [k for k, v in ((TOKEN_KEY, token), (CHAT_ID_KEY, chat_id)) if not v]
    if missing:
        raise NotifyError(
            f"설정 파일에 값이 비어 있습니다 ({path}): {', '.join(missing)}"
        )
    if token.startswith("여기에") or chat_id.startswith("여기에"):
        raise NotifyError(
            f"설정 파일이 예시 그대로입니다 ({path}). 실제 값으로 바꿔주세요."
        )

    _warn_if_world_readable(path)
    return token, chat_id


def _warn_if_world_readable(path: Path) -> None:
    """자격증명 파일이 남에게도 읽히는 권한이면 경고한다."""
    try:
        mode = path.stat().st_mode
    except OSError:
        return
    if mode & (stat.S_IRGRP | stat.S_IROTH):
        log.warning(
            "%s 파일을 다른 사용자도 읽을 수 있습니다. `chmod 600 %s` 를 권장합니다.",
            path, path,
        )


def format_message(
    picks: list[list[int]],
    draw_no: int,
    strategy: str,
    note: str | None = None,
    draw_date: str | None = None,
    history: str | None = None,
) -> str:
    """추천 조합을 텔레그램 HTML 메시지로 만든다.

    draw_date: 추첨일 표기 (예: '2026-07-25 (토)')
    history:   이 전략의 과거 시뮬레이션 당첨 이력 한 줄

    사용자 입력이 섞일 수 있는 값은 모두 이스케이프한다.
    """
    lines = [f"🎱 <b>{draw_no}회 추천 번호</b>"]
    if draw_date:
        lines.append(f"추첨일: {html.escape(draw_date)}")
    lines += [
        f"전략: <code>{html.escape(strategy)}</code> · {len(picks)}게임",
        "",
    ]
    for i, combo in enumerate(picks, start=1):
        numbers = " ".join(f"{n:02d}" for n in sorted(combo))
        lines.append(f"<b>{chr(64 + i)}</b>  <code>{numbers}</code>  (합 {sum(combo)})")

    if history:
        lines += ["", f"📊 <b>이 전략 과거 시뮬레이션</b>", html.escape(history)]

    if note:
        lines += ["", html.escape(note)]

    return "\n".join(lines)


def get_me(token: str, timeout: float = DEFAULT_TIMEOUT) -> dict:
    """봇 정보를 조회한다. 토큰이 유효한지 확인하는 용도."""
    try:
        resp = requests.get(f"{API_BASE}/bot{token}/getMe", timeout=timeout)
        body = resp.json()
    except (requests.RequestException, ValueError) as exc:
        raise NotifyError(f"봇 정보를 조회하지 못했습니다: {exc}") from exc
    if not body.get("ok"):
        raise NotifyError(f"토큰이 유효하지 않습니다: {body.get('description', '알 수 없는 오류')}")
    return body.get("result", {})


def detect_chat_ids(token: str, timeout: float = DEFAULT_TIMEOUT) -> dict[str, str]:
    """봇이 받은 메시지에서 chat_id를 찾는다.

    {chat_id: 표시이름} 형태로 돌려준다. 봇에게 아직 아무도 말을 걸지 않았으면
    빈 딕셔너리다(텔레그램은 봇이 먼저 대화를 시작할 수 없다).
    """
    try:
        resp = requests.get(f"{API_BASE}/bot{token}/getUpdates", timeout=timeout)
        body = resp.json()
    except (requests.RequestException, ValueError) as exc:
        raise NotifyError(f"업데이트를 조회하지 못했습니다: {exc}") from exc
    if not body.get("ok"):
        raise NotifyError(f"업데이트 조회 실패: {body.get('description', '알 수 없는 오류')}")

    found: dict[str, str] = {}
    for update in body.get("result", []):
        message = update.get("message") or update.get("channel_post") or {}
        chat = message.get("chat") or {}
        chat_id = chat.get("id")
        if chat_id is None:
            continue
        label = (
            chat.get("title")
            or " ".join(filter(None, [chat.get("first_name"), chat.get("last_name")]))
            or chat.get("username")
            or chat.get("type", "")
        )
        found[str(chat_id)] = label
    return found


def save_credentials(
    token: str,
    chat_id: str,
    path: Path | str = DEFAULT_CONFIG_PATH,
) -> Path:
    """설정 파일에 자격증명을 쓰고 본인만 읽도록 권한을 조인다."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps({TOKEN_KEY: token, CHAT_ID_KEY: str(chat_id)},
                   indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    path.chmod(0o600)
    return path


class TelegramNotifier:
    """텔레그램 Bot API로 메시지를 보낸다."""

    def __init__(
        self,
        token: str | None = None,
        chat_id: str | None = None,
        timeout: float = DEFAULT_TIMEOUT,
        config_path: Path | str = DEFAULT_CONFIG_PATH,
    ) -> None:
        if token and chat_id:
            self.token, self.chat_id = token, chat_id
        else:
            self.token, self.chat_id = load_credentials(config_path)
        self.timeout = timeout

    def send(self, text: str) -> dict:
        """메시지를 보내고 API 응답(result)을 돌려준다."""
        url = f"{API_BASE}/bot{self.token}/sendMessage"
        payload = {
            "chat_id": self.chat_id,
            "text": text,
            "parse_mode": "HTML",
            "disable_web_page_preview": True,
        }
        try:
            resp = requests.post(url, json=payload, timeout=self.timeout)
        except requests.RequestException as exc:
            raise NotifyError(f"텔레그램 요청 실패: {exc}") from exc

        try:
            body = resp.json()
        except ValueError:
            raise NotifyError(
                f"텔레그램 응답을 해석하지 못했습니다 (HTTP {resp.status_code})."
            ) from None

        if not body.get("ok"):
            # 오류 메시지에 토큰이 섞이지 않도록 API가 준 설명만 전달한다
            raise NotifyError(
                f"텔레그램 전송 실패 (HTTP {resp.status_code}): "
                f"{body.get('description', '알 수 없는 오류')}"
            )
        log.info("텔레그램 전송 완료 (chat_id=%s).", self.chat_id)
        return body.get("result", {})


def send_picks(
    picks: list[list[int]],
    draw_no: int,
    strategy: str,
    note: str | None = None,
    **kwargs,
) -> dict:
    """추천 조합을 포맷해서 바로 전송하는 편의 함수."""
    return TelegramNotifier(**kwargs).send(
        format_message(picks, draw_no, strategy, note)
    )
