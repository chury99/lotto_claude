"""추천 번호를 텔레그램으로 보낸다.

봇 토큰과 채팅 ID는 환경변수로만 읽는다. 코드나 저장소에 절대 넣지 말 것.

    export TELEGRAM_BOT_TOKEN="123456:ABC-DEF..."
    export TELEGRAM_CHAT_ID="123456789"

봇 만드는 법:
  1. 텔레그램에서 @BotFather 에게 /newbot — 토큰을 받는다.
  2. 만든 봇과 대화를 시작한다(아무 메시지나 한 번 보내야 봇이 답할 수 있다).
  3. https://api.telegram.org/bot<토큰>/getUpdates 를 열어 chat.id 를 확인한다.
"""

from __future__ import annotations

import html
import logging
import os

import requests

log = logging.getLogger(__name__)

API_BASE = "https://api.telegram.org"
TOKEN_ENV = "TELEGRAM_BOT_TOKEN"
CHAT_ID_ENV = "TELEGRAM_CHAT_ID"

DEFAULT_TIMEOUT = 15.0


class NotifyError(RuntimeError):
    """전송 실패 또는 설정 누락."""


def format_message(
    picks: list[list[int]],
    draw_no: int,
    strategy: str,
    note: str | None = None,
) -> str:
    """추천 조합을 텔레그램 HTML 메시지로 만든다.

    사용자 입력이 섞일 수 있는 값은 모두 이스케이프한다.
    """
    lines = [
        f"🎱 <b>{draw_no}회 추천 번호</b>",
        f"전략: <code>{html.escape(strategy)}</code> · {len(picks)}게임",
        "",
    ]
    for i, combo in enumerate(picks, start=1):
        numbers = " ".join(f"{n:02d}" for n in sorted(combo))
        lines.append(f"<b>{chr(64 + i)}</b>  <code>{numbers}</code>  (합 {sum(combo)})")

    if note:
        lines += ["", html.escape(note)]

    lines += [
        "",
        "<i>로또는 매 회차 독립적인 무작위 추첨입니다. "
        "이 번호가 당첨 확률을 높여주지 않습니다.</i>",
    ]
    return "\n".join(lines)


class TelegramNotifier:
    """텔레그램 Bot API로 메시지를 보낸다."""

    def __init__(
        self,
        token: str | None = None,
        chat_id: str | None = None,
        timeout: float = DEFAULT_TIMEOUT,
    ) -> None:
        self.token = token or os.environ.get(TOKEN_ENV, "")
        self.chat_id = chat_id or os.environ.get(CHAT_ID_ENV, "")
        self.timeout = timeout

        missing = [
            name for name, value in ((TOKEN_ENV, self.token), (CHAT_ID_ENV, self.chat_id))
            if not value
        ]
        if missing:
            raise NotifyError(
                f"환경변수가 설정되지 않았습니다: {', '.join(missing)}\n"
                f"  export {TOKEN_ENV}='봇토큰'\n"
                f"  export {CHAT_ID_ENV}='채팅ID'"
            )

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
