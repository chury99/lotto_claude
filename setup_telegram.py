#!/usr/bin/env python3
"""텔레그램 설정 도우미.

봇 토큰을 확인하고, 봇이 받은 메시지에서 chat_id를 찾아 설정 파일에 저장한다.

    python setup_telegram.py                 # 저장된 토큰으로 chat_id 채우기
    python setup_telegram.py --token 123:ABC # 토큰부터 새로 저장
    python setup_telegram.py --check         # 저장된 설정으로 연결만 확인

텔레그램은 봇이 먼저 대화를 시작할 수 없으므로, chat_id를 찾으려면 먼저
봇에게 아무 메시지나 한 번 보내야 한다.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from lotto import notify


def read_existing(path: Path) -> dict:
    """저장된 설정을 읽는다(값이 비어 있어도 그대로 돌려준다)."""
    if not path.is_file():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return data if isinstance(data, dict) else {}


def resolve_token(args_token: str | None, path: Path) -> str:
    token = (args_token or read_existing(path).get(notify.TOKEN_KEY, "")).strip()
    if not token:
        print("봇 토큰이 없습니다. --token 으로 전달하거나 설정 파일에 먼저 적어주세요.",
              file=sys.stderr)
        print("  @BotFather 에게 /newbot 으로 발급받을 수 있습니다.", file=sys.stderr)
        raise SystemExit(1)
    return token


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="텔레그램 봇 토큰 확인 및 chat_id 자동 설정",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--token", help="봇 토큰 (생략하면 설정 파일에서 읽음)")
    parser.add_argument("--chat-id", help="chat_id를 직접 지정 (자동 탐지 건너뜀)")
    parser.add_argument("--config", default=str(notify.DEFAULT_CONFIG_PATH),
                        help="설정 파일 경로")
    parser.add_argument("--check", action="store_true",
                        help="저장된 설정으로 연결만 확인하고 끝냄")
    args = parser.parse_args(argv)

    path = Path(args.config)

    if args.check:
        token, chat_id = notify.load_credentials(path)
        bot = notify.get_me(token)
        print(f"토큰 유효 — {bot.get('first_name')} (@{bot.get('username')})")
        print(f"chat_id: {chat_id}")
        print(f"설정 파일: {path}")
        return 0

    token = resolve_token(args.token, path)

    # 1. 토큰 확인
    bot = notify.get_me(token)
    print(f"봇 확인: {bot.get('first_name')} (@{bot.get('username')})")

    # 2. chat_id 결정
    chat_id = (args.chat_id or "").strip()
    if not chat_id:
        found = notify.detect_chat_ids(token)
        if not found:
            notify.save_credentials(token, "", path)
            print(f"\n토큰은 {path} 에 저장했습니다.")
            print("\nchat_id를 찾지 못했습니다. 텔레그램은 봇이 먼저 말을 걸 수 없으므로,")
            print(f"  1) 텔레그램에서 @{bot.get('username')} 을 열고")
            print("  2) 아무 메시지나 한 번 보낸 뒤 (예: 안녕)")
            print("  3) 이 스크립트를 다시 실행하세요: python setup_telegram.py")
            return 1

        if len(found) == 1:
            chat_id, label = next(iter(found.items()))
            print(f"chat_id 발견: {chat_id} ({label})")
        else:
            print("\n여러 대화가 발견됐습니다. --chat-id 로 하나를 지정하세요:")
            for cid, label in found.items():
                print(f"  {cid}  {label}")
            return 1

    # 3. 저장
    saved = notify.save_credentials(token, chat_id, path)
    print(f"\n저장 완료: {saved}")
    print("이제 아래로 바로 실행할 수 있습니다:")
    print("  python run.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
