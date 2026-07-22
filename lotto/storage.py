"""수집한 회차 데이터를 CSV로 저장/로드한다.

크롤링은 느리고 서버에 부담을 주므로, 한 번 받은 회차는 CSV에 캐시해 두고
새로 열린 회차만 추가로 수집한다(증분 업데이트).
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from .crawler import DrawResult, LottoCrawler

log = logging.getLogger(__name__)

DEFAULT_CSV = Path(__file__).resolve().parent.parent / "data" / "lotto_history.csv"

COLUMNS = [
    "draw_no", "draw_date",
    "n1", "n2", "n3", "n4", "n5", "n6", "bonus",
    "first_prize_winners", "first_prize_amount", "total_sales",
]

NUMBER_COLUMNS = ["n1", "n2", "n3", "n4", "n5", "n6"]

# 수집 도중 중단되어도 진행분이 남도록 이 개수마다 중간 저장한다.
CHECKPOINT_EVERY = 50


def load(path: Path | str = DEFAULT_CSV) -> pd.DataFrame:
    """저장된 회차 이력을 회차 오름차순 DataFrame으로 읽는다. 없으면 빈 DataFrame."""
    path = Path(path)
    if not path.exists():
        return pd.DataFrame(columns=COLUMNS)
    df = pd.read_csv(path)
    if df.empty:
        return pd.DataFrame(columns=COLUMNS)
    return df.sort_values("draw_no").reset_index(drop=True)


def save(df: pd.DataFrame, path: Path | str = DEFAULT_CSV) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.sort_values("draw_no").to_csv(path, index=False)
    return path


def append(df: pd.DataFrame, results: list[DrawResult]) -> pd.DataFrame:
    """새 회차를 기존 DataFrame에 합친다(회차 기준 중복 제거)."""
    if not results:
        return df
    new = pd.DataFrame([r.to_dict() for r in results], columns=COLUMNS)
    merged = new if df.empty else pd.concat([df, new], ignore_index=True)
    return (
        merged.drop_duplicates(subset="draw_no", keep="last")
        .sort_values("draw_no")
        .reset_index(drop=True)
    )


def update(path: Path | str = DEFAULT_CSV, start: int = 1, force: bool = False) -> pd.DataFrame:
    """빠진 회차만 크롤링해 CSV를 최신 상태로 만든다.

    force=True면 기존 캐시를 무시하고 start회차부터 전부 다시 수집한다.
    수집은 최신 -> 과거 순으로 진행하며, 중간에 끊겨도 받은 만큼은 저장된다.
    """
    df = pd.DataFrame(columns=COLUMNS) if force else load(path)

    with LottoCrawler() as crawler:
        latest = crawler.latest_draw_no()
        have = set(df["draw_no"].astype(int)) if not df.empty else set()
        missing = [n for n in range(start, latest + 1) if n not in have]

        if not missing:
            log.info("이미 최신입니다 (최신 %d회, 보유 %d회차).", latest, len(have))
            return df

        log.info("최신 %d회 — %d개 회차를 수집합니다 (%d ~ %d).",
                 latest, len(missing), missing[0], missing[-1])

        buffer: list[DrawResult] = []
        try:
            for result in crawler.fetch_range(missing[0], missing[-1]):
                if result.draw_no in have:
                    continue  # 배치에 섞여 온 이미 보유한 회차
                buffer.append(result)
                log.debug("%s", result)
                if len(buffer) >= CHECKPOINT_EVERY:
                    df = append(df, buffer)
                    save(df, path)
                    log.info("중간 저장 (%d회차까지, 누적 %d건).", result.draw_no, len(df))
                    buffer.clear()
        except KeyboardInterrupt:
            log.warning("사용자 중단 — 여기까지 저장합니다.")
        finally:
            df = append(df, buffer)
            save(df, path)

    log.info("저장 완료: %s (총 %d회차)", path, len(df))
    return df
