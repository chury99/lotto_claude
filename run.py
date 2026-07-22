#!/usr/bin/env python3
"""전 과정 자동 실행 런처.

이 파일 하나만 실행하면 아래를 순서대로 수행한다.

    1. 수집 — 동행복권에서 새 회차를 크롤링해 CSV 갱신
    2. 분석 — 최신 데이터로 통계 요약
    3. 생성 — 전략에 따라 추천 조합 생성
    4. 발송 — 텔레그램으로 전송 (설정돼 있을 때만)

사용:
    python run.py                      # 기본값으로 전 과정 실행
    python run.py -n 5 -s unpopular    # 게임 수·전략 지정
    python run.py --no-telegram        # 발송 없이 화면 출력만
    python run.py --skip-update        # 크롤링 건너뛰고 기존 CSV 사용

cron 등록 예 (매주 토요일 18시):
    0 18 * * 6 cd /path/to/lotto_claude && .venv/bin/python run.py >> run.log 2>&1

종료 코드: 0=성공, 1=실패(수집 불가/발송 실패 등). cron에서 실패 감지에 쓴다.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from datetime import datetime

import pandas as pd

from lotto import analyzer, notify, popularity, predictor, storage

log = logging.getLogger("run")

DEFAULT_STRATEGY = "unpopular"
DEFAULT_GAMES = 5


def setup_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    # 단계별 진행 상황만 보이도록 라이브러리 로그는 낮춘다
    logging.getLogger("urllib3").setLevel(logging.WARNING)


def step(number: int, title: str) -> None:
    print(f"\n{'=' * 60}\n[{number}/4] {title}\n{'=' * 60}")


# ------------------------------------------------------------------ 1. 수집

def collect(csv_path: str, skip: bool) -> pd.DataFrame:
    step(1, "당첨번호 수집")
    if skip:
        df = storage.load(csv_path)
        if df.empty:
            print("저장된 데이터가 없습니다. --skip-update 없이 다시 실행하세요.", file=sys.stderr)
            raise SystemExit(1)
        print(f"크롤링 건너뜀 — 기존 데이터 {len(df)}회차 사용")
        return df

    try:
        df = storage.update(csv_path)
    except Exception as exc:  # 네트워크/사이트 구조 변경 등
        log.error("수집 실패: %s", exc)
        df = storage.load(csv_path)
        if df.empty:
            print("수집에 실패했고 사용할 기존 데이터도 없습니다.", file=sys.stderr)
            raise SystemExit(1) from exc
        print(f"수집에 실패해 기존 데이터 {len(df)}회차로 계속 진행합니다.")
        return df

    last = df.iloc[-1]
    print(f"보유 {len(df)}회차 · 최신 {int(last['draw_no'])}회 ({last['draw_date']})")
    print(f"최신 당첨번호: {last[analyzer.NUMBER_COLUMNS].astype(int).tolist()} "
          f"+ 보너스 {int(last['bonus'])}")
    return df


# ------------------------------------------------------------------ 2. 분석

def analyze(df: pd.DataFrame, strategy: str) -> None:
    step(2, "데이터 분석")
    freq = analyzer.frequency(df)
    recent = analyzer.frequency(df, last_n=100)
    gap = analyzer.gaps(df)

    def fmt(series: pd.Series, ascending: bool = False) -> str:
        top = series.sort_values(ascending=ascending).head(6)
        return ", ".join(f"{n}({v})" for n, v in top.items())

    print(f"전체 최다 출현 : {fmt(freq)}")
    print(f"최근 100회 최다: {fmt(recent)}")
    print(f"장기 미출현    : {fmt(gap)}")
    print(f"직전 회차 중복 : 평균 {analyzer.carryover_rate(df):.2f}개")

    if strategy == "unpopular":
        # 이 전략은 인기도 모델에 근거하므로 그 적합 결과를 함께 보여준다
        try:
            model = popularity.fit(df)
            summary = model.summary()
            print("\n인기도 모델 (계수 음수 = 사람들이 덜 사는 특성):")
            for name in ("sum", "consec", "low12"):
                row = summary.loc[name]
                print(f"  {name:8s} 계수 {row['계수']:+.4f}  z {row['z']:+.2f}")
        except ValueError as exc:
            log.warning("인기도 모델을 적합하지 못했습니다: %s", exc)


# ------------------------------------------------------------------ 3. 생성

def generate(df: pd.DataFrame, strategy: str, games: int, seed: int | None) -> tuple[list[list[int]], int]:
    step(3, "추천 조합 생성")
    next_draw = int(df["draw_no"].max()) + 1
    picks = predictor.predict(df, strategy=strategy, games=games, seed=seed)

    print(f"{next_draw}회 추천 번호 (전략: {strategy}, {games}게임)\n")
    for i, combo in enumerate(picks, start=1):
        numbers = "  ".join(f"{n:2d}" for n in combo)
        print(f"  {chr(64 + i)}. {numbers}   (합계 {sum(combo)})")
    return picks, next_draw


# ------------------------------------------------------------------ 4. 발송

def dispatch(picks: list[list[int]], next_draw: int, strategy: str, enabled: bool) -> bool:
    """텔레그램 발송. 전송했으면 True, 건너뛰었으면 False."""
    step(4, "텔레그램 발송")
    if not enabled:
        print("--no-telegram 지정 — 발송하지 않았습니다.")
        return False

    if not (os.environ.get(notify.TOKEN_ENV) and os.environ.get(notify.CHAT_ID_ENV)):
        print("텔레그램이 설정되지 않아 건너뜁니다. 발송하려면 아래를 설정하세요:")
        print(f"  export {notify.TOKEN_ENV}='봇토큰'")
        print(f"  export {notify.CHAT_ID_ENV}='채팅ID'")
        return False

    note = f"생성 시각: {datetime.now():%Y-%m-%d %H:%M}"
    try:
        notify.send_picks(picks, next_draw, strategy, note=note)
    except notify.NotifyError as exc:
        print(f"발송 실패: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc

    print("발송 완료.")
    return True


# ------------------------------------------------------------------ 진입점

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="수집 → 분석 → 생성 → 발송을 한 번에 실행",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("-n", "--games", type=int, default=DEFAULT_GAMES,
                        help=f"추천 게임 수 (기본 {DEFAULT_GAMES})")
    parser.add_argument("-s", "--strategy", default=DEFAULT_STRATEGY,
                        choices=predictor.available_strategies(),
                        help=f"예측 전략 (기본 {DEFAULT_STRATEGY})")
    parser.add_argument("--seed", type=int, help="난수 시드 (같은 조합 재현용)")
    parser.add_argument("--csv", default=str(storage.DEFAULT_CSV), help="데이터 CSV 경로")
    parser.add_argument("--skip-update", action="store_true", help="크롤링 없이 기존 CSV 사용")
    parser.add_argument("--no-telegram", action="store_true", help="텔레그램 발송 건너뛰기")
    parser.add_argument("-v", "--verbose", action="store_true", help="상세 로그")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    setup_logging(args.verbose)

    started = datetime.now()
    print(f"로또 자동 실행 시작 — {started:%Y-%m-%d %H:%M:%S}")

    df = collect(args.csv, args.skip_update)
    analyze(df, args.strategy)
    picks, next_draw = generate(df, args.strategy, args.games, args.seed)
    sent = dispatch(picks, next_draw, args.strategy, not args.no_telegram)

    elapsed = (datetime.now() - started).total_seconds()
    print(f"\n{'=' * 60}")
    print(f"완료 ({elapsed:.1f}초) — {next_draw}회 {len(picks)}게임 생성"
          f"{', 텔레그램 발송함' if sent else ''}")
    print("※ 로또는 매 회차 독립적인 무작위 추첨입니다. 추천 번호가 당첨 확률을")
    print("   높여주지 않으며, 기대수익률은 여전히 음수입니다.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
