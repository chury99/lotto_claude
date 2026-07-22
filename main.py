#!/usr/bin/env python3
"""로또 당첨번호 분석/예측 CLI.

    python main.py update              # 당첨번호 크롤링 (증분)
    python main.py stats               # 통계 요약
    python main.py combos              # 2개/3개 번호 조합 출현 분석
    python main.py prize               # 전략별 당첨 시 기대 상금 비교
    python main.py predict -n 5        # 다음 회차 번호 추천
    python main.py backtest            # 전략 성능 검증
"""

from __future__ import annotations

import argparse
import json
import logging
import sys

import pandas as pd

from lotto import analyzer, backtest, popularity, predictor, storage


def setup_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )


def _load_or_exit(path: str) -> pd.DataFrame:
    df = storage.load(path)
    if df.empty:
        print("저장된 데이터가 없습니다. 먼저 `python main.py update`를 실행하세요.", file=sys.stderr)
        raise SystemExit(1)
    return df


def cmd_update(args: argparse.Namespace) -> None:
    df = storage.update(args.csv, start=args.start, force=args.force)
    if df.empty:
        print("수집된 데이터가 없습니다.", file=sys.stderr)
        raise SystemExit(1)
    last = df.iloc[-1]
    print(f"\n총 {len(df)}회차 보유 · 최신 {int(last['draw_no'])}회 ({last['draw_date']})")
    print(f"당첨번호: {last[analyzer.NUMBER_COLUMNS].astype(int).tolist()} + 보너스 {int(last['bonus'])}")


def cmd_stats(args: argparse.Namespace) -> None:
    df = _load_or_exit(args.csv)
    print(json.dumps(analyzer.summary(df, last_n=args.recent), ensure_ascii=False, indent=2))


def cmd_predict(args: argparse.Namespace) -> None:
    df = _load_or_exit(args.csv)
    next_draw = int(df["draw_no"].max()) + 1
    picks = predictor.predict(
        df,
        strategy=args.strategy,
        games=args.games,
        seed=args.seed,
        use_filter=not args.no_filter,
    )

    print(f"\n=== {next_draw}회 추천 번호 (전략: {args.strategy}) ===")
    for i, combo in enumerate(picks, start=1):
        print(f"  {chr(64 + i)}. " + "  ".join(f"{n:2d}" for n in combo) + f"   (합계 {sum(combo)})")

    if args.show_scores:
        print("\n--- 번호별 점수 상위 15 ---")
        print(predictor.score_table(df, args.strategy).head(15).to_string())

    print("\n※ 로또는 매 회차 독립적인 무작위 추첨입니다. 이 추천은 과거 데이터의")
    print("   통계적 편차에 기반한 참고용이며, 당첨 확률을 높여주지 않습니다.")

    if args.telegram:
        from lotto import notify  # requests만 필요하지만 사용 시점에 임포트
        kwargs = {"config_path": args.telegram_config} if args.telegram_config else {}
        try:
            notify.send_picks(picks, next_draw, args.strategy, **kwargs)
        except notify.NotifyError as exc:
            print(f"\n[텔레그램] 전송하지 못했습니다: {exc}", file=sys.stderr)
            raise SystemExit(1)
        print("\n[텔레그램] 전송 완료.")


def cmd_combos(args: argparse.Namespace) -> None:
    df = _load_or_exit(args.csv)
    for r, name in ((2, "2개 조합"), (3, "3개 조합")):
        counts = analyzer.combo_frequency(df, r=r)
        stats = analyzer.combo_uniformity(counts)
        print(f"\n=== {name} ({stats['categories']:,}가지) ===")
        print(f"관측 {stats['observations']:,}건 / 조합당 기대 {stats['expected_per_combo']:.2f}회")
        print(f"\n최다 출현 top {args.top}:")
        print(analyzer.top_combos(df, r=r, k=args.top).to_string(index=False))
        print(f"\n균등 가설 검정: 카이제곱 z-점수 {stats['z_score']:+.2f}, "
              f"분산/평균 {stats['dispersion']:.3f} (균등 무작위면 z≈0, 분산/평균≈1)")
        verdict = "균등 가설과 부합 — 특별히 잘 나오는 조합은 없습니다." \
            if abs(stats["z_score"]) < 3 else "균등 가설에서 벗어난 것으로 보입니다."
        print(f"해석: {verdict}")
        if args.poisson:
            print("\n출현 횟수 분포 vs 포아송 이론값:")
            print(analyzer.poisson_table(counts).to_string(index=False))


def cmd_prize(args: argparse.Namespace) -> None:
    df = _load_or_exit(args.csv)

    model = popularity.fit(df)
    print(f"\n=== 조합 인기도 모델 ({model.n_draws}회차 적합) ===")
    print("(계수 양수 = 그 특성이 있을수록 인기 조합 → 당첨 시 상금이 쪼개짐)")
    print(model.summary().round(4).to_string())

    strategies = args.strategies or predictor.available_strategies()
    table = popularity.prize_comparison(df, strategies, games=args.games, seed=args.seed)
    print(f"\n=== 전략별 기대 상금 비교 (전략당 {args.games}조합 표본) ===")
    print(table.to_string(index=False))
    print("\n※ 당첨 확률은 모든 조합이 동일합니다. 이 표의 차이는 '당첨됐을 때")
    print("   남들과 얼마나 겹치는가'에서만 나오며, 전체 기대수익률은 여전히 음수입니다.")


def cmd_backtest(args: argparse.Namespace) -> None:
    df = _load_or_exit(args.csv)
    if args.strategy:
        result = backtest.run(
            df, strategy=args.strategy, test_draws=args.draws,
            games_per_draw=args.games, seed=args.seed,
        )
        print(result)
        print(result.as_frame().to_string(index=False))
    else:
        table = backtest.compare(
            df, test_draws=args.draws, games_per_draw=args.games, seed=args.seed
        )
        print(f"\n=== 전략 비교 (최근 {args.draws}회차 x {args.games}게임) ===")
        print(table.to_string(index=False))
        print(f"\n무작위 선택 시 평균 적중 기댓값: {6 * 6 / 45:.4f}개")


def notify_config_hint() -> str:
    """--telegram 도움말에 설정 파일 위치를 안내한다."""
    from lotto import notify
    return f"설정 파일 {notify.DEFAULT_CONFIG_PATH} 필요"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="로또 6/45 당첨번호 크롤링 · 분석 · 예측",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--csv", default=str(storage.DEFAULT_CSV), help="데이터 CSV 경로")
    parser.add_argument("-v", "--verbose", action="store_true", help="상세 로그")
    sub = parser.add_subparsers(dest="command", required=True)

    p_update = sub.add_parser("update", help="당첨번호 크롤링 (없는 회차만 수집)")
    p_update.add_argument("--start", type=int, default=1, help="수집 시작 회차 (기본 1)")
    p_update.add_argument("--force", action="store_true", help="캐시 무시하고 전부 재수집")
    p_update.set_defaults(func=cmd_update)

    p_stats = sub.add_parser("stats", help="통계 요약 출력")
    p_stats.add_argument("--recent", type=int, default=100, help="최근 N회차 기준 집계")
    p_stats.set_defaults(func=cmd_stats)

    p_predict = sub.add_parser("predict", help="다음 회차 번호 추천")
    p_predict.add_argument("-n", "--games", type=int, default=5, help="추천 게임 수")
    p_predict.add_argument(
        "-s", "--strategy", default="balanced",
        choices=predictor.available_strategies(), help="예측 전략",
    )
    p_predict.add_argument("--seed", type=int, help="난수 시드 (재현용)")
    p_predict.add_argument("--no-filter", action="store_true", help="조합 필터 끄기")
    p_predict.add_argument("--show-scores", action="store_true", help="번호별 점수 표 출력")
    p_predict.add_argument("--telegram", action="store_true",
                           help=f"추천 번호를 텔레그램으로 전송 "
                                f"({notify_config_hint()})")
    p_predict.add_argument("--telegram-config", metavar="PATH",
                           help="텔레그램 설정 파일 경로 (기본 config/telegram.json)")
    p_predict.set_defaults(func=cmd_predict)

    p_combos = sub.add_parser("combos", help="2개/3개 번호 조합 출현 분석")
    p_combos.add_argument("--top", type=int, default=10, help="상위 몇 개 조합을 볼지")
    p_combos.add_argument("--poisson", action="store_true", help="포아송 분포 비교표 출력")
    p_combos.set_defaults(func=cmd_combos)

    p_prize = sub.add_parser("prize", help="전략별 '당첨 시 기대 상금' 비교 (인기도 모델)")
    p_prize.add_argument("-n", "--games", type=int, default=300, help="전략당 표본 조합 수")
    p_prize.add_argument("-s", "--strategies", nargs="*", help="비교할 전략 (생략 시 전체)")
    p_prize.add_argument("--seed", type=int, default=42, help="난수 시드")
    p_prize.set_defaults(func=cmd_prize)

    p_back = sub.add_parser("backtest", help="전략 성능 검증")
    p_back.add_argument("-s", "--strategy", choices=predictor.available_strategies(),
                        help="생략하면 전체 전략 비교")
    p_back.add_argument("-d", "--draws", type=int, default=100, help="검증할 최근 회차 수")
    p_back.add_argument("-n", "--games", type=int, default=5, help="회차당 게임 수")
    p_back.add_argument("--seed", type=int, default=42, help="난수 시드")
    p_back.set_defaults(func=cmd_backtest)

    return parser


def main() -> None:
    args = build_parser().parse_args()
    setup_logging(args.verbose)
    args.func(args)


if __name__ == "__main__":
    main()
