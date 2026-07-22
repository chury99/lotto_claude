"""전략 백테스트.

과거 시점으로 돌아가 "그때까지의 데이터만" 가지고 번호를 뽑았다면 실제 당첨번호와
몇 개나 맞았을지 계산한다. uniform(균등 무작위) 전략과 비교해서, 어떤 전략이
정말로 기준선보다 나은지 확인하는 용도다.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

from . import predictor
from .analyzer import NUMBER_COLUMNS

log = logging.getLogger(__name__)

# 맞춘 개수 -> 등수 (보너스 번호는 단순화를 위해 무시)
RANK_BY_MATCH = {6: "1등", 5: "3등", 4: "4등", 3: "5등"}


@dataclass
class BacktestResult:
    strategy: str
    draws_tested: int
    games_per_draw: int
    match_counts: dict[int, int]  # 맞춘 개수 -> 게임 수
    mean_matches: float
    expected_mean: float  # 무작위 선택 시 이론적 기댓값 (6*6/45 = 0.8)

    def as_frame(self) -> pd.DataFrame:
        total = sum(self.match_counts.values())
        rows = [
            {
                "맞춘개수": k,
                "게임수": self.match_counts.get(k, 0),
                "비율": self.match_counts.get(k, 0) / total if total else 0.0,
                "등수": RANK_BY_MATCH.get(k, "-"),
            }
            for k in range(7)
        ]
        return pd.DataFrame(rows)

    def __str__(self) -> str:
        return (
            f"[{self.strategy}] {self.draws_tested}회차 x {self.games_per_draw}게임 — "
            f"평균 적중 {self.mean_matches:.4f}개 (무작위 기댓값 {self.expected_mean:.4f})"
        )


def run(
    df: pd.DataFrame,
    strategy: str = "balanced",
    test_draws: int = 100,
    games_per_draw: int = 5,
    min_history: int = 200,
    seed: int | None = 42,
) -> BacktestResult:
    """최근 test_draws개 회차에 대해 워크포워드 백테스트를 수행한다."""
    df = df.sort_values("draw_no").reset_index(drop=True)
    if len(df) < min_history + test_draws:
        raise ValueError(
            f"데이터가 부족합니다. 최소 {min_history + test_draws}회차가 필요하지만 "
            f"{len(df)}회차만 있습니다."
        )

    start = len(df) - test_draws
    rng = np.random.default_rng(seed)
    match_counts: dict[int, int] = {k: 0 for k in range(7)}
    all_matches: list[int] = []

    for i in range(start, len(df)):
        history = df.iloc[:i]  # i회차 시점에서는 그 이전 데이터만 사용 가능
        actual = set(df.iloc[i][NUMBER_COLUMNS].astype(int).tolist())

        sample = predictor.build_sampler(history, strategy)
        for _ in range(games_per_draw):
            combo = sample(rng)
            hits = len(actual & set(combo))
            match_counts[hits] += 1
            all_matches.append(hits)

    return BacktestResult(
        strategy=strategy,
        draws_tested=test_draws,
        games_per_draw=games_per_draw,
        match_counts=match_counts,
        mean_matches=float(np.mean(all_matches)),
        expected_mean=6 * 6 / 45,
    )


def compare(
    df: pd.DataFrame,
    strategies: list[str] | None = None,
    **kwargs,
) -> pd.DataFrame:
    """여러 전략을 같은 조건에서 비교한다."""
    strategies = strategies or predictor.available_strategies()
    rows = []
    for name in strategies:
        try:
            result = run(df, strategy=name, **kwargs)
        except ImportError as exc:  # 선택 의존성(torch 등) 미설치 전략은 건너뜀
            log.warning("전략 %s 건너뜀: %s", name, exc)
            continue
        rows.append({
            "전략": name,
            "평균적중": round(result.mean_matches, 4),
            "3개이상": sum(v for k, v in result.match_counts.items() if k >= 3),
            "4개이상": sum(v for k, v in result.match_counts.items() if k >= 4),
            "5개이상": sum(v for k, v in result.match_counts.items() if k >= 5),
        })
    return pd.DataFrame(rows).sort_values("평균적중", ascending=False).reset_index(drop=True)
