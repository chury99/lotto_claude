"""과거 당첨번호 분석 기반 다음 회차 번호 추천.

전략별로 45개 번호에 점수(가중치)를 매긴 뒤, 그 가중치로 6개를 비복원 추출한다.
마지막에 조합 필터(합계 범위, 홀짝 균형 등)로 통계적으로 드문 조합을 걸러낸다.

주의: 로또 추첨은 매 회차 독립적인 균등 무작위 시행이다. 아래 전략들은 과거
데이터의 편차를 근거로 번호를 고르지만, 그 편차가 다음 회차 확률을 바꾸지는
않는다. backtest 모듈로 실제 성능을 직접 확인해 보길 권한다.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd

from . import analyzer

NUMBERS = np.arange(1, 46)

# 전략 이름 -> 점수 함수(df -> 45개 가중치 Series)
Strategy = Callable[[pd.DataFrame], pd.Series]
_REGISTRY: dict[str, Strategy] = {}


def register(name: str) -> Callable[[Strategy], Strategy]:
    def deco(fn: Strategy) -> Strategy:
        _REGISTRY[name] = fn
        return fn
    return deco


def available_strategies() -> list[str]:
    return sorted(_REGISTRY)


def _normalize(scores: pd.Series) -> pd.Series:
    """음수를 제거하고 합이 1이 되도록 정규화한다."""
    s = scores.clip(lower=0).astype(float)
    total = s.sum()
    if total <= 0:
        return pd.Series(1 / len(s), index=s.index)
    return s / total


# ---------------------------------------------------------------- 전략들

@register("uniform")
def uniform_scores(df: pd.DataFrame) -> pd.Series:
    """균등 무작위. 다른 전략을 비교할 기준선(baseline)."""
    return pd.Series(1.0, index=NUMBERS)


@register("hot")
def hot_scores(df: pd.DataFrame, half_life: int = 100) -> pd.Series:
    """최근 자주 나온 번호(핫넘버)에 가중치."""
    return _normalize(analyzer.weighted_frequency(df, half_life=half_life))


@register("cold")
def cold_scores(df: pd.DataFrame, half_life: int = 100) -> pd.Series:
    """최근 덜 나온 번호(콜드넘버)에 가중치. hot의 반대 가정."""
    w = analyzer.weighted_frequency(df, half_life=half_life)
    return _normalize(w.max() - w + w.mean() * 0.1)


@register("overdue")
def overdue_scores(df: pd.DataFrame) -> pd.Series:
    """평균 출현 간격 대비 오래 안 나온 번호에 가중치."""
    gap = analyzer.gaps(df).astype(float)
    mean = analyzer.mean_gap(df).replace(0, np.nan)
    ratio = (gap / mean).fillna(1.0)
    return _normalize(ratio)


@register("pair")
def pair_scores(df: pd.DataFrame, last_n: int = 200) -> pd.Series:
    """직전 회차 번호들과 자주 함께 나온 번호에 가중치."""
    pairs = analyzer.pair_matrix(df.tail(last_n) if len(df) > last_n else df)
    last_numbers = df.sort_values("draw_no")[analyzer.NUMBER_COLUMNS].iloc[-1].tolist()
    affinity = pairs.loc[:, last_numbers].sum(axis=1).astype(float)
    # 직전 회차 번호가 그대로 반복되는 경우는 드무므로 약하게 눌러 준다.
    affinity.loc[last_numbers] *= 0.5
    return _normalize(affinity)


@register("balanced")
def balanced_scores(df: pd.DataFrame) -> pd.Series:
    """hot / overdue / pair를 섞은 기본 전략."""
    parts = {
        "hot": (hot_scores(df), 0.4),
        "overdue": (overdue_scores(df), 0.35),
        "pair": (pair_scores(df), 0.25),
    }
    total = sum(s * w for s, w in parts.values())
    return _normalize(total)


# ---------------------------------------------------------------- 조합 필터

@dataclass
class CombinationFilter:
    """통계적으로 드문 조합을 걸러내는 규칙 묶음.

    과거 당첨 조합의 실제 분포에서 뽑은 경계를 쓴다.
    """

    sum_min: int
    sum_max: int
    min_odd: int = 1
    max_odd: int = 5
    max_consecutive: int = 3
    max_same_decade: int = 4

    @classmethod
    def from_history(cls, df: pd.DataFrame) -> "CombinationFilter":
        stats = analyzer.sum_stats(df)
        return cls(sum_min=int(stats["p05"]), sum_max=int(stats["p95"]))

    def accepts(self, combo: list[int]) -> bool:
        combo = sorted(combo)
        if not self.sum_min <= sum(combo) <= self.sum_max:
            return False

        odd = sum(1 for n in combo if n % 2 == 1)
        if not self.min_odd <= odd <= self.max_odd:
            return False

        # 연속된 숫자가 너무 길게 이어지는 조합 배제 (예: 11,12,13,14)
        run = longest = 1
        for prev, cur in zip(combo, combo[1:]):
            run = run + 1 if cur == prev + 1 else 1
            longest = max(longest, run)
        if longest > self.max_consecutive:
            return False

        # 한 십의 자리에 몰린 조합 배제
        decades = pd.Series([n // 10 for n in combo]).value_counts()
        return int(decades.max()) <= self.max_same_decade


# ---------------------------------------------------------------- 추천 생성

def draw_combination(
    weights: pd.Series,
    rng: np.random.Generator,
    combo_filter: CombinationFilter | None = None,
    max_attempts: int = 500,
) -> list[int]:
    """가중치에 따라 번호 6개를 비복원 추출한다.

    필터를 통과하는 조합을 max_attempts까지 시도하고, 실패하면 마지막 조합을
    그대로 돌려준다(필터가 지나치게 빡빡한 경우 무한 루프 방지).
    """
    p = _normalize(weights).to_numpy()
    combo: list[int] = []
    for _ in range(max_attempts):
        combo = sorted(int(n) for n in rng.choice(NUMBERS, size=6, replace=False, p=p))
        if combo_filter is None or combo_filter.accepts(combo):
            return combo
    return combo


def predict(
    df: pd.DataFrame,
    strategy: str = "balanced",
    games: int = 5,
    seed: int | None = None,
    use_filter: bool = True,
) -> list[list[int]]:
    """다음 회차 추천 번호를 games개 만든다 (조합 중복 없음)."""
    if strategy not in _REGISTRY:
        raise ValueError(
            f"알 수 없는 전략: {strategy!r} (사용 가능: {', '.join(available_strategies())})"
        )
    if df.empty:
        raise ValueError("분석할 데이터가 없습니다. 먼저 `python main.py update`를 실행하세요.")

    weights = _REGISTRY[strategy](df)
    combo_filter = CombinationFilter.from_history(df) if use_filter else None
    rng = np.random.default_rng(seed)

    picks: list[list[int]] = []
    seen: set[tuple[int, ...]] = set()
    while len(picks) < games:
        combo = draw_combination(weights, rng, combo_filter)
        key = tuple(combo)
        if key in seen:
            continue
        seen.add(key)
        picks.append(combo)
    return picks


def score_table(df: pd.DataFrame, strategy: str = "balanced") -> pd.DataFrame:
    """번호별 점수와 근거 지표를 한 표로 정리한다."""
    weights = _normalize(_REGISTRY[strategy](df))
    return pd.DataFrame({
        "score": weights.round(5),
        "frequency": analyzer.frequency(df),
        "recent_100": analyzer.frequency(df, last_n=100),
        "gap": analyzer.gaps(df),
        "mean_gap": analyzer.mean_gap(df).round(1),
    }).sort_values("score", ascending=False)
