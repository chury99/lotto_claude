"""정규분포 이탈도(z-score) 기반 번호 선정.

"모든 현상은 정규분포한다"는 가정 아래, 관측된 출현 횟수가 이론값에서 가장 크게
벗어난 조합을 고른다. 이탈 방향은 **하향**으로 고정한다 — 즉 기대보다 적게
나온 쪽(z가 가장 작은 쪽)을 고른다.

  k개 번호 조합이 한 회차에 모두 나올 확률은  p_k = C(6,k) / C(45,k) 이고,
  N회차 중 출현 횟수는 Binomial(N, p_k) — 정규근사하면
      기대값 μ = N·p_k,  표준편차 σ = √(N·p_k·(1−p_k))
  이탈도 z = (관측 − μ) / σ.

선정 절차(그리디):
  1단계: 45개 번호 중 z가 가장 작은(가장 덜 나온) 번호를 고른다.
  2단계: 이미 고른 번호와 짝지었을 때 z가 가장 작은 번호를 고른다.
  ...
  6단계까지 반복. 단, 완성된 6개 조합이 과거 당첨 조합과 일치하면 그 후보는
  제외한다(로또에서 같은 조합이 두 번 나온 적은 없다).

동률 처리: z가 같으면 unpopular 전략과 같은 기준 — 사람들이 덜 고르는 번호
(인기도 η가 낮은 쪽)를 우선한다.

주의: 큰 k에서는 p_k가 극도로 작아 대부분의 조합이 관측 0회이고, 관측 0회는
모두 같은 z(최솟값)를 갖는다. 따라서 실질적으로 앞 단계는 이탈도가, 뒷 단계는
동률 처리(인기도)가 번호를 결정한다. zscore_trace()로 직접 확인할 수 있다.
"""

from __future__ import annotations

import logging
import math
from math import comb

import numpy as np
import pandas as pd

from .analyzer import NUMBER_COLUMNS

log = logging.getLogger(__name__)

NUMBERS = list(range(1, 46))


def membership_matrix(df: pd.DataFrame) -> np.ndarray:
    """(회차, 45) bool 행렬. [i, n-1]이 True면 i회차에 번호 n이 나왔다."""
    draws = df.sort_values("draw_no")[NUMBER_COLUMNS].to_numpy(int)
    out = np.zeros((len(draws), 45), dtype=bool)
    for i, row in enumerate(draws):
        out[i, row - 1] = True
    return out


def combo_probability(k: int) -> float:
    """k개 번호가 한 회차에 모두 포함될 확률 C(6,k)/C(45,k)."""
    if not 1 <= k <= 6:
        raise ValueError(f"k는 1~6이어야 합니다 (받은 값 {k}).")
    return comb(6, k) / comb(45, k)


def deviation_scores(counts: np.ndarray, n_draws: int, k: int) -> np.ndarray:
    """출현 횟수 배열을 z-score로 바꾼다 (이항분포의 정규근사)."""
    p = combo_probability(k)
    mean = n_draws * p
    sd = math.sqrt(n_draws * p * (1 - p))
    return (counts - mean) / sd


def _winning_combos(df: pd.DataFrame) -> set[tuple[int, ...]]:
    """과거 당첨 조합 집합 (6개 정렬 튜플)."""
    return {
        tuple(sorted(row))
        for row in df[NUMBER_COLUMNS].to_numpy(int).tolist()
    }


def _popularity_etas(df: pd.DataFrame) -> pd.Series:
    """동률 처리용 인기도. 값이 낮을수록 사람들이 덜 고르는 번호.

    인기도 모델을 적합할 수 없으면(데이터 부족 등) 생일 편향을 근사해
    '번호가 클수록 덜 인기'로 대체한다.
    """
    from . import popularity
    try:
        return popularity.fit(df).number_etas()
    except (ValueError, KeyError) as exc:
        log.warning("인기도 모델을 적합하지 못해 번호 크기로 대체합니다: %s", exc)
        return pd.Series(-np.arange(1, 46, dtype=float), index=NUMBERS)


def _candidates_at(chosen, mask, membership, winning, n_draws):
    """다음 단계의 후보 목록과 각 후보의 z를 돌려준다."""
    k = len(chosen) + 1
    candidates = [n for n in NUMBERS if n not in chosen]

    # 마지막 번호는 과거 당첨 조합을 만들지 않는 후보만 허용
    if k == 6:
        allowed = [n for n in candidates
                   if tuple(sorted(chosen + [n])) not in winning]
        if allowed:
            candidates = allowed
        else:  # 이론상 도달 불가 — 전부 막히면 원래 후보를 쓴다
            log.warning("모든 후보가 과거 당첨 조합이라 제외 규칙을 적용하지 못했습니다.")

    counts = np.array([np.count_nonzero(mask & membership[:, n - 1])
                       for n in candidates], dtype=float)
    return candidates, counts, deviation_scores(counts, n_draws, k)


def _tied_candidates(candidates, z, etas) -> list[int]:
    """z 최솟값이 같은 후보들을 인기도가 낮은(덜 고르는) 순으로 돌려준다."""
    bottom = round(float(np.min(z)), 12)
    tied = [c for c, zi in zip(candidates, z) if round(float(zi), 12) == bottom]
    return sorted(tied, key=lambda n: etas.loc[n])


def select_combo(
    df: pd.DataFrame,
    fixed: list[int] | None = None,
    membership: np.ndarray | None = None,
    etas: pd.Series | None = None,
    winning: set[tuple[int, ...]] | None = None,
) -> list[int]:
    """기대보다 가장 적게 나온(z가 가장 작은) 방향으로 6개를 그리디로 고른다.

    fixed를 주면 그 번호들을 이미 고른 것으로 보고 이어서 채운다.
    """
    membership = membership_matrix(df) if membership is None else membership
    etas = _popularity_etas(df) if etas is None else etas
    winning = _winning_combos(df) if winning is None else winning
    n_draws = len(membership)

    chosen: list[int] = list(fixed or [])
    mask = np.ones(n_draws, dtype=bool)  # 이미 고른 번호를 모두 포함하는 회차
    for n in chosen:
        mask = mask & membership[:, n - 1]

    while len(chosen) < 6:
        candidates, _, z = _candidates_at(chosen, mask, membership, winning, n_draws)
        # z 최소(가장 덜 나온 쪽) → 동률이면 인기도가 낮은(덜 고르는) 번호
        pick = _tied_candidates(candidates, z, etas)[0]
        chosen.append(pick)
        mask = mask & membership[:, pick - 1]

    return sorted(chosen)


def fixed_prefix(
    df: pd.DataFrame,
    membership: np.ndarray | None = None,
    etas: pd.Series | None = None,
    winning: set[tuple[int, ...]] | None = None,
) -> list[int]:
    """이탈도가 단독으로 결정한(동률이 아닌) 번호들.

    이 번호들은 근거가 명확하므로 모든 세트에 고정된다. 동률이 처음 생기는
    단계에서 멈춘다.
    """
    membership = membership_matrix(df) if membership is None else membership
    etas = _popularity_etas(df) if etas is None else etas
    winning = _winning_combos(df) if winning is None else winning
    n_draws = len(membership)

    chosen: list[int] = []
    mask = np.ones(n_draws, dtype=bool)
    while len(chosen) < 6:
        candidates, _, z = _candidates_at(chosen, mask, membership, winning, n_draws)
        tied = _tied_candidates(candidates, z, etas)
        if len(tied) > 1:      # 여기서부터는 이탈도가 결정하지 못한다
            break
        chosen.append(tied[0])
        mask = mask & membership[:, tied[0] - 1]
    return chosen


def iter_sets(df: pd.DataFrame):
    """이탈도로 확정된 번호는 고정하고, 동률 단계에서만 갈라져 조합을 만든다.

    첫 세트는 원래 절차 그대로(동률마다 인기도 최저를 선택)다. 이후 세트는
    동률 단계의 차순위 후보로 갈아 끼워 만든다 — 근거가 있는 번호는 버리지 않고
    근거가 없는 자리만 바꾸는 것이다.
    """
    membership = membership_matrix(df)
    etas = _popularity_etas(df)
    winning = _winning_combos(df)
    n_draws = len(membership)

    prefix = fixed_prefix(df, membership, etas, winning)

    def complete(base: list[int]) -> list[int]:
        return select_combo(df, fixed=base, membership=membership,
                            etas=etas, winning=winning)

    seen: set[tuple[int, ...]] = set()

    def emit(combo: list[int]):
        key = tuple(combo)
        if key not in seen:
            seen.add(key)
            return combo
        return None

    primary = complete(prefix)
    first = emit(primary)
    if first:
        yield first

    # 동률이 생긴 단계들을 순서대로 훑으며 차순위 후보로 분기한다
    branch_base = list(prefix)
    mask = np.ones(n_draws, dtype=bool)
    for n in branch_base:
        mask = mask & membership[:, n - 1]

    while len(branch_base) < 6:
        candidates, _, z = _candidates_at(branch_base, mask, membership,
                                          winning, n_draws)
        tied = _tied_candidates(candidates, z, etas)
        for alt in tied[1:]:
            combo = emit(complete(branch_base + [alt]))
            if combo:
                yield combo
        branch_base.append(tied[0])
        mask = mask & membership[:, tied[0] - 1]


def select_sets(df: pd.DataFrame, n_sets: int = 5) -> list[list[int]]:
    """이탈도 확정 번호를 공유하는 조합 n_sets개를 만든다."""
    out = []
    for combo in iter_sets(df):
        out.append(combo)
        if len(out) >= n_sets:
            break
    return out


def zscore_trace(df: pd.DataFrame, fixed: list[int] | None = None) -> pd.DataFrame:
    """단계별로 어떤 번호가 왜 뽑혔는지 기록한다(진단용).

    각 단계의 선택 번호, 그때의 관측 횟수·기대값·z, 그리고 최솟값이 동률이었는지.
    """
    membership = membership_matrix(df)
    etas = _popularity_etas(df)
    winning = _winning_combos(df)
    n_draws = len(membership)

    chosen: list[int] = list(fixed or [])
    mask = np.ones(n_draws, dtype=bool)
    for n in chosen:
        mask = mask & membership[:, n - 1]

    rows = []
    while len(chosen) < 6:
        k = len(chosen) + 1
        candidates = [n for n in NUMBERS if n not in chosen]
        if k == 6:
            allowed = [n for n in candidates
                       if tuple(sorted(chosen + [n])) not in winning]
            candidates = allowed or candidates

        counts = np.array([np.count_nonzero(mask & membership[:, n - 1])
                           for n in candidates], dtype=float)
        z = deviation_scores(counts, n_draws, k)
        best = min(range(len(candidates)),
                   key=lambda i: (round(z[i], 12), etas.loc[candidates[i]]))
        pick = candidates[best]

        bottom = round(z[best], 12)
        tied_idx = np.flatnonzero(np.round(z, 12) == bottom)
        tied = len(tied_idx)
        p = combo_probability(k)
        rows.append({
            "단계": k,
            "선택": pick,
            "관측": int(counts[best]),
            "기대": round(n_draws * p, 3),
            "z": round(z[best], 2),
            "동률후보수": tied,
            "결정요인": "이탈도" if tied == 1 else "인기도(동률)",
            "동률후보": [candidates[i] for i in tied_idx],
        })
        chosen.append(pick)
        mask = mask & membership[:, pick - 1]

    return pd.DataFrame(rows)
