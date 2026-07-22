"""당첨번호 통계 분석.

여기서 계산하는 지표들은 예측기(predictor)가 번호에 점수를 매길 때 쓰는 재료다.
"""

from __future__ import annotations

import math
from collections import Counter
from itertools import combinations

import numpy as np
import pandas as pd

NUMBER_RANGE = range(1, 46)
NUMBER_COLUMNS = ["n1", "n2", "n3", "n4", "n5", "n6"]


def numbers_matrix(df: pd.DataFrame) -> np.ndarray:
    """(회차 수, 6) 정수 배열. 행 순서는 회차 오름차순."""
    return df.sort_values("draw_no")[NUMBER_COLUMNS].to_numpy(dtype=int)


def frequency(df: pd.DataFrame, last_n: int | None = None) -> pd.Series:
    """번호별 출현 횟수. last_n을 주면 최근 n회차만 집계한다."""
    data = df.sort_values("draw_no")
    if last_n is not None:
        data = data.tail(last_n)
    counts = Counter(numbers_matrix(data).ravel().tolist())
    return pd.Series(
        {n: counts.get(n, 0) for n in NUMBER_RANGE}, name="frequency"
    ).sort_index()


def weighted_frequency(df: pd.DataFrame, half_life: int = 100) -> pd.Series:
    """최근 회차에 가중치를 준 출현 빈도.

    half_life 회차 전의 결과는 절반의 무게만 갖는다. 오래된 추첨의 영향력을
    부드럽게 줄여 '최근 흐름'을 반영하기 위한 지표.
    """
    data = df.sort_values("draw_no")
    matrix = numbers_matrix(data)
    ages = np.arange(len(matrix) - 1, -1, -1)  # 최신 회차의 age = 0
    weights = 0.5 ** (ages / half_life)

    scores = np.zeros(46)
    for row, w in zip(matrix, weights):
        scores[row] += w
    return pd.Series(scores[1:], index=list(NUMBER_RANGE), name="weighted_frequency")


def gaps(df: pd.DataFrame) -> pd.Series:
    """번호별 미출현 기간(최근 몇 회차 동안 안 나왔는지).

    마지막 회차에 나온 번호는 0, 한 번도 안 나온 번호는 전체 회차 수.
    """
    matrix = numbers_matrix(df)
    total = len(matrix)
    last_seen = {n: -1 for n in NUMBER_RANGE}
    for i, row in enumerate(matrix):
        for n in row:
            last_seen[int(n)] = i
    return pd.Series(
        {n: (total - 1 - i if i >= 0 else total) for n, i in last_seen.items()},
        name="gap",
    ).sort_index()


def mean_gap(df: pd.DataFrame) -> pd.Series:
    """번호별 평균 출현 간격. 데이터가 부족하면 전체 회차 수로 대체한다."""
    matrix = numbers_matrix(df)
    total = len(matrix)
    positions: dict[int, list[int]] = {n: [] for n in NUMBER_RANGE}
    for i, row in enumerate(matrix):
        for n in row:
            positions[int(n)].append(i)

    out = {}
    for n, pos in positions.items():
        out[n] = float(np.mean(np.diff(pos))) if len(pos) >= 2 else float(total)
    return pd.Series(out, name="mean_gap").sort_index()


def pair_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """번호쌍 동시 출현 횟수 (46x46, 대각선은 0)."""
    matrix = numbers_matrix(df)
    counts = np.zeros((46, 46), dtype=int)
    for row in matrix:
        for i, a in enumerate(row):
            for b in row[i + 1:]:
                counts[a, b] += 1
                counts[b, a] += 1
    return pd.DataFrame(
        counts[1:, 1:], index=list(NUMBER_RANGE), columns=list(NUMBER_RANGE)
    )


def combo_frequency(df: pd.DataFrame, r: int = 2, last_n: int | None = None) -> pd.Series:
    """r개 번호 조합별 동시 출현 횟수. 한 번도 안 나온 조합도 0으로 포함한다.

    한 회차(6개)에는 C(6,r)개의 r-조합이 들어 있다.
      r=2: 조합 990가지, 회차당 15개 관측
      r=3: 조합 14,190가지, 회차당 20개 관측
    """
    data = df.sort_values("draw_no")
    if last_n is not None:
        data = data.tail(last_n)
    counter: Counter = Counter()
    for row in numbers_matrix(data):
        counter.update(combinations(sorted(row.tolist()), r))
    return pd.Series(
        {c: counter.get(c, 0) for c in combinations(range(1, 46), r)},
        name=f"combo{r}_count",
    )


def combo_uniformity(counts: pd.Series) -> dict[str, float]:
    """조합 출현 횟수가 균등 가설(모든 조합이 같은 확률)과 부합하는지 검정한다.

    - chi2 / z_score: 카이제곱 적합도 통계량과 그 정규 근사 z-점수.
      자유도가 크므로 z = (chi2 - dof) / sqrt(2·dof)가 표준정규에 가깝다.
      |z|가 2~3 이내면 균등 가설과 부합한다.
    - dispersion: 분산/평균. 균등 무작위(포아송 근사)라면 1에 가깝다.
    """
    k = len(counts)
    total = int(counts.sum())
    expected = total / k
    chi2 = float(((counts - expected) ** 2 / expected).sum())
    dof = k - 1
    return {
        "categories": k,
        "observations": total,
        "expected_per_combo": expected,
        "chi2": chi2,
        "dof": dof,
        "z_score": (chi2 - dof) / math.sqrt(2 * dof),
        "dispersion": float(counts.var(ddof=0) / counts.mean()),
    }


def poisson_table(counts: pd.Series) -> pd.DataFrame:
    """출현 횟수 분포를 포아송 이론값과 나란히 놓는다.

    균등 무작위라면 각 조합의 출현 횟수는 근사적으로 Poisson(λ=평균)을 따른다.
    관측 분포가 이 이론값과 잘 맞는지 눈으로 확인하는 용도.
    """
    lam = float(counts.mean())
    k = len(counts)
    observed = counts.value_counts().sort_index()
    max_count = int(observed.index.max())
    rows = []
    for i in range(max_count + 1):
        pmf = math.exp(-lam) * lam**i / math.factorial(i)
        rows.append({
            "출현횟수": i,
            "관측_조합수": int(observed.get(i, 0)),
            "포아송_기대": round(k * pmf, 1),
        })
    return pd.DataFrame(rows)


def top_combos(df: pd.DataFrame, r: int = 2, k: int = 10) -> pd.DataFrame:
    """가장 자주 함께 나온 r개 조합 상위 k개 (기대값과 비교)."""
    counts = combo_frequency(df, r=r)
    expected = counts.sum() / len(counts)
    top = counts.sort_values(ascending=False).head(k)
    return pd.DataFrame({
        "조합": ["-".join(map(str, c)) for c in top.index],
        "출현": top.to_numpy(),
        "기대": round(expected, 2),
    }).reset_index(drop=True)


def carryover_rate(df: pd.DataFrame) -> float:
    """직전 회차 번호가 다음 회차에 다시 나오는 평균 개수."""
    matrix = numbers_matrix(df)
    if len(matrix) < 2:
        return 0.0
    overlaps = [
        len(set(matrix[i].tolist()) & set(matrix[i - 1].tolist()))
        for i in range(1, len(matrix))
    ]
    return float(np.mean(overlaps))


def sum_stats(df: pd.DataFrame) -> dict[str, float]:
    """당첨번호 6개 합계의 분포. 조합 필터링 기준으로 쓴다."""
    sums = numbers_matrix(df).sum(axis=1)
    return {
        "mean": float(np.mean(sums)),
        "std": float(np.std(sums)),
        "min": int(np.min(sums)),
        "max": int(np.max(sums)),
        "p05": float(np.percentile(sums, 5)),
        "p95": float(np.percentile(sums, 95)),
    }


def odd_even_distribution(df: pd.DataFrame) -> pd.Series:
    """홀수 개수(0~6)별 회차 비율."""
    odd_counts = (numbers_matrix(df) % 2 == 1).sum(axis=1)
    dist = pd.Series(Counter(odd_counts.tolist())).reindex(range(7), fill_value=0)
    return (dist / dist.sum()).rename("odd_ratio")


def range_distribution(df: pd.DataFrame) -> pd.Series:
    """1~10, 11~20, ... 구간별 출현 비율."""
    flat = numbers_matrix(df).ravel()
    bins = [(1, 10), (11, 20), (21, 30), (31, 40), (41, 45)]
    counts = {f"{lo}-{hi}": int(((flat >= lo) & (flat <= hi)).sum()) for lo, hi in bins}
    total = sum(counts.values())
    return pd.Series({k: v / total for k, v in counts.items()}, name="range_ratio")


def summary(df: pd.DataFrame, last_n: int = 100) -> dict:
    """리포트용 종합 통계."""
    freq = frequency(df)
    recent = frequency(df, last_n=last_n)
    gap = gaps(df)
    return {
        "총_회차": int(len(df)),
        "기간": f"{df['draw_date'].iloc[0]} ~ {df['draw_date'].iloc[-1]}",
        "최다출현_top10": freq.sort_values(ascending=False).head(10).to_dict(),
        "최소출현_bottom10": freq.sort_values().head(10).to_dict(),
        f"최근{last_n}회_최다_top10": recent.sort_values(ascending=False).head(10).to_dict(),
        "장기미출현_top10": gap.sort_values(ascending=False).head(10).to_dict(),
        "합계_통계": sum_stats(df),
        "홀수개수_분포": odd_even_distribution(df).round(3).to_dict(),
        "구간별_비율": range_distribution(df).round(3).to_dict(),
        "직전회차_중복_평균": round(carryover_rate(df), 3),
    }
