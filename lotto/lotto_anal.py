"""lotto-anal(2023) 프로젝트 로직 재현 — 검증용.

원본: https://github.com/chury99/lotto-anal
  m20_prob_maker.py  — 확률 예측
  m31_selector_algorithm.py — 번호 선정 (마지막 커밋 기준 '따라가기')

원본 구조를 그대로 옮긴다.

  특징(x): 과거 N회차(기본 10)의 당첨 여부를 one-hot으로 편 것.
           45개 단일번호 + 990개 2개조합 = 1,035개 × N회차 = 10,350개 특징.
  목표(y): 다음 회차에 그 번호(또는 2개조합)가 나오는지 (0/1).
  모델   : 번호마다 별도의 RandomForestClassifier(150그루, 깊이 20).
           45 + 990 = 1,035개 모델을 매 회차 학습한다.
  학습량 : 직전 500회차(config의 학습진행차수).

  선정   : '따라가기' — 단일번호 확률 상위 5개를 각각 시작점으로 삼아,
           그 번호가 포함된 2개조합 중 확률이 가장 높은 것의 상대 번호를
           이어 붙이길 반복해 6개를 채운다. 그렇게 5세트를 만든다.

이 모듈은 검증 전용이라 원본의 파일 입출력·로깅은 생략하고 계산만 남겼다.
"""

from __future__ import annotations

import logging
from itertools import combinations

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from .analyzer import NUMBER_COLUMNS

log = logging.getLogger(__name__)

PAST_DRAWS = 10      # config: 고려할과거회차
TRAIN_DRAWS = 500    # config: 학습진행차수
N_ESTIMATORS = 150
MAX_DEPTH = 20
RANDOM_STATE = 99    # 원본과 동일

NUMBERS = list(range(1, 46))
PAIRS = list(combinations(NUMBERS, 2))  # 990개


def build_indicator_matrix(df: pd.DataFrame) -> np.ndarray:
    """(회차, 1035) 0/1 행렬. 앞 45열=단일번호, 뒤 990열=2개조합."""
    draws = df.sort_values("draw_no")[NUMBER_COLUMNS].to_numpy(int)
    out = np.zeros((len(draws), 45 + len(PAIRS)), dtype=np.int8)

    pair_index = {p: i for i, p in enumerate(PAIRS)}
    for row_i, row in enumerate(draws):
        nums = sorted(row.tolist())
        for n in nums:
            out[row_i, n - 1] = 1
        for pair in combinations(nums, 2):
            out[row_i, 45 + pair_index[pair]] = 1
    return out


def build_dataset(
    indicators: np.ndarray,
    past: int = PAST_DRAWS,
) -> tuple[np.ndarray, np.ndarray]:
    """과거 past회차를 이어붙여 x, 그 다음 회차를 y로 만든다."""
    rows = len(indicators)
    if rows <= past:
        raise ValueError(f"데이터가 부족합니다 (최소 {past + 1}회차 필요, 현재 {rows}).")
    x = np.stack([indicators[i:i + past].ravel() for i in range(rows - past)])
    y = indicators[past:]
    return x, y


def predict_probabilities(
    history: pd.DataFrame,
    past: int = PAST_DRAWS,
    train_draws: int = TRAIN_DRAWS,
    targets: list[int] | None = None,
    n_jobs: int = -1,
) -> np.ndarray:
    """다음 회차의 번호별·조합별 출현 확률 1,035개를 예측한다.

    targets를 주면 그 열만 학습한다(속도용). 나머지는 0으로 남는다.
    """
    indicators = build_indicator_matrix(history)
    x_all, y_all = build_dataset(indicators, past)

    # 원본과 동일하게 최근 train_draws개만 학습에 쓴다
    x = x_all[-train_draws:]
    y = y_all[-train_draws:]

    # 예측용 입력: 가장 최근 past회차
    x_pred = indicators[-past:].ravel().reshape(1, -1)

    n_cols = y.shape[1]
    probs = np.zeros(n_cols)
    columns = range(n_cols) if targets is None else targets

    for col in columns:
        y_col = y[:, col]
        if y_col.max() == 0:  # 학습 구간에 한 번도 안 나온 조합
            continue
        model = RandomForestClassifier(
            n_estimators=N_ESTIMATORS, max_depth=MAX_DEPTH,
            random_state=RANDOM_STATE, n_jobs=n_jobs,
        )
        model.fit(x, y_col)
        # 원본: prob_1 = 1 - predict_proba[0][0]
        probs[col] = 1.0 - model.predict_proba(x_pred)[0][0]
    return probs


def follow_the_pairs(probs: np.ndarray, n_sets: int = 5) -> list[list[int]]:
    """'따라가기' 선정 로직 (원본 번호선정로직_2개번호_따라가기).

    단일번호 확률 상위 5개를 각각 시작점으로, 그 번호가 들어간 2개조합 중
    확률이 가장 높은 것의 상대 번호를 이어 붙여 6개를 채운다.
    """
    single = probs[:45]
    pair_probs = probs[45:]

    # 번호별로 (상대번호, 확률) 목록을 확률 내림차순으로 준비
    partners: dict[int, list[tuple[int, float]]] = {n: [] for n in NUMBERS}
    for idx, (a, b) in enumerate(PAIRS):
        p = pair_probs[idx]
        partners[a].append((b, p))
        partners[b].append((a, p))
    for n in NUMBERS:
        partners[n].sort(key=lambda t: t[1], reverse=True)

    starts = list(np.argsort(single)[::-1][:n_sets] + 1)

    sets: list[list[int]] = []
    for start in starts:
        combo = [int(start)]
        while len(combo) < 6:
            current = combo[-1]
            nxt = next((p for p, _ in partners[current] if p not in combo), None)
            if nxt is None:  # 이론상 도달 불가(44개 상대가 있음)
                nxt = next(n for n in NUMBERS if n not in combo)
            combo.append(int(nxt))
        sets.append(sorted(combo))
    return sets


def predict_next(
    history: pd.DataFrame,
    n_sets: int = 5,
    **kwargs,
) -> tuple[list[list[int]], np.ndarray]:
    """원본 파이프라인 그대로: 확률 예측 → 따라가기 선정."""
    probs = predict_probabilities(history, **kwargs)
    return follow_the_pairs(probs, n_sets=n_sets), probs


# ------------------------------------------------------------------ 캐시

# 1,035개 모델 학습에 회차당 2분 이상 걸리므로, 백테스트에서 매 회차 다시
# 학습하면 현실적으로 끝나지 않는다. lstm과 같은 방식으로 일정 회차마다만
# 재학습하고, 캐시가 현재 시점보다 미래의 데이터로 만들어졌으면 강제 재계산한다.
RETRAIN_EVERY = 50

_cache: tuple[int, np.ndarray] | None = None  # (학습에 쓴 마지막 회차, 확률)


def clear_cache() -> None:
    global _cache
    _cache = None


def cached_probabilities(history: pd.DataFrame, **kwargs) -> np.ndarray:
    """필요할 때만 재학습하고 1,035개 확률을 돌려준다."""
    global _cache
    max_draw = int(history["draw_no"].max())

    stale = (
        _cache is None
        or _cache[0] > max_draw                     # 미래 데이터로 학습된 캐시 -> 누출 방지
        or max_draw - _cache[0] >= RETRAIN_EVERY
    )
    if stale:
        log.info("RandomForest 1,035개 학습 시작 (%d회차까지) — 수 분 걸립니다…", max_draw)
        _cache = (max_draw, predict_probabilities(history, **kwargs))
        log.info("RandomForest 학습 완료.")
    return _cache[1]
