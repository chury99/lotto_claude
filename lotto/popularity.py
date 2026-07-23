"""조합 인기도 모델 — 당첨 확률이 아니라 '당첨 시 몇 명과 나누는가'를 다룬다.

로또 1등 상금은 당첨자끼리 나누는 패리뮤추얼 방식이므로, 남들이 많이 사는 조합은
당첨돼도 상금이 쪼개진다. 회차별 1등 당첨자 수는 그 회차 당첨 조합을 '몇 명이
샀는가'의 직접 관측치이므로, 이를 이용해 조합 특성별 인기도를 추정할 수 있다.

모델:  당첨자수_i ~ Poisson( 판매량_i × exp(β·x_i) )
       (판매량은 오프셋, x는 당첨 조합의 특성 벡터)

특성은 구매자 선호가 알려진 것들만 쓴다.
  - sum: 번호 합계 (작은 수 선호 → 계수 음수 예상)
  - low31: 생일 범위(1~31) 번호 개수
  - low12: 월 범위(1~12) 번호 개수
  - consec: 연속 번호 포함 여부 (사람들이 기피 → 계수 음수 예상)

적합은 IRLS(반복 재가중 최소제곱)로 하며 외부 의존성이 없다.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .analyzer import NUMBER_COLUMNS, numbers_matrix

log = logging.getLogger(__name__)

FEATURE_NAMES = ["intercept", "sum", "low31", "low12", "consec"]


def combo_features(combo: list[int]) -> np.ndarray:
    """조합 하나의 특성 벡터 (절편 포함)."""
    combo = sorted(combo)
    return np.array([
        1.0,
        float(sum(combo)),
        float(sum(1 for n in combo if n <= 31)),
        float(sum(1 for n in combo if n <= 12)),
        float(any(b - a == 1 for a, b in zip(combo, combo[1:]))),
    ])


def _design_matrix(df: pd.DataFrame) -> np.ndarray:
    nums = numbers_matrix(df)
    X = np.empty((len(nums), len(FEATURE_NAMES)))
    for i, row in enumerate(nums):
        X[i] = combo_features(row.tolist())
    return X


@dataclass
class PopularityModel:
    """적합된 포아송 회귀 계수와 그 표준오차."""

    coef: np.ndarray        # (5,) — FEATURE_NAMES 순서
    stderr: np.ndarray      # (5,)
    n_draws: int

    def summary(self) -> pd.DataFrame:
        z = self.coef / self.stderr
        return pd.DataFrame({
            "계수": self.coef,
            "표준오차": self.stderr,
            "z": z,
        }, index=FEATURE_NAMES)

    # ---------------------------------------------------------- 인기도 계산

    def eta(self, combo: list[int]) -> float:
        """선형 예측값 β·x (절편 포함). 클수록 인기 있는 조합."""
        return float(self.coef @ combo_features(combo))

    def expected_winners(self, combo: list[int], sales: float) -> float:
        """이 조합이 당첨됐을 때 예상되는 1등 당첨자 수(나 제외)."""
        return sales * math.exp(self.eta(combo))

    def number_etas(self) -> pd.Series:
        """번호별 인기도 기여분.

        sum/low31/low12는 번호 단위로 정확히 분해된다(조합 특성이 번호별 기여의
        합이므로). consec만 진짜 조합 수준 특성이라 여기 못 들어간다.
        """
        b = dict(zip(FEATURE_NAMES, self.coef))
        idx = np.arange(1, 46)
        etas = b["sum"] * idx + b["low31"] * (idx <= 31) + b["low12"] * (idx <= 12)
        return pd.Series(etas, index=idx, name="popularity_eta")


def expected_share(lam: float) -> float:
    """내가 추가 당첨자일 때 상금 배분 비율의 기댓값 E[1/(1+N)], N~Poisson(λ)."""
    if lam <= 0:
        return 1.0
    return (1.0 - math.exp(-lam)) / lam


def fit(df: pd.DataFrame, max_iter: int = 25, tol: float = 1e-8) -> PopularityModel:
    """회차별 (당첨 조합, 판매량, 1등 당첨자 수)로 인기도 계수를 IRLS 적합한다."""
    data = df.dropna(subset=["first_prize_winners", "total_sales"])
    data = data[data["total_sales"] > 0]
    if len(data) < 50:
        raise ValueError(f"적합에 필요한 회차가 부족합니다 (유효 {len(data)}회차).")

    X = _design_matrix(data)
    y = data["first_prize_winners"].to_numpy(float)
    offset = np.log(data["total_sales"].to_numpy(float))

    beta = np.zeros(X.shape[1])
    beta[0] = math.log(max(y.sum(), 1.0) / data["total_sales"].sum())  # 절편 초기값

    for iteration in range(max_iter):
        eta = offset + X @ beta
        mu = np.exp(np.clip(eta, -30, 30))
        # 가중 최소제곱: W=μ, 작업 반응 z = (η - offset) + (y-μ)/μ
        z = (eta - offset) + (y - mu) / mu
        XtW = X.T * mu
        hessian = XtW @ X
        new_beta = np.linalg.solve(hessian + 1e-9 * np.eye(len(beta)), XtW @ z)
        if np.max(np.abs(new_beta - beta)) < tol:
            beta = new_beta
            break
        beta = new_beta
    else:
        log.warning("IRLS가 %d회 안에 수렴하지 않았습니다.", max_iter)

    # 표준오차 = (X' W X)^-1 대각의 제곱근
    mu = np.exp(np.clip(offset + X @ beta, -30, 30))
    cov = np.linalg.inv((X.T * mu) @ X + 1e-9 * np.eye(len(beta)))
    stderr = np.sqrt(np.diag(cov))

    model = PopularityModel(coef=beta, stderr=stderr, n_draws=len(data))
    log.debug("인기도 모델 적합 완료 (%d회차):\n%s", len(data), model.summary())
    return model


def prize_comparison(
    df: pd.DataFrame,
    strategies: list[str],
    games: int = 300,
    seed: int = 42,
    recent: int = 52,
) -> pd.DataFrame:
    """전략별로 '당첨됐다면 기대 상금이 얼마인가'를 비교한다.

    최근 recent회차의 평균 판매량과 평균 1등 총 배분금(pool)을 기준으로,
    각 전략이 뽑는 조합의 예상 동시 당첨자 수 λ와 기대 상금 pool·E[1/(1+N)]을
    계산한다. 당첨 확률은 모든 조합이 동일하므로 이 표의 차이는 순수하게
    '남들과 얼마나 겹치는가'의 차이다.
    """
    from . import backtest as bt, predictor  # 순환 임포트 방지

    strategies = [s for s in strategies if s not in bt.SLOW_STRATEGIES] or strategies
    model = fit(df)
    tail = df.dropna(subset=["first_prize_winners", "first_prize_amount", "total_sales"]).tail(recent)
    sales = float(tail["total_sales"].mean())
    pool = float((tail["first_prize_winners"] * tail["first_prize_amount"]).mean())

    rows = []
    for name in strategies:
        try:
            sample = predictor.build_sampler(df, name)
        except ImportError as exc:
            log.warning("전략 %s 건너뜀: %s", name, exc)
            continue
        rng = np.random.default_rng(seed)
        lams = [model.expected_winners(sample(rng), sales) for _ in range(games)]
        prizes = [pool * expected_share(lam) for lam in lams]
        rows.append({
            "전략": name,
            "예상_동시당첨자": round(float(np.mean(lams)), 2),
            "기대상금_당첨시_억원": round(float(np.mean(prizes)) / 1e8, 2),
        })
    return (
        pd.DataFrame(rows)
        .sort_values("기대상금_당첨시_억원", ascending=False)
        .reset_index(drop=True)
    )
