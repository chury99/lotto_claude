"""인기도 모델과 unpopular 전략 테스트."""

import math

import numpy as np
import pandas as pd
import pytest

from lotto import popularity, predictor


def make_history(n_draws: int = 600, low12_effect: float = 0.15, seed: int = 0) -> pd.DataFrame:
    """알려진 인기도 효과가 심어진 합성 이력.

    1등 당첨자 수 ~ Poisson(판매량 × exp(base + low12_effect × low12개수)).
    월 범위(1~12) 번호가 많은 조합일수록 당첨자가 많게 만든다.
    """
    rng = np.random.default_rng(seed)
    sales = 5e10
    base = math.log(10 / sales)  # 평균 10명 수준
    rows = []
    for i in range(1, n_draws + 1):
        picks = rng.choice(np.arange(1, 46), size=7, replace=False)
        nums = sorted(picks[:6].tolist())
        low12 = sum(1 for n in nums if n <= 12)
        lam = sales * math.exp(base + low12_effect * low12)
        rows.append({
            "draw_no": i,
            "draw_date": "2020-01-01",
            **{f"n{j+1}": nums[j] for j in range(6)},
            "bonus": int(picks[6]),
            "first_prize_winners": int(rng.poisson(lam)),
            "first_prize_amount": 2_000_000_000,
            "total_sales": sales,
        })
    return pd.DataFrame(rows)


@pytest.fixture
def df():
    return make_history()


def test_combo_features():
    x = popularity.combo_features([1, 5, 12, 31, 32, 45])
    assert x.tolist() == [1.0, 126.0, 4.0, 3.0, 1.0]  # 절편, 합, ≤31, ≤12, 연속(31-32)


def test_combo_features_no_consec():
    x = popularity.combo_features([2, 10, 20, 30, 40, 44])
    assert x[4] == 0.0


def test_fit_recovers_planted_effect(df):
    """심어둔 low12 효과(+0.15)를 회귀가 복원해야 한다."""
    model = popularity.fit(df)
    b = dict(zip(popularity.FEATURE_NAMES, model.coef))
    assert b["low12"] == pytest.approx(0.15, abs=0.05)
    # 심지 않은 특성들은 0 근처
    assert abs(b["consec"]) < 0.15
    z = model.summary()["z"]
    assert abs(z["low12"]) > 3       # 유의
    assert abs(z["consec"]) < 3      # 비유의


def test_fit_requires_enough_data(df):
    with pytest.raises(ValueError, match="부족"):
        popularity.fit(df.head(10))


def test_expected_winners_scale(df):
    """평균 조합의 예상 당첨자 수가 데이터 평균과 같은 자릿수여야 한다."""
    model = popularity.fit(df)
    lam = model.expected_winners([5, 13, 21, 28, 34, 41], sales=5e10)
    assert 1 < lam < 100


def test_number_etas_decompose(df):
    """번호별 기여 합 + consec 항 = 전체 η (절편 제외)."""
    model = popularity.fit(df)
    combo = [3, 11, 25, 31, 32, 44]
    etas = model.number_etas()
    number_part = sum(etas.loc[n] for n in combo)
    consec_part = model.coef[popularity.FEATURE_NAMES.index("consec")]  # 31-32 연속
    intercept = model.coef[0]
    assert number_part + consec_part + intercept == pytest.approx(model.eta(combo))


def test_expected_share():
    assert popularity.expected_share(0) == 1.0
    # 몬테카를로 대조
    rng = np.random.default_rng(0)
    lam = 5.0
    mc = np.mean(1.0 / (1.0 + rng.poisson(lam, size=200_000)))
    assert popularity.expected_share(lam) == pytest.approx(mc, rel=0.01)


def test_unpopular_strategy_available():
    assert "unpopular" in predictor.available_strategies()


def test_unpopular_predict_valid(df):
    picks = predictor.predict(df, strategy="unpopular", games=5, seed=1)
    assert len(picks) == 5
    for combo in picks:
        assert len(set(combo)) == 6
        assert all(1 <= n <= 45 for n in combo)


def test_unpopular_avoids_popular_numbers(df):
    """low12가 인기 특성인 이력에서, unpopular는 1~12를 균등보다 덜 뽑아야 한다.

    샘플링은 인기도 역수 비례(∝ exp(-η))라 완전 회피가 아니라 기울임이다.
    심어둔 효과 +0.15면 1~12 번호의 상대 가중치는 exp(-0.15) ≈ 0.86배,
    조합당 low12 개수 기댓값은 약 1.6 → 1.43으로 줄어드는 게 이론값이다.
    """
    unpop = predictor.predict(df, strategy="unpopular", games=200, seed=2, use_filter=False)
    uni = predictor.predict(df, strategy="uniform", games=200, seed=2, use_filter=False)

    def low12_rate(picks):
        return np.mean([sum(1 for n in c if n <= 12) for c in picks])

    assert low12_rate(unpop) < low12_rate(uni) * 0.93


def test_prize_comparison_orders_unpopular_first(df):
    table = popularity.prize_comparison(df, ["uniform", "unpopular"], games=100, seed=0)
    assert list(table.columns) == ["전략", "예상_동시당첨자", "기대상금_당첨시_억원"]
    # 비인기 전략이 동시 당첨자는 적고 기대 상금은 커야 한다
    row = table.set_index("전략")
    assert row.loc["unpopular", "예상_동시당첨자"] < row.loc["uniform", "예상_동시당첨자"]
    assert table.iloc[0]["전략"] == "unpopular"
