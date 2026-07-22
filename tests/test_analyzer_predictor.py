"""분석/예측 로직 테스트 (합성 데이터 사용)."""

import numpy as np
import pandas as pd
import pytest

from lotto import analyzer, backtest, predictor


@pytest.fixture
def df():
    """무작위 회차 300개로 만든 합성 이력."""
    rng = np.random.default_rng(0)
    rows = []
    for i in range(1, 301):
        nums = sorted(rng.choice(np.arange(1, 46), size=7, replace=False).tolist())
        rows.append({
            "draw_no": i,
            "draw_date": f"2020-01-{(i % 28) + 1:02d}",
            **{f"n{j+1}": nums[j] for j in range(6)},
            "bonus": nums[6],
        })
    return pd.DataFrame(rows)


@pytest.fixture
def fixed_df():
    """1~6이 반복되는 결정적 이력. 지표 계산을 정확히 검증하기 위한 것."""
    rows = [
        {"draw_no": i, "draw_date": "2020-01-01",
         "n1": 1, "n2": 2, "n3": 3, "n4": 4, "n5": 5, "n6": 6, "bonus": 7}
        for i in range(1, 11)
    ]
    return pd.DataFrame(rows)


def test_frequency_totals(df):
    freq = analyzer.frequency(df)
    assert len(freq) == 45
    assert freq.sum() == len(df) * 6


def test_frequency_fixed(fixed_df):
    freq = analyzer.frequency(fixed_df)
    assert freq.loc[1] == 10
    assert freq.loc[45] == 0


def test_gaps_fixed(fixed_df):
    gap = analyzer.gaps(fixed_df)
    assert gap.loc[1] == 0          # 마지막 회차에 출현
    assert gap.loc[45] == len(fixed_df)  # 한 번도 미출현


def test_weighted_frequency_favors_recent():
    """같은 횟수라면 최근에 나온 번호가 더 높은 가중치를 받는다.

    100회차 중 앞 5회차에만 13이, 뒤 5회차에만 42가 나오도록 통제한다.
    """
    rows = []
    for i in range(1, 101):
        nums = [20, 21, 22, 23, 24]
        if i <= 5:
            nums.append(13)
        elif i > 95:
            nums.append(42)
        else:
            nums.append(30)
        rows.append({
            "draw_no": i, "draw_date": "2020-01-01",
            **{f"n{j+1}": n for j, n in enumerate(sorted(nums))}, "bonus": 45,
        })
    data = pd.DataFrame(rows)

    w = analyzer.weighted_frequency(data, half_life=50)
    assert w.loc[42] > w.loc[13]           # 출현 횟수는 같지만 최근이 우세
    assert w.loc[45] == 0                  # 보너스 번호는 집계 대상이 아님


def test_pair_matrix_symmetric(df):
    pairs = analyzer.pair_matrix(df)
    assert pairs.shape == (45, 45)
    assert (pairs.to_numpy() == pairs.to_numpy().T).all()
    assert np.trace(pairs.to_numpy()) == 0


def test_summary_keys(df):
    s = analyzer.summary(df)
    assert s["총_회차"] == 300
    assert "합계_통계" in s and "구간별_비율" in s


@pytest.mark.parametrize("strategy", predictor.available_strategies())
def test_predict_shape(df, strategy):
    picks = predictor.predict(df, strategy=strategy, games=5, seed=1)
    assert len(picks) == 5
    for combo in picks:
        assert len(combo) == 6
        assert len(set(combo)) == 6
        assert all(1 <= n <= 45 for n in combo)
        assert combo == sorted(combo)


def test_predict_no_duplicate_games(df):
    picks = predictor.predict(df, games=20, seed=7)
    assert len({tuple(c) for c in picks}) == 20


def test_predict_is_reproducible(df):
    assert predictor.predict(df, seed=99) == predictor.predict(df, seed=99)


def test_predict_unknown_strategy(df):
    with pytest.raises(ValueError, match="알 수 없는 전략"):
        predictor.predict(df, strategy="없는전략")


def test_predict_empty_df():
    with pytest.raises(ValueError, match="데이터가 없습니다"):
        predictor.predict(pd.DataFrame())


def test_filter_rejects_consecutive_run():
    f = predictor.CombinationFilter(sum_min=0, sum_max=300, max_consecutive=3)
    assert not f.accepts([1, 2, 3, 4, 20, 30])
    assert f.accepts([1, 2, 3, 15, 25, 35])


def test_filter_rejects_all_odd():
    f = predictor.CombinationFilter(sum_min=0, sum_max=300)
    assert not f.accepts([1, 3, 5, 7, 9, 11])


def test_filter_rejects_sum_outside_range():
    f = predictor.CombinationFilter(sum_min=100, sum_max=180)
    assert not f.accepts([1, 2, 3, 4, 5, 7])


def test_filtered_predictions_pass_filter(df):
    combo_filter = predictor.CombinationFilter.from_history(df)
    for combo in predictor.predict(df, games=10, seed=3):
        assert combo_filter.accepts(combo)


def test_backtest_runs(df):
    result = backtest.run(df, strategy="hot", test_draws=20, games_per_draw=3, min_history=100)
    assert result.draws_tested == 20
    assert sum(result.match_counts.values()) == 60
    assert 0 <= result.mean_matches <= 6


def test_backtest_requires_enough_history(df):
    with pytest.raises(ValueError, match="데이터가 부족"):
        backtest.run(df, test_draws=200, min_history=200)


def test_backtest_compare(df):
    table = backtest.compare(
        df, strategies=["uniform", "hot"], test_draws=10, games_per_draw=2, min_history=100
    )
    assert list(table["전략"].sort_values()) == ["hot", "uniform"]
