"""분석/예측 로직 테스트 (합성 데이터 사용)."""

import numpy as np
import pandas as pd
import pytest

from lotto import analyzer, backtest, predictor


@pytest.fixture
def df():
    """무작위 회차 300개로 만든 합성 이력.

    주의: 7개를 정렬한 뒤 마지막을 보너스로 쓰면 보너스가 항상 최댓값이 되어
    당첨번호 6개가 작은 쪽으로 치우친다. 정렬 전에 분리해야 균등하다.
    """
    rng = np.random.default_rng(0)
    rows = []
    for i in range(1, 301):
        picks = rng.choice(np.arange(1, 46), size=7, replace=False)
        nums = sorted(picks[:6].tolist())
        rows.append({
            "draw_no": i,
            "draw_date": f"2020-01-{(i % 28) + 1:02d}",
            **{f"n{j+1}": nums[j] for j in range(6)},
            "bonus": int(picks[6]),
            # unpopular 전략(인기도 회귀)이 쓰는 컬럼
            "first_prize_winners": int(rng.poisson(10)),
            "first_prize_amount": 2_000_000_000,
            "total_sales": 5e10,
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


# randomforest는 1,035개 모델을 실제로 학습해 회차당 2분 이상 걸린다.
# 전용 테스트(test_lotto_anal.py)가 확률을 모의해 따로 검증한다.
SLOW_IN_TESTS = {"randomforest"}


@pytest.mark.parametrize(
    "strategy",
    [s for s in predictor.available_strategies() if s not in SLOW_IN_TESTS],
)
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


def test_combo_frequency_totals(df):
    pairs = analyzer.combo_frequency(df, r=2)
    triples = analyzer.combo_frequency(df, r=3)
    assert len(pairs) == 990        # C(45,2)
    assert len(triples) == 14190    # C(45,3)
    assert pairs.sum() == len(df) * 15   # 회차당 C(6,2)
    assert triples.sum() == len(df) * 20  # 회차당 C(6,3)


def test_combo_frequency_fixed(fixed_df):
    pairs = analyzer.combo_frequency(fixed_df, r=2)
    assert pairs.loc[(1, 2)] == len(fixed_df)   # 매 회차 함께 출현
    assert pairs.loc[(1, 45)] == 0
    triples = analyzer.combo_frequency(fixed_df, r=3)
    assert triples.loc[(1, 2, 3)] == len(fixed_df)


def test_combo_uniformity_random_data(df):
    """무작위 합성 데이터는 균등 가설과 부합해야 한다 (|z| 작음, 분산/평균 ≈ 1)."""
    stats = analyzer.combo_uniformity(analyzer.combo_frequency(df, r=2))
    assert abs(stats["z_score"]) < 4
    assert 0.7 < stats["dispersion"] < 1.3


def test_combo_uniformity_degenerate_data(fixed_df):
    """같은 조합만 반복되는 데이터는 균등 가설에서 크게 벗어나야 한다."""
    stats = analyzer.combo_uniformity(analyzer.combo_frequency(fixed_df, r=2))
    assert stats["z_score"] > 10


def test_poisson_table(df):
    counts = analyzer.combo_frequency(df, r=3)
    table = analyzer.poisson_table(counts)
    assert table["관측_조합수"].sum() == 14190
    # 포아송 기대의 합도 전체 조합 수와 비슷해야 한다
    assert table["포아송_기대"].sum() == pytest.approx(14190, rel=0.05)


def test_top_combos(df):
    top = analyzer.top_combos(df, r=2, k=5)
    assert len(top) == 5
    assert (top["출현"].diff().dropna() <= 0).all()  # 내림차순


def test_pairwise_strategy_available():
    assert "pairwise" in predictor.available_strategies()


def test_pairwise_predict_valid(df):
    picks = predictor.predict(df, strategy="pairwise", games=5, seed=1)
    assert len(picks) == 5
    for combo in picks:
        assert len(set(combo)) == 6
        assert all(1 <= n <= 45 for n in combo)
        assert combo == sorted(combo)


def test_pairwise_reproducible(df):
    assert (predictor.predict(df, strategy="pairwise", seed=3)
            == predictor.predict(df, strategy="pairwise", seed=3))


def test_pairwise_follows_pair_structure():
    """1~6과 7~12가 각각 항상 함께 나온 이력이라면, 표본도 같은 무리에서 나와야 한다."""
    rows = []
    for i in range(1, 201):
        nums = [1, 2, 3, 4, 5, 6] if i % 2 == 0 else [7, 8, 9, 10, 11, 12]
        rows.append({
            "draw_no": i, "draw_date": "2020-01-01",
            **{f"n{j+1}": n for j, n in enumerate(nums)}, "bonus": 45,
        })
    history = pd.DataFrame(rows)

    # predict()는 중복 조합을 제거하는데 순수 조합은 두 가지뿐이므로,
    # 중복을 허용하는 샘플러를 직접 사용해 분포를 본다.
    sample = predictor.build_sampler(history, "pairwise", use_filter=False)
    rng = np.random.default_rng(0)
    picks = [sample(rng) for _ in range(50)]

    group_a, group_b = set(range(1, 7)), set(range(7, 13))
    pure = sum(1 for c in picks if set(c) == group_a or set(c) == group_b)
    # 첫 번호가 무리 밖(약 19%)이거나 평활(α=1) 탓에 혼합이 일부 생긴다
    assert pure >= 30


def test_clt_theoretical_constants():
    """이론값 검증: 합계 평균 138, 분산 = 6σ²(N-n)/(N-1) ≈ 897."""
    assert predictor.CLT_SUM_MEAN == 138.0
    assert predictor.CLT_SUM_VAR == pytest.approx(6 * (45**2 - 1) / 12 * 39 / 44)
    assert predictor.CLT_SUM_STD == pytest.approx(29.95, abs=0.01)


def test_clt_scorer_peaks_at_mean(df):
    score = predictor.clt_combo_scorer(df)
    center = [18, 20, 22, 24, 26, 28]      # 합계 138
    extreme = [1, 2, 3, 4, 5, 6]           # 합계 21
    assert score(center) == pytest.approx(1.0)
    assert score(extreme) < 0.001
    assert 0.0 < score([1, 5, 10, 20, 30, 40]) < 1.0


def test_clt_sums_concentrate_near_mean(df):
    """CLT 전략의 합계 분포가 균등 추출보다 138 주변에 집중되는지 확인한다.

    필터를 꺼서 순수한 기각 샘플링 효과만 본다.
    """
    clt_sums = [sum(c) for c in predictor.predict(
        df, strategy="clt", games=300, seed=11, use_filter=False)]
    uni_sums = [sum(c) for c in predictor.predict(
        df, strategy="uniform", games=300, seed=11, use_filter=False)]

    assert np.mean(np.abs(np.array(clt_sums) - 138)) < np.mean(np.abs(np.array(uni_sums) - 138))
    assert np.std(clt_sums) < np.std(uni_sums)
    # 채택 확률이 정규 밀도이므로 표본 표준편차는 이론값(≈30)보다 크게 작아야 한다
    assert np.std(clt_sums) < predictor.CLT_SUM_STD


def test_draw_combination_scorer_rejects(df):
    """조합 점수 0이면 절대 채택되지 않고 max_attempts 후 마지막 조합을 돌려준다."""
    rng = np.random.default_rng(0)
    weights = pd.Series(1.0, index=range(1, 46))
    combo = predictor.draw_combination(
        weights, rng, combo_scorer=lambda c: 0.0, max_attempts=50)
    assert len(combo) == 6  # 무한 루프 없이 반환


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


# --------------------------------------------------------------- 추첨일 / 등수

def test_next_draw_date_adds_week(df):
    data = df.copy()
    data.loc[data.index[-1], "draw_date"] = "2026-07-18"  # 토요일
    assert analyzer.next_draw_date(data) == "2026-07-25 (토)"


def test_next_draw_date_weekday_label():
    data = pd.DataFrame([{
        "draw_no": 1, "draw_date": "2026-01-01",  # 목요일
        "n1": 1, "n2": 2, "n3": 3, "n4": 4, "n5": 5, "n6": 6, "bonus": 7,
    }])
    assert analyzer.next_draw_date(data) == "2026-01-08 (목)"


@pytest.mark.parametrize("combo,expected", [
    ([1, 2, 3, 4, 5, 6], "1등"),           # 6개
    ([1, 2, 3, 4, 5, 7], "2등"),           # 5개 + 보너스(7)
    ([1, 2, 3, 4, 5, 9], "3등"),           # 5개, 보너스 없음
    ([1, 2, 3, 4, 9, 10], "4등"),          # 4개
    ([1, 2, 3, 9, 10, 11], "5등"),         # 3개
    ([1, 2, 9, 10, 11, 12], None),         # 2개 — 미당첨
    ([9, 10, 11, 12, 13, 14], None),       # 0개
])
def test_rank_of(combo, expected):
    winning = {1, 2, 3, 4, 5, 6}
    assert backtest.rank_of(combo, winning, bonus=7) == expected


def test_rank_of_bonus_only_matters_at_five():
    """보너스는 5개 맞췄을 때만 등수를 바꾼다."""
    winning = {1, 2, 3, 4, 5, 6}
    assert backtest.rank_of([1, 2, 3, 4, 7, 8], winning, bonus=7) == "4등"


def test_rank_history_counts(df):
    result = backtest.rank_history(df, strategy="uniform", games_per_draw=2,
                                   min_history=250, seed=1)
    assert result.draws_tested == 50
    assert result.total_games == 100
    assert set(result.counts) == set(backtest.RANK_NAMES)
    assert all(v >= 0 for v in result.counts.values())
    # 당첨 횟수 합은 전체 게임 수를 넘을 수 없다
    assert sum(result.counts.values()) <= result.total_games


def test_rank_history_summary_line(df):
    result = backtest.rank_history(df, strategy="uniform", games_per_draw=1,
                                   min_history=280, seed=1)
    line = result.summary_line()
    assert line.startswith("1등 ")
    assert "5등 " in line
    assert line.count("·") == 4


def test_rank_history_requires_enough_data(df):
    with pytest.raises(ValueError, match="부족"):
        backtest.rank_history(df.head(50), min_history=100)


def test_rank_history_is_reproducible(df):
    kwargs = dict(strategy="uniform", games_per_draw=2, min_history=250, seed=9)
    assert backtest.rank_history(df, **kwargs).counts == backtest.rank_history(df, **kwargs).counts


# --------------------------------------------------------------- oracle (부정 데모)

@pytest.fixture
def oracle_source(df, monkeypatch):
    """오라클이 몰래 읽는 '전체 데이터'를 합성 이력으로 교체한다."""
    monkeypatch.setattr(predictor, "ORACLE_SOURCE", lambda: df)
    return df


def test_oracle_available():
    assert "oracle" in predictor.available_strategies()


def test_oracle_weights_peak_at_next_winning_numbers(oracle_source, df):
    """history가 i회차까지면 (i+1)회차의 실제 당첨번호에 가중치가 몰린다."""
    history = df.iloc[:200]
    next_row = df.iloc[200]  # 201회차
    winning = set(next_row[[f"n{i}" for i in range(1, 7)]].astype(int))

    weights = predictor.oracle_scores(history)
    top6 = set(weights.sort_values(ascending=False).head(6).index)
    assert top6 == winning


def test_oracle_hits_first_and_second_prize_in_backtest(oracle_source, df):
    """미래를 보면 시뮬레이션에서 1등·2등이 나온다 — 그게 유일한 방법이다."""
    result = backtest.rank_history(df, strategy="oracle", games_per_draw=5,
                                   min_history=250, seed=0)
    assert result.counts["1등"] >= 1
    assert result.counts["2등"] >= 1
    # 대부분의 게임이 1등이어야 한다 (누출의 규모 확인)
    assert result.counts["1등"] > result.total_games * 0.5


def test_oracle_degrades_to_uniform_for_real_future(oracle_source, df, caplog):
    """전체 데이터의 마지막 회차까지 주면 볼 미래가 없어 균등이 된다."""
    with caplog.at_level("WARNING"):
        weights = predictor.oracle_scores(df)  # 다음 회차는 아직 없음
    assert weights.nunique() == 1              # 모든 번호 동일 가중치
    assert "실전 성능" in caplog.text


def test_honest_strategies_cannot_hit_first_prize(df):
    """정직한 전략(누출 없음)은 같은 조건에서 1등이 나오지 않는다.

    P(1등)=1/8,145,060이므로 250게임에서 1등이 나올 확률은 사실상 0이다.
    """
    for strategy in ("uniform", "unpopular"):
        result = backtest.rank_history(df, strategy=strategy, games_per_draw=5,
                                       min_history=250, seed=0)
        assert result.counts["1등"] == 0
