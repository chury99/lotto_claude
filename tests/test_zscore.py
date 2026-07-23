"""정규분포 이탈도(zscore) 전략 테스트."""

import math
from math import comb

import numpy as np
import pandas as pd
import pytest

from lotto import predictor, zscore


@pytest.fixture
def df():
    rng = np.random.default_rng(0)
    rows = []
    for i in range(1, 301):
        picks = rng.choice(np.arange(1, 46), size=7, replace=False)
        nums = sorted(picks[:6].tolist())
        rows.append({
            "draw_no": i, "draw_date": "2020-01-01",
            **{f"n{j+1}": nums[j] for j in range(6)}, "bonus": int(picks[6]),
            "first_prize_winners": int(rng.poisson(10)),
            "first_prize_amount": 2_000_000_000, "total_sales": 5e10,
        })
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ 기초 계산

def test_membership_matrix(df):
    m = zscore.membership_matrix(df)
    assert m.shape == (300, 45)
    assert (m.sum(axis=1) == 6).all()


@pytest.mark.parametrize("k", [1, 2, 3, 4, 5, 6])
def test_combo_probability(k):
    assert zscore.combo_probability(k) == comb(6, k) / comb(45, k)


def test_combo_probability_range():
    with pytest.raises(ValueError):
        zscore.combo_probability(7)


def test_combo_probability_values():
    assert zscore.combo_probability(1) == pytest.approx(6 / 45)
    assert zscore.combo_probability(6) == pytest.approx(1 / 8_145_060)


def test_deviation_scores_formula():
    p = zscore.combo_probability(1)
    n = 1000
    mean, sd = n * p, math.sqrt(n * p * (1 - p))
    z = zscore.deviation_scores(np.array([mean, mean + sd, mean - sd]), n, k=1)
    assert z == pytest.approx([0.0, 1.0, -1.0])


def test_deviation_scores_sign():
    """기대보다 적게 나오면 음수, 많이 나오면 양수."""
    z = zscore.deviation_scores(np.array([0.0, 1000.0]), 1000, k=1)
    assert z[0] < 0 < z[1]


# ------------------------------------------------------------------ 선정 로직

def test_select_combo_shape(df):
    combo = zscore.select_combo(df)
    assert len(combo) == 6
    assert len(set(combo)) == 6
    assert combo == sorted(combo)
    assert all(1 <= n <= 45 for n in combo)


def test_select_combo_is_deterministic(df):
    assert zscore.select_combo(df) == zscore.select_combo(df)


def test_first_pick_is_least_frequent_number():
    """1단계는 가장 덜 나온 번호여야 한다 (하향 이탈)."""
    rows = []
    for i in range(1, 201):
        # 1~6만 계속 나오고 45는 한 번도 안 나옴
        nums = [1, 2, 3, 4, 5, 6] if i % 2 else [1, 2, 3, 4, 5, 7]
        rows.append({
            "draw_no": i, "draw_date": "2020-01-01",
            **{f"n{j+1}": n for j, n in enumerate(nums)}, "bonus": 44,
        })
    data = pd.DataFrame(rows)
    trace = zscore.zscore_trace(data)
    assert trace.iloc[0]["관측"] == 0          # 한 번도 안 나온 번호를 골랐다
    assert trace.iloc[0]["z"] < 0              # 하향 이탈


def test_never_picks_most_frequent_first():
    """가장 많이 나온 번호는 1단계에서 절대 뽑히지 않는다."""
    rows = []
    rng = np.random.default_rng(1)
    for i in range(1, 201):
        nums = [1] + sorted(rng.choice(np.arange(2, 46), size=5, replace=False).tolist())
        rows.append({
            "draw_no": i, "draw_date": "2020-01-01",
            **{f"n{j+1}": n for j, n in enumerate(nums)}, "bonus": 45,
        })
    data = pd.DataFrame(rows)   # 번호 1이 매 회차 등장
    assert zscore.zscore_trace(data).iloc[0]["선택"] != 1


def test_excludes_past_winning_combos():
    """완성 조합이 과거 당첨 조합과 같으면 그 후보를 제외한다."""
    rows = []
    for i in range(1, 121):
        nums = [1, 2, 3, 4, 5, 6]
        rows.append({
            "draw_no": i, "draw_date": "2020-01-01",
            **{f"n{j+1}": n for j, n in enumerate(nums)}, "bonus": 7,
        })
    data = pd.DataFrame(rows)
    past = zscore._winning_combos(data)
    assert (1, 2, 3, 4, 5, 6) in past
    # 어떤 시작점에서 출발해도 과거 조합이 그대로 나오면 안 된다
    for combo in zscore.select_sets(data, n_sets=5):
        assert tuple(combo) not in past


def test_tie_break_prefers_unpopular(df):
    """동률 단계에서는 인기도가 낮은(덜 고르는) 번호가 선택된다."""
    trace = zscore.zscore_trace(df)
    tie_rows = trace[trace["동률후보수"] > 1]
    assert len(tie_rows) > 0                      # 뒷 단계는 동률이 생긴다

    from lotto import popularity
    etas = popularity.fit(df).number_etas()
    for _, row in tie_rows.iterrows():
        # 선택된 번호는 '동률인 후보들' 중 인기도가 가장 낮아야 한다
        assert etas.loc[row["선택"]] == pytest.approx(
            min(etas.loc[n] for n in row["동률후보"]))


def test_trace_tied_candidates_include_pick(df):
    trace = zscore.zscore_trace(df)
    for _, row in trace.iterrows():
        assert row["선택"] in row["동률후보"]


def test_trace_columns(df):
    trace = zscore.zscore_trace(df)
    assert list(trace["단계"]) == [1, 2, 3, 4, 5, 6]
    assert set(trace["결정요인"]) <= {"이탈도", "인기도(동률)"}
    assert (trace["z"] <= 0).all()                # 항상 하향 이탈


def test_select_sets_distinct(df):
    sets = zscore.select_sets(df, n_sets=5)
    assert len(sets) == 5
    assert len({tuple(s) for s in sets}) == 5


# ------------------------------------------------------------------ 전략 등록

def test_registered():
    assert "zscore" in predictor.available_strategies()


def test_predict(df):
    picks = predictor.predict(df, strategy="zscore", games=5, seed=1)
    assert len(picks) == 5
    for combo in picks:
        assert len(set(combo)) == 6
        assert combo == sorted(combo)


def test_predict_matches_select_sets(df):
    """필터를 끄면 select_sets 결과와 정확히 같아야 한다."""
    assert (predictor.predict(df, strategy="zscore", games=5, seed=1, use_filter=False)
            == zscore.select_sets(df, n_sets=5))


def test_predict_beyond_pool(df):
    """풀(45개)보다 많이 요청해도 끝난다."""
    picks = predictor.predict(df, strategy="zscore", games=50, seed=2)
    assert len({tuple(c) for c in picks}) == 50


def test_scores_favor_least_frequent(df):
    """가중치는 덜 나온 번호에 높게 매겨진다."""
    w = predictor.zscore_scores(df)
    counts = zscore.membership_matrix(df).sum(axis=0)
    assert w.loc[int(np.argmin(counts)) + 1] > w.loc[int(np.argmax(counts)) + 1]
