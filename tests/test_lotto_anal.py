"""lotto-anal(2023) 로직 재현 검증 테스트.

RandomForest 1,035개 학습은 회차당 2분대라 테스트에서는 소수 열만 쓴다.
"""

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("sklearn")

from lotto import lotto_anal as la  # noqa: E402


@pytest.fixture
def df():
    rng = np.random.default_rng(0)
    rows = []
    for i in range(1, 121):
        picks = rng.choice(np.arange(1, 46), size=7, replace=False)
        nums = sorted(picks[:6].tolist())
        rows.append({
            "draw_no": i, "draw_date": "2020-01-01",
            **{f"n{j+1}": nums[j] for j in range(6)}, "bonus": int(picks[6]),
        })
    return pd.DataFrame(rows)


def test_pair_count():
    assert len(la.PAIRS) == 990  # C(45,2)


def test_indicator_matrix_shape_and_counts(df):
    ind = la.build_indicator_matrix(df)
    assert ind.shape == (120, 1035)
    assert (ind[:, :45].sum(axis=1) == 6).all()    # 회차당 번호 6개
    assert (ind[:, 45:].sum(axis=1) == 15).all()   # 회차당 2개조합 C(6,2)=15


def test_indicator_matrix_marks_right_columns():
    one = pd.DataFrame([{
        "draw_no": 1, "draw_date": "2020-01-01",
        "n1": 1, "n2": 2, "n3": 3, "n4": 4, "n5": 5, "n6": 6, "bonus": 7,
    }])
    ind = la.build_indicator_matrix(one)
    assert ind[0, :45].nonzero()[0].tolist() == [0, 1, 2, 3, 4, 5]
    assert ind[0, 45 + la.PAIRS.index((1, 2))] == 1
    assert ind[0, 45 + la.PAIRS.index((1, 45))] == 0


def test_build_dataset_alignment(df):
    ind = la.build_indicator_matrix(df)
    x, y = la.build_dataset(ind, past=10)
    assert x.shape == (110, 10 * 1035)
    assert y.shape == (110, 1035)
    # x의 첫 행은 0~9회차, y의 첫 행은 10회차여야 한다
    assert np.array_equal(x[0], ind[0:10].ravel())
    assert np.array_equal(y[0], ind[10])


def test_build_dataset_requires_enough_rows(df):
    with pytest.raises(ValueError, match="부족"):
        la.build_dataset(la.build_indicator_matrix(df.head(5)), past=10)


def test_follow_the_pairs_shape():
    rng = np.random.default_rng(0)
    probs = rng.random(1035)
    sets = la.follow_the_pairs(probs, n_sets=5)
    assert len(sets) == 5
    for combo in sets:
        assert len(combo) == 6
        assert len(set(combo)) == 6           # 세트 내 중복 없음
        assert combo == sorted(combo)
        assert all(1 <= n <= 45 for n in combo)


def test_follow_the_pairs_starts_from_top_singles():
    """단일번호 확률 상위 5개가 각 세트의 시작점이 된다."""
    probs = np.zeros(1035)
    probs[:45] = 0.01
    for n in (10, 20, 30, 40, 44):        # 0-based -> 번호 11,21,31,41,45
        probs[n] = 0.9
    sets = la.follow_the_pairs(probs, n_sets=5)
    starts = {11, 21, 31, 41, 45}
    assert all(starts & set(c) for c in sets)


def test_follow_the_pairs_chains_strongest_partner():
    """가장 확률 높은 2개조합의 상대 번호를 이어 붙이는지 확인."""
    probs = np.zeros(1035)
    probs[0] = 1.0                                    # 번호 1이 최고 확률
    probs[45 + la.PAIRS.index((1, 42))] = 1.0         # 1과 가장 강한 짝은 42
    probs[45 + la.PAIRS.index((42, 43))] = 0.9        # 42의 다음 짝은 43
    combo = la.follow_the_pairs(probs, n_sets=1)[0]
    assert {1, 42, 43} <= set(combo)


def test_predict_probabilities_subset(df):
    """소수 열만 학습해도 확률이 0~1 범위로 나온다."""
    probs = la.predict_probabilities(df, train_draws=50, targets=[0, 1, 2])
    assert probs.shape == (1035,)
    assert ((probs >= 0) & (probs <= 1)).all()
    assert probs[3:].sum() == 0  # 학습하지 않은 열은 0
