"""LSTM 전략 테스트. torch 미설치 환경에서는 전체 건너뜀."""

import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")

from lotto import lstm, predictor  # noqa: E402  (importorskip 이후여야 함)


@pytest.fixture
def df():
    rng = np.random.default_rng(0)
    rows = []
    for i in range(1, 121):
        nums = sorted(rng.choice(np.arange(1, 46), size=7, replace=False).tolist())
        rows.append({
            "draw_no": i,
            "draw_date": "2020-01-01",
            **{f"n{j+1}": nums[j] for j in range(6)},
            "bonus": nums[6],
        })
    return pd.DataFrame(rows)


@pytest.fixture(autouse=True)
def fresh_cache():
    lstm.clear_cache()
    yield
    lstm.clear_cache()


def test_multi_hot():
    hot = lstm._multi_hot(np.array([[1, 2, 3, 4, 5, 45]]))
    assert hot.shape == (1, 45)
    assert hot.sum() == 6
    assert hot[0, 0] == 1.0 and hot[0, 44] == 1.0 and hot[0, 5] == 0.0


def test_build_dataset_shapes(df):
    X, y = lstm.build_dataset(df, seq_len=10)
    assert X.shape == (110, 10, 45)
    assert y.shape == (110, 45)
    assert float(y.sum(dim=1).unique()) == 6.0  # 모든 타깃은 번호 6개


def test_build_dataset_insufficient_data(df):
    with pytest.raises(ValueError, match="부족"):
        lstm.build_dataset(df.head(10), seq_len=32)


def test_train_and_predict_probs(df, monkeypatch):
    monkeypatch.setattr(lstm, "SEQ_LEN", 10)
    model = lstm.train(df, epochs=2, seq_len=10)
    probs = lstm.predict_probs(model, df, seq_len=10)
    assert len(probs) == 45
    assert list(probs.index) == list(range(1, 46))
    assert ((probs > 0) & (probs < 1)).all()


def test_number_weights_caches(df, monkeypatch):
    monkeypatch.setattr(lstm, "SEQ_LEN", 10)
    calls = []
    original_train = lstm.train

    def counting_train(*args, **kwargs):
        calls.append(1)
        return original_train(*args, **kwargs)

    monkeypatch.setattr(lstm, "train", counting_train)

    lstm.number_weights(df, epochs=2)
    lstm.number_weights(df, epochs=2)          # 같은 시점 -> 재학습 없음
    assert len(calls) == 1

    lstm.number_weights(df.head(60), epochs=2)  # 과거 시점 -> 누출 방지 재학습
    assert len(calls) == 2

    lstm.number_weights(df.head(100), epochs=2)  # 60 + RETRAIN_EVERY(50) 이내 -> 재사용
    assert len(calls) == 2


def test_predictor_integration(df, monkeypatch):
    monkeypatch.setattr(lstm, "SEQ_LEN", 10)
    monkeypatch.setattr(lstm, "EPOCHS", 2)
    assert "lstm" in predictor.available_strategies()
    picks = predictor.predict(df, strategy="lstm", games=3, seed=5)
    assert len(picks) == 3
    for combo in picks:
        assert len(set(combo)) == 6
        assert all(1 <= n <= 45 for n in combo)


def test_training_is_reproducible(df, monkeypatch):
    monkeypatch.setattr(lstm, "SEQ_LEN", 10)
    m1 = lstm.train(df, epochs=2, seq_len=10)
    m2 = lstm.train(df, epochs=2, seq_len=10)
    p1 = lstm.predict_probs(m1, df, seq_len=10)
    p2 = lstm.predict_probs(m2, df, seq_len=10)
    assert np.allclose(p1.to_numpy(), p2.to_numpy())
