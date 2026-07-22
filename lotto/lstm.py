"""LSTM 시계열 전략.

과거 SEQ_LEN회차의 당첨번호(각 회차를 45차원 멀티-핫 벡터로 인코딩)를 입력으로,
다음 회차에 각 번호가 포함될 확률 45개를 출력하도록 LSTM을 학습한다. 학습 목표는
번호별 이진 분류(BCE)이고, 출력 확률을 predictor의 번호별 가중치로 그대로 쓴다.

백테스트에서는 회차가 하나 넘어갈 때마다 재학습하면 지나치게 느리므로, 학습 시점을
기억해 두고 RETRAIN_EVERY회차마다만 재학습한다(그 사이에는 학습된 모델에 최신
윈도우만 넣어 추론). 캐시된 모델이 현재 이력보다 미래의 데이터로 학습된 것이면
미래 정보 누출이므로 무조건 다시 학습한다.

이 모듈은 선택 의존성인 torch가 필요하다:  pip install -r requirements-lstm.txt
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

try:
    import torch
    from torch import nn
except ImportError as exc:  # pragma: no cover - torch 미설치 환경 안내
    raise ImportError(
        "lstm 전략에는 torch가 필요합니다: pip install -r requirements-lstm.txt"
    ) from exc

from .analyzer import NUMBER_COLUMNS, numbers_matrix

log = logging.getLogger(__name__)

SEQ_LEN = 32        # 입력으로 쓰는 과거 회차 수
HIDDEN = 96         # LSTM 은닉 차원
LAYERS = 2
DROPOUT = 0.2
EPOCHS = 40
BATCH_SIZE = 64
LR = 1e-3
VAL_RATIO = 0.1     # 시간순 뒤쪽 10%를 검증용으로 떼어 조기 종료에 사용
PATIENCE = 6
SEED = 0
RETRAIN_EVERY = 50  # 마지막 학습 후 이 회차 수만큼 지나면 재학습

DEVICE = torch.device("cpu")  # 모델이 작아 CPU로 충분하고 결과 재현이 쉽다


class DrawLSTM(nn.Module):
    def __init__(self, hidden: int = HIDDEN, layers: int = LAYERS) -> None:
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=45, hidden_size=hidden, num_layers=layers,
            batch_first=True, dropout=DROPOUT if layers > 1 else 0.0,
        )
        self.head = nn.Sequential(nn.Dropout(DROPOUT), nn.Linear(hidden, 45))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)          # (B, T, H)
        return self.head(out[:, -1])   # 마지막 시점만 사용 -> (B, 45) 로짓


def _multi_hot(matrix: np.ndarray) -> np.ndarray:
    """(T, 6) 번호 행렬 -> (T, 45) 멀티-핫 float32."""
    hot = np.zeros((len(matrix), 45), dtype=np.float32)
    for i, row in enumerate(matrix):
        hot[i, row - 1] = 1.0
    return hot


def build_dataset(df: pd.DataFrame, seq_len: int = SEQ_LEN) -> tuple[torch.Tensor, torch.Tensor]:
    """(X, y) 텐서. X[i]=연속 seq_len회차, y[i]=그 다음 회차의 멀티-핫."""
    hot = _multi_hot(numbers_matrix(df))
    if len(hot) <= seq_len:
        raise ValueError(f"데이터가 부족합니다. 최소 {seq_len + 1}회차가 필요합니다 (현재 {len(hot)}회차).")
    X = np.stack([hot[i:i + seq_len] for i in range(len(hot) - seq_len)])
    y = hot[seq_len:]
    return torch.from_numpy(X), torch.from_numpy(y)


def train(df: pd.DataFrame, epochs: int | None = None, seq_len: int = SEQ_LEN) -> DrawLSTM:
    """이력 전체로 모델을 학습한다. 시간순 뒤쪽 VAL_RATIO를 검증 손실로 조기 종료."""
    epochs = EPOCHS if epochs is None else epochs
    torch.manual_seed(SEED)

    X, y = build_dataset(df, seq_len)
    n_val = max(1, int(len(X) * VAL_RATIO))
    X_tr, y_tr, X_val, y_val = X[:-n_val], y[:-n_val], X[-n_val:], y[-n_val:]

    model = DrawLSTM().to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    loss_fn = nn.BCEWithLogitsLoss()

    best_val = float("inf")
    best_state = {k: v.clone() for k, v in model.state_dict().items()}
    patience_left = PATIENCE

    for epoch in range(1, epochs + 1):
        model.train()
        perm = torch.randperm(len(X_tr))
        for start in range(0, len(X_tr), BATCH_SIZE):
            idx = perm[start:start + BATCH_SIZE]
            optimizer.zero_grad()
            loss = loss_fn(model(X_tr[idx].to(DEVICE)), y_tr[idx].to(DEVICE))
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            val_loss = float(loss_fn(model(X_val.to(DEVICE)), y_val.to(DEVICE)))
        log.debug("epoch %d/%d val_loss=%.5f", epoch, epochs, val_loss)

        if val_loss < best_val - 1e-5:
            best_val = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            patience_left = PATIENCE
        else:
            patience_left -= 1
            if patience_left <= 0:
                log.debug("조기 종료 (epoch %d, best val_loss=%.5f)", epoch, best_val)
                break

    model.load_state_dict(best_state)
    model.eval()
    return model


def predict_probs(model: DrawLSTM, df: pd.DataFrame, seq_len: int = SEQ_LEN) -> pd.Series:
    """이력의 마지막 seq_len회차를 넣어 다음 회차의 번호별 포함 확률을 얻는다."""
    hot = _multi_hot(numbers_matrix(df))
    if len(hot) < seq_len:
        raise ValueError(f"데이터가 부족합니다. 최소 {seq_len}회차가 필요합니다 (현재 {len(hot)}회차).")
    window = torch.from_numpy(hot[-seq_len:]).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        probs = torch.sigmoid(model(window)).squeeze(0).cpu().numpy()
    return pd.Series(probs.astype(float), index=range(1, 46), name="lstm_prob")


# ------------------------------------------------------------ 캐시/전략 진입점

_cache: tuple[int, DrawLSTM] | None = None  # (학습에 쓴 마지막 회차 번호, 모델)


def clear_cache() -> None:
    global _cache
    _cache = None


def number_weights(df: pd.DataFrame, epochs: int | None = None) -> pd.Series:
    """predictor에 등록되는 진입점. 필요할 때만 재학습하고 확률을 가중치로 돌려준다."""
    global _cache
    max_draw = int(df["draw_no"].max())

    stale = (
        _cache is None
        or _cache[0] > max_draw                      # 미래 데이터로 학습된 모델 -> 누출 방지
        or max_draw - _cache[0] >= RETRAIN_EVERY     # 오래된 모델 -> 재학습
    )
    if stale:
        log.info("LSTM 학습 시작 (%d회차까지, %d행)…", max_draw, len(df))
        _cache = (max_draw, train(df, epochs=epochs))
        log.info("LSTM 학습 완료.")
    return predict_probs(_cache[1], df)
