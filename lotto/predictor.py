"""과거 당첨번호 분석 기반 다음 회차 번호 추천.

전략별로 45개 번호에 점수(가중치)를 매긴 뒤, 그 가중치로 6개를 비복원 추출한다.
마지막에 조합 필터(합계 범위, 홀짝 균형 등)로 통계적으로 드문 조합을 걸러낸다.

주의: 로또 추첨은 매 회차 독립적인 균등 무작위 시행이다. 아래 전략들은 과거
데이터의 편차를 근거로 번호를 고르지만, 그 편차가 다음 회차 확률을 바꾸지는
않는다. backtest 모듈로 실제 성능을 직접 확인해 보길 권한다.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd

from . import analyzer

log = logging.getLogger(__name__)

NUMBERS = np.arange(1, 46)

# 전략 이름 -> 점수 함수(df -> 45개 가중치 Series)
Strategy = Callable[[pd.DataFrame], pd.Series]
_REGISTRY: dict[str, Strategy] = {}

# 조합 단위 점수: 전략 이름 -> (df -> (조합 -> 0~1 채택 확률))
# 번호별 가중치로 표현할 수 없는, 조합 전체의 성질(합계 분포 등)을 다루는 전략용.
ComboScorer = Callable[[list[int]], float]
_COMBO_REGISTRY: dict[str, Callable[[pd.DataFrame], ComboScorer]] = {}

# 커스텀 샘플러: 전략 이름 -> (df -> (rng -> 조합)).
# 번호를 독립적으로 뽑지 않는 전략(순차 조건부 샘플링 등)용. 등록돼 있으면
# build_sampler가 번호별 가중치 경로 대신 이것을 쓴다(필터는 공통 적용).
Sampler = Callable[[np.random.Generator], list[int]]
_SAMPLER_REGISTRY: dict[str, Callable[[pd.DataFrame], Sampler]] = {}


def register(name: str) -> Callable[[Strategy], Strategy]:
    def deco(fn: Strategy) -> Strategy:
        _REGISTRY[name] = fn
        return fn
    return deco


def register_combo(name: str) -> Callable:
    def deco(fn: Callable[[pd.DataFrame], ComboScorer]) -> Callable:
        _COMBO_REGISTRY[name] = fn
        return fn
    return deco


def register_sampler(name: str) -> Callable:
    def deco(fn: Callable[[pd.DataFrame], Sampler]) -> Callable:
        _SAMPLER_REGISTRY[name] = fn
        return fn
    return deco


def available_strategies() -> list[str]:
    return sorted(set(_REGISTRY) | set(_SAMPLER_REGISTRY))


def _normalize(scores: pd.Series) -> pd.Series:
    """음수를 제거하고 합이 1이 되도록 정규화한다."""
    s = scores.clip(lower=0).astype(float)
    total = s.sum()
    if total <= 0:
        return pd.Series(1 / len(s), index=s.index)
    return s / total


# ---------------------------------------------------------------- 전략들

@register("uniform")
def uniform_scores(df: pd.DataFrame) -> pd.Series:
    """균등 무작위. 다른 전략을 비교할 기준선(baseline)."""
    return pd.Series(1.0, index=NUMBERS)


@register("hot")
def hot_scores(df: pd.DataFrame, half_life: int = 100) -> pd.Series:
    """최근 자주 나온 번호(핫넘버)에 가중치."""
    return _normalize(analyzer.weighted_frequency(df, half_life=half_life))


@register("cold")
def cold_scores(df: pd.DataFrame, half_life: int = 100) -> pd.Series:
    """최근 덜 나온 번호(콜드넘버)에 가중치. hot의 반대 가정."""
    w = analyzer.weighted_frequency(df, half_life=half_life)
    return _normalize(w.max() - w + w.mean() * 0.1)


@register("overdue")
def overdue_scores(df: pd.DataFrame) -> pd.Series:
    """평균 출현 간격 대비 오래 안 나온 번호에 가중치."""
    gap = analyzer.gaps(df).astype(float)
    mean = analyzer.mean_gap(df).replace(0, np.nan)
    ratio = (gap / mean).fillna(1.0)
    return _normalize(ratio)


@register("pair")
def pair_scores(df: pd.DataFrame, last_n: int = 200) -> pd.Series:
    """직전 회차 번호들과 자주 함께 나온 번호에 가중치."""
    pairs = analyzer.pair_matrix(df.tail(last_n) if len(df) > last_n else df)
    last_numbers = df.sort_values("draw_no")[analyzer.NUMBER_COLUMNS].iloc[-1].tolist()
    affinity = pairs.loc[:, last_numbers].sum(axis=1).astype(float)
    # 직전 회차 번호가 그대로 반복되는 경우는 드무므로 약하게 눌러 준다.
    affinity.loc[last_numbers] *= 0.5
    return _normalize(affinity)


@register("balanced")
def balanced_scores(df: pd.DataFrame) -> pd.Series:
    """hot / overdue / pair를 섞은 기본 전략."""
    parts = {
        "hot": (hot_scores(df), 0.4),
        "overdue": (overdue_scores(df), 0.35),
        "pair": (pair_scores(df), 0.25),
    }
    total = sum(s * w for s, w in parts.values())
    return _normalize(total)


# --------------------------------------------------- 오라클(oracle) — 부정 데모
#
# ⚠️ 이 전략은 일부러 만든 '부정 행위 시연'이다.
#
# 백테스트에서 1등·2등을 맞추는 유일한 방법은 미래 정보를 보는 것임을 증명하기
# 위해, 전달받은 history 밖의 전체 데이터를 몰래 읽어 "다음 회차의 실제 당첨번호"
# 에 가중치를 준다. 시뮬레이션에서는 1등을 쏟아내지만, 진짜 미래(아직 추첨 전인
# 회차)를 예측할 때는 볼 데이터가 없으므로 균등 무작위로 전락한다.
#
# 시중의 "백테스트에서 1등 다수 적중" 광고가 정확히 이 구조다. 정직한 전략의
# 1등 기대 횟수는 5,665게임 기준 0.0007회 — 기대 1회가 되려면 8,145,060게임
# (최근 1133회차 기준 회차당 7,190게임 = 719만원어치)이 필요하다.

ORACLE_BONUS_WEIGHT = 0.2    # 보너스 번호 가중치 — 가끔 5개+보너스(2등)가 나오게
ORACLE_OTHER_WEIGHT = 1e-4   # 나머지 번호 — 0이면 조합 다양성이 없어 predict가 막힌다


def _oracle_source() -> pd.DataFrame:
    """오라클이 몰래 읽는 전체 데이터. 테스트에서 교체할 수 있도록 분리."""
    from . import storage
    return storage.load()


ORACLE_SOURCE = _oracle_source


@register("oracle")
def oracle_scores(df: pd.DataFrame) -> pd.Series:
    """다음 회차의 '실제' 당첨번호에 가중치를 준다 (미래 정보 누출).

    다음 회차가 전체 데이터에 없으면(=진짜 미래) 균등으로 전락한다.
    """
    full = ORACLE_SOURCE()
    next_no = int(df["draw_no"].max()) + 1
    row = full[full["draw_no"] == next_no] if not full.empty else full

    if row is None or len(row) == 0:
        log.warning(
            "oracle: %d회는 아직 추첨 전이라 볼 미래가 없습니다. "
            "균등 무작위로 동작합니다 — 이것이 이 전략의 실전 성능입니다.", next_no,
        )
        return pd.Series(1.0, index=NUMBERS)

    winning = row.iloc[0][analyzer.NUMBER_COLUMNS].astype(int).tolist()
    bonus = int(row.iloc[0]["bonus"])

    weights = pd.Series(ORACLE_OTHER_WEIGHT, index=NUMBERS)
    weights.loc[winning] = 1.0
    weights.loc[bonus] = ORACLE_BONUS_WEIGHT
    return weights


# --------------------------------------------------- 비인기(unpopular) 전략
#
# 당첨 확률을 높이는 게 아니라 '당첨 시 나눠 갖는 사람 수'를 줄이는 전략.
# popularity 모듈이 회차별 1등 당첨자 수로 조합 인기도를 회귀 추정하고,
# 여기서는 인기도의 역수에 비례하게 조합을 뽑는다: 샘플링 분포 ∝ exp(-η(조합)).
#
# η의 번호 단위 항(sum/low31/low12)은 번호별 가중치 exp(-η_n)으로,
# 조합 수준 항(consec)은 조합 점수(기각 샘플링)로 나눠 처리하면
# 곱해서 정확히 exp(-η(조합))이 된다.


# 기울임 강도 γ: 샘플링 분포 ∝ exp(-γ·η). γ=1이면 인기도의 정확한 역수 비례인데,
# 추정된 η의 폭이 좁아(조합 간 최대 ~0.5) 실질 효과가 약하다. γ를 키우면
# 비인기 조합에 더 집중하되 조합 다양성은 유지된다.
UNPOPULAR_SHARPNESS = 4.0


@register("unpopular")
def unpopular_scores(df: pd.DataFrame) -> pd.Series:
    """번호별 인기도 기여분의 역수. 인기 없는 번호(주로 32~45)에 가중치."""
    from . import popularity
    etas = popularity.fit(df).number_etas()
    return _normalize(np.exp(-UNPOPULAR_SHARPNESS * (etas - etas.min())))


@register_combo("unpopular")
def unpopular_combo_scorer(df: pd.DataFrame) -> ComboScorer:
    """조합 수준 특성(연속 번호 여부)의 인기도 항을 채택 확률로 변환한다."""
    from . import popularity
    beta_consec = float(popularity.fit(df).coef[popularity.FEATURE_NAMES.index("consec")])

    def score(combo: list[int]) -> float:
        combo = sorted(combo)
        has_consec = any(b - a == 1 for a, b in zip(combo, combo[1:]))
        eta = beta_consec * has_consec
        # exp(-γη)를 최댓값 1로 정규화: η가 낮은(비인기) 쪽이 항상 1
        eta_min = min(0.0, beta_consec)
        return math.exp(-UNPOPULAR_SHARPNESS * (eta - eta_min))

    return score


# --------------------------------------------------- 2개 조합(pairwise) 전략
#
# 번호를 독립적으로 6개 뽑는 대신, "이미 뽑은 번호들과 과거에 자주 함께 나온
# 번호"를 다음 번호로 뽑는 순차 조건부 샘플링. 다음 번호 n의 가중치는
#   w(n) = Π_{m ∈ 이미 뽑은 번호} (pair(m, n) + α)
# 로, 2개 조합의 경험적 동시 출현 확률을 그대로 쓴다. α=1(라플라스 평활)은
# 한 번도 같이 안 나온 쌍의 확률이 0이 되는 것을 막는다. 곱은 로그 공간에서
# 계산해 언더플로를 피한다.

PAIRWISE_SMOOTHING = 1.0


@register("pairwise")
def pairwise_scores(df: pd.DataFrame) -> pd.Series:
    """score_table용 번호별 관점: 다른 번호들과의 동시 출현 횟수 합(주변 강도).

    실제 샘플링은 pairwise_sampler(순차 조건부)가 담당한다.
    """
    pairs = analyzer.pair_matrix(df)
    return _normalize(pairs.sum(axis=1).astype(float))


@register_sampler("pairwise")
def pairwise_sampler(df: pd.DataFrame) -> Sampler:
    counts = analyzer.pair_matrix(df).to_numpy().astype(float) + PAIRWISE_SMOOTHING
    log_pairs = np.log(counts)  # (45, 45), 대각선은 log(α)지만 선택에서 제외됨

    marginal = counts.sum(axis=1)
    marginal = marginal / marginal.sum()

    def sample(rng: np.random.Generator) -> list[int]:
        chosen = [int(rng.choice(45, p=marginal))]  # 0-based 인덱스
        while len(chosen) < 6:
            log_w = log_pairs[chosen].sum(axis=0)
            w = np.exp(log_w - log_w.max())
            w[chosen] = 0.0
            w /= w.sum()
            chosen.append(int(rng.choice(45, p=w)))
        return sorted(n + 1 for n in chosen)

    return sample


# --------------------------------------------------- zscore (정규분포 이탈도)
#
# "모든 현상은 정규분포한다"는 가정 아래, 관측 출현 횟수가 이론값에서 가장 크게
# 벗어난 조합을 그리디로 고른다. 자세한 절차는 lotto/zscore.py 참고.
# 선택이 결정적이라 커스텀 샘플러로 붙인다.

ZSCORE_POOL = 45  # 시작점을 z가 작은 순 45개(전체)로 확장한 후보 풀


@register("zscore")
def zscore_scores(df: pd.DataFrame) -> pd.Series:
    """번호별 가중치 (score_table·폴백용).

    이탈 방향이 하향이므로 z가 작을수록(덜 나왔을수록) 큰 가중치를 준다.
    """
    from . import zscore
    membership = zscore.membership_matrix(df)
    counts = membership.sum(axis=0).astype(float)
    z = zscore.deviation_scores(counts, len(membership), k=1)
    return _normalize(pd.Series(z.max() - z, index=NUMBERS))


@register_sampler("zscore")
def zscore_sampler(df: pd.DataFrame) -> Sampler:
    from . import zscore

    # 요청한 만큼만 계산한다(45개를 미리 다 만들면 백테스트가 느려진다)
    pool = zscore.iter_sets(df)
    weights = zscore_scores(df)

    def sample(rng: np.random.Generator) -> list[int]:
        """덜 나온 순 후보를 내주고, 후보가 떨어지면 가중 추출로 넘어간다."""
        combo = next(pool, None)
        return combo if combo is not None else draw_combination(weights, rng)

    return sample


# --------------------------------------------------- randomforest (lotto-anal)
#
# 이전 프로젝트 lotto-anal(2023)의 로직. 45개 단일번호와 990개 2개조합 각각에
# RandomForest를 학습해 다음 회차 출현 확률을 예측하고, '따라가기'로 조합을
# 만든다(자세한 구조와 검증 결과는 lotto/lotto_anal.py 와 README 참고).
#
# 조합 생성이 확률 가중 추출이 아니라 결정적 체인이므로 커스텀 샘플러로 붙인다.
# 원본과 완전히 동일한 5세트를 얻으려면 조합 필터를 꺼야 한다:
#     predictor.predict(df, strategy="randomforest", games=5, use_filter=False)
# 필터를 켜면 합계가 극단적인 세트가 걸러지고 그다음 후보로 대체된다.

RANDOMFOREST_POOL = 45  # 따라가기 시작점을 45개 번호 전체로 확장한 후보 풀


@register("randomforest")
def randomforest_scores(df: pd.DataFrame) -> pd.Series:
    """단일번호 45개에 대한 RandomForest 예측 확률 (score_table·폴백용).

    scikit-learn이 필요하다: pip install -r requirements.txt
    """
    from . import lotto_anal
    probs = lotto_anal.cached_probabilities(df)
    return _normalize(pd.Series(probs[:45], index=NUMBERS))


@register_sampler("randomforest")
def randomforest_sampler(df: pd.DataFrame) -> Sampler:
    from . import lotto_anal

    probs = lotto_anal.cached_probabilities(df)
    pool = lotto_anal.follow_the_pairs(probs, n_sets=RANDOMFOREST_POOL)
    weights = _normalize(pd.Series(probs[:45], index=NUMBERS))
    position = 0

    def sample(rng: np.random.Generator) -> list[int]:
        """후보 풀을 순서대로 내주고, 풀이 떨어지면 확률 가중 추출로 넘어간다."""
        nonlocal position
        if position < len(pool):
            combo = pool[position]
            position += 1
            return combo
        return draw_combination(weights, rng)

    return sample


@register("lstm")
def lstm_strategy(df: pd.DataFrame) -> pd.Series:
    """LSTM 시계열 모델이 예측한 번호별 포함 확률을 가중치로 쓴다.

    torch가 필요하다(선택 설치): pip install -r requirements-lstm.txt
    무거운 의존성이라 실제 사용 시점에 지연 임포트한다.
    """
    from . import lstm
    return lstm.number_weights(df)


# --------------------------------------------------- CLT(중심극한정리) 전략
#
# 1~45 모집단의 평균 μ=23, 분산 σ²=(45²-1)/12 ≈ 168.67.
# 비복원 추출 6개 합계의 이론 분포는
#   평균  = 6μ = 138
#   분산  = 6σ² · (N-n)/(N-1) = 6·168.67·(39/44) ≈ 897   (표준편차 ≈ 29.9)
# 이고, 중심극한정리에 따라 근사적으로 정규분포를 따른다.
#
# 개별 번호는 모두 균등하게 두되(가정상 특정 번호에 우열이 없음), 조합의 합계가
# 이론 평균 138에 가까울수록 채택 확률을 높이는 기각 샘플링으로 구현한다.
# 결과적으로 추천 조합의 합계 분포가 위 정규분포를 따라가게 된다.

CLT_SUM_MEAN = 6 * (1 + 45) / 2  # 138.0
CLT_SUM_VAR = 6 * ((45**2 - 1) / 12) * ((45 - 6) / (45 - 1))  # ≈ 897.0
CLT_SUM_STD = math.sqrt(CLT_SUM_VAR)


@register("clt")
def clt_scores(df: pd.DataFrame) -> pd.Series:
    """번호별로는 균등 — CLT 전략의 핵심은 조합 점수(clt_combo_scorer)에 있다."""
    return pd.Series(1.0, index=NUMBERS)


@register_combo("clt")
def clt_combo_scorer(df: pd.DataFrame) -> ComboScorer:
    """합계가 이론 평균 138에 가까울수록 1에 가까운 채택 확률을 준다.

    정규 밀도를 최댓값(합계=138)으로 나눠 0~1로 정규화한 값이라, 기각 샘플링의
    채택 확률로 그대로 쓸 수 있다.
    """
    def score(combo: list[int]) -> float:
        return math.exp(-((sum(combo) - CLT_SUM_MEAN) ** 2) / (2 * CLT_SUM_VAR))
    return score


# ---------------------------------------------------------------- 조합 필터

@dataclass
class CombinationFilter:
    """통계적으로 드문 조합을 걸러내는 규칙 묶음.

    과거 당첨 조합의 실제 분포에서 뽑은 경계를 쓴다.
    """

    sum_min: int
    sum_max: int
    min_odd: int = 1
    max_odd: int = 5
    max_consecutive: int = 3
    max_same_decade: int = 4

    @classmethod
    def from_history(cls, df: pd.DataFrame) -> "CombinationFilter":
        stats = analyzer.sum_stats(df)
        return cls(sum_min=int(stats["p05"]), sum_max=int(stats["p95"]))

    def accepts(self, combo: list[int]) -> bool:
        combo = sorted(combo)
        if not self.sum_min <= sum(combo) <= self.sum_max:
            return False

        odd = sum(1 for n in combo if n % 2 == 1)
        if not self.min_odd <= odd <= self.max_odd:
            return False

        # 연속된 숫자가 너무 길게 이어지는 조합 배제 (예: 11,12,13,14)
        run = longest = 1
        for prev, cur in zip(combo, combo[1:]):
            run = run + 1 if cur == prev + 1 else 1
            longest = max(longest, run)
        if longest > self.max_consecutive:
            return False

        # 한 십의 자리에 몰린 조합 배제
        decades = pd.Series([n // 10 for n in combo]).value_counts()
        return int(decades.max()) <= self.max_same_decade


# ---------------------------------------------------------------- 추천 생성

def draw_combination(
    weights: pd.Series,
    rng: np.random.Generator,
    combo_filter: CombinationFilter | None = None,
    combo_scorer: ComboScorer | None = None,
    max_attempts: int = 500,
) -> list[int]:
    """가중치에 따라 번호 6개를 비복원 추출한다.

    combo_scorer가 있으면 그 값(0~1)을 채택 확률로 쓰는 기각 샘플링을 한다.
    필터/기각을 통과하는 조합을 max_attempts까지 시도하고, 실패하면 마지막
    조합을 그대로 돌려준다(조건이 지나치게 빡빡한 경우 무한 루프 방지).
    """
    p = _normalize(weights).to_numpy()
    combo: list[int] = []
    for _ in range(max_attempts):
        combo = sorted(int(n) for n in rng.choice(NUMBERS, size=6, replace=False, p=p))
        if combo_filter is not None and not combo_filter.accepts(combo):
            continue
        if combo_scorer is not None and rng.random() >= combo_scorer(combo):
            continue
        return combo
    return combo


def build_sampler(
    df: pd.DataFrame,
    strategy: str = "balanced",
    use_filter: bool = True,
) -> Callable[[np.random.Generator], list[int]]:
    """전략 이름으로 '조합 1개를 뽑는 함수'를 만든다. predict와 backtest가 공유한다."""
    if strategy not in _REGISTRY and strategy not in _SAMPLER_REGISTRY:
        raise ValueError(
            f"알 수 없는 전략: {strategy!r} (사용 가능: {', '.join(available_strategies())})"
        )
    if df.empty:
        raise ValueError("분석할 데이터가 없습니다. 먼저 `python main.py update`를 실행하세요.")

    combo_filter = CombinationFilter.from_history(df) if use_filter else None

    if strategy in _SAMPLER_REGISTRY:
        base = _SAMPLER_REGISTRY[strategy](df)

        def sample(rng: np.random.Generator, max_attempts: int = 500) -> list[int]:
            combo: list[int] = []
            for _ in range(max_attempts):
                combo = base(rng)
                if combo_filter is None or combo_filter.accepts(combo):
                    return combo
            return combo

        return sample

    weights = _REGISTRY[strategy](df)
    combo_scorer = _COMBO_REGISTRY[strategy](df) if strategy in _COMBO_REGISTRY else None

    def sample(rng: np.random.Generator) -> list[int]:
        return draw_combination(weights, rng, combo_filter, combo_scorer)

    return sample


def predict(
    df: pd.DataFrame,
    strategy: str = "balanced",
    games: int = 5,
    seed: int | None = None,
    use_filter: bool = True,
) -> list[list[int]]:
    """다음 회차 추천 번호를 games개 만든다 (조합 중복 없음)."""
    sample = build_sampler(df, strategy, use_filter)
    rng = np.random.default_rng(seed)

    picks: list[list[int]] = []
    seen: set[tuple[int, ...]] = set()
    while len(picks) < games:
        combo = sample(rng)
        key = tuple(combo)
        if key in seen:
            continue
        seen.add(key)
        picks.append(combo)
    return picks


def score_table(df: pd.DataFrame, strategy: str = "balanced") -> pd.DataFrame:
    """번호별 점수와 근거 지표를 한 표로 정리한다."""
    weights = _normalize(_REGISTRY[strategy](df))
    return pd.DataFrame({
        "score": weights.round(5),
        "frequency": analyzer.frequency(df),
        "recent_100": analyzer.frequency(df, last_n=100),
        "gap": analyzer.gaps(df),
        "mean_gap": analyzer.mean_gap(df).round(1),
    }).sort_values("score", ascending=False)
