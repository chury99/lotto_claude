"""런처(run.py) 테스트. 네트워크·전송 없이 각 단계를 모의로 검증한다."""

import json

import numpy as np
import pandas as pd
import pytest

import run
from lotto import notify


@pytest.fixture
def df():
    rng = np.random.default_rng(0)
    rows = []
    for i in range(1, 301):
        picks = rng.choice(np.arange(1, 46), size=7, replace=False)
        nums = sorted(picks[:6].tolist())
        rows.append({
            "draw_no": i, "draw_date": "2020-01-01",
            **{f"n{j+1}": nums[j] for j in range(6)},
            "bonus": int(picks[6]),
            "first_prize_winners": int(rng.poisson(10)),
            "first_prize_amount": 2_000_000_000,
            "total_sales": 5e10,
        })
    return pd.DataFrame(rows)


@pytest.fixture
def csv_path(df, tmp_path):
    path = tmp_path / "history.csv"
    df.to_csv(path, index=False)
    return str(path)


@pytest.fixture
def config_file(tmp_path):
    path = tmp_path / "telegram.json"
    path.write_text(json.dumps({
        notify.TOKEN_KEY: "tok", notify.CHAT_ID_KEY: "42",
    }), encoding="utf-8")
    return path


@pytest.fixture
def no_config(tmp_path):
    """존재하지 않는 설정 파일 경로."""
    return str(tmp_path / "none.json")


# --------------------------------------------------------------- 단계별 동작

def test_collect_skip_update_uses_existing(csv_path, capsys):
    out = run.collect(csv_path, skip=True)
    assert len(out) == 300
    assert "크롤링 건너뜀" in capsys.readouterr().out


def test_collect_skip_update_without_data_exits(tmp_path):
    with pytest.raises(SystemExit) as exc:
        run.collect(str(tmp_path / "none.csv"), skip=True)
    assert exc.value.code == 1


def test_collect_falls_back_when_crawl_fails(csv_path, monkeypatch, capsys):
    """수집이 실패해도 기존 데이터가 있으면 계속 진행한다."""
    def boom(*a, **k):
        raise RuntimeError("네트워크 끊김")

    monkeypatch.setattr(run.storage, "update", boom)
    out = run.collect(csv_path, skip=False)
    assert len(out) == 300
    assert "기존 데이터" in capsys.readouterr().out


def test_collect_exits_when_crawl_fails_and_no_cache(tmp_path, monkeypatch):
    def boom(*a, **k):
        raise RuntimeError("네트워크 끊김")

    monkeypatch.setattr(run.storage, "update", boom)
    with pytest.raises(SystemExit) as exc:
        run.collect(str(tmp_path / "none.csv"), skip=False)
    assert exc.value.code == 1


def test_analyze_prints_indicators(df, capsys):
    run.analyze(df, strategy="balanced")
    out = capsys.readouterr().out
    assert "전체 최다 출현" in out
    assert "장기 미출현" in out


def test_analyze_includes_popularity_for_unpopular(df, capsys):
    run.analyze(df, strategy="unpopular")
    assert "인기도 모델" in capsys.readouterr().out


def test_generate_returns_picks(df, capsys):
    picks, next_draw = run.generate(df, strategy="uniform", games=5, seed=1)
    assert next_draw == 301
    assert len(picks) == 5
    for combo in picks:
        assert len(set(combo)) == 6
    assert "301회 추천 번호" in capsys.readouterr().out


def test_dispatch_disabled(capsys):
    assert run.dispatch([[1, 2, 3, 4, 5, 6]], 1, "uniform", enabled=False) is False
    assert "발송하지 않았습니다" in capsys.readouterr().out


def test_dispatch_skips_without_config_file(no_config, capsys):
    """설정 파일이 없으면 안내만 하고 정상 진행(오류 아님)."""
    assert run.dispatch([[1, 2, 3, 4, 5, 6]], 1, "uniform",
                        enabled=True, config_path=no_config) is False
    assert "설정 파일이 없습니다" in capsys.readouterr().out


def test_dispatch_sends_with_config_file(monkeypatch, config_file, capsys):
    sent = {}

    def fake_send_picks(picks, draw_no, strategy, note=None, **kw):
        sent.update(picks=picks, draw_no=draw_no, strategy=strategy,
                    note=note, config_path=kw.get("config_path"))
        return {"message_id": 1}

    monkeypatch.setattr(run.notify, "send_picks", fake_send_picks)

    assert run.dispatch([[1, 2, 3, 4, 5, 6]], 1234, "unpopular",
                        enabled=True, config_path=str(config_file)) is True
    assert sent["draw_no"] == 1234
    assert "생성 시각" in sent["note"]
    assert str(sent["config_path"]) == str(config_file)
    assert "발송 완료" in capsys.readouterr().out


def test_dispatch_exits_on_send_failure(monkeypatch, config_file):
    def boom(*a, **k):
        raise notify.NotifyError("chat not found")

    monkeypatch.setattr(run.notify, "send_picks", boom)
    with pytest.raises(SystemExit) as exc:
        run.dispatch([[1, 2, 3, 4, 5, 6]], 1, "uniform",
                     enabled=True, config_path=str(config_file))
    assert exc.value.code == 1


# --------------------------------------------------------------- 전체 흐름

def test_main_skips_send_when_no_config(csv_path, no_config, capsys):
    """설정 파일이 없어도 전체 실행은 성공한다."""
    code = run.main(["--csv", csv_path, "--skip-update", "-n", "2",
                     "-s", "uniform", "--telegram-config", no_config])
    assert code == 0
    assert "설정 파일이 없습니다" in capsys.readouterr().out


def test_main_end_to_end_without_telegram(csv_path, capsys):
    code = run.main(["--csv", csv_path, "--skip-update", "--no-telegram",
                     "-n", "3", "-s", "uniform", "--seed", "1"])
    out = capsys.readouterr().out
    assert code == 0
    assert "[1/4]" in out and "[2/4]" in out and "[3/4]" in out and "[4/4]" in out
    assert "301회 3게임 생성" in out


def test_main_sends_when_configured(csv_path, config_file, monkeypatch, capsys):
    calls = []
    monkeypatch.setattr(run.notify, "send_picks",
                        lambda *a, **k: calls.append(a) or {"message_id": 1})

    code = run.main(["--csv", csv_path, "--skip-update", "-n", "2",
                     "-s", "uniform", "--seed", "1",
                     "--telegram-config", str(config_file)])
    assert code == 0
    assert len(calls) == 1
    assert "텔레그램 발송함" in capsys.readouterr().out


def test_main_runs_update_by_default(csv_path, monkeypatch):
    """--skip-update 없으면 storage.update가 호출된다."""
    called = []

    def fake_update(path, **kwargs):
        called.append(path)
        return pd.read_csv(csv_path)

    monkeypatch.setattr(run.storage, "update", fake_update)
    code = run.main(["--csv", csv_path, "--no-telegram", "-n", "1", "-s", "uniform"])
    assert code == 0
    assert called == [csv_path]


def test_main_reproducible_with_seed(csv_path, capsys):
    run.main(["--csv", csv_path, "--skip-update", "--no-telegram",
              "-n", "3", "-s", "uniform", "--seed", "7"])
    first = capsys.readouterr().out
    run.main(["--csv", csv_path, "--skip-update", "--no-telegram",
              "-n", "3", "-s", "uniform", "--seed", "7"])
    second = capsys.readouterr().out

    def combos(text):
        return [l for l in text.splitlines() if l.strip().startswith(("A.", "B.", "C."))]

    assert combos(first) == combos(second)
