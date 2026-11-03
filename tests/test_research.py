"""Small execution checks; synthetic scores are never competition evidence."""

import numpy as np
import pandas as pd
import pytest
import hashlib
from zipfile import ZipFile

from src import import_official_data
from src.research.compare_models import encode_lgbm, fit_component, raw_signatures


def test_relocated_research_uses_repository_root():
    from pathlib import Path
    from src.research.compare_models import ROOT
    assert ROOT == Path(__file__).resolve().parents[1]
    assert (ROOT / "src/research/compare_models.py").is_file()


@pytest.mark.parametrize("module", ["src.audits.public_source_diagnostic", "src.research.build_submission_attributes"])
def test_quarantined_entries_refuse_default_execution(module):
    import subprocess
    import sys
    from pathlib import Path
    result = subprocess.run([sys.executable, "-m", module], cwd=Path(__file__).resolve().parents[1],
                            capture_output=True, text=True)
    assert result.returncode != 0
    assert "Disabled audit" in result.stderr or "Abandoned prototype" in result.stderr


def test_query_vocabulary_cannot_change_training_encoding():
    train = pd.DataFrame({"place": ["a", "b", "a"], "area": [60, 70, 80]})
    query = pd.DataFrame({"place": ["never_seen"], "area": [90]})
    left, right, vocabulary = encode_lgbm(train, query)
    assert vocabulary["place"] == ["a", "b"]
    assert right["place"].isna().all()
    assert left["place"].cat.codes.tolist() == [0, 1, 0]


@pytest.mark.parametrize("learner", ["catboost", "lgbm"])
@pytest.mark.parametrize("mode", ["direct", "residual", "relative", "log"])
def test_fit_predicts_finite_rent_in_original_units(learner, mode):
    if learner == "lgbm":
        pytest.importorskip("lightgbm", reason="Optional research dependency; install requirements-research.txt")
    rng = np.random.default_rng(42)
    count = 200
    frame = pd.DataFrame({
        "area": rng.uniform(50, 130, count),
        "place": ["a", "b"] * (count // 2),
        "ADAPT_ADDRESS_FLAT_AREA_LEVEL": rng.uniform(2200, 2800, count),
    })
    target = frame["ADAPT_ADDRESS_FLAT_AREA_LEVEL"] + 2 * frame["area"]
    _, prediction, _ = fit_component(
        frame.iloc[:160], target.iloc[:160], frame.iloc[160:],
        learner=learner, mode=mode, seed=42, threads=2,
        config={"iterations": 12, "learning_rate": 0.1, "depth": 3, "l2_leaf_reg": 6},
    )
    assert prediction.shape == (40,)
    assert np.isfinite(prediction).all()
    assert 2000 < prediction.mean() < 3500


def test_raw_signatures_preserve_case_without_mutating_input():
    raw = pd.DataFrame({
        "TOWN": ["TEST", "test"], "BLOCK": ["1", "1"],
        "STREET": ["Test Road", "test road"], "FLAT_TYPE": ["3-room"] * 2,
        "FLAT_MODEL": ["Model A"] * 2, "FLOOR_AREA_SQM": [64, 67],
    })
    features = pd.DataFrame({"canonical_town": ["test"] * 2})
    result = raw_signatures(features, raw)
    assert result["RAW_TOWN"].tolist() == ["TEST", "test"]
    assert result["RAW_TOWN_UPPER_COUNT"].tolist() == [4, 0]
    assert result["AREA_BIN_5"].tolist() == [60, 65]
    assert features.columns.tolist() == ["canonical_town"]


def test_import_rejects_bad_archive_before_any_write(tmp_path, monkeypatch):
    expected = {"train.csv": hashlib.sha256(b"correct").hexdigest()}
    monkeypatch.setattr(import_official_data, "RAW_SHA256", expected)
    archive = tmp_path / "wrong.zip"
    with ZipFile(archive, "w") as source:
        source.writestr("train.csv", b"wrong")
    destination = tmp_path / "raw"
    with pytest.raises(ValueError, match="checksum"):
        import_official_data.import_archive(archive, destination)
    assert not destination.exists()


def test_import_writes_only_fixed_destination_paths(tmp_path, monkeypatch):
    payload = b"correct"
    expected = {"train.csv": hashlib.sha256(payload).hexdigest()}
    monkeypatch.setattr(import_official_data, "RAW_SHA256", expected)
    archive = tmp_path / "valid.zip"
    with ZipFile(archive, "w") as source:
        source.writestr("../../train.csv", payload)
        source.writestr("../../unrequested.txt", b"do not extract")
    destination = tmp_path / "raw"
    import_official_data.import_archive(archive, destination)
    assert (destination / "train.csv").read_bytes() == payload
    assert sorted(p.name for p in destination.iterdir()) == ["train.csv"]
