import pytest


@pytest.fixture(autouse=True)
def _isolated_db(tmp_path, monkeypatch):
    # never write to results/playground.sqlite from tests
    monkeypatch.setenv("NCM_DB_PATH", str(tmp_path / "test.sqlite"))
