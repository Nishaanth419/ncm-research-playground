import os
import tempfile


def pytest_configure(config):
    # Set before any fixture (including module-scoped ones) builds an app,
    # so tests never write to results/playground.sqlite.
    os.environ["NCM_DB_PATH"] = os.path.join(tempfile.mkdtemp(prefix="ncm-tests-"), "test.sqlite")
