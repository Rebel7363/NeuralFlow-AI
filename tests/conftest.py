import pytest
from fastapi.testclient import TestClient
from train import main as train_main


@pytest.fixture(scope="session")
def trained_dir(tmp_path_factory):
    """Train once for the whole test session into a temp folder (does not touch your real models/)."""
    d = tmp_path_factory.mktemp("artifacts")
    train_main(out=str(d / "results"), model_dir=str(d / "models"))
    return d


@pytest.fixture()
def client(trained_dir, monkeypatch):
    monkeypatch.setenv("MODEL_DIR", str(trained_dir / "models"))
    from app.main import app
    with TestClient(app) as c:      # `with` runs the startup code that loads the model
        yield c
