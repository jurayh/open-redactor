import pytest

pytest.importorskip("fastapi")
pytest.importorskip("multipart")

from fastapi.testclient import TestClient

from open_redactor.server import create_app


def test_server_health(tmp_path):
    client = TestClient(create_app(tmp_path))
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_server_rejects_unsupported_upload(tmp_path):
    client = TestClient(create_app(tmp_path), raise_server_exceptions=False)
    response = client.post("/audit", files={"file": ("notes.txt", b"hello", "text/plain")})
    assert response.status_code == 400
