"""Tests for the FastAPI web application."""

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from src.web.app import app


@pytest.fixture
def client() -> TestClient:
    """Create a test client."""
    return TestClient(app)


class TestRoutes:
    """Test basic route availability."""

    def test_chat_page_returns_html(self, client: TestClient) -> None:
        resp = client.get("/")
        assert resp.status_code == 200
        assert "cfa-optimization-agent" in resp.text

    def test_upload_requires_file(self, client: TestClient) -> None:
        resp = client.post("/upload")
        assert resp.status_code == 422

    def test_upload_csv(self, client: TestClient, tmp_path: Path) -> None:
        csv_content = b"id,name,value\n1,Alice,10\n"
        resp = client.post(
            "/upload",
            files={"file": ("test.csv", csv_content, "text/csv")},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert "run_id" in data

    def test_nonexistent_run_returns_404(self, client: TestClient) -> None:
        resp = client.get("/run/nonexistent/data")
        assert resp.status_code == 404
