from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient


class TestAnalysisAPI:
    def test_start_analysis_requires_analysts(self, client: TestClient):
        """POST /api/v1/analysis with empty analysts should return 400."""
        response = client.post("/api/v1/analysis", json={
            "ticker": "SPY",
            "analysts": [],
        })
        assert response.status_code == 400

    def test_start_analysis_returns_job_id(self, client: TestClient):
        """POST /api/v1/analysis with valid params should return job_id."""
        with patch("app.services.analysis_service.AnalysisService._run_analysis"):
            response = client.post("/api/v1/analysis", json={
                "ticker": "SPY",
                "analysis_date": "2026-05-14",
                "analysts": ["market"],
                "research_depth": 1,
            })
        assert response.status_code == 202
        data = response.json()
        assert "job_id" in data
        assert data["status"] == "started"

    def test_get_job_not_found(self, client: TestClient):
        """GET non-existent job should return 404."""
        response = client.get("/api/v1/analysis/nonexistent-id")
        assert response.status_code == 404

    def test_get_report_not_found(self, client: TestClient):
        """GET report for non-existent job should return 404."""
        response = client.get("/api/v1/analysis/nonexistent-id/report")
        assert response.status_code == 404

    def test_get_job_status_after_creation(self, client: TestClient):
        """GET /{job_id} should return job details after creation."""
        with patch("app.services.analysis_service.AnalysisService._run_analysis"):
            create_resp = client.post("/api/v1/analysis", json={
                "ticker": "AAPL",
                "analysts": ["market", "news"],
                "research_depth": 1,
            })
        job_id = create_resp.json()["job_id"]

        response = client.get(f"/api/v1/analysis/{job_id}")
        assert response.status_code == 200
        data = response.json()
        assert data["id"] == job_id
        assert data["ticker"] == "AAPL"
        assert data["status"] == "running"

    def test_report_in_progress_returns_425(self, client: TestClient):
        """GET report for running job should return 425 Too Early."""
        with patch("app.services.analysis_service.AnalysisService._run_analysis"):
            create_resp = client.post("/api/v1/analysis", json={
                "ticker": "SPY",
                "analysts": ["market"],
                "research_depth": 1,
            })
        job_id = create_resp.json()["job_id"]

        response = client.get(f"/api/v1/analysis/{job_id}/report")
        assert response.status_code == 425
