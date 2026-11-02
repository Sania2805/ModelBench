def test_health_reports_database_connected(client, settings):
    response = client.get("/api/health")
    assert response.status_code == 200
    body = response.json()
    assert body == {"status": "ok", "database": "connected"}
    # The health check must never leak connection details.
    assert settings.database_user not in response.text
    assert settings.database_host not in response.text


def test_unknown_route_returns_json_404(client):
    response = client.get("/api/does-not-exist")
    assert response.status_code == 404
    assert "detail" in response.json()
