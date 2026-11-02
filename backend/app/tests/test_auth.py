"""Authentication API tests."""

def test_register_creates_account_and_sets_http_only_cookie(client):
    response = client.post(
        "/api/auth/register",
        json={"full_name": "Test Person", "email": "test.person@example.com", "password": "correct-horse-123"},
    )
    assert response.status_code == 201, response.text
    assert response.json()["email"] == "test.person@example.com"
    assert "modelbench_access_token" in response.cookies
    assert "password_hash" not in response.json()


def test_duplicate_registration_is_rejected(client):
    payload = {"full_name": "Test Person", "email": "duplicate@example.com", "password": "correct-horse-123"}
    assert client.post("/api/auth/register", json=payload).status_code == 201
    response = client.post("/api/auth/register", json=payload)
    assert response.status_code == 409


def test_login_rejects_incorrect_password(client):
    client.post(
        "/api/auth/register",
        json={"full_name": "Test Person", "email": "login@example.com", "password": "correct-horse-123"},
    )
    response = client.post(
        "/api/auth/login",
        json={"email": "login@example.com", "password": "incorrect-password"},
    )
    assert response.status_code == 401
