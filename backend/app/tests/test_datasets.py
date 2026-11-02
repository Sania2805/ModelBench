"""Upload validation and dataset profiling (needs the MySQL test database)."""

import pandas as pd

from app.core.config import get_settings
from app.database.models import Dataset
from app.tests.conftest import make_classification_df, to_csv_bytes


def post_csv(client, content: bytes, filename: str = "data.csv"):
    return client.post("/api/datasets/upload", files={"file": (filename, content, "text/csv")})


def test_upload_valid_csv_returns_profile(client, classification_csv, settings, db_session):
    response = post_csv(client, classification_csv, "my data.csv")
    assert response.status_code == 201
    body = response.json()

    assert body["dataset"]["row_count"] == 80
    assert body["dataset"]["column_count"] == 4
    assert body["dataset"]["original_filename"] == "my data.csv"
    assert body["numeric_column_count"] == 2
    assert body["categorical_column_count"] == 2
    assert body["total_missing_values"] == 5
    assert len(body["preview"]) == 10
    assert body["insight_id"]
    assert body["insight_version"] == 1
    assert "missing_by_column" in body
    assert "recommendations" in body
    assert set(body["numeric_statistics"]) == {"x1", "x2"}
    columns = {c["name"]: c for c in body["columns"]}
    assert columns["x1"]["missing_count"] == 3
    assert columns["color"]["kind"] == "categorical"

    # Metadata is in MySQL; the file is on disk under a generated name, unchanged.
    dataset = db_session.get(Dataset, body["dataset"]["id"])
    assert dataset is not None
    assert dataset.stored_filename == f"{dataset.id}.csv"
    assert (settings.upload_path / dataset.stored_filename).read_bytes() == classification_csv
    # The server-side filename/path is never exposed to the client.
    assert "stored_filename" not in body["dataset"]


def test_profile_endpoint_matches_upload(client, upload, classification_csv):
    dataset_id = upload(classification_csv)["dataset"]["id"]
    response = client.get(f"/api/datasets/{dataset_id}/profile")
    assert response.status_code == 200
    assert response.json()["dataset"]["id"] == dataset_id
    assert response.json()["duplicate_row_count"] == 0


def test_profile_detects_duplicates_and_identifiers(client):
    df = make_classification_df(40)
    df.insert(0, "customer_id", range(1000, 1040))
    df = pd.concat([df, df.head(3)])  # 3 duplicate rows
    response = post_csv(client, to_csv_bytes(df))
    body = response.json()
    assert body["duplicate_row_count"] == 3
    assert any("customer_id" in w for w in body["warnings"])


def test_list_datasets(client, upload, classification_csv, regression_csv):
    upload(classification_csv)
    upload(regression_csv)
    body = client.get("/api/datasets").json()
    assert body["total"] == 2
    assert len(body["items"]) == 2


def test_rejects_non_csv_extension(client, classification_csv):
    response = post_csv(client, classification_csv, "data.xlsx")
    assert response.status_code == 422
    assert "csv" in response.json()["detail"].lower()


def test_rejects_empty_file(client):
    response = post_csv(client, b"")
    assert response.status_code == 422
    assert "empty" in response.json()["detail"].lower()


def test_rejects_binary_content_with_csv_extension(client):
    response = post_csv(client, b"PK\x03\x04\x00\x00binarystuff" * 10)
    assert response.status_code == 422


def test_rejects_malformed_csv(client):
    content = b"a,b,c\n1,2,3\n" + b"4,5,6,7,8\n" * 12
    response = post_csv(client, content)
    assert response.status_code == 422
    assert "parsed" in response.json()["detail"]


def test_rejects_duplicate_column_names(client):
    content = b"a,b,a\n" + b"1,2,3\n" * 12
    response = post_csv(client, content)
    assert response.status_code == 422
    assert "unique" in response.json()["detail"]


def test_rejects_too_few_rows(client):
    response = post_csv(client, b"a,b\n1,2\n3,4\n")
    assert response.status_code == 422
    assert "rows" in response.json()["detail"]


def test_rejects_single_column(client):
    response = post_csv(client, b"only\n" + b"1\n" * 20)
    assert response.status_code == 422


def test_rejects_file_over_size_limit(client, classification_csv, settings):
    from app.main import app

    tiny = settings.model_copy(update={"max_upload_size_mb": 0.001})  # about 1 KB
    app.dependency_overrides[get_settings] = lambda: tiny
    response = post_csv(client, classification_csv)
    assert response.status_code == 413


def test_upload_ignores_path_in_filename(client, classification_csv):
    response = post_csv(client, classification_csv, "../../etc/evil.csv")
    assert response.status_code == 201
    assert response.json()["dataset"]["original_filename"] == "evil.csv"


def test_profile_unknown_dataset_returns_404(client):
    response = client.get("/api/datasets/00000000-0000-0000-0000-000000000000/profile")
    assert response.status_code == 404


def test_saved_insights_endpoints(client, upload, classification_csv):
    dataset_id = upload(classification_csv)["dataset"]["id"]
    latest = client.get(f"/api/datasets/{dataset_id}/insights")
    assert latest.status_code == 200
    assert latest.json()["version"] == 1
    assert "correlation_matrix" in latest.json()["profile"]

    regenerated = client.post(f"/api/datasets/{dataset_id}/insights/regenerate")
    assert regenerated.status_code == 201
    assert regenerated.json()["version"] == 2

    history = client.get(f"/api/datasets/{dataset_id}/insights/history")
    assert history.status_code == 200
    assert history.json()["total"] == 2
    assert [item["version"] for item in history.json()["items"]] == [2, 1]


def test_datasets_and_insights_are_isolated_between_users(client, upload, classification_csv):
    """A second authenticated account cannot list or fetch another user's data."""
    from datetime import datetime
    from app.api.auth import require_current_user
    from app.database.connection import get_session_factory
    from app.database.models import User
    from app.main import app

    dataset_id = upload(classification_csv)["dataset"]["id"]

    other_user = User(
        id="other-test-user",
        full_name="Other User",
        email="other@example.com",
        password_hash="not-used",
        created_at=datetime.utcnow(),
    )
    seed = get_session_factory()()
    seed.merge(other_user)
    seed.commit()
    seed.close()

    app.dependency_overrides[require_current_user] = lambda: other_user
    try:
        assert client.get("/api/datasets").json()["total"] == 0
        assert client.get(f"/api/datasets/{dataset_id}/profile").status_code == 404
        assert client.get(f"/api/datasets/{dataset_id}/insights").status_code == 404
        assert client.get(f"/api/datasets/{dataset_id}/insights/history").status_code == 404
        assert client.post(f"/api/datasets/{dataset_id}/insights/regenerate").status_code == 404
        assert client.get(f"/api/datasets/{dataset_id}/target-classes?target_column=label").status_code == 404
    finally:
        app.dependency_overrides[require_current_user] = lambda: User(
            id="test-user", full_name="Test User", email="test@example.com",
            password_hash="not-used", created_at=datetime.utcnow(),
        )
