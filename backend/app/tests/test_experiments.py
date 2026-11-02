"""
Full experiment workflow against the MySQL test database: training, saving,
retrieval, reports, artifact downloads and error handling.
"""

import io

import joblib
import pytest

from app.database.models import Experiment, ModelResult
from app.services import training_service
from app.tests.conftest import make_classification_df, to_csv_bytes

CLASSIFIERS = ["logistic_regression", "decision_tree_classifier", "random_forest_classifier"]
REGRESSORS = ["linear_regression", "decision_tree_regressor", "random_forest_regressor"]


def run(client, dataset_id, target, task, models, **extra):
    payload = {"dataset_id": dataset_id, "target_column": target, "task_type": task, "model_names": models, **extra}
    return client.post("/api/experiments", json=payload)


def test_available_models_are_filtered_by_task(client):
    names = {m["name"] for m in client.get("/api/models?task_type=regression").json()}
    assert names == set(REGRESSORS)


def test_classification_experiment_end_to_end(client, upload, classification_csv, db_session, settings):
    dataset_id = upload(classification_csv)["dataset"]["id"]
    response = run(client, dataset_id, "label", "classification", CLASSIFIERS, test_size=0.25, random_seed=7)
    assert response.status_code == 201, response.text
    body = response.json()

    assert body["status"] == "completed"
    assert body["primary_metric"] == "f1"
    assert body["run_metadata"]["test_rows"] == 20
    assert len(body["results"]) == 3
    assert body["best_model_name"] in CLASSIFIERS
    assert [r["is_best"] for r in body["results"]].count(True) == 1
    assert body["results"][0]["model_name"] == body["best_model_name"]
    for result in body["results"]:
        assert result["status"] == "success"
        assert result["has_artifact"]
        assert result["chart_data"]["confusion_matrix"]["labels"] == ["no", "yes"]
        assert "actual_vs_predicted" not in result["chart_data"]

    # Results are persisted in MySQL and readable in a fresh session.
    stored = db_session.get(Experiment, body["id"])
    assert stored.status == "completed"
    assert len(stored.results) == 3

    # Retrieval endpoint returns the same saved data.
    detail = client.get(f"/api/experiments/{body['id']}").json()
    assert detail["best_model_name"] == body["best_model_name"]
    assert {r["model_name"] for r in detail["results"]} == set(CLASSIFIERS)
    assert len(detail["comparison_chart"]) == 3


def test_downloaded_pipeline_predicts_on_raw_features(client, upload, classification_csv):
    dataset_id = upload(classification_csv)["dataset"]["id"]
    exp_id = run(client, dataset_id, "label", "classification", ["random_forest_classifier"]).json()["id"]

    response = client.get(f"/api/experiments/{exp_id}/models/random_forest_classifier/download")
    assert response.status_code == 200
    assert "attachment" in response.headers["content-disposition"]
    pipeline = joblib.load(io.BytesIO(response.content))  # trusted: we just created it
    raw = make_classification_df(20, seed=99).drop(columns=["label"])
    predictions = pipeline.predict(raw)
    assert set(predictions) <= {"yes", "no"}


def test_regression_experiment_end_to_end(client, upload, regression_csv):
    dataset_id = upload(regression_csv)["dataset"]["id"]
    response = run(client, dataset_id, "price", "regression", REGRESSORS)
    assert response.status_code == 201, response.text
    body = response.json()
    assert body["status"] == "completed"
    assert body["primary_metric"] == "rmse"
    assert body["best_model_name"] in REGRESSORS
    assert body["results"][0]["model_name"] == body["best_model_name"]
    for result in body["results"]:
        assert "confusion_matrix" not in result["chart_data"]
        assert len(result["chart_data"]["actual_vs_predicted"]) == body["run_metadata"]["test_rows"]


def test_invalid_target_column_creates_no_experiment(client, upload, classification_csv, db_session):
    dataset_id = upload(classification_csv)["dataset"]["id"]
    response = run(client, dataset_id, "not_a_column", "classification", CLASSIFIERS)
    assert response.status_code == 422
    assert "does not exist" in response.json()["detail"]
    assert db_session.query(Experiment).count() == 0


def test_unsuitable_target_for_task_is_rejected(client, upload, regression_csv):
    dataset_id = upload(regression_csv)["dataset"]["id"]
    response = run(client, dataset_id, "price", "classification", ["logistic_regression"])
    assert response.status_code == 422
    assert "regression" in response.json()["detail"]


def test_incompatible_task_and_model_rejected(client, upload, classification_csv):
    dataset_id = upload(classification_csv)["dataset"]["id"]
    response = run(client, dataset_id, "label", "classification", ["linear_regression"])
    assert response.status_code == 422
    assert "cannot be used for classification" in response.json()["detail"]


def test_unknown_model_and_bad_parameters_rejected(client, upload, classification_csv):
    dataset_id = upload(classification_csv)["dataset"]["id"]
    assert run(client, dataset_id, "label", "classification", ["svm"]).status_code == 422
    assert run(client, dataset_id, "label", "classification", CLASSIFIERS, test_size=0.9).status_code == 422
    assert run(client, dataset_id, "label", "classification", []).status_code == 422
    assert run(client, dataset_id, "label", "clustering", CLASSIFIERS).status_code == 422
    dup = run(client, dataset_id, "label", "classification", ["logistic_regression", "logistic_regression"])
    assert dup.status_code == 422


def test_unknown_dataset_returns_404(client):
    response = run(client, "00000000-0000-0000-0000-000000000000", "label", "classification", CLASSIFIERS)
    assert response.status_code == 404


def test_one_failing_model_gives_partial_status(client, upload, classification_csv, monkeypatch):
    real = training_service.train_and_evaluate

    def flaky(spec, *args, **kwargs):
        if spec.name == "decision_tree_classifier":
            raise RuntimeError("simulated training failure")
        return real(spec, *args, **kwargs)

    monkeypatch.setattr("app.services.experiment_service.train_and_evaluate", flaky)
    dataset_id = upload(classification_csv)["dataset"]["id"]
    body = run(client, dataset_id, "label", "classification", CLASSIFIERS).json()

    assert body["status"] == "partial"
    failed = [r for r in body["results"] if r["status"] == "failed"]
    assert len(failed) == 1 and "simulated" in failed[0]["error_message"]
    assert failed[0]["has_artifact"] is False
    assert body["best_model_name"] != "decision_tree_classifier"
    # Failed models have nothing to download.
    response = client.get(f"/api/experiments/{body['id']}/models/decision_tree_classifier/download")
    assert response.status_code == 409


def test_all_models_failing_marks_experiment_failed(client, upload, classification_csv, monkeypatch):
    def always_fail(*args, **kwargs):
        raise RuntimeError("boom")

    monkeypatch.setattr("app.services.experiment_service.train_and_evaluate", always_fail)
    dataset_id = upload(classification_csv)["dataset"]["id"]
    body = run(client, dataset_id, "label", "classification", ["logistic_regression"]).json()
    assert body["status"] == "failed"
    assert body["best_model_name"] is None
    # No report for an experiment without results.
    assert client.get(f"/api/experiments/{body['id']}/report").status_code == 409


def test_experiment_history_is_paginated_newest_first(client, upload, classification_csv, regression_csv):
    first = upload(classification_csv)["dataset"]["id"]
    second = upload(regression_csv)["dataset"]["id"]
    run(client, first, "label", "classification", ["logistic_regression"])
    newest = run(client, second, "price", "regression", ["linear_regression"]).json()

    page = client.get("/api/experiments?page=1&page_size=1").json()
    assert page["total"] == 2
    assert len(page["items"]) == 1
    item = page["items"][0]
    assert item["id"] == newest["id"]
    assert item["dataset_name"] == "data.csv"
    assert item["best_metric_value"] == pytest.approx(newest["results"][0]["metrics"]["rmse"])


def test_get_unknown_experiment_returns_404(client):
    assert client.get("/api/experiments/nope").status_code == 404


def test_report_contains_saved_results_and_escapes_html(client, upload, settings):
    df = make_classification_df(60)
    df = df.rename(columns={"color": "<script>alert(1)</script>"})
    dataset_id = upload(to_csv_bytes(df), "report<test>.csv")["dataset"]["id"]
    body = run(client, dataset_id, "label", "classification", CLASSIFIERS).json()

    response = client.get(f"/api/experiments/{body['id']}/report")
    assert response.status_code == 200
    html = response.text
    assert "Random Forest Classifier" in html
    assert body["id"] in html
    assert "<script>" not in html
    assert "does not guarantee" in html or "do not guarantee" in html
    assert (settings.report_path / f"modelbench_report_{body['id']}.html").exists()


def test_artifact_path_traversal_is_blocked(client, upload, classification_csv, db_session):
    dataset_id = upload(classification_csv)["dataset"]["id"]
    exp_id = run(client, dataset_id, "label", "classification", ["logistic_regression"]).json()["id"]
    # Tamper with the stored path to point outside artifacts/.
    result = db_session.query(ModelResult).filter_by(experiment_id=exp_id).one()
    result.artifact_filename = "../../../backend/.env"
    db_session.commit()
    response = client.get(f"/api/experiments/{exp_id}/models/logistic_regression/download")
    assert response.status_code == 404


def test_missing_artifact_file_returns_404(client, upload, regression_csv, settings):
    dataset_id = upload(regression_csv)["dataset"]["id"]
    exp_id = run(client, dataset_id, "price", "regression", ["linear_regression"]).json()["id"]
    (settings.artifact_path / exp_id / "linear_regression.joblib").unlink()
    response = client.get(f"/api/experiments/{exp_id}/models/linear_regression/download")
    assert response.status_code == 404


def test_artifacts_are_unique_per_experiment(client, upload, regression_csv, settings):
    dataset_id = upload(regression_csv)["dataset"]["id"]
    a = run(client, dataset_id, "price", "regression", ["linear_regression"]).json()["id"]
    b = run(client, dataset_id, "price", "regression", ["linear_regression"]).json()["id"]
    assert (settings.artifact_path / a / "linear_regression.joblib").exists()
    assert (settings.artifact_path / b / "linear_regression.joblib").exists()



def test_target_classes_endpoint_returns_binary_labels(client, upload, classification_csv):
    dataset_id = upload(classification_csv)["dataset"]["id"]
    response = client.get(f"/api/datasets/{dataset_id}/target-classes", params={"target_column": "label"})
    assert response.status_code == 200
    body = response.json()
    assert body["classes"] == ["no", "yes"]
    assert body["positive_class_selection"] is True


def test_experiment_accepts_user_selected_positive_class(client, upload, classification_csv):
    dataset_id = upload(classification_csv)["dataset"]["id"]
    response = client.post("/api/experiments", json={
        "dataset_id": dataset_id, "target_column": "label", "task_type": "classification",
        "model_names": ["logistic_regression"], "positive_class": "no"
    })
    assert response.status_code == 201, response.text
    body = response.json()
    assert body["positive_class"] == "no"
    assert "positive class 'no'" in body["results"][0]["metric_notes"]["averaging"]


def test_experiment_history_and_detail_are_isolated_between_users(client, upload, classification_csv):
    from datetime import datetime
    from app.api.auth import require_current_user
    from app.database.connection import get_session_factory
    from app.database.models import User
    from app.main import app

    dataset_id = upload(classification_csv)["dataset"]["id"]
    experiment_id = run(client, dataset_id, "label", "classification", ["logistic_regression"]).json()["id"]

    other_user = User(
        id="other-test-user", full_name="Other User", email="other@example.com",
        password_hash="not-used", created_at=datetime.utcnow(),
    )
    seed = get_session_factory()()
    seed.merge(other_user)
    seed.commit()
    seed.close()

    app.dependency_overrides[require_current_user] = lambda: other_user
    try:
        assert client.get("/api/experiments").json()["total"] == 0
        assert client.get(f"/api/experiments/{experiment_id}").status_code == 404
        assert client.get(f"/api/experiments/{experiment_id}/report").status_code == 404
        assert client.get(f"/api/experiments/{experiment_id}/models/logistic_regression/download").status_code == 404
        # Another account cannot start a run against someone else's dataset.
        assert run(client, dataset_id, "label", "classification", ["logistic_regression"]).status_code == 404
    finally:
        app.dependency_overrides[require_current_user] = lambda: User(
            id="test-user", full_name="Test User", email="test@example.com",
            password_hash="not-used", created_at=datetime.utcnow(),
        )
