"""
HTML experiment reports.

The report is built ONLY from data already saved in MySQL — no model is
retrained, so the report always matches what the results page showed.

We produce a single self-contained HTML file (inline CSS and an inline SVG bar
chart), which opens in any browser and can be printed to PDF. This avoids
extra dependencies such as matplotlib or a PDF library.

Every value that came from the user (file names, column names, labels) is
passed through html.escape() so a malicious CSV header cannot inject HTML or
JavaScript into the report.
"""

from datetime import timezone
from html import escape
from pathlib import Path
from typing import Any

from app.core.config import Settings
from app.core.errors import ConflictError
from app.database.models import Experiment
from app.services.experiment_service import build_experiment_detail

METRIC_LABELS = {
    "accuracy": "Accuracy",
    "precision": "Precision",
    "recall": "Recall",
    "f1": "F1-score",
    "roc_auc": "ROC-AUC",
    "mae": "MAE",
    "mse": "MSE",
    "rmse": "RMSE",
    "r2": "R²",
}
CLASSIFICATION_METRICS = ["accuracy", "precision", "recall", "f1", "roc_auc"]
REGRESSION_METRICS = ["mae", "mse", "rmse", "r2"]


def _fmt(value: Any, digits: int = 4) -> str:
    if value is None:
        return "N/A"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return escape(str(value))


def _bar_chart_svg(rows: list[dict[str, Any]], metric: str) -> str:
    """Draw a simple horizontal bar chart of one metric across models."""
    points = [(r["display_name"], r["metrics"][metric]) for r in rows if r["metrics"].get(metric) is not None]
    if not points:
        return "<p>No valid values to chart.</p>"
    width, bar_height, gap, label_width = 640, 26, 10, 210
    max_value = max(abs(v) for _, v in points) or 1.0
    height = len(points) * (bar_height + gap) + gap
    bars = []
    for i, (name, value) in enumerate(points):
        y = gap + i * (bar_height + gap)
        bar_width = max(2.0, (width - label_width - 80) * abs(value) / max_value)
        bars.append(
            f'<text x="{label_width - 8}" y="{y + 17}" text-anchor="end" font-size="12" fill="#1e293b">{escape(name)}</text>'
            f'<rect x="{label_width}" y="{y}" width="{bar_width:.1f}" height="{bar_height}" rx="3" fill="#2563eb"/>'
            f'<text x="{label_width + bar_width + 6:.1f}" y="{y + 17}" font-size="12" fill="#1e293b">{value:.4f}</text>'
        )
    return f'<svg width="{width}" height="{height}" role="img" aria-label="{METRIC_LABELS[metric]} by model">{"".join(bars)}</svg>'


def _confusion_matrix_html(chart_data: dict[str, Any] | None) -> str:
    matrix_data = (chart_data or {}).get("confusion_matrix")
    if not matrix_data:
        return ""
    labels = matrix_data["labels"]
    header = "".join(f"<th>Pred: {escape(label)}</th>" for label in labels)
    body = "".join(
        f"<tr><th>Actual: {escape(label)}</th>" + "".join(f"<td>{count}</td>" for count in row) + "</tr>"
        for label, row in zip(labels, matrix_data["matrix"])
    )
    return f'<table class="matrix"><tr><th></th>{header}</tr>{body}</table>'


def _summary_sentences(detail: dict[str, Any]) -> list[str]:
    """Short, factual findings. We describe what happened on THIS split only."""
    metric = detail["primary_metric"]
    label = METRIC_LABELS[metric]
    successful = [r for r in detail["results"] if r["status"] == "success" and r["metrics"].get(metric) is not None]
    failed = [r for r in detail["results"] if r["status"] == "failed"]
    sentences = []
    if successful:
        best = successful[0]  # results are already sorted best-first
        sentences.append(
            f"{best['display_name']} was selected using validation {label}. "
            f"Its final held-out test-set {label} was {best['metrics'][metric]:.4f}."
        )
        if len(successful) > 1:
            runner_up = successful[1]
            gap = abs(best["metrics"][metric] - runner_up["metrics"][metric])
            sentences.append(
                f"The next model in the displayed comparison, {runner_up['display_name']}, differed by {gap:.4f} on the test set. "
                "Small differences on a single split may not be meaningful."
            )
        fastest = min(successful, key=lambda r: r["training_time_seconds"] or 0)
        sentences.append(f"{fastest['display_name']} trained fastest ({fastest['training_time_seconds']:.3f} s).")
    if failed:
        sentences.append(f"{len(failed)} model(s) failed to train: {', '.join(r['display_name'] for r in failed)}.")
    sentences.append(
        "Models are selected using a validation split; the reported metrics are calculated on a separate, untouched test split. "
        "A single split and default hyperparameters do not guarantee which model will generalise best to new data."
    )
    return sentences


def render_report_html(experiment: Experiment) -> str:
    detail = build_experiment_detail(experiment)
    metric_keys = CLASSIFICATION_METRICS if detail["task_type"] == "classification" else REGRESSION_METRICS
    meta = detail["run_metadata"] or {}
    created = experiment.created_at.replace(tzinfo=timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    metric_header = "".join(f"<th>{METRIC_LABELS[k]}</th>" for k in metric_keys)
    result_rows = []
    for r in detail["results"]:
        if r["status"] == "success":
            cells = "".join(f"<td>{_fmt(r['metrics'].get(k))}</td>" for k in metric_keys)
            timing = f"<td>{_fmt(r['training_time_seconds'])}</td><td>{_fmt(r['prediction_time_seconds'])}</td>"
        else:
            cells = f'<td colspan="{len(metric_keys)}">Failed: {escape(r["error_message"] or "unknown error")}</td>'
            timing = "<td>-</td><td>-</td>"
        best_tag = ' <span class="best">Best</span>' if r["is_best"] else ""
        result_rows.append(f"<tr><td>{escape(r['display_name'])}{best_tag}</td>{cells}{timing}</tr>")

    notes = []
    for r in detail["results"]:
        for key, note in (r["metric_notes"] or {}).items():
            notes.append(f"<li><b>{escape(r['display_name'])}</b> ({escape(key)}): {escape(note)}</li>")

    matrices = ""
    if detail["task_type"] == "classification":
        matrices = "".join(
            f"<h3>{escape(r['display_name'])}</h3>{_confusion_matrix_html(r['chart_data'])}"
            for r in detail["results"]
            if r["status"] == "success"
        )
        matrices = f"<h2>Confusion matrices</h2>{matrices}" if matrices else ""

    warnings_html = "".join(f"<li>{escape(w)}</li>" for w in meta.get("warnings", []))
    best_name = next((r["display_name"] for r in detail["results"] if r["is_best"]), "None")

    return f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<title>ModelBench report {escape(detail['id'][:8])}</title>
<style>
 :root {{ color-scheme: light; }}
 body {{ font-family: Segoe UI, Arial, sans-serif; color: #0f172a; background: #ffffff; max-width: 960px; margin: 32px auto; padding: 0 16px; }}
 h1 {{ margin-bottom: 4px; }} h2 {{ margin-top: 32px; border-bottom: 1px solid #e2e8f0; padding-bottom: 6px; }}
 .muted {{ color: #64748b; }}
 table {{ border-collapse: collapse; width: 100%; margin: 12px 0; font-size: 14px; }}
 th, td {{ border: 1px solid #e2e8f0; padding: 6px 10px; text-align: left; }}
 th {{ background: #f8fafc; }}
 table.matrix {{ width: auto; }} table.matrix td {{ text-align: center; min-width: 48px; }}
 .best {{ background: #dbeafe; color: #1d4ed8; padding: 1px 6px; border-radius: 4px; font-size: 12px; }}
 .grid {{ display: grid; grid-template-columns: 220px 1fr; gap: 4px 16px; }}
</style></head><body>
<h1>ModelBench experiment report</h1>
<p class="muted">Experiment {escape(detail['id'])} &middot; created {created} &middot; status {escape(detail['status'])}</p>

<h2>Dataset and configuration</h2>
<div class="grid">
 <span>Dataset</span><span>{escape(detail['dataset']['original_filename'])} ({detail['dataset']['row_count']} rows &times; {detail['dataset']['column_count']} columns)</span>
 <span>Target column</span><span>{escape(detail['target_column'])}</span>
 <span>Task type</span><span>{escape(detail['task_type'])}</span>
 <span>Train / validation / test</span><span>{meta.get('train_rows', '?')} / {meta.get('validation_rows', '?')} / {meta.get('test_rows', '?')} rows (test size {detail['test_size']:.0%})</span>
 <span>Random seed</span><span>{detail['random_seed']}</span>
 <span>Models evaluated</span><span>{escape(', '.join(r['display_name'] for r in detail['results']))}</span>
 <span>Primary metric</span><span>{METRIC_LABELS[detail['primary_metric']]} ({'higher' if detail['primary_metric_higher_is_better'] else 'lower'} is better)</span>
 <span>Best model (validation-selected)</span><span>{escape(best_name)}</span>
</div>

<h2>Summary</h2>
<ul>{''.join(f'<li>{escape(s)}</li>' for s in _summary_sentences(detail))}</ul>

<h2>Metrics (test set)</h2>
<table><tr><th>Model</th>{metric_header}<th>Train time (s)</th><th>Predict time (s)</th></tr>{''.join(result_rows)}</table>

<h2>{METRIC_LABELS[detail['primary_metric']]} comparison</h2>
{_bar_chart_svg([r for r in detail['results'] if r['status'] == 'success'], detail['primary_metric'])}
{matrices}
{'<h2>Metric notes</h2><ul>' + ''.join(notes) + '</ul>' if notes else ''}
{'<h2>Data warnings</h2><ul>' + warnings_html + '</ul>' if warnings_html else ''}
<p class="muted">Generated by ModelBench from saved experiment data.</p>
</body></html>"""


def generate_report(experiment: Experiment, settings: Settings) -> Path:
    """Write the report to reports/ and return its path. One file per experiment ID."""
    if experiment.status not in ("completed", "partial"):
        raise ConflictError("A report is only available for experiments with at least one successful model.")
    settings.report_path.mkdir(parents=True, exist_ok=True)
    # The experiment ID is a server-generated UUID, so this name is unique and safe.
    path = settings.report_path / f"modelbench_report_{experiment.id}.html"
    path.write_text(render_report_html(experiment), encoding="utf-8")
    return path
