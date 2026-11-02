# Persistent Dataset Insights

ModelBench now stores each dataset analysis as an immutable MySQL snapshot in `dataset_insights`. Uploads automatically save version 1. Users can regenerate the analysis from the stored CSV and save additional versions; previous versions remain available in the dataset detail page.

## Database update

From the `backend` directory, with the existing `.env` configured and the virtual environment activated, run:

```powershell
python -m app.database.init_db
```

The existing initializer uses SQLAlchemy `create_all()`: it creates the new `dataset_insights` table and leaves existing tables and rows untouched. No manual `CREATE TABLE` statement is needed. Existing datasets are backfilled automatically the first time their profile is opened.

Verify in MySQL Workbench:

```sql
USE modelbench_db;
SHOW TABLES;
DESCRIBE dataset_insights;
SELECT dataset_id, version, generated_at FROM dataset_insights ORDER BY generated_at DESC;
```

## API

- `GET /api/datasets/{dataset_id}/insights` — latest saved snapshot.
- `GET /api/datasets/{dataset_id}/insights/history` — all saved versions, newest first.
- `POST /api/datasets/{dataset_id}/insights/regenerate` — recompute from the stored CSV and save a new version.
- `GET /api/datasets/{dataset_id}/profile` — existing profile endpoint, now backed by the latest saved snapshot (and backfills older datasets when needed).

The profile includes overview counts, numeric and categorical statistics, per-column missing-value information, a bounded Pearson correlation matrix (up to 40 numeric columns), and automated data-quality recommendations. Correlation is descriptive and does not imply causation.

## Frontend

The dashboard lists recent datasets and links to their saved analysis. The dataset analysis page supports viewing prior snapshots and regenerating the latest one. The upload page continues to show the immediate profile and offers a link to the saved analysis.
