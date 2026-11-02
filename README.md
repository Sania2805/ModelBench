# ModelBench

**ModelBench** is a web platform for comparing classic machine learning models on your own data.
Upload a CSV file, pick the column you want to predict, choose classification or regression, and
ModelBench trains several scikit-learn models with a leakage-safe preprocessing pipeline. It then
compares them with real metrics and charts and lets you download an HTML report and the trained
model pipelines. Every experiment is stored in MySQL, so your history survives restarts.

> Built as a portfolio project with FastAPI, React, scikit-learn and MySQL.

---

## Contents

1. [Main features](#main-features)
2. [Technology stack](#technology-stack)
3. [How it works (architecture)](#how-it-works-architecture)
4. [Project structure](#project-structure)
5. [Prerequisites](#prerequisites)
6. [Setup on Windows (PowerShell)](#setup-on-windows-powershell)
7. [Running the tests](#running-the-tests)
8. [Using ModelBench](#using-modelbench)
9. [API reference](#api-reference)
10. [Where files are stored](#where-files-are-stored)
11. [Using a downloaded model pipeline](#using-a-downloaded-model-pipeline)
12. [Machine learning design decisions](#machine-learning-design-decisions)
13. [Known limitations](#known-limitations)
14. [Security considerations](#security-considerations)
15. [Troubleshooting](#troubleshooting)
16. [Future improvements](#future-improvements)

---

## Main features

- **CSV upload with validation.** Checks the extension, size limit (10 MB), encoding, malformed
  rows, duplicate headers, binary files and minimum rows, and returns a clear message for each problem.
- **Dataset profiling.** Shows row and column counts, inferred types, missing values, duplicate rows,
  descriptive statistics and a 10-row preview. It also warns about likely ID columns and
  high-cardinality text.
- **Explicit task selection.** You choose classification or regression, and ModelBench never guesses.
  Unsuitable targets are rejected with an explanation.
- **Six models:**
  - Classification: Logistic Regression, Decision Tree, Random Forest
  - Regression: Linear Regression, Decision Tree, Random Forest
- **Leakage-safe preprocessing.** The data is split first; imputation, scaling and one-hot encoding
  are then fitted on the training rows only, inside a single scikit-learn `Pipeline`.
- **Honest evaluation:**
  - Classification metrics: accuracy, precision, recall, F1, ROC-AUC, confusion matrix and per-class metrics.
  - Regression metrics: MAE, MSE, RMSE and R², plus an actual-vs-predicted chart.
  - Metrics that cannot be computed are shown as **N/A with a reason** instead of a made-up number.
- **Best-model indicator.** Uses F1 for classification and RMSE for regression.
- **Downloads.** An HTML report for each experiment and a `.joblib` pipeline for each model.
- **Persistent history** in MySQL, with pagination.

## Technology stack

| Layer    | Technology |
|----------|------------|
| Frontend | React 19 + Vite, JavaScript/JSX, plain CSS, Recharts, Axios, React Router |
| Backend  | Python, FastAPI, Pydantic v2, pydantic-settings |
| Database | **MySQL 8** via SQLAlchemy 2.0 + PyMySQL (no SQLite anywhere) |
| ML       | pandas, NumPy, scikit-learn, joblib |
| Testing  | pytest, FastAPI TestClient, a dedicated MySQL test database |

## How it works (architecture)

```
 Browser (React, port 5173)
      │  HTTP/JSON (Axios)            The frontend never talks to MySQL directly.
      ▼
 FastAPI (port 8000)  ── app/api/*.py         thin routes: parse request → call service → return schema
      │
      ├── services/dataset_service.py         upload validation, profiling, safe file storage
      ├── services/preprocessing_service.py   target checks, feature selection, split, ColumnTransformer
      ├── services/training_service.py        model registry, Pipeline(preprocessor + model), timing
      ├── services/evaluation_service.py      metrics, N/A reasons, chart data, best-model choice
      ├── services/experiment_service.py      orchestrates a run, saves results + artifacts
      └── services/report_service.py          HTML report built from saved data
      │
      ├── MySQL (modelbench_db)               metadata, configs, metrics, chart data
      └── Disk: uploads/ artifacts/ reports/  the CSV files, .joblib pipelines and HTML reports
```

**What happens when you click "Run experiment":**

1. The API validates the request with Pydantic: task type, test size between 10% and 50%, the seed,
   and that no model is listed twice.
2. The service checks that the dataset exists and that every model matches the task.
3. The CSV is loaded and the target is checked. For example, decimals can't be a classification
   target and text can't be a regression target. If the data is unsuitable, **no experiment record
   is created**.
4. The data is split into train and test sets (stratified for classification when possible).
5. An `experiments` row is saved with status `running`.
6. Each model is trained and evaluated. If one model fails, the error is recorded for that model
   and the other models still run.
7. Each fitted pipeline is saved to `artifacts/<experiment_id>/<model>.joblib`.
8. All results and the final status are committed in **one transaction**. The status is
   `completed` (all models succeeded), `partial` (some failed) or `failed` (none succeeded).
   If saving fails, the artifact files are deleted again.

### How MySQL stores the data

```
datasets (1) ──< experiments (many) ──< model_results (many)
```

| Table | Key columns |
|-------|-------------|
| `datasets` | `id` (UUID), `original_filename`, `stored_filename` (server-generated), `file_size_bytes`, `row_count`, `column_count`, `columns_metadata` (JSON), `uploaded_at` |
| `experiments` | `id`, `dataset_id` (FK), `target_column`, `task_type`, `selected_models` (JSON), `test_size`, `random_seed`, `status`, `primary_metric`, `best_model_name`, `run_metadata` (JSON: split sizes, features, warnings), `error_message`, `created_at`, `completed_at` |
| `model_results` | `id`, `experiment_id` (FK, cascade delete), `model_name`, `status`, `metrics` (JSON), `metric_notes` (JSON), `chart_data` (JSON), `hyperparameters` (JSON), `training_time_seconds`, `prediction_time_seconds`, `artifact_filename`, `error_message` |

The tables also have foreign keys, indexes on common filters (`created_at`, `status`, `dataset_id`),
a unique `(experiment_id, model_name)` pair, and CHECK constraints on task type and status.
**CSV files and model files are never stored as BLOBs.** Only their server-generated relative names
are stored in MySQL.

## Project structure

```
ModelBench/
├── backend/
│   ├── app/
│   │   ├── main.py                  FastAPI app, CORS, error handlers, startup checks
│   │   ├── api/                     routes: health.py, datasets.py, experiments.py
│   │   ├── core/                    config.py (settings), security.py (safe paths), errors.py
│   │   ├── schemas/                 Pydantic request/response models
│   │   ├── services/                all business and ML logic (see above)
│   │   ├── database/                connection.py, models.py, init_db.py
│   │   └── tests/                   pytest suite (uses modelbench_test_db)
│   ├── requirements.txt
│   ├── pytest.ini
│   ├── .env.example                 placeholders — copy to .env
│   └── .env                         your real settings (git-ignored)
├── frontend/
│   ├── src/
│   │   ├── components/              Layout, tables, charts, dropzone, shared UI
│   │   ├── pages/                   Dashboard, Upload, Configure, Results, History
│   │   ├── services/api.js          the only place that calls the backend
│   │   ├── hooks/useAsync.js        loading / error / data state helper
│   │   ├── utils/format.js          number, date and metric formatting
│   │   └── styles/global.css
│   ├── package.json
│   └── .env.example
├── sample_data/                     two sample CSVs + the script that generated them
├── uploads/  artifacts/  reports/   created automatically, git-ignored
├── .gitignore
└── README.md
```

## Prerequisites

| Tool | Version | Check with |
|------|---------|------------|
| Python | 3.11 or newer (tested with 3.14) | `python --version` |
| Node.js | 20 or newer (tested with 24) | `node --version` |
| MySQL Server | 8.0 or 8.4 LTS | `mysql --version` |
| Git | any recent version | `git --version` |

## Setup on Windows (PowerShell)

> Each block says **which folder** to run it from. `ModelBench` means the project root folder.

### 1. Install and start MySQL

**Option A (recommended for most people): MySQL Installer**

1. Download **MySQL Installer for Windows** from <https://dev.mysql.com/downloads/installer/>.
2. Choose **Server only**, keep port **3306**, set a root password, and let it run MySQL as a Windows service.
3. MySQL now starts automatically with Windows. To check it, open the Services app and look for
   `MySQL84` (or `MySQL80`).

**Option B: portable ZIP (no installer and no admin rights)**

1. Download the "Windows (x86, 64-bit), ZIP Archive" from <https://dev.mysql.com/downloads/mysql/>
   and extract it, for example to `C:\mysql`. Avoid OneDrive-synced folders.
2. Create `C:\mysql\my.ini`:
   ```ini
   [mysqld]
   basedir=C:/mysql
   datadir=C:/mysql/data
   port=3306
   bind-address=127.0.0.1
   ```
3. Initialize the data folder once, then start the server. Keep that window open while you work:
   ```powershell
   C:\mysql\bin\mysqld.exe --defaults-file=C:\mysql\my.ini --initialize-insecure --console
   C:\mysql\bin\mysqld.exe --defaults-file=C:\mysql\my.ini --console
   ```
4. In a new window, set a root password:
   ```powershell
   C:\mysql\bin\mysql.exe -u root -e "ALTER USER 'root'@'localhost' IDENTIFIED BY 'choose-a-root-password';"
   ```

### 2. Create the databases and an application user

Open a MySQL shell as root (`mysql -u root -p`, or `C:\mysql\bin\mysql.exe -u root -p`) and run:

```sql
CREATE DATABASE modelbench_db
  CHARACTER SET utf8mb4
  COLLATE utf8mb4_unicode_ci;

-- Separate database used only by the automated tests.
CREATE DATABASE modelbench_test_db
  CHARACTER SET utf8mb4
  COLLATE utf8mb4_unicode_ci;

-- A dedicated user. The app does not need admin rights.
CREATE USER 'modelbench_user'@'localhost' IDENTIFIED BY 'choose-a-strong-password';
GRANT ALL PRIVILEGES ON modelbench_db.*      TO 'modelbench_user'@'localhost';
GRANT ALL PRIVILEGES ON modelbench_test_db.* TO 'modelbench_user'@'localhost';
FLUSH PRIVILEGES;
```

The application never creates the database itself. That is intentional, so the app's MySQL user
does not need database-administration privileges.

### 3. Configure the backend `.env`

From `ModelBench\backend`:

```powershell
Copy-Item .env.example .env
notepad .env
```

Fill in your values:

```env
DATABASE_HOST=localhost
DATABASE_PORT=3306
DATABASE_NAME=modelbench_db
DATABASE_USER=modelbench_user
DATABASE_PASSWORD=choose-a-strong-password
TEST_DATABASE_NAME=modelbench_test_db
```

Passwords with special characters such as `@ : / #` are fine. The app builds the connection URL
with SQLAlchemy's `URL.create()`, which escapes them correctly.

### 4. Create a virtual environment and install backend dependencies

From `ModelBench\backend`:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If PowerShell says *"running scripts is disabled on this system"*, allow local scripts for your
user account once, then activate again:

```powershell
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
```

You'll know the virtual environment is active when your prompt starts with `(.venv)`.

### 5. Create the MySQL tables

From `ModelBench\backend`, with the venv active:

```powershell
python -m app.database.init_db
```

Expected output: `Done. Tables: datasets, experiments, model_results`. It is safe to run this
again; it only creates tables that are missing.

### 6. Start the backend

From `ModelBench\backend`, with the venv active:

```powershell
uvicorn app.main:app --reload --port 8000
```

- Health check: <http://localhost:8000/api/health> should return `{"status":"ok","database":"connected"}`.
- Interactive API docs: <http://localhost:8000/docs>

### 7. Install and start the frontend

Open a **second** PowerShell window. From `ModelBench\frontend`:

```powershell
npm install
Copy-Item .env.example .env   # optional; the default API URL is http://localhost:8000/api
npm run dev
```

Open <http://localhost:5173>. The sidebar footer should say **"API and MySQL connected"**.

To create a production build: `npm run build` (output goes to `frontend\dist`).

## Running the tests

The tests use **MySQL only**. They connect to `TEST_DATABASE_NAME` (default `modelbench_test_db`),
never to your normal database. As safety checks, the suite refuses to start if the test database
name does not contain `test` or is the same as `DATABASE_NAME`. Uploads, artifacts and reports go
to a temporary folder that is deleted afterwards. The test tables are created at the start of the
run and dropped at the end.

From `ModelBench\backend`, with the venv active and MySQL running:

```powershell
python -m pytest
```

| Test file | What it covers |
|-----------|----------------|
| `test_health.py` | health endpoint, no leaked connection details, JSON 404s |
| `test_datasets.py` | upload validation (extension, empty, binary, malformed, duplicate headers, too few rows, size limit, path in filename), profiling, duplicates/ID detection, MySQL + disk storage |
| `test_preprocessing.py` | imputation, numeric/categorical/boolean handling, unknown categories, **fit-on-train-only**, target validation, leakage column removal, stratification, rare classes, metrics, ROC-AUC/R² N/A cases, best-model ranking (runs without MySQL) |
| `test_experiments.py` | full classification and regression runs, persistence, retrieval, history pagination, invalid targets and model combinations, partial/failed runs, report generation and HTML escaping, artifact download and prediction, path-traversal protection, missing artifact files |

If MySQL is not reachable, the database tests are **skipped** with a message telling you why. The
pure ML tests still run.

## Using ModelBench

The `sample_data` folder contains two synthetic datasets with real, learnable patterns and some
deliberate messiness (missing values and an ID column):

| File | Task | Target column |
|------|------|---------------|
| `customer_churn_classification.csv` (600 rows) | Classification | `churned` (yes/no) |
| `house_prices_regression.csv` (500 rows) | Regression | `price_thousands` |

To regenerate them: `python sample_data/generate_samples.py` (from `ModelBench`, with the backend venv active).

**Walkthrough:**

1. **Upload dataset.** Drag `customer_churn_classification.csv` onto the drop area and review the
   profile. You'll notice that `customer_id` is flagged as a likely identifier.
2. Click **Continue to experiment setup**.
3. Choose `churned` as the target and **Classification** as the task. All three classifiers are
   selected by default.
4. Keep test size 20% and seed 42, then click **Run experiment**.
5. On the **Results** page:
   - compare the metrics table and the bar chart (you can switch metrics),
   - inspect each model's confusion matrix,
   - click **Download report** or **Download** for a model pipeline.
6. Open **Experiment history** at any time to reopen past results.
7. Repeat with `house_prices_regression.csv`, target `price_thousands` and **Regression**.

## API reference

| Method | Path | Description |
|--------|------|-------------|
| GET | `/api/health` | API status and whether MySQL is reachable (no connection details) |
| POST | `/api/datasets/upload` | Upload a CSV (multipart field `file`); returns the profile |
| GET | `/api/datasets` | List uploaded datasets (`limit`, `offset`) |
| GET | `/api/datasets/{dataset_id}/profile` | Statistics, columns and a 10-row preview |
| GET | `/api/models?task_type=classification` | Models available for a task |
| POST | `/api/experiments` | Run an experiment (body below) |
| GET | `/api/experiments?page=1&page_size=10` | Paginated history, newest first |
| GET | `/api/experiments/{experiment_id}` | Full results and chart-ready data |
| GET | `/api/experiments/{experiment_id}/report` | Download the HTML report |
| GET | `/api/experiments/{experiment_id}/models/{model_name}/download` | Download a `.joblib` pipeline |

Request body for `POST /api/experiments`:

```json
{
  "dataset_id": "87b46c2e-...",
  "target_column": "churned",
  "task_type": "classification",
  "model_names": ["logistic_regression", "random_forest_classifier"],
  "test_size": 0.2,
  "random_seed": 42
}
```

Errors always look like `{"detail": "human readable message"}` and use these status codes:

| Code | Meaning |
|------|---------|
| 404 | Not found |
| 409 | Not possible in the current state, for example a report for a failed experiment |
| 413 | File too large |
| 422 | Invalid input or unsuitable data |
| 503 | Database unavailable |

## Where files are stored

| What | Location | Name |
|------|----------|------|
| Uploaded CSVs (original bytes, unchanged) | `ModelBench/uploads/` | `<dataset_id>.csv` |
| Trained pipelines | `ModelBench/artifacts/<experiment_id>/` | `<model_name>.joblib` |
| Reports | `ModelBench/reports/` | `modelbench_report_<experiment_id>.html` |
| Everything else (metadata, metrics, history) | MySQL `modelbench_db` | — |

These folders are created automatically and are git-ignored. You can change them with
`UPLOAD_DIR`, `ARTIFACT_DIR` and `REPORT_DIR` in `.env`.

## Using a downloaded model pipeline

Each `.joblib` file is a complete scikit-learn `Pipeline`: the fitted imputers, scaler and one-hot
encoder plus the fitted model. Give it **raw feature columns**, named exactly as in your training
CSV and without the target, and it handles the preprocessing itself:

```python
import joblib
import pandas as pd

pipeline = joblib.load("modelbench_random_forest_classifier_a4b38d28.joblib")
new_rows = pd.read_csv("new_customers.csv")          # same feature columns, no target
print(pipeline.predict(new_rows))
print(pipeline.predict_proba(new_rows))              # classifiers only
```

Use the same (or a compatible) scikit-learn version that trained the model.

> ⚠️ **Security warning:** `.joblib` and pickle files can execute arbitrary code when they are
> loaded. Only load files you created yourself or fully trust. Never load a `.joblib` file from
> an unknown source.

## Machine learning design decisions

- **Split before fitting.** Medians, scaling parameters and categories are learned from the
  training rows only. The test set plays the role of unseen future data, so no information from it
  should leak into training.
- **One Pipeline per model.** `Pipeline([("preprocessor", ColumnTransformer), ("model", estimator)])`.
  The exact same preprocessing is applied during training, evaluation and later predictions.
- **Numeric columns:** median imputation, which is robust to outliers. `StandardScaler` is applied
  only for the linear models; tree models don't need scaling.
- **Categorical columns:** most-frequent imputation, then `OneHotEncoder(handle_unknown="ignore")`,
  so categories that appear only at prediction time don't cause a crash.
- **Leakage guards:**
  - Columns that are exact copies of the target are excluded, with a warning.
  - Likely ID columns are **kept but flagged**, so you decide what to do with them.
  - Completely empty columns are excluded.
- **Stratified split** for classification when every class has at least 2 rows. Otherwise
  ModelBench falls back to a random split and tells you.
- **Averaging.** Binary problems use `average="binary"` with the positive class being the label
  that sorts last (`1`, `yes`, `True`). Multiclass problems use `average="weighted"`.
- **ROC-AUC** uses predicted probabilities. Binary problems use the positive-class column;
  multiclass problems use one-vs-rest with weighted averaging. If a class is missing from the test
  set, ROC-AUC is reported as N/A with the reason.
- **Primary metrics.** F1 for classification (higher is better) and RMSE for regression (lower is
  better). Models with an invalid primary metric are not ranked.
- **Default hyperparameters:**
  - Logistic Regression: `C=1.0`, `max_iter=1000`
  - Decision Trees: scikit-learn defaults
  - Random Forests: `n_estimators=100`

  The experiment's seed is used as `random_state`.

## Known limitations

- **Single train/test split.** There is no cross-validation or hyperparameter tuning. Small
  differences between models may come from the particular split, so try a few seeds.
- Training runs **synchronously** inside the HTTP request. Fine for files up to 10 MB, but the
  browser waits while models train.
- Date/time columns are treated as text categories. There is no automatic feature engineering.
- High-cardinality text columns, such as IDs, create many one-hot features. ModelBench warns about
  them but does not drop them.
- No authentication, so anyone who can reach the API can see all experiments. It is meant for
  local, single-user use.
- No delete endpoints yet. Remove data manually from MySQL and the folders if needed.
- Tables are created with `create_all()`. Schema changes would need a migration tool.

## Security considerations

- Secrets live only in `backend/.env`, which is git-ignored. `.env.example` contains placeholders only.
- The database password is never logged. Logs show only `mysql://host:port/database`.
- Database errors return a generic 503 message. SQL statements and credentials are never sent to
  the client.
- Uploaded files are saved under server-generated UUID names. The client's filename is used only
  for display, after sanitising.
- File paths stored in MySQL are resolved with a check that they stay inside `uploads/`,
  `artifacts/` or `reports/`. Path traversal returns 404.
- Report HTML escapes every user-supplied string, such as file names, column names and labels.
- CORS only allows the configured frontend origins (`CORS_ORIGINS`). Set it to your real domain
  in production.
- The frontend never connects to MySQL and contains no secrets.
- Only load `.joblib` files you trust (see the warning above).

## Troubleshooting

| Symptom | Fix |
|---------|-----|
| `ConfigurationError ... missing ... DATABASE_USER` | Create `backend/.env` from `.env.example` and fill in every value. |
| Health shows `"database": "unavailable"` | Is MySQL running? Are the user, password and database in `.env` correct? Did you create `modelbench_db`? |
| Startup log says `Missing tables` | Run `python -m app.database.init_db` from `backend`. |
| `Access denied for user` | Re-run the `GRANT` statements from step 2. |
| Frontend says "Cannot reach the ModelBench API" | Start the backend on port 8000, or set `VITE_API_BASE_URL` in `frontend/.env`. |
| Tests are all skipped | Create `modelbench_test_db` and grant your user access to it. |
| PowerShell blocks `Activate.ps1` | Run `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`. |

## Future improvements

- k-fold cross-validation and simple hyperparameter search.
- Background job queue for long trainings, with progress updates.
- Feature importance and model explanation charts.
- More models (gradient boosting, SVM, k-NN) and class-imbalance handling.
- Prediction page: upload new rows and get predictions from a saved pipeline.
- Delete/archive for datasets and experiments.
- Alembic migrations, user accounts, and Docker Compose for one-command setup.
