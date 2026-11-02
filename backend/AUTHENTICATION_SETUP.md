# ModelBench authentication setup

## Configuration

1. Keep your existing `backend/.env` file private. Do not replace it with `.env.example`.
2. Add these new variables to the existing `backend/.env` without changing your database settings:

```env
AUTH_SECRET_KEY=paste_a_new_random_secret_here
AUTH_TOKEN_EXPIRE_MINUTES=60
AUTH_COOKIE_SECURE=false
AUTH_COOKIE_SAMESITE=lax
```

Generate a secret in PowerShell from the `backend` folder:

```powershell
python -c "import secrets; print(secrets.token_urlsafe(48))"
```

Copy the generated value into `AUTH_SECRET_KEY`. Do not commit `.env` to GitHub.

## Database

From the `backend` folder, after activating your virtual environment and installing requirements:

```powershell
python -m app.database.init_db
```

This adds the `users` table through SQLAlchemy `create_all()`. It does not drop or alter the existing datasets, experiments, or model_results tables.

## Run the backend

```powershell
python -m uvicorn app.main:app --reload --port 8000
```

Authentication endpoints:
- `POST /api/auth/register`
- `POST /api/auth/login`
- `GET /api/auth/me`
- `POST /api/auth/logout`

The ML dataset and experiment endpoints now require a valid login cookie. The health endpoint remains public.

## Important scope note

This release adds accounts and requires authentication for the ML API, but existing datasets and experiments are still in the shared application database. It does not yet implement per-user data ownership or workspace isolation. Add that as a separate migration before using this as a multi-tenant production service.
