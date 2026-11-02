
## Run locally

1. Extract this folder.
2. Open a terminal in `frontend`.
3. Run `npm install` (or `npm ci` if you want to use the lockfile).
4. Copy `.env.example` to `.env` if needed and set `VITE_API_BASE_URL`.
5. Run `npm run dev`.

The frontend continues to use the existing FastAPI endpoints. No backend changes are included.
`node_modules` is intentionally not included; install dependencies on your machine.


## Persistent dataset insights
- Upload automatically saves the initial dataset profile.
- Dashboard now lists recently uploaded datasets and links to their analysis pages.
- Dataset analysis pages show saved overview, statistics, missing values, correlation and recommendations.
- Users can inspect prior analysis versions and regenerate/save a new version.
- Existing upload and experiment configuration flows are retained.
