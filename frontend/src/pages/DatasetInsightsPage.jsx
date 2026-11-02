import { useMemo, useState } from 'react'
import { Link } from 'react-router-dom'
import { useAsync } from '../hooks/useAsync'
import { listDatasets } from '../services/api'
import { EmptyState, ErrorMessage, LoadingState, PageHeader } from '../components/common'

async function loadDatasets() {
  return listDatasets(200)
}

export default function DatasetInsightsPage() {
  const { data, loading, error, reload } = useAsync(loadDatasets, [])
  const [search, setSearch] = useState('')
  const datasets = data?.items || []
  const filtered = useMemo(() => {
    const query = search.trim().toLowerCase()
    if (!query) return datasets
    return datasets.filter((dataset) => dataset.original_filename.toLowerCase().includes(query))
  }, [datasets, search])

  return (
    <div className="page">
      <PageHeader
        title="Dataset Insights"
        description="Browse your uploaded datasets and open their saved analysis reports. Each upload automatically saves an initial insight snapshot."
        actions={<Link className="btn btn-primary" to="/upload">＋ Upload dataset</Link>}
      />

      <section className="card recent-card">
        <div className="card-header">
          <div><span className="eyebrow">YOUR DATA LIBRARY</span><h2>Choose a dataset</h2><p>{data ? `${data.total} dataset${data.total === 1 ? '' : 's'} available for analysis` : 'Loading your saved datasets…'}</p></div>
          <label className="field" style={{ minWidth: 220 }}>
            <span className="muted small">Search datasets</span>
            <input className="input" type="search" placeholder="Search by filename…" value={search} onChange={(event) => setSearch(event.target.value)} />
          </label>
        </div>

        {loading && <LoadingState message="Loading your datasets…" />}
        <ErrorMessage message={error} onRetry={reload} />
        {!loading && data && filtered.length === 0 && (
          <EmptyState
            title={datasets.length ? 'No matching datasets' : 'No datasets uploaded yet'}
            description={datasets.length ? 'Try a different filename.' : 'Upload a CSV file to generate and save its first dataset insights.'}
            action={!datasets.length && <Link className="btn btn-primary" to="/upload">Upload your first dataset</Link>}
          />
        )}
        {!loading && filtered.length > 0 && (
          <div className="dataset-library-list">
            {filtered.map((dataset) => (
              <div className="dataset-library-item" key={dataset.id}>
                <div className="dataset-library-icon">▤</div>
                <div className="dataset-library-copy">
                  <strong>{dataset.original_filename}</strong>
                  <small>{Number(dataset.row_count).toLocaleString()} rows · {dataset.column_count} columns · Uploaded {new Date(dataset.uploaded_at).toLocaleDateString()}</small>
                </div>
                <span className="badge success">Saved</span>
                <Link className="btn btn-outline btn-sm" to={`/datasets/${dataset.id}`}>View insights ↗</Link>
              </div>
            ))}
          </div>
        )}
      </section>
    </div>
  )
}
