import { useState } from 'react'
import { Link } from 'react-router-dom'
import { useAsync } from '../hooks/useAsync'
import { listExperiments } from '../services/api'
import ExperimentTable from '../components/ExperimentTable'
import { EmptyState, ErrorMessage, LoadingState, PageHeader } from '../components/common'

const PAGE_SIZE = 10

export default function HistoryPage() {
  const [page, setPage] = useState(1)
  const { data, loading, error, reload } = useAsync(() => listExperiments(page, PAGE_SIZE), [page])
  const totalPages = data ? Math.max(1, Math.ceil(data.total / PAGE_SIZE)) : 1

  return (
    <div className="page">
      <PageHeader
        title="Experiment history"
        description="Every experiment is stored in MySQL, so history survives page refreshes and server restarts."
        actions={
          <>
            <button className="btn" onClick={reload} disabled={loading}>Refresh</button>
            <Link className="btn btn-primary" to="/experiments/new">New experiment</Link>
          </>
        }
      />
      <ErrorMessage message={error} onRetry={reload} />

      <section className="card">
        {loading && !data && <LoadingState message="Loading experiments…" />}
        {data && data.total === 0 && (
          <EmptyState
            title="No experiments yet"
            description="Run an experiment and it will appear here."
            action={<Link className="btn btn-primary" to="/experiments/new">Configure experiment</Link>}
          />
        )}
        {data && data.total > 0 && (
          <div className="stack">
            <ExperimentTable experiments={data.items} />
            <div className="spread">
              <span className="muted small">
                {data.total} experiment{data.total === 1 ? '' : 's'} · page {page} of {totalPages}
              </span>
              <div className="row">
                <button className="btn btn-sm" disabled={page <= 1 || loading} onClick={() => setPage((p) => p - 1)}>
                  Previous
                </button>
                <button className="btn btn-sm" disabled={page >= totalPages || loading} onClick={() => setPage((p) => p + 1)}>
                  Next
                </button>
              </div>
            </div>
          </div>
        )}
      </section>
    </div>
  )
}
