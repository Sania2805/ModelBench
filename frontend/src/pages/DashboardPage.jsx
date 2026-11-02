import { Link } from 'react-router-dom'
import { listDatasets, listExperiments } from '../services/api'
import { useAsync } from '../hooks/useAsync'
import ExperimentTable from '../components/ExperimentTable'
import { EmptyState, ErrorMessage, LoadingState, PageHeader, StatCard, StatusBadge } from '../components/common'
import { METRIC_LABELS, formatDateTime, formatMetric, formatTask, shortModelName } from '../utils/format'

async function loadDashboard() {
  const [datasets, experiments] = await Promise.all([listDatasets(1), listExperiments(1, 5)])
  return { datasetCount: datasets.total, experimentCount: experiments.total, recent: experiments.items }
}

export default function DashboardPage() {
  const { data, loading, error, reload } = useAsync(loadDashboard, [])
  const latest = data?.recent?.[0]

  return (
    <div className="page">
      <PageHeader
        title="Welcome to ModelBench"
        description="Upload a CSV, pick a target column and compare classic machine learning models side by side — with real training, honest metrics and downloadable pipelines."
        actions={
          <>
            <Link className="btn" to="/upload">Upload dataset</Link>
            <Link className="btn btn-primary" to="/experiments/new">Start new experiment</Link>
          </>
        }
      />

      {loading && <LoadingState message="Loading your workspace…" />}
      <ErrorMessage message={error} onRetry={reload} />

      {data && (
        <>
          <div className="grid grid-3">
            <StatCard label="Datasets uploaded" value={data.datasetCount} />
            <StatCard label="Experiments run" value={data.experimentCount} />
            <StatCard
              label="Most recent experiment"
              value={latest ? latest.dataset_name : '—'}
              hint={
                latest
                  ? `${formatTask(latest.task_type)} · ${latest.target_column} · ${formatDateTime(latest.created_at)}`
                  : 'No experiments yet'
              }
            />
          </div>

          {latest && (
            <section className="card highlight">
              <div className="spread">
                <div className="stack" style={{ gap: 4 }}>
                  <div className="row">
                    <h2>Latest result</h2>
                    <StatusBadge status={latest.status} />
                  </div>
                  <p className="muted">
                    {latest.best_model_name
                      ? `${shortModelName(latest.best_model_name)} scored ${METRIC_LABELS[latest.primary_metric]} ${formatMetric(latest.best_metric_value)} predicting "${latest.target_column}" in ${latest.dataset_name}.`
                      : `No model produced a valid ${METRIC_LABELS[latest.primary_metric]} for this experiment.`}
                  </p>
                </div>
                <Link className="btn" to={`/experiments/${latest.id}`}>View results</Link>
              </div>
            </section>
          )}

          <section className="card">
            <div className="card-header">
              <h2>Recent experiments</h2>
              {data.experimentCount > 0 && <Link to="/history">View all</Link>}
            </div>
            {data.recent.length === 0 ? (
              <EmptyState
                title="No experiments yet"
                description={
                  data.datasetCount === 0
                    ? 'Start by uploading a CSV dataset. Sample datasets are in the sample_data folder.'
                    : 'You have datasets ready. Configure an experiment to train your first models.'
                }
                action={
                  <Link className="btn btn-primary" to={data.datasetCount === 0 ? '/upload' : '/experiments/new'}>
                    {data.datasetCount === 0 ? 'Upload a dataset' : 'Configure experiment'}
                  </Link>
                }
              />
            ) : (
              <ExperimentTable experiments={data.recent} compact />
            )}
          </section>

          <section className="card">
            <div className="card-header"><h2>How it works</h2></div>
            <div className="grid grid-4">
              {[
                ['1. Upload', 'Upload a CSV. ModelBench validates it and profiles every column.'],
                ['2. Configure', 'Choose the target, the task type, models, test split and seed.'],
                ['3. Train', 'Each model is trained in a leakage-safe scikit-learn pipeline.'],
                ['4. Compare', 'Inspect metrics and charts, then download a report or pipeline.'],
              ].map(([title, text]) => (
                <div key={title} className="stack" style={{ gap: 4 }}>
                  <h3>{title}</h3>
                  <p className="muted">{text}</p>
                </div>
              ))}
            </div>
          </section>
        </>
      )}
    </div>
  )
}
