import { Link } from 'react-router-dom'
import { listDatasets, listExperiments } from '../services/api'
import { useAsync } from '../hooks/useAsync'
import ExperimentTable from '../components/ExperimentTable'
import { EmptyState, ErrorMessage, LoadingState, StatusBadge } from '../components/common'
import { METRIC_LABELS, formatDateTime, formatMetric, formatTask, shortModelName } from '../utils/format'

async function loadDashboard() {
  const [datasets, experiments] = await Promise.all([listDatasets(6), listExperiments(1, 5)])
  return { datasetCount: datasets.total, datasets: datasets.items, experimentCount: experiments.total, recent: experiments.items }
}

export default function DashboardPage() {
  const { data, loading, error, reload } = useAsync(loadDashboard, [])
  const latest = data?.recent?.[0]
  return (
    <div className="page dashboard-page">
      <section className="welcome-panel">
        <div className="welcome-copy">
          <span className="eyebrow"><span className="sparkle">✦</span> YOUR MACHINE LEARNING WORKSPACE</span>
          <h1>Build. Train. <span>Compare.</span></h1>
          <p>Turn your datasets into insights. Train machine learning models, compare their performance, and find the right fit for your problem.</p>
          <div className="welcome-actions">
            <Link className="btn btn-primary" to="/upload"><span>↑</span> Upload dataset</Link>
            <Link className="btn btn-secondary" to="/experiments/new"><span>▶</span> Start an experiment</Link>
          </div>
        </div>
        <div className="welcome-art" aria-hidden="true">
          <div className="art-orbit orbit-one"></div><div className="art-orbit orbit-two"></div>
          <div className="art-core"><span>ML</span><small>MODEL<br/>LAB</small></div>
          <div className="art-chip chip-a">01 / DATA</div><div className="art-chip chip-b">02 / TRAIN</div><div className="art-chip chip-c">03 / INSIGHT</div>
        </div>
      </section>

      {loading && <LoadingState message="Loading your workspace…" />}
      <ErrorMessage message={error} onRetry={reload} />

      {data && <>
        <div className="section-heading"><div><span className="eyebrow">AT A GLANCE</span><h2>Your workspace overview</h2></div><span className="live-label"><i/> LIVE WORKSPACE DATA</span></div>
        <div className="grid grid-3 overview-grid">
          <div className="card stat-card stat-datasets"><div className="stat-top"><span className="label">DATASETS UPLOADED</span><span className="stat-icon">▤</span></div><div className="value">{data.datasetCount}</div><div className="hint">Datasets available in your workspace</div><div className="stat-bottom"><span className="stat-line"></span><span>DATA LIBRARY</span></div></div>
          <div className="card stat-card stat-experiments"><div className="stat-top"><span className="label">EXPERIMENTS RUN</span><span className="stat-icon">⌁</span></div><div className="value">{data.experimentCount}</div><div className="hint">Training runs saved to your history</div><div className="stat-bottom"><span className="stat-line"></span><span>EXPERIMENT LOG</span></div></div>
          <div className="card stat-card stat-latest"><div className="stat-top"><span className="label">LATEST ACTIVITY</span><span className="stat-icon">◷</span></div><div className="value stat-text">{latest ? formatDateTime(latest.created_at) : '—'}</div><div className="hint">{latest ? `${formatTask(latest.task_type)} · ${latest.dataset_name}` : 'Your latest run will appear here'}</div><div className="stat-bottom"><span className="stat-line"></span><span>RECENT RUN</span></div></div>
        </div>

        <section className="card recent-card" style={{ marginBottom: 22 }}>
          <div className="card-header"><div><span className="eyebrow">DATA LIBRARY</span><h2>Recently uploaded datasets</h2><p>Open a dataset to revisit its saved insights and analysis history.</p></div><Link className="text-link" to="/upload">Upload dataset ↗</Link></div>
          {data.datasets.length === 0 ? <p className="muted">No datasets uploaded yet.</p> : (
            <div className="dataset-library-list">
              {data.datasets.map((dataset) => <div className="dataset-library-item" key={dataset.id}>
                <div className="dataset-library-icon">▤</div>
                <div className="dataset-library-copy"><strong>{dataset.original_filename}</strong><small>{dataset.row_count.toLocaleString()} rows · {dataset.column_count} columns · {new Date(dataset.uploaded_at).toLocaleDateString()}</small></div>
                <Link className="btn btn-outline btn-sm" to={`/datasets/${dataset.id}`}>View insights ↗</Link>
              </div>)}
            </div>
          )}
        </section>

        {latest && <section className="latest-banner">
          <div className="latest-symbol">✦</div><div className="latest-copy"><span className="eyebrow">LATEST EXPERIMENT</span><strong>{latest.best_model_name ? shortModelName(latest.best_model_name) : 'Experiment completed'}</strong><p>{latest.best_model_name ? `${METRIC_LABELS[latest.primary_metric]}: ${formatMetric(latest.best_metric_value)} · ${latest.dataset_name}` : `No valid ${METRIC_LABELS[latest.primary_metric]} was produced.`}</p></div>
          <StatusBadge status={latest.status}/><Link className="btn btn-outline" to={`/experiments/${latest.id}`}>View results <span>↗</span></Link>
        </section>}

        <div className="dashboard-lower">
          <section className="card recent-card">
            <div className="card-header"><div><span className="eyebrow">YOUR WORK</span><h2>Recent experiments</h2><p>Pick up where you left off.</p></div>{data.experimentCount > 0 && <Link className="text-link" to="/history">View all experiments ↗</Link>}</div>
            {data.recent.length === 0 ? <EmptyState title="No experiments yet" description={data.datasetCount === 0 ? 'Upload a CSV dataset to get started. Sample datasets are in the sample_data folder.' : 'You have datasets ready. Configure an experiment to train your first models.'} action={<Link className="btn btn-primary" to={data.datasetCount === 0 ? '/upload' : '/experiments/new'}>{data.datasetCount === 0 ? 'Upload a dataset' : 'Configure experiment'}</Link>} /> : <ExperimentTable experiments={data.recent} compact />}
          </section>
          <aside className="quick-card">
            <span className="eyebrow">QUICK START</span><h2>What would you like to do?</h2><p>Jump straight into your next machine learning task.</p>
            <Link to="/upload" className="quick-action"><span className="quick-icon">＋</span><span><strong>Upload a dataset</strong><small>CSV files · data profiling</small></span><b>↗</b></Link>
            <Link to="/experiments/new" className="quick-action"><span className="quick-icon">⚗</span><span><strong>Run an experiment</strong><small>Train and compare models</small></span><b>↗</b></Link>
            <Link to="/history" className="quick-action"><span className="quick-icon">◷</span><span><strong>Explore history</strong><small>Review previous results</small></span><b>↗</b></Link>
            <div className="quick-foot"><span className="quick-foot-mark">M</span> MODEL BENCH · MACHINE LEARNING, MADE CLEAR.</div>
          </aside>
        </div>
        <section className="workflow-strip"><div><span className="eyebrow">THE WORKFLOW</span><h2>From raw data to a better model.</h2></div><div className="workflow-steps"><div><b>01</b><strong>Upload</strong><small>Bring your CSV data</small></div><i>→</i><div><b>02</b><strong>Configure</strong><small>Choose target & models</small></div><i>→</i><div><b>03</b><strong>Compare</strong><small>Understand performance</small></div></div></section>
      </>}
    </div>
  )
}
