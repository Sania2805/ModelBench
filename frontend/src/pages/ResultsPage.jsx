import { useState } from 'react'
import { Link, useParams } from 'react-router-dom'
import { useAsync } from '../hooks/useAsync'
import { downloadModel, downloadReport, getErrorMessage, getExperiment } from '../services/api'
import { ErrorMessage, LoadingState, Notice, PageHeader, StatCard, StatusBadge } from '../components/common'
import { ActualVsPredictedChart, ConfusionMatrix, MetricBarChart } from '../components/charts'
import {
  LOWER_IS_BETTER,
  METRIC_LABELS,
  TASK_METRICS,
  formatDateTime,
  formatMetric,
  formatSeconds,
  formatTask,
} from '../utils/format'

function useDownloads() {
  const [busy, setBusy] = useState(null)
  const [error, setError] = useState(null)
  const run = async (key, fn) => {
    setBusy(key)
    setError(null)
    try {
      await fn()
    } catch (err) {
      setError(getErrorMessage(err))
    } finally {
      setBusy(null)
    }
  }
  return { busy, error, run }
}

function ComparisonTable({ experiment, onDownload, busy }) {
  const metrics = TASK_METRICS[experiment.task_type]
  return (
    <div className="table-wrap">
      <table className="table">
        <thead>
          <tr>
            <th>Model</th>
            {metrics.map((m) => (
              <th key={m} className="num">
                {METRIC_LABELS[m]}
                {m === experiment.primary_metric && ' ★'}
              </th>
            ))}
            <th className="num">Train time</th>
            <th className="num">Predict time</th>
            <th>Pipeline</th>
          </tr>
        </thead>
        <tbody>
          {experiment.results.map((r) => (
            <tr key={r.id} className={r.is_best ? 'best-row' : ''}>
              <td>
                <strong>{r.display_name}</strong> {r.is_best && <span className="badge info">Best</span>}
                {r.status === 'failed' && <span className="badge danger">failed</span>}
              </td>
              {r.status === 'success' ? (
                metrics.map((m) => (
                  <td key={m} className={`num ${r.metrics[m] === null ? 'null' : ''}`} title={r.metric_notes?.[m] || ''}>
                    {formatMetric(r.metrics[m])}
                  </td>
                ))
              ) : (
                <td colSpan={metrics.length} className="null">Training failed — see details below</td>
              )}
              <td className="num">{formatSeconds(r.training_time_seconds)}</td>
              <td className="num">{formatSeconds(r.prediction_time_seconds)}</td>
              <td>
                <button
                  className="btn btn-sm"
                  disabled={!r.has_artifact || busy === r.model_name}
                  onClick={() => onDownload(r.model_name)}
                >
                  {busy === r.model_name ? 'Downloading…' : 'Download'}
                </button>
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}

function MetricPill({ label, value, note }) {
  return (
    <div className="metric-pill" title={note || ''}>
      <div className="k">{label}</div>
      <div className={`v ${value === null || value === undefined ? 'na' : ''}`}>{formatMetric(value)}</div>
    </div>
  )
}

function PerClassTable({ rows }) {
  if (!rows || rows.length === 0) return null
  return (
    <div className="table-wrap">
      <table className="table">
        <thead>
          <tr>
            <th>Class</th>
            <th className="num">Precision</th>
            <th className="num">Recall</th>
            <th className="num">F1</th>
            <th className="num">Support</th>
          </tr>
        </thead>
        <tbody>
          {rows.map((row) => (
            <tr key={row.label}>
              <td>{row.label}</td>
              <td className="num">{formatMetric(row.precision, 3)}</td>
              <td className="num">{formatMetric(row.recall, 3)}</td>
              <td className="num">{formatMetric(row.f1, 3)}</td>
              <td className="num">{row.support}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}

function ModelDetails({ result, experiment, onDownload, busy }) {
  const metrics = TASK_METRICS[experiment.task_type]
  const notes = Object.entries(result.metric_notes || {})

  return (
    <section className={`card model-section ${result.is_best ? 'highlight' : ''}`} id={`model-${result.model_name}`}>
      <div className="spread">
        <div className="row">
          <h2>{result.display_name}</h2>
          {result.is_best && <span className="badge info">Best by {METRIC_LABELS[experiment.primary_metric]}</span>}
          <StatusBadge status={result.status} />
        </div>
        <button
          className="btn btn-sm"
          disabled={!result.has_artifact || busy === result.model_name}
          onClick={() => onDownload(result.model_name)}
        >
          {busy === result.model_name ? 'Downloading…' : 'Download pipeline (.joblib)'}
        </button>
      </div>

      {result.status === 'failed' ? (
        <Notice type="error" title="This model failed to train">{result.error_message}</Notice>
      ) : (
        <>
          <div className="metric-pills">
            {metrics.map((m) => (
              <MetricPill key={m} label={METRIC_LABELS[m]} value={result.metrics[m]} note={result.metric_notes?.[m]} />
            ))}
          </div>
          <div className="meta-row">
            <span>Training: {formatSeconds(result.training_time_seconds)}</span>
            <span>Prediction: {formatSeconds(result.prediction_time_seconds)}</span>
            <span className="mono">
              {Object.entries(result.hyperparameters || {})
                .map(([k, v]) => `${k}=${v === null ? 'None' : v}`)
                .join(', ')}
            </span>
          </div>

          {experiment.task_type === 'classification' ? (
            <div className="grid grid-2">
              <div className="stack">
                <h3>Confusion matrix</h3>
                <ConfusionMatrix data={result.chart_data?.confusion_matrix} />
              </div>
              <div className="stack">
                <h3>Per-class metrics</h3>
                <PerClassTable rows={result.chart_data?.per_class} />
              </div>
            </div>
          ) : (
            <div className="stack">
              <h3>Actual vs predicted (test set)</h3>
              <p className="muted small">Points on the dashed diagonal are perfect predictions.</p>
              <ActualVsPredictedChart points={result.chart_data?.actual_vs_predicted} />
            </div>
          )}

          {notes.length > 0 && (
            <Notice type="info" title="Notes">
              <ul>
                {notes.map(([key, note]) => (
                  <li key={key}>
                    <strong>{METRIC_LABELS[key] || key.replace(/_/g, ' ')}:</strong> {note}
                  </li>
                ))}
              </ul>
            </Notice>
          )}
        </>
      )}
    </section>
  )
}

export default function ResultsPage() {
  const { experimentId } = useParams()
  const { data: experiment, loading, error, reload } = useAsync(() => getExperiment(experimentId), [experimentId])
  const [chartMetric, setChartMetric] = useState(null)
  const downloads = useDownloads()

  if (loading) return <div className="page"><LoadingState message="Loading experiment…" /></div>
  if (error) {
    return (
      <div className="page">
        <PageHeader title="Experiment results" />
        <ErrorMessage message={error} onRetry={reload} />
        <Link to="/history">Back to experiment history</Link>
      </div>
    )
  }

  const meta = experiment.run_metadata || {}
  const best = experiment.results.find((r) => r.is_best)
  const metric = chartMetric || experiment.primary_metric
  const metricOptions = TASK_METRICS[experiment.task_type]
  const hasResults = experiment.results.some((r) => r.status === 'success')
  const handleModelDownload = (name) =>
    downloads.run(name, () => downloadModel(experiment.id, name))

  return (
    <div className="page">
      <PageHeader
        title="Experiment results"
        description={`${formatTask(experiment.task_type)} on ${experiment.dataset.original_filename}, predicting "${experiment.target_column}". Run ${formatDateTime(experiment.created_at)}.`}
        actions={
          <>
            <Link className="btn" to={`/experiments/new?datasetId=${experiment.dataset.id}`}>New experiment on this data</Link>
            <button
              className="btn btn-primary"
              disabled={!hasResults || downloads.busy === 'report'}
              onClick={() => downloads.run('report', () => downloadReport(experiment.id))}
            >
              {downloads.busy === 'report' ? 'Preparing report…' : 'Download report'}
            </button>
          </>
        }
      />
      <ErrorMessage message={downloads.error} />

      {experiment.status === 'failed' && (
        <Notice type="error" title="This experiment did not produce any valid results">
          {experiment.error_message}
        </Notice>
      )}
      {experiment.status === 'partial' && (
        <Notice type="warning">Some models failed to train. Results below include only the models that succeeded.</Notice>
      )}

      <div className="grid grid-4">
        <StatCard label="Status" value={<StatusBadge status={experiment.status} />} hint={`Finished ${formatDateTime(experiment.completed_at)}`} />
        <StatCard
          label="Best model"
          value={best ? best.display_name : '—'}
          hint={best ? `${METRIC_LABELS[experiment.primary_metric]} ${formatMetric(best.metrics[experiment.primary_metric])}` : 'No valid primary metric'}
        />
        <StatCard
          label="Train / validation / test"
          value={`${meta.model_fit_rows ?? '?'} / ${meta.validation_rows ?? '?'} / ${meta.test_rows ?? '?'}`}
          hint={`${Math.round(experiment.test_size * 100)}% final test · ${meta.split_strategy === 'chronological' ? 'chronological' : 'random'} split`}
        />
        <StatCard label="Random seed" value={experiment.random_seed} hint={`${experiment.results.length} model(s) compared`} />
      </div>

      <section className="card">
        <div className="card-header"><h2>Dataset and configuration</h2></div>
        <dl className="kv">
          <dt>Dataset</dt>
          <dd>{experiment.dataset.original_filename} ({experiment.dataset.row_count} rows × {experiment.dataset.column_count} columns)</dd>
          <dt>Target column</dt>
          <dd>{experiment.target_column}</dd>
          <dt>Task type</dt>
          <dd>{formatTask(experiment.task_type)}</dd>
          <dt>Primary metric</dt>
          <dd>
            {METRIC_LABELS[experiment.primary_metric]} ({experiment.primary_metric_higher_is_better ? 'higher' : 'lower'} is better)
          </dd>
          {meta.numeric_features && (
            <>
              <dt>Numerical features</dt>
              <dd>{meta.numeric_features.join(', ') || '—'}</dd>
              <dt>Categorical features</dt>
              <dd>{meta.categorical_features.join(', ') || '—'}</dd>
            </>
          )}
          {meta.class_labels && (
            <>
              <dt>Classes</dt>
              <dd>{meta.class_labels.join(', ')}</dd>
              {experiment.positive_class && <><dt>Positive class</dt><dd>{experiment.positive_class}</dd></>}
            </>
          )}
          {meta.datetime_features?.length > 0 && <><dt>Date/time features</dt><dd>{meta.datetime_features.join(', ')}</dd></>}
          {meta.time_column && <><dt>Chronological order</dt><dd>{meta.time_column}</dd></>}
          {meta.excluded_columns?.length > 0 && <><dt>Excluded features</dt><dd>{meta.excluded_columns.join(', ')}</dd></>}
          {meta.balance_classes && <><dt>Class balancing</dt><dd>Balanced class weights enabled</dd></>}
        </dl>
      </section>

      <Notice type="warning" title="Data warnings" items={meta.warnings || []} />

      {hasResults && (
        <>
          <section className="card">
            <div className="card-header">
              <h2>Model comparison</h2>
              <p>★ marks the primary metric. Models are ranked on validation data; the displayed scores below are from the untouched test set. Hover N/A values for an explanation.</p>
            </div>
            <ComparisonTable experiment={experiment} onDownload={handleModelDownload} busy={downloads.busy} />
          </section>

          <section className="card">
            <div className="card-header">
              <div>
                <h2>{METRIC_LABELS[metric]} by model</h2>
                <p>{LOWER_IS_BETTER.has(metric) ? 'Lower is better.' : 'Higher is better.'}</p>
              </div>
              <select className="select metric-select" value={metric} onChange={(e) => setChartMetric(e.target.value)}>
                {metricOptions.map((m) => (
                  <option key={m} value={m}>{METRIC_LABELS[m]}</option>
                ))}
              </select>
            </div>
            <MetricBarChart data={experiment.comparison_chart} metric={metric} bestModel={best?.display_name} />
          </section>

          <Notice type="info">
            The best model is selected using validation F1 (classification) or validation RMSE (regression). The metrics shown in this comparison are calculated separately on the untouched test set. A single split still gives an estimate rather than a guarantee of performance on new data; try another random seed to check stability.
          </Notice>
        </>
      )}

      {experiment.results.map((result) => (
        <ModelDetails
          key={result.id}
          result={result}
          experiment={experiment}
          onDownload={handleModelDownload}
          busy={downloads.busy}
        />
      ))}
    </div>
  )
}
