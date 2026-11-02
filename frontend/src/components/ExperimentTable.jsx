import { useNavigate } from 'react-router-dom'
import { StatusBadge } from './common'
import { METRIC_LABELS, formatDateTime, formatMetric, formatTask, shortModelName } from '../utils/format'

export default function ExperimentTable({ experiments, compact = false }) {
  const navigate = useNavigate()

  return (
    <div className="table-wrap">
      <table className="table">
        <thead>
          <tr>
            <th>Dataset</th>
            <th>Target</th>
            <th>Task</th>
            {!compact && <th>Models</th>}
            <th>Best model</th>
            <th className="num">Primary metric</th>
            <th>Status</th>
            <th>Created</th>
            <th />
          </tr>
        </thead>
        <tbody>
          {experiments.map((exp) => (
            <tr key={exp.id} className="clickable" onClick={() => navigate(`/experiments/${exp.id}`)}>
              <td><strong>{exp.dataset_name}</strong></td>
              <td>{exp.target_column}</td>
              <td>{formatTask(exp.task_type)}</td>
              {!compact && <td>{exp.selected_models.map(shortModelName).join(', ')}</td>}
              <td>{exp.best_model_name ? shortModelName(exp.best_model_name) : '—'}</td>
              <td className="num">
                {exp.best_metric_value !== null ? (
                  <>
                    <span className="muted small">{METRIC_LABELS[exp.primary_metric]} </span>
                    {formatMetric(exp.best_metric_value)}
                  </>
                ) : (
                  <span className="null">N/A</span>
                )}
              </td>
              <td><StatusBadge status={exp.status} /></td>
              <td>{formatDateTime(exp.created_at)}</td>
              <td>
                <button
                  className="btn btn-sm"
                  onClick={(e) => {
                    e.stopPropagation()
                    navigate(`/experiments/${exp.id}`)
                  }}
                >
                  Open
                </button>
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}
