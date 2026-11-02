export function PageHeader({ title, description, actions }) {
  return (
    <div className="page-header">
      <div>
        <h1>{title}</h1>
        {description && <p>{description}</p>}
      </div>
      {actions && <div className="actions">{actions}</div>}
    </div>
  )
}

export function LoadingState({ message = 'Loading…' }) {
  return (
    <div className="state">
      <div className="spinner" />
      <p>{message}</p>
    </div>
  )
}

export function ErrorMessage({ message, onRetry }) {
  if (!message) return null
  return (
    <div className="notice error" role="alert">
      <div className="spread">
        <span>{message}</span>
        {onRetry && (
          <button className="btn btn-sm" onClick={onRetry}>
            Try again
          </button>
        )}
      </div>
    </div>
  )
}

export function EmptyState({ title, description, action }) {
  return (
    <div className="state">
      <h3>{title}</h3>
      {description && <p>{description}</p>}
      {action}
    </div>
  )
}

export function Notice({ type = 'info', title, items, children }) {
  if (items && items.length === 0 && !children) return null
  return (
    <div className={`notice ${type}`}>
      {title && <div className="notice-title">{title}</div>}
      {children}
      {items && items.length > 0 && (
        <ul>
          {items.map((item, i) => (
            <li key={i}>{item}</li>
          ))}
        </ul>
      )}
    </div>
  )
}

const STATUS_STYLES = {
  completed: 'success',
  success: 'success',
  partial: 'warning',
  running: 'info',
  failed: 'danger',
}

export function StatusBadge({ status }) {
  return <span className={`badge ${STATUS_STYLES[status] || ''}`}>{status}</span>
}

export function StatCard({ label, value, hint }) {
  return (
    <div className="card stat-card">
      <div className="label">{label}</div>
      <div className="value">{value}</div>
      {hint && <div className="hint">{hint}</div>}
    </div>
  )
}
