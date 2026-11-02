export const METRIC_LABELS = {
  accuracy: 'Accuracy',
  precision: 'Precision',
  recall: 'Recall',
  f1: 'F1-score',
  roc_auc: 'ROC-AUC',
  mae: 'MAE',
  mse: 'MSE',
  rmse: 'RMSE',
  r2: 'R²',
}

export const TASK_METRICS = {
  classification: ['accuracy', 'precision', 'recall', 'f1', 'roc_auc'],
  regression: ['mae', 'mse', 'rmse', 'r2'],
}

export const LOWER_IS_BETTER = new Set(['mae', 'mse', 'rmse'])

export function formatMetric(value, digits = 4) {
  if (value === null || value === undefined || Number.isNaN(value)) return 'N/A'
  const abs = Math.abs(value)
  if (abs !== 0 && (abs >= 1e6 || abs < 1e-4)) return value.toExponential(2)
  if (abs >= 1000) return value.toLocaleString(undefined, { maximumFractionDigits: 2 })
  return value.toFixed(digits)
}

export function formatSeconds(value) {
  if (value === null || value === undefined) return 'N/A'
  if (value < 0.001) return '<1 ms'
  if (value < 1) return `${(value * 1000).toFixed(0)} ms`
  return `${value.toFixed(2)} s`
}

export function formatDateTime(value) {
  if (!value) return '—'
  return new Date(value).toLocaleString(undefined, {
    year: 'numeric',
    month: 'short',
    day: 'numeric',
    hour: '2-digit',
    minute: '2-digit',
  })
}

export function formatBytes(bytes) {
  if (bytes < 1024) return `${bytes} B`
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`
  return `${(bytes / (1024 * 1024)).toFixed(2)} MB`
}

export function formatTask(task) {
  return task ? task.charAt(0).toUpperCase() + task.slice(1) : ''
}

const MODEL_NAMES = {
  logistic_regression: 'Logistic Regression',
  decision_tree_classifier: 'Decision Tree',
  random_forest_classifier: 'Random Forest',
  linear_regression: 'Linear Regression',
  decision_tree_regressor: 'Decision Tree',
  random_forest_regressor: 'Random Forest',
}

export function shortModelName(name) {
  return MODEL_NAMES[name] || name
}
