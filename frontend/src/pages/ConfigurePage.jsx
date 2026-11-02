import { useEffect, useMemo, useState } from 'react'
import { Link, useNavigate, useSearchParams } from 'react-router-dom'
import { useAsync } from '../hooks/useAsync'
import {
  createExperiment,
  getAvailableModels,
  getDatasetProfile,
  getErrorMessage,
  getTargetClasses,
  listDatasets,
} from '../services/api'
import { EmptyState, ErrorMessage, LoadingState, Notice, PageHeader } from '../components/common'
import { formatDateTime } from '../utils/format'

const TASKS = [
  { value: 'classification', label: 'Classification', hint: 'Predict a category (e.g. yes / no, species).' },
  { value: 'regression', label: 'Regression', hint: 'Predict a number (e.g. price, temperature).' },
]

function targetAdvice(column, taskType) {
  if (!column || !taskType) return { error: null, warning: null }
  if (taskType === 'regression' && column.kind !== 'numeric') {
    return { error: `"${column.name}" contains text values. Regression needs a numeric target — choose classification.`, warning: null }
  }
  if (column.unique_count < 2) {
    return { error: `"${column.name}" has only one distinct value, so there is nothing to predict.`, warning: null }
  }
  if (taskType === 'classification' && column.kind === 'numeric' && column.unique_count > 20) {
    return { error: null, warning: `"${column.name}" has ${column.unique_count} distinct numeric values. If it is a quantity, regression is likely the better choice.` }
  }
  if (taskType === 'regression' && column.unique_count <= 10) {
    return { error: null, warning: `"${column.name}" has only ${column.unique_count} distinct values. If these are categories, use classification.` }
  }
  if (column.likely_identifier) {
    return { error: null, warning: `"${column.name}" looks like an identifier, which is rarely a meaningful target.` }
  }
  return { error: null, warning: null }
}

// The profile endpoint provides a small preview rather than a separate date schema.
// Use column names/dtypes plus preview values to suggest likely date/time features.
function findDateColumns(columns, preview = []) {
  return columns.filter((column) => {
    const name = column.name.toLowerCase()
    if (/date|time|timestamp|datetime|created_at|updated_at|_at$/.test(name) || /datetime|date/.test(column.dtype.toLowerCase())) return true
    if (column.kind !== 'categorical' || !preview.length) return false
    const values = preview.map((row) => row[column.name]).filter((value) => value !== null && value !== undefined && value !== '')
    if (!values.length) return false
    const parseable = values.filter((value) => typeof value === 'string' && !Number.isNaN(Date.parse(value))).length
    return parseable / values.length >= 0.8
  })
}

export default function ConfigurePage() {
  const navigate = useNavigate()
  const [searchParams, setSearchParams] = useSearchParams()
  const datasetId = searchParams.get('datasetId') || ''

  const [target, setTarget] = useState('')
  const [taskType, setTaskType] = useState('')
  const [selectedModels, setSelectedModels] = useState([])
  const [testSize, setTestSize] = useState(0.2)
  const [seed, setSeed] = useState('42')
  const [positiveClass, setPositiveClass] = useState('')
  const [targetClasses, setTargetClasses] = useState(null)
  const [classesLoading, setClassesLoading] = useState(false)
  const [classesError, setClassesError] = useState(null)
  const [excludedColumns, setExcludedColumns] = useState([])
  const [balanceClasses, setBalanceClasses] = useState(false)
  const [splitStrategy, setSplitStrategy] = useState('random')
  const [timeColumn, setTimeColumn] = useState('')
  const [submitting, setSubmitting] = useState(false)
  const [submitError, setSubmitError] = useState(null)

  const datasets = useAsync(() => listDatasets(100), [])
  const profile = useAsync(() => (datasetId ? getDatasetProfile(datasetId) : Promise.resolve(null)), [datasetId])
  const models = useAsync(() => (taskType ? getAvailableModels(taskType) : Promise.resolve([])), [taskType])

  useEffect(() => {
    setTarget('')
    setSubmitError(null)
    setExcludedColumns([])
    setPositiveClass('')
    setTargetClasses(null)
    setSplitStrategy('random')
    setTimeColumn('')
  }, [datasetId])

  useEffect(() => {
    setSelectedModels(models.data ? models.data.map((m) => m.name) : [])
  }, [models.data])

  useEffect(() => {
    let active = true
    setTargetClasses(null)
    setPositiveClass('')
    setClassesError(null)
    if (!datasetId || !target || taskType !== 'classification') {
      setClassesLoading(false)
      return () => { active = false }
    }
    setClassesLoading(true)
    getTargetClasses(datasetId, target)
      .then((data) => {
        if (!active) return
        setTargetClasses(data)
        if (data.positive_class_selection && data.classes?.length) {
          setPositiveClass(data.classes[data.classes.length - 1])
        }
      })
      .catch((error) => {
        if (active) setClassesError(getErrorMessage(error))
      })
      .finally(() => {
        if (active) setClassesLoading(false)
      })
    return () => { active = false }
  }, [datasetId, target, taskType])

  const columns = profile.data?.columns || []
  const targetColumn = columns.find((c) => c.name === target)
  const featureColumns = columns.filter((column) => column.name !== target)
  const dateColumns = useMemo(
    () => findDateColumns(featureColumns, profile.data?.preview || []),
    [columns, target, profile.data?.preview],
  )
  const eligibleDateColumns = dateColumns.filter((column) => !excludedColumns.includes(column.name))
  const advice = targetAdvice(targetColumn, taskType)
  const seedNumber = Number(seed)
  const seedValid = seed !== '' && Number.isInteger(seedNumber) && seedNumber >= 0 && seedNumber <= 2147483647
  const rowCount = profile.data?.dataset.row_count || 0
  const testRows = Math.ceil(rowCount * testSize)
  const remainingRows = Math.max(0, rowCount - testRows)
  const validationRows = Math.ceil(remainingRows * 0.2)
  const trainingRows = Math.max(0, remainingRows - validationRows)

  const problems = useMemo(() => {
    const list = []
    if (!datasetId) list.push('Select a dataset.')
    if (!target) list.push('Select the target column you want to predict.')
    if (!taskType) list.push('Choose classification or regression.')
    if (taskType && selectedModels.length === 0) list.push('Select at least one model.')
    if (!seedValid) list.push('Random seed must be a whole number between 0 and 2147483647.')
    if (advice.error) list.push(advice.error)
    if (splitStrategy === 'chronological' && !timeColumn) list.push('Choose a date/time feature for chronological splitting.')
    if (splitStrategy === 'chronological' && excludedColumns.includes(timeColumn)) list.push('The time column cannot also be excluded from features.')
    return list
  }, [datasetId, target, taskType, selectedModels, seedValid, advice.error, splitStrategy, timeColumn, excludedColumns])

  const toggleModel = (name) =>
    setSelectedModels((current) => (current.includes(name) ? current.filter((n) => n !== name) : [...current, name]))

  const toggleExcludedColumn = (name) =>
    setExcludedColumns((current) => current.includes(name) ? current.filter((item) => item !== name) : [...current, name])

  const handleSubmit = async (event) => {
    event.preventDefault()
    if (problems.length > 0) return
    setSubmitting(true)
    setSubmitError(null)
    try {
      const experiment = await createExperiment({
        dataset_id: datasetId,
        target_column: target,
        task_type: taskType,
        model_names: selectedModels,
        test_size: testSize,
        random_seed: seedNumber,
        ...(taskType === 'classification' && targetClasses?.positive_class_selection && positiveClass
          ? { positive_class: positiveClass } : {}),
        exclude_columns: excludedColumns,
        balance_classes: taskType === 'classification' && balanceClasses,
        split_strategy: splitStrategy,
        ...(splitStrategy === 'chronological' ? { time_column: timeColumn } : {}),
      })
      navigate(`/experiments/${experiment.id}`)
    } catch (err) {
      setSubmitError(getErrorMessage(err))
    } finally {
      setSubmitting(false)
    }
  }

  if (datasets.loading) return <div className="page"><LoadingState /></div>

  if (datasets.data && datasets.data.total === 0) {
    return (
      <div className="page">
        <PageHeader title="Configure experiment" />
        <section className="card">
          <EmptyState
            title="No datasets yet"
            description="Upload a CSV file before configuring an experiment."
            action={<Link className="btn btn-primary" to="/upload">Upload dataset</Link>}
          />
        </section>
      </div>
    )
  }

  return (
    <div className="page">
      <PageHeader
        title="Configure experiment"
        description="Models are ranked on validation data. Final comparison metrics are calculated on an untouched test set."
      />
      <ErrorMessage message={datasets.error} onRetry={datasets.reload} />

      <form className="stack" style={{ gap: 20 }} onSubmit={handleSubmit}>
        <section className="card stack">
          <h2>1. Data</h2>
          <div className="grid grid-2">
            <div className="field">
              <label htmlFor="dataset">Dataset</label>
              <select id="dataset" className="select" value={datasetId}
                onChange={(e) => setSearchParams(e.target.value ? { datasetId: e.target.value } : {})}>
                <option value="">Select a dataset…</option>
                {datasets.data?.items.map((d) => (
                  <option key={d.id} value={d.id}>{d.original_filename} — {d.row_count} rows · {formatDateTime(d.uploaded_at)}</option>
                ))}
              </select>
              {profile.data && <span className="help">{profile.data.dataset.row_count} rows · {profile.data.dataset.column_count} columns · {profile.data.total_missing_values} missing values</span>}
            </div>
            <div className="field">
              <label htmlFor="target">Target column</label>
              <select id="target" className="select" value={target} disabled={!profile.data}
                onChange={(e) => setTarget(e.target.value)}>
                <option value="">{profile.loading ? 'Loading columns…' : 'Select the column to predict…'}</option>
                {columns.map((c) => <option key={c.name} value={c.name}>{c.name} ({c.kind}, {c.unique_count} unique)</option>)}
              </select>
              {targetColumn && <span className="help">Other non-excluded columns are used as features{targetColumn.missing_count > 0 && `; ${targetColumn.missing_count} rows with a missing target will be dropped`}.</span>}
            </div>
          </div>
          <ErrorMessage message={profile.error} onRetry={profile.reload} />
        </section>

        <section className="card stack">
          <h2>2. Task type</h2>
          <p className="muted">ModelBench never guesses the task — choose the one that matches your target.</p>
          <div className="segmented" role="radiogroup" aria-label="Task type">
            {TASKS.map((task) => (
              <button type="button" key={task.value} role="radio" aria-checked={taskType === task.value}
                className={taskType === task.value ? 'selected' : ''} onClick={() => setTaskType(task.value)}>{task.label}</button>
            ))}
          </div>
          {taskType && <span className="help muted">{TASKS.find((t) => t.value === taskType).hint}</span>}
          {advice.error && <Notice type="error">{advice.error}</Notice>}
          {advice.warning && <Notice type="warning">{advice.warning}</Notice>}
          {taskType === 'classification' && target && (
            <div className="field">
              <label htmlFor="positive-class">Positive class (binary classification)</label>
              {classesLoading ? <span className="help">Loading target labels…</span> : targetClasses?.positive_class_selection ? (
                <>
                  <select id="positive-class" className="select" value={positiveClass} onChange={(e) => setPositiveClass(e.target.value)}>
                    {targetClasses.classes.map((label) => <option key={label} value={label}>{label}</option>)}
                  </select>
                  <span className="help">Precision, recall, F1-score and ROC-AUC will treat this label as the positive class.</span>
                </>
              ) : targetClasses ? <span className="help">This target has {targetClasses.classes?.length ?? 0} classes. Positive-class selection applies only to binary classification; multiclass metrics use weighted averaging.</span> : null}
              <ErrorMessage message={classesError} />
            </div>
          )}
        </section>

        <section className="card stack">
          <div className="spread">
            <h2>3. Models</h2>
            {models.data?.length > 0 && <span className="muted small">{selectedModels.length} of {models.data.length} selected</span>}
          </div>
          {!taskType && <p className="muted">Choose a task type to see compatible models.</p>}
          {models.loading && taskType && <LoadingState message="Loading models…" />}
          <ErrorMessage message={models.error} onRetry={models.reload} />
          <div className="option-list">
            {models.data?.map((model) => {
              const checked = selectedModels.includes(model.name)
              return (
                <label key={model.name} className={`option ${checked ? 'checked' : ''}`}>
                  <input type="checkbox" checked={checked} onChange={() => toggleModel(model.name)} />
                  <div>
                    <div className="title">{model.display_name}</div>
                    <div className="desc">{model.description}</div>
                    <div className="desc mono">{Object.entries(model.hyperparameters).filter(([k]) => k !== 'random_state' && k !== 'n_jobs').map(([k, v]) => `${k}=${v === null ? 'None' : v}`).join(', ')}</div>
                  </div>
                </label>
              )
            })}
          </div>
        </section>

        <section className="card stack">
          <h2>4. Feature handling</h2>
          <p className="muted">Date/time columns are automatically converted into useful calendar features when detected.</p>
          {featureColumns.length > 0 ? (
            <div className="option-list">
              {featureColumns.map((column) => (
                <label key={column.name} className={`option ${excludedColumns.includes(column.name) ? 'checked' : ''}`}>
                  <input type="checkbox" checked={excludedColumns.includes(column.name)} onChange={() => toggleExcludedColumn(column.name)} />
                  <div>
                    <div className="title">Exclude {column.name} {column.likely_identifier && <span className="badge warning">Likely ID</span>}</div>
                    <div className="desc">{column.kind} · {column.unique_count} unique value(s){column.likely_identifier ? ' · Consider excluding this identifier.' : ''}</div>
                  </div>
                </label>
              ))}
            </div>
          ) : <p className="help">Choose a target column to configure features.</p>}
          <label className="check-row">
            <input type="checkbox" checked={balanceClasses} onChange={(e) => setBalanceClasses(e.target.checked)} disabled={taskType !== 'classification'} />
            <span><strong>Balance classes using class weights</strong><span className="help">Applies class_weight=balanced where the selected estimator supports it. Does not resample the dataset.</span></span>
          </label>
        </section>

        <section className="card stack">
          <h2>5. Evaluation settings</h2>
          <div className="grid grid-2">
            <div className="field">
              <label htmlFor="test-size">Final test set: {Math.round(testSize * 100)}%</label>
              <input id="test-size" type="range" min="0.1" max="0.5" step="0.05" value={testSize} onChange={(e) => setTestSize(Number(e.target.value))} />
              <span className="help">Approximately {trainingRows} training, {validationRows} validation and {testRows} final test rows (before rows with missing targets are dropped).</span>
            </div>
            <div className="field">
              <label htmlFor="seed">Random seed</label>
              <input id="seed" className="input" type="number" min="0" step="1" value={seed} onChange={(e) => setSeed(e.target.value)} />
              {seedValid ? <span className="help">The same seed reproduces the random split.</span> : <span className="field-error">Enter a whole number from 0 to 2147483647.</span>}
            </div>
          </div>
          <div className="grid grid-2">
            <div className="field">
              <label htmlFor="split-strategy">Split strategy</label>
              <select id="split-strategy" className="select" value={splitStrategy} onChange={(e) => { setSplitStrategy(e.target.value); setTimeColumn('') }}>
                <option value="random">Random split</option>
                <option value="chronological" disabled={eligibleDateColumns.length === 0}>Chronological split (time-series data)</option>
              </select>
              <span className="help">Use chronological splitting when future rows must be kept out of training.</span>
            </div>
            {splitStrategy === 'chronological' && (
              <div className="field">
                <label htmlFor="time-column">Date/time column</label>
                <select id="time-column" className="select" value={timeColumn} onChange={(e) => setTimeColumn(e.target.value)}>
                  <option value="">Choose a date/time feature…</option>
                  {eligibleDateColumns.map((column) => <option key={column.name} value={column.name}>{column.name}</option>)}
                </select>
                <span className="help">Rows are ordered by this column: earlier rows train, later rows validate and test.</span>
              </div>
            )}
          </div>
          {splitStrategy === 'chronological' && eligibleDateColumns.length === 0 && <Notice type="warning">No date/time feature was detected in the dataset preview. Check your date column format or use a random split.</Notice>}
          <Notice type="info">
            ModelBench uses training data to fit models, validation data to select the best model (highest F1 for classification or lowest RMSE for regression), and the untouched test set for the final displayed metrics. Preprocessing is fitted without using validation or test data.
          </Notice>
        </section>

        <section className="card">
          {problems.length > 0 && !submitting && <Notice type="warning" title="Before you can start" items={problems} />}
          {submitError && <div style={{ marginTop: 12 }}><ErrorMessage message={submitError} /></div>}
          <div className="spread" style={{ marginTop: problems.length > 0 || submitError ? 16 : 0 }}>
            <span className="muted">{submitting ? `Training ${selectedModels.length} model(s)… larger datasets can take a minute.` : 'Training runs on the server and results are saved to MySQL.'}</span>
            <button className="btn btn-primary" type="submit" disabled={problems.length > 0 || submitting}>{submitting ? 'Training…' : 'Run experiment'}</button>
          </div>
          {submitting && <LoadingState message="Training and evaluating models…" />}
        </section>
      </form>
    </div>
  )
}
