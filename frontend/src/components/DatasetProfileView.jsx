import { Notice, StatCard } from './common'
import { formatBytes, formatMetric } from '../utils/format'

function PreviewTable({ columns, rows }) {
  return (
    <div className="table-wrap">
      <table className="table">
        <thead>
          <tr>
            {columns.map((c) => (
              <th key={c.name}>{c.name}</th>
            ))}
          </tr>
        </thead>
        <tbody>
          {rows.map((row, i) => (
            <tr key={i}>
              {columns.map((c) => {
                const value = row[c.name]
                const missing = value === null || value === undefined
                return (
                  <td key={c.name} className={missing ? 'null' : ''}>
                    {missing ? 'null' : String(value)}
                  </td>
                )
              })}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}

function ColumnTable({ columns }) {
  return (
    <div className="table-wrap">
      <table className="table">
        <thead>
          <tr>
            <th>Column</th>
            <th>Type</th>
            <th>Kind</th>
            <th className="num">Unique</th>
            <th className="num">Missing</th>
            <th className="num">Missing %</th>
            <th>Notes</th>
          </tr>
        </thead>
        <tbody>
          {columns.map((c) => (
            <tr key={c.name}>
              <td><strong>{c.name}</strong></td>
              <td className="mono">{c.dtype}</td>
              <td><span className={`badge ${c.kind === 'numeric' ? 'info' : ''}`}>{c.kind}</span></td>
              <td className="num">{c.unique_count}</td>
              <td className="num">{c.missing_count}</td>
              <td className="num">{c.missing_percent.toFixed(1)}%</td>
              <td>{c.likely_identifier && <span className="badge warning">likely ID</span>}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}

const NUMERIC_STATS = ['count', 'mean', 'std', 'min', '25%', '50%', '75%', 'max']

function NumericStatsTable({ stats }) {
  const names = Object.keys(stats)
  if (names.length === 0) return <p className="muted">No numerical columns.</p>
  return (
    <div className="table-wrap">
      <table className="table">
        <thead>
          <tr>
            <th>Column</th>
            {NUMERIC_STATS.map((k) => (
              <th key={k} className="num">{k}</th>
            ))}
          </tr>
        </thead>
        <tbody>
          {names.map((name) => (
            <tr key={name}>
              <td><strong>{name}</strong></td>
              {NUMERIC_STATS.map((k) => (
                <td key={k} className="num">
                  {k === 'count' ? stats[name][k] : formatMetric(stats[name][k], 3)}
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}

function CategoricalStatsTable({ stats }) {
  const names = Object.keys(stats)
  if (names.length === 0) return <p className="muted">No categorical columns.</p>
  return (
    <div className="table-wrap">
      <table className="table">
        <thead>
          <tr>
            <th>Column</th>
            <th className="num">Distinct</th>
            <th>Most frequent</th>
            <th className="num">Frequency</th>
          </tr>
        </thead>
        <tbody>
          {names.map((name) => (
            <tr key={name}>
              <td><strong>{name}</strong></td>
              <td className="num">{stats[name].unique}</td>
              <td>{stats[name].top ?? '—'}</td>
              <td className="num">{stats[name].top_frequency}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}

export default function DatasetProfileView({ profile, onRegenerate, regenerating = false }) {
  const { dataset } = profile
  return (
    <div className="stack" style={{ gap: 20 }}>
      <div className="spread" style={{ alignItems: 'center', flexWrap: 'wrap', gap: 12 }}>
        <div>
          <span className="eyebrow">SAVED DATA PROFILE</span>
          <p className="muted small" style={{ margin: '4px 0 0' }}>
            {profile.insight_version ? `Analysis version ${profile.insight_version}` : 'Analysis preview'}
            {profile.insights_generated_at ? ` · ${new Date(profile.insights_generated_at).toLocaleString()}` : ''}
          </p>
        </div>
        {onRegenerate && (
          <button className="btn btn-secondary" type="button" onClick={onRegenerate} disabled={regenerating}>
            {regenerating ? 'Regenerating…' : '↻ Regenerate & save'}
          </button>
        )}
      </div>
      <div className="grid grid-4">
        <StatCard label="Rows" value={dataset.row_count.toLocaleString()} hint={formatBytes(dataset.file_size_bytes)} />
        <StatCard
          label="Columns"
          value={dataset.column_count}
          hint={`${profile.numeric_column_count} numerical · ${profile.categorical_column_count} categorical`}
        />
        <StatCard label="Missing values" value={profile.total_missing_values.toLocaleString()} />
        <StatCard label="Duplicate rows" value={profile.duplicate_row_count.toLocaleString()} />
      </div>

      <Notice type="warning" title="Things to check before training" items={profile.warnings} />

      <section className="card">
        <div className="card-header">
          <h2>Preview</h2>
          <p>First {profile.preview.length} rows of {dataset.original_filename}</p>
        </div>
        <PreviewTable columns={profile.columns} rows={profile.preview} />
      </section>

      <section className="card">
        <div className="card-header">
          <h2>Columns</h2>
          <p>Inferred types and missing values</p>
        </div>
        <ColumnTable columns={profile.columns} />
      </section>

      <div className="grid grid-2">
        <section className="card">
          <div className="card-header"><h2>Numerical statistics</h2></div>
          <NumericStatsTable stats={profile.numeric_statistics} />
        </section>
        <section className="card">
          <div className="card-header"><h2>Categorical statistics</h2></div>
          <CategoricalStatsTable stats={profile.categorical_statistics} />
        </section>
      </div>

      <div className="grid grid-2">
        <section className="card">
          <div className="card-header"><h2>Missing values analysis</h2><p>Missing cells by column</p></div>
          {Object.keys(profile.missing_by_column || {}).length === 0 ? <p className="muted">No missing-value data available.</p> : (
            <div className="table-wrap"><table className="table"><thead><tr><th>Column</th><th className="num">Missing</th><th className="num">Missing %</th></tr></thead><tbody>
              {Object.entries(profile.missing_by_column).map(([name, item]) => <tr key={name}><td><strong>{name}</strong></td><td className="num">{item.missing_count}</td><td className="num">{Number(item.missing_percent).toFixed(1)}%</td></tr>)}
            </tbody></table></div>
          )}
        </section>
        <section className="card">
          <div className="card-header"><h2>Automated recommendations</h2><p>Data-quality checks to review before training</p></div>
          {profile.recommendations?.length ? <ul className="recommendation-list">{profile.recommendations.map((item, i) => <li key={i}>{item}</li>)}</ul> : <p className="muted">No recommendations were generated.</p>}
        </section>
      </div>

      <section className="card">
        <div className="card-header"><h2>Correlation analysis</h2><p>Pearson correlation between numeric columns. Correlation does not imply causation.</p></div>
        {Object.keys(profile.correlation_matrix || {}).length < 2 ? <p className="muted">At least two numeric columns are needed for correlation analysis.</p> : (
          <div className="table-wrap"><table className="table"><thead><tr><th>Column</th>{Object.keys(profile.correlation_matrix).map((name) => <th key={name} className="num">{name}</th>)}</tr></thead><tbody>
            {Object.entries(profile.correlation_matrix).map(([rowName, values]) => <tr key={rowName}><td><strong>{rowName}</strong></td>{Object.keys(profile.correlation_matrix).map((colName) => <td key={colName} className="num">{values[colName] == null ? '—' : Number(values[colName]).toFixed(3)}</td>)}</tr>)}
          </tbody></table></div>
        )}
        {profile.correlation_columns_analyzed > 0 && <p className="muted small" style={{ marginTop: 10 }}>Analyzed {profile.correlation_columns_analyzed} numeric columns.</p>}
      </section>
    </div>
  )
}
