import {
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  LabelList,
  ReferenceLine,
  ResponsiveContainer,
  Scatter,
  ScatterChart,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts'
import { METRIC_LABELS, formatMetric } from '../utils/format'

const BLUE = '#2563eb'
const BLUE_LIGHT = '#9dbaf5'
const AXIS = { fontSize: 12, fill: '#5b6b85' }

export function MetricBarChart({ data, metric, bestModel }) {
  const rows = data.filter((row) => row[metric] !== null && row[metric] !== undefined)
  if (rows.length === 0) {
    return <p className="muted">No valid {METRIC_LABELS[metric]} values to compare.</p>
  }
  return (
    <div className="chart-box">
      <ResponsiveContainer width="100%" height="100%">
        <BarChart data={rows} margin={{ top: 24, right: 16, left: 8, bottom: 8 }}>
          <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="#e2e8f0" />
          <XAxis dataKey="model" tick={AXIS} tickLine={false} axisLine={{ stroke: '#cbd5e1' }} />
          <YAxis tick={AXIS} tickLine={false} axisLine={false} tickFormatter={(v) => formatMetric(v, 2)} width={70} />
          <Tooltip formatter={(value) => [formatMetric(value), METRIC_LABELS[metric]]} cursor={{ fill: '#f1f5f9' }} />
          <Bar dataKey={metric} radius={[4, 4, 0, 0]} maxBarSize={72}>
            {rows.map((row) => (
              <Cell key={row.model} fill={row.model === bestModel ? BLUE : BLUE_LIGHT} />
            ))}
            <LabelList dataKey={metric} position="top" formatter={(v) => formatMetric(v, 3)} style={AXIS} />
          </Bar>
        </BarChart>
      </ResponsiveContainer>
    </div>
  )
}

export function ConfusionMatrix({ data }) {
  if (!data) return null
  const { labels, matrix } = data
  const max = Math.max(1, ...matrix.flat())
  const columns = `auto repeat(${labels.length}, minmax(56px, 1fr))`

  return (
    <div style={{ overflowX: 'auto' }}>
      <div className="cm-axis">Rows: actual class · Columns: predicted class</div>
      <div className="cm-grid" style={{ gridTemplateColumns: columns }}>
        <div />
        {labels.map((label) => (
          <div key={`h-${label}`} className="cm-head">{label}</div>
        ))}
        {matrix.map((row, i) => (
          <div key={`r-${labels[i]}`} style={{ display: 'contents' }}>
            <div className="cm-head" style={{ justifyContent: 'flex-end' }}>{labels[i]}</div>
            {row.map((count, j) => {
              const intensity = count / max
              const diagonal = i === j
              const background = diagonal
                ? `rgba(37, 99, 235, ${0.08 + intensity * 0.82})`
                : `rgba(220, 38, 38, ${count === 0 ? 0.03 : 0.08 + intensity * 0.5})`
              return (
                <div
                  key={j}
                  className="cm-cell"
                  title={`Actual ${labels[i]}, predicted ${labels[j]}: ${count}`}
                  style={{ background, color: diagonal && intensity > 0.55 ? '#fff' : '#0f1e3a' }}
                >
                  {count}
                </div>
              )
            })}
          </div>
        ))}
      </div>
    </div>
  )
}

function niceScale(min, max, targetTicks = 6) {
  if (min === max) {
    min -= 1
    max += 1
  }
  const rough = (max - min) / targetTicks
  const magnitude = 10 ** Math.floor(Math.log10(rough))
  const step = [1, 2, 2.5, 5, 10].map((m) => m * magnitude).find((s) => s >= rough)
  const low = Math.floor(min / step) * step
  const high = Math.ceil(max / step) * step
  const ticks = []
  for (let t = low; t <= high + step / 2; t += step) ticks.push(Number(t.toPrecision(12)))
  return { domain: [low, high], ticks }
}

export function ActualVsPredictedChart({ points }) {
  if (!points || points.length === 0) return <p className="muted">No predictions to plot.</p>
  const values = points.flatMap((p) => [p.actual, p.predicted]).filter((v) => v !== null)
  const { domain, ticks } = niceScale(Math.min(...values), Math.max(...values))

  return (
    <div className="chart-box">
      <ResponsiveContainer width="100%" height="100%">
        <ScatterChart margin={{ top: 12, right: 20, left: 8, bottom: 24 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
          <XAxis
            type="number"
            dataKey="actual"
            name="Actual"
            domain={domain}
            ticks={ticks}
            tick={AXIS}
            tickFormatter={(v) => v.toLocaleString()}
            label={{ value: 'Actual', position: 'insideBottom', offset: -12, ...AXIS }}
          />
          <YAxis
            type="number"
            dataKey="predicted"
            name="Predicted"
            domain={domain}
            ticks={ticks}
            tick={AXIS}
            width={70}
            tickFormatter={(v) => v.toLocaleString()}
            label={{ value: 'Predicted', angle: -90, position: 'insideLeft', ...AXIS }}
          />
          <Tooltip formatter={(value, name) => [formatMetric(value, 3), name]} cursor={{ strokeDasharray: '3 3' }} />
          <ReferenceLine
            segment={[
              { x: domain[0], y: domain[0] },
              { x: domain[1], y: domain[1] },
            ]}
            stroke="#94a3b8"
            strokeDasharray="5 5"
            ifOverflow="hidden"
          />
          <Scatter data={points} fill={BLUE} fillOpacity={0.6} />
        </ScatterChart>
      </ResponsiveContainer>
    </div>
  )
}
