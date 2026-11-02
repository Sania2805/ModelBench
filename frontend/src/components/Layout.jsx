import { NavLink, Outlet } from 'react-router-dom'
import { useEffect, useState } from 'react'
import { getHealth } from '../services/api'

const NAV_ITEMS = [
  { to: '/', label: 'Dashboard', end: true },
  { to: '/upload', label: 'Upload dataset' },
  { to: '/experiments/new', label: 'New experiment' },
  { to: '/history', label: 'Experiment history' },
]

function HealthIndicator() {
  const [health, setHealth] = useState(null)

  useEffect(() => {
    let active = true
    const check = () =>
      getHealth()
        .then((data) => active && setHealth(data))
        .catch(() => active && setHealth({ status: 'down' }))
    check()
    const timer = setInterval(check, 30000)
    return () => {
      active = false
      clearInterval(timer)
    }
  }, [])

  if (!health) return <span><span className="status-dot" />Checking API…</span>
  if (health.status === 'down') return <span><span className="status-dot bad" />API unreachable</span>
  if (health.database !== 'connected') return <span><span className="status-dot bad" />MySQL unavailable</span>
  return <span><span className="status-dot ok" />API and MySQL connected</span>
}

export default function Layout() {
  return (
    <div className="app-shell">
      <aside className="sidebar">
        <div className="brand">
          <img src="/favicon.svg" alt="" />
          ModelBench
        </div>
        <nav className="nav">
          {NAV_ITEMS.map((item) => (
            <NavLink key={item.to} to={item.to} end={item.end}>
              {item.label}
            </NavLink>
          ))}
        </nav>
        <div className="sidebar-footer">
          <HealthIndicator />
        </div>
      </aside>
      <main className="main">
        <Outlet />
      </main>
    </div>
  )
}
