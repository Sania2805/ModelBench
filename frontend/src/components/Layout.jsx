import { NavLink, Outlet, useLocation, useNavigate } from 'react-router-dom'
import { useEffect, useState } from 'react'
import { getHealth } from '../services/api'
import { useAuth } from '../context/AuthContext'

const NAV_ITEMS = [
  { to: '/', label: 'Dashboard', icon: '▦', end: true },
  { to: '/upload', label: 'Datasets', icon: '▤' },
  { to: '/dataset-insights', label: 'Dataset Insights', icon: '▥' },
  { to: '/experiments/new', label: 'New experiment', icon: '⚗' },
  { to: '/history', label: 'Experiments', icon: '◷' },
]

function HealthIndicator() {
  const [health, setHealth] = useState(null)
  useEffect(() => {
    let active = true
    const check = () => getHealth().then((data) => active && setHealth(data)).catch(() => active && setHealth({ status: 'down' }))
    check()
    const timer = setInterval(check, 30000)
    return () => { active = false; clearInterval(timer) }
  }, [])
  const ok = health && health.status !== 'down' && health.database === 'connected'
  return <span className="health-label"><i className={`status-dot ${ok ? 'ok' : health ? 'bad' : ''}`} />{!health ? 'Checking connection' : ok ? 'API connected' : 'API unavailable'}</span>
}

export default function Layout() {
  const location = useLocation()
  const navigate = useNavigate()
  const { user, logout } = useAuth()
  const initials = (user?.full_name || 'User').split(' ').map((part) => part[0]).join('').slice(0, 2).toUpperCase()
  const handleLogout = async () => {
    try { await logout(); navigate('/login', { replace: true }) }
    catch { window.alert('Could not reach the server to end your session. Please try again.') }
  }
  const title = location.pathname.startsWith('/upload') ? 'Datasets' : location.pathname.startsWith('/dataset-insights') || location.pathname.startsWith('/datasets/') ? 'Dataset Insights' : location.pathname.startsWith('/experiments/new') ? 'New experiment' : location.pathname.startsWith('/history') ? 'Experiments' : location.pathname.startsWith('/experiments/') ? 'Experiment results' : 'Dashboard'
  return (
    <div className="app-shell">
      <aside className="sidebar">
        <div className="brand">
          <span className="brand-mark">M</span>
          <span className="brand-copy"><strong>ModelBench</strong><small>ML WORKSPACE</small></span>
        </div>
        <div className="nav-caption">WORKSPACE</div>
        <nav className="nav">
          {NAV_ITEMS.map((item) => (
            <NavLink key={item.to} to={item.to} end={item.end} className={({ isActive }) => (isActive || (item.to === '/dataset-insights' && location.pathname.startsWith('/datasets/'))) ? 'active' : undefined}>
              <span className="nav-icon">{item.icon}</span><span>{item.label}</span>
              {item.to === '/history' && <span className="nav-arrow">↗</span>}
            </NavLink>
          ))}
        </nav>
        <div className="sidebar-bottom">
          <div className="profile-mini">
            <div className="avatar">{initials}</div>
            <div className="profile-identity"><strong>{user?.full_name || 'Account'}</strong><small>{user?.email || ''}</small></div>
            <button className="logout-button" type="button" onClick={handleLogout} title="Sign out" aria-label="Sign out">↪</button>
          </div>
          <HealthIndicator />
        </div>
      </aside>
      <main className="main">
        <header className="topbar">
          <div className="breadcrumb"><span>Workspace</span><b>/</b><strong>{title}</strong></div>
          <div className="topbar-right"><span className="topbar-date">MODEL DEVELOPMENT WORKSPACE</span><div className="topbar-avatar" title={user?.full_name || "Account"}>{initials}</div></div>
        </header>
        <div className="main-content"><Outlet /></div>
      </main>
    </div>
  )
}
