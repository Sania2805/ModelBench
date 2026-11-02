import { useState } from 'react'
import { Link, Navigate, useLocation, useNavigate } from 'react-router-dom'
import { useAuth } from '../context/AuthContext'
import { getErrorMessage } from '../services/api'

export default function AuthPage({ mode = 'login' }) {
  const isRegister = mode === 'register'
  const { user, loading, login, register } = useAuth()
  const navigate = useNavigate()
  const location = useLocation()
  const [fullName, setFullName] = useState('')
  const [email, setEmail] = useState('')
  const [password, setPassword] = useState('')
  const [confirmPassword, setConfirmPassword] = useState('')
  const [showPassword, setShowPassword] = useState(false)
  const [submitting, setSubmitting] = useState(false)
  const [error, setError] = useState('')

  if (loading) return <div className="auth-loading"><div className="spinner" /></div>
  if (user) return <Navigate to="/" replace />

  const submit = async (event) => {
    event.preventDefault()
    setError('')
    if (isRegister && password !== confirmPassword) { setError('Your passwords do not match.'); return }
    setSubmitting(true)
    try {
      if (isRegister) await register({ full_name: fullName.trim(), email: email.trim(), password })
      else await login({ email: email.trim(), password })
      navigate(location.state?.from?.pathname || '/', { replace: true })
    } catch (err) { setError(getErrorMessage(err)) }
    finally { setSubmitting(false) }
  }

  return (
    <main className="auth-page">
      <section className="auth-story">
        <Link to="/login" className="auth-brand"><span className="auth-brand-mark">M</span><span><strong>ModelBench</strong><small>ML WORKSPACE</small></span></Link>
        <div className="auth-story-content">
          <span className="auth-kicker"><span>✳</span> BENCHMARK YOUR ML WORKFLOW</span>
          <h1>Train ML models with clarity, <em>not guesswork.</em></h1>
          <p>Compare machine learning models on your own tabular datasets. Understand how your pipeline works, evaluate results, and keep every experiment organised.</p>
          <div className="auth-feature-list">
            <div><span>01</span><div><strong>Prepare your data</strong><small>Upload CSV files and inspect data quality.</small></div></div>
            <div><span>02</span><div><strong>Train and compare</strong><small>Run multiple models with consistent splits.</small></div></div>
            <div><span>03</span><div><strong>Understand results</strong><small>Review metrics and export trained pipelines.</small></div></div>
          </div>
        </div>
        <div className="auth-story-footer"><span>MODELBENCH</span><span>BUILT FOR CLEARER ML EXPERIMENTS</span></div>
      </section>
      <section className="auth-panel">
        <div className="auth-form-wrap">
          <div className="auth-tabs" role="tablist" aria-label="Account access">
            <Link role="tab" aria-selected={isRegister} className={isRegister ? 'selected' : ''} to="/register">Create account</Link>
            <Link role="tab" aria-selected={!isRegister} className={!isRegister ? 'selected' : ''} to="/login">Sign in</Link>
          </div>
          <div className="auth-heading"><span className="eyebrow">{isRegister ? 'GET STARTED' : 'WELCOME BACK'}</span><h2>{isRegister ? 'Create your account' : 'Sign in to ModelBench'}</h2><p>{isRegister ? 'Set up your account to start running and comparing ML experiments.' : 'Continue to your machine learning workspace.'}</p></div>
          <form className="auth-form" onSubmit={submit}>
            {isRegister && <label className="auth-field"><span>Full name</span><input autoComplete="name" value={fullName} onChange={(e) => setFullName(e.target.value)} placeholder="Your name" minLength={2} maxLength={120} required /></label>}
            <label className="auth-field"><span>Email address</span><input type="email" autoComplete="email" value={email} onChange={(e) => setEmail(e.target.value)} placeholder="you@example.com" required /></label>
            <label className="auth-field"><span>Password</span><div className="auth-password"><input type={showPassword ? 'text' : 'password'} autoComplete={isRegister ? 'new-password' : 'current-password'} value={password} onChange={(e) => setPassword(e.target.value)} placeholder={isRegister ? 'At least 8 characters' : 'Enter your password'} minLength={isRegister ? 8 : 1} maxLength={128} required /><button type="button" onClick={() => setShowPassword((v) => !v)} aria-label={showPassword ? 'Hide password' : 'Show password'}>{showPassword ? 'Hide' : 'Show'}</button></div>{isRegister && <small className="auth-help">Use at least 8 characters.</small>}</label>
            {isRegister && <label className="auth-field"><span>Confirm password</span><input type={showPassword ? 'text' : 'password'} autoComplete="new-password" value={confirmPassword} onChange={(e) => setConfirmPassword(e.target.value)} placeholder="Re-enter your password" minLength={8} maxLength={128} required /></label>}
            {error && <div className="auth-error" role="alert">{error}</div>}
            <button className="btn btn-primary auth-submit" type="submit" disabled={submitting}>{submitting ? (isRegister ? 'Creating account…' : 'Signing in…') : (isRegister ? 'Create account' : 'Sign in')} <span aria-hidden="true">→</span></button>
          </form>
          <p className="auth-switch">{isRegister ? 'Already have an account?' : 'New to ModelBench?'} <Link to={isRegister ? '/login' : '/register'}>{isRegister ? 'Sign in' : 'Create an account'}</Link></p>
          <p className="auth-security"><span>▣</span> Passwords are securely hashed. Sessions use an HttpOnly cookie.</p>
        </div>
      </section>
    </main>
  )
}
