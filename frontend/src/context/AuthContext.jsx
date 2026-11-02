import { createContext, useContext, useEffect, useMemo, useState } from 'react'
import { getCurrentUser, loginRequest, logoutRequest, registerRequest } from '../services/auth'

const AuthContext = createContext(null)

export function AuthProvider({ children }) {
  const [user, setUser] = useState(null)
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    let active = true
    getCurrentUser().then((current) => { if (active) setUser(current) })
      .catch(() => { if (active) setUser(null) })
      .finally(() => { if (active) setLoading(false) })
    return () => { active = false }
  }, [])

  const login = async (credentials) => { const current = await loginRequest(credentials); setUser(current); return current }
  const register = async (details) => { const current = await registerRequest(details); setUser(current); return current }
  const logout = async () => { await logoutRequest(); setUser(null) }
  const value = useMemo(() => ({ user, loading, login, register, logout }), [user, loading])
  return <AuthContext.Provider value={value}>{children}</AuthContext.Provider>
}

export function useAuth() {
  const value = useContext(AuthContext)
  if (!value) throw new Error('useAuth must be used inside AuthProvider.')
  return value
}
