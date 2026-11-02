import api from './api'
export async function registerRequest(payload) { const { data } = await api.post('/auth/register', payload); return data }
export async function loginRequest(payload) { const { data } = await api.post('/auth/login', payload); return data }
export async function getCurrentUser() { const { data } = await api.get('/auth/me'); return data }
export async function logoutRequest() { await api.post('/auth/logout') }
