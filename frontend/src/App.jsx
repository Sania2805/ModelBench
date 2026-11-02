import { Link, Route, Routes } from 'react-router-dom'
import Layout from './components/Layout'
import ProtectedRoute from './components/ProtectedRoute'
import AuthPage from './pages/AuthPage'
import { EmptyState } from './components/common'
import DashboardPage from './pages/DashboardPage'
import UploadPage from './pages/UploadPage'
import ConfigurePage from './pages/ConfigurePage'
import ResultsPage from './pages/ResultsPage'
import HistoryPage from './pages/HistoryPage'

function NotFound() {
  return <div className="page"><EmptyState title="Page not found" description="The page you are looking for does not exist." action={<Link className="btn btn-primary" to="/">Go to dashboard</Link>} /></div>
}

export default function App() {
  return (
    <Routes>
      <Route path="/login" element={<AuthPage mode="login" />} />
      <Route path="/register" element={<AuthPage mode="register" />} />
      <Route element={<ProtectedRoute />}>
        <Route element={<Layout />}>
          <Route index element={<DashboardPage />} />
          <Route path="upload" element={<UploadPage />} />
          <Route path="experiments/new" element={<ConfigurePage />} />
          <Route path="experiments/:experimentId" element={<ResultsPage />} />
          <Route path="history" element={<HistoryPage />} />
          <Route path="*" element={<NotFound />} />
        </Route>
      </Route>
    </Routes>
  )
}
