import { useCallback, useEffect, useState } from 'react'
import { Link, useParams } from 'react-router-dom'
import DatasetProfileView from '../components/DatasetProfileView'
import { ErrorMessage, LoadingState, PageHeader } from '../components/common'
import { getDatasetInsightHistory, getDatasetProfile, getErrorMessage, regenerateDatasetInsights } from '../services/api'

export default function DatasetDetailPage() {
  const { datasetId } = useParams()
  const [profile, setProfile] = useState(null)
  const [history, setHistory] = useState([])
  const [selectedVersion, setSelectedVersion] = useState('latest')
  const [loading, setLoading] = useState(true)
  const [regenerating, setRegenerating] = useState(false)
  const [error, setError] = useState(null)

  const load = useCallback(async () => {
    setLoading(true)
    setError(null)
    try {
      const [savedProfile, savedHistory] = await Promise.all([getDatasetProfile(datasetId), getDatasetInsightHistory(datasetId)])
      setProfile(savedProfile)
      setHistory(savedHistory.items || [])
      setSelectedVersion('latest')
    } catch (err) { setError(getErrorMessage(err)) }
    finally { setLoading(false) }
  }, [datasetId])

  useEffect(() => { load() }, [load])

  const handleRegenerate = async () => {
    setRegenerating(true)
    setError(null)
    try {
      const snapshot = await regenerateDatasetInsights(datasetId)
      setProfile((current) => ({ ...current, ...snapshot.profile, insight_id: snapshot.id, insight_version: snapshot.version, insights_generated_at: snapshot.generated_at }))
      setHistory((current) => [snapshot, ...current])
      setSelectedVersion('latest')
    } catch (err) { setError(getErrorMessage(err)) }
    finally { setRegenerating(false) }
  }

  const selected = selectedVersion === 'latest' ? null : history.find((item) => item.id === selectedVersion)
  const visibleProfile = selected && profile ? { ...profile, ...selected.profile, insight_id: selected.id, insight_version: selected.version, insights_generated_at: selected.generated_at } : profile

  return (
    <div className="page">
      <PageHeader title="Dataset analysis" description="Revisit saved data profiles, compare analysis versions, and regenerate insights when needed." actions={<Link className="btn btn-secondary" to="/upload">Upload another dataset</Link>} />
      {loading && <LoadingState message="Loading saved dataset analysis…" />}
      <ErrorMessage message={error} onRetry={load} />
      {!loading && visibleProfile && (
        <>
          <section className="card" style={{ marginBottom: 18 }}>
            <div className="spread" style={{ gap: 14, flexWrap: 'wrap' }}>
              <div><span className="eyebrow">DATASET</span><h2 style={{ margin: '5px 0' }}>{profile.dataset.original_filename}</h2><p className="muted small">Uploaded {new Date(profile.dataset.uploaded_at).toLocaleString()} · {history.length} saved analysis version(s)</p></div>
              <label className="field" style={{ minWidth: 220 }}>
                <span className="muted small">Saved analysis version</span>
                <select className="input" value={selectedVersion} onChange={(event) => setSelectedVersion(event.target.value)}>
                  <option value="latest">Latest analysis</option>
                  {history.map((item) => <option key={item.id} value={item.id}>Version {item.version} · {new Date(item.generated_at).toLocaleString()}</option>)}
                </select>
              </label>
            </div>
          </section>
          <DatasetProfileView profile={visibleProfile} onRegenerate={selected ? undefined : handleRegenerate} regenerating={regenerating} />
          <div style={{ marginTop: 18 }}><Link className="btn btn-primary" to={`/experiments/new?datasetId=${datasetId}`}>Configure experiment with this dataset</Link></div>
        </>
      )}
    </div>
  )
}
