import { useState } from 'react'
import { Link } from 'react-router-dom'
import FileDropzone from '../components/FileDropzone'
import DatasetProfileView from '../components/DatasetProfileView'
import { ErrorMessage, Notice, PageHeader } from '../components/common'
import { getErrorMessage, uploadDataset } from '../services/api'
import { formatBytes } from '../utils/format'

const MAX_SIZE_MB = 10

export default function UploadPage() {
  const [file, setFile] = useState(null)
  const [progress, setProgress] = useState(0)
  const [uploading, setUploading] = useState(false)
  const [error, setError] = useState(null)
  const [profile, setProfile] = useState(null)

  const handleFile = async (selected) => {
    setError(null)
    setProfile(null)
    setFile(selected)

    if (!selected.name.toLowerCase().endsWith('.csv')) {
      setError('Only .csv files are accepted. Export spreadsheets as "CSV (Comma delimited)" first.')
      return
    }
    if (selected.size > MAX_SIZE_MB * 1024 * 1024) {
      setError(`The file is ${formatBytes(selected.size)}, which is over the ${MAX_SIZE_MB} MB limit.`)
      return
    }
    if (selected.size === 0) {
      setError('The file is empty.')
      return
    }

    setUploading(true)
    setProgress(0)
    try {
      setProfile(await uploadDataset(selected, setProgress))
    } catch (err) {
      setError(getErrorMessage(err))
    } finally {
      setUploading(false)
    }
  }

  return (
    <div className="page">
      <PageHeader
        title="Upload dataset"
        description="Upload a CSV file with a header row. The original file is stored unchanged; ModelBench only reads it."
        actions={
          profile && (
            <Link className="btn btn-primary" to={`/experiments/new?datasetId=${profile.dataset.id}`}>
              Continue to experiment setup
            </Link>
          )
        }
      />

      <section className="card">
        <FileDropzone onFileSelected={handleFile} disabled={uploading} maxSizeMb={MAX_SIZE_MB} />
        {file && (
          <div style={{ marginTop: 14 }}>
            <div className="spread small">
              <span>
                <strong>{file.name}</strong> <span className="muted">· {formatBytes(file.size)}</span>
              </span>
              <span className="muted">
                {uploading ? (progress < 100 ? `Uploading ${progress}%` : 'Validating and profiling…') : ''}
              </span>
            </div>
            {uploading && (
              <div className="progress">
                <div style={{ width: `${progress}%` }} />
              </div>
            )}
          </div>
        )}
      </section>

      <ErrorMessage message={error} />

      {profile && (
        <>
          <Notice type="success">
            <div className="spread">
              <span>
                <strong>{profile.dataset.original_filename}</strong> was uploaded and validated.
              </span>
              <Link className="btn btn-primary btn-sm" to={`/experiments/new?datasetId=${profile.dataset.id}`}>
                Continue to experiment setup
              </Link>
            </div>
          </Notice>
          <DatasetProfileView profile={profile} />
        </>
      )}

      {!file && (
        <Notice type="info" title="Tips for a good dataset">
          <ul>
            <li>One row per example, one column per feature, plus the column you want to predict.</li>
            <li>At least 10 rows are required; a few hundred or more gives more reliable comparisons.</li>
            <li>Missing values are fine — they are imputed inside the training pipeline.</li>
            <li>Try the sample files in the project&apos;s <code>sample_data</code> folder.</li>
          </ul>
        </Notice>
      )}
    </div>
  )
}
