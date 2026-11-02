import axios from 'axios'

const api = axios.create({
  baseURL: import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000/api',
})

export function getErrorMessage(error) {
  if (error?.response) {
    const detail = error.response.data?.detail
    if (typeof detail === 'string') return detail
    if (Array.isArray(detail)) return detail.map((d) => d.msg).join('; ')
    return `Request failed with status ${error.response.status}.`
  }
  if (error?.request) {
    return 'Cannot reach the ModelBench API. Make sure the backend is running on the configured URL.'
  }
  return error?.message || 'Something went wrong.'
}

export async function getHealth() {
  const { data } = await api.get('/health')
  return data
}

export async function uploadDataset(file, onProgress) {
  const form = new FormData()
  form.append('file', file)
  const { data } = await api.post('/datasets/upload', form, {
    onUploadProgress: (event) => {
      if (onProgress && event.total) onProgress(Math.round((event.loaded / event.total) * 100))
    },
  })
  return data
}

export async function listDatasets(limit = 50, offset = 0) {
  const { data } = await api.get('/datasets', { params: { limit, offset } })
  return data
}

export async function getDatasetProfile(datasetId) {
  const { data } = await api.get(`/datasets/${encodeURIComponent(datasetId)}/profile`)
  return data
}

export async function getTargetClasses(datasetId, targetColumn) {
  const { data } = await api.get(`/datasets/${encodeURIComponent(datasetId)}/target-classes`, {
    params: { target_column: targetColumn },
  })
  return data
}

export async function getAvailableModels(taskType) {
  const { data } = await api.get('/models', { params: { task_type: taskType } })
  return data
}

export async function createExperiment(payload) {
  const { data } = await api.post('/experiments', payload)
  return data
}

export async function listExperiments(page = 1, pageSize = 10) {
  const { data } = await api.get('/experiments', { params: { page, page_size: pageSize } })
  return data
}

export async function getExperiment(experimentId) {
  const { data } = await api.get(`/experiments/${encodeURIComponent(experimentId)}`)
  return data
}

function filenameFromHeader(header, fallback) {
  const match = /filename\*?=(?:UTF-8'')?"?([^";]+)"?/i.exec(header || '')
  return match ? decodeURIComponent(match[1]) : fallback
}

async function downloadFile(url, fallbackName) {
  try {
    const response = await api.get(url, { responseType: 'blob' })
    const name = filenameFromHeader(response.headers['content-disposition'], fallbackName)
    const objectUrl = URL.createObjectURL(response.data)
    const link = document.createElement('a')
    link.href = objectUrl
    link.download = name
    document.body.appendChild(link)
    link.click()
    link.remove()
    URL.revokeObjectURL(objectUrl)
  } catch (error) {
    if (error?.response?.data instanceof Blob) {
      try {
        error.response.data = JSON.parse(await error.response.data.text())
      } catch {
        error.response.data = {}
      }
    }
    throw error
  }
}

export function downloadReport(experimentId) {
  return downloadFile(
    `/experiments/${encodeURIComponent(experimentId)}/report`,
    `modelbench_report_${experimentId}.html`,
  )
}

export function downloadModel(experimentId, modelName) {
  return downloadFile(
    `/experiments/${encodeURIComponent(experimentId)}/models/${encodeURIComponent(modelName)}/download`,
    `${modelName}.joblib`,
  )
}
