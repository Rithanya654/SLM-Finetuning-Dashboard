const API_BASE_FROM_ENV = import.meta.env.VITE_API_BASE_URL
const API_BASE_FROM_HOST =
  typeof window !== 'undefined'
    ? `${window.location.protocol}//${window.location.hostname}:8000`
    : 'http://localhost:8000'

const BASE = (API_BASE_FROM_ENV || API_BASE_FROM_HOST).replace(/\/$/, '')

async function req(path, options = {}) {
  const res = await fetch(`${BASE}${path}`, options)
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }))
    throw new Error(err.detail || `HTTP ${res.status}`)
  }
  return res.json()
}

async function reqBlob(path, options = {}) {
  const res = await fetch(`${BASE}${path}`, options)
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }))
    throw new Error(err.detail || `HTTP ${res.status}`)
  }
  const blob = await res.blob()
  const cd = res.headers.get('content-disposition') || ''
  const m = cd.match(/filename\*=UTF-8''([^;]+)|filename=\"?([^\";]+)\"?/i)
  const filename = decodeURIComponent((m?.[1] || m?.[2] || '').trim()) || 'dataset.jsonl'
  return { blob, filename }
}

export const api = {
  health: () => req('/health'),
  models: () => req('/api/models'),

  uploadDataset: (file) => {
    const fd = new FormData()
    fd.append('file', file)
    return req('/api/dataset/upload', { method: 'POST', body: fd })
  },
  downloadDataset: (path) => reqBlob(`/api/dataset/download?path=${encodeURIComponent(path)}`),
  datasetInfo: (path) => req(`/api/dataset/info?path=${encodeURIComponent(path)}`),

  startTraining: (config) =>
    req('/api/train/start', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(config),
    }),

  getJob: (id) => req(`/api/train/job/${id}`),
  listJobs: () => req('/api/train/jobs'),
  cancelJob: (id) => req(`/api/train/cancel/${id}`, { method: 'POST' }),

  // context: optional extracted doc text sent silently alongside prompt
  predict: (payload) =>
    req('/api/predict', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    }),

  runEval: (payload) =>
    req('/api/eval/run', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    }),

  getEval: (id) => req(`/api/eval/${id}`),

  startDataPrep: (config) =>
    req('/api/dataprep/start', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(config),
    }),

  getDataPrepJob: (id) => req(`/api/dataprep/job/${id}`),
  listDataPrepJobs: () => req('/api/dataprep/list'),

  analyzeDataset: (payload) =>
    req('/api/dataprep/analyze', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    }),

  analyzeTrainingDataset: (payload) =>
    req('/analyze-dataset', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    }),

  extractDocument: (file) => {
    const fd = new FormData()
    fd.append('file', file)
    return req('/api/extract', { method: 'POST', body: fd })
  },

  sampleDocuments: () => req('/api/sample-documents'),
  sampleDocument: (id) => req(`/api/sample-documents/${encodeURIComponent(id)}`),
}
