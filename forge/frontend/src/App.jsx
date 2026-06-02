import React, { useState, useEffect, useCallback } from 'react'
import Header from './components/Header.jsx'
// import DataPrepTab from './components/DataPrepTab.jsx'  // commented out: causes OOM
import TrainTab from './components/TrainTab.jsx'
import PredictTab from './components/PredictTab.jsx'
import EvalTab from './components/EvalTab.jsx'
import RunsTab from './components/RunsTab.jsx'
import HelpDrawer from './components/HelpDrawer.jsx'
import { api } from './api.js'

class TabErrorBoundary extends React.Component {
  constructor(props) {
    super(props)
    this.state = { hasError: false, message: '' }
  }

  static getDerivedStateFromError(error) {
    return { hasError: true, message: error?.message || 'Unknown UI error' }
  }

  componentDidCatch(error) {
    console.error('Tab render error:', error)
  }

  render() {
    if (this.state.hasError) {
      return (
        <div className="card" style={{ borderColor: 'rgba(248, 113, 113, 0.32)', background: 'rgba(248, 113, 113, 0.08)' }}>
          <div style={{ color: 'var(--red)', marginBottom: 8, fontSize: 15, fontWeight: 600 }}>UI render error</div>
          <div>{this.state.message}</div>
        </div>
      )
    }
    return this.props.children
  }
}

export default function App() {
  const [activeTab, setActiveTab] = useState('train')
  const [visitedTabs, setVisitedTabs] = useState(['train'])
  const [runResetNonce, setRunResetNonce] = useState(0)
  const [apiOnline, setApiOnline] = useState(false)
  const [models, setModels] = useState({ base: [], finetuned: [] })
  const [datasetPath, setDatasetPath] = useState(null)

  // Tracks model IDs hidden by the user in the Runs tab
  const [removedModelIds, setRemovedModelIds] = useState([])

  const checkAPI = useCallback(async () => {
    try {
      await api.health()
      setApiOnline(true)
    } catch {
      setApiOnline(false)
    }
  }, [])

  const loadModels = useCallback(async () => {
    try {
      const data = await api.models()
      setModels(data)
    } catch (e) {
      console.warn('Could not load models:', e.message)
    }
  }, [])

  useEffect(() => {
    checkAPI()
    loadModels()

    const iv1 = setInterval(() => {
      if (navigator.onLine) checkAPI()
    }, 30_000)

    const iv2 = setInterval(() => {
      if (navigator.onLine) loadModels()
    }, 15_000)

    return () => {
      clearInterval(iv1)
      clearInterval(iv2)
    }
  }, [checkAPI, loadModels])

  useEffect(() => {
    if (!visitedTabs.includes(activeTab)) {
      setVisitedTabs(prev => [...prev, activeTab])
    }
  }, [activeTab, visitedTabs])

  const handleResetRuns = useCallback(() => {
    setRunResetNonce(n => n + 1)
  }, [])

  const handleRemoveModel = useCallback((modelId) => {
    setRemovedModelIds(prev => prev.includes(modelId) ? prev : [...prev, modelId])
  }, [])

  const handleRestoreAll = useCallback(() => {
    setRemovedModelIds([])
  }, [])

  // Filter out removed models before passing to Predict / Eval
  const visibleModels = {
    base: models.base || [],
    finetuned: (models.finetuned || []).filter(m => !removedModelIds.includes(m.id)),
  }

  return (
    <div className="app-shell">
      <Header
        activeTab={activeTab}
        setActiveTab={setActiveTab}
        apiOnline={apiOnline}
        fineTunedCount={visibleModels.finetuned.length}
        onResetRuns={handleResetRuns}
      />
      <HelpDrawer />

      <main className="app-main">

        {/* ── Prep Data tab — COMMENTED OUT (causes OOM) ──────────────────────
        {visitedTabs.includes('prep') && (
          <div style={{ display: activeTab === 'prep' ? 'block' : 'none' }}>
            <TabErrorBoundary>
              <DataPrepTab key={`prep-${runResetNonce}`} onDatasetReady={(path) => {
                  setDatasetPath(path);
                  setVisitedTabs(prev => prev.includes('train') ? prev : [...prev, 'train']);
                  setActiveTab('train');
                }} />
            </TabErrorBoundary>
          </div>
        )}
        ──────────────────────────────────────────────────────────────────────── */}

        {visitedTabs.includes('train') && (
          <div style={{ display: activeTab === 'train' ? 'block' : 'none' }}>
            <TabErrorBoundary>
              <TrainTab
                key={`train-${runResetNonce}`}
                onModelAdded={loadModels}
                onDatasetReady={(path) => {
                  setDatasetPath(path)
                  setActiveTab('train')
                }}
                datasetPath={datasetPath}
              />
            </TabErrorBoundary>
          </div>
        )}

        {visitedTabs.includes('runs') && (
          <div style={{ display: activeTab === 'runs' ? 'block' : 'none' }}>
            <TabErrorBoundary>
              <RunsTab
                key={`runs-${runResetNonce}`}
                allFinetunedModels={models.finetuned || []}
                removedModelIds={removedModelIds}
                onRemoveModel={handleRemoveModel}
                onRestoreAll={handleRestoreAll}
              />
            </TabErrorBoundary>
          </div>
        )}

        {visitedTabs.includes('predict') && (
          <div style={{ display: activeTab === 'predict' ? 'block' : 'none' }}>
            <TabErrorBoundary>
              <PredictTab key={`predict-${runResetNonce}`} models={visibleModels} />
            </TabErrorBoundary>
          </div>
        )}

        {visitedTabs.includes('eval') && (
          <div style={{ display: activeTab === 'eval' ? 'block' : 'none' }}>
            <TabErrorBoundary>
              <EvalTab key={`eval-${runResetNonce}`} models={visibleModels} datasetPath={datasetPath} />
            </TabErrorBoundary>
          </div>
        )}
      </main>
    </div>
  )
}
