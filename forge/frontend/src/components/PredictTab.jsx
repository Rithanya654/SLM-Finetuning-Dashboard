import React, { useState, useRef, useEffect } from 'react'
import { Card, CardTitle, Btn, Slider, SectionLabel, Empty, ModelGrid, Spinner } from './UI.jsx'
import { api } from '../api.js'

/* ── Doc preview panel ───────────────────────────── */
function DocPanel({ file, extractedText, extracting, onClear, onTogglePreview, showPreview }) {
  const ext = file?.name?.split('.').pop()?.toUpperCase() || 'DOC'
  const lines = extractedText ? extractedText.split('\n').filter(l => l.trim()).length : 0

  return (
    <div className="surface-muted" style={{ marginBottom: '.75rem', overflow: 'hidden', animation: 'fadeIn .2s ease' }}>
      {/* Header row */}
      <div style={{
        display: 'flex', alignItems: 'center', gap: '.75rem',
        padding: '14px 16px',
        borderBottom: showPreview ? '1px solid rgba(49, 56, 70, 0.7)' : 'none',
      }}>
        {/* File icon */}
        <div style={{
          width: 40, height: 40, borderRadius: 12, flexShrink: 0,
          background: 'rgba(20, 184, 166, 0.14)', border: '1px solid rgba(20, 184, 166, 0.24)',
          display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center',
        }}>
          <span className="mono" style={{ fontSize: 10, color: 'var(--accent)', fontWeight: 600 }}>{ext}</span>
        </div>

        <div style={{ flex: 1, minWidth: 0 }}>
          <div style={{ fontSize: 14, fontWeight: 600, color: 'var(--text)', overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
            {file.name}
          </div>
          <div style={{ fontSize: 12, color: 'var(--text3)', marginTop: '.2rem' }}>
            {extracting ? 'Extracting with Azure Document Intelligence…' : (
              extractedText ? `${extractedText.length.toLocaleString()} chars · ${lines} lines extracted` : 'Processing…'
            )}
          </div>
        </div>

        <div style={{ display: 'flex', alignItems: 'center', gap: '.5rem', flexShrink: 0 }}>
          {extracting && <Spinner size={13} />}
          {!extracting && extractedText && (
            <button onClick={onTogglePreview} className="btn btn-ghost btn-sm" style={{
              minHeight: 32,
              background: showPreview ? 'rgba(20, 184, 166, 0.12)' : undefined,
              borderColor: showPreview ? 'rgba(20, 184, 166, 0.24)' : undefined,
              color: showPreview ? 'var(--text)' : undefined,
            }}>
              {showPreview ? 'hide' : 'preview'}
            </button>
          )}
          <button onClick={onClear} style={{
            fontSize: 14, color: 'var(--text3)', background: 'none', border: 'none',
            cursor: 'pointer', lineHeight: 1, padding: '.2rem',
            transition: 'color .15s',
          }} title="Remove document">×</button>
        </div>
      </div>

      {/* Extracted content preview */}
      {showPreview && extractedText && (
        <div style={{
          maxHeight: 180, overflowY: 'auto',
          padding: '14px 16px',
          fontFamily: 'var(--mono)', fontSize: 12,
          color: 'var(--text2)', lineHeight: 1.7,
          background: 'rgba(11, 13, 16, 0.92)',
          borderTop: '1px solid rgba(49, 56, 70, 0.7)',
          whiteSpace: 'pre-wrap', wordBreak: 'break-word',
        }}>
          {extractedText.slice(0, 2000)}{extractedText.length > 2000 ? '\n…[truncated for preview]' : ''}
        </div>
      )}

      {/* Usage hint */}
      {!extracting && extractedText && (
        <div style={{
          padding: '10px 16px',
          fontSize: 12, color: 'var(--text3)',
          borderTop: '1px solid rgba(49, 56, 70, 0.7)',
          background: 'rgba(20, 184, 166, 0.04)',
        }}>
          Document context is attached. Ask for extraction, summarization, or a direct answer grounded in the uploaded file.
        </div>
      )}
    </div>
  )
}

/* ── Chat area ───────────────────────────────────── */
function ChatArea({ messages, minHeight = 300 }) {
  const ref = useRef(null)
  useEffect(() => {
    if (ref.current) ref.current.scrollTop = ref.current.scrollHeight
  }, [messages])

  return (
    <div ref={ref} style={{
      background: 'rgba(11, 13, 16, 0.88)', border: '1px solid rgba(49, 56, 70, 0.8)', borderRadius: 18,
      minHeight, maxHeight: 460, overflowY: 'auto', padding: '1rem',
      display: 'flex', flexDirection: 'column', gap: '.6rem',
    }}>
      {messages.length === 0 ? (
        <Empty>Select a model and send a prompt</Empty>
      ) : messages.map((m, i) => (
        <div key={i} style={{
          maxWidth: '84%',
          alignSelf: m.role === 'user' ? 'flex-end' : 'flex-start',
          animation: 'fadeIn .2s ease',
        }}>
          <div style={{
            background: m.role === 'user' ? 'rgba(20, 184, 166, 0.12)' : 'rgba(23, 27, 33, 0.92)',
            border: m.role === 'user' ? '1px solid rgba(20, 184, 166, 0.24)' : '1px solid rgba(49, 56, 70, 0.84)',
            borderRadius: m.role === 'user' ? '16px 16px 6px 16px' : '16px 16px 16px 6px',
            padding: '12px 14px',
          }}>
            <div style={{ fontSize: 11, color: 'var(--text3)', marginBottom: '.35rem' }}>
              {m.role === 'user' ? 'you' : (m.modelId?.split('/').pop() || 'model')}
            </div>
            <div style={{ fontSize: 14, color: m.error ? 'var(--red)' : 'var(--text)', whiteSpace: 'pre-wrap', lineHeight: 1.7 }}>
              {m.loading ? (
                <span style={{ display: 'flex', gap: '.3rem', alignItems: 'center' }}>
                  <Spinner size={10} /> <span style={{ color: 'var(--text3)' }}>Generating…</span>
                </span>
              ) : m.text}
            </div>
          </div>
        </div>
      ))}
    </div>
  )
}

const INFERENCE_TIPS = {
  maxTokens: 'Maximum length of the model response. Higher values allow longer answers, but take more time and may include extra text. Use 128-512 for most extraction/demo questions.',
  temperature: 'Controls randomness. Lower values give more stable, factual answers. For document extraction and comparisons, 0.0-0.3 is safest; higher values are more creative.',
  topP: 'Limits generation to the most likely token choices. Lower values are stricter and more focused. 0.8-0.95 is a good default for chat; use lower with low temperature for extraction.',
}

const GEMINI_FINETUNED_MODEL_ID = 'finetuned-Qwen2.5-3B-Instruct-gemini-demo'

function InferenceTooltip({ tipKey }) {
  const [show, setShow] = useState(false)
  const tip = INFERENCE_TIPS[tipKey]
  if (!tip) return null
  return (
    <span style={{ position: 'relative', display: 'inline-flex', alignItems: 'center' }}>
      <button
        type="button"
        onMouseEnter={() => setShow(true)}
        onMouseLeave={() => setShow(false)}
        onFocus={() => setShow(true)}
        onBlur={() => setShow(false)}
        style={{
          width: 16, height: 16, borderRadius: '50%', border: '1px solid var(--border2)',
          background: 'var(--bg4)', color: 'var(--text3)', fontSize: 10, fontWeight: 700,
          cursor: 'help', display: 'inline-flex', alignItems: 'center', justifyContent: 'center',
          marginLeft: 5, flexShrink: 0, lineHeight: 1,
        }}
        tabIndex={0}
        aria-label="Inference parameter help"
      >?</button>
      {show && (
        <div style={{
          position: 'absolute', bottom: '120%', left: '50%', transform: 'translateX(-50%)',
          width: 260, background: 'rgba(23,27,33,0.98)', border: '1px solid var(--border)',
          borderRadius: 12, padding: '10px 12px', fontSize: 12, color: 'var(--text2)',
          lineHeight: 1.6, zIndex: 999, boxShadow: '0 8px 32px rgba(0,0,0,.5)',
          pointerEvents: 'none', whiteSpace: 'normal',
        }}>
          {tip}
          <div style={{
            position: 'absolute', top: '100%', left: '50%', transform: 'translateX(-50%)',
            width: 0, height: 0,
            borderLeft: '5px solid transparent',
            borderRight: '5px solid transparent',
            borderTop: '5px solid var(--border)',
          }} />
        </div>
      )}
    </span>
  )
}

function InferenceLabel({ children, tipKey }) {
  return (
    <span style={{ display: 'inline-flex', alignItems: 'center' }}>
      {children}
      <InferenceTooltip tipKey={tipKey} />
    </span>
  )
}

function isFinetunedModel(model) {
  return Boolean(model && (model.type === 'finetuned' || String(model.id || '').startsWith('finetuned-')))
}

/* ── Main component ──────────────────────────────── */
export default function PredictTab({ models }) {
  const [selectedModel, setSelectedModel] = useState(null)
  const [compareMode, setCompareMode] = useState(true)
  const [compareFtModel, setCompareFtModel] = useState(null)
  const [compareBaseModel, setCompareBaseModel] = useState(null)
  const [messages, setMessages] = useState([])
  const [baseMessages, setBaseMessages] = useState([])
  const [ftMessages, setFtMessages] = useState([])
  const [prompt, setPrompt] = useState('')
  const [comparePrompt, setComparePrompt] = useState('')
  const [loading, setLoading] = useState(false)

  // Doc attachment state — extracted text stored silently
  const [attachedFile, setAttachedFile] = useState(null)
  const [extractedContext, setExtractedContext] = useState(null)
  const [extracting, setExtracting] = useState(false)
  const [showPreview, setShowPreview] = useState(false)
  const [sampleDocs, setSampleDocs] = useState([])
  const [loadingSampleDoc, setLoadingSampleDoc] = useState(null)

  // Gen params
  const [maxTok, setMaxTok] = useState(256)
  const [temp, setTemp] = useState(7)
  const [topP, setTopP] = useState(9)

  const genParams = {
    max_new_tokens: maxTok,
    temperature: temp / 10,
    top_p: topP / 10,
  }

  useEffect(() => {
    const ft = models.finetuned || []
    if (ft.length === 0) {
      setCompareFtModel(null)
      return
    }
    setCompareFtModel(prev => (prev && ft.some(m => m.id === prev.id) ? prev : ft[0]))
  }, [models.finetuned])

  useEffect(() => {
    const base = models.base || []
    if (base.length === 0) {
      setCompareBaseModel(null)
      return
    }
    const matchedBase = getPairedBaseModel(compareFtModel)
    setCompareBaseModel(prev => {
      if (matchedBase) return matchedBase
      return (prev && base.some(m => m.id === prev.id)) ? prev : base[0]
    })
  }, [models.base, compareFtModel])

  useEffect(() => {
    let cancelled = false
    api.sampleDocuments()
      .then(res => {
        if (!cancelled) setSampleDocs(res.documents || [])
      })
      .catch(e => console.warn('Could not load sample documents:', e.message))
    return () => { cancelled = true }
  }, [])

  function getPairedBaseModel(ftModel) {
    if (!ftModel) return null
    const baseModels = models.base || []
    const modelText = `${ftModel.id || ''} ${ftModel.name || ''} ${ftModel.base_model || ''}`.toLowerCase()

    if (modelText.includes('3b')) {
      const threeB = baseModels.find(m => `${m.id} ${m.name} ${m.params || ''}`.toLowerCase().includes('3b'))
      if (threeB) return threeB
    }

    if (modelText.includes('1.5b') || modelText.includes('1_5b') || modelText.includes('1-5b')) {
      const oneFiveB = baseModels.find(m => `${m.id} ${m.name} ${m.params || ''}`.toLowerCase().includes('1.5b'))
      if (oneFiveB) return oneFiveB
    }

    const ftBase = (ftModel.base_model || '').trim()
    if (!ftBase) return null
    const exact = baseModels.find(m => m.id === ftBase)
    if (exact) return exact

    const normalized = ftBase.toLowerCase()
    const byNormalized = baseModels.find(m => m.id.toLowerCase() === normalized)
    if (byNormalized) return byNormalized

    const ftShort = normalized.split('/').pop()
    return baseModels.find(m => m.id.toLowerCase().endsWith(`/${ftShort}`)) || null
  }

  const pairedBaseModel = compareBaseModel || getPairedBaseModel(compareFtModel)

  async function handleFileAttach(e) {
    const f = e.target.files[0]
    if (!f) return
    setMessages([])
    setBaseMessages([])
    setFtMessages([])
    setAttachedFile(f)
    setExtractedContext(null)
    setExtracting(true)
    setShowPreview(false)
    try {
      const res = await api.extractDocument(f)
      if (res.text && res.text.trim().length > 0) {
        setExtractedContext(res.text)
      } else {
        alert('Extraction returned empty text. The document may not contain readable content.')
        setAttachedFile(null)
      }
    } catch (err) {
      alert(`Extraction failed: ${err.message}`)
      setAttachedFile(null)
    } finally {
      setExtracting(false)
    }
  }

  async function attachSampleDocument(doc) {
    setLoadingSampleDoc(doc.id)
    setExtracting(false)
    setShowPreview(false)
    setMessages([])
    setBaseMessages([])
    setFtMessages([])
    try {
      const res = await api.sampleDocument(doc.id)
      setAttachedFile({ name: res.filename, size: res.char_count || 0 })
      setExtractedContext(res.text || '')
      setShowPreview(false)
    } catch (err) {
      alert(`Could not load sample document: ${err.message}`)
    } finally {
      setLoadingSampleDoc(null)
    }
  }

  function clearAttachment() {
    setAttachedFile(null)
    setExtractedContext(null)
    setShowPreview(false)
    setMessages([])
    setBaseMessages([])
    setFtMessages([])
  }

  async function send() {
    if (!selectedModel || !prompt.trim()) return
    const p = prompt.trim()
    setPrompt('')

    const userMsg = {
      role: 'user',
      text: p,
      hasDoc: !!extractedContext,
    }
    setMessages(prev => [...prev, userMsg, { role: 'assistant', text: '', loading: true, modelId: selectedModel.id }])
    setLoading(true)

    try {
      const requestModelId = isFinetunedModel(selectedModel) ? GEMINI_FINETUNED_MODEL_ID : selectedModel.id
      const res = await api.predict({
        model_id: requestModelId,
        prompt: p,
        context: extractedContext || undefined,  // sent silently, not shown in UI
        ...genParams,
      })
      setMessages(prev => {
        const copy = [...prev]
        copy[copy.length - 1] = { role: 'assistant', text: res.response, modelId: selectedModel.id }
        return copy
      })
    } catch (e) {
      setMessages(prev => {
        const copy = [...prev]
        copy[copy.length - 1] = { role: 'assistant', text: `Error: ${e.message}`, error: true }
        return copy
      })
    } finally {
      setLoading(false)
    }
  }

  async function sendCompare() {
    const p = comparePrompt.trim()
    if (!p || extracting) return
    const ftModel = compareFtModel
    const baseModel = pairedBaseModel
    if (!ftModel) { alert('Select a finetuned model for comparison'); return }
    if (!baseModel) { alert(`No matching base model found for ${ftModel.base_model || ftModel.name}`); return }

    setComparePrompt('')
    const userMsg = { role: 'user', text: p, hasDoc: !!extractedContext }
    setBaseMessages(prev => [...prev, userMsg, { role: 'assistant', text: '', loading: true, modelId: baseModel.id }])
    setFtMessages(prev => [...prev, userMsg, { role: 'assistant', text: '', loading: true, modelId: ftModel.id }])

    await Promise.all([
      api.predict({ model_id: baseModel.id, prompt: p, context: extractedContext || undefined, ...genParams })
        .then(res => setBaseMessages(prev => { const c = [...prev]; c[c.length - 1] = { role: 'assistant', text: res.response, modelId: baseModel.id }; return c }))
        .catch(e => setBaseMessages(prev => { const c = [...prev]; c[c.length - 1] = { role: 'assistant', text: `Error: ${e.message}`, error: true }; return c })),
      api.predict({ model_id: GEMINI_FINETUNED_MODEL_ID, prompt: p, context: extractedContext || undefined, ...genParams })
        .then(res => setFtMessages(prev => { const c = [...prev]; c[c.length - 1] = { role: 'assistant', text: res.response, modelId: ftModel.id }; return c }))
        .catch(e => setFtMessages(prev => { const c = [...prev]; c[c.length - 1] = { role: 'assistant', text: `Error: ${e.message}`, error: true }; return c })),
    ])
  }

  return (
    <div className="dashboard-layout">

      {/* ── Side Pane ── */}
      <Card className="sticky-panel">
        <CardTitle style={{ marginBottom: '.75rem' }}>Select Model</CardTitle>

        {!compareMode ? (
          <>
            {(models.base || []).length > 0 && (
              <>
                <SectionLabel>Base Models</SectionLabel>
                <ModelGrid
                  models={models.base || []}
                  selectedId={selectedModel?.id}
                  onSelect={m => { setSelectedModel(m); setMessages([]) }}
                />
              </>
            )}

            {(models.finetuned || []).length > 0 && (
              <>
                <SectionLabel>Finetuned Models</SectionLabel>
                <ModelGrid
                  models={models.finetuned || []}
                  selectedId={selectedModel?.id}
                  onSelect={m => { setSelectedModel(m); setMessages([]) }}
                />
              </>
            )}

            {(!models.finetuned || models.finetuned.length === 0) && (
              <p style={{ fontSize: 10, color: 'var(--text3)', fontFamily: 'var(--mono)', padding: '0 .25rem' }}>
                No finetuned models available.
              </p>
            )}
          </>
        ) : (
          <>
            {(models.base || []).length > 0 && (
              <>
                <SectionLabel>Base Models</SectionLabel>
                <ModelGrid
                  models={models.base || []}
                  selectedId={compareBaseModel?.id}
                  onSelect={m => { setCompareBaseModel(m); setBaseMessages([]); setFtMessages([]) }}
                />
              </>
            )}

            <SectionLabel>Finetuned (Compare)</SectionLabel>
            <ModelGrid
              models={models.finetuned || []}
              selectedId={compareFtModel?.id}
              onSelect={m => {
                setCompareFtModel(m)
                setCompareBaseModel(getPairedBaseModel(m))
                setBaseMessages([])
                setFtMessages([])
              }}
              emptyText="No finetuned models available."
            />
          </>
        )}

        <div style={{ marginTop: '1rem' }}>
          <SectionLabel>Sample Documents</SectionLabel>
          <div style={{ display: 'flex', flexDirection: 'column', gap: '.4rem' }}>
            {sampleDocs.length === 0 ? (
              <div className="surface-muted" style={{ fontSize: 12, color: 'var(--text3)', padding: '10px 12px' }}>
                Sample documents unavailable.
              </div>
            ) : sampleDocs.map(doc => (
              <button
                key={doc.id}
                onClick={() => attachSampleDocument(doc)}
                disabled={loadingSampleDoc === doc.id}
                style={{
                  display: 'flex', alignItems: 'center', gap: '.65rem',
                  padding: '10px 12px', borderRadius: 12,
                  background: attachedFile?.name === doc.filename ? 'rgba(20, 184, 166, 0.10)' : 'var(--bg3)',
                  border: `1px solid ${attachedFile?.name === doc.filename ? 'rgba(20, 184, 166, 0.28)' : 'var(--border)'}`,
                  color: 'var(--text)', cursor: loadingSampleDoc === doc.id ? 'wait' : 'pointer',
                  textAlign: 'left', opacity: loadingSampleDoc && loadingSampleDoc !== doc.id ? 0.65 : 1,
                }}
              >
                <span className="mono" style={{
                  width: 34, height: 26, borderRadius: 8,
                  display: 'inline-flex', alignItems: 'center', justifyContent: 'center',
                  background: 'rgba(248,113,113,.12)', border: '1px solid rgba(248,113,113,.22)',
                  color: '#ffb4b4', fontSize: 10, fontWeight: 800, flexShrink: 0,
                }}>
                  PDF
                </span>
                <span style={{
                  flex: 1, minWidth: 0, overflow: 'hidden', textOverflow: 'ellipsis',
                  whiteSpace: 'nowrap', fontSize: 12, fontWeight: 600,
                }}>
                  {doc.filename}
                </span>
                {loadingSampleDoc === doc.id && <Spinner size={11} />}
              </button>
            ))}
          </div>
        </div>
      </Card>

      {/* ── Main Pane ── */}
      <div className="stack-md">
        <Card>
          <div className="toolbar" style={{ marginBottom: '1.25rem' }}>
            <div style={{ minWidth: 0 }}>
              <CardTitle style={{ margin: 0 }}>Inference</CardTitle>
              {selectedModel && (
                <div className="mono" style={{ fontSize: 12, color: 'var(--text3)', marginTop: '.35rem', overflow: 'hidden', textOverflow: 'ellipsis' }}>
                  {selectedModel.id}
                </div>
              )}
            </div>
            <div style={{ display: 'flex', gap: '.5rem', flexShrink: 0 }}>
              <Btn size="sm" variant="ghost" onClick={() => { setCompareMode(m => !m); setMessages([]); setBaseMessages([]); setFtMessages([]) }}>
                {compareMode ? 'Single view' : 'Compare'}
              </Btn>
              <Btn size="sm" variant="ghost" onClick={() => { setMessages([]); setBaseMessages([]); setFtMessages([]) }}>Clear</Btn>
            </div>
          </div>

          <div className="triple-grid" style={{ marginBottom: '1.5rem' }}>
            <Slider label={<InferenceLabel tipKey="maxTokens">Max tokens</InferenceLabel>} min={32} max={2048} step={32} value={maxTok} onChange={setMaxTok} />
            <Slider label={<InferenceLabel tipKey="temperature">Temperature</InferenceLabel>} min={0} max={20} value={temp} displayValue={(temp / 10).toFixed(1)} onChange={setTemp} />
            <Slider label={<InferenceLabel tipKey="topP">Top-p</InferenceLabel>} min={0} max={10} value={topP} displayValue={(topP / 10).toFixed(1)} onChange={setTopP} />
          </div>

          {!compareMode ? (
            <>
              {attachedFile && (
                <DocPanel
                  file={attachedFile}
                  extractedText={extractedContext}
                  extracting={extracting}
                  onClear={clearAttachment}
                  onTogglePreview={() => setShowPreview(v => !v)}
                  showPreview={showPreview}
                />
              )}

              <ChatArea messages={messages} minHeight={320} />

              <div style={{ display: 'flex', gap: '.6rem', marginTop: '1rem', alignItems: 'flex-end', flexWrap: 'wrap' }}>
                <label className="btn btn-ghost" style={{
                  width: 96, height: 42, flexShrink: 0,
                  display: 'flex', alignItems: 'center', justifyContent: 'center',
                  cursor: extracting ? 'wait' : 'pointer',
                  background: attachedFile ? 'rgba(124, 156, 255, 0.12)' : undefined,
                  borderColor: attachedFile ? 'rgba(124, 156, 255, 0.24)' : undefined,
                }}>
                  {extracting ? <Spinner size={12} /> : 'Attach'}
                  <input
                    type="file"
                    accept=".pdf,.png,.jpg,.jpeg,.tiff,.bmp,.docx,.xlsx,.pptx"
                    style={{ display: 'none' }}
                    disabled={extracting}
                    onChange={handleFileAttach}
                  />
                </label>

                <textarea
                  value={prompt}
                  onChange={e => setPrompt(e.target.value)}
                  onKeyDown={e => e.key === 'Enter' && e.ctrlKey && !loading && send()}
                  placeholder={
                    !selectedModel
                      ? 'Select a model from the left pane…'
                      : attachedFile
                        ? (extracting ? 'Extracting…' : 'Document ready — ask a question… (Ctrl+Enter)')
                        : 'Enter prompt… (Ctrl+Enter to send)'
                  }
                  disabled={!selectedModel || extracting}
                  rows={2}
                  className="textarea"
                  style={{
                    flex: 1, background: 'var(--bg3)', border: '1px solid var(--border)',
                    borderRadius: 16, color: 'var(--text)',
                    fontSize: 14, padding: '12px 14px', resize: 'vertical',
                    opacity: (!selectedModel || extracting) ? 0.6 : 1,
                    minWidth: 240,
                  }}
                />
                <Btn variant="primary" onClick={send} disabled={loading || !selectedModel || extracting || !prompt.trim()} style={{ minWidth: 100 }}>
                  Send
                </Btn>
              </div>
            </>
          ) : (
            <>
              {attachedFile && (
                <DocPanel
                  file={attachedFile}
                  extractedText={extractedContext}
                  extracting={extracting}
                  onClear={clearAttachment}
                  onTogglePreview={() => setShowPreview(v => !v)}
                  showPreview={showPreview}
                />
              )}

              <div className="split-grid" style={{ gap: '1rem' }}>
                <div>
                  <div style={{
                    display: 'inline-flex', alignItems: 'center', gap: '.4rem',
                    padding: '0 12px', minHeight: 30, borderRadius: 999, fontSize: 12,
                    background: 'rgba(96,165,250,.12)', color: 'var(--blue)',
                    border: '1px solid rgba(96,165,250,.24)', marginBottom: '.6rem',
                  }}>
                    Base model · {pairedBaseModel?.name || 'N/A'}
                  </div>
                  <ChatArea messages={baseMessages} minHeight={240} />
                </div>
                <div>
                  <div style={{
                    display: 'inline-flex', alignItems: 'center', gap: '.4rem',
                    padding: '0 12px', minHeight: 30, borderRadius: 999, fontSize: 12,
                    background: 'rgba(20, 184, 166, 0.05)', color: 'var(--accent2)',
                    border: '1px solid rgba(20, 184, 166, 0.18)', marginBottom: '.6rem',
                  }}>
                    Finetuned model · {compareFtModel?.name || 'N/A'}
                  </div>
                  <ChatArea messages={ftMessages} minHeight={240} />
                </div>
              </div>

              <div style={{ display: 'flex', gap: '.6rem', marginTop: '1rem', alignItems: 'center', flexWrap: 'wrap' }}>
                <label className="btn btn-ghost" style={{
                  width: 96, height: 42, flexShrink: 0,
                  display: 'flex', alignItems: 'center', justifyContent: 'center',
                  cursor: extracting ? 'wait' : 'pointer',
                  background: attachedFile ? 'rgba(20, 184, 166, 0.12)' : undefined,
                  borderColor: attachedFile ? 'rgba(20, 184, 166, 0.24)' : undefined,
                }}>
                  {extracting ? <Spinner size={12} /> : 'Attach'}
                  <input
                    type="file"
                    accept=".pdf,.png,.jpg,.jpeg,.tiff,.bmp,.docx,.xlsx,.pptx"
                    style={{ display: 'none' }}
                    disabled={extracting}
                    onChange={handleFileAttach}
                  />
                </label>

                <input
                  value={comparePrompt} onChange={e => setComparePrompt(e.target.value)}
                  onKeyDown={e => e.key === 'Enter' && !extracting && sendCompare()}
                  placeholder={
                    attachedFile
                      ? (extracting ? 'Extracting…' : 'Document ready — ask common question for comparison…')
                      : 'Ask common question for comparison…'
                  }
                  disabled={extracting}
                  className="input"
                  style={{
                    flex: 1, background: 'var(--bg3)', border: '1px solid var(--border)',
                    borderRadius: 16, color: 'var(--text)',
                    fontSize: 14, padding: '0 14px',
                    opacity: extracting ? 0.6 : 1,
                    minWidth: 240,
                  }}
                />
                <Btn
                  variant="primary"
                  onClick={sendCompare}
                  disabled={extracting || !comparePrompt.trim() || !compareFtModel || !pairedBaseModel}
                  style={{ minWidth: 110 }}
                >
                  Compare
                </Btn>
              </div>
            </>
          )}
        </Card>
      </div>
    </div>
  )
}
