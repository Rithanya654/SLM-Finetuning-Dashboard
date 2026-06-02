import React, { useState, useCallback, useEffect, useRef } from 'react'
import { Card, CardTitle, Btn, Field, Label, Input, Select, Slider, Toggle, SectionLabel, ModelGrid, GpuSelector, InnerTabs } from './UI.jsx'
import { api } from '../api.js'

const PRESETS = {
  quick:     { epochs: 1,  lrIdx: 50, batch: 2, gradAcc: 2,  loraR: 8,  loraAlpha: 16,  warmup: 3, seq: 256,  quantization: '4bit' },
  balanced:  { epochs: 3,  lrIdx: 50, batch: 4, gradAcc: 4,  loraR: 16, loraAlpha: 32,  warmup: 3, seq: 512,  quantization: '4bit' },
  full:      { epochs: 5,  lrIdx: 45, batch: 8, gradAcc: 4,  loraR: 32, loraAlpha: 64,  warmup: 5, seq: 1024, quantization: 'none' },
  loraSmall: { epochs: 2,  lrIdx: 40, batch: 4, gradAcc: 4,  loraR: 4,  loraAlpha: 8,   warmup: 3, seq: 512,  quantization: '4bit' },
  loraLarge: { epochs: 4,  lrIdx: 55, batch: 4, gradAcc: 8,  loraR: 64, loraAlpha: 128, warmup: 5, seq: 512,  quantization: '8bit' },
}

const BASE_MODELS = [
  { id: 'Qwen/Qwen2.5-1.5B-Instruct', name: 'Qwen 2.5 1.5B', params: '1.5B', type: 'base' },
  { id: 'Qwen/Qwen2.5-3B-Instruct',   name: 'Qwen 2.5 3B',   params: '3B',   type: 'base' },
]

function lrFromIdx(idx) { return Math.pow(10, -6 + (idx / 100) * 5) }
function formatLR(idx) {
  const v = lrFromIdx(idx)
  return v < 0.0001 ? v.toExponential(1) : v.toFixed(5)
}
const lrToIdx   = (val) => { const v = Math.max(1e-7, val || 1e-6); return Math.round(((Math.log10(v) + 6) / 5) * 100) }
const idxToLr   = (idx) => Math.pow(10, -6 + (idx / 100) * 5)

// ── Quantization pill selector ─────────────────────────────────────────────
const QUANT_OPTIONS = [
  { value: 'none',  label: 'None',        desc: 'Full precision. More VRAM, best quality.',                  color: '#ffffff' },
  { value: '8bit',  label: '8-bit',       desc: 'LLM.int8(). ~2× VRAM savings.',                            color: 'var(--accent2)' },
  { value: '4bit',  label: '4-bit (QLoRA)', desc: 'NF4 + double quant. ~4× VRAM savings. Recommended.',    color: 'var(--accent)' },
]

function QuantSelector({ value, onChange }) {
  return (
    <div className="stack-sm" style={{ marginBottom: '1rem' }}>
      {QUANT_OPTIONS.map(opt => {
        const active = value === opt.value
        return (
          <button key={opt.value} onClick={() => onChange(opt.value)} style={{
            display: 'flex', alignItems: 'center', gap: '.75rem',
            padding: '14px 16px', borderRadius: 16,
            background: active ? `${opt.color}12` : 'rgba(29, 35, 43, 0.82)',
            border: `1px solid ${active ? opt.color + '55' : 'rgba(49, 56, 70, 0.84)'}`,
            cursor: 'pointer', textAlign: 'left', transition: 'all .15s',
          }}>
            <span style={{
              width: 8, height: 8, borderRadius: '50%', flexShrink: 0,
              background: active ? opt.color : 'var(--border2)',
              boxShadow: active ? `0 0 0 6px ${opt.color}18` : 'none', transition: 'all .15s',
            }} />
            <div>
              <div style={{ fontSize: 14, fontWeight: active ? 700 : 500, color: active ? opt.color : 'var(--text)' }}>{opt.label}</div>
              <div style={{ fontSize: 12, color: 'var(--text3)', marginTop: 4 }}>{opt.desc}</div>
            </div>
          </button>
        )
      })}
    </div>
  )
}

// ── Param Tooltip ──────────────────────────────────────────────────────────
const PARAM_TIPS = {
  epochs:     'How many times the model sees your entire dataset. More epochs = more learning, but risk overfitting (memorising rather than generalising). Start with 2–4.',
  batch:      'How many samples the model learns from in one step. Larger batch = more stable learning but uses more GPU memory. If you hit OOM errors, reduce this first.',
  learningRate: 'How fast the model updates its knowledge. Too high = unstable training (loss spikes). Too low = slow or no improvement. 2e-4 is a safe default for QLoRA.',
  gradAcc:    'Simulates a larger batch by accumulating gradients over N steps before updating. Lets you run large effective batches on small GPUs without OOM.',
  warmup:     'Gradually ramps learning rate up for the first X% of training, preventing large early updates that can destabilise the model.',
  gradClip:   'Caps the size of gradient updates to prevent "gradient explosions" that crash training. 1.0 is almost always fine.',
  weightDecay:'Adds a small penalty for large weights, helping the model generalise better rather than overfit to training data.',
  optimizer:  'The algorithm that updates model weights. AdamW is the standard choice. Lion uses less memory but is more sensitive to learning rate.',
  mixedPrec:  'Use half-precision math to speed up training and save VRAM. bf16 is preferred on A10G. fp16 is more compatible with T4.',
  seq:        'Maximum number of tokens processed per sample. Longer = handles longer documents but uses exponentially more memory. Match to your typical document length.',
  packing:    'Packs multiple short samples into one sequence to avoid wasting context space. Speeds up training on short-text datasets.',
  loraR:      'LoRA rank: controls how many new parameters are added. Higher rank = more capacity but more memory and slower training. 16–32 is typical.',
  loraAlpha:  'Scales the LoRA update magnitude. Usually set to 2× the rank. Higher = stronger adaptation effect.',
  loraDrop:   'Randomly disables some LoRA connections during training to prevent overfitting. 0.05–0.1 is a safe range.',
  targetMods: 'Which attention layers LoRA adapts. q_proj and v_proj are the standard default. Adding k_proj and o_proj gives more capacity.',
  useRslora:  'Improved rank-normalised LoRA scaling from 2024. Recommended for ranks > 16 — helps training stability.',
  splitRatio: 'Splits your dataset into training and validation sets. More training data = better learning; more validation = better early stopping accuracy.',
  shuffle:    'Randomises the order samples are seen each epoch. Almost always beneficial — prevents the model from learning order patterns.',
  evalSteps:  'How often to check validation loss during training. More frequent = earlier detection of overfitting. Too frequent = slower training.',
  esPat:      'Stops training automatically if validation loss does not improve for N consecutive evaluations. Prevents wasted compute on overfit models.',
  saveSteps:  'How often to save a checkpoint. Lower = more checkpoints (larger storage). Useful for recovery if training crashes.',
  mergeWeights:'After training, merges the LoRA weights back into the base model for simpler deployment. Slightly larger file but no adapter needed at inference.',
  pushHub:    'Automatically uploads your trained model to Hugging Face Hub for easy sharing and inference.',
}

function ParamTooltip({ tipKey }) {
  const [show, setShow] = useState(false)
  const tip = PARAM_TIPS[tipKey]
  if (!tip) return null
  return (
    <span style={{ position: 'relative', display: 'inline-flex', alignItems: 'center' }}>
      <button
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
        aria-label="Parameter help"
      >?</button>
      {show && (
        <div style={{
          position: 'absolute', bottom: '120%', left: '50%', transform: 'translateX(-50%)',
          width: 260, background: 'rgba(23,27,33,0.98)', border: '1px solid var(--border)',
          borderRadius: 12, padding: '10px 12px', fontSize: 12, color: 'var(--text2)',
          lineHeight: 1.6, zIndex: 999, boxShadow: '0 8px 32px rgba(0,0,0,.5)',
          pointerEvents: 'none',
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

// ── Dataset Analysis Section ───────────────────────────────────────────────
function DatasetAnalyzer({ dataset, modelId, gpu, quantization, onApply }) {
  const [loading, setLoading] = useState(false)
  const [result, setResult] = useState(null)
  const [error, setError] = useState(null)
  const [open, setOpen] = useState(true)

  async function analyze() {
    if (!dataset) { alert('Upload a dataset first'); return }
    setLoading(true)
    setError(null)
    setResult(null)
    try {
      const res = await api.analyzeTrainingDataset({
        dataset_path: dataset.path,
        model_id: modelId,
        gpu_type: gpu,
        quantization,
      })
      setResult(res)
    } catch (e) {
      setError(e.message)
    } finally {
      setLoading(false)
    }
  }

  return (
    <Card>
      {/* Header */}
      <div
        onClick={() => setOpen(o => !o)}
        style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', cursor: 'pointer', marginBottom: open ? '1rem' : 0 }}
      >
        <div style={{ display: 'flex', alignItems: 'center', gap: '.5rem' }}>
          <CardTitle style={{ margin: 0 }}>4. Smart Dataset Analysis</CardTitle>
          <span style={{
            fontSize: 10, padding: '3px 8px', borderRadius: 999,
            background: 'rgba(124,156,255,0.14)', color: 'var(--accent)', fontWeight: 700, border: '1px solid rgba(124,156,255,0.28)',
          }}>AI-powered</span>
        </div>
        <span style={{ fontSize: 14, color: 'var(--text3)', userSelect: 'none' }}>{open ? '▲' : '▼'}</span>
      </div>

      {open && (
        <div className="stack-md">
          <p style={{ fontSize: 13, color: 'var(--text3)', margin: 0, lineHeight: 1.6 }}>
            Analyses your dataset structure (row count, token lengths, field density) using Python — without sending your data to an LLM.
            Then recommends safe hyperparameters for your chosen model and GPU.
          </p>

          {!dataset && (
            <div className="surface-muted" style={{ fontSize: 13, color: 'var(--text3)', padding: '12px 14px' }}>
              Upload a dataset above to enable analysis.
            </div>
          )}

          {dataset && (
            <div style={{ display: 'flex', gap: '.5rem', alignItems: 'center', flexWrap: 'wrap' }}>
              <div className="surface-muted" style={{ fontSize: 12, color: 'var(--text2)', padding: '8px 14px', flex: 1, minWidth: 180 }}>
                <span className="mono">{dataset.row_count} rows</span>
                <span style={{ color: 'var(--text3)', margin: '0 .4rem' }}>·</span>
                <span className="mono">{(dataset.size_bytes / 1024).toFixed(1)} KB</span>
                <span style={{ color: 'var(--text3)', margin: '0 .4rem' }}>·</span>
                <span style={{ color: 'var(--text3)' }}>{modelId?.split('/').pop() || 'no model'}</span>
                <span style={{ color: 'var(--text3)', margin: '0 .4rem' }}>·</span>
                <span style={{ color: 'var(--text3)' }}>{gpu}</span>
                <span style={{ color: 'var(--text3)', margin: '0 .4rem' }}>·</span>
                <span style={{ color: 'var(--text3)' }}>{quantization}</span>
              </div>
              <Btn
                variant="secondary"
                onClick={analyze}
                disabled={loading}
                style={{ flexShrink: 0 }}
              >
                {loading ? 'Analysing…' : '🔍 Analyse & Recommend'}
              </Btn>
            </div>
          )}

          {error && (
            <div className="surface-muted" style={{ color: 'var(--red)', fontSize: 13, padding: '12px 14px' }}>
              Error: {error}
            </div>
          )}

          {loading && (
            <div className="surface-muted" style={{ fontSize: 13, color: 'var(--text3)', padding: '14px', textAlign: 'center' }}>
              <div style={{ marginBottom: '.5rem' }}>Running structural checks…</div>
              <div style={{
                width: 200, height: 4, borderRadius: 2, background: 'var(--bg4)',
                margin: '0 auto', overflow: 'hidden',
              }}>
                <div style={{
                  height: '100%', background: 'linear-gradient(90deg, var(--accent), var(--accent2))',
                  borderRadius: 2, animation: 'loading-bar 1.5s ease-in-out infinite',
                  width: '40%',
                }} />
              </div>
              <div style={{ marginTop: '.5rem', fontSize: 12 }}>Then generating safe parameter recommendations…</div>
            </div>
          )}

          {result && !loading && (
            <div className="stack-md">
              {/* Structural stats from Python */}
              <div>
                <SectionLabel style={{ marginTop: 0 }}>Dataset Structure (Python analysis)</SectionLabel>
                <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: '.5rem' }}>
                  {[
                    { label: 'Total rows',      value: result.stats?.row_count ?? dataset.row_count },
                    { label: 'Avg tokens/row',  value: result.stats?.avg_tokens != null ? Math.round(result.stats.avg_tokens) : '—' },
                    { label: 'Max tokens/row',  value: result.stats?.max_tokens != null ? result.stats.max_tokens : '—' },
                    { label: 'P95 tokens',      value: result.stats?.p95_tokens != null ? result.stats.p95_tokens : '—' },
                    { label: 'Est. train rows', value: result.stats?.train_rows ?? '—' },
                    { label: 'Est. val rows',   value: result.stats?.val_rows ?? '—' },
                  ].map(s => (
                    <div key={s.label} style={{
                      background: 'var(--bg3)', border: '1px solid var(--border)',
                      borderRadius: 10, padding: '10px 12px',
                    }}>
                      <div style={{ fontSize: 11, color: 'var(--text3)', marginBottom: '.2rem' }}>{s.label}</div>
                      <div className="mono" style={{ fontSize: 16, fontWeight: 700, color: 'var(--text)' }}>{s.value}</div>
                    </div>
                  ))}
                </div>
              </div>

              {/* Gemini recommendation */}
              {result.recommendation && (
                <div>
                  <SectionLabel>Parameter Recommendations</SectionLabel>
                  <div style={{
                    background: 'rgba(124,156,255,0.05)', border: '1px solid rgba(124,156,255,0.2)',
                    borderRadius: 14, padding: '14px 16px',
                  }}>
                    {/* Reasoning text */}
                    {result.recommendation.reasoning && (
                      <p style={{ fontSize: 13, color: 'var(--text2)', marginBottom: '1rem', lineHeight: 1.7, marginTop: 0 }}>
                        {result.recommendation.reasoning}
                      </p>
                    )}

                    {/* Recommended param chips */}
                    {result.recommendation.params && (
                      <div style={{ display: 'flex', flexWrap: 'wrap', gap: '.4rem', marginBottom: '1rem' }}>
                        {Object.entries(result.recommendation.params).map(([k, v]) => (
                          <span key={k} style={{
                            fontSize: 12, padding: '4px 10px', borderRadius: 999,
                            background: 'rgba(124,156,255,0.12)', color: 'var(--accent)',
                            border: '1px solid rgba(124,156,255,0.25)', fontFamily: 'var(--mono)',
                          }}>
                            {k}: <strong>{String(v)}</strong>
                          </span>
                        ))}
                      </div>
                    )}

                    {/* OOM warning if any */}
                    {result.recommendation.oom_warning && (
                      <div style={{
                        fontSize: 12, color: 'var(--orange)', padding: '8px 12px',
                        background: 'rgba(251,146,60,0.08)', borderRadius: 8, marginBottom: '1rem',
                        border: '1px solid rgba(251,146,60,0.22)',
                      }}>
                        ⚠ {result.recommendation.oom_warning}
                      </div>
                    )}

                    {/* Apply button */}
                    {result.recommendation.params && onApply && (
                      <Btn
                        variant="primary"
                        size="sm"
                        onClick={() => onApply(result.recommendation.params)}
                      >
                        ✓ Apply Recommended Settings
                      </Btn>
                    )}
                  </div>
                </div>
              )}
            </div>
          )}
        </div>
      )}
    </Card>
  )
}

// ── Sample dataset constant ────────────────────────────────────────────────
const SAMPLE_TRAIN_DATASET = {
  filename: 'training_data.jsonl',
  row_count: 93,
  size_bytes: 185000,
  path: '__sample_train__',
  preview: [],
}

// ── Main Component ─────────────────────────────────────────────────────────
export default function TrainConfig({ onJobStart, onDatasetReady, prefilledDatasetPath }) {
  const [dataset, setDataset] = useState(null)
  const [uploading, setUploading] = useState(false)
  const [dsFormat, setDsFormat] = useState('chat')
  const [dragging, setDragging] = useState(false)

  const [modelId, setModelId] = useState('Qwen/Qwen2.5-1.5B-Instruct')
  const [gpu, setGpu] = useState('A10G')
  const [runName, setRunName] = useState('my-finetune-run')
  const [notes, setNotes] = useState('')

  const [quantization, setQuantization] = useState('4bit')

  const [epochs, setEpochs] = useState(3)
  const [learningRate, setLearningRate] = useState(0.00002)
  const [batch, setBatch] = useState(4)
  const [gradAcc, setGradAcc] = useState(4)
  const [warmup, setWarmup] = useState(0.03)
  const [maxSteps, setMaxSteps] = useState(-1)
  const [gradClip, setGradClip] = useState(1.0)
  const [weightDecay, setWeightDecay] = useState(0.1)
  const [optimizer, setOptimizer] = useState('adamw_torch')
  const [mixedPrec, setMixedPrec] = useState('bf16')
  const [seq, setSeq] = useState(2048)
  const [packing, setPacking] = useState(false)
  const [showAdv, setShowAdv] = useState(false)

  const [loraR, setLoraR] = useState(16)
  const [loraAlpha, setLoraAlpha] = useState(32)
  const [loraDrop, setLoraDrop] = useState(0.05)
  const [targetMods, setTargetMods] = useState('q_proj,v_proj')
  const [useRslora, setUseRslora] = useState(false)

  const [splitRatio, setSplitRatio] = useState(90)
  const [shuffle, setShuffle] = useState(true)
  const [evalSteps, setEvalSteps] = useState(50)
  const [esPat, setEsPat] = useState(3)

  const [saveSteps, setSaveSteps] = useState(100)
  const [mergeWeights, setMergeWeights] = useState(true)
  const [pushHub, setPushHub] = useState(false)
  const [hfToken, setHfToken] = useState('')

  const [launching, setLaunching] = useState(false)

  useEffect(() => {
    if (!prefilledDatasetPath) return
    if (prefilledDatasetPath === '__sample_train__') { setDataset(SAMPLE_TRAIN_DATASET); return }
    let cancelled = false
    ;(async () => {
      try {
        const data = await api.datasetInfo(prefilledDatasetPath)
        if (!cancelled) setDataset(data)
      } catch (e) { console.warn('Could not load prefilled dataset info:', e.message) }
    })()
    return () => { cancelled = true }
  }, [prefilledDatasetPath])

  const uploadFile = useCallback(async (file) => {
    if (!file) return
    setUploading(true)
    try {
      const data = await api.uploadDataset(file)
      setDataset(data)
      onDatasetReady?.(data.path)
    } catch (e) {
      alert(`Upload failed: ${e.message}`)
    } finally {
      setUploading(false)
    }
  }, [onDatasetReady])

  const handleDrop = useCallback((e) => {
    e.preventDefault(); setDragging(false)
    uploadFile(e.dataTransfer.files[0])
  }, [uploadFile])

  const is4b = modelId.includes('4b') || modelId.includes('3B')
  const gpuWarn = is4b && gpu === 'T4'

  function applyPreset(key) {
    const p = PRESETS[key]
    setEpochs(p.epochs); setLearningRate(idxToLr(p.lrIdx)); setBatch(p.batch)
    setGradAcc(p.gradAcc); setLoraR(p.loraR); setLoraAlpha(p.loraAlpha)
    setWarmup(p.warmup / 100); setSeq(p.seq); setQuantization(p.quantization)
  }

  // Apply recommended settings from dataset analysis
  function applyRecommended(params) {
    if (params.epochs !== undefined)       setEpochs(Number(params.epochs))
    if (params.learning_rate !== undefined) setLearningRate(Number(params.learning_rate))
    if (params.batch_size !== undefined)   setBatch(Number(params.batch_size))
    if (params.grad_accumulation !== undefined) setGradAcc(Number(params.grad_accumulation))
    if (params.max_seq_length !== undefined) setSeq(Number(params.max_seq_length))
    if (params.lora_r !== undefined)       setLoraR(Number(params.lora_r))
    if (params.lora_alpha !== undefined)   setLoraAlpha(Number(params.lora_alpha))
    if (params.warmup_ratio !== undefined) setWarmup(Number(params.warmup_ratio))
    if (params.quantization !== undefined) setQuantization(params.quantization)
    if (params.max_steps !== undefined)    setMaxSteps(Number(params.max_steps))
    if (params.merge_weights !== undefined) {
      setMergeWeights(params.merge_weights === true || params.merge_weights === 'true')
    }
  }

  async function launch() {
    if (!dataset) { alert('Upload a dataset first'); return }
    setLaunching(true)
    try {
      const config = {
        model_id: modelId, gpu_type: gpu, dataset_path: dataset.path,
        dataset_format: dsFormat, run_name: runName, notes,
        train_split: splitRatio / 100, shuffle, quantization,
        lora_r: loraR, lora_alpha: loraAlpha, lora_dropout: loraDrop,
        target_modules: targetMods.split(',').map(s => s.trim()).filter(Boolean),
        use_rslora: useRslora, learning_rate: learningRate, batch_size: batch,
        grad_accumulation: gradAcc, epochs, max_steps: maxSteps,
        warmup_ratio: warmup, grad_clip: gradClip, weight_decay: weightDecay,
        optimizer, mixed_precision: mixedPrec, max_seq_length: seq, packing,
        save_steps: saveSteps, eval_steps: evalSteps, early_stopping_patience: esPat,
        merge_weights: mergeWeights, push_to_hub: pushHub, hf_token: hfToken || null,
      }
      const res = await api.startTraining(config)
      onJobStart(res.job_id)
    } catch (e) {
      alert(`Launch failed: ${e.message}`)
    } finally {
      setLaunching(false)
    }
  }

  return (
    <div className="stack-lg">
      <div className="split-grid" style={{ alignItems: 'start' }}>

        {/* ── LEFT COLUMN ── */}
        <div className="stack-lg">

          {/* 1. Dataset */}
          <Card>
            <CardTitle>1. Dataset</CardTitle>

            {/* Sample dataset pill */}
            {!dataset && (
              <div style={{ marginBottom: '.6rem' }}>
                <button
                  onClick={() => { setDataset(SAMPLE_TRAIN_DATASET); onDatasetReady?.('__sample_train__') }}
                  style={{
                    fontSize: 12, padding: '5px 12px', borderRadius: 999, cursor: 'pointer',
                    background: 'rgba(124,156,255,0.10)', color: 'var(--accent)',
                    border: '1px solid rgba(124,156,255,0.28)', fontWeight: 600,
                    display: 'inline-flex', alignItems: 'center', gap: 6,
                  }}
                >
                  <span style={{ fontSize: 14 }}>📂</span> Use sample training dataset (93 rows)
                </button>
              </div>
            )}

            <div
              onDragOver={e => { e.preventDefault(); setDragging(true) }}
              onDragLeave={() => setDragging(false)}
              onDrop={handleDrop}
              onClick={() => document.getElementById('file-upload').click()}
              className={`dropzone${dragging ? ' is-active' : ''}`}
              style={{ cursor: 'pointer' }}
            >
              <input id="file-upload" type="file" accept=".json,.jsonl,.csv,.txt" style={{ display: 'none' }}
                onChange={e => uploadFile(e.target.files[0])} />
              {uploading ? (
                <p className="dropzone-title" style={{ color: 'var(--text2)' }}>Uploading dataset...</p>
              ) : dataset ? (
                <>
                  <div className="dropzone-title" style={{ color: 'var(--green)' }}>Dataset ready</div>
                  <p className="dropzone-subtitle" style={{ color: 'var(--text2)', overflow: 'hidden', textOverflow: 'ellipsis' }}>{dataset.filename}</p>
                  <p style={{ color: 'var(--text3)', fontSize: 12, marginTop: 8 }}>
                    {dataset.row_count.toLocaleString()} rows · {(dataset.size_bytes / 1024).toFixed(1)} KB
                  </p>
                  <button
                    onClick={e => { e.stopPropagation(); setDataset(null) }}
                    style={{ marginTop: 8, fontSize: 11, color: 'var(--text3)', background: 'none', border: 'none', cursor: 'pointer', textDecoration: 'underline' }}
                  >Remove</button>
                </>
              ) : (
                <>
                  <div className="dropzone-title" style={{ color: 'var(--text2)' }}>Upload a training dataset</div>
                  <p className="dropzone-subtitle">Drag a file here or click to browse</p>
                  <p style={{ fontSize: 12, color: 'var(--text4)', marginTop: 10 }}>.json · .jsonl · .csv · .txt</p>
                </>
              )}
            </div>

            {dataset && (
              <div style={{ marginTop: '1rem' }}>
                <div className="toolbar" style={{ marginBottom: '.5rem' }}>
                  <Label style={{ margin: 0 }}>Format</Label>
                  <div style={{ display: 'flex', gap: '.3rem' }}>
                    {['alpaca', 'sharegpt', 'chat', 'raw'].map(f => (
                      <button key={f} onClick={() => setDsFormat(f)} style={{
                        fontSize: 12, minHeight: 32, padding: '0 12px', borderRadius: 999,
                        background: dsFormat === f ? 'rgba(124, 156, 255, 0.14)' : 'rgba(29, 35, 43, 0.9)',
                        color: dsFormat === f ? 'var(--text)' : 'var(--text3)',
                        border: `1px solid ${dsFormat === f ? 'rgba(124, 156, 255, 0.32)' : 'rgba(49, 56, 70, 0.8)'}`, cursor: 'pointer', fontWeight: 600,
                      }}>{f}</button>
                    ))}
                  </div>
                </div>
                {dataset.preview?.length > 0 && <DatasetPreview rows={dataset.preview} />}
              </div>
            )}
          </Card>

          {/* 2. Model & Run Info */}
          <Card>
            <CardTitle>2. Base Model</CardTitle>
            <ModelGrid models={BASE_MODELS} selectedId={modelId} onSelect={m => setModelId(m.id)} />
            <div className="split-grid" style={{ gap: '.75rem', marginTop: '1.25rem' }}>
              <Field label="Run name" style={{ marginBottom: 0 }}>
                <Input value={runName} onChange={e => setRunName(e.target.value)} />
              </Field>
              <Field label="Notes" style={{ marginBottom: 0 }}>
                <Input value={notes} onChange={e => setNotes(e.target.value)} placeholder="Run details…" />
              </Field>
            </div>
          </Card>

          {/* 3. Hardware */}
          <Card>
            <CardTitle>3. Hardware</CardTitle>
            <div className="stack-md">
              <GpuSelector value={gpu} onChange={setGpu} />
              {gpuWarn && (
                <div className="surface-muted" style={{ fontSize: 13, color: 'var(--orange)', padding: '12px 14px' }}>
                  Higher-capacity models may run out of memory on T4 hardware.
                </div>
              )}
            </div>
          </Card>

          {/* 4. Smart Dataset Analysis */}
          <DatasetAnalyzer
            dataset={dataset}
            modelId={modelId}
            gpu={gpu}
            quantization={quantization}
            onApply={applyRecommended}
          />
        </div>

        {/* ── RIGHT COLUMN ── */}
        <div className="stack-lg">

          {/* 5. Initialization / Presets */}
          <Card>
            <CardTitle>5. Initialization</CardTitle>
            <SectionLabel style={{ marginTop: 0 }}>Quick presets</SectionLabel>
            <div style={{ display: 'flex', gap: '.4rem', flexWrap: 'wrap', marginBottom: '1rem' }}>
              {Object.keys(PRESETS).map(k => (
                <button key={k} onClick={() => applyPreset(k)} className="preset-chip">{k}</button>
              ))}
            </div>
            <Toggle label="Advanced configuration" checked={showAdv} onChange={setShowAdv} />
          </Card>

          {/* 6. Quantization */}
          <Card>
            <CardTitle>6. Quantization</CardTitle>
            <QuantSelector value={quantization} onChange={setQuantization} />
          </Card>

          {/* 7. Hyperparameters */}
          <ParamPanel
            showAdv={showAdv}
            epochs={epochs} setEpochs={setEpochs}
            learningRate={learningRate} setLearningRate={setLearningRate}
            batch={batch} setBatch={setBatch}
            gradAcc={gradAcc} setGradAcc={setGradAcc}
            warmup={warmup} setWarmup={setWarmup}
            gradClip={gradClip} setGradClip={setGradClip}
            weightDecay={weightDecay} setWeightDecay={setWeightDecay}
            optimizer={optimizer} setOptimizer={setOptimizer}
            mixedPrec={mixedPrec} setMixedPrec={setMixedPrec}
            seq={seq} setSeq={setSeq}
            packing={packing} setPacking={setPacking}
            loraR={loraR} setLoraR={setLoraR}
            loraAlpha={loraAlpha} setLoraAlpha={setLoraAlpha}
            loraDrop={loraDrop} setLoraDrop={setLoraDrop}
            targetMods={targetMods} setTargetMods={setTargetMods}
            useRslora={useRslora} setUseRslora={setUseRslora}
            splitRatio={splitRatio} setSplitRatio={setSplitRatio}
            shuffle={shuffle} setShuffle={setShuffle}
            evalSteps={evalSteps} setEvalSteps={setEvalSteps}
            esPat={esPat} setEsPat={setEsPat}
            saveSteps={saveSteps} setSaveSteps={setSaveSteps}
            mergeWeights={mergeWeights} setMergeWeights={setMergeWeights}
            pushHub={pushHub} setPushHub={setPushHub}
            hfToken={hfToken} setHfToken={setHfToken}
          />

          {/* 8. Launch */}
          <div style={{ marginTop: 'auto', paddingTop: '.5rem' }}>
            <Btn variant="primary" size="lg" onClick={launch} disabled={launching || !dataset} style={{ width: '100%', padding: '.85rem' }}>
              {launching ? 'Submitting to Modal…' : 'Launch Training Job'}
            </Btn>
            {!dataset && (
              <p style={{ textAlign: 'center', fontSize: 12, color: 'var(--text3)', marginTop: '.55rem' }}>
                Upload a dataset to begin
              </p>
            )}
          </div>
        </div>
      </div>
    </div>
  )
}

// ── Dataset Preview ────────────────────────────────────────────────────────
function DatasetPreview({ rows }) {
  const keys = Object.keys(rows[0]).slice(0, 4)
  return (
    <div style={{ marginTop: '.75rem' }}>
      <p style={{ fontSize: 13, color: 'var(--text3)', marginBottom: '.5rem' }}>Preview ({rows.length} rows)</p>
      <div className="table-wrap">
        <table className="data-table">
          <thead>
            <tr>{keys.map(k => <th key={k}>{k}</th>)}</tr>
          </thead>
          <tbody>
            {rows.map((row, i) => (
              <tr key={i}>{keys.map(k => {
                const val = row[k]
                const display = typeof val === 'object' ? JSON.stringify(val) : String(val ?? '')
                const v = display.slice(0, 50)
                return <td key={k} title={display} style={{ maxWidth: 160, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>{v}{display.length > 50 ? '…' : ''}</td>
              })}</tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  )
}

// ── Param Panel with ? tooltips ────────────────────────────────────────────
function LabelWithTip({ children, tipKey }) {
  return (
    <span style={{ display: 'inline-flex', alignItems: 'center' }}>
      {children}
      <ParamTooltip tipKey={tipKey} />
    </span>
  )
}

function ParamPanel(p) {
  const [innerTab, setInnerTab] = useState('training')
  const tabs = [
    { id: 'training', label: 'Training' },
    { id: 'lora',     label: 'LoRA / PEFT' },
    { id: 'data',     label: 'Data' },
    { id: 'output',   label: 'Output' },
  ]

  return (
    <Card>
      <InnerTabs tabs={tabs} active={innerTab} onChange={setInnerTab} />

      {innerTab === 'training' && (
        <div>
          <div className="split-grid" style={{ gap: '1rem' }}>
            <Slider label={<LabelWithTip tipKey="epochs">Epochs</LabelWithTip>} min={1} max={10} value={p.epochs} onChange={p.setEpochs} />
            <Slider label={<LabelWithTip tipKey="batch">Batch size</LabelWithTip>} min={1} max={32} value={p.batch} onChange={p.setBatch} />
          </div>
          <Slider
            label={<LabelWithTip tipKey="learningRate">Learning rate</LabelWithTip>}
            min={1e-6} max={1e-1} step={0.000001}
            value={p.learningRate}
            displayValue={p.learningRate < 0.0001 ? p.learningRate.toExponential(1) : p.learningRate.toFixed(5)}
            onChange={p.setLearningRate}
            isLogarithmic={{ minIdx: 0, maxIdx: 100, toIdx: lrToIdx, fromIdx: idxToLr }}
          />
          {p.showAdv && <>
            <div className="split-grid" style={{ gap: '1rem' }}>
              <Slider label={<LabelWithTip tipKey="gradAcc">Grad accumulation</LabelWithTip>} min={1} max={32} value={p.gradAcc} onChange={p.setGradAcc} />
              <Slider label={<LabelWithTip tipKey="warmup">Warmup ratio</LabelWithTip>} min={0} max={0.5} step={0.01} value={p.warmup} onChange={p.setWarmup} />
            </div>
            <div className="split-grid" style={{ gap: '1rem' }}>
              <Slider label={<LabelWithTip tipKey="gradClip">Grad clip</LabelWithTip>} min={0} max={10} step={0.1} value={p.gradClip} onChange={p.setGradClip} />
              <Slider label={<LabelWithTip tipKey="weightDecay">Weight decay</LabelWithTip>} min={0} max={1} step={0.01} value={p.weightDecay} onChange={p.setWeightDecay} />
            </div>
            <div className="split-grid" style={{ gap: '1rem' }}>
              <Field label={<LabelWithTip tipKey="optimizer">Optimizer</LabelWithTip>} style={{ margin: 0 }}>
                <Select value={p.optimizer} onChange={e => p.setOptimizer(e.target.value)}>
                  <option value="adamw_torch">AdamW</option>
                  <option value="lion_8bit">Lion (8-bit)</option>
                  <option value="sgd">SGD</option>
                </Select>
              </Field>
              <Field label={<LabelWithTip tipKey="mixedPrec">Mixed precision</LabelWithTip>} style={{ margin: 0 }}>
                <Select value={p.mixedPrec} onChange={e => p.setMixedPrec(e.target.value)}>
                  <option value="bf16">bf16 (A10G recommended)</option>
                  <option value="fp16">fp16 (T4 compatible)</option>
                </Select>
              </Field>
            </div>
            <div className="split-grid" style={{ gap: '1rem', marginTop: '.5rem' }}>
              <Slider label={<LabelWithTip tipKey="seq">Max seq length</LabelWithTip>} min={64} max={16384} step={64} value={p.seq} onChange={p.setSeq} />
            </div>
            <div style={{ marginTop: '.5rem' }}>
              <Toggle label={<LabelWithTip tipKey="packing">Sequence packing (bin-pack short samples to fill context)</LabelWithTip>} checked={p.packing} onChange={p.setPacking} />
            </div>
          </>}
          <div className="surface-muted" style={{ fontSize: 13, color: 'var(--text3)', padding: '12px 14px', marginTop: '.25rem' }}>
            Effective batch size: <span className="mono" style={{ color: 'var(--accent)' }}>{p.batch * p.gradAcc}</span>
          </div>
        </div>
      )}

      {innerTab === 'lora' && (
        <div>
          <div className="split-grid" style={{ gap: '1rem' }}>
            <Slider label={<LabelWithTip tipKey="loraR">LoRA rank (r)</LabelWithTip>} min={4} max={128} step={4} value={p.loraR} onChange={p.setLoraR} />
            <Slider label={<LabelWithTip tipKey="loraAlpha">LoRA alpha</LabelWithTip>} min={4} max={256} step={4} value={p.loraAlpha} onChange={p.setLoraAlpha} />
          </div>
          <Slider label={<LabelWithTip tipKey="loraDrop">LoRA dropout</LabelWithTip>} min={0} max={1} step={0.01} value={p.loraDrop} onChange={p.setLoraDrop} />
          <Field label={<LabelWithTip tipKey="targetMods">Target modules (comma-separated)</LabelWithTip>}>
            <Input value={p.targetMods} onChange={e => p.setTargetMods(e.target.value)} />
          </Field>
          <Toggle label={<LabelWithTip tipKey="useRslora">Use RSLoRA (improved rank scaling)</LabelWithTip>} checked={p.useRslora} onChange={p.setUseRslora} />
          <div className="surface-muted mono" style={{ fontSize: 12, color: 'var(--text3)', padding: '12px 14px', marginTop: '.5rem' }}>
            Effective LoRA scale: <span style={{ color: 'var(--accent)' }}>
              {p.useRslora ? (p.loraAlpha / Math.sqrt(p.loraR)).toFixed(2) : (p.loraAlpha / p.loraR).toFixed(2)}
            </span>
            {' '}· Trainable param ratio scales with r²
          </div>
        </div>
      )}

      {innerTab === 'data' && (
        <div>
          <Slider label={<LabelWithTip tipKey="splitRatio">Train / val split</LabelWithTip>} min={60} max={99} value={p.splitRatio} displayValue={`${p.splitRatio}% / ${100 - p.splitRatio}%`} onChange={p.setSplitRatio} />
          <Toggle label={<LabelWithTip tipKey="shuffle">Shuffle dataset before training</LabelWithTip>} checked={p.shuffle} onChange={p.setShuffle} />
          <div className="split-grid" style={{ gap: '1rem' }}>
            <Field label={<LabelWithTip tipKey="evalSteps">Eval every N steps</LabelWithTip>} style={{ margin: 0 }}>
              <Input type="number" value={p.evalSteps} onChange={e => p.setEvalSteps(Number(e.target.value))} />
            </Field>
            <Field label={<LabelWithTip tipKey="esPat">Early stopping patience</LabelWithTip>} style={{ margin: 0 }}>
              <Input type="number" value={p.esPat} onChange={e => p.setEsPat(Number(e.target.value))} />
            </Field>
          </div>
        </div>
      )}

      {innerTab === 'output' && (
        <div>
          <Field label={<LabelWithTip tipKey="saveSteps">Save checkpoint every N steps</LabelWithTip>}>
            <Input type="number" value={p.saveSteps} onChange={e => p.setSaveSteps(Number(e.target.value))} />
          </Field>
          <Toggle label={<LabelWithTip tipKey="mergeWeights">Merge LoRA weights after training</LabelWithTip>} checked={p.mergeWeights} onChange={p.setMergeWeights} />
          <Toggle label={<LabelWithTip tipKey="pushHub">Push to Hugging Face Hub</LabelWithTip>} checked={p.pushHub} onChange={p.setPushHub} />
          {p.pushHub && (
            <Field label="HuggingFace token">
              <Input type="password" value={p.hfToken} onChange={e => p.setHfToken(e.target.value)} placeholder="hf_…" />
            </Field>
          )}
        </div>
      )}
    </Card>
  )
}
