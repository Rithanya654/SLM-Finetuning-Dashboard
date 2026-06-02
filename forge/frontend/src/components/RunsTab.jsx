import React, { useState, useMemo } from 'react'
import {
  AreaChart, Area, XAxis, YAxis, CartesianGrid,
  Tooltip, ResponsiveContainer,
} from 'recharts'

// ─── Simulated 20-epoch demo run (pinned, never removable) ────────────────────
function buildDemoMetrics() {
  const trainLoss = []
  const valLoss = []
  const totalSteps = 440
  const stepsPerEpoch = totalSteps / 20

  for (let i = 0; i <= totalSteps; i += 10) {
    const epoch = i / stepsPerEpoch
    const trainBase = 0.28 + 1.55 * Math.exp(-0.38 * epoch)
    const trainNoise = (Math.random() - 0.5) * 0.045
    const train = Math.max(0.24, trainBase + trainNoise)
    trainLoss.push({ step: i, value: parseFloat(train.toFixed(4)) })

    const valBase = 0.34 + 1.42 * Math.exp(-0.33 * epoch) + (epoch > 16 ? (epoch - 16) * 0.004 : 0)
    const valNoise = (Math.random() - 0.5) * 0.038
    const val = Math.max(0.31, valBase + valNoise)
    valLoss.push({ step: i, value: parseFloat(val.toFixed(4)) })
  }

  return { train_loss: trainLoss, val_loss: valLoss }
}

const DEMO_RUN = {
  id: 'demo-20ep-qwen25-3b',
  _isPinned: true,
  model_id: 'qwen2.5-3b-invoice-v1',
  base_model: 'Qwen/Qwen2.5-3B-Instruct',
  run_name: 'invoice-extraction-v1',
  status: 'completed',
  gpu: 'A10G',
  quantization: '8bit',
  epochs: 20,
  lora_r: 32,
  lora_alpha: 64,
  learning_rate: '8e-5',
  batch_size: 2,
  grad_accumulation: 8,
  max_seq_length: 4096,
  samples: 94,
  final_train_loss: 0.2841,
  final_val_loss: 0.3512,
  duration_mins: 47,
  cost_usd: 1.84,
  completed_at: new Date(Date.now() - 2 * 24 * 60 * 60 * 1000).toISOString(),
  metrics: buildDemoMetrics(),
}

// ─── Helpers ──────────────────────────────────────────────────────────────────
function fmt(iso) {
  if (!iso) return '—'
  const d = new Date(iso)
  return d.toLocaleDateString('en-US', { month: 'short', day: 'numeric', year: 'numeric' }) +
    ' ' + d.toLocaleTimeString('en-US', { hour: '2-digit', minute: '2-digit' })
}

function statusColor(s) {
  if (s === 'completed') return 'var(--accent)'
  if (s === 'failed') return '#ffffff'
  if (s === 'cancelled') return 'var(--text3)'
  return 'var(--accent2)'
}

// ─── Loss chart ───────────────────────────────────────────────────────────────
function LossChart({ metrics }) {
  const tl = metrics?.train_loss || []
  const vl = metrics?.val_loss || []
  const byStep = new Map()
  tl.forEach(pt => {
    const step = pt.step ?? 0
    byStep.set(step, { ...(byStep.get(step) || { step }), train: pt.value })
  })
  vl.forEach(pt => {
    const step = pt.step ?? 0
    byStep.set(step, { ...(byStep.get(step) || { step }), val: pt.value })
  })
  const data = [...byStep.values()].sort((a, b) => a.step - b.step)

  if (data.length === 0) {
    return (
      <div style={{ color: 'var(--text3)', fontSize: 13, padding: '1rem 0' }}>
        No loss data recorded for this run.
      </div>
    )
  }

  const finalTrain = tl[tl.length - 1]?.value
  const finalVal = vl[vl.length - 1]?.value
  const finalStep = Math.max(tl[tl.length - 1]?.step || 0, vl[vl.length - 1]?.step || 0)
  const hasSinglePoint = data.length === 1

  return (
    <div>
      {hasSinglePoint && (
        <div style={{
          fontSize: 12, color: 'var(--text3)', padding: '8px 12px',
          background: 'rgba(124,156,255,.06)', border: '1px solid rgba(124,156,255,.16)',
          borderRadius: 10, marginBottom: '.85rem',
        }}>
          Only one logging point was captured for this short demo run, so the chart shows the final recorded point.
        </div>
      )}

      {/* Mini stat row */}
      <div style={{ display: 'flex', gap: '1.5rem', marginBottom: '.85rem', flexWrap: 'wrap' }}>
        {[
          { label: 'Final Train Loss', value: finalTrain?.toFixed(4), color: 'var(--accent)' },
          { label: 'Final Val Loss', value: finalVal?.toFixed(4), color: 'var(--accent2)' },
          { label: 'Steps', value: finalStep || '—', color: '#ffffff' },
        ].map(s => (
          <div key={s.label}>
            <div style={{ fontSize: 12, color: 'var(--text3)', marginBottom: '.25rem' }}>{s.label}</div>
            <div className="mono" style={{ fontSize: 22, fontWeight: 700, color: s.color }}>{s.value ?? '—'}</div>
          </div>
        ))}
      </div>

      {/* Legend */}
      <div style={{ display: 'flex', gap: '1rem', marginBottom: '.75rem', fontSize: 12, color: 'var(--text3)', flexWrap: 'wrap' }}>
        <span style={{ display: 'flex', alignItems: 'center', gap: '.35rem' }}>
          <span style={{ display: 'inline-block', width: 16, height: 2, background: 'var(--accent)', borderRadius: 1 }} />
          Train loss
        </span>
        <span style={{ display: 'flex', alignItems: 'center', gap: '.35rem' }}>
          <span style={{ display: 'inline-block', width: 16, height: 2, background: 'var(--accent2)', borderRadius: 1 }} />
          Val loss
        </span>
      </div>

      <ResponsiveContainer width="100%" height={220}>
        <AreaChart data={data} margin={{ top: 4, right: 8, bottom: 0, left: -20 }}>
          <defs>
            <linearGradient id="trainGrad" x1="0" y1="0" x2="0" y2="1">
              <stop offset="5%" stopColor="var(--accent)" stopOpacity={0.22} />
              <stop offset="95%" stopColor="var(--accent)" stopOpacity={0} />
            </linearGradient>
            <linearGradient id="valGrad" x1="0" y1="0" x2="0" y2="1">
              <stop offset="5%" stopColor="var(--accent2)" stopOpacity={0.15} />
              <stop offset="95%" stopColor="var(--accent2)" stopOpacity={0} />
            </linearGradient>
          </defs>
          <CartesianGrid strokeDasharray="3 3" stroke="rgba(31,213,195,.10)" vertical={false} />
          <XAxis
            dataKey="step"
            tick={{ fontSize: 11, fill: '#94a0b2' }}
            minTickGap={30}
            label={{ value: 'Step', position: 'insideBottomRight', offset: -4, fontSize: 10, fill: '#94a0b2' }}
          />
          <YAxis tick={{ fontSize: 11, fill: '#94a0b2' }} />
          <Tooltip
            contentStyle={{
              background: 'rgba(8, 12, 11, 0.96)', border: '1px solid rgba(31, 213, 195, 0.28)',
              borderRadius: 14, fontSize: 12,
            }}
            labelStyle={{ color: 'var(--text2)' }}
            itemStyle={{ color: 'var(--text)' }}
          />
          <Area type="monotone" dataKey="train" stroke="var(--accent)" strokeWidth={2.2} fill="url(#trainGrad)" dot={hasSinglePoint ? { r: 4 } : false} name="Train" />
          <Area type="monotone" dataKey="val" stroke="var(--accent2)" strokeWidth={2.2} fill="url(#valGrad)" dot={hasSinglePoint ? { r: 4 } : false} name="Val" strokeDasharray="5 4" />
        </AreaChart>
      </ResponsiveContainer>
    </div>
  )
}

// ─── Config pill grid ─────────────────────────────────────────────────────────
function ConfigGrid({ run }) {
  const items = [
    { k: 'GPU', v: run.gpu },
    { k: 'Quantization', v: run.quantization },
    { k: 'Epochs', v: run.epochs },
    { k: 'LoRA r', v: run.lora_r },
    { k: 'LoRA α', v: run.lora_alpha },
    { k: 'LR', v: run.learning_rate },
    { k: 'Batch', v: run.batch_size },
    { k: 'Grad accum', v: run.grad_accumulation },
    { k: 'Seq len', v: run.max_seq_length },
    { k: 'Samples', v: run.samples },
    { k: 'Duration', v: run.duration_mins ? `${run.duration_mins}m` : '—' },
    { k: 'Cost', v: run.cost_usd ? `$${run.cost_usd.toFixed(2)}` : '—' },
  ].filter(i => i.v !== undefined && i.v !== null && i.v !== '')

  return (
    <div style={{ display: 'flex', flexWrap: 'wrap', gap: '.5rem', marginTop: '.9rem' }}>
      {items.map(({ k, v }) => (
        <div key={k} style={{
          fontSize: 12,
          background: 'rgba(29, 35, 43, 0.9)', border: '1px solid rgba(49, 56, 70, 0.84)',
          borderRadius: 999, padding: '6px 12px',
          color: 'var(--text2)',
          display: 'flex', gap: '.35rem', alignItems: 'center',
        }}>
          <span style={{ color: 'var(--text3)' }}>{k}</span>
          <span style={{ color: 'var(--accent2)', fontWeight: 600 }}>{v}</span>
        </div>
      ))}
    </div>
  )
}

// ─── Single run card ──────────────────────────────────────────────────────────
function RunCard({ run, isRemoved, onRemove }) {
  const [expanded, setExpanded] = useState(run._isPinned)

  return (
    <div style={{
      background: 'linear-gradient(180deg, rgba(23, 27, 33, 0.96), rgba(17, 20, 24, 0.98))',
      border: `1px solid ${run._isPinned ? 'rgba(124, 156, 255, 0.22)' : 'rgba(49, 56, 70, 0.8)'}`,
      borderRadius: 18,
      overflow: 'hidden',
      opacity: isRemoved ? 0.45 : 1,
      transition: 'opacity .2s, border-color .2s',
      boxShadow: 'var(--shadow-sm)',
    }}>
      {/* Header row */}
      <div
        onClick={() => setExpanded(e => !e)}
        style={{
          display: 'flex', alignItems: 'center', gap: '.75rem',
          padding: '16px 18px',
          cursor: 'pointer',
          background: expanded ? 'rgba(255,255,255,.015)' : 'transparent',
          transition: 'background .15s',
        }}
      >
        {/* Expand chevron */}
        <span style={{
          fontSize: 12, color: 'var(--text3)', transition: 'transform .2s',
          transform: expanded ? 'rotate(90deg)' : 'rotate(0deg)',
          flexShrink: 0,
        }}>{'>'}</span>

        {/* Model name + run name */}
        <div style={{ flex: 1, minWidth: 0 }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '.5rem', flexWrap: 'wrap' }}>
            <span className="mono" style={{ fontSize: 14, fontWeight: 700, color: 'var(--text)' }}>
              {run.model_id || run.run_name || run.id}
            </span>
            {run._isPinned && (
              <span style={{
                fontSize: 11, padding: '4px 10px',
                borderRadius: 999, background: 'rgba(124, 156, 255, 0.12)',
                border: '1px solid rgba(124, 156, 255, 0.24)', color: 'var(--accent2)',
              }}>Sample run</span>
            )}
            {isRemoved && (
              <span style={{
                fontSize: 11, padding: '4px 10px',
                borderRadius: 999, background: 'rgba(255,255,255,.05)',
                border: '1px solid rgba(255,255,255,.12)', color: 'var(--text3)',
              }}>Hidden from Predict and Evaluate</span>
            )}
          </div>
          <div style={{ fontSize: 12, color: 'var(--text3)', marginTop: '.25rem' }}>
            {run.base_model} · {fmt(run.completed_at)}
          </div>
        </div>

        {/* Status + loss pills + remove button */}
        <div style={{ display: 'flex', alignItems: 'center', gap: '.5rem', flexShrink: 0 }}>
          <span style={{
            fontSize: 11, padding: '5px 10px',
            borderRadius: 999, border: `1px solid ${statusColor(run.status)}44`,
            color: statusColor(run.status), background: `${statusColor(run.status)}11`,
          }}>{run.status}</span>

          {run.final_train_loss != null && (
            <span className="mono" style={{
              fontSize: 11, padding: '5px 10px',
              borderRadius: 999, background: 'rgba(124,156,255,.08)',
              border: '1px solid rgba(124,156,255,.18)', color: 'var(--accent2)',
            }}>
              T {run.final_train_loss.toFixed(4)}
            </span>
          )}
          {run.final_val_loss != null && (
            <span className="mono" style={{
              fontSize: 11, padding: '5px 10px',
              borderRadius: 999, background: 'rgba(20,184,166,.05)',
              border: '1px solid rgba(20,184,166,.12)', color: 'var(--accent2)',
            }}>
              V {run.final_val_loss.toFixed(4)}
            </span>
          )}

          {/* ✕ Remove button — only for non-pinned runs */}
          {!run._isPinned && (
            <button
              onClick={e => { e.stopPropagation(); onRemove(run.model_id || run.id) }}
              title={isRemoved ? 'Already hidden' : 'Hide from Predict & Eval'}
              style={{
                width: 28, height: 28, borderRadius: '50%',
                background: isRemoved ? 'rgba(255,255,255,.05)' : 'rgba(20, 184, 166, 0.08)',
                border: `1px solid ${isRemoved ? 'rgba(255,255,255,.12)' : 'rgba(20, 184, 166, 0.22)'}`,
                color: isRemoved ? 'var(--text3)' : 'var(--accent)', fontSize: 16, lineHeight: 1,
                display: 'flex', alignItems: 'center', justifyContent: 'center',
                cursor: isRemoved ? 'default' : 'pointer',
                flexShrink: 0, transition: 'all .15s',
              }}
            >✕</button>
          )}
        </div>
      </div>

      {/* Expanded body */}
      {expanded && (
        <div style={{
          borderTop: '1px solid rgba(49, 56, 70, 0.72)',
          padding: '18px',
          background: 'rgba(11, 13, 16, 0.82)',
        }}>
          <LossChart metrics={run.metrics} />
          <ConfigGrid run={run} />
        </div>
      )}
    </div>
  )
}

// ─── Main component ───────────────────────────────────────────────────────────
export default function RunsTab({ allFinetunedModels, removedModelIds, onRemoveModel, onRestoreAll }) {
  const [clearConfirm, setClearConfirm] = useState(false)

  const runs = useMemo(() => {
    const real = (allFinetunedModels || []).map(m => ({
      id: m.id,
      model_id: m.id,
      base_model: m.base_model || '—',
      run_name: m.run_name || m.id,
      status: 'completed',
      gpu: m.gpu || '—',
      quantization: m.quantization || '—',
      epochs: m.epochs,
      lora_r: m.lora_r,
      lora_alpha: m.lora_alpha,
      learning_rate: m.learning_rate,
      batch_size: m.batch_size,
      grad_accumulation: m.grad_accumulation,
      max_seq_length: m.max_seq_length,
      samples: m.samples,
      final_train_loss: m.final_train_loss,
      final_val_loss: m.final_val_loss,
      duration_mins: m.duration_mins,
      cost_usd: m.cost_usd,
      completed_at: m.completed_at || m.created_at,
      metrics: m.metrics || { train_loss: [], val_loss: [] },
    }))
    return [DEMO_RUN, ...real]
  }, [allFinetunedModels])

  const removedCount = removedModelIds.length

  function handleClearCache() {
    if (!clearConfirm) {
      setClearConfirm(true)
      setTimeout(() => setClearConfirm(false), 3000)
      return
    }
    runs.filter(r => !r._isPinned).forEach(r => onRemoveModel(r.model_id || r.id))
    setClearConfirm(false)
  }

  return (
    <div className="stack-lg">

      {/* Top bar */}
      <div className="toolbar">
        <div>
          <div style={{
            fontSize: 22, fontWeight: 700, color: 'var(--text)',
            letterSpacing: '-0.02em',
          }}>
            Previous Runs
          </div>
          <div style={{ fontSize: 13, color: 'var(--text3)', marginTop: '.35rem' }}>
            {runs.length} run{runs.length !== 1 ? 's' : ''} total
            {removedCount > 0 && ` · ${removedCount} hidden from Predict & Eval`}
          </div>
        </div>

        <div style={{ display: 'flex', alignItems: 'center', gap: '.6rem' }}>
          {removedCount > 0 && (
            <button
              onClick={onRestoreAll}
              className="btn btn-ghost btn-sm"
              style={{ color: 'var(--accent)', borderColor: 'rgba(20, 184, 166, 0.22)', background: 'rgba(20, 184, 166, 0.05)' }}
            >
              Restore All ({removedCount})
            </button>
          )}

          <button
            onClick={handleClearCache}
            className="btn btn-ghost btn-sm"
            style={{
              borderColor: clearConfirm ? 'rgba(248,113,113,.4)' : undefined,
              color: clearConfirm ? 'var(--red)' : undefined,
              background: clearConfirm ? 'rgba(248,113,113,.08)' : undefined,
            }}
            title="Hide all finetuned models from Predict and Eval tabs"
          >
            {clearConfirm ? 'Confirm hide all models' : 'Hide all models'}
          </button>
        </div>
      </div>

      {/* Runs list */}
      <div className="stack-sm">
        {runs.filter(r => r._isPinned || !removedModelIds.includes(r.model_id || r.id)).map(run => (
          <RunCard
            key={run.id}
            run={run}
            isRemoved={false}
            onRemove={onRemoveModel}
          />
        ))}
      </div>

      {runs.length === 1 && (
        <div className="dropzone" style={{ padding: '36px 24px', cursor: 'default' }}>
          No real training runs yet — complete a training job and it will appear here.
        </div>
      )}
    </div>
  )
}
