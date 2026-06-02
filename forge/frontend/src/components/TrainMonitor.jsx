import React, { useEffect, useRef, useState, useCallback } from 'react'
import {
  LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Legend
} from 'recharts'
import { Card, CardTitle, Badge, Btn, MetricCard, Empty, Spinner } from './UI.jsx'
import { api } from '../api.js'

export default function TrainMonitor({ jobId, onComplete }) {
  const [job, setJob] = useState(null)
  const [error, setError] = useState(null)
  const logRef = useRef(null)
  const prevLogLen = useRef(0)

  const poll = useCallback(async () => {
    if (!jobId) return
    try {
      const data = await api.getJob(jobId)
      setJob(data)
      if (['completed', 'failed', 'cancelled'].includes(data.status)) {
        if (data.status === 'completed') onComplete?.()
        return true // stop polling
      }
    } catch (e) {
      setError(e.message)
      return true
    }
    return false
  }, [jobId, onComplete])

  useEffect(() => {
    if (!jobId) return
    setJob(null); setError(null); prevLogLen.current = 0
    poll()
    let busy = false
      const iv = setInterval(async () => {
        if (busy) return
        busy = true
        const stop = await poll()
        busy = false
        if (stop) clearInterval(iv)
      }, 2500) 
    return () => clearInterval(iv)
  }, [jobId, poll])

  // Auto-scroll logs
  useEffect(() => {
    if (!job) return
    const newLogs = job.logs.length
    if (newLogs > prevLogLen.current) {
      prevLogLen.current = newLogs
      if (logRef.current) logRef.current.scrollTop = logRef.current.scrollHeight
    }
  }, [job])

  if (!jobId) {
    return (
      <Card>
        <CardTitle>Training Monitor</CardTitle>
        <Empty>Launch a training job to see live metrics here</Empty>
      </Card>
    )
  }

  if (error) {
    return (
      <Card>
        <CardTitle>Training Monitor</CardTitle>
        <div style={{ color: 'var(--red)', fontSize: 13, padding: '1rem' }}>
          Poll error: {error}
        </div>
      </Card>
    )
  }

  if (!job) {
    return (
      <Card>
        <CardTitle>Training Monitor</CardTitle>
        <div style={{ padding: '2rem', display: 'flex', justifyContent: 'center' }}>
          <Spinner size={20} />
        </div>
      </Card>
    )
  }

  const tl = job.metrics.train_loss || []
  const vl = job.metrics.val_loss || []
  const chartData = tl.map((pt, i) => ({
    step: pt.step,
    train: pt.value,
    val: vl[i]?.value ?? null,
  }))

  const isRunning = ['queued', 'initializing', 'training', 'deploying'].includes(job.status)

  async function cancel() {
    try { await api.cancelJob(jobId) } catch (e) { alert(e.message) }
  }

  return (
    <Card>
      {/* Header */}
      <div className="toolbar" style={{ marginBottom: '1rem' }}>
        <CardTitle style={{ margin: 0 }}>Training Monitor</CardTitle>
        <div style={{ display: 'flex', alignItems: 'center', gap: '.5rem' }}>
          <Badge status={job.status} />
          {isRunning && (
            <Btn variant="danger" size="sm" onClick={cancel}>Cancel</Btn>
          )}
        </div>
      </div>

      {/* Metrics row */}
      <div className="quad-grid" style={{ marginBottom: '.9rem' }}>
        <MetricCard label="Progress" value={`${job.progress}%`} />
        <MetricCard
          label="Train Loss"
          value={tl.length ? tl[tl.length - 1].value.toFixed(4) : '—'}
        />
        <MetricCard
          label="Val Loss"
          value={vl.length ? vl[vl.length - 1].value.toFixed(4) : '—'}
        />
        <MetricCard label="Est. Cost" value={`$${job.cost_usd.toFixed(3)}`} />
      </div>

      {/* Progress bar */}
      <div style={{ background: 'rgba(29, 35, 43, 0.92)', borderRadius: 999, height: 8, overflow: 'hidden', marginBottom: '.55rem' }}>
        <div style={{ height: '100%', width: `${job.progress}%`, background: 'linear-gradient(90deg, var(--accent), var(--accent2))', borderRadius: 999, transition: 'width .3s' }} />
      </div>
      <div style={{ fontSize: 13, color: 'var(--text3)', marginBottom: '1rem' }}>
        Step {job.current_step} / {job.total_steps}
        {job.output_model_id && (
          <span className="mono" style={{ marginLeft: '1rem', color: 'var(--green)' }}>
            Model ready: {job.output_model_id}
          </span>
        )}
      </div>

      {/* Loss chart */}
      {chartData.length > 1 && (
        <div className="surface-strong" style={{ padding: '1rem', marginBottom: '1rem' }}>
          <div style={{ fontSize: 13, color: 'var(--text3)', marginBottom: '.9rem', display: 'flex', gap: '1rem', flexWrap: 'wrap' }}>
            <span><span style={{ display: 'inline-block', width: 10, height: 2, background: 'var(--accent)', marginRight: 4, verticalAlign: 'middle' }} />Train loss</span>
            <span><span style={{ display: 'inline-block', width: 10, height: 2, background: 'var(--blue)', marginRight: 4, verticalAlign: 'middle', borderTop: '1px dashed var(--blue)' }} />Val loss</span>
          </div>
          <ResponsiveContainer width="100%" height={220}>
            <LineChart data={chartData} margin={{ top: 4, right: 8, bottom: 0, left: -20 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="rgba(148,160,178,.12)" vertical={false} />
              <XAxis dataKey="step" tick={{ fontSize: 11, fill: '#94a0b2' }} minTickGap={30} />
              <YAxis tick={{ fontSize: 11, fill: '#94a0b2' }} />
              <Tooltip
                contentStyle={{ background: 'rgba(23, 27, 33, 0.96)', border: '1px solid rgba(49, 56, 70, 0.92)', borderRadius: 14, fontSize: 12 }}
                labelStyle={{ color: 'var(--text2)' }}
                itemStyle={{ color: 'var(--text)' }}
              />
              <Line type="monotone" dataKey="train" stroke="#7c9cff" dot={false} strokeWidth={2.2} name="Train" />
              <Line type="monotone" dataKey="val" stroke="#60a5fa" dot={false} strokeWidth={2.2} strokeDasharray="5 4" name="Val" />
            </LineChart>
          </ResponsiveContainer>
        </div>
      )}

      {/* Logs */}
      <div style={{ fontSize: 14, color: 'var(--text2)', marginBottom: '.55rem', fontWeight: 600 }}>
        Logs
      </div>
      <div
        ref={logRef}
        className="mono-panel"
        style={{ height: 240, overflowY: 'auto', fontSize: 12, lineHeight: 1.8 }}
      >
        {job.logs.length === 0 ? (
          <span style={{ color: 'var(--text3)' }}>Waiting for logs…</span>
        ) : job.logs.map((l, i) => (
          <div key={i} style={{ display: 'flex', gap: '.5rem' }}>
            <span style={{ color: 'var(--text3)', flexShrink: 0 }}>{l.ts.split('T')[1]?.slice(0, 8)}</span>
            <span style={{
              flexShrink: 0, minWidth: 38,
              color: l.level === 'ERROR' ? 'var(--red)' : l.level === 'WARN' ? 'var(--orange)' : 'var(--blue)',
            }}>{l.level}</span>
            <span style={{ color: 'var(--text)' }}>{l.msg}</span>
          </div>
        ))}
      </div>
    </Card>
  )
}
