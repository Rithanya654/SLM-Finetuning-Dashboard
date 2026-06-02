import React from 'react'

function cx(...parts) {
  return parts.filter(Boolean).join(' ')
}

export function Card({ children, style, className }) {
  return (
    <div className={cx('card', className)} style={style}>
      {children}
    </div>
  )
}

export function CardTitle({ children, style, className }) {
  return (
    <div className={cx('card-title', className)} style={style}>
      {children}
    </div>
  )
}

const MODEL_META = {
  'google/gemma-3-1b-it': { org: 'Google', desc: 'Fast, lightweight' },
  'google/gemma-3-4b-it': { org: 'Google', desc: 'More capable' },
  'Qwen/Qwen2.5-1.5B-Instruct': { org: 'Alibaba', desc: 'Efficient for smaller workloads' },
  'Qwen/Qwen2.5-3B-Instruct': { org: 'Alibaba', desc: 'Balanced quality and speed' },
}

export function ModelGrid({ models, selectedId, onSelect, emptyText }) {
  if (!models || models.length === 0) {
    return (
      <p style={{ fontSize: 14, color: 'var(--text3)', paddingTop: 4 }}>
        {emptyText || 'No models available.'}
      </p>
    )
  }

  return (
    <div className="model-grid">
      {models.map((model, index) => {
        const meta = MODEL_META[model.id] || {}
        const isFT = model.type === 'finetuned'
        const selected = selectedId === model.id
        const subtitle = isFT && model.run_name
          ? `Run: ${model.run_name}`
          : meta.desc || meta.org || model.id.split('/')[0]

        return (
          <button
            key={model.id}
            type="button"
            className={cx('model-card', selected && 'selected')}
            style={{ animationDelay: `${index * 25}ms`, textAlign: 'left' }}
            onClick={() => onSelect(model)}
          >
            <div className="model-card-title">{model.name}</div>
            <div className="model-card-subtitle">{subtitle}</div>
          </button>
        )
      })}
    </div>
  )
}

export function GpuSelector({ value, onChange }) {
  const options = [
    { id: 'T4', label: 'T4', vram: '16 GB VRAM', tag: 'Budget', desc: 'Best for small training runs' },
    { id: 'A10G', label: 'A10G', vram: '24 GB VRAM', tag: 'Standard', desc: 'Recommended for 3B models' },
    { id: 'A100', label: 'A100', vram: '40 GB VRAM', tag: 'Premium', desc: 'Most headroom for long context' },
  ]

  return (
    <div className="gpu-grid">
      {options.map((option) => {
        const selected = value === option.id
        return (
          <button
            key={option.id}
            type="button"
            className={cx('gpu-card', selected && 'selected')}
            onClick={() => onChange(option.id)}
          >
            <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: 8, gap: 8 }}>
              <span style={{ fontSize: 15, fontWeight: 600, color: 'var(--text)' }}>{option.label}</span>
              <span className="chip" style={{ minHeight: 26, padding: '0 10px', fontSize: 11 }}>
                {option.tag}
              </span>
            </div>
            <div style={{ fontSize: 13, color: 'var(--text2)', marginBottom: 4 }}>{option.vram}</div>
            <div style={{ fontSize: 12, color: 'var(--text3)' }}>{option.desc}</div>
          </button>
        )
      })}
    </div>
  )
}

export function Btn({ children, onClick, variant = 'default', size = 'md', disabled, style, type, className }) {
  return (
    <button
      type={type || 'button'}
      onClick={onClick}
      disabled={disabled}
      className={cx('btn', `btn-${variant}`, size !== 'md' && `btn-${size}`, className)}
      style={style}
    >
      {children}
    </button>
  )
}

export function Field({ label, children, style, className }) {
  return (
    <div className={cx('field', className)} style={style}>
      {label && <Label>{label}</Label>}
      {children}
    </div>
  )
}

export function Label({ children, style, className }) {
  return (
    <label className={cx('field-label', className)} style={style}>
      {children}
    </label>
  )
}

const inputBaseClass = 'input'

export function Input({ style, className, ...props }) {
  return <input className={cx(inputBaseClass, className)} style={style} {...props} />
}

export function Select({ children, style, className, ...props }) {
  return (
    <select className={cx('select', className)} style={style} {...props}>
      {children}
    </select>
  )
}

export function Slider({ label, value, displayValue, min, max, step = 1, onChange, isLogarithmic }) {
  const sliderValue = isLogarithmic ? isLogarithmic.toIdx(value) : value
  const sliderMin = isLogarithmic ? isLogarithmic.minIdx : min
  const sliderMax = isLogarithmic ? isLogarithmic.maxIdx : max

  return (
    <div className="field">
      {label && (
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', gap: 12, marginBottom: 10 }}>
          <Label style={{ marginBottom: 0 }}>{label}</Label>
          <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
            {displayValue && <span style={{ fontSize: 12, color: 'var(--text3)' }}>{displayValue}</span>}
            <input
              type="number"
              min={min}
              max={max}
              step={step}
              value={value}
              onChange={(e) => onChange(Number(e.target.value))}
              className="input mono"
              style={{ width: 90, minHeight: 36, padding: '0 10px', fontSize: 12, textAlign: 'right' }}
            />
          </div>
        </div>
      )}
      <input
        type="range"
        min={sliderMin}
        max={sliderMax}
        step={isLogarithmic ? 1 : step}
        value={sliderValue}
        onChange={(e) => {
          const v = Number(e.target.value)
          onChange(isLogarithmic ? isLogarithmic.fromIdx(v) : v)
        }}
      />
    </div>
  )
}

export function Toggle({ label, checked, onChange }) {
  return (
    <div className="toggle">
      <span className="toggle-label">{label}</span>
      <button
        type="button"
        className={cx('toggle-track', checked && 'is-on')}
        onClick={() => onChange(!checked)}
        aria-pressed={checked}
      >
        <span className="toggle-thumb" />
      </button>
    </div>
  )
}

const badgeColors = {
  queued: { color: 'var(--text2)', border: 'rgba(49, 56, 70, 0.95)', bg: 'rgba(29, 35, 43, 0.92)' },
  initializing: { color: 'var(--orange)', border: 'rgba(251, 146, 60, 0.32)', bg: 'rgba(251, 146, 60, 0.12)' },
  training: { color: 'var(--orange)', border: 'rgba(251, 146, 60, 0.32)', bg: 'rgba(251, 146, 60, 0.12)' },
  deploying: { color: 'var(--blue)', border: 'rgba(96, 165, 250, 0.32)', bg: 'rgba(96, 165, 250, 0.12)' },
  completed: { color: 'var(--green)', border: 'rgba(74, 222, 128, 0.3)', bg: 'rgba(74, 222, 128, 0.12)' },
  failed: { color: 'var(--red)', border: 'rgba(248, 113, 113, 0.3)', bg: 'rgba(248, 113, 113, 0.12)' },
  cancelled: { color: 'var(--text3)', border: 'rgba(49, 56, 70, 0.95)', bg: 'rgba(29, 35, 43, 0.92)' },
  running: { color: 'var(--orange)', border: 'rgba(251, 146, 60, 0.32)', bg: 'rgba(251, 146, 60, 0.12)' },
}

export function Badge({ status }) {
  const colorSet = badgeColors[status] || badgeColors.queued
  const pulsing = ['training', 'initializing', 'deploying', 'running'].includes(status)

  return (
    <span className="badge" style={{ color: colorSet.color, borderColor: colorSet.border, background: colorSet.bg }}>
      <span className="badge-dot" style={{ animation: pulsing ? 'pulse 1.4s ease-in-out infinite' : 'none' }} />
      {status}
    </span>
  )
}

export function SectionLabel({ children, style, className }) {
  return (
    <div className={cx('section-label', className)} style={style}>
      {children}
    </div>
  )
}

export function MetricCard({ label, value, sub }) {
  const stringValue = typeof value === 'string' ? value : String(value)
  const isMono = /[%$./0-9]/.test(stringValue)

  return (
    <div className="metric-card">
      <div className="metric-card-label">{label}</div>
      <div className={cx('metric-card-value', isMono && 'is-mono')}>{value}</div>
      {sub && <div className="metric-card-sub">{sub}</div>}
    </div>
  )
}

export function Spinner({ size = 14 }) {
  return <span className="spinner" style={{ width: size, height: size }} />
}

export function Empty({ children }) {
  return (
    <div className="empty-state">
      <p>{children}</p>
    </div>
  )
}

export function InnerTabs({ tabs, active, onChange, trailing }) {
  return (
    <div className="inner-tabs">
      {tabs.map((tab) => (
        <button
          key={tab.id}
          type="button"
          onClick={() => onChange(tab.id)}
          className={cx('tab-btn', active === tab.id && 'active')}
        >
          {tab.label}
        </button>
      ))}
      {trailing && <div style={{ marginLeft: 'auto' }}>{trailing}</div>}
    </div>
  )
}
