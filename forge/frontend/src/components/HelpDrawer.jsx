import React, { useState } from 'react'

// ── Help content ──────────────────────────────────────────────────────────────
const HELP_SECTIONS = [
  {
    id: 'overview',
    icon: '',
    title: 'How This Platform Works',
    color: 'var(--accent)',
    content: `This is an end-to-end LLM fine-tuning platform. You don't need to write any training code — the platform handles GPU provisioning, model loading, training loops, and evaluation automatically.

The workflow is: prepare data → fine-tune → evaluate → deploy. Each tab in the platform maps to one of these steps. You can iterate: evaluate your model, spot weak fields, improve your dataset, retrain, repeat.`,
  },
  {
    id: 'training',
    icon: '',
    title: 'Train Tab',
    color: '#a78bfa',
    subsections: [
      {
        title: 'Dataset',
        body: 'Upload your JSONL training file. The format should be chat-style: a system prompt, a user message (the document), and an assistant message (the correct JSON extraction). Use the "Use sample dataset" button to try the platform immediately.',
      },
      {
        title: 'Base Model',
        body: 'Choose the starting model. Qwen 2.5 1.5B is fast and cheap — good for testing. Qwen 2.5 3B is better for complex extraction tasks but needs more GPU memory.',
      },
      {
        title: 'Hardware',
        body: 'T4 is cheaper but limited to smaller models and smaller batch sizes. A10G has 24GB VRAM — recommended for anything serious. A100 80GB is for large models or very long sequences.',
      },
      {
        title: 'Smart Dataset Analysis',
        body: 'Click "Analyse & Recommend" to let the platform inspect your dataset (row count, token lengths, field density) without sending your data to an AI. It then asks Gemini to recommend safe hyperparameters for your chosen model and GPU, with OOM prevention built in. Hit "Apply Recommended Settings" to fill in all params automatically.',
      },
      {
        title: 'Presets',
        body: '"quick" is a single-epoch smoke test to verify your data loads correctly. "balanced" is the standard starting point. "full" is for final production runs with more capacity. "loraSmall" uses a tiny rank for minimal compute. "loraLarge" maximises model capacity.',
      },
      {
        title: 'LoRA / PEFT',
        body: 'LoRA trains a tiny set of adapter weights rather than the full model, making fine-tuning possible on a single GPU. Rank (r) controls how much capacity the adapter has. Higher rank = more powerful but more memory. Alpha controls the learning strength — usually set to 2× rank.',
      },
      {
        title: 'Training Monitor',
        body: 'Live training loss and validation loss curves update every few seconds. A good run shows both losses falling together. If val loss starts rising while train loss falls, that is overfitting — early stopping will catch this automatically.',
      },
    ],
  },
  {
    id: 'runs',
    icon: '',
    title: 'Runs Tab',
    color: 'var(--accent)',
    subsections: [
      {
        title: 'What It Shows',
        body: 'Use this tab to review completed fine-tuning jobs and the fine-tuned models registered by the backend. It helps you keep track of which runs produced usable models.',
      },
      {
        title: 'When To Use It',
        body: 'After training finishes, check Runs to confirm the model is available before using it in Predict or Eval. If multiple demo runs exist, keep the newest successful model visible and hide older noisy runs.',
      },
      {
        title: 'Demo Tip',
        body: 'For demos, use the run status, final loss, GPU, and timestamp to explain that each model is reproducible and tied to a specific training configuration.',
      },
    ],
  },
  {
    id: 'predict',
    icon: '',
    title: 'Predict Tab',
    color: 'var(--accent2)',
    subsections: [
      {
        title: 'What It Does',
        body: 'Use Predict to test a base or fine-tuned model on one document or prompt. It is the fastest way to sanity-check whether the model returns the expected JSON schema.',
      },
      {
        title: 'How To Use It',
        body: 'Select the model, paste invoice/document text, and run prediction. Compare the output against the fields you expect before doing a larger evaluation.',
      },
      {
        title: 'Demo Tip',
        body: 'Show the same prompt on a base model and then on the fine-tuned model. The fine-tuned output should look more consistent, schema-aligned, and less verbose.',
      },
    ],
  },
  {
    id: 'eval',
    icon: '',
    title: 'Eval Tab',
    color: 'var(--green)',
    subsections: [
      {
        title: 'What It Measures',
        body: 'The evaluator compares your model\'s extracted JSON output against ground-truth labels for each document. It scores every field individually: exact match (perfect), partial match (truncated or slightly off), or miss (wrong/null).',
      },
      {
        title: 'Field-Level F1',
        body: 'The primary metric. Tells you which specific fields the model is good at (e.g. vendor_name: 90%) versus struggling with (e.g. vendor_address: 40%). This directly tells you where to add more training examples.',
      },
      {
        title: 'Reading the Results',
        body: '"Weakest Fields" on the Overview tab shows the top 5 most problematic fields. The Field Analysis tab shows every tracked field with exact/partial/miss/hallucination counts. Samples lets you drill into individual documents to see exactly what the model predicted versus the truth.',
      },
      {
        title: 'Sample Dataset',
        body: 'The built-in 20-invoice sample dataset covers diverse vendor types: pharma, utilities, retail, NON-PO, credit memos, and non-invoices. This gives a realistic benchmark across all invoice types your pipeline is expected to handle.',
      },
    ],
  },
  {
    id: 'vspretrained',
    icon: '',
    title: 'Why Fine-tune vs. Use a Base Model?',
    color: 'var(--accent)',
    content: `A base model like Qwen 2.5 1.5B can extract invoice fields with ~40–55% accuracy out of the box. Fine-tuning on 50–200 domain examples typically pushes this to 80–95%.

More importantly, fine-tuning makes the model reliable and consistent: it learns your exact output schema, handles your specific vendors, and stops hallucinating fields that do not exist. A base model might extract "total" in 10 different key names across 10 invoices. A fine-tuned model always uses "invoice_total".

Cost comparison: a single A10G training run costs ~$2–5 and takes 20–60 minutes. Running a large frontier model (GPT-4o, Gemini Pro) at scale for the same extraction costs 10–100× more per document over time.`,
  },
  {
    id: 'tips',
    icon: '',
    title: 'Tips for Better Results',
    color: 'var(--accent2)',
    items: [
      'Start with 50+ diverse examples. Diversity of vendor types matters more than raw count.',
      'If a field has <60% accuracy, add 10–20 more examples specifically for that field.',
      'Keep your system prompt consistent between training and inference.',
      'Use the "balanced" preset as your first real training run.',
      'Run dataset analysis before training — it prevents OOM crashes and suggests a starting seq length.',
      'A training loss that never falls below 1.0 usually means your data format is wrong.',
      'If val loss diverges early, reduce learning rate or increase early stopping patience.',
      'Credit memos and Non-Invoice types often need extra examples — they have different structures.',
    ],
  },
]

// ── Drawer Component ──────────────────────────────────────────────────────────
export default function HelpDrawer() {
  const [open, setOpen] = useState(false)
  const [expanded, setExpanded] = useState({})

  function toggle(id) {
    setExpanded(prev => ({ ...prev, [id]: !prev[id] }))
  }

  return (
    <>
      {/* Trigger button — fixed top right */}
      <button
        onClick={() => setOpen(true)}
        title="Platform guide"
        style={{
          position: 'fixed', top: 16, right: 16, zIndex: 1000,
          display: 'flex', alignItems: 'center', gap: 6,
          padding: '7px 14px', borderRadius: 999,
          background: 'rgba(23,27,33,0.92)', backdropFilter: 'blur(12px)',
          border: '1px solid rgba(124,156,255,0.3)', cursor: 'pointer',
          color: 'var(--accent)', fontSize: 13, fontWeight: 700,
          boxShadow: '0 4px 24px rgba(0,0,0,0.4)',
          transition: 'all .15s',
        }}
      >
        <span style={{ fontSize: 15 }}></span>
        <span>Help & Guide</span>
      </button>

      {/* Overlay */}
      {open && (
        <div
          onClick={() => setOpen(false)}
          style={{
            position: 'fixed', inset: 0, zIndex: 1001,
            background: 'rgba(0,0,0,0.55)', backdropFilter: 'blur(2px)',
          }}
        />
      )}

      {/* Drawer */}
      <div style={{
        position: 'fixed', top: 0, right: 0, bottom: 0, zIndex: 1002,
        width: Math.min(520, window.innerWidth - 32),
        background: 'rgba(17,20,26,0.98)', backdropFilter: 'blur(24px)',
        borderLeft: '1px solid rgba(49,56,70,0.9)',
        transform: open ? 'translateX(0)' : 'translateX(105%)',
        transition: 'transform .28s cubic-bezier(.4,0,.2,1)',
        display: 'flex', flexDirection: 'column',
        boxShadow: '-8px 0 48px rgba(0,0,0,.6)',
      }}>
        {/* Drawer header */}
        <div style={{
          display: 'flex', alignItems: 'center', justifyContent: 'space-between',
          padding: '1.1rem 1.25rem', borderBottom: '1px solid rgba(49,56,70,0.7)',
          flexShrink: 0,
        }}>
          <div>
            <div style={{ fontSize: 16, fontWeight: 800, color: 'var(--text)' }}>Platform Guide</div>
            <div style={{ fontSize: 12, color: 'var(--text3)', marginTop: 2 }}>Everything you need to know about using this platform</div>
          </div>
          <button
            onClick={() => setOpen(false)}
            style={{
              width: 32, height: 32, borderRadius: '50%',
              background: 'var(--bg3)', border: '1px solid var(--border)',
              color: 'var(--text2)', fontSize: 16, cursor: 'pointer',
              display: 'flex', alignItems: 'center', justifyContent: 'center',
            }}
          >×</button>
        </div>

        {/* Drawer body */}
        <div style={{ flex: 1, overflowY: 'auto', padding: '1rem 1.25rem' }}>
          <div style={{ display: 'flex', flexDirection: 'column', gap: '.5rem' }}>
            {HELP_SECTIONS.map(section => {
              const isOpen = expanded[section.id]
              return (
                <div key={section.id} style={{
                  border: `1px solid ${isOpen ? section.color + '30' : 'rgba(49,56,70,0.7)'}`,
                  borderRadius: 16, overflow: 'hidden',
                  background: isOpen ? `${section.color}06` : 'var(--bg3)',
                  transition: 'all .15s',
                }}>
                  {/* Section header */}
                  <button
                    onClick={() => toggle(section.id)}
                    style={{
                      width: '100%', display: 'flex', alignItems: 'center', gap: '.75rem',
                      padding: '14px 16px', cursor: 'pointer', textAlign: 'left',
                      background: 'transparent', border: 'none',
                    }}
                  >
                    <span style={{ fontSize: 18, flexShrink: 0 }}>{section.icon}</span>
                    <span style={{ flex: 1, fontSize: 14, fontWeight: 700, color: isOpen ? section.color : 'var(--text)' }}>
                      {section.title}
                    </span>
                    <span style={{ fontSize: 12, color: 'var(--text3)', flexShrink: 0 }}>
                      {isOpen ? '▲' : '▼'}
                    </span>
                  </button>

                  {/* Section body */}
                  {isOpen && (
                    <div style={{ padding: '0 16px 16px', borderTop: '1px solid rgba(49,56,70,0.5)' }}>
                      {/* Plain text content */}
                      {section.content && (
                        <p style={{ fontSize: 13, color: 'var(--text2)', lineHeight: 1.75, marginTop: '1rem', marginBottom: 0, whiteSpace: 'pre-line' }}>
                          {section.content}
                        </p>
                      )}

                      {/* Subsections */}
                      {section.subsections && (
                        <div style={{ marginTop: '1rem', display: 'flex', flexDirection: 'column', gap: '.6rem' }}>
                          {section.subsections.map(sub => (
                            <div key={sub.title} style={{
                              background: 'var(--bg4)', borderRadius: 10,
                              padding: '10px 12px', border: '1px solid rgba(49,56,70,0.6)',
                            }}>
                              <div style={{ fontSize: 12, fontWeight: 700, color: section.color, marginBottom: '.35rem' }}>
                                {sub.title}
                              </div>
                              <div style={{ fontSize: 12, color: 'var(--text2)', lineHeight: 1.7 }}>{sub.body}</div>
                            </div>
                          ))}
                        </div>
                      )}

                      {/* Bullet items */}
                      {section.items && (
                        <ul style={{ marginTop: '1rem', paddingLeft: '1.2rem', display: 'flex', flexDirection: 'column', gap: '.45rem' }}>
                          {section.items.map((item, i) => (
                            <li key={i} style={{ fontSize: 13, color: 'var(--text2)', lineHeight: 1.65 }}>{item}</li>
                          ))}
                        </ul>
                      )}
                    </div>
                  )}
                </div>
              )
            })}
          </div>

          {/* Footer note */}
          <div style={{ marginTop: '1.5rem', padding: '12px 14px', borderRadius: 12, background: 'var(--bg3)', border: '1px solid var(--border)' }}>
            <div style={{ fontSize: 12, color: 'var(--text3)', lineHeight: 1.6 }}>
               Hover over any <span style={{
                display: 'inline-flex', alignItems: 'center', justifyContent: 'center',
                width: 14, height: 14, borderRadius: '50%', border: '1px solid var(--border2)',
                background: 'var(--bg4)', fontSize: 9, fontWeight: 700, color: 'var(--text3)',
              }}>?</span> icon next to a parameter for a plain-English explanation of what it does.
            </div>
          </div>
        </div>
      </div>
    </>
  )
}
