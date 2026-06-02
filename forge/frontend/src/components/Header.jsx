import React from 'react'

export default function Header({ activeTab, setActiveTab, apiOnline, fineTunedCount, onResetRuns }) {
    const tabs = [
        { id: 'train', label: 'Train' },
        { id: 'runs', label: 'Runs' },
        { id: 'predict', label: 'Predict' },
        { id: 'eval', label: 'Evaluate' },
    ]

    return (
        <header className="app-header">
            <div className="app-brand">
                <div className="app-brand-copy">
                    <div className="app-brand-title">SLM Finetuning Platform</div>
                    <div className="app-brand-subtitle">Training, inference, and evaluation workspace</div>
                </div>
            </div>

            <nav className="app-nav" aria-label="Primary navigation">
                {tabs.map((tab) => (
                    <button
                        key={tab.id}
                        type="button"
                        className={`app-nav-btn${activeTab === tab.id ? ' active' : ''}`}
                        onClick={() => setActiveTab(tab.id)}
                    >
                        {tab.label}
                    </button>
                ))}
            </nav>

            <div className="app-header-actions">
                <button type="button" className="btn btn-ghost btn-sm" onClick={onResetRuns} title="Reset active runs and in-tab process state">
                    Reset runs
                </button>
            </div>
        </header>
    )
}
