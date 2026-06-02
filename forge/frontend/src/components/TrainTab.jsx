import React, { useState } from 'react'
import TrainConfig from './TrainConfig.jsx'
import TrainMonitor from './TrainMonitor.jsx'

export default function TrainTab({ onModelAdded, onDatasetReady, datasetPath }) {
    const [jobId, setJobId] = useState(null)

    return (
        <div className="stack-lg">
            {/* Top — Config (now handles its own two-column grid) */}
            <TrainConfig
                onJobStart={(id) => setJobId(id)}
                onDatasetReady={onDatasetReady}
                prefilledDatasetPath={datasetPath}
            />

            {/* Bottom — Monitor (Full Width) */}
            <TrainMonitor jobId={jobId} onComplete={onModelAdded} />
        </div>
    )
}
