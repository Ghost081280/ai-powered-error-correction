import React from 'react';

const Controls = ({ 
  selectedNoiseType, 
  setSelectedNoiseType, 
  showAI, 
  setShowAI, 
  showClassical, 
  setShowClassical,
  selectedSNR,
  setSelectedSNR,
  availableSNRs,
  summaryData
}) => {

  const noiseTypeDescriptions = {
    awgn: 'Additive White Gaussian Noise - Standard thermal noise',
    burst: 'Burst Noise - Intermittent high-power interference',
    narrowband: 'Narrowband Interference - Continuous sinusoidal jamming'
  };

  const formatPercentage = (value) => {
    if (value === undefined || value === null) return 'N/A';
    return `${value >= 0 ? '+' : ''}${value.toFixed(1)}%`;
  };

  const formatBER = (value) => {
    if (value === undefined || value === null) return 'N/A';
    return value.toExponential(2);
  };

  return (
    <div className="controls-panel">
      <div className="controls-section">
        <h3>Visualization Controls</h3>
        
        <div className="control-group">
          <label htmlFor="noise-type">Channel Condition:</label>
          <select 
            id="noise-type"
            value={selectedNoiseType} 
            onChange={(e) => setSelectedNoiseType(e.target.value)}
            className="control-select"
          >
            <option value="awgn">AWGN</option>
            <option value="burst">Burst Noise</option>
            <option value="narrowband">Narrowband Interference</option>
          </select>
          <span className="control-description">
            {noiseTypeDescriptions[selectedNoiseType]}
          </span>
        </div>

        <div className="control-group">
          <label>Display Options:</label>
          <div className="checkbox-group">
            <label className="checkbox-label">
              <input 
                type="checkbox" 
                checked={showClassical} 
                onChange={(e) => setShowClassical(e.target.checked)}
              />
              <span className="checkmark"></span>
              Classical FEC (Hamming)
            </label>
            <label className="checkbox-label">
              <input 
                type="checkbox" 
                checked={showAI} 
                onChange={(e) => setShowAI(e.target.checked)}
              />
              <span className="checkmark"></span>
              AI Denoising
            </label>
          </div>
        </div>

        <div className="control-group">
          <label htmlFor="snr-select">Constellation SNR:</label>
          <select 
            id="snr-select"
            value={selectedSNR} 
            onChange={(e) => setSelectedSNR(parseInt(e.target.value))}
            className="control-select"
          >
            {availableSNRs.map(snr => (
              <option key={snr} value={snr}>{snr} dB</option>
            ))}
          </select>
        </div>
      </div>

      {summaryData && (
        <div className="controls-section">
          <h3>Performance Summary</h3>
          <div className="performance-grid">
            {Object.entries(summaryData.performance_summary || {}).map(([noiseType, stats]) => (
              <div key={noiseType} className="performance-card">
                <h4>{noiseType.toUpperCase()}</h4>
                <div className="stat-row">
                  <span className="stat-label">Classical BER:</span>
                  <span className="stat-value">{formatBER(stats.average_classical_ber)}</span>
                </div>
                <div className="stat-row">
                  <span className="stat-label">AI BER:</span>
                  <span className="stat-value">{formatBER(stats.average_ai_ber)}</span>
                </div>
                <div className="stat-row">
                  <span className="stat-label">Improvement:</span>
                  <span className={`stat-value ${stats.improvement_percent > 0 ? 'positive' : 'negative'}`}>
                    {formatPercentage(stats.improvement_percent)}
                  </span>
                </div>
                <div className="performance-indicator">
                  {stats.ai_better ? (
                    <span className="indicator-good">✓ AI Better</span>
                  ) : (
                    <span className="indicator-poor">✗ Classical Better</span>
                  )}
                </div>
              </div>
            ))}
          </div>
        </div>
      )}

      <div className="controls-section">
        <h3>About This Demo</h3>
        <div className="info-content">
          <p>
            This demonstration compares classical Forward Error Correction (FEC) using 
            Hamming(7,4) codes against an AI-based signal denoising approach using a 
            PyTorch autoencoder neural network.
          </p>
          <div className="method-comparison">
            <div className="method-card">
              <h4>Classical FEC</h4>
              <ul>
                <li>Hamming(7,4) encoding</li>
                <li>Hard decision demodulation</li>
                <li>Single error correction</li>
                <li>Fixed overhead (75%)</li>
              </ul>
            </div>
            <div className="method-card">
              <h4>AI Denoising</h4>
              <ul>
                <li>Neural network autoencoder</li>
                <li>Signal-level processing</li>
                <li>Adaptive noise reduction</li>
                <li>No coding overhead</li>
              </ul>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default Controls;
