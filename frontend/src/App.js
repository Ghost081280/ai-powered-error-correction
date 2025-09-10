import React, { useState, useEffect } from 'react';
import BERChart from './components/BERChart';
import Constellation from './components/Constellation';
import Spectrogram from './components/Spectrogram';
import Controls from './components/Controls';
import './styles.css';

function App() {
  const [berData, setBerData] = useState(null);
  const [constellationData, setConstellationData] = useState(null);
  const [spectrogramData, setSpectrogramData] = useState(null);
  const [summaryData, setSummaryData] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  
  // Control states
  const [selectedNoiseType, setSelectedNoiseType] = useState('awgn');
  const [showAI, setShowAI] = useState(true);
  const [showClassical, setShowClassical] = useState(true);
  const [selectedSNR, setSelectedSNR] = useState(5);

  useEffect(() => {
    loadData();
  }, []);

  const loadData = async () => {
    try {
      setLoading(true);
      setError(null);

      // Load all data files
      const dataPromises = [
        fetch('/data/ber_results.json').then(res => res.ok ? res.json() : null),
        fetch('/data/constellation_data.json').then(res => res.ok ? res.json() : null),
        fetch('/data/spectrogram_data.json').then(res => res.ok ? res.json() : null),
        fetch('/data/simulation_summary.json').then(res => res.ok ? res.json() : null)
      ];

      const [ber, constellation, spectrogram, summary] = await Promise.all(dataPromises);

      setBerData(ber);
      setConstellationData(constellation);
      setSpectrogramData(spectrogram);
      setSummaryData(summary);

      // Set default SNR to middle value if data is available
      if (constellation) {
        const snrKeys = Object.keys(constellation);
        if (snrKeys.length > 0) {
          const snrValues = snrKeys.map(key => parseInt(key.match(/\d+/)[0])).sort((a, b) => a - b);
          setSelectedSNR(snrValues[Math.floor(snrValues.length / 2)]);
        }
      }

    } catch (err) {
      console.error('Error loading data:', err);
      setError('Failed to load simulation data. Please ensure the backend simulation has been run and data files are available.');
    } finally {
      setLoading(false);
    }
  };

  if (loading) {
    return (
      <div className="loading-container">
        <div className="loading-spinner"></div>
        <div className="loading-text">Loading AI Error Correction Demo...</div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="error-container">
        <div className="error-message">
          <h2>⚠️ Data Loading Error</h2>
          <p>{error}</p>
          <div className="error-instructions">
            <h3>To fix this:</h3>
            <ol>
              <li>Run the backend simulation: <code>cd backend && python train_model.py && python simulation.py</code></li>
              <li>Copy generated data: <code>cp -r backend/data/* frontend/public/data/</code></li>
              <li>Refresh this page</li>
            </ol>
          </div>
          <button onClick={loadData} className="retry-button">Retry Loading Data</button>
        </div>
      </div>
    );
  }

  const hasData = berData || constellationData || spectrogramData;

  return (
    <div className="app">
      <header className="app-header">
        <div className="header-content">
          <h1>AI-Powered Error Correction for UHF Communications</h1>
          <p className="subtitle">
            Interactive demonstration comparing classical FEC vs. AI-based error correction under jamming conditions
          </p>
        </div>
      </header>

      <main className="app-main">
        {hasData ? (
          <>
            <Controls
              selectedNoiseType={selectedNoiseType}
              setSelectedNoiseType={setSelectedNoiseType}
              showAI={showAI}
              setShowAI={setShowAI}
              showClassical={showClassical}
              setShowClassical={setShowClassical}
              selectedSNR={selectedSNR}
              setSelectedSNR={setSelectedSNR}
              availableSNRs={constellationData ? 
                Object.keys(constellationData).map(key => parseInt(key.match(/\d+/)[0])).sort((a, b) => a - b) : 
                [0, 5, 10, 15]
              }
              summaryData={summaryData}
            />

            <div className="charts-grid">
              {berData && (
                <div className="chart-container full-width">
                  <BERChart 
                    data={berData}
                    selectedNoiseType={selectedNoiseType}
                    showAI={showAI}
                    showClassical={showClassical}
                  />
                </div>
              )}

              {constellationData && (
                <div className="chart-container">
                  <Constellation 
                    data={constellationData}
                    selectedSNR={selectedSNR}
                  />
                </div>
              )}

              {spectrogramData && (
                <div className="chart-container">
                  <Spectrogram 
                    data={spectrogramData}
                  />
                </div>
              )}
            </div>
          </>
        ) : (
          <div className="no-data-message">
            <h2>No Simulation Data Available</h2>
            <p>Please run the backend simulation to generate visualization data.</p>
            <div className="instructions">
              <h3>Quick Start:</h3>
              <ol>
                <li><code>cd backend</code></li>
                <li><code>pip install -r requirements.txt</code></li>
                <li><code>python train_model.py</code></li>
                <li><code>python simulation.py</code></li>
                <li><code>cp -r data/* ../frontend/public/data/</code></li>
                <li>Refresh this page</li>
              </ol>
            </div>
          </div>
        )}
      </main>

      <footer className="app-footer">
        <div className="footer-content">
          <p>
            🚀 AI Error Correction Demo | 
            <a href="https://github.com/yourusername/ai-error-correction-demo" target="_blank" rel="noopener noreferrer">
              View on GitHub
            </a>
          </p>
          <p className="tech-stack">
            Built with PyTorch, React, and Recharts
          </p>
        </div>
      </footer>
    </div>
  );
}

export default App;
