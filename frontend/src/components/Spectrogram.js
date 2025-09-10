import React, { useState } from 'react';
import Plot from 'react-plotly.js';

const Spectrogram = ({ data }) => {
  const [selectedView, setSelectedView] = useState('clean');

  if (!data) {
    return (
      <div className="chart-placeholder">
        <h3>Spectrogram Analysis</h3>
        <p>No spectrogram data available</p>
      </div>
    );
  }

  const getSpectrogramData = () => {
    let zData, title;
    
    switch (selectedView) {
      case 'clean':
        zData = data.clean_spectrogram;
        title = 'Clean Signal Spectrogram';
        break;
      case 'noisy':
        zData = data.noisy_spectrogram;
        title = 'Noisy Signal Spectrogram';
        break;
      case 'ai_denoised':
        zData = data.ai_denoised_spectrogram;
        title = 'AI Denoised Signal Spectrogram';
        break;
      default:
        zData = data.clean_spectrogram;
        title = 'Clean Signal Spectrogram';
    }

    return { zData, title };
  };

  const { zData, title } = getSpectrogramData();

  const plotData = [{
    z: zData,
    x: data.time,
    y: data.frequency,
    type: 'heatmap',
    colorscale: [
      [0, '#000080'],      // Dark blue (low power)
      [0.25, '#0000ff'],   // Blue
      [0.5, '#00ff00'],    // Green
      [0.75, '#ffff00'],   // Yellow
      [1, '#ff0000']       // Red (high power)
    ],
    showscale: true,
    colorbar: {
      title: 'Power (dB)',
      titleside: 'right'
    }
  }];

  const layout = {
    title: {
      text: title,
      font: { size: 16, color: '#333' }
    },
    xaxis: {
      title: 'Time',
      gridcolor: '#e0e0e0'
    },
    yaxis: {
      title: 'Normalized Frequency',
      gridcolor: '#e0e0e0'
    },
    plot_bgcolor: 'white',
    paper_bgcolor: 'white',
    margin: { t: 60, l: 60, r: 60, b: 60 },
    height: 400
  };

  const config = {
    displayModeBar: true,
    displaylogo: false,
    modeBarButtonsToRemove: ['pan2d', 'lasso2d', 'select2d']
  };

  return (
    <div className="chart-wrapper">
      <div className="chart-header">
        <h3>Spectrogram Analysis</h3>
        <div className="spectrogram-controls">
          <label htmlFor="spectrogram-view">View: </label>
          <select 
            id="spectrogram-view"
            value={selectedView} 
            onChange={(e) => setSelectedView(e.target.value)}
            className="control-select"
          >
            <option value="clean">Clean Signal</option>
            <option value="noisy">Noisy Signal</option>
            <option value="ai_denoised">AI Denoised</option>
          </select>
        </div>
      </div>
      
      <Plot
        data={plotData}
        layout={layout}
        config={config}
        style={{ width: '100%', height: '400px' }}
      />
      
      <div className="chart-description">
        <p>
          <strong>Time-frequency analysis</strong> showing signal power distribution. 
          Compare the spectrograms to see how the AI denoising algorithm reduces 
          interference while preserving the desired signal characteristics.
        </p>
        <div className="spectrogram-legend">
          <span className="legend-item">
            <span className="color-box" style={{ backgroundColor: '#000080' }}></span>
            Low Power
          </span>
          <span className="legend-item">
            <span className="color-box" style={{ backgroundColor: '#00ff00' }}></span>
            Medium Power
          </span>
          <span className="legend-item">
            <span className="color-box" style={{ backgroundColor: '#ff0000' }}></span>
            High Power
          </span>
        </div>
      </div>
    </div>
  );
};

export default Spectrogram;
