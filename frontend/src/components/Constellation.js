import React from 'react';
import Plot from 'react-plotly.js';

const Constellation = ({ data, selectedSNR }) => {
  if (!data) {
    return (
      <div className="chart-placeholder">
        <h3>Constellation Diagram</h3>
        <p>No constellation data available</p>
      </div>
    );
  }

  const snrKey = `snr_${selectedSNR}db`;
  const constellationData = data[snrKey];

  if (!constellationData) {
    return (
      <div className="chart-placeholder">
        <h3>Constellation Diagram</h3>
        <p>No data available for SNR = {selectedSNR} dB</p>
      </div>
    );
  }

  const plotData = [
    {
      x: constellationData.clean.real,
      y: constellationData.clean.imag,
      mode: 'markers',
      type: 'scatter',
      name: 'Clean Signal',
      marker: {
        color: 'green',
        size: 6,
        opacity: 0.8
      }
    },
    {
      x: constellationData.noisy.real,
      y: constellationData.noisy.imag,
      mode: 'markers',
      type: 'scatter',
      name: 'Noisy Signal',
      marker: {
        color: 'red',
        size: 4,
        opacity: 0.6
      }
    },
    {
      x: constellationData.ai_denoised.real,
      y: constellationData.ai_denoised.imag,
      mode: 'markers',
      type: 'scatter',
      name: 'AI Denoised',
      marker: {
        color: 'blue',
        size: 5,
        opacity: 0.7
      }
    }
  ];

  const layout = {
    title: {
      text: `Constellation Diagram (SNR = ${selectedSNR} dB)`,
      font: { size: 16, color: '#333' }
    },
    xaxis: {
      title: 'In-Phase (I)',
      gridcolor: '#e0e0e0',
      zerolinecolor: '#999',
      range: [-2, 2]
    },
    yaxis: {
      title: 'Quadrature (Q)',
      gridcolor: '#e0e0e0',
      zerolinecolor: '#999',
      range: [-2, 2]
    },
    plot_bgcolor: 'white',
    paper_bgcolor: 'white',
    legend: {
      x: 0.02,
      y: 0.98,
      bgcolor: 'rgba(255,255,255,0.8)',
      bordercolor: '#999',
      borderwidth: 1
    },
    margin: { t: 50, l: 60, r: 40, b: 60 },
    height: 400
  };

  const config = {
    displayModeBar: true,
    displaylogo: false,
    modeBarButtonsToRemove: ['pan2d', 'lasso2d', 'select2d', 'autoScale2d']
  };

  return (
    <div className="chart-wrapper">
      <div className="chart-header">
        <h3>Constellation Diagram</h3>
        <div className="chart-info">
          <span>Signal points in I/Q plane showing noise effects and AI correction</span>
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
          <strong>BPSK constellation points</strong> showing the effect of channel noise and AI denoising. 
          Green points represent clean transmitted symbols, red points show noise corruption, 
          and blue points demonstrate AI recovery performance.
        </p>
      </div>
    </div>
  );
};

export default Constellation;
