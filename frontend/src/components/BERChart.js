import React from 'react';
import { 
  LineChart, 
  Line, 
  XAxis, 
  YAxis, 
  CartesianGrid, 
  Tooltip, 
  Legend, 
  ResponsiveContainer 
} from 'recharts';

const BERChart = ({ data, selectedNoiseType, showAI, showClassical }) => {
  if (!data || !data.snr_db) {
    return (
      <div className="chart-placeholder">
        <h3>Bit Error Rate vs Signal-to-Noise Ratio</h3>
        <p>No BER data available</p>
      </div>
    );
  }

  // Transform data for Recharts
  const chartData = data.snr_db.map((snr, index) => ({
    snr: snr,
    classical: data.classical_ber[selectedNoiseType]?.[index] || 0,
    ai: data.ai_ber[selectedNoiseType]?.[index] || 0
  }));

  const CustomTooltip = ({ active, payload, label }) => {
    if (active && payload && payload.length) {
      return (
        <div className="custom-tooltip">
          <p className="tooltip-label">{`SNR: ${label} dB`}</p>
          {payload.map((entry, index) => (
            <p key={index} className="tooltip-value" style={{ color: entry.color }}>
              {`${entry.name}: ${entry.value.toExponential(2)}`}
            </p>
          ))}
        </div>
      );
    }
    return null;
  };

  return (
    <div className="chart-wrapper">
      <div className="chart-header">
        <h3>Bit Error Rate vs Signal-to-Noise Ratio</h3>
        <div className="chart-info">
          <span className="noise-type-indicator">
            Noise Type: <strong>{selectedNoiseType.toUpperCase()}</strong>
          </span>
        </div>
      </div>
      
      <ResponsiveContainer width="100%" height={400}>
        <LineChart
          data={chartData}
          margin={{
            top: 20,
            right: 30,
            left: 40,
            bottom: 60,
          }}
        >
          <CartesianGrid strokeDasharray="3 3" stroke="#e0e0e0" />
          <XAxis 
            dataKey="snr" 
            label={{ value: 'SNR (dB)', position: 'insideBottom', offset: -10 }}
            stroke="#666"
          />
          <YAxis 
            scale="log"
            domain={['dataMin', 'dataMax']}
            label={{ value: 'Bit Error Rate', angle: -90, position: 'insideLeft' }}
            stroke="#666"
            tickFormatter={(value) => value.toExponential(0)}
          />
          <Tooltip content={<CustomTooltip />} />
          <Legend />
          
          {showClassical && (
            <Line 
              type="monotone" 
              dataKey="classical" 
              stroke="#ff7300" 
              strokeWidth={3}
              dot={{ fill: '#ff7300', strokeWidth: 2, r: 4 }}
              name="Classical FEC"
              connectNulls={false}
            />
          )}
          
          {showAI && (
            <Line 
              type="monotone" 
              dataKey="ai" 
              stroke="#007fff" 
              strokeWidth={3}
              dot={{ fill: '#007fff', strokeWidth: 2, r: 4 }}
              name="AI Denoising"
              connectNulls={false}
            />
          )}
        </LineChart>
      </ResponsiveContainer>
      
      <div className="chart-description">
        <p>
          <strong>Lower is better.</strong> Comparison of classical Forward Error Correction (Hamming codes) 
          vs AI-based signal denoising under different channel conditions. The AI approach shows improved 
          performance especially at low SNR conditions.
        </p>
      </div>
    </div>
  );
};

export default BERChart;
