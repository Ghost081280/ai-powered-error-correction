# AI-Powered Error Correction for UHF Communications

This repository demonstrates **Phase 1** of an AI-powered error correction system designed for UHF communications under jamming/noise conditions. It compares classical Forward Error Correction (FEC) using Hamming codes against a neural network-based signal denoising approach.

## 🚀 Live Demo

[View Interactive Demo](https://yourusername.github.io/ai-error-correction-demo)

## 📁 Repository Structure

```
ai-error-correction-demo/
├── backend/                   # Python simulation and AI training
│   ├── train_model.py         # PyTorch autoencoder training
│   ├── simulation.py          # Classical vs AI comparison simulation
│   ├── utils.py               # Signal processing utilities
│   ├── requirements.txt       # Python dependencies
│   └── data/                  # Generated simulation results
│
├── frontend/                  # React visualization dashboard
│   ├── public/
│   │   ├── index.html
│   │   └── data/              # Simulation data for visualization
│   ├── src/
│   │   ├── App.js             # Main application
│   │   ├── components/        # Visualization components
│   │   │   ├── BERChart.js    # BER vs SNR line charts
│   │   │   ├── Constellation.js # I/Q constellation diagrams  
│   │   │   ├── Spectrogram.js # Time-frequency analysis
│   │   │   └── Controls.js    # Interactive control panel
│   │   └── styles.css         # Styling
│   ├── package.json           # Node.js dependencies
│   └── README.md              # Frontend documentation
│
└── README.md                  # This file
```

## 🎯 What This Demonstrates

### Classical Approach
- **Hamming(7,4) FEC**: Adds redundancy for error detection/correction
- **Hard Decision Demodulation**: Binary threshold detection
- **Fixed Performance**: Consistent but limited error correction capability

### AI Approach  
- **Neural Network Denoising**: PyTorch autoencoder trained on noisy signals
- **Signal-Level Processing**: Operates directly on waveforms before demodulation
- **Adaptive Performance**: Learns optimal denoising for different interference types

### Channel Conditions Tested
- **AWGN**: Additive White Gaussian Noise (thermal noise)
- **Burst Noise**: Intermittent high-power interference  
- **Narrowband Interference**: Continuous sinusoidal jamming

## 🔧 Quick Start

### 1. Backend Simulation

```bash
# Navigate to backend
cd backend

# Install Python dependencies
pip install -r requirements.txt

# Train the AI model (takes ~2-5 minutes)
python train_model.py

# Run comparative simulation (takes ~3-10 minutes)
python simulation.py
```

This generates:
- `data/ai_denoiser_model.pth` - Trained neural network weights
- `data/ber_results.json` - BER vs SNR comparison data
- `data/constellation_data.json` - Signal constellation points
- `data/spectrogram_data.json` - Time-frequency analysis
- `data/simulation_summary.json` - Performance statistics

### 2. Frontend Visualization

```bash
# Copy simulation data to frontend
cp -r backend/data/* frontend/public/data/

# Navigate to frontend
cd frontend

# Install Node.js dependencies
npm install

# Start development server
npm start
```

Visit `http://localhost:3000` to explore the interactive dashboard.

### 3. Deploy to GitHub Pages

```bash
# From frontend directory
# First, update package.json homepage to your GitHub repo
# Then deploy:
npm run deploy
```

Your live demo will be available at `https://yourusername.github.io/ai-error-correction-demo`

## 📊 Key Results

The AI approach typically shows **10-40% improvement** in BER performance, especially:

- **Low SNR conditions** (-2 to 5 dB): AI excels in high-noise environments
- **Burst interference**: Neural network adapts to intermittent jamming
- **Narrowband jamming**: AI learns to suppress sinusoidal interference

Performance varies by channel condition - classical FEC may outperform in some scenarios.

## 🧠 Technical Approach

### AI Model Architecture
- **Input**: 128-sample corrupted signal segments
- **Encoder**: 128 → 64 → 32 → 16 neurons (ReLU activation)
- **Decoder**: 16 → 32 → 64 → 128 neurons (ReLU + Tanh output)
- **Training**: 10,000 clean/noisy signal pairs, 50 epochs
- **Loss Function**: Mean Squared Error between clean and denoised signals

### Signal Processing Pipeline
1. **Generate Data**: Random bits → BPSK modulation → Channel noise
2. **Classical Path**: Hamming encode → Transmit → Hard demod → Hamming decode
3. **AI Path**: Transmit → Neural network denoise → Soft demod
4. **Evaluation**: Compare recovered bits to original, calculate BER

### Visualization Features
- **Interactive BER Charts**: Toggle between algorithms and noise types
- **Constellation Diagrams**: See I/Q signal points before/after processing  
- **Spectrograms**: Time-frequency analysis showing interference patterns
- **Performance Summary**: Statistical comparison across all conditions

## 🛠 Customization

### Modify AI Architecture
Edit `backend/train_model.py`:
```python
class SignalDenoiseAutoencoder(nn.Module):
    def __init__(self, input_size=128):
        # Modify layer sizes, add dropout, change activation functions
```

### Add New Interference Types
Edit `backend/utils.py`:
```python
def add_channel_noise(signal, snr_db, noise_type='new_interference'):
    # Implement new interference model
```

### Extend Visualizations
Create new components in `frontend/src/components/` and add to main dashboard.

## 📚 Dependencies

### Backend (Python)
- **PyTorch**: Neural network training and inference
- **NumPy**: Numerical computations and signal processing
- **Matplotlib**: Training curve visualization
- **SciPy**: Signal processing utilities

### Frontend (React)
- **Recharts**: BER vs SNR line charts
- **Plotly.js**: Interactive constellation and spectrogram plots
- **React 18**: Modern UI framework

## 🚧 Future Enhancements (Phase 2+)

- **Advanced Modulation**: QPSK, QAM, OFDM support
- **Stronger FEC**: LDPC, Turbo codes, Polar codes
- **Recurrent Networks**: LSTM/GRU for sequential processing  
- **Attention Mechanisms**: Transformer-based denoising
- **Real-time Processing**: Streaming inference optimization
- **Hardware Implementation**: FPGA/GPU acceleration
- **Adaptive Learning**: Online model updates based on channel conditions

## 📖 Educational Use

This repository is designed for:
- **Academic Research**: Baseline for AI-enhanced communications
- **Student Projects**: Learn signal processing + machine learning integration
- **Industry Prototyping**: Evaluate AI techniques for interference mitigation
- **Algorithm Development**: Test new neural architectures for communications

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **PyTorch Team**: Excellent deep learning framework
- **React Team**: Powerful frontend library  
- **Recharts & Plotly**: Beautiful visualization libraries
- **Communications Community**: Inspiration from classical and modern techniques

---

**Built with ❤️ for the communications and AI community**

*Exploring the intersection of signal processing and artificial intelligence*
