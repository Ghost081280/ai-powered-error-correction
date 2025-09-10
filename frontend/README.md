# AI Error Correction Frontend

Interactive React dashboard for visualizing AI-powered error correction performance vs classical FEC methods.

## Features

- **BER vs SNR Charts**: Compare bit error rates between classical Hamming codes and AI denoising
- **Constellation Diagrams**: Visualize signal points showing noise effects and AI correction
- **Spectrogram Analysis**: Time-frequency view of signals before/after processing
- **Interactive Controls**: Switch between noise types, toggle display options, adjust parameters
- **Performance Summary**: Statistical comparison of methods across different channel conditions

## Quick Start

### Prerequisites
- Node.js 16+ and npm
- Simulation data from backend (see [Backend README](../backend/README.md))

### Setup and Run Locally

```bash
# Install dependencies
npm install

# Start development server
npm start
```

The app will open at `http://localhost:3000`

### Deploy to GitHub Pages

1. **Update package.json**: Replace `yourusername` with your GitHub username:
   ```json
   "homepage": "https://yourusername.github.io/ai-error-correction-demo"
   ```

2. **Build and deploy**:
   ```bash
   npm run deploy
   ```

   This will:
   - Build the production version
   - Push to `gh-pages` branch
   - Make the site available at your GitHub Pages URL

### Data Setup

The frontend expects simulation data in `public/data/`:

```
frontend/public/data/
├── ber_results.json          # BER vs SNR data
├── constellation_data.json   # I/Q constellation points
├── spectrogram_data.json     # Time-frequency analysis
└── simulation_summary.json   # Performance statistics
```

To populate this data:

1. **Run backend simulation**:
   ```bash
   cd ../backend
   python train_model.py
   python simulation.py
   ```

2. **Copy data to frontend**:
   ```bash
   cp -r ../backend/data/* public/data/
   ```

3. **Verify data files**:
   ```bash
   ls -la public/data/
   ```

## Architecture

### Components

- **App.js**: Main application with data loading and state management
- **BERChart.js**: Recharts-based BER vs SNR visualization
- **Constellation.js**: Plotly.js I/Q constellation diagram
- **Spectrogram.js**: Plotly.js time-frequency heatmap
- **Controls.js**: Interactive control panel with performance summary

### Data Flow

1. App loads JSON data from `/public/data/` on startup
2. Control changes update visualization parameters
3. Charts re-render based on selected noise type, SNR, display options
4. Performance summary shows comparative statistics

### Styling

- **CSS**: Custom styles in `src/styles.css`
- **Responsive**: Mobile-friendly grid layouts
- **Theme**: Professional blue/white with glass-morphism effects

## Dependencies

### Core
- **React 18**: UI framework
- **Recharts**: BER charts and line graphs
- **Plotly.js**: Constellation and spectrogram visualizations

### Development
- **react-scripts**: Create React App tooling
- **gh-pages**: GitHub Pages deployment

## Configuration

### Environment Variables

Create `.env.local` for local development:

```env
REACT_APP_DATA_PATH=/data
```

### Build Configuration

The app is configured for static hosting:
- All data loaded via client-side fetch
- No backend API dependencies
- Builds to static files in `build/`

## Deployment Options

### GitHub Pages (Recommended)
```bash
npm run deploy
```

### Manual Static Hosting
```bash
npm run build
# Upload build/ directory to your web server
```

### Docker
```dockerfile
FROM nginx:alpine
COPY build/ /usr/share/nginx/html/
```

## Troubleshooting

### Data Loading Issues

**Problem**: "No simulation data available"
**Solution**: 
1. Verify backend simulation completed successfully
2. Check data files exist in `public/data/`
3. Ensure JSON files are valid (use `jq` or JSON validator)

**Problem**: Charts show "No data available"
**Solution**:
1. Check browser DevTools network tab for 404 errors
2. Verify JSON structure matches expected format
3. Check console for parsing errors

### Deployment Issues

**Problem**: GitHub Pages shows 404
**Solution**:
1. Verify `homepage` in package.json matches your repo
2. Check GitHub Pages settings in repo settings
3. Ensure `gh-pages` branch was created

**Problem**: Blank page after deployment
**Solution**:
1. Check browser console for errors
2. Verify all assets loading correctly
3. Check relative path issues (use `/` prefix for absolute paths)

### Performance Issues

**Problem**: Slow loading or rendering
**Solution**:
1. Reduce data size in backend simulation
2. Implement data pagination for large datasets
3. Use React.memo for expensive components

## Development

### Adding New Visualizations

1. Create component in `src/components/`
2. Add data loading logic in `App.js`
3. Update controls if needed
4. Add styling to `styles.css`

### Modifying Data Format

1. Update backend simulation output
2. Modify frontend data loading/parsing
3. Update component prop interfaces
4. Test with new data format

### Customizing Styling

- Edit `src/styles.css` for global styles
- Use CSS custom properties for theming
- Maintain responsive design principles

## License

MIT License - see [LICENSE](../LICENSE) for details.
