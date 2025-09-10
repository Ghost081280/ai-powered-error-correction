#!/usr/bin/env python3
"""
AI Error Correction Model Training
Trains a PyTorch autoencoder to denoise corrupted communication signals
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
import json
import os
from utils import generate_bpsk_signal, add_channel_noise

class SignalDenoiseAutoencoder(nn.Module):
    """Simple autoencoder for signal denoising"""
    
    def __init__(self, input_size=128):
        super(SignalDenoiseAutoencoder, self).__init__()
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU()
        )
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(16, 32),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, input_size),
            nn.Tanh()  # Output bounded between -1 and 1
        )
    
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

def generate_training_data(num_samples=10000, signal_length=128):
    """Generate training data with clean and noisy signal pairs"""
    clean_signals = []
    noisy_signals = []
    
    print(f"Generating {num_samples} training samples...")
    
    for i in range(num_samples):
        # Generate random bit sequence
        bits = np.random.randint(0, 2, signal_length // 4)  # Fewer bits for oversampling
        
        # Create clean BPSK signal
        clean_signal = generate_bpsk_signal(bits, samples_per_symbol=4)
        
        # Add various types of noise and interference
        snr_db = np.random.uniform(-5, 20)  # Random SNR between -5 and 20 dB
        noise_type = np.random.choice(['awgn', 'burst', 'narrowband'])
        
        noisy_signal = add_channel_noise(clean_signal, snr_db, noise_type)
        
        # Normalize signals
        clean_signal = clean_signal / np.max(np.abs(clean_signal))
        noisy_signal = noisy_signal / np.max(np.abs(noisy_signal))
        
        clean_signals.append(clean_signal)
        noisy_signals.append(noisy_signal)
        
        if (i + 1) % 1000 == 0:
            print(f"Generated {i + 1}/{num_samples} samples")
    
    return np.array(clean_signals), np.array(noisy_signals)

def train_model(num_epochs=50, batch_size=32, learning_rate=0.001):
    """Train the signal denoising autoencoder"""
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Training on device: {device}")
    
    # Generate training data
    clean_signals, noisy_signals = generate_training_data()
    
    # Convert to PyTorch tensors
    clean_tensor = torch.FloatTensor(clean_signals).to(device)
    noisy_tensor = torch.FloatTensor(noisy_signals).to(device)
    
    # Create dataset and dataloader
    dataset = TensorDataset(noisy_tensor, clean_tensor)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # Initialize model
    model = SignalDenoiseAutoencoder(input_size=clean_signals.shape[1]).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Training loop
    train_losses = []
    
    print(f"Starting training for {num_epochs} epochs...")
    
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        num_batches = 0
        
        for batch_noisy, batch_clean in dataloader:
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(batch_noisy)
            loss = criterion(outputs, batch_clean)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            num_batches += 1
        
        avg_loss = epoch_loss / num_batches
        train_losses.append(avg_loss)
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.6f}")
    
    # Save model
    os.makedirs('data', exist_ok=True)
    torch.save(model.state_dict(), 'data/ai_denoiser_model.pth')
    
    # Save training metrics
    training_metrics = {
        'epochs': num_epochs,
        'final_loss': train_losses[-1],
        'training_losses': train_losses,
        'model_architecture': {
            'input_size': clean_signals.shape[1],
            'hidden_layers': [64, 32, 16],
            'activation': 'ReLU'
        }
    }
    
    with open('data/training_metrics.json', 'w') as f:
        json.dump(training_metrics, f, indent=2)
    
    # Plot training curve
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses)
    plt.title('Training Loss Over Time')
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.grid(True)
    plt.savefig('data/training_curve.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Training completed! Final loss: {train_losses[-1]:.6f}")
    print("Model saved to: data/ai_denoiser_model.pth")
    print("Training metrics saved to: data/training_metrics.json")
    
    return model, train_losses

if __name__ == "__main__":
    # Train the model
    model, losses = train_model(num_epochs=50)
    
    print("\nTraining complete! Run simulation.py to test the model performance.")
