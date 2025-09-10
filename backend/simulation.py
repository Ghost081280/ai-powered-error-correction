#!/usr/bin/env python3
"""
Communication System Simulation
Compares classical FEC vs AI-based error correction under various channel conditions
"""

import numpy as np
import torch
import json
import csv
import os
from train_model import SignalDenoiseAutoencoder
from utils import (
    generate_bpsk_signal, generate_qpsk_signal,
    hamming_encode, hamming_decode,
    add_channel_noise, calculate_ber, demodulate_bpsk, demodulate_qpsk,
    compute_constellation_points, compute_spectrogram, normalize_signal
)

class CommunicationSimulator:
    """Main simulation class for comparing classical vs AI error correction"""
    
    def __init__(self, model_path='data/ai_denoiser_model.pth'):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.ai_model = None
        
        # Load trained AI model if available
        if os.path.exists(model_path):
            self.ai_model = SignalDenoiseAutoencoder(input_size=128)
            self.ai_model.load_state_dict(torch.load(model_path, map_location=self.device))
            self.ai_model.to(self.device)
            self.ai_model.eval()
            print(f"Loaded AI model from {model_path}")
        else:
            print(f"Warning: AI model not found at {model_path}")
            print("Run train_model.py first to train the AI model")
    
    def simulate_transmission(self, bits, snr_db, noise_type='awgn', modulation='bpsk'):
        """Simulate transmission through noisy channel"""
        
        # Modulate signal
        if modulation == 'bpsk':
            clean_signal = generate_bpsk_signal(bits, samples_per_symbol=4)
        elif modulation == 'qpsk':
            clean_signal = generate_qpsk_signal(bits, samples_per_symbol=4)
        else:
            raise ValueError(f"Unknown modulation: {modulation}")
        
        # Add channel noise
        noisy_signal = add_channel_noise(clean_signal, snr_db, noise_type)
        
        return clean_signal, noisy_signal
    
    def classical_error_correction(self, bits, snr_db, noise_type='awgn'):
        """Classical approach: Hamming FEC + hard decision demodulation"""
        
        # Encode with Hamming code
        encoded_bits = hamming_encode(bits)
        
        # Modulate and transmit
        clean_signal, noisy_signal = self.simulate_transmission(
            encoded_bits, snr_db, noise_type, 'bpsk'
        )
        
        # Demodulate
        received_bits = demodulate_bpsk(noisy_signal)
        
        # Ensure received bits match expected length
        if len(received_bits) > len(encoded_bits):
            received_bits = received_bits[:len(encoded_bits)]
        elif len(received_bits) < len(encoded_bits):
            # Pad with zeros if needed
            padding = len(encoded_bits) - len(received_bits)
            received_bits = np.append(received_bits, np.zeros(padding, dtype=int))
        
        # Decode with error correction
        decoded_bits, errors_corrected = hamming_decode(received_bits)
        
        # Trim to original length
        if len(decoded_bits) > len(bits):
            decoded_bits = decoded_bits[:len(bits)]
        
        return decoded_bits, clean_signal, noisy_signal, errors_corrected
    
    def ai_error_correction(self, bits, snr_db, noise_type='awgn'):
        """AI approach: Neural network denoising + demodulation"""
        
        if self.ai_model is None:
            # Fallback to classical if AI model not available
            return self.classical_error_correction(bits, snr_db, noise_type)
        
        # Modulate and transmit (no FEC encoding for AI approach)
        clean_signal, noisy_signal = self.simulate_transmission(
            bits, snr_db, noise_type, 'bpsk'
        )
        
        # Ensure signal has correct length for AI model
        target_length = 128
        if len(noisy_signal) > target_length:
            noisy_signal = noisy_signal[:target_length]
        elif len(noisy_signal) < target_length:
            # Pad with zeros
            padding = target_length - len(noisy_signal)
            noisy_signal = np.append(noisy_signal, np.zeros(padding))
        
        # Normalize and denoise with AI
        normalized_noisy = normalize_signal(noisy_signal)
        
        with torch.no_grad():
            input_tensor = torch.FloatTensor(normalized_noisy).unsqueeze(0).to(self.device)
            denoised_tensor = self.ai_model(input_tensor)
            denoised_signal = denoised_tensor.cpu().numpy().squeeze()
        
        # Demodulate denoised signal
        decoded_bits = demodulate_bpsk(denoised_signal)
        
        # Trim to original length
        if len(decoded_bits) > len(bits):
            decoded_bits = decoded_bits[:len(bits)]
        
        return decoded_bits, clean_signal, denoised_signal, 0
    
    def run_ber_simulation(self, snr_range=(-5, 15, 21), num_bits=1000, 
                          noise_types=['awgn', 'burst', 'narrowband']):
        """Run BER vs SNR simulation comparing classical and AI approaches"""
        
        snr_values = np.linspace(snr_range[0], snr_range[1], snr_range[2])
        results = {
            'snr_db': snr_values.tolist(),
            'classical_ber': {nt: [] for nt in noise_types},
            'ai_ber': {nt: [] for nt in noise_types},
        }
        
        print("Running BER simulation...")
        print(f"SNR range: {snr_range[0]} to {snr_range[1]} dB")
        print(f"Noise types: {noise_types}")
        
        for snr_db in snr_values:
            print(f"\nTesting SNR = {snr_db:.1f} dB")
            
            for noise_type in noise_types:
                # Generate random bits
                bits = np.random.randint(0, 2, num_bits)
                
                # Classical approach
                classical_decoded, _, _, _ = self.classical_error_correction(
                    bits, snr_db, noise_type
                )
                classical_ber = calculate_ber(bits, classical_decoded)
                results['classical_ber'][noise_type].append(classical_ber)
                
                # AI approach
                ai_decoded, _, _, _ = self.ai_error_correction(
                    bits, snr_db, noise_type
                )
                ai_ber = calculate_ber(bits, ai_decoded)
                results['ai_ber'][noise_type].append(ai_ber)
                
                print(f"  {noise_type:12s}: Classical BER = {classical_ber:.4f}, AI BER = {ai_ber:.4f}")
        
        return results
    
    def generate_constellation_data(self, snr_values=[0, 5, 10], noise_type='awgn'):
        """Generate constellation diagram data"""
        
        results = {}
        bits = np.random.randint(0, 2, 100)  # 100 bits for constellation
        
        for snr_db in snr_values:
            # Classical approach
            classical_decoded, clean_sig, noisy_sig, _ = self.classical_error_correction(
                bits, snr_db, noise_type
            )
            
            # AI approach  
            ai_decoded, _, denoised_sig, _ = self.ai_error_correction(
                bits, snr_db, noise_type
            )
            
            # Extract constellation points
            clean_points = compute_constellation_points(clean_sig)
            noisy_points = compute_constellation_points(noisy_sig)
            denoised_points = compute_constellation_points(denoised_sig)
            
            results[f'snr_{snr_db}db'] = {
                'clean': {
                    'real': clean_points.real.tolist(),
                    'imag': clean_points.imag.tolist()
                },
                'noisy': {
                    'real': noisy_points.real.tolist(),
                    'imag': noisy_points.imag.tolist()
                },
                'ai_denoised': {
                    'real': denoised_points.real.tolist(),
                    'imag': denoised_points.imag.tolist()
                }
            }
        
        return results
    
    def generate_spectrogram_data(self, snr_db=5, noise_type='awgn'):
        """Generate spectrogram data for visualization"""
        
        bits = np.random.randint(0, 2, 200)  # Longer sequence for spectrogram
        
        # Classical approach
        classical_decoded, clean_sig, noisy_sig, _ = self.classical_error_correction(
            bits, snr_db, noise_type
        )
        
        # AI approach
        ai_decoded, _, denoised_sig, _ = self.ai_error_correction(
            bits, snr_db, noise_type
        )
        
        # Compute spectrograms
        f_clean, t_clean, S_clean = compute_spectrogram(clean_sig)
        f_noisy, t_noisy, S_noisy = compute_spectrogram(noisy_sig)
        f_denoised, t_denoised, S_denoised = compute_spectrogram(denoised_sig)
        
        return {
            'frequency': f_clean.tolist(),
            'time': t_clean.tolist(),
            'clean_spectrogram': S_clean.tolist(),
            'noisy_spectrogram': S_noisy.tolist(),
            'ai_denoised_spectrogram': S_denoised.tolist()
        }

def main():
    """Main simulation function"""
    
    # Create data directory
    os.makedirs('data', exist_ok=True)
    
    # Initialize simulator
    simulator = CommunicationSimulator()
    
    print("=== AI Error Correction Communication System Simulation ===\n")
    
    # 1. Run BER simulation
    print("1. Running BER vs SNR simulation...")
    ber_results = simulator.run_ber_simulation(
        snr_range=(-2, 12, 15),
        num_bits=1000,
        noise_types=['awgn', 'burst', 'narrowband']
    )
    
    # Save BER results as JSON
    with open('data/ber_results.json', 'w') as f:
        json.dump(ber_results, f, indent=2)
    
    # Save BER results as CSV
    with open('data/ber_results.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['SNR_dB', 'Noise_Type', 'Classical_BER', 'AI_BER'])
        
        for i, snr in enumerate(ber_results['snr_db']):
            for noise_type in ['awgn', 'burst', 'narrowband']:
                writer.writerow([
                    snr,
                    noise_type,
                    ber_results['classical_ber'][noise_type][i],
                    ber_results['ai_ber'][noise_type][i]
                ])
    
    print("   BER results saved to data/ber_results.json and data/ber_results.csv")
    
    # 2. Generate constellation data
    print("\n2. Generating constellation diagram data...")
    constellation_results = simulator.generate_constellation_data(
        snr_values=[0, 5, 10, 15],
        noise_type='awgn'
    )
    
    with open('data/constellation_data.json', 'w') as f:
        json.dump(constellation_results, f, indent=2)
    
    print("   Constellation data saved to data/constellation_data.json")
    
    # 3. Generate spectrogram data
    print("\n3. Generating spectrogram data...")
    spectrogram_results = simulator.generate_spectrogram_data(
        snr_db=5,
        noise_type='awgn'
    )
    
    with open('data/spectrogram_data.json', 'w') as f:
        json.dump(spectrogram_results, f, indent=2)
    
    print("   Spectrogram data saved to data/spectrogram_data.json")
    
    # 4. Generate summary statistics
    print("\n4. Computing summary statistics...")
    
    # Find best performing approach for each noise type
    summary = {
        'simulation_parameters': {
            'num_bits': 1000,
            'snr_range_db': [-2, 12],
            'noise_types': ['awgn', 'burst', 'narrowband']
        },
        'performance_summary': {}
    }
    
    for noise_type in ['awgn', 'burst', 'narrowband']:
        classical_bers = ber_results['classical_ber'][noise_type]
        ai_bers = ber_results['ai_ber'][noise_type]
        
        avg_classical = np.mean(classical_bers)
        avg_ai = np.mean(ai_bers)
        
        improvement = (avg_classical - avg_ai) / avg_classical * 100 if avg_classical > 0 else 0
        
        summary['performance_summary'][noise_type] = {
            'average_classical_ber': avg_classical,
            'average_ai_ber': avg_ai,
            'improvement_percent': improvement,
            'ai_better': avg_ai < avg_classical
        }
    
    with open('data/simulation_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    
    print("   Summary statistics saved to data/simulation_summary.json")
    
    print("\n=== Simulation Complete ===")
    print("Results saved in the data/ directory:")
    print("  - ber_results.json/csv: BER vs SNR data")
    print("  - constellation_data.json: Constellation diagram points")
    print("  - spectrogram_data.json: Time-frequency analysis")
    print("  - simulation_summary.json: Performance summary")
    print("\nCopy the data/ directory contents to frontend/public/data/ to visualize results!")

if __name__ == "__main__":
    main()
