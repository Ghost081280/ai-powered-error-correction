#!/usr/bin/env python3
"""
Utility functions for communication system simulation
Includes modulation, FEC encoding/decoding, and channel models
"""

import numpy as np
import scipy.signal

def generate_bpsk_signal(bits, samples_per_symbol=4, symbol_rate=1000):
    """Generate BPSK modulated signal from bit sequence"""
    # Map bits to symbols: 0 -> -1, 1 -> +1
    symbols = 2 * bits - 1
    
    # Upsample to create pulse-shaped signal
    upsampled = np.repeat(symbols, samples_per_symbol)
    
    # Apply simple rectangular pulse shaping
    # In practice, you'd use raised cosine filtering
    return upsampled.astype(np.float32)

def generate_qpsk_signal(bits, samples_per_symbol=4):
    """Generate QPSK modulated signal from bit sequence"""
    # Ensure even number of bits for QPSK
    if len(bits) % 2 != 0:
        bits = np.append(bits, 0)
    
    # Group bits into pairs and map to complex symbols
    bit_pairs = bits.reshape(-1, 2)
    symbols = []
    
    qpsk_map = {
        (0, 0): 1 + 1j,
        (0, 1): -1 + 1j,
        (1, 0): 1 - 1j,
        (1, 1): -1 - 1j
    }
    
    for pair in bit_pairs:
        symbols.append(qpsk_map[tuple(pair)])
    
    symbols = np.array(symbols)
    
    # Upsample
    upsampled = np.repeat(symbols, samples_per_symbol)
    
    return upsampled

def hamming_encode(data_bits):
    """Simple Hamming(7,4) encoder"""
    # Generator matrix for Hamming(7,4)
    G = np.array([
        [1, 1, 0, 1],
        [1, 0, 1, 1],
        [1, 0, 0, 0],
        [0, 1, 1, 1],
        [0, 1, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ])
    
    # Pad data to multiple of 4
    remainder = len(data_bits) % 4
    if remainder != 0:
        data_bits = np.append(data_bits, np.zeros(4 - remainder))
    
    # Encode in blocks of 4
    encoded = []
    for i in range(0, len(data_bits), 4):
        block = data_bits[i:i+4]
        encoded_block = np.dot(G, block) % 2
        encoded.extend(encoded_block)
    
    return np.array(encoded, dtype=int)

def hamming_decode(received_bits):
    """Simple Hamming(7,4) decoder with error correction"""
    # Parity check matrix for Hamming(7,4)
    H = np.array([
        [1, 0, 1, 0, 1, 0, 1],
        [0, 1, 1, 0, 0, 1, 1],
        [0, 0, 0, 1, 1, 1, 1]
    ])
    
    # Decode in blocks of 7
    decoded = []
    errors_corrected = 0
    
    for i in range(0, len(received_bits), 7):
        if i + 7 > len(received_bits):
            break
            
        block = received_bits[i:i+7]
        
        # Calculate syndrome
        syndrome = np.dot(H, block) % 2
        
        # Check for errors
        if np.any(syndrome):
            # Find error position (syndrome as binary number)
            error_pos = syndrome[0] * 4 + syndrome[1] * 2 + syndrome[2] * 1 - 1
            if 0 <= error_pos < 7:
                block[error_pos] = 1 - block[error_pos]  # Flip bit
                errors_corrected += 1
        
        # Extract data bits (positions 2, 4, 5, 6 in 0-indexed)
        data_bits = block[[2, 4, 5, 6]]
        decoded.extend(data_bits)
    
    return np.array(decoded, dtype=int), errors_corrected

def add_channel_noise(signal, snr_db, noise_type='awgn'):
    """Add various types of noise and interference to signal"""
    signal_power = np.mean(np.abs(signal) ** 2)
    snr_linear = 10 ** (snr_db / 10)
    noise_power = signal_power / snr_linear
    
    if noise_type == 'awgn':
        # Additive White Gaussian Noise
        if np.iscomplexobj(signal):
            noise = (np.random.normal(0, np.sqrt(noise_power/2), len(signal)) + 
                    1j * np.random.normal(0, np.sqrt(noise_power/2), len(signal)))
        else:
            noise = np.random.normal(0, np.sqrt(noise_power), len(signal))
    
    elif noise_type == 'burst':
        # Burst noise (intermittent high-power noise)
        noise = np.random.normal(0, np.sqrt(noise_power/4), len(signal))
        burst_locations = np.random.random(len(signal)) < 0.05  # 5% burst probability
        noise[burst_locations] *= 10  # 10x higher noise during bursts
        
        if np.iscomplexobj(signal):
            noise = noise + 1j * np.random.normal(0, np.sqrt(noise_power/4), len(signal))
    
    elif noise_type == 'narrowband':
        # Narrowband interference
        t = np.arange(len(signal))
        interference_freq = 0.1  # Normalized frequency
        interference = np.sqrt(noise_power * 2) * np.sin(2 * np.pi * interference_freq * t)
        
        # Add some AWGN as well
        base_noise = np.random.normal(0, np.sqrt(noise_power/4), len(signal))
        noise = base_noise + interference
        
        if np.iscomplexobj(signal):
            noise = noise + 1j * np.random.normal(0, np.sqrt(noise_power/4), len(signal))
    
    else:
        raise ValueError(f"Unknown noise type: {noise_type}")
    
    return signal + noise

def calculate_ber(original_bits, recovered_bits):
    """Calculate Bit Error Rate between original and recovered bits"""
    if len(original_bits) != len(recovered_bits):
        min_len = min(len(original_bits), len(recovered_bits))
        original_bits = original_bits[:min_len]
        recovered_bits = recovered_bits[:min_len]
    
    if len(original_bits) == 0:
        return 0.0
    
    errors = np.sum(original_bits != recovered_bits)
    return errors / len(original_bits)

def demodulate_bpsk(signal, samples_per_symbol=4):
    """Demodulate BPSK signal to bits"""
    # Downsample by taking every samples_per_symbol-th sample
    symbols = signal[::samples_per_symbol]
    
    # Hard decision: positive -> 1, negative -> 0
    bits = (symbols > 0).astype(int)
    
    return bits

def demodulate_qpsk(signal, samples_per_symbol=4):
    """Demodulate QPSK signal to bits"""
    # Downsample
    symbols = signal[::samples_per_symbol]
    
    bits = []
    qpsk_demap = {
        (1, 1): [0, 0],    # First quadrant
        (-1, 1): [0, 1],   # Second quadrant  
        (1, -1): [1, 0],   # Fourth quadrant
        (-1, -1): [1, 1]   # Third quadrant
    }
    
    for symbol in symbols:
        # Hard decision based on sign of real and imaginary parts
        real_bit = 0 if symbol.real > 0 else 1
        imag_bit = 0 if symbol.imag > 0 else 1
        bits.extend([real_bit, imag_bit])
    
    return np.array(bits, dtype=int)

def compute_constellation_points(signal, samples_per_symbol=4):
    """Extract constellation points from modulated signal"""
    # Downsample to symbol rate
    symbols = signal[::samples_per_symbol]
    
    if np.iscomplexobj(symbols):
        return symbols
    else:
        # For real signals (like BPSK), create complex representation
        return symbols + 0j

def compute_spectrogram(signal, nperseg=64, noverlap=32):
    """Compute spectrogram of signal for visualization"""
    if np.iscomplexobj(signal):
        # For complex signals, compute magnitude
        signal = np.abs(signal)
    
    f, t, Sxx = scipy.signal.spectrogram(signal, nperseg=nperseg, noverlap=noverlap)
    
    # Convert to dB scale
    Sxx_db = 10 * np.log10(Sxx + 1e-10)
    
    return f, t, Sxx_db

def normalize_signal(signal):
    """Normalize signal to unit power"""
    power = np.mean(np.abs(signal) ** 2)
    if power > 0:
        return signal / np.sqrt(power)
    else:
        return signal
