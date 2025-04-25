from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import numpy as np
from scipy.special import erfc
from scipy import signal
import matplotlib.pyplot as plt
import base64
from io import BytesIO
from typing import List, Optional

app = FastAPI(title="BPSK SRRC Simulation API")

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, restrict this to your frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class SimulationParams(BaseModel):
    K: int = 100000  # Number of symbols
    EbN0_dB: float = 4.0  # Eb/N0 ratio in dB
    alpha: float = 0.22  # Roll-off factor
    sps: int = 8  # Samples per symbol
    span: int = 10  # Number of symbols covered by filter
    seed: Optional[int] = None  # Random seed for reproducibility
    display_length: int = 20  # Number of bits to display in plots

# Function to create the SRRC filter
def srrc_pulse(alpha, span, sps):
    """
    Generates a Square Root Raised Cosine (SRRC) filter.

    Args:
        alpha: Roll-off factor (0 <= alpha <= 1)
        span: Half-length of filter in symbols
        sps: Samples per symbol

    Returns:
        Normalized SRRC filter
    """
    n = np.arange(-span * sps, span * sps + 1)
    t = n / sps

    # Avoid division by zero
    pulse = np.zeros_like(t, dtype=float)

    # t = 0
    pulse[t == 0] = 1.0 - alpha + (4 * alpha / np.pi)

    # t = ±Ts/(4*alpha)
    idx1 = np.abs(np.abs(t) - 1.0 / (4.0 * alpha)) < 1e-10
    if np.any(idx1):
        pulse[idx1] = (alpha / np.sqrt(2)) * (
            (1.0 + 2.0 / np.pi) * np.sin(np.pi / (4.0 * alpha)) +
            (1.0 - 2.0 / np.pi) * np.cos(np.pi / (4.0 * alpha))
        )

    # Other values of t
    idx2 = (t != 0) & (np.abs(np.abs(t) - 1.0 / (4.0 * alpha)) >= 1e-10)
    numer = np.sin(np.pi * t[idx2] * (1.0 - alpha)) + 4.0 * alpha * t[idx2] * np.cos(np.pi * t[idx2] * (1.0 + alpha))
    denom = np.pi * t[idx2] * (1.0 - (4.0 * alpha * t[idx2])**2)
    pulse[idx2] = numer / denom

    # Normalize for unit energy
    return pulse / np.sqrt(np.sum(pulse**2))

def plot_to_base64(fig):
    """Convert matplotlib figure to base64 string"""
    buf = BytesIO()
    fig.savefig(buf, format='png')
    buf.seek(0)
    img_str = base64.b64encode(buf.read()).decode('utf-8')
    buf.close()
    plt.close(fig)
    return img_str

@app.get("/")
async def root():
    # Root function to check API health
    return {"message": "BPSK SRRC Simulation API is running"}

@app.post("/advanced-simulate")
async def run_advanced_simulation(params: SimulationParams):
    try:
        # Set random seed if provided
        if params.seed is not None:
            np.random.seed(params.seed)

        # Convert Eb/N0
        EbN0_linear = 10 ** (params.EbN0_dB / 10)
        sigma = np.sqrt(1 / (2 * EbN0_linear))

        # Create SRRC filter
        srrc_filter = srrc_pulse(params.alpha, params.span, params.sps)

        # Generate bits
        bits = np.random.randint(0, 2, params.K)

        # BPSK modulation
        symbols = 2 * bits - 1

        # Upsampling with zeros
        symbol_upsampled = np.zeros(params.K * params.sps)
        symbol_upsampled[::params.sps] = symbols

        # Filter with SRRC at transmitter
        tx_signal = signal.fftconvolve(symbol_upsampled, srrc_filter)

        # Add noise
        noise = sigma * np.random.randn(len(tx_signal))
        rx_signal = tx_signal + noise

        # Matched filtering at receiver
        matched_filtered = signal.fftconvolve(rx_signal, srrc_filter[::-1])

        # Total delay is 2*span*sps due to two convolutions
        delay = 2 * params.span * params.sps

        # Sample at optimal instants
        sampled_signal = matched_filtered[delay + np.arange(0, params.K) * params.sps]

        # Decision
        decoded_bits = (sampled_signal > 0).astype(int)

        # Calculate errors
        errors = np.sum(bits != decoded_bits)
        ber = errors / params.K
        theoretical_ber = 0.5 * erfc(np.sqrt(EbN0_linear))

        # Generate plots
        plots = {}
        
        # SRRC filter plot
        fig_filter = plt.figure(figsize=(10, 6))
        time_axis = np.arange(-params.span, params.span + 1 / params.sps, 1 / params.sps)
        plt.plot(time_axis, srrc_filter)
        plt.grid(True)
        plt.xlabel('Time (symbol periods)')
        plt.ylabel('Amplitude')
        plt.title(f'SRRC Filter (α = {params.alpha})')
        plt.axhline(y=0, color='r', linestyle='-', alpha=0.3)
        plt.axvline(x=0, color='r', linestyle='-', alpha=0.3)
        plots['srrc_filter'] = plot_to_base64(fig_filter)
        
        # 1. BPSK Constellation at Transmitter
        fig_constellation_tx = plt.figure(figsize=(8, 8))
        plt.plot(symbols[:100].real, np.zeros_like(symbols[:100]), 'bo')
        plt.grid(True)
        plt.xlim([-1.5, 1.5])
        plt.ylim([-1.5, 1.5])
        plt.title('BPSK Constellation at Transmitter')
        plt.xlabel('In-Phase')
        plt.ylabel('Quadrature')
        plots['bpsk_constellation_tx'] = plot_to_base64(fig_constellation_tx)
        
        # 2. Constellation of the filtered transmitted signal
        fig_constellation_tx_filtered = plt.figure(figsize=(8, 8))
        sample_points = tx_signal[params.sps - 1::params.sps][:100]
        plt.plot(sample_points.real, np.zeros_like(sample_points), 'bo')
        plt.grid(True)
        plt.xlim([-1.5, 1.5])
        plt.ylim([-1.5, 1.5])
        plt.title('Constellation of Filtered Signal (Tx)')
        plt.xlabel('In-Phase')
        plt.ylabel('Quadrature')
        plots['constellation_tx_filtered'] = plot_to_base64(fig_constellation_tx_filtered)
        
        # 3. Eye diagram at transmitter
        fig_eye_tx = plt.figure(figsize=(10, 6))
        eye_length = 2 * params.sps  # 2 symbol periods
        num_traces = 50
        
        for i in range(num_traces):
            start_idx = params.span * params.sps + i * params.sps
            if start_idx + eye_length <= len(tx_signal):
                plt.plot(np.arange(eye_length) / params.sps, tx_signal[start_idx:start_idx + eye_length])
                
        plt.grid(True)
        plt.title('Eye Diagram at Transmitter')
        plt.xlabel('Time (symbol periods)')
        plt.ylabel('Amplitude')
        plots['eye_diagram_tx'] = plot_to_base64(fig_eye_tx)
        
        # 4. Constellation of the received signal (with noise)
        fig_constellation_rx = plt.figure(figsize=(8, 8))
        sample_points_rx = rx_signal[params.sps - 1::params.sps][:100]
        plt.plot(sample_points_rx.real, np.zeros_like(sample_points_rx), 'ro', alpha=0.5)
        plt.grid(True)
        plt.xlim([-2, 2])
        plt.ylim([-2, 2])
        plt.title('Constellation of Received Signal (with noise)')
        plt.xlabel('In-Phase')
        plt.ylabel('Quadrature')
        plots['constellation_rx'] = plot_to_base64(fig_constellation_rx)
        
        # 5. Eye diagram at receiver (before filtering)
        fig_eye_rx = plt.figure(figsize=(10, 6))
        
        for i in range(num_traces):
            start_idx = params.span * params.sps + i * params.sps
            if start_idx + eye_length <= len(rx_signal):
                plt.plot(np.arange(eye_length) / params.sps, rx_signal[start_idx:start_idx + eye_length])
                
        plt.grid(True)
        plt.title('Eye Diagram at Receiver (before filtering)')
        plt.xlabel('Time (symbol periods)')
        plt.ylabel('Amplitude')
        plots['eye_diagram_rx'] = plot_to_base64(fig_eye_rx)
        
        # 6. Constellation after matched filtering
        fig_constellation_rx_filtered = plt.figure(figsize=(8, 8))
        plt.plot(sampled_signal[:100].real, np.zeros_like(sampled_signal[:100]), 'go')
        plt.grid(True)
        plt.xlim([-2, 2])
        plt.ylim([-2, 2])
        plt.title('Constellation after Matched Filtering')
        plt.xlabel('In-Phase')
        plt.ylabel('Quadrature')
        plots['constellation_rx_filtered'] = plot_to_base64(fig_constellation_rx_filtered)
        
        # 7. Eye diagram after matched filtering
        fig_eye_rx_filtered = plt.figure(figsize=(10, 6))
        
        for i in range(num_traces):
            trace_start = delay + i * params.sps
            if trace_start + eye_length <= len(matched_filtered):
                plt.plot(np.arange(eye_length) / params.sps, matched_filtered[trace_start:trace_start + eye_length])
        
        plt.axvline(x=1.0, color='r', linestyle='--', label='Sampling Point')
        plt.grid(True)
        plt.title('Eye Diagram after Matched Filtering')
        plt.xlabel('Time (symbol periods)')
        plt.ylabel('Amplitude')
        plt.legend()
        plots['eye_diagram_rx_filtered'] = plot_to_base64(fig_eye_rx_filtered)

        # Return results
        return {
            "plots": plots,
            "results": {
                "ber": ber,
                "theoretical_ber": theoretical_ber,
                "errors": int(errors),
                "total_bits": params.K,
                "input_bits": bits[:params.display_length].tolist(),
                "decoded_bits": decoded_bits[:params.display_length].tolist(),
                "sampled_signal": sampled_signal[:params.display_length].tolist()
            }
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))