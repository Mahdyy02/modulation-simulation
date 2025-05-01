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
from enum import Enum

class ModulationType(str, Enum):
    BPSK = "bpsk"
    QPSK = "qpsk"
    QAM = "qam"

app = FastAPI(title="Digital Modulation SRRC Simulation API")

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, restrict this to your frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class SimulationParams(BaseModel):
    modulation_type: ModulationType = ModulationType.BPSK  # Type of modulation
    K: int = 100000  # Number of symbols
    EbN0_dB: float = 4.0  # Eb/N0 ratio in dB
    alpha: float = 0.22  # Roll-off factor
    sps: int = 8  # Samples per symbol
    span: int = 10  # Number of symbols covered by filter
    seed: Optional[int] = None  # Random seed for reproducibility
    display_length: int = 20  # Number of bits to display in plots
    qam_order: Optional[int] = None  # QAM order (2^l, where l is the number of bits per symbol)

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
    return {"message": "Digital Modulation SRRC Simulation API is running"}

@app.post("/advanced-simulate-bpsk")
async def run_bpsk_simulation(params: SimulationParams):
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
        
        # Theoretical BER for BPSK
        theoretical_ber = 0.5 * erfc(np.sqrt(EbN0_linear))

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

        # Decision for BPSK
        decoded_bits = (sampled_signal > 0).astype(int)

        # Calculate errors
        errors = np.sum(bits != decoded_bits)
        ber = errors / params.K

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
        
        # BPSK Constellation at Transmitter
        fig_constellation_tx = plt.figure(figsize=(8, 8))
        plt.plot(symbols[:100].real, np.zeros_like(symbols[:100]), 'bo')
        plt.grid(True)
        plt.xlim([-1.5, 1.5])
        plt.ylim([-1.5, 1.5])
        plt.title('BPSK Constellation at Transmitter')
        plt.xlabel('In-Phase')
        plt.ylabel('Quadrature')
        plots['constellation_tx'] = plot_to_base64(fig_constellation_tx)
        
        # Constellation of the filtered transmitted signal
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
        
        # Eye diagram at transmitter
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
        
        # Constellation of the received signal (with noise)
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
        
        # Eye diagram at receiver (before filtering)
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
        
        # Constellation after matched filtering
        fig_constellation_rx_filtered = plt.figure(figsize=(8, 8))
        plt.plot(sampled_signal[:100].real, np.zeros_like(sampled_signal[:100]), 'go')
        plt.grid(True)
        plt.xlim([-2, 2])
        plt.ylim([-2, 2])
        plt.title('Constellation after Matched Filtering')
        plt.xlabel('In-Phase')
        plt.ylabel('Quadrature')
        plots['constellation_rx_filtered'] = plot_to_base64(fig_constellation_rx_filtered)
        
        # Eye diagram after matched filtering
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

@app.post("/advanced-simulate-qpsk")
async def run_qpsk_simulation(params: SimulationParams):
    try:
        # Set random seed if provided
        if params.seed is not None:
            np.random.seed(params.seed)

        # Convert Eb/N0
        EbN0_linear = 10 ** (params.EbN0_dB / 10)
        sigma = np.sqrt(1 / (2 * EbN0_linear))

        # Create SRRC filter
        srrc_filter = srrc_pulse(params.alpha, params.span, params.sps)

        # Generate bits (2 bits per symbol)
        bits = np.random.randint(0, 2, params.K * 2)
        
        # QPSK modulation
        # Map bits to symbols: 00 -> (1+j)/sqrt(2), 01 -> (-1+j)/sqrt(2), 11 -> (-1-j)/sqrt(2), 10 -> (1-j)/sqrt(2)
        symbols = np.zeros(params.K, dtype=complex)
        for i in range(0, len(bits), 2):
            if bits[i] == 0 and bits[i+1] == 0:
                symbols[i//2] = (1 + 1j) / np.sqrt(2)
            elif bits[i] == 0 and bits[i+1] == 1:
                symbols[i//2] = (-1 + 1j) / np.sqrt(2)
            elif bits[i] == 1 and bits[i+1] == 1:
                symbols[i//2] = (-1 - 1j) / np.sqrt(2)
            else:  # bits[i] == 1 and bits[i+1] == 0
                symbols[i//2] = (1 - 1j) / np.sqrt(2)
        
        # Theoretical BER for QPSK
        theoretical_ber = erfc(np.sqrt(EbN0_linear)) - 0.25 * erfc(np.sqrt(EbN0_linear))**2

        # Upsampling with zeros
        symbol_upsampled = np.zeros(params.K * params.sps, dtype=complex)
        symbol_upsampled[::params.sps] = symbols

        # Filter with SRRC at transmitter
        tx_signal = signal.fftconvolve(symbol_upsampled, srrc_filter)

        # Add noise
        noise = sigma * (np.random.randn(len(tx_signal)) + 1j * np.random.randn(len(tx_signal))) / np.sqrt(2)
        rx_signal = tx_signal + noise

        # Matched filtering at receiver
        matched_filtered = signal.fftconvolve(rx_signal, srrc_filter[::-1])

        # Total delay is 2*span*sps due to two convolutions
        delay = 2 * params.span * params.sps

        # Sample at optimal instants
        sampled_signal = matched_filtered[delay + np.arange(0, params.K) * params.sps]

        # Decision for QPSK
        decoded_bits = np.zeros(params.K * 2, dtype=int)
        for i in range(params.K):
            # Map complex symbol to bits
            if sampled_signal[i].real > 0 and sampled_signal[i].imag > 0:
                decoded_bits[2*i] = 0
                decoded_bits[2*i+1] = 0
            elif sampled_signal[i].real < 0 and sampled_signal[i].imag > 0:
                decoded_bits[2*i] = 0
                decoded_bits[2*i+1] = 1
            elif sampled_signal[i].real < 0 and sampled_signal[i].imag < 0:
                decoded_bits[2*i] = 1
                decoded_bits[2*i+1] = 1
            else:  # sampled_signal[i].real > 0 and sampled_signal[i].imag < 0
                decoded_bits[2*i] = 1
                decoded_bits[2*i+1] = 0

        # Calculate errors
        errors = np.sum(bits != decoded_bits)
        ber = errors / len(bits)

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
        
        # QPSK Constellation at Transmitter
        fig_constellation_tx = plt.figure(figsize=(8, 8))
        plt.plot(symbols[:100].real, symbols[:100].imag, 'bo')
        plt.grid(True)
        plt.xlim([-1.5, 1.5])
        plt.ylim([-1.5, 1.5])
        plt.title('QPSK Constellation at Transmitter')
        plt.xlabel('In-Phase')
        plt.ylabel('Quadrature')
        plots['constellation_tx'] = plot_to_base64(fig_constellation_tx)
        
        # Constellation of the filtered transmitted signal
        fig_constellation_tx_filtered = plt.figure(figsize=(8, 8))
        sample_points = tx_signal[params.sps - 1::params.sps][:100]
        plt.plot(sample_points.real, sample_points.imag, 'bo')
        plt.grid(True)
        plt.xlim([-1.5, 1.5])
        plt.ylim([-1.5, 1.5])
        plt.title('Constellation of Filtered Signal (Tx)')
        plt.xlabel('In-Phase')
        plt.ylabel('Quadrature')
        plots['constellation_tx_filtered'] = plot_to_base64(fig_constellation_tx_filtered)
        
        # Eye diagram at transmitter
        fig_eye_tx = plt.figure(figsize=(10, 6))
        eye_length = 2 * params.sps  # 2 symbol periods
        num_traces = 50
        
        for i in range(num_traces):
            start_idx = params.span * params.sps + i * params.sps
            if start_idx + eye_length <= len(tx_signal):
                plt.plot(np.arange(eye_length) / params.sps, tx_signal[start_idx:start_idx + eye_length].real, 'b-', alpha=0.5)
                plt.plot(np.arange(eye_length) / params.sps, tx_signal[start_idx:start_idx + eye_length].imag, 'r-', alpha=0.5)
                
        plt.grid(True)
        plt.title('Eye Diagram at Transmitter')
        plt.xlabel('Time (symbol periods)')
        plt.ylabel('Amplitude')
        plots['eye_diagram_tx'] = plot_to_base64(fig_eye_tx)
        
        # Constellation of the received signal (with noise)
        fig_constellation_rx = plt.figure(figsize=(8, 8))
        sample_points_rx = rx_signal[params.sps - 1::params.sps][:100]
        plt.plot(sample_points_rx.real, sample_points_rx.imag, 'ro', alpha=0.5)
        plt.grid(True)
        plt.xlim([-2, 2])
        plt.ylim([-2, 2])
        plt.title('Constellation of Received Signal (with noise)')
        plt.xlabel('In-Phase')
        plt.ylabel('Quadrature')
        plots['constellation_rx'] = plot_to_base64(fig_constellation_rx)
        
        # Eye diagram at receiver (before filtering)
        fig_eye_rx = plt.figure(figsize=(10, 6))
        
        for i in range(num_traces):
            start_idx = params.span * params.sps + i * params.sps
            if start_idx + eye_length <= len(rx_signal):
                plt.plot(np.arange(eye_length) / params.sps, rx_signal[start_idx:start_idx + eye_length].real, 'b-', alpha=0.5)
                plt.plot(np.arange(eye_length) / params.sps, rx_signal[start_idx:start_idx + eye_length].imag, 'r-', alpha=0.5)
                
        plt.grid(True)
        plt.title('Eye Diagram at Receiver (before filtering)')
        plt.xlabel('Time (symbol periods)')
        plt.ylabel('Amplitude')
        plots['eye_diagram_rx'] = plot_to_base64(fig_eye_rx)
        
        # Constellation after matched filtering
        fig_constellation_rx_filtered = plt.figure(figsize=(8, 8))
        plt.plot(sampled_signal[:100].real, sampled_signal[:100].imag, 'go')
        plt.grid(True)
        plt.xlim([-2, 2])
        plt.ylim([-2, 2])
        plt.title('Constellation after Matched Filtering')
        plt.xlabel('In-Phase')
        plt.ylabel('Quadrature')
        plots['constellation_rx_filtered'] = plot_to_base64(fig_constellation_rx_filtered)
        
        # Eye diagram after matched filtering
        fig_eye_rx_filtered = plt.figure(figsize=(10, 6))
        
        for i in range(num_traces):
            trace_start = delay + i * params.sps
            if trace_start + eye_length <= len(matched_filtered):
                plt.plot(np.arange(eye_length) / params.sps, matched_filtered[trace_start:trace_start + eye_length].real, 'b-', alpha=0.5)
                plt.plot(np.arange(eye_length) / params.sps, matched_filtered[trace_start:trace_start + eye_length].imag, 'r-', alpha=0.5)
        
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
                "total_bits": len(bits),
                "input_bits": bits[:params.display_length].tolist(),
                "decoded_bits": decoded_bits[:params.display_length].tolist(),
                "sampled_signal": [{"real": s.real, "imag": s.imag} for s in sampled_signal[:params.display_length]]
            }
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/advanced-simulate-qam")
async def run_qam_simulation(params: SimulationParams):
    try:
        if params.qam_order is None or params.qam_order < 4 or params.qam_order > 1024:
            raise HTTPException(status_code=400, detail="QAM order must be between 4 and 1024")
        
        # Set random seed if provided
        if params.seed is not None:
            np.random.seed(params.seed)

        # Convert Eb/N0
        EbN0_linear = 10 ** (params.EbN0_dB / 10)
        l = int(np.log2(params.qam_order))  # Number of bits per symbol
        
        # Create SRRC filter
        srrc_filter = srrc_pulse(params.alpha, params.span, params.sps)

        # Generate bits
        bits = np.random.randint(0, 2, params.K * l)
        
        # QAM modulation
        # Create constellation points
        constellation = np.array([(2*i - np.sqrt(params.qam_order) + 1) + 1j*(2*j - np.sqrt(params.qam_order) + 1) 
                                for i in range(int(np.sqrt(params.qam_order))) 
                                for j in range(int(np.sqrt(params.qam_order)))])
        
        # Normalize constellation to have unit average energy
        constellation = constellation / np.sqrt(np.mean(np.abs(constellation)**2))
        
        # Map bits to symbols
        symbols = np.zeros(params.K, dtype=complex)
        for i in range(params.K):
            bits_idx = bits[i*l:(i+1)*l]
            idx = int(''.join(map(str, bits_idx)), 2)
            symbols[i] = constellation[idx]

        # Theoretical BER for QAM
        # For M-QAM, the theoretical BER is approximately:
        # BER ≈ (4/log2(M)) * (1-1/sqrt(M)) * Q(sqrt(3*log2(M)*EbN0/(M-1)))
        M = params.qam_order
        theoretical_ber = (4 * (1 - 1/np.sqrt(M)) * erfc(np.sqrt(3 * EbN0_linear * np.log2(M) / (M - 1)))) / np.log2(M)

        # Upsampling with zeros
        symbol_upsampled = np.zeros(params.K * params.sps, dtype=complex)
        symbol_upsampled[::params.sps] = symbols

        # Filter with SRRC at transmitter
        tx_signal = signal.fftconvolve(symbol_upsampled, srrc_filter)

        # Calculate noise variance
        # For complex noise, the variance is halved for each real/imag component
        # The total noise power is N0 = 1/EbN0_linear
        N0 = 1 / EbN0_linear
        sigma = np.sqrt(N0 / 2)  # For each real/imag component

        # Add noise
        noise = sigma * (np.random.randn(len(tx_signal)) + 1j * np.random.randn(len(tx_signal)))
        rx_signal = tx_signal + noise

        # Matched filtering at receiver
        matched_filtered = signal.fftconvolve(rx_signal, srrc_filter[::-1])

        # Total delay is 2*span*sps due to two convolutions
        delay = 2 * params.span * params.sps

        # Sample at optimal instants
        sampled_signal = matched_filtered[delay + np.arange(0, params.K) * params.sps]

        # Decision for QAM
        decoded_bits = np.zeros(params.K * l, dtype=int)
        for i in range(params.K):
            # Find closest constellation point
            distances = np.abs(sampled_signal[i] - constellation)
            closest_idx = np.argmin(distances)
            # Convert index to bits
            decoded_bits[i*l:(i+1)*l] = [int(b) for b in format(closest_idx, f'0{l}b')]

        # Calculate errors
        errors = np.sum(bits != decoded_bits)
        ber = errors / len(bits)

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
        
        # QAM Constellation at Transmitter
        fig_constellation_tx = plt.figure(figsize=(8, 8))
        plt.plot(symbols[:100].real, symbols[:100].imag, 'bo')
        plt.grid(True)
        plt.xlim([-2, 2])
        plt.ylim([-2, 2])
        plt.title(f'{params.qam_order}-QAM Constellation at Transmitter')
        plt.xlabel('In-Phase')
        plt.ylabel('Quadrature')
        plots['constellation_tx'] = plot_to_base64(fig_constellation_tx)
        
        # Constellation of the filtered transmitted signal
        fig_constellation_tx_filtered = plt.figure(figsize=(8, 8))
        sample_points = tx_signal[params.sps - 1::params.sps][:100]
        plt.plot(sample_points.real, sample_points.imag, 'bo')
        plt.grid(True)
        plt.xlim([-2, 2])
        plt.ylim([-2, 2])
        plt.title(f'Constellation of Filtered {params.qam_order}-QAM Signal (Tx)')
        plt.xlabel('In-Phase')
        plt.ylabel('Quadrature')
        plots['constellation_tx_filtered'] = plot_to_base64(fig_constellation_tx_filtered)
        
        # Eye diagram at transmitter
        fig_eye_tx = plt.figure(figsize=(10, 6))
        eye_length = 2 * params.sps  # 2 symbol periods
        num_traces = 50
        
        for i in range(num_traces):
            start_idx = params.span * params.sps + i * params.sps
            if start_idx + eye_length <= len(tx_signal):
                plt.plot(np.arange(eye_length) / params.sps, tx_signal[start_idx:start_idx + eye_length].real, 'b-', alpha=0.5)
                plt.plot(np.arange(eye_length) / params.sps, tx_signal[start_idx:start_idx + eye_length].imag, 'r-', alpha=0.5)
                
        plt.grid(True)
        plt.title(f'Eye Diagram at Transmitter ({params.qam_order}-QAM)')
        plt.xlabel('Time (symbol periods)')
        plt.ylabel('Amplitude')
        plots['eye_diagram_tx'] = plot_to_base64(fig_eye_tx)
        
        # Constellation of the received signal (with noise)
        fig_constellation_rx = plt.figure(figsize=(8, 8))
        sample_points_rx = rx_signal[params.sps - 1::params.sps][:100]
        plt.plot(sample_points_rx.real, sample_points_rx.imag, 'ro', alpha=0.5)
        plt.grid(True)
        plt.xlim([-2, 2])
        plt.ylim([-2, 2])
        plt.title(f'Constellation of Received {params.qam_order}-QAM Signal (with noise)')
        plt.xlabel('In-Phase')
        plt.ylabel('Quadrature')
        plots['constellation_rx'] = plot_to_base64(fig_constellation_rx)
        
        # Eye diagram at receiver (before filtering)
        fig_eye_rx = plt.figure(figsize=(10, 6))
        
        for i in range(num_traces):
            start_idx = params.span * params.sps + i * params.sps
            if start_idx + eye_length <= len(rx_signal):
                plt.plot(np.arange(eye_length) / params.sps, rx_signal[start_idx:start_idx + eye_length].real, 'b-', alpha=0.5)
                plt.plot(np.arange(eye_length) / params.sps, rx_signal[start_idx:start_idx + eye_length].imag, 'r-', alpha=0.5)
                
        plt.grid(True)
        plt.title(f'Eye Diagram at Receiver (before filtering) ({params.qam_order}-QAM)')
        plt.xlabel('Time (symbol periods)')
        plt.ylabel('Amplitude')
        plots['eye_diagram_rx'] = plot_to_base64(fig_eye_rx)
        
        # Constellation after matched filtering
        fig_constellation_rx_filtered = plt.figure(figsize=(8, 8))
        plt.plot(sampled_signal[:100].real, sampled_signal[:100].imag, 'go')
        plt.grid(True)
        plt.xlim([-2, 2])
        plt.ylim([-2, 2])
        plt.title(f'Constellation after Matched Filtering ({params.qam_order}-QAM)')
        plt.xlabel('In-Phase')
        plt.ylabel('Quadrature')
        plots['constellation_rx_filtered'] = plot_to_base64(fig_constellation_rx_filtered)
        
        # Eye diagram after matched filtering
        fig_eye_rx_filtered = plt.figure(figsize=(10, 6))
        
        for i in range(num_traces):
            trace_start = delay + i * params.sps
            if trace_start + eye_length <= len(matched_filtered):
                plt.plot(np.arange(eye_length) / params.sps, matched_filtered[trace_start:trace_start + eye_length].real, 'b-', alpha=0.5)
                plt.plot(np.arange(eye_length) / params.sps, matched_filtered[trace_start:trace_start + eye_length].imag, 'r-', alpha=0.5)
        
        plt.axvline(x=1.0, color='r', linestyle='--', label='Sampling Point')
        plt.grid(True)
        plt.title(f'Eye Diagram after Matched Filtering ({params.qam_order}-QAM)')
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
                "total_bits": len(bits),
                "input_bits": bits[:params.display_length].tolist(),
                "decoded_bits": decoded_bits[:params.display_length].tolist(),
                "sampled_signal": [{"real": s.real, "imag": s.imag} for s in sampled_signal[:params.display_length]]
            }
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))