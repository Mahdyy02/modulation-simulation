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

class BERCurveParams(BaseModel):
    modulation_type: ModulationType
    K: int = 100000  # Match frontend default
    alpha: float = 0.22
    sps: int = 8
    span: int = 10
    seed: Optional[int] = None
    qam_order: Optional[int] = None
    ebno_range: List[float] = [0, 20]
    num_points: int = 41  # One point per 0.5 dB for better resolution

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
        if params.modulation_type == ModulationType.BPSK:
            M = 2
            Eb = 1
            Es = Eb * np.log2(M)
            N0 = 1 / EbN0_linear
            Sigma2 = N0 / 2
        elif params.modulation_type == ModulationType.QPSK:
            M = 4
            Eb = 1
            Es = Eb * np.log2(M)
            N0 = 1 / EbN0_linear
            Sigma2 = N0 / 2
        else:  # QAM
            M = params.qam_order
            Eb = 1
            Es = Eb * np.log2(M)
            N0 = 1 / EbN0_linear
            Sigma2 = N0 / 2

        # Add noise with proper scaling
        if params.modulation_type == ModulationType.BPSK:
            noise = np.sqrt(Sigma2) * np.random.randn(len(tx_signal))
        else:  # QPSK and QAM
            br = np.sqrt(Sigma2) * np.random.randn(len(tx_signal))
            bi = np.sqrt(Sigma2) * np.random.randn(len(tx_signal))
            noise = br + 1j * bi

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

@app.post("/ber-curve")
async def calculate_ber_curve(params: BERCurveParams):
    try:

        print(f"Modulation type: {params.modulation_type}")
        # print(f"True: {modulation == ModulationType.QPSK}")

        modulation = params.modulation_type
        # Augmenter significativement le nombre de bits pour une meilleure précision
        N = max(params.K, 5000000)  # Utiliser au moins 5 millions de bits
        # Create EbN0 values based on the range and number of points
        EbN0_dB = np.linspace(params.ebno_range[0], params.ebno_range[1], params.num_points)
        if params.seed is not None:
            np.random.seed(params.seed)

        if modulation == ModulationType.BPSK:
            ber_sim = []
            bits = np.random.randint(0, 2, N)
            symbols = 2*bits - 1  # BPSK mapping: 0->-1, 1->1
            
            for ebn0 in EbN0_dB:
                EbN0 = 10**(ebn0 / 10)
                # Correct noise scaling for BPSK
                noise_std = np.sqrt(1 / (2 * EbN0))
                noise = noise_std * np.random.randn(N)
                r = symbols + noise
                bits_hat = (r >= 0).astype(int)
                errors = np.sum(bits != bits_hat)
                # Utiliser une valeur minimale plus petite pour les forts Eb/N0
                min_ber = 1e-7 if ebn0 < 10 else 1e-8
                ber = max(errors / N, min_ber)
                ber_sim.append(ber)
            
            # Theoretical BER for BPSK
            EbN0_lin = 10**(EbN0_dB / 10)
            ber_theo = 0.5 * erfc(np.sqrt(EbN0_lin))
            
            fig = plt.figure(figsize=(10, 6))
            plt.semilogy(EbN0_dB, ber_sim, 'o-', label='Simulated BER (BPSK)', markersize=4)
            plt.semilogy(EbN0_dB, ber_theo, 'r--', label='Theoretical BER (0.5·erfc(√Eb/N0))')
            plt.grid(True, which='both')
            plt.xlabel('Eb/N0 [dB]')
            plt.ylabel('Bit Error Rate (BER)')
            plt.title('BPSK BER Performance')
            plt.legend()
            plt.ylim([1e-8, 1])  # Ajusté pour montrer des BER plus faibles
            ber_curve_plot = plot_to_base64(fig)
            
            return {
                "ebno_points": EbN0_dB.tolist(),
                "ber_simulated": ber_sim,
                "ber_theoretical": ber_theo.tolist(),
                "ber_curve_plot": ber_curve_plot,
                "parameters": {
                    "modulation": modulation,
                    "symbol_count": N,
                }
            }
            
        elif modulation == ModulationType.QPSK:

            N = int(1e6)  # number of bits
            EbN0_dB = np.arange(0, 21, 1)
            BER_sim = []

            for ebn0_db in EbN0_dB:
                # 1. Generate random bits
                bits = np.random.randint(0, 2, N)

                # 2. Group bits into I and Q
                bits_i = bits[0::2]
                bits_q = bits[1::2]
                symbols = (2 * bits_i - 1) + 1j * (2 * bits_q - 1)  # QPSK mapping

                # 3. Normalize to unit average power
                symbols /= np.sqrt(2)

                # 4. Add AWGN noise
                EbN0 = 10**(ebn0_db / 10)
                noise_std = np.sqrt(1 / (4 * EbN0))
                noise = noise_std * (np.random.randn(len(symbols)) + 1j * np.random.randn(len(symbols)))
                rx = symbols + noise

                # 5. Demodulate
                bits_i_hat = (np.real(rx) > 0).astype(int)
                bits_q_hat = (np.imag(rx) > 0).astype(int)
                bits_hat = np.empty_like(bits)
                bits_hat[0::2] = bits_i_hat
                bits_hat[1::2] = bits_q_hat

                # 6. Compute BER
                BER = np.mean(bits != bits_hat)
                BER_sim.append(BER)

            # Theoretical BER
            BER_theo = 0.5 * erfc(np.sqrt(10**(EbN0_dB / 10)))

            # Plot
            fig = plt.figure(figsize=(10, 6))
            plt.semilogy(EbN0_dB, BER_sim, 'o-', label='Simulated BER (QPSK)', markersize=4)
            plt.semilogy(EbN0_dB, BER_theo, 'r--', label='Theoretical BER (0.5·erfc(√Eb/N0))')
            plt.title('QPSK BER Performance')
            plt.xlabel('Eb/N0 [dB]')
            plt.ylabel('Bit Error Rate (BER)')
            plt.grid(True, which='both')
            plt.legend()
            plt.ylim([1e-5, 1])
            ber_curve_plot = plot_to_base64(fig)

            return {
                "ebno_points": EbN0_dB.tolist(),
                "ber_simulated": BER_sim,
                "ber_theoretical": BER_theo.tolist(),
                "ber_curve_plot": ber_curve_plot,
                "parameters": {
                    "modulation": modulation,
                    "symbol_count": N // 2,  # Each QPSK symbol = 2 bits
                }
            }

        elif modulation == ModulationType.QAM:

            M = params.qam_order if params.qam_order is not None else 16
            k = int(np.log2(M))

            max_bits = int(1e5)  # cap bits to speed up simulation
            if N > max_bits:
                N_adj = (max_bits // k) * k
            else:
                N_adj = (N // k) * k

            bits = np.random.randint(0, 2, N_adj)
            num_symbols = N_adj // k
            
            # Regroupement bits en symboles
            bits_grouped = bits.reshape((num_symbols, k))
            
            # Générer constellation QAM carré
            m = int(np.sqrt(M))
            if m*m != M:
                raise ValueError(f"L'ordre QAM {M} n'est pas un carré parfait")
            
            # Création de la constellation Gray-coded
            constellation = np.zeros((M, 2))
            for i in range(m):
                for j in range(m):
                    gray_i = i ^ (i >> 1)
                    gray_j = j ^ (j >> 1)
                    I = 2*i - (m - 1)
                    Q = 2*j - (m - 1)
                    constellation[gray_i * m + gray_j] = [I, Q]
            
            # Normaliser la puissance moyenne à 1
            avg_power = np.mean(np.sum(constellation**2, axis=1))
            constellation /= np.sqrt(avg_power)
            
            # Map bits vers symboles complexes
            symbols = np.zeros(num_symbols, dtype=complex)
            for idx in range(M):
                bit_pattern = np.array([int(b) for b in format(idx, f'0{k}b')])
                mask = np.all(bits_grouped == bit_pattern, axis=1)
                symbols[mask] = constellation[idx, 0] + 1j * constellation[idx, 1]
            
            ber_sim = []
            for ebn0 in EbN0_dB:
                EbN0 = 10**(ebn0 / 10)
                EsN0 = EbN0 * k
                
                noise_std = np.sqrt(1 / (2 * EsN0))
                noise = noise_std * (np.random.randn(num_symbols) + 1j * np.random.randn(num_symbols))
                
                r = symbols + noise
                
                # Démodulation vectorisée : calcul distances entre r et constellation
                r_vec = np.vstack([np.real(r), np.imag(r)]).T[:, np.newaxis, :]  # shape (num_symbols,1,2)
                const_vec = constellation[np.newaxis, :, :]                     # shape (1,M,2)
                dist = np.sum((r_vec - const_vec)**2, axis=2)                   # shape (num_symbols, M)
                
                idx_hat = np.argmin(dist, axis=1)
                
                # Convertir indices estimés en bits
                bits_hat = np.array([list(map(int, format(i, f'0{k}b'))) for i in idx_hat])
                
                errors = np.sum(bits_grouped != bits_hat)
                ber = errors / N_adj
                ber_sim.append(ber)
            
            # BER théorique pour M-QAM carré
            EbN0_lin = 10**(EbN0_dB / 10)
            if M == 4:
                ber_theo = 0.5 * erfc(np.sqrt(EbN0_lin))
            else:
                ber_theo = (4/k) * (1 - 1/np.sqrt(M)) * 0.5 * erfc(np.sqrt((3*k*EbN0_lin)/(2*(M-1))))
            
            fig = plt.figure(figsize=(10, 6))
            plt.semilogy(EbN0_dB, ber_sim, 'o-', label=f'Simulated BER ({M}-QAM)', markersize=4)
            plt.semilogy(EbN0_dB, ber_theo, 'r--', label=f'Theoretical BER ({M}-QAM)')
            plt.grid(True, which='both')
            plt.xlabel('Eb/N0 [dB]')
            plt.ylabel('Bit Error Rate (BER)')
            plt.title(f'{M}-QAM BER Performance')
            plt.legend()
            plt.ylim([1e-8, 1])
            
            ber_curve_plot = plot_to_base64(fig)
            
            return {
                "ebno_points": EbN0_dB.tolist(),
                "ber_simulated": ber_sim,
                "ber_theoretical": ber_theo.tolist(),
                "ber_curve_plot": ber_curve_plot,
                "parameters": {
                    "modulation": modulation,
                    "qam_order": M,
                    "symbol_count": num_symbols,
                }
            }

        else:
            raise HTTPException(status_code=400, detail="Unknown modulation type")
    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}\n{error_details}")