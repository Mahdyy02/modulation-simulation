import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, fftshift

# Parameters
alpha = [0.1, 0.22, 0.5, 1]  # Roll-off factors
T = 1                         # Symbol period
Fs = 10                       # Sampling frequency (samples per T)
t = np.linspace(-5*T, 5*T, int(10 * T * Fs) + 1, endpoint=True)  # Time axis (-5T to 5T)

# Raised cosine impulse response
def raised_cosine(t, alpha, T):
    t += 1e-9  # Avoid division by zero (t=0)
    return np.sinc(t / T) * np.cos(np.pi * alpha * t / T) / (1 - (2 * alpha * t / T)**2)

# Create figure with subplots
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

# Plot time-domain responses
for a in alpha:
    g = raised_cosine(t, a, T)
    g /= np.sqrt(np.sum(g**2))  # Normalize energy
    ax1.plot(t, g, linewidth=2, label=f'α = {a}')

ax1.set_title('Raised Cosine Filter - Impulse Response')
ax1.set_xlabel('Time (t/T)')
ax1.set_ylabel('Amplitude')
ax1.grid(True, linestyle='--')
ax1.axhline(0, color='black', linewidth=0.5)
ax1.axvline(0, color='black', linewidth=0.5)
ax1.legend()

# Plot frequency-domain PSD
n_fft = 1024  # FFT length
for a in alpha:
    g = raised_cosine(t, a, T)
    g /= np.sqrt(np.sum(g**2))  # Normalize energy
    
    # Compute PSD
    G = fft(g, n_fft)
    psd = np.abs(fftshift(G))**2
    psd = 10 * np.log10(psd / np.max(psd))  # Normalized dB scale
    
    # Frequency axis
    freq = np.fft.fftshift(np.fft.fftfreq(n_fft, d=1/Fs))
    
    ax2.plot(freq, psd, label=f'α = {a}')

ax2.set_title('Power Spectral Density (PSD)')
ax2.set_xlabel('Frequency (1/T)')
ax2.set_ylabel('Magnitude (dB)')
ax2.grid(True, linestyle='--')
ax2.legend()
ax2.set_xlim(-2, 2)  # Focus on the main lobe
ax2.set_ylim(-60, 5)  # Typical dB range for PSD

plt.tight_layout()
plt.show()