import numpy as np
import matplotlib.pyplot as plt

# Parameters
alpha = [0.22, 0.5, 0.8, 1]  # Roll-off factors
T = 1                         # Symbol period
Fs = 8                        # Sampling frequency (samples per T)
t = np.linspace(-5*T, 5*T, int(10 * T * Fs) + 1, endpoint=True)  # Time axis (-5T to 5T)

# Raised cosine impulse response
def raised_cosine(t, alpha, T):
    t += 1e-9  # Avoid division by zero (t=0)
    return np.sinc(t / T) * np.cos(np.pi * alpha * t / T) / (1 - (2 * alpha * t / T)**2)

# Create a single figure
plt.figure(figsize=(10, 6))

# Plot all alpha curves
for a in alpha:
    g = raised_cosine(t, a, T)
    g /= np.sqrt(np.sum(g**2))  # Normalize energy
    plt.plot(t, g, linewidth=2, label=f'α = {a}')

# Formatting
plt.title('Raised Cosine Filter - Impulse Response (Different Roll-off Factors)')
plt.xlabel('Time (t/T)')
plt.ylabel('Amplitude')
plt.grid(True, linestyle='--')
plt.axhline(0, color='black', linewidth=0.5)
plt.axvline(0, color='black', linewidth=0.5)
plt.legend()
plt.show()