import numpy as np
from scipy.special import erfc
from scipy import signal
import matplotlib.pyplot as plt

# Paramètres
K = 100000                # Nombre de symboles
EbN0_dB = 4               # Rapport Eb/N0 en dB
alpha = 0.22              # Facteur de roll-off du filtre
sps = 8                   # Échantillons par symbole (samples per symbol)
span = 10                 # Nombre de symboles couverts par le filtre
Eb = 1                    # Énergie par bit

# Fonction pour créer le filtre en racine de cosinus surélevé
def srrc_pulse(alpha, span, sps):
    """
    Génère un filtre en racine de cosinus surélevé (SRRC).
    
    Args:
        alpha: Facteur de roll-off (0 <= alpha <= 1)
        span: Demi-longueur du filtre en nombre de symboles
        sps: Nombre d'échantillons par symbole
        
    Returns:
        Un filtre SRRC normalisé
    """
    n = np.arange(-span * sps, span * sps + 1)
    t = n / sps
    
    # Éviter la division par zéro
    pulse = np.zeros_like(t, dtype=float)
    
    # t = 0
    pulse[t == 0] = 1.0 - alpha + (4 * alpha / np.pi)
    
    # t = ±Ts/(4*alpha)
    idx1 = np.abs(np.abs(t) - 1.0/(4.0*alpha)) < 1e-10
    if np.any(idx1):
        pulse[idx1] = (alpha/np.sqrt(2)) * ((1.0 + 2.0/np.pi) * np.sin(np.pi/(4.0*alpha)) + 
                                        (1.0 - 2.0/np.pi) * np.cos(np.pi/(4.0*alpha)))
    
    # Autres valeurs de t
    idx2 = (t != 0) & (np.abs(np.abs(t) - 1.0/(4.0*alpha)) >= 1e-10)
    numer = np.sin(np.pi * t[idx2] * (1.0 - alpha)) + 4.0 * alpha * t[idx2] * np.cos(np.pi * t[idx2] * (1.0 + alpha))
    denom = np.pi * t[idx2] * (1.0 - (4.0 * alpha * t[idx2])**2)
    pulse[idx2] = numer / denom
    
    # Normaliser pour une énergie unitaire
    return pulse / np.sqrt(np.sum(pulse**2))

# Conversion Eb/N0
EbN0_linear = 10 ** (EbN0_dB / 10)
sigma = np.sqrt(1/(2 * EbN0_linear))  # Écart-type du bruit

# Créer le filtre SRRC
srrc_filter = srrc_pulse(alpha, span, sps)

# Visualiser le filtre
plt.figure(figsize=(10, 6))
time_axis = np.arange(-span, span + 1/sps, 1/sps)
plt.plot(time_axis, srrc_filter)
plt.grid(True)
plt.xlabel('Temps (en périodes symbole)')
plt.ylabel('Amplitude')
plt.title(f'Filtre SRRC (α = {alpha})')
plt.axhline(y=0, color='r', linestyle='-', alpha=0.3)
plt.axvline(x=0, color='r', linestyle='-', alpha=0.3)
plt.savefig('srrc_filter.png')
plt.show()

# Simulation de Monte Carlo
nb_errors = 0
nb_bits = 0
max_errors = 500
nb_runs = 0

while nb_errors < max_errors:
    nb_runs += 1
    
    # Génération des bits
    bits = np.random.randint(0, 2, K)
    
    # Modulation BPSK
    symbols = 2*bits - 1
    
    # Suréchantillonnage avec zéros
    symbol_upsampled = np.zeros(K * sps)
    symbol_upsampled[::sps] = symbols
    
    # Filtrage par le SRRC à l'émission
    tx_signal = signal.fftconvolve(symbol_upsampled, srrc_filter)
    
    # Ajout du bruit
    noise = sigma * np.random.randn(len(tx_signal))
    rx_signal = tx_signal + noise
    
    # Filtrage adapté à la réception
    matched_filtered = signal.fftconvolve(rx_signal, srrc_filter[::-1])
    
    # Décalage pour l'échantillonnage correct
    # Le délai total est 2*span*sps car nous avons deux convolutions
    delay = 2*span*sps
    
    # Échantillonnage aux instants optimaux
    sampled_signal = matched_filtered[delay + np.arange(0, K) * sps]
    
    # Décision
    decoded_bits = (sampled_signal > 0).astype(int)
    
    # Calcul des erreurs
    errors = np.sum(bits != decoded_bits)
    nb_errors += errors
    nb_bits += K
    
    # Visualisation pour la première exécution seulement
    if nb_runs == 1:
        plt.figure(figsize=(15, 10))
        
        # Affichage des 20 premiers bits
        display_length = 20
        
        # Bits d'entrée
        plt.subplot(511)
        plt.stem(np.arange(display_length), bits[:display_length], basefmt='')
        plt.title('Bits d\'entrée')
        plt.grid(True)
        plt.xticks(np.arange(0, display_length, 1))
        
        # Symboles suréchantillonnés
        plt.subplot(512)
        plt.step(np.arange(display_length * sps) / sps, 
                symbol_upsampled[:display_length * sps])
        plt.title('Symboles suréchantillonnés')
        plt.grid(True)
        plt.xlim(0, display_length)
        
        # Signal après filtrage SRRC (émission)
        plt.subplot(513)
        signal_length = display_length * sps + 2 * span * sps
        plt.plot(np.arange(signal_length) / sps, tx_signal[:signal_length])
        plt.title('Signal après filtrage SRRC (émission)')
        plt.grid(True)
        plt.xlim(span, span + display_length)
        
        # Signal après filtrage adapté (réception)
        plt.subplot(514)
        matched_length = signal_length + 2 * span * sps
        plt.plot(np.arange(matched_length) / sps, matched_filtered[:matched_length])
        plt.title('Signal après filtrage adapté (réception)')
        plt.grid(True)
        plt.xlim(2 * span, 2 * span + display_length)
        
        # Points d'échantillonnage
        plt.subplot(515)
        plt.stem(np.arange(display_length), sampled_signal[:display_length], basefmt='')
        plt.title('Signal échantillonné aux instants nT')
        plt.grid(True)
        plt.xticks(np.arange(0, display_length, 1))
        
        plt.tight_layout()
        plt.savefig('bpsk_srrc_simulation.png')
        plt.show()
        
        # Diagramme de l'œil
        plt.figure(figsize=(10, 6))
        eye_length = 2 * sps  # 2 périodes symbole
        num_traces = 20
        start_idx = delay - sps // 2  # Commencer un demi-symbole avant le premier échantillon
        
        for i in range(num_traces):
            trace_start = start_idx + i * sps
            plt.plot(np.arange(eye_length) / sps, matched_filtered[trace_start:trace_start + eye_length])
            
        plt.axvline(x=0.5, color='r', linestyle='--', label='Point d\'échantillonnage')
        plt.axvline(x=1.5, color='r', linestyle='--')
        plt.grid(True)
        plt.title('Diagramme de l\'œil')
        plt.xlabel('Temps (en périodes symbole)')
        plt.ylabel('Amplitude')
        plt.legend()
        plt.savefig('eye_diagram.png')
        plt.show()

# Calcul du taux d'erreur
ber = nb_errors / nb_bits
theoretical_ber = 0.5 * erfc(np.sqrt(EbN0_linear))

print(f"Taux d'erreur mesuré: {ber * 100:.4f}%")
print(f"Taux d'erreur théorique (BPSK): {theoretical_ber * 100:.4f}%")