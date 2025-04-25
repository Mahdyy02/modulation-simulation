import numpy as np
from scipy.special import erfc  # Import erfc function

# Paramètres
K = 1000       # Taille augmentée pour observer des erreurs
EbN0_dB = 8     # Rapport Eb/N0 réduit pour plus de bruit
Es = 1
Eb = 1

# Conversion Eb/N0
EbN0_linear = 10 ** (EbN0_dB / 10)
sigma2 = 1/(2 * EbN0_linear)  # Es=1, donc sigma² = 1/(2*EbN0)

Nerr=0
Nb=0
Nerr_MAX = 500


while Nerr < Nerr_MAX:

    # Génération des bits
    d = np.random.randint(0, 2, K)

    # Modulation BPSK
    SBPSK = 2 * d - 1

    # Bruit gaussien
    b = np.random.normal(0, np.sqrt(sigma2), K)

    # Signal reçu
    Sr = SBPSK + b

    # Détection
    dr = np.where(Sr > 0, 1, 0)

    # Calcul des erreurs
    Nerr+= np.sum(np.abs(d - dr))
    Nb+=K


taux_erreur=(Nerr/Nb)*100
print(f"Taux d'erruer: {taux_erreur:.4f}%")
print(f"Taux d'erreur theorique (BPSK): {100 * 0.5 * erfc(np.sqrt(EbN0_linear)):.4f}%")
