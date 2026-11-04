import numpy as np

# Modèle du lacet identifié sur le simulateur (voir identification_lacet/lacet.ipynb) :
#   moteurs u_i = u + s_i·δ bornés à [-1, 1], filtrés par 1/(0.2 s + 1)  ->  m_i
#   w_dot_z = 50 · Σ s_i·tanh(m_i)          (exact sur 100 vols, sans amortissement)
TS = 0.05                                   # pas de temps imposé [s]
K_LACET = 50.0                              # gain identifié [rad/s²]
TAU = 0.2                                   # constante de temps des moteurs [s]
S = np.array([1.0, -1.0, 1.0, -1.0])        # moteurs 1 et 3 à +δ, 2 et 4 à -δ : δ > 0 -> wz > 0
N_SOUS_PAS = 10                             # le filtre moteur bouge pendant Ts : on intègre plus finement


# créer une class pour récupérer le lacet

class ModeleLacet:

    def __init__(self):
        self.reset()

    # moteurs au repos, drone immobile
    def reset(self):
        self.m = np.zeros(4)           # commandes moteur filtrées
        self.wz = 0.0                  # vitesse de lacet [rad/s]
        self.psi = 0.0                 # angle de lacet [rad], pas ramené dans [-π, π] (comme le simulateur)

    # avancer d'un pas : u et δ tenus pendant Ts, renvoie w_dot_z à la fin du pas
    def step(self, u, delta):
        moteurs = np.clip(u + delta * S, -1.0, 1.0)     # le mélangeur
        h = TS / N_SOUS_PAS
        a = np.exp(-h / TAU)                            # filtre moteur discrétisé exactement
        for _ in range(N_SOUS_PAS):
            wdz = K_LACET * np.sum(S * np.tanh(self.m))
            self.m = a * self.m + (1 - a) * moteurs
            wdz_fin = K_LACET * np.sum(S * np.tanh(self.m))
            wz_fin = self.wz + 0.5 * (wdz + wdz_fin) * h          # trapèzes
            self.psi += 0.5 * (self.wz + wz_fin) * h
            self.wz = wz_fin
        return wdz_fin
