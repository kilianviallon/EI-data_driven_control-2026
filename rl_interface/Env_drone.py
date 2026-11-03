import gymnasium as gym
from gymnasium import spaces
import numpy as np
from modele_rnn import ModeleRNN

TS = 0.05        # pas de temps imposé [s]
N_STEPS = 200    # 10 s de vol
SOL = -0.25      # altitude du sol [m]
CRASH = -50.0    # récompense du crash, qui arrête le vol


class DroneEnv(gym.Env):

    def __init__(self):
        super().__init__()
        # Action : une commande moteur entre -1 et 1
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)
        # Observation 
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(4,), dtype=np.float32)
        # Le modèle du drone : le RNN du groupe
        self.modele = ModeleRNN()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.modele.reset()          # mémoire du RNN vide : moteurs au repos
        self.k = 0                   # numéro du pas
        self.t = 0.0
        self.z = 0.0
        self.vz = 0.0
        self.az_prev = 0.0
        self.u_prev = 0.0
        m = self._mesures()
        obs = self._observation(m)
        return obs, {"m": m}

# Définir l'objectif de la mission 
    def mission(self, t):
        if t < 3.0:
            return 0.0
        elif t < 7.0:
            return 0.5
        else:
            return -0.25

# définir mesures et observation
    # ce que les capteurs mesurent 
    def _mesures(self):
        return {"t": self.t, "z": self.z, "vz": self.vz,
            "az_prev": self.az_prev, "z_ref": self.mission(self.t), "u_prev": self.u_prev}
    # ce que l'agent voit (observation)
    def _observation(self, m):
        erreur = m["z_ref"] - m["z"]
        return np.array([erreur, m["vz"], m["az_prev"] / 10, m["u_prev"]], dtype=np.float32)    # az_prev / 10 : pour que toutes les valeurs soient de l'ordre de 1
   
# A CHAQUE PAS 
    def step(self, action):
        # 1. La commande de l'IA
        u = float(np.clip(action[0], -1.0, 1.0))   # tableau [0.38] -> nombre 0.38, borné à [-1, 1]
        z_ref = self.mission(self.t)               # consigne pendant ce pas

        # 2. Physique : le RNN donne az, puis on intègre
        az = self.modele.step(u)
        self.z  = self.z + self.vz * TS + 0.5 * az * TS**2
        self.vz = self.vz + az * TS

        # 3. Récompense
        reward = self._recompense(z_ref, u)

        # 4. Mise à jour pour le pas suivant
        self.az_prev = az          # devient "l'accélération du pas précédent"
        self.u_prev = u            # devient "la commande du pas précédent"
        self.k += 1
        self.t = self.k * TS

        # 5. Fin du vol ?
        crash = abs(z_ref - self.z) > 1.0 or self.z < SOL - 0.1   # trop loin de la consigne ou dans le sol
        if crash:
            reward = CRASH
        terminated = crash                 # le crash arrête le vol
        truncated = self.k >= N_STEPS      # les 10 s sont écoulées

        # 6. Ce qu'on renvoie
        m = self._mesures()
        obs = self._observation(m)
        return obs, float(reward), terminated, truncated, {"m": m}


    # Récompense d'un pas (z, vz : état après le pas ; u_prev : commande du pas précédent)
    # Positive quand le drone suit : rester en vol rapporte, donc le crash (fin du vol) est évité.
    def _recompense(self, z_ref, u):
        erreur = z_ref - self.z
        return (1.0 - abs(erreur)                         # être sur la consigne
                + 0.3 * np.exp(-(erreur / 0.03) ** 2)     # bonus de précision (à 3 cm près)
                - 0.05 * abs(self.vz)                     # ne pas foncer
                - 0.3 * (u - self.u_prev) ** 2            # commande sans à-coups
                - 10.0 * max(0.0, SOL - self.z))          # ne pas passer sous le sol
