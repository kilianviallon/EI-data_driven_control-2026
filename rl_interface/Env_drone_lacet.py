import gymnasium as gym
from gymnasium import spaces
import numpy as np
from modele_rnn import ModeleRNN
from modele_lacet import ModeleLacet

TS = 0.05        # pas de temps imposé [s]
N_STEPS = 200    # 10 s de vol
SOL = -0.25      # altitude du sol [m]
CRASH = -50.0    # récompense du crash, qui arrête le vol
DELTA_MAX = 0.05     # commande de lacet max : le lacet est très sensible (un tour en 2.5 s demande δ ≈ 0.04)
PSI_CRASH = 1.0      # erreur de lacet qui arrête le vol [rad] (~57°)
T_ROT = 4.0          # début du tour [s] : le drone est stabilisé à 0.5 m
D_ROT = 2.5          # durée du tour [s] : fini à 6.5 s, avant la descente à 7 s


# écart entre deux angles, ramené dans [-π, π]
def ecart_angle(e):
    return np.arctan2(np.sin(e), np.cos(e))


class DroneLacetEnv(gym.Env):

    def __init__(self):
        super().__init__()
        # Action : [commande moteur u, commande de lacet a], entre -1 et 1 ; δ = DELTA_MAX · a
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
        # Observation : les 4 valeurs de l'altitude (comme Env_drone) + 3 valeurs du lacet
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(7,), dtype=np.float32)
        # Les modèles du drone : le RNN du groupe (altitude) et le modèle de lacet identifié
        self.modele = ModeleRNN()
        self.lacet = ModeleLacet()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.modele.reset()          # mémoire du RNN vide : moteurs au repos
        self.lacet.reset()           # moteurs au repos, drone immobile en lacet
        self.k = 0                   # numéro du pas
        self.t = 0.0
        self.z = 0.0
        self.vz = 0.0
        self.az_prev = 0.0
        self.u_prev = 0.0
        self.psi = 0.0
        self.wz = 0.0
        self.a_prev = 0.0
        m = self._mesures()
        obs = self._observation(m)
        return obs, {"m": m}

# Définir l'objectif de la mission
    # renvoie la consigne d'altitude, la consigne de lacet et la vitesse de lacet voulue
    def mission(self, t):
        # altitude : les 3 paliers de la figure 1
        if t < 3.0:
            z_ref = 0.0
        elif t < 7.0:
            z_ref = 0.5
        else:
            z_ref = -0.25
        # lacet : un tour sur soi-même pendant le palier à 0.5 m, sans à-coup (vitesse nulle au début et à la fin)
        # pas un échelon de 2π : une fois ramené dans [-π, π], il serait invisible
        x = np.clip((t - T_ROT) / D_ROT, 0.0, 1.0)                         # avancement du tour, de 0 à 1
        psi_ref = 2 * np.pi * x - np.sin(2 * np.pi * x)
        wz_ref = 2 * np.pi / D_ROT * (1 - np.cos(2 * np.pi * x))            # au plus 5 rad/s
        return z_ref, psi_ref, wz_ref

# définir mesures et observation
    # ce que les capteurs mesurent
    def _mesures(self):
        z_ref, psi_ref, wz_ref = self.mission(self.t)
        return {"t": self.t, "z": self.z, "vz": self.vz,
            "az_prev": self.az_prev, "z_ref": z_ref, "u_prev": self.u_prev,
            "psi": self.psi, "wz": self.wz, "psi_ref": psi_ref, "wz_ref": wz_ref, "a_prev": self.a_prev}
    # ce que l'agent voit (observation)
    def _observation(self, m):
        erreur = m["z_ref"] - m["z"]
        erreur_psi = ecart_angle(m["psi_ref"] - m["psi"])      # ramenée dans [-π, π]
        erreur_wz = m["wz_ref"] - m["wz"]                       # sans amortissement : il faut voir wz pour freiner
        return np.array([erreur, m["vz"], m["az_prev"] / 10, m["u_prev"],
                         erreur_psi, erreur_wz / 5, m["a_prev"]], dtype=np.float32)    # / 10, / 5 : valeurs de l'ordre de 1

# A CHAQUE PAS
    def step(self, action):
        # 1. Les commandes de l'IA
        u = float(np.clip(action[0], -1.0, 1.0))   # commande moteur, bornée à [-1, 1]
        a = float(np.clip(action[1], -1.0, 1.0))   # commande de lacet, bornée à [-1, 1]
        delta = DELTA_MAX * a                      # commande différentielle envoyée aux moteurs
        z_ref, _, _ = self.mission(self.t)               # consigne d'altitude pendant ce pas
        _, psi_ref, _ = self.mission(self.t + TS)        # consigne de lacet à la fin du pas : elle bouge pendant le tour

        # 2. Physique : le RNN donne az, puis on intègre ; le modèle de lacet donne psi et wz
        # (le couplage avec l'altitude, -8.5·δ² m/s², est négligé : voir lacet.ipynb partie 6)
        az = self.modele.step(u)
        self.z  = self.z + self.vz * TS + 0.5 * az * TS**2
        self.vz = self.vz + az * TS
        self.lacet.step(u, delta)
        self.psi = self.lacet.psi
        self.wz = self.lacet.wz

        # 3. Récompense
        reward = self._recompense(z_ref, u, psi_ref, a)

        # 4. Mise à jour pour le pas suivant
        self.az_prev = az          # devient "l'accélération du pas précédent"
        self.u_prev = u            # devient "la commande du pas précédent"
        self.a_prev = a            # devient "la commande de lacet du pas précédent"
        self.k += 1
        self.t = self.k * TS

        # 5. Fin du vol ?
        crash = (abs(z_ref - self.z) > 1.0 or self.z < SOL - 0.1         # trop loin de la consigne ou dans le sol
                 or abs(ecart_angle(psi_ref - self.psi)) > PSI_CRASH)    # ou trop loin de la consigne de lacet
        if crash:
            reward = CRASH
        terminated = crash                 # le crash arrête le vol
        truncated = self.k >= N_STEPS      # les 10 s sont écoulées

        # 6. Ce qu'on renvoie
        m = self._mesures()
        obs = self._observation(m)
        return obs, float(reward), terminated, truncated, {"m": m}


    # Récompense d'un pas (z, vz, psi : état après le pas ; u_prev, a_prev : commandes du pas précédent)
    # Positive quand le drone suit : rester en vol rapporte, donc le crash (fin du vol) est évité.
    # Les 5 premiers termes sont ceux de Env_drone ; les 3 derniers, ceux du lacet.
    def _recompense(self, z_ref, u, psi_ref, a):
        erreur = z_ref - self.z
        erreur_psi = ecart_angle(psi_ref - self.psi)
        return (1.0 - abs(erreur)                         # être sur la consigne
                + 0.3 * np.exp(-(erreur / 0.03) ** 2)     # bonus de précision (à 3 cm près)
                - 0.05 * abs(self.vz)                     # ne pas foncer
                - 0.3 * (u - self.u_prev) ** 2            # commande sans à-coups
                - 10.0 * max(0.0, SOL - self.z)           # ne pas passer sous le sol
                - 0.5 * abs(erreur_psi)                   # être sur la consigne de lacet
                + 0.2 * np.exp(-(erreur_psi / 0.05) ** 2) # bonus de précision (à 3° près)
                - 0.3 * (a - self.a_prev) ** 2)           # commande de lacet sans à-coups
