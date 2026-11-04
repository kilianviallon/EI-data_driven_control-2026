import os

import gymnasium as gym
from gymnasium import spaces
import numpy as np

TS = 0.05        # pas de temps imposé [s]
N_STEPS = 200    # 10 s de vol
ICI = os.path.dirname(os.path.abspath(__file__))
DOSSIER_ATHETA = os.path.join(ICI, "../identification_RNN_x_theta/RNN_a_theta_16neurones/lightning_logs/version_2/export_NN")
DOSSIER_AX = os.path.join(ICI, "../identification_RNN_x_theta/RNN_ax_32neurones/lightning_logs/version_1/export_NN")
U_MAX = 0.02     # commande de tangage autorisée : la plage de la base d'apprentissage des RNN (±0,021)
PENTE_AX = 9.45  # ax = 9,45.theta, mesuré sur la base (poussée d'équilibre x sin(theta), proche de g.theta)
THETA_MAX = 0.6  # au-delà, le drone sort du domaine appris par les RNN (theta de la base : ±0,47 rad)
Z_VOL = 0.5      # altitude du drone, supposée tenue par le contrôleur de z (pour l'affichage seulement)

# Récupérer les paramètres du RNN
def lire_txt(chemin):
    with open(chemin) as f:
        texte = f.read().strip().strip("[]")      # enlève les crochets
    lignes = texte.split(";")                      # coupe aux ";"
    return np.array([[float(x) for x in ligne.split()] for ligne in lignes])

# créer une class RNN pour récupérer une accélération (a_theta ou ax)

class ModeleRNN:
    # implémenter les paramètres du RNN (n'importe quel nombre de neurones, normalisation lue dans le dossier)
    def __init__(self, dossier):
        self.Wih = lire_txt(dossier + "/weight_ih_l0.txt").reshape(-1)   # n nombres
        self.Whh = lire_txt(dossier + "/weight_hh_l0.txt")               # matrice n x n
        self.bh  = lire_txt(dossier + "/bias_h.txt").reshape(-1)         # n nombres
        self.Wo  = lire_txt(dossier + "/head_weight.txt").reshape(-1)    # n nombres
        self.bo  = lire_txt(dossier + "/head_bias.txt")[0, 0]            # 1 nombre
        self.u_scale, self.y_offset, self.y_scale = lire_txt(dossier + "/normalization.txt").reshape(-1)
        self.h = np.zeros(len(self.bh))

    # reset la mémoire de l'élément caché
    def reset(self):
        self.h = np.zeros(len(self.bh))     # on vide la mémoire au début de chaque vol
    # recalculer l'accélération grâce au RNN
    def step(self, u):
        self.h = np.tanh(self.Wih * (u / self.u_scale) + self.Whh @ self.h + self.bh)    # équation 1
        return self.y_scale * (self.Wo @ self.h + self.bo) + self.y_offset


class DroneEnvX(gym.Env):
    """
    Le drone suit la voiture sur x (2D : x horizontal, z vertical tenu à Z_VOL par le contrôleur de z).
    Commande : u_theta (tangage). Le RNN donne a_theta ; theta s'intègre ; ax = 9,45.theta (ou le RNN de ax) ;
    x s'intègre.
    modele_ax : 'theta' (ax = 9,45.theta, cohérent avec le theta de l'environnement) ou 'rnn' (RNN ax, 32 neurones)
    voiture_aleatoire : True pour l'entraînement (profils de vitesse tirés au hasard), False pour l'évaluation
    """

    def __init__(self, modele_ax='theta', voiture_aleatoire=False):
        super().__init__()
        # Action : une commande entre -1 et 1, convertie en tangage u_theta = U_MAX.action
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)
        # Observation : [erreur x, vitesse relative, theta, w, u_prev], chacune de l'ordre de 1
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(5,), dtype=np.float32)
        # Les modèles du drone : les RNN du groupe
        self.modele = ModeleRNN(DOSSIER_ATHETA)
        self.modele_ax = modele_ax
        self.rnn_ax = ModeleRNN(DOSSIER_AX) if modele_ax == 'rnn' else None
        self.voiture_aleatoire = voiture_aleatoire

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.modele.reset()          # mémoire du RNN vide : moteurs au repos
        if self.rnn_ax is not None:
            self.rnn_ax.reset()
        self.k = 0                   # numéro du pas
        self.t = 0.0
        self.x = 0.0                 # le drone décolle de la plateforme de la voiture
        self.vx = 0.0
        self.theta = 0.0
        self.w = 0.0                 # vitesse de tangage (gyromètre wy)
        self.atheta_prev = 0.0
        self.u_prev = 0.0
        self.x_car = 0.0
        self.v_car = 0.0
        self._tirer_voiture()
        m = self._mesures()
        obs = self._observation(m)
        return obs, {"m": m}

# Définir l'objectif de la mission : la voiture roule, le drone la suit
    def _tirer_voiture(self):
        # profil de vitesse de la voiture : liste de (instant, vitesse visée) et accélération maximale
        if self.voiture_aleatoire:
            t1 = self.np_random.uniform(0.5, 3.0)
            t2 = t1 + self.np_random.uniform(2.0, 5.0)
            self.profil = [(t1, self.np_random.uniform(0.5, 2.0)), (t2, self.np_random.uniform(0.0, 2.0))]
            self.a_car = self.np_random.uniform(0.5, 1.5)
        else:                        # mission d'évaluation : démarre à 1 s, roule à 1,5 m/s, ralentit à 0,5 m/s à 6 s
            self.profil = [(1.0, 1.5), (6.0, 0.5)]
            self.a_car = 1.0

    def mission(self, t):
        # vitesse visée par la voiture à l'instant t
        v = 0.0
        for t_palier, vitesse in self.profil:
            if t >= t_palier:
                v = vitesse
        return v

# définir mesures et observation
    # ce que les capteurs mesurent (theta : intégration du gyromètre, le capteur de theta est inutilisable)
    def _mesures(self):
        return {"t": self.t, "x": self.x, "vx": self.vx, "theta": self.theta, "w": self.w,
                "atheta_prev": self.atheta_prev, "x_car": self.x_car, "v_car": self.v_car, "u_prev": self.u_prev}
    # ce que l'agent voit (observation)
    def _observation(self, m):
        erreur = m["x_car"] - m["x"]
        v_rel = m["v_car"] - m["vx"]
        return np.array([erreur, v_rel, m["theta"] / 0.3, m["w"], m["u_prev"] / U_MAX], dtype=np.float32)    # divisions : valeurs de l'ordre de 1

# A CHAQUE PAS
    def step(self, action):
        # 1. La commande de l'IA
        u = U_MAX * float(np.clip(action[0], -1.0, 1.0))   # tableau [0.2] -> nombre 0.2, borné à [-1, 1], puis u_theta = 0.004

        # 2. Physique : le RNN donne a_theta, on intègre theta ; ax vient de theta (ou du RNN de ax) ; on intègre x
        atheta = self.modele.step(u)
        self.theta = self.theta + self.w * TS + 0.5 * atheta * TS**2
        self.w = self.w + atheta * TS
        ax = PENTE_AX * self.theta if self.rnn_ax is None else self.rnn_ax.step(u)
        self.x = self.x + self.vx * TS + 0.5 * ax * TS**2
        self.vx = self.vx + ax * TS
        # la voiture : accélère ou freine vers sa vitesse visée
        v_vise = self.mission(self.t)
        self.v_car = self.v_car + float(np.clip(v_vise - self.v_car, -self.a_car * TS, self.a_car * TS))
        self.x_car = self.x_car + self.v_car * TS

        # 3. Récompense
        erreur = self.x_car - self.x
        v_rel = self.v_car - self.vx
        reward = (-abs(erreur)                              # suivre la voiture
                  - 0.1 * v_rel**2                          # rouler à sa vitesse (amortit les oscillations)
                  - 1.0 * self.theta**2                     # rester peu incliné
                  - 0.5 * ((u - self.u_prev) / U_MAX)**2    # commande lisse
                  + 0.5 * (abs(erreur) < 0.1))              # bonus de précision : à moins de 10 cm

        # 4. Mise à jour pour le pas suivant
        self.atheta_prev = atheta  # devient "l'accélération angulaire du pas précédent"
        self.u_prev = u            # devient "la commande du pas précédent"
        self.k += 1
        self.t = self.k * TS

        # 5. Fin du vol ?
        terminated = bool(abs(self.theta) > THETA_MAX or abs(erreur) > 5.0)    # drone trop incliné, ou voiture perdue
        if terminated:
            reward -= 2.0 * (N_STEPS - self.k)                           # pire que tout ce qui restait à perdre
        truncated = self.k >= N_STEPS      # les 10 s sont écoulées

        # 6. Ce qu'on renvoie
        m = self._mesures()
        obs = self._observation(m)
        return obs, float(reward), terminated, truncated, {"m": m}
