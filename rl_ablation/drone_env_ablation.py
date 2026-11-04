"""
Environnement du drone pour l'étude d'ablation.

Réglage par défaut = DroneEnv de rl_interface/Env_drone.py (le modèle de départ).
Chaque amélioration s'active séparément (voir VARIANTES et README.md) :
  A  action limitée à [0 ; 0,8] : le centre (0,4) est proche de la commande d'équilibre (0,386)
  B  nouvelle récompense + fin du vol si le drone sort de la zone
  C  intégrale de l'erreur dans l'observation (le « I » d'un PI)
  D  missions aléatoires + bruit de mesure + biais sur az (entraînement seulement)
  E  coefficient d'entropie de SAC fixé (réglage de l'algorithme, pas de l'environnement)
"""
import os
import sys

import numpy as np
import gymnasium as gym
from gymnasium import spaces

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', 'rl_interface'))
from Env_drone import ModeleRNN, TS, N_STEPS        # même RNN et mêmes constantes que l'environnement de l'équipe

DOSSIER_RNN = os.path.join(HERE, '..', 'identification_RNN', 'model_rnn', 'lightning_logs', 'version_3', 'export_NN')

# Les 7 entraînements : le modèle de départ, une amélioration à la fois, puis toutes ensemble
VARIANTES = {
    'base':  dict(),
    'A':     dict(action=(0.0, 0.8)),
    'B':     dict(recompense='nouvelle'),
    'C':     dict(integrale=True),
    'D':     dict(robuste=True),
    'E':     dict(ent_coef=0.02),
    'ABCDE': dict(action=(0.0, 0.8), recompense='nouvelle', integrale=True, robuste=True, ent_coef=0.02),
    # Suite de l'étude : les deux améliorations qui donnent le vol le plus lisse, sans les autres
    'AC':      dict(action=(0.0, 0.8), integrale=True),
    'AC_200k': dict(action=(0.0, 0.8), integrale=True, n_pas=200_000),     # même chose, entraînée 2 fois plus longtemps
}


def make_env(variante, usage='train'):
    """
    usage : 'train'  environnement d'entraînement de la variante
            'eval'   mission de la Figure 1, sans bruit, sans fin anticipée (évaluation et export MATLAB)
            'random' missions aléatoires, sans bruit (test de généralisation)
            'robust' mission de la Figure 1 avec bruit de mesure et biais sur az (test de robustesse)
    """
    cfg = VARIANTES[variante]
    kw = dict(action=cfg.get('action', (-1.0, 1.0)), integrale=cfg.get('integrale', False))
    if usage == 'train':
        kw.update(recompense=cfg.get('recompense', 'base'),
                  missions_aleatoires=cfg.get('robuste', False), bruit=cfg.get('robuste', False))
    elif usage == 'random':
        kw.update(missions_aleatoires=True)
    elif usage == 'robust':
        kw.update(bruit=True)
    return DroneEnvAblation(**kw)


class DroneEnvAblation(gym.Env):

    def __init__(self, action=(-1.0, 1.0), recompense='base', integrale=False, missions_aleatoires=False, bruit=False):
        super().__init__()
        self.u_min, self.u_max = action
        self.recompense = recompense
        self.integrale = integrale
        self.missions_aleatoires = missions_aleatoires
        self.bruit = bruit
        # Action : une commande moteur (A : plage réduite autour de l'équilibre)
        self.action_space = spaces.Box(low=self.u_min, high=self.u_max, shape=(1,), dtype=np.float32)
        # Observation : [erreur, vz, az_prev/10, u_prev] (+ intégrale de l'erreur avec C)
        n_obs = 5 if integrale else 4
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(n_obs,), dtype=np.float32)
        self.modele = ModeleRNN(DOSSIER_RNN)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.modele.reset()                  # mémoire du RNN vide : moteurs au repos
        self.k = 0
        self.t = 0.0
        self.z = 0.0
        self.vz = 0.0
        self.az_prev = 0.0
        self.u_prev = 0.0
        self.I = 0.0                         # intégrale de l'erreur (C)
        # D : erreur du modèle sur az, tirée au hasard pour tout le vol (ordre de grandeur de l'erreur du RNN)
        self.biais = self.np_random.uniform(-0.3, 0.3) if self.bruit else 0.0
        if self.missions_aleatoires:         # D : décollage puis descente, instants et niveaux tirés au hasard
            t1 = self.np_random.uniform(1.0, 4.0)
            t2 = t1 + self.np_random.uniform(2.0, 4.0)
            self.paliers = [(t1, self.np_random.uniform(0.2, 1.0)), (t2, self.np_random.uniform(-0.25, 0.75))]
        else:                                # mission de la Figure 1, comme Env_drone.py
            self.paliers = [(3.0, 0.5), (7.0, -0.25)]
        m = self._mesures()
        return self._observation(m), {"m": m}

    # Objectif de la mission : 0 m sur la plateforme, puis les paliers
    def mission(self, t):
        z_ref = 0.0
        for t_palier, niveau in self.paliers:
            if t >= t_palier:
                z_ref = niveau
        return z_ref

    # ce que les capteurs mesurent (D : bruités)
    def _mesures(self):
        m = {"t": self.t, "z": self.z, "vz": self.vz,
             "az_prev": self.az_prev, "z_ref": self.mission(self.t), "u_prev": self.u_prev}
        if self.bruit:
            m["z"] += self.np_random.normal(0.0, 0.01)
            m["vz"] += self.np_random.normal(0.0, 0.05)
            m["az_prev"] += self.np_random.normal(0.0, 0.2)
        return m

    # ce que l'agent voit (observation) : appelée une fois par pas, comme observation() de rl_controller.m
    def _observation(self, m):
        erreur = m["z_ref"] - m["z"]
        obs = [erreur, m["vz"], m["az_prev"] / 10, m["u_prev"]]
        if self.integrale:                   # C : somme des erreurs passées, bornée à ±2 m.s (anti-emballement)
            self.I = float(np.clip(self.I + erreur * TS, -2.0, 2.0))
            obs.append(self.I)
        return np.array(obs, dtype=np.float32)

    def step(self, action):
        # 1. La commande de l'IA
        u = float(np.clip(action[0], self.u_min, self.u_max))
        z_ref = self.mission(self.t)

        # 2. Physique : le RNN donne az (+ biais avec D), puis on intègre
        az = self.modele.step(u) + self.biais
        self.z = self.z + self.vz * TS + 0.5 * az * TS**2
        self.vz = self.vz + az * TS

        # 3. Récompense
        erreur = z_ref - self.z
        terminated = False
        if self.recompense == 'base':        # récompense de départ (Env_drone.py)
            reward = (-10 * abs(erreur)
                      - 0.05 * abs(self.vz)
                      - 10.0 * max(0.0, -0.25 - self.z))
        else:                                # B
            reward = (-abs(erreur)                     # position : être sur la consigne
                      - 0.1 * self.vz**2               # vitesse : amortir les oscillations
                      - 0.5 * (u - self.u_prev)**2     # commande lisse (plutôt que pénaliser az)
                      + 0.5 * (abs(erreur) < 0.05))    # bonus de précision : à moins de 5 cm
            if self.z < -0.30 or self.z > 1.5:         # sous le sol ou trop haut : fin du vol
                terminated = True
                reward -= 2.0 * (N_STEPS - self.k)     # pire que tout ce qui restait à perdre en volant

        # 4. Mise à jour pour le pas suivant
        self.az_prev = az
        self.u_prev = u
        self.k += 1
        self.t = self.k * TS

        # 5. Fin du vol ?
        truncated = self.k >= N_STEPS

        # 6. Ce qu'on renvoie
        m = self._mesures()
        return self._observation(m), float(reward), terminated, truncated, {"m": m}
