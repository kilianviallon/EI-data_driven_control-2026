"""
Environnement Gymnasium du drone (axe vertical) pour la Track B.

La physique vient du modèle identifié aux étapes 1 à 3 : le RNN u -> az
(identification_RNN/, version_0), recodé ici en numpy à partir des poids exportés
en .txt (mêmes équations que dans le README de identification_RNN) :

    h_k  = tanh(Wih * u_k + Whh * h_{k-1} + bh)
    az_k = 20 * (Wo * h_k + bo) - 9.81

puis on intègre az (accélération constante pendant Ts) pour obtenir vz et z.

    from drone_env import DroneEnv
    env_train = DroneEnv(randomize=True)    # missions aléatoires + perturbations (entraînement)
    env_eval  = DroneEnv()                  # mission de la Figure 1, sans bruit (export MATLAB)

IMPORTANT : mission() et _get_obs() doivent être recopiées À L'IDENTIQUE dans
rl_controller.m (fonctions mission() et observation()).
"""
import os
import numpy as np
import gymnasium as gym
from gymnasium import spaces

TS = 0.05            # période d'échantillonnage [s] (étape 1)
N_STEPS = 200        # 10 s de vol
Z_GROUND = -0.25     # le sol [m] (Figure 1)
I_MAX = 1.0          # saturation de l'intégrale de l'erreur [m.s]

HERE = os.path.dirname(os.path.abspath(__file__))
RNN_DIR = os.path.join(HERE, "..", "identification_RNN", "model_rnn",
                       "lightning_logs", "version_0", "export_NN")


# ---------------------------------------------------------------------------
# Modèle identifié
# ---------------------------------------------------------------------------
def _read_matlab_txt(path):
    """Lit un fichier '[a b; c d]' écrit par RNN_drone.py -> np.ndarray 2D."""
    with open(path) as f:
        txt = f.read().strip().strip("[]")
    return np.array([[float(x) for x in row.split()] for row in txt.split(";")])


class RNNModel:
    """Le RNN u -> az identifié, pas à pas (état caché h)."""

    def __init__(self, export_dir=RNN_DIR):
        r = lambda name: _read_matlab_txt(os.path.join(export_dir, name + ".txt"))
        self.Wih = r("weight_ih_l0").reshape(-1)        # (16,)
        self.Whh = r("weight_hh_l0")                    # (16, 16)
        self.bh = r("bias_h").reshape(-1)               # (16,)
        self.Wo = r("head_weight").reshape(-1)          # (16,)
        self.bo = float(r("head_bias").reshape(-1)[0])
        u_scale, y_offset, y_scale = r("normalization").reshape(-1)
        self.u_scale, self.y_offset, self.y_scale = u_scale, y_offset, y_scale
        self.h = np.zeros_like(self.bh)

    def reset(self):
        self.h = np.zeros_like(self.bh)                 # h0 = 0, comme à l'entraînement

    def step(self, u):
        """Applique u pendant un pas, renvoie az [m/s²]."""
        self.h = np.tanh(self.Wih * (u / self.u_scale) + self.Whh @ self.h + self.bh)
        return self.y_scale * (self.Wo @ self.h + self.bo) + self.y_offset


# ---------------------------------------------------------------------------
# Mission (Figure 1) -- à recopier dans mission() de rl_controller.m
# ---------------------------------------------------------------------------
T1, T2 = 3.0, 7.0                    # instants des changements de palier [s] (votre choix)
Z0, Z1, Z2 = 0.0, 0.5, -0.25         # plateforme, palier haut, sol [m]


def mission(t):
    if t < T1:
        return Z0
    elif t < T2:
        return Z1
    else:
        return Z2


# ---------------------------------------------------------------------------
# Environnement
# ---------------------------------------------------------------------------
class DroneEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, randomize=False, noise=0.0, export_dir=RNN_DIR):
        """
        randomize : True pour l'entraînement : missions aléatoires (instants et paliers)
                    + biais d'accélération aléatoire (erreur de modèle). False : Figure 1.
        noise     : écart-type du bruit ajouté aux mesures z [m] (et vz, az en proportion).
        """
        super().__init__()
        self.model = RNNModel(export_dir)
        self.randomize = randomize
        self.noise = noise
        self.Ts = TS
        self.action_space = spaces.Box(-1.0, 1.0, shape=(1,), dtype=np.float32)
        # obs = [erreur, vz, az_prev/10, u_prev, intégrale de l'erreur]
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(5,), dtype=np.float32)

    # -- mission de cet épisode ------------------------------------------------
    def _new_mission(self):
        r = self.np_random
        if not self.randomize or r.random() < 0.3:      # 30 % du temps : la vraie mission
            self.ref = (T1, T2, Z0, Z1, Z2)
        else:
            t1 = r.uniform(1.0, 4.0)
            t2 = r.uniform(t1 + 2.0, 8.0)
            z1 = r.uniform(0.1, 0.8)
            z2 = r.uniform(Z_GROUND, z1 - 0.1)
            self.ref = (t1, t2, Z0, z1, z2)

    def z_ref(self, t):
        t1, t2, z0, z1, z2 = self.ref
        return z0 if t < t1 else (z1 if t < t2 else z2)

    # -- mesures et observation ----------------------------------------------
    def _measurements(self):
        z, vz, az = self.z, self.vz, self.az_prev
        if self.noise > 0:                              # capteurs bruités (robustesse)
            r = self.np_random
            z += self.noise * r.normal()
            vz += 2 * self.noise * r.normal()
            az += 10 * self.noise * r.normal()
        return {"t": self.t, "z": z, "vz": vz, "az_prev": az,
                "z_ref": self.z_ref(self.t), "u_prev": self.u_prev}

    def _get_obs(self, m):
        """À recopier dans observation() de rl_controller.m (même ordre, mêmes échelles)."""
        e = m["z_ref"] - m["z"]
        self.I = float(np.clip(self.I + e * TS, -I_MAX, I_MAX))
        return np.array([e, m["vz"], m["az_prev"] / 10.0, m["u_prev"], self.I],
                        dtype=np.float32)

    # -- API Gymnasium ---------------------------------------------------------
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._new_mission()
        self.model.reset()
        self.k = 0
        self.t = 0.0
        self.z, self.vz = Z0, 0.0                       # posé sur la plateforme
        self.az_prev, self.u_prev = 0.0, 0.0
        self.I = 0.0
        # erreur de modèle : biais constant sur az (le RNN a ~0.25 m/s² d'erreur RMS)
        self.az_bias = self.np_random.uniform(-0.5, 0.5) if self.randomize else 0.0
        m = self._measurements()
        return self._get_obs(m), {"m": m}

    def step(self, action):
        u = float(np.clip(np.asarray(action).reshape(-1)[0], -1.0, 1.0))
        z_ref = self.z_ref(self.t)                      # consigne pendant ce pas

        # physique : modèle identifié, puis intégration (az constante sur Ts)
        az = self.model.step(u) + self.az_bias
        self.z += self.vz * TS + 0.5 * az * TS**2
        self.vz += az * TS

        # récompense : suivre la consigne, sans brutalité, sans passer sous le sol
        e = z_ref - self.z
        reward = (-abs(e)
                  - 0.05 * abs(self.vz)
                  - 0.1 * (u - self.u_prev) ** 2
                  - 10.0 * max(0.0, Z_GROUND - self.z))

        self.az_prev, self.u_prev = az, u
        self.k += 1
        self.t = self.k * TS                            # pas de t += Ts (erreurs d'arrondi)

        lost = abs(self.z) > 3.0                         # parti trop loin : on arrête
        if lost:
            reward -= 100.0
        terminated = lost
        truncated = self.k >= N_STEPS                   # fin des 10 s (limite de temps)

        m = self._measurements()
        return self._get_obs(m), float(reward), terminated, truncated, {"m": m}


# ---------------------------------------------------------------------------
# Baseline linéaire (PID) pour comparer, comme demandé dans le sujet
# ---------------------------------------------------------------------------
def u_hover(model=None):
    """Commande d'équilibre (az = 0) du modèle identifié, par dichotomie."""
    model = model or RNNModel()
    lo, hi = -1.0, 1.0
    for _ in range(40):
        mid = 0.5 * (lo + hi)
        model.reset()
        for _ in range(100):
            az = model.step(mid)
        lo, hi = (mid, hi) if az < 0 else (lo, mid)
    return 0.5 * (lo + hi)


class PID:
    """u = u_hover + Kp e + Ki ∫e - Kd vz (sortie saturée à [-1, 1])."""

    def __init__(self, Kp=0.6, Ki=0.3, Kd=0.35, uh=None):
        self.Kp, self.Ki, self.Kd = Kp, Ki, Kd
        self.uh = u_hover() if uh is None else uh

    def __call__(self, obs):
        e, vz, _, _, I = obs
        return np.array([np.clip(self.uh + self.Kp * e + self.Ki * I - self.Kd * vz, -1, 1)],
                        dtype=np.float32)


def run_episode(env, policy, seed=0):
    """Joue un épisode. policy(obs) -> action. Renvoie un dict de tableaux + retour total."""
    obs, info = env.reset(seed=seed)
    log = {k: [] for k in ("t", "z", "z_ref", "u", "vz")}
    total = 0.0
    while True:
        a = policy(obs)
        for k in ("t", "z", "z_ref", "vz"):
            log[k].append(info["m"][k])
        log["u"].append(float(np.asarray(a).reshape(-1)[0]))
        obs, r, term, trunc, info = env.step(a)
        total += r
        if term or trunc:
            break
    log = {k: np.array(v) for k, v in log.items()}
    log["return"] = total
    log["rmse"] = float(np.sqrt(np.mean((log["z_ref"] - log["z"]) ** 2)))
    return log
