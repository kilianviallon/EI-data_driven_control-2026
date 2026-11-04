"""
Vérification de Env_drone_x.py avant l'entraînement RL :
  1. check_env de Stable-Baselines3 ;
  2. politique « rien faire » (u = 0) : la voiture s'éloigne, le vol s'arrête ;
  3. PD en cascade de référence (position -> theta visé -> u_theta) : l'environnement doit être pilotable ;
  4. figure 2D (voiture et drone dans le plan x-z) et courbes au cours du temps : figures/test_env_x.png
"""
import os
import itertools

import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3.common.env_checker import check_env

from Env_drone_x import DroneEnvX, TS, U_MAX, Z_VOL

os.chdir(os.path.dirname(os.path.abspath(__file__)))
os.makedirs('figures', exist_ok=True)


def voler(env, politique, seed=0):
    """Un vol complet ; renvoie les signaux et la somme des récompenses."""
    obs, info = env.reset(seed=seed)
    log = {k: [] for k in ('t', 'x', 'x_car', 'theta', 'u', 'r')}
    fini = False
    while not fini:
        u = politique(info['m'])
        for k in ('t', 'x', 'x_car', 'theta'):
            log[k].append(info['m'][k])
        obs, r, terminated, truncated, info = env.step(np.array([u / U_MAX], dtype=np.float32))   # action = u / U_MAX
        log['u'].append(u)
        log['r'].append(r)
        fini = terminated or truncated
    return {k: np.array(v) for k, v in log.items()}, terminated


def pd_cascade(kx, kv, kp=0.02, kd=0.015):
    """theta visé = kx.erreur + kv.vitesse relative (borné à ±0,4 rad), puis u = kp.(theta visé - theta) - kd.w."""
    def politique(m):
        theta_vise = np.clip(kx * (m['x_car'] - m['x']) + kv * (m['v_car'] - m['vx']), -0.4, 0.4)
        return float(np.clip(kp * (theta_vise - m['theta']) - kd * m['w'], -U_MAX, U_MAX))
    return politique


# 1. Conformité Gymnasium / Stable-Baselines3
for modele_ax in ('theta', 'rnn'):
    check_env(DroneEnvX(modele_ax=modele_ax, voiture_aleatoire=True))
print("check_env OK (modele_ax = 'theta' et 'rnn')")

# 2. Politique « rien faire »
env = DroneEnvX()
vol0, crash0 = voler(env, lambda m: 0.0)
print(f"u = 0 : {len(vol0['t'])} pas, arrêt anticipé {crash0}, somme des récompenses {vol0['r'].sum():.0f}")

# 3. PD en cascade : réglage des 4 gains sur la mission d'évaluation, critère = somme des récompenses de
#    l'environnement (l'objectif même de l'agent RL), sans arrêt anticipé du vol
meilleur = None
for kp, kd, kx, kv in itertools.product([0.02, 0.04, 0.06, 0.08], [0.015, 0.03, 0.045, 0.06],
                                        [0.03, 0.05, 0.1, 0.15, 0.2, 0.3], [0.05, 0.1, 0.2, 0.3, 0.5]):
    vol, crash = voler(env, pd_cascade(kx, kv, kp, kd))
    if not crash and (meilleur is None or vol['r'].sum() > meilleur[0]):
        meilleur = (vol['r'].sum(), kx, kv, kp, kd)
_, kx, kv, kp, kd = meilleur
vol, _ = voler(env, pd_cascade(kx, kv, kp, kd))
vol_rnn, _ = voler(DroneEnvX(modele_ax='rnn'), pd_cascade(kx, kv, kp, kd))
e = vol['x_car'] - vol['x']
print(f"PD (kx {kx}, kv {kv}, kp {kp}, kd {kd}) : RMSE de suivi {np.sqrt(np.mean(e**2)):.3f} m, erreur max {np.abs(e).max():.2f} m, "
      f"theta max {np.abs(vol['theta']).max():.2f} rad, somme des récompenses {vol['r'].sum():.0f}")
print(f"  même PD avec ax du RNN (32 neurones) : RMSE de suivi {np.sqrt(np.mean((vol_rnn['x_car'] - vol_rnn['x'])**2)):.3f} m")
rs = [voler(DroneEnvX(voiture_aleatoire=True), pd_cascade(kx, kv, kp, kd), seed=s) for s in range(20)]
print(f"  sur 20 profils de voiture aléatoires : RMSE moyen {np.mean([np.sqrt(np.mean((v['x_car'] - v['x'])**2)) for v, _ in rs]):.3f} m, "
      f"arrêts anticipés {sum(c for _, c in rs)}")

# 4. Figure : vue 2D et courbes
fig = plt.figure(figsize=(12, 9))
gs = fig.add_gridspec(4, 1, height_ratios=[1.6, 1, 1, 1], hspace=0.45)
ax2d = fig.add_subplot(gs[0])
ax2d.axhline(-0.25, color='#898781', linewidth=1)
for k in range(0, len(vol['t']), 20):                                   # une image par seconde
    xc, xd, th = vol['x_car'][k], vol['x'][k], vol['theta'][k]
    ax2d.add_patch(plt.Rectangle((xc - 0.3, -0.25), 0.6, 0.2, color='#c3c2b7'))   # la voiture
    dx, dz = 0.25 * np.cos(th), -0.25 * np.sin(th)                      # le drone, incliné de theta
    ax2d.plot([xd - dx, xd + dx], [Z_VOL - dz, Z_VOL + dz], color='#2a78d6', linewidth=2.5)
    ax2d.annotate(f"{vol['t'][k]:.0f} s", (xc, -0.35), ha='center', fontsize=7, color='#52514e')
ax2d.plot(vol['x'], np.full_like(vol['x'], Z_VOL), ':', color='#2a78d6', linewidth=1)
ax2d.set_ylim(-0.5, 1.0)
ax2d.set_aspect('equal', adjustable='datalim')
ax2d.set_title(f"Vue 2D, une image par seconde : voiture (gris) et drone (bleu), PD de référence", loc='left')
ax2d.set_xlabel('x [m]')
ax2d.set_ylabel('z [m]')
axs = [fig.add_subplot(gs[i]) for i in (1, 2, 3)]
axs[0].plot(vol['t'], vol['x_car'], '--', color='#52514e', label='voiture')
axs[0].plot(vol['t'], vol['x'], color='#2a78d6', label='drone (PD)')
axs[0].plot(vol0['t'], vol0['x'], color='#898781', linewidth=1, label='drone (u = 0)')
axs[0].set_ylabel('x [m]')
axs[0].legend(fontsize=8)
axs[1].plot(vol['t'], vol['theta'], color='#2a78d6')
axs[1].set_ylabel('theta [rad]')
axs[2].step(vol['t'], vol['u'], where='post', color='#2a78d6')
axs[2].set_ylabel('u_theta')
axs[2].set_xlabel('t [s]')
for a in [ax2d] + axs:
    a.grid(True, alpha=0.3)
fig.savefig('figures/test_env_x.png', dpi=130, bbox_inches='tight')
print("Figure : figures/test_env_x.png")
