"""
Un entraînement de l'étude d'ablation :  python train_variant.py <variante> <graine> [pas]

Écrit dans runs/<variante>_s<graine>/ :
  evaluations.json   RMSE sur la mission de la Figure 1 tous les pas/20 pas (courbe d'apprentissage)
  meilleur_modele    politique avec le RMSE le plus bas ; modele_final : celle de la fin
  metriques.json     résultats du meilleur modèle (mission, missions aléatoires, bruit + biais)
  vol_mission.npz    vol du meilleur modèle sur la mission (t, z, z_ref, vz, u)
  policy.mat, traj_model.mat   export MATLAB du meilleur modèle (export_policy.py)
"""
import os
import sys
import json
import time

import numpy as np
import torch
torch.set_num_threads(1)              # petit réseau : 1 thread est plus rapide, et laisse les autres cœurs aux autres entraînements

from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor

from drone_env_ablation import HERE, VARIANTES, make_env
from export_policy import export_for_matlab          # rl_interface/ est dans le chemin (drone_env_ablation)

N_MISSIONS_TEST = 20                  # test de généralisation : 20 missions aléatoires (toujours les mêmes)
N_BRUIT_TEST = 10                     # test de robustesse : 10 vols bruités (toujours les mêmes)


def voler(model, env, seed=None):
    """Un vol de 10 s avec la politique déterministe ; renvoie l'état vrai (sans bruit) à chaque pas."""
    obs, _ = env.reset(seed=seed)
    vol = {k: [] for k in ('t', 'z', 'z_ref', 'vz', 'u')}
    fini = False
    while not fini:
        u = float(model.predict(obs, deterministic=True)[0][0])
        for k, v in (('t', env.t), ('z', env.z), ('z_ref', env.mission(env.t)), ('vz', env.vz), ('u', u)):
            vol[k].append(v)
        obs, _, terminated, truncated, _ = env.step(np.array([u], dtype=np.float32))
        fini = terminated or truncated
    return {k: np.array(v) for k, v in vol.items()}


def rmse(vol):
    return float(np.sqrt(np.mean((vol['z_ref'] - vol['z'])**2)))


class EvaluationMission(BaseCallback):
    """Tous les `tous_les` pas : un vol sur la mission de la Figure 1 ; garde le modèle au RMSE le plus bas."""

    def __init__(self, variante, dossier, tous_les):
        super().__init__()
        self.env = make_env(variante, 'eval')
        self.dossier = dossier
        self.tous_les = tous_les
        self.historique = []
        self.meilleur = np.inf
        self.t0 = time.time()

    def _on_step(self):
        if self.num_timesteps % self.tous_les == 0:
            vol = voler(self.model, self.env)
            e = rmse(vol)
            self.historique.append(dict(pas=self.num_timesteps, rmse=e, z_min=float(vol['z'].min()),
                                        duree_s=round(time.time() - self.t0, 1)))
            if e < self.meilleur:
                self.meilleur = e
                self.model.save(os.path.join(self.dossier, 'meilleur_modele'))
            with open(os.path.join(self.dossier, 'evaluations.json'), 'w') as f:
                json.dump(self.historique, f, indent=1)
        return True


if __name__ == '__main__':
    variante, graine = sys.argv[1], int(sys.argv[2])
    n_pas = int(sys.argv[3]) if len(sys.argv) > 3 else 100_000
    n_pas = VARIANTES[variante].get('n_pas', n_pas)          # une variante peut imposer sa durée (AC_200k)
    dossier = os.path.join(HERE, 'runs', f'{variante}_s{graine}')
    os.makedirs(dossier, exist_ok=True)

    # Réglages communs à toutes les variantes : réseau 64x64, même nombre de pas, mêmes graines
    env = Monitor(make_env(variante, 'train'), os.path.join(dossier, 'monitor'))
    agent = SAC("MlpPolicy", env, seed=graine, verbose=0,
                ent_coef=VARIANTES[variante].get('ent_coef', 'auto'),     # E : coefficient d'entropie fixé
                policy_kwargs=dict(net_arch=[64, 64]))
    evaluation = EvaluationMission(variante, dossier, tous_les=max(n_pas // 20, 1))
    t0 = time.time()
    agent.learn(total_timesteps=n_pas, callback=evaluation)
    duree = time.time() - t0
    agent.save(os.path.join(dossier, 'modele_final'))

    # Résultats du meilleur modèle
    meilleur = SAC.load(os.path.join(dossier, 'meilleur_modele'), device='cpu')
    vol = voler(meilleur, make_env(variante, 'eval'))
    np.savez(os.path.join(dossier, 'vol_mission.npz'), **vol)
    t, e = vol['t'], np.abs(vol['z_ref'] - vol['z'])
    fin_palier = ((t >= 2) & (t < 3)) | ((t >= 6) & (t < 7)) | (t >= 9)      # dernière seconde de chaque palier
    env_random, env_robust = make_env(variante, 'random'), make_env(variante, 'robust')
    metriques = dict(
        variante=variante, graine=graine, n_pas=n_pas, duree_min=round(duree / 60, 2),
        pas_meilleur=min(evaluation.historique, key=lambda h: h['rmse'])['pas'],
        rmse=rmse(vol),                                                        # mission de la Figure 1
        rmse_modele_final=rmse(voler(agent, make_env(variante, 'eval'))),     # même mission, modèle de la fin
        erreur_statique=float(e[fin_palier].mean()),                          # |erreur| en fin de palier
        z_min=float(vol['z'].min()),                                          # sol à -0,25 m
        variation_u=float(np.abs(np.diff(vol['u'])).mean()),                  # douceur de la commande
        rmse_missions_aleatoires=float(np.mean([rmse(voler(meilleur, env_random, seed=1000 + i))
                                                for i in range(N_MISSIONS_TEST)])),
        rmse_bruit=float(np.mean([rmse(voler(meilleur, env_robust, seed=2000 + i))
                                  for i in range(N_BRUIT_TEST)])),
    )
    with open(os.path.join(dossier, 'metriques.json'), 'w') as f:
        json.dump(metriques, f, indent=1)

    # Export MATLAB du meilleur modèle (policy.mat, traj_model.mat écrits dans le dossier courant)
    os.chdir(dossier)
    export_for_matlab(meilleur, make_env(variante, 'eval'))
    print(json.dumps(metriques))
