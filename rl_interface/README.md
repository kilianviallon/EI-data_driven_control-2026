# Track B : commande par apprentissage par renforcement

Agent PPO entraîné sur le modèle RNN identifié ([`../identification_RNN`](../identification_RNN), `version_3`),
puis exporté vers MATLAB et testé sur `UAV_model.slx`.

| Fichier | Contenu |
|---|---|
| `modele_rnn.py` | Modèle RNN u → az (`version_3`), lu depuis les poids `export_NN/` |
| `Env_drone.py` | Environnement Gymnasium : mission, observation, récompense, crash |
| `modele_lacet.py`, `Env_drone_lacet.py` | Tâche avancée : modèle du lacet et environnement avec un tour sur soi-même (voir [`../identification_lacet`](../identification_lacet)) |
| `track_b.ipynb` | Entraînement PPO et export (`policy.mat`, `traj_model.mat`) |
| `rl_controller.m`, `EINT_RL_eval.m`, `EINT_RL_closed_loop.slx` | Policy sous MATLAB et vol sur le simulateur |
| `policy.mat`, `traj_model.mat`, `ppo_drone.zip` | Policy entraînée (export MATLAB, trajectoire sur le modèle, agent SB3) |
| `figures/` | Comparaison des récompenses, vol sur le modèle et sur le simulateur |

## Environnement
- Observation : `[z_ref - z, vz, az_prev / 10, u_prev]` ; action u ∈ [-1, 1] ; Ts = 0.05 s, 200 pas (10 s)
- Récompense, avec e = z_ref - z :
  `1 - |e| + 0.3·exp(-(e/0.03)²) - 0.05·|vz| - 0.3·(u - u_prev)² - 10·max(0, -0.25 - z)`
- Crash : si |e| > 1 m ou z < -0.35 m, le vol s'arrête avec une récompense de -50

Le +1 rend la récompense positive quand le drone suit la consigne : rester en vol rapporte, donc le crash est
évité. Sans fin de vol (premier essai), aucune récompense ne fonctionne : le drone tombe de plusieurs centaines
de mètres et l'agent n'apprend rien (`figures/rewards_essai1_sans_crash.png`).

## Choix de la récompense
PPO, 500k pas, même graine, crash activé (`figures/rewards_essai2_avec_crash.png`) :

| Récompense | RMSE [m] | z min [m] | Erreur fin de palier 0 / 0.5 / -0.25 m |
|---|---|---|---|
| `base` : 1 - \|e\| - 0.05·\|vz\| - sol | 0.161 | -0.240 | 0.033 / 0.047 / 0.010 |
| `exp` : exp(-(e/0.1)²) + 0.5·exp(-(e/0.02)²) - 0.02·vz² - 0.5·Δu² - sol | 0.165 | -0.248 | 0.027 / 0.040 / 0.002 |
| `quad` : 1 - \|e\| - 2e² - 0.02·vz² - 0.3·Δu² - sol | 0.161 | -0.244 | 0.030 / 0.043 / 0.006 |
| **`abs_du`** (retenue) | **0.160** | -0.252 | **0.027 / 0.039 / 0.001** |

Les quatre se valent : c'est la fin de vol sur crash qui rend l'apprentissage possible. `abs_du` donne les plus
petites erreurs en fin de palier.

## Résultats
| Vol | RMSE [m] | z min [m] |
|---|---|---|
| Modèle RNN (Python) | 0.160 | -0.249 |
| `UAV_model.slx`, g_noise = 0.1 | 0.157 | -0.260 |

Le RMSE vient surtout du temps de montée sur les échelons (~0.7 s, commande saturée à ±1). Sur le simulateur,
le drone descend 1 cm sous le sol à l'atterrissage à cause du bruit.

## Utilisation
- **Entraîner** : exécuter `track_b.ipynb` (PPO, 16 vols en parallèle, ~5 min ; SAC était ~50 fois plus lent).
- **Tester sous MATLAB** : lancer `EINT_RL_eval`. `EINT_RL_closed_loop.slx` a été enregistré avec R2025b :
  sous R2025a, exécuter d'abord `set_param(0, 'ErrorIfLoadNewModel', 'off')`.
