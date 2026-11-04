# Poids du RNN u_theta → ax (32 neurones)

RNN de l'accélération sur x (sortie `ax` du simulateur) en fonction de la commande de tangage u_theta, avec
32 neurones au lieu de 16. Entraîné par `RNN_drone_xtheta.py ax 32` sur les bases de `generation_donnees_x_theta/`
(3000 essais de 100 s, Ts = 0,05 s). **C'est le modèle de ax à utiliser** (meilleur que la version à 16 neurones).

**Modèle retenu : `lightning_logs/version_1/`** (meilleur des 3 entraînements en validation).
Poids au format MATLAB : `lightning_logs/version_1/export_NN/`.

| Version | Graine | Meilleure époque | RMSE validation [m/s²] |
|---|---|---|---|
| version_0 | 1 | 30 / 30 | 0.721 |
| **version_1** | **2** | **30 / 30** | **0.676** |
| version_2 | 3 | 29 / 30 | 0.690 |

Comparaison avec 16 neurones (écart-type de ax : 1,76 m/s²) :

| | 16 neurones | **32 neurones** |
|---|---|---|
| Test Python, 450 essais de 100 s jamais vus | RMSE 0,828 m/s², R² = 0,779 | **RMSE 0,661 m/s², R² = 0,859** |
| Test MATLAB face au simulateur, même signal de 20 s | RMSE 0,611 m/s² | **RMSE 0,361 m/s²** |

La forme et les instants sont bien suivis ; l'erreur grandit encore sur les séquences longues (ax dépend de
theta, la double intégrale de a_theta). Le meilleur score est atteint à la dernière époque : plus d'époques
amélioreraient encore le modèle.

## Utiliser les poids
RNN simple (tanh), 1 couche, 32 neurones. Pas à pas, h0 = 0 :

```
u_n = u_theta / 0.02
h   = tanh(Wih*u_n + Whh*h + bh)
ax  = 4.5*(Wo*h + bo)
```

| Fichier de `export_NN/` | Contenu |
|---|---|
| `weight_ih_l0.txt` | Wih (32×1) |
| `weight_hh_l0.txt` | Whh (32×32) |
| `bias_h.txt` | bh (32), somme des deux biais de PyTorch |
| `head_weight.txt`, `head_bias.txt` | Wo (1×32), bo |
| `normalization.txt` | `[u_scale y_offset y_scale]` = `[0.02 0 4.5]` |

Lecture en MATLAB : `Whh = str2num(fileread('weight_hh_l0.txt'));` (voir `matlab_test/test_RNN_xtheta.m`).
Chaque version contient aussi le checkpoint PyTorch Lightning (`.ckpt`) et `run_info.json` (données,
normalisation, hyperparamètres, résultats).
