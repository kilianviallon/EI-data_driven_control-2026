# Poids du RNN u_theta → a_theta (16 neurones)

RNN de l'accélération angulaire de tangage a_theta (sortie `w_dot_y` du simulateur) en fonction de la commande
de tangage u_theta. Entraîné par `RNN_drone_xtheta.py atheta` sur les bases de `generation_donnees_x_theta/`
(3000 essais de 100 s, Ts = 0,05 s).

**Modèle retenu : `lightning_logs/version_2/`** (meilleur des 3 entraînements en validation).
Poids au format MATLAB : `lightning_logs/version_2/export_NN/`.

| Version | Graine | Meilleure époque | RMSE validation [rad/s²] |
|---|---|---|---|
| version_0 | 1 | 29 / 30 | 0.0118 |
| version_1 | 2 | 27 / 30 | 0.0110 |
| **version_2** | **3** | **30 / 30** | **0.0065** |

Résultats du modèle retenu (écart-type de a_theta : 0,55 rad/s²) :
- test Python, 450 essais jamais vus : RMSE = 0,006 rad/s², R² = 0,99986 ;
- test MATLAB face au simulateur (`boucle_tangage.slx`, signal de 20 s jamais vu) : RMSE = 0,005 rad/s².

## Utiliser les poids
RNN simple (tanh), 1 couche, 16 neurones. Pas à pas, h0 = 0 :

```
u_n     = u_theta / 0.02
h       = tanh(Wih*u_n + Whh*h + bh)
a_theta = 2.6*(Wo*h + bo)
```

| Fichier de `export_NN/` | Contenu |
|---|---|
| `weight_ih_l0.txt` | Wih (16×1) |
| `weight_hh_l0.txt` | Whh (16×16) |
| `bias_h.txt` | bh (16), somme des deux biais de PyTorch |
| `head_weight.txt`, `head_bias.txt` | Wo (1×16), bo |
| `normalization.txt` | `[u_scale y_offset y_scale]` = `[0.02 0 2.6]` |

Lecture en MATLAB : `Whh = str2num(fileread('weight_hh_l0.txt'));` (voir `matlab_test/test_RNN_xtheta.m`).
Chaque version contient aussi le checkpoint PyTorch Lightning (`.ckpt`) et `run_info.json` (données,
normalisation, hyperparamètres, résultats).
