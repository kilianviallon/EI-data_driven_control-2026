# Poids du RNN u_theta → ax (16 neurones)

RNN de l'accélération sur x (sortie `ax` du simulateur) en fonction de la commande de tangage u_theta.
Entraîné par `RNN_drone_xtheta.py ax` sur les bases de `generation_donnees_x_theta/`
(3000 essais de 100 s, Ts = 0,05 s).

**Modèle retenu : `lightning_logs/version_0/`** (meilleur des 3 entraînements en validation).
Poids au format MATLAB : `lightning_logs/version_0/export_NN/`.

| Version | Graine | Meilleure époque | RMSE validation [m/s²] |
|---|---|---|---|
| **version_0** | **1** | **29 / 30** | **0.828** |
| version_1 | 2 | 30 / 30 | 0.848 |
| version_2 | 3 | 30 / 30 | 0.844 |

Résultats du modèle retenu (écart-type de ax : 1,76 m/s²) :
- test Python, 450 essais jamais vus : RMSE = 0,828 m/s², R² = 0,779 ;
- test MATLAB face au simulateur (`boucle_tangage.slx`, signal de 20 s jamais vu) : RMSE = 0,560 m/s².

Modèle moyen : ax dépend de theta, la double intégrale de a_theta, que 16 neurones reconstruisent mal à partir
de u_theta seul (les 3 graines butent au même niveau). **Utiliser plutôt la version à 32 neurones**
([`../RNN_ax_32neurones`](../RNN_ax_32neurones) : RMSE 0,661 m/s² en test, 0,361 m/s² face au simulateur).

## Utiliser les poids
RNN simple (tanh), 1 couche, 16 neurones. Pas à pas, h0 = 0 :

```
u_n = u_theta / 0.02
h   = tanh(Wih*u_n + Whh*h + bh)
ax  = 4.5*(Wo*h + bo)
```

| Fichier de `export_NN/` | Contenu |
|---|---|
| `weight_ih_l0.txt` | Wih (16×1) |
| `weight_hh_l0.txt` | Whh (16×16) |
| `bias_h.txt` | bh (16), somme des deux biais de PyTorch |
| `head_weight.txt`, `head_bias.txt` | Wo (1×16), bo |
| `normalization.txt` | `[u_scale y_offset y_scale]` = `[0.02 0 4.5]` |

Lecture en MATLAB : `Whh = str2num(fileread('weight_hh_l0.txt'));` (voir `matlab_test/test_RNN_xtheta.m`).
Chaque version contient aussi le checkpoint PyTorch Lightning (`.ckpt`) et `run_info.json` (données,
normalisation, hyperparamètres, résultats).
