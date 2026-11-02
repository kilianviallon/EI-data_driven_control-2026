# Identification du drone par RNN

Modèle **u → az** (commande moteur → accélération verticale), appris sur les bases nettoyées de
[`../data_base_cleaning`](../data_base_cleaning). C'est le modèle sur lequel l'agent RL sera entraîné.

| Dossier / fichier | Contenu |
|---|---|
| `RNN_drone.py` | Algorithme : entraînement (PyTorch Lightning), test, export des poids au format MATLAB |
| `model_rnn/lightning_logs/version_0/` | Modèle entraîné : checkpoint `.ckpt`, `run_info.json` (données, normalisation, hyperparamètres, résultats) |
| `model_rnn/.../export_NN/` | Poids du RNN en `.txt`, lisibles par MATLAB |
| `matlab_test/` | `test_RNN.m` + `test_RNN_harness.slx` : même u envoyé à `UAV_model.slx` et au RNN, comparaison de az |
| `figures/` | Captures des tests de validation sous MATLAB |

## Modèle
- RNN simple (tanh), 1 couche, 16 neurones, 321 paramètres ; Ts = 0.05 s
- Normalisation : u inchangé (déjà dans [-1, 1]), az_n = (az + 9.81) / 20
- 30 époques, lr = 3e-3, environ 3,5 min sur CPU ; essais répartis 70 / 15 / 15 % (train / val / test)
- Équations (pas à pas, h0 = 0) :
  `h = tanh(Wih*u + Whh*h + bh)` puis `az = 20*(Wo*h + bo) - 9.81`

## Résultats
- **Python**, 450 essais de test : RMSE = 0.239 m/s², R² = 0.9996
- **MATLAB contre `UAV_model.slx`**, signaux aléatoires de 20 s jamais vus :

| Figure | g_noise | RMSE [m/s²] |
|---|---|---|
| `test1_RNN.png` | 0 | 0.177 |
| `test2_RNN.png` | 0 | 0.270 |
| `test3_RNN.png` | 0 | 0.253 |
| `test1_RNN_Noise.png` | 0.1 | 0.233 |
| `test2_RNN_Noise.png` | 0.1 | 0.266 |

L'erreur est la même qu'en Python : le dataset représente bien le simulateur à Ts = 0.05 s.

## Utilisation
- **Réentraîner** : `python RNN_drone.py` (crée `model_rnn/lightning_logs/version_N`).
  `TRAIN = False` recharge le checkpoint existant sans réentraîner. Les poids MATLAB sont écrits
  après la fermeture des figures.
- **Tester sous MATLAB** : lancer `matlab_test/test_RNN.m`, pas le `.slx` (c'est le script qui crée
  `u_ts` et `Ts`). Pour un nouveau modèle, changer `version_0` dans `test_RNN.m`.
