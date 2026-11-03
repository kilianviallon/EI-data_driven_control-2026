# EI-data-driven-control-2026

| Dossier | Contenu |
|---|---|
| `data_base_cleaning/` | Nettoyage des données d'identification ; bases nettoyées en `.csv.gz` |
| `identification_RNN/` | Modèle RNN u → az : entraînement, poids, test MATLAB, figures (voir son README) |
| `Matlab files/`, `rl_interface/` | Fichiers fournis (simulateur du drone, interface RL) |

Environnement Python : `python -m venv .venv` puis `.venv\Scripts\pip install -r requirements.txt`.
Les données brutes `bdd_in_mat_05.csv` / `bdd_out_mat_05.csv` sont à placer à la racine (non versionnées).