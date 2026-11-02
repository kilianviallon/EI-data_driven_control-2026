"""
Nettoyage des données d'identification du drone.

Entrées : bdd_in_mat_05.csv   commande u   (1 ligne = 1 essai de 2000 pas)
          bdd_out_mat_05.csv  accélération verticale az mesurée [m/s²] (même format)
          (à la racine du dépôt, non versionnés : voir .gitignore)
Sorties : bdd_in_clean.csv.gz, bdd_out_clean.csv.gz  (mêmes données, essais corrompus retirés,
          compressées en gzip pour passer sous la limite de 100 Mo de GitHub)
          cleaning.png                               (essais retirés vs essai normal)

Lecture des données nettoyées : np.loadtxt("bdd_out_clean.csv.gz", delimiter=",")

Un essai est retiré en entier s'il est corrompu : un modèle dynamique a besoin de l'historique
continu (az[k-1], u[k-1]...), on ne peut pas garder un essai avec un trou au milieu.
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).parent
IN_FILE = HERE.parent / "bdd_in_mat_05.csv"
OUT_FILE = HERE.parent / "bdd_out_mat_05.csv"

# Plage physique de az : avec u dans [-1, 1], az reste entre ~-30 et ~+10 m/s² (on garde de la marge)
AZ_MIN, AZ_MAX = -40.0, 20.0
# Un capteur bruité ne donne jamais 2 fois exactement la même valeur : au-delà, il est figé
MAX_REPEATS = 10


# 1. Chargement
U = np.loadtxt(IN_FILE, delimiter=",")
Y = np.loadtxt(OUT_FILE, delimiter=",")
assert U.shape == Y.shape, "les fichiers d'entrée et de sortie n'ont pas la même taille"
print(f"Données brutes : {U.shape[0]} essais x {U.shape[1]} pas")

# 2. Détection des essais corrompus (un booléen par essai)
has_nan = np.isnan(U).any(axis=1) | np.isnan(Y).any(axis=1)
out_of_range = (np.abs(U) > 1).any(axis=1) | ((Y < AZ_MIN) | (Y > AZ_MAX)).any(axis=1)
n_repeats = (np.diff(Y, axis=1) == 0).sum(axis=1)   # nb de fois où az[k] == az[k-1]
frozen = n_repeats > MAX_REPEATS

print(f"  valeurs manquantes (NaN)     : {np.flatnonzero(has_nan)}")
print(f"  valeurs hors plage physique  : {np.flatnonzero(out_of_range)}")
print(f"  capteur figé                 : {np.flatnonzero(frozen)}")

# 3. Suppression des essais corrompus (mêmes lignes dans U et Y pour garder l'alignement)
bad = has_nan | out_of_range | frozen
U_clean, Y_clean = U[~bad], Y[~bad]
print(f"Essais retirés : {np.flatnonzero(bad)} -> {U_clean.shape[0]} essais conservés")
print(f"az nettoyé dans [{Y_clean.min():.2f}, {Y_clean.max():.2f}] m/s²")

# 4. Sauvegarde (même format que les fichiers d'origine, gzip car l'extension est .gz)
np.savetxt(HERE / "bdd_in_clean.csv.gz", U_clean, delimiter=",", fmt="%.15g")
np.savetxt(HERE / "bdd_out_clean.csv.gz", Y_clean, delimiter=",", fmt="%.15g")
print(f"Écrit : {HERE / 'bdd_in_clean.csv.gz'}, {HERE / 'bdd_out_clean.csv.gz'}")

# 5. Figure : un essai normal vs les essais retirés
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
ax1.plot(Y_clean[0], label="az (essai normal)")
ax1.plot(U_clean[0] * 10, label="u x 10")
ax1.set_ylabel("az [m/s²]")
ax1.legend()
for i in np.flatnonzero(bad):
    ax2.plot(Y[i], label=f"essai {i}")
ax2.set_ylabel("az [m/s²]")
ax2.set_xlabel("pas k")
ax2.set_title("Essais retirés")
ax2.legend()
fig.tight_layout()
fig.savefig(HERE / "cleaning.png", dpi=120)
print(f"Écrit : {HERE / 'cleaning.png'}")
