import os
import numpy as np

# Poids du RNN identifié (u -> az) : version_3, le meilleur en validation (voir identification_RNN/README.md)
DOSSIER = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "..", "identification_RNN", "model_rnn", "lightning_logs", "version_3", "export_NN")

# Récupérer les paramètres du RNN
def lire_txt(chemin):
    with open(chemin) as f:
        texte = f.read().strip().strip("[]")      # enlève les crochets
    lignes = texte.split(";")                      # coupe aux ";"
    return np.array([[float(x) for x in ligne.split()] for ligne in lignes])

# créer une class RNN pour récupérer l'accélération

class ModeleRNN:
    # implémenter les paramètres du RNN
    def __init__(self, dossier=DOSSIER):
        self.Wih = lire_txt(os.path.join(dossier, "weight_ih_l0.txt")).reshape(-1)   # 16 nombres
        self.Whh = lire_txt(os.path.join(dossier, "weight_hh_l0.txt"))               # matrice 16 x 16
        self.bh  = lire_txt(os.path.join(dossier, "bias_h.txt")).reshape(-1)         # 16 nombres
        self.Wo  = lire_txt(os.path.join(dossier, "head_weight.txt")).reshape(-1)    # 16 nombres
        self.bo  = lire_txt(os.path.join(dossier, "head_bias.txt"))[0, 0]            # 1 nombre
        self.h = np.zeros(16)

    # reset la mémoire de l'élement caché
    def reset(self):
        self.h = np.zeros(16)          # on vide la mémoire au début de chaque vol

    # recalculer l'accélération grace au RNN
    def step(self, u):
        self.h = np.tanh(self.Wih * u + self.Whh @ self.h + self.bh)    # équation 1
        return 20 * (self.Wo @ self.h + self.bo) - 9.81
