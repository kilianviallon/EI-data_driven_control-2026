# Bases de données du tangage : u_theta → ax et u_theta → a_theta

Données générées avec le simulateur `UAV_model.slx` pour identifier la dynamique en x et en theta
(tangage), comme `bdd_in_mat_05` / `bdd_out_mat_05` l'ont été pour z.

| Fichier | Contenu |
|---|---|
| `bdd_in_utheta.csv.gz` | **Entrée** u_theta : commande différentielle de tangage appliquée aux moteurs |
| `bdd_out_ax.csv.gz` | **Sortie** ax : accélération sur x [m/s²] (sortie 13 du simulateur) |
| `bdd_out_atheta.csv.gz` | **Sortie** a_theta = w_dot_y : accélération angulaire de tangage [rad/s²] (sortie 17) |
| `generer_bdd_x_theta.m` | Script MATLAB de génération (construit `boucle_tangage.slx`, lance les essais, écrit les fichiers) |
| `boucle_tangage.slx` | Modèle Simulink de génération (ouvrable pour voir la boucle) |
| `essai_exemple.png` | Un essai : u_theta, theta estimé, a_theta, ax |

Format identique à la base fournie : **1 ligne = 1 essai de 100 s, 1 colonne = 1 pas** (Ts = 0,05 s),
3000 essais × 2000 pas, 40 pas au repos au début de chaque essai, bruit des capteurs `g_noise = 0.01`.
Les fichiers `.csv.gz` se lisent directement avec `pandas.read_csv` ou `np.loadtxt`.

## Comment les données sont produites

Le drone est en configuration « + » : les moteurs 1 (avant) et 3 (arrière) font tanguer le drone autour de y,
les moteurs 2 et 4 le font rouler autour de x. Une commande de tangage pure s'écrit :

```
u1 = u0 - u_theta      u3 = u0 + u_theta      u2 = u4 = u0 + u_psi
```

Trois contraintes physiques, découvertes en explorant le simulateur, imposent la façon de générer :

1. **Poussée collective à l'équilibre (u0 = 0,386), pas en chute libre.** ax ≈ (poussée / m)·sin(theta) :
   en chute libre il n'y a pas de poussée à incliner, ax resterait nul.
2. **Theta stabilisé par un PD pendant la génération.** Le tangage est un double intégrateur rapide
   (environ 170 rad/s² par unité de u_theta) : en boucle ouverte, le drone se retourne en moins d'une seconde.
   `u_theta = excitation - Kp·theta_est - Kd·wy`, avec une excitation aléatoire en paliers (5 à 39 pas,
   comme la base fournie). On enregistre le u_theta **réellement appliqué** (PD + excitation).
   Theta est estimé en intégrant le gyromètre wy : avec le bruit activé, la sortie theta du simulateur a un
   écart-type de 3,5 rad (capteur inutilisable), contre 0,0005 pour wy, ax et a_theta.
3. **Lacet tenu à 0 par un second PD** sur les moteurs 2 et 4 (`u_psi = Kpsi·psi_est + Kdpsi·wz`). Les couples
   de lacet des moteurs 1 et 3 ne se compensent qu'au premier ordre : sans correction, le drone tournait de
   105° en 100 s et l'accélération passait de x vers y.

## Ce que contiennent les données

| Grandeur | Min | Max | Écart-type |
|---|---|---|---|
| u_theta | -0.021 | +0.021 | 0.0039 |
| ax [m/s²] | -4.35 | +4.47 | 1.76 |
| a_theta [rad/s²] | -2.58 | +2.56 | 0.54 |

Theta reste dans ±0,47 rad (27°), le lacet dans ±0,001 rad. Vérifications physiques :
- a_theta suit u_theta avec environ 3 pas de retard (corrélation 0,96), comme az suit u ;
- ax suit theta (reconstruit en intégrant deux fois a_theta) : corrélation 0,985, pente 9,45 m/s² par rad
  (proche de g = 9,81 : ax ≈ g·theta pour de petits angles).

## Pour entraîner un RNN (`identification_RNN/RNN_drone.py`)
Changer les fichiers et la normalisation :

```python
csv_u = '../generation_donnees_x_theta/bdd_in_utheta.csv.gz'
csv_y = '../generation_donnees_x_theta/bdd_out_atheta.csv.gz'   # ou bdd_out_ax.csv.gz
u_scale = 0.02          # u_theta dans environ [-1, 1]
y_offset = 0.0          # pas de décalage : a_theta et ax valent 0 au repos
y_scale = 2.6           # 4.5 pour ax
```

- **a_theta** dépend directement de u_theta (dynamique des moteurs) : même type de modèle que az.
- **ax** dépend de theta, la **double intégrale** de a_theta : un RNN u_theta → ax doit apprendre cette double
  intégration lui-même, ce qui est plus difficile. Alternative : enchaîner le RNN de a_theta, une double
  intégration, puis ax ≈ 9,45·theta.

## Régénérer
Lancer `generer_bdd_x_theta.m` dans MATLAB (environ 2 min 30). Paramètres en tête de script : nombre d'essais,
gains des PD, amplitude de l'excitation (`A_EXC`, theta d'environ ±A_EXC/Kp).
