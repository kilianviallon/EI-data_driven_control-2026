"""
Analyse de l'étude d'ablation (lancée automatiquement à la fin de run_all.py) :

  figures/1_apprentissage.png   RMSE sur la mission au cours de l'entraînement : chaque amélioration face au départ
  figures/2_vols.png            vol sur la mission de la Figure 1 (z) : chaque amélioration face au départ
  figures/3_commandes.png       commande u des mêmes vols
  figures/4_bilan.png           mission, missions aléatoires, bruit + biais : toutes les variantes et le PID
  resultats.md                  tableau des métriques : médiane (min - max) sur les graines

Le vol montré pour une variante est celui de la graine médiane (ni la meilleure, ni la pire).
Un PID réglé sur le même RNN sert de référence (le sujet demande une comparaison à une loi linéaire).
"""
import os
import glob
import json

import numpy as np
import matplotlib.pyplot as plt

from drone_env_ablation import HERE, VARIANTES, TS, make_env
from train_variant import voler, rmse, N_MISSIONS_TEST, N_BRUIT_TEST

DESCRIPTION = {
    'base':  'modèle de départ (Env_drone.py)',
    'A':     'action limitée à [0 ; 0,8]',
    'B':     'nouvelle récompense + fin du vol hors zone',
    'C':     "intégrale de l'erreur dans l'observation",
    'D':     'missions aléatoires + bruit + biais',
    'E':     "entropie de SAC fixée à 0,02",
    'ABCDE': 'toutes les améliorations',
    'AC':      'action [0 ; 0,8] + intégrale (100k pas)',
    'AC_200k': 'action [0 ; 0,8] + intégrale (200k pas)',
}
AMELIORATIONS = ['A', 'B', 'C', 'D', 'E', 'ABCDE']

# Couleurs : départ en gris, une amélioration en bleu, toutes ensemble en orange, PID en pointillés
GRIS, BLEU, ORANGE = '#898781', '#2a78d6', '#eb6834'
ENCRE, ENCRE_2, GRILLE, AXE, FOND = '#0b0b0b', '#52514e', '#e1e0d9', '#c3c2b7', '#fcfcfb'
couleur = lambda v: GRIS if v == 'base' else ORANGE if v == 'ABCDE' else BLEU
os.makedirs(os.path.join(HERE, 'figures'), exist_ok=True)


def fin_palier(t):
    """Dernière seconde de chaque palier de la mission de la Figure 1 (le drone devrait y être immobile)."""
    return ((t >= 2) & (t < 3)) | ((t >= 6) & (t < 7)) | (t >= 9)


def metriques_vol(vol):
    """Mêmes métriques que train_variant.py, pour un vol sur la mission de la Figure 1."""
    t, e = vol['t'], np.abs(vol['z_ref'] - vol['z'])
    return dict(rmse=rmse(vol), erreur_statique=float(e[fin_palier(t)].mean()), z_min=float(vol['z'].min()),
                variation_u=float(np.abs(np.diff(vol['u'])).mean()), vitesse_residuelle=vitesse_residuelle(vol))


def vitesse_residuelle(vol):
    """|vz| moyen en fin de palier [m/s] : mesure des oscillations (0 pour un drone stabilisé)."""
    return float(np.abs(vol['vz'][fin_palier(vol['t'])]).mean())


# --- Résultats de chaque entraînement ---
runs = {}
for v in VARIANTES:
    for dossier in sorted(glob.glob(os.path.join(HERE, 'runs', f'{v}_s*'))):
        if not os.path.isfile(os.path.join(dossier, 'metriques.json')):
            continue
        with open(os.path.join(dossier, 'metriques.json')) as f:
            m = json.load(f)
        with open(os.path.join(dossier, 'evaluations.json')) as f:
            m['evaluations'] = json.load(f)
        m['vol'] = dict(np.load(os.path.join(dossier, 'vol_mission.npz')))
        m['vitesse_residuelle'] = vitesse_residuelle(m['vol'])
        runs.setdefault(v, []).append(m)
variantes = [v for v in VARIANTES if v in runs]
median_run = {v: sorted(runs[v], key=lambda m: m['rmse'])[len(runs[v]) // 2] for v in variantes}


# --- Référence : PID réglé sur le RNN (grille de gains, RMSE le plus bas sur la mission de la Figure 1) ---
U_EQ = 0.386                                    # commande d'équilibre du RNN (az = 0)

def voler_pid(gains, env, seed=None):
    """u = u_eq + Kp.e + Ki.intégrale(e) - Kd.vz, calculé sur les mesures (bruitées avec 'robust')."""
    kp, ki, kd = gains
    _, info = env.reset(seed=seed)
    I, vol, fini = 0.0, {k: [] for k in ('t', 'z', 'z_ref', 'vz', 'u')}, False
    while not fini:
        m = info['m']
        e = m['z_ref'] - m['z']
        I = float(np.clip(I + e * TS, -2.0, 2.0))
        u = float(np.clip(U_EQ + kp * e + ki * I - kd * m['vz'], -1.0, 1.0))
        for k, x in (('t', env.t), ('z', env.z), ('z_ref', env.mission(env.t)), ('vz', env.vz), ('u', u)):
            vol[k].append(x)
        _, _, terminated, truncated, info = env.step(np.array([u], dtype=np.float32))
        fini = terminated or truncated
    return {k: np.array(x) for k, x in vol.items()}

env_pid = make_env('base', 'eval')
grille = [(kp, ki, kd) for kp in np.geomspace(0.05, 1.0, 12) for ki in (0.0, 0.05, 0.1, 0.2)
          for kd in np.geomspace(0.02, 0.5, 12)]
gains = min(grille, key=lambda g: rmse(voler_pid(g, env_pid)))
vol_pid = voler_pid(gains, env_pid)
pid = dict(metriques_vol(vol_pid),
           rmse_missions_aleatoires=float(np.mean([rmse(voler_pid(gains, make_env('base', 'random'), 1000 + i))
                                                   for i in range(N_MISSIONS_TEST)])),
           rmse_bruit=float(np.mean([rmse(voler_pid(gains, make_env('base', 'robust'), 2000 + i))
                                     for i in range(N_BRUIT_TEST)])))
print(f"PID de référence : Kp = {gains[0]:.3f}, Ki = {gains[1]:.2f}, Kd = {gains[2]:.3f}  ->  RMSE mission {pid['rmse']:.3f} m")


# --- Mise en forme commune ---
def habiller(ax):
    ax.set_facecolor(FOND)
    ax.grid(True, color=GRILLE, linewidth=0.8)
    ax.set_axisbelow(True)
    for cote in ('top', 'right'):
        ax.spines[cote].set_visible(False)
    for cote in ('left', 'bottom'):
        ax.spines[cote].set_color(AXE)
    ax.tick_params(colors=ENCRE_2, labelsize=8)
    ax.xaxis.label.set_color(ENCRE_2)
    ax.yaxis.label.set_color(ENCRE_2)


def grille_ameliorations(titre):
    fig, axes = plt.subplots(2, 3, figsize=(15, 8), sharex=True, sharey=True, facecolor=FOND)
    fig.suptitle(titre, x=0.01, ha='left', fontsize=13, color=ENCRE)
    return fig, axes.ravel()


def legende(fig, handles):
    fig.legend(handles=handles, loc='lower center', ncol=len(handles), frameon=False, fontsize=9)
    fig.tight_layout(rect=(0, 0.04, 1, 0.97))


def sauver(fig, nom):
    fig.savefig(os.path.join(HERE, 'figures', nom), dpi=130, facecolor=FOND)
    plt.close(fig)


def courbe_apprentissage(v):
    """Pas d'évaluation, RMSE médian, min et max sur les graines (graines terminées seulement)."""
    n = min(len(m['evaluations']) for m in runs[v])
    pas = np.array([h['pas'] for h in runs[v][0]['evaluations'][:n]])
    R = np.array([[h['rmse'] for h in m['evaluations'][:n]] for m in runs[v]])
    return pas / 1000, np.median(R, axis=0), R.min(axis=0), R.max(axis=0)


# --- Figure 1 : apprentissage ---
fig, axes = grille_ameliorations("Apprentissage : RMSE sur la mission de la Figure 1 (médiane et étendue sur les graines)")
for ax, v in zip(axes, AMELIORATIONS):
    for w in ('base', v):
        if w not in runs:
            continue
        x, med, lo, hi = courbe_apprentissage(w)
        ax.fill_between(x, lo, hi, color=couleur(w), alpha=0.15, linewidth=0)
        ax.plot(x, med, color=couleur(w), linewidth=2.2 if w != 'base' else 1.6)
    ax.axhline(pid['rmse'], color=ENCRE_2, linestyle=':', linewidth=1.2)
    ax.set_yscale('log')
    ax.set_yticks([0.1, 0.2, 0.5, 1, 2, 5, 10, 20], ['0.1', '0.2', '0.5', '1', '2', '5', '10', '20'])
    ax.minorticks_off()
    ax.set_title(f"{v} : {DESCRIPTION[v]}", loc='left', fontsize=10, color=ENCRE)
    habiller(ax)
for ax in axes[3:]:
    ax.set_xlabel("pas d'entraînement [milliers]")
for ax in axes[::3]:
    ax.set_ylabel('RMSE [m]')
legende(fig, [plt.Line2D([], [], color=GRIS, lw=1.6, label='départ'),
              plt.Line2D([], [], color=BLEU, lw=2.2, label='une amélioration'),
              plt.Line2D([], [], color=ORANGE, lw=2.2, label='toutes les améliorations'),
              plt.Line2D([], [], color=ENCRE_2, ls=':', lw=1.2, label='PID de référence')])
sauver(fig, '1_apprentissage.png')


# --- Figures 2 et 3 : vols sur la mission (graine médiane) ---
for nom, cle, ylabel, titre in (('2_vols.png', 'z', 'z [m]', "Vol sur la mission de la Figure 1 (graine médiane)"),
                                ('3_commandes.png', 'u', 'u [-]', "Commande pendant le vol (graine médiane)")):
    fig, axes = grille_ameliorations(titre)
    for ax, v in zip(axes, AMELIORATIONS):
        if cle == 'z':
            vol = median_run['base']['vol'] if 'base' in runs else vol_pid
            ax.plot(vol['t'], vol['z_ref'], color=ENCRE, linestyle='--', linewidth=1.2)
            ax.axhline(-0.25, color=AXE, linestyle=':', linewidth=1)
        ax.plot(vol_pid['t'], vol_pid[cle], color=ENCRE_2, linestyle=':', linewidth=1.2)
        for w in ('base', v):
            if w not in runs:
                continue
            vol = median_run[w]['vol']
            epaisseur = 1.5 if w == 'base' else 2
            if cle == 'u':                      # commande tenue pendant chaque pas
                ax.step(vol['t'], vol['u'], where='post', color=couleur(w), linewidth=epaisseur)
            else:
                ax.plot(vol['t'], vol['z'], color=couleur(w), linewidth=epaisseur)
        rm = median_run[v]['rmse'] if v in runs else float('nan')
        ax.set_title(f"{v} : {DESCRIPTION[v]}  (RMSE {rm:.2f} m)", loc='left', fontsize=10, color=ENCRE)
        habiller(ax)
    for ax in axes[3:]:
        ax.set_xlabel('t [s]')
    for ax in axes[::3]:
        ax.set_ylabel(ylabel)
    poignees = [plt.Line2D([], [], color=ENCRE, ls='--', lw=1.2, label='consigne')] if cle == 'z' else []
    legende(fig, poignees + [plt.Line2D([], [], color=GRIS, lw=1.5, label='départ'),
                             plt.Line2D([], [], color=BLEU, lw=2, label='une amélioration'),
                             plt.Line2D([], [], color=ORANGE, lw=2, label='toutes les améliorations'),
                             plt.Line2D([], [], color=ENCRE_2, ls=':', lw=1.2, label='PID de référence')])
    sauver(fig, nom)


# --- Figure 4 : bilan des trois tests ---
tests = [('rmse', 'Mission de la Figure 1'), ('rmse_missions_aleatoires', f'{N_MISSIONS_TEST} missions aléatoires'),
         ('rmse_bruit', f'Mission + bruit de mesure + biais ({N_BRUIT_TEST} vols)')]
fig, axes = plt.subplots(1, 3, figsize=(15, 4.8), sharey=True, facecolor=FOND)
fig.suptitle("Bilan : RMSE du meilleur modèle (barre : médiane, trait : min - max sur les graines)",
             x=0.01, ha='left', fontsize=13, color=ENCRE)
y = np.arange(len(variantes))[::-1]
for ax, (cle, titre) in zip(axes, tests):
    vals = [np.array([m[cle] for m in runs[v]]) for v in variantes]
    med = np.array([np.median(x) for x in vals])
    ax.barh(y, med, height=0.6, color=[couleur(v) for v in variantes])
    ax.errorbar(med, y, xerr=[med - [x.min() for x in vals], [x.max() for x in vals] - med],
                fmt='none', ecolor=ENCRE, elinewidth=1, capsize=3)
    for yi, mi, x in zip(y, med, vals):
        ax.annotate(f"{mi:.2f}", (x.max(), yi), xytext=(4, 0), textcoords='offset points', va='center',
                    fontsize=8, color=ENCRE_2)
    ax.axvline(pid[cle], color=ENCRE_2, linestyle=':', linewidth=1.2)
    ax.annotate('PID', (pid[cle], y[0] + 0.55), xytext=(3, 0), textcoords='offset points', fontsize=8, color=ENCRE_2)
    ax.set_title(titre, loc='left', fontsize=10, color=ENCRE)
    ax.set_xlabel('RMSE [m]')
    ax.set_xlim(0, max(x.max() for x in vals) * 1.18)
    habiller(ax)
    ax.grid(False, axis='y')
axes[0].set_yticks(y, variantes)
fig.tight_layout(rect=(0, 0, 1, 0.94))
sauver(fig, '4_bilan.png')


# --- Figures de présentation : chaque amélioration face au départ, puis toutes ensemble (une figure 16:9 par fichier) ---
CHANGEMENT = {
    'A':     "Action u limitée à [0 ; 0,8] au lieu de [-1 ; 1] : le centre (0,4) est presque la commande d'équilibre (0,386).",
    'B':     "Récompense  -|e| - 0,1·vz² - 0,5·(u - u_prev)² + 0,5 si |e| < 5 cm,  et fin du vol si z < -0,30 m ou z > 1,5 m.",
    'C':     "Intégrale de l'erreur ajoutée à l'observation de l'agent (le « I » d'un correcteur PI).",
    'D':     "Entraînement sur des missions aléatoires, avec bruit de mesure et biais sur az (écart modèle / drone).",
    'E':     "Coefficient d'entropie de SAC fixé à 0,02 au lieu du réglage automatique (moins d'exploration aléatoire).",
    'ABCDE': "A + B + C + D + E dans le même entraînement.",
}
NOMS_FICHIERS = {'A': '1_A_action', 'B': '2_B_recompense', 'C': '3_C_integrale', 'D': '4_D_robustesse',
                 'E': '5_E_entropie', 'ABCDE': '6_ABCDE_toutes'}
METRIQUES = [('rmse', 'RMSE sur la mission [m]'), ('erreur_statique', 'Erreur en fin de palier [m]'),
             ('vitesse_residuelle', 'Oscillation : |vz| en fin de palier [m/s]'),
             ('rmse_missions_aleatoires', 'RMSE missions aléatoires [m]'),
             ('rmse_modele_final', 'RMSE du modèle final [m]'), ('variation_u', 'Variation de u par pas')]
dossier_pres = os.path.join(HERE, 'figures', 'presentation')
os.makedirs(dossier_pres, exist_ok=True)

for v in AMELIORATIONS:
    if v not in runs or 'base' not in runs:
        continue
    final = v == 'ABCDE'
    fig = plt.figure(figsize=(13.33, 7.5), facecolor=FOND)
    gs = fig.add_gridspec(2, 2, width_ratios=[1.55, 1], height_ratios=[1.7, 1], hspace=0.38, wspace=0.18,
                          left=0.06, right=0.98, top=0.84, bottom=0.08)
    ax_z, ax_u = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[1, 0])
    ax_a, ax_t = fig.add_subplot(gs[0, 1]), fig.add_subplot(gs[1, 1])
    titre = 'Toutes les améliorations' if final else f'Amélioration {v} : {DESCRIPTION[v]}'
    fig.text(0.06, 0.94, titre, fontsize=17, color=ENCRE, weight='bold')
    fig.text(0.06, 0.895, CHANGEMENT[v], fontsize=11, color=ENCRE_2)

    # Vol et commande : meilleur modèle de la graine médiane
    vb, vv = median_run['base']['vol'], median_run[v]['vol']
    ax_z.plot(vb['t'], vb['z_ref'], color=ENCRE, linestyle='--', linewidth=1.3, label='consigne')
    ax_z.axhline(-0.25, color=AXE, linestyle=':', linewidth=1.2)
    ax_z.annotate('sol', (2.0, -0.25), xytext=(0, 3), textcoords='offset points', ha='center', fontsize=8, color=ENCRE_2)
    ax_z.plot(vb['t'], vb['z'], color=GRIS, linewidth=1.8, label=f"départ (RMSE {median_run['base']['rmse']:.2f} m)")
    ax_z.plot(vv['t'], vv['z'], color=couleur(v), linewidth=2.4, label=f"{v} (RMSE {median_run[v]['rmse']:.2f} m)")
    ax_u.step(vb['t'], vb['u'], where='post', color=GRIS, linewidth=1.5)
    ax_u.step(vv['t'], vv['u'], where='post', color=couleur(v), linewidth=2)
    if final:
        ax_z.plot(vol_pid['t'], vol_pid['z'], color=ENCRE_2, linestyle=':', linewidth=1.4, label=f"PID (RMSE {pid['rmse']:.2f} m)")
        ax_u.step(vol_pid['t'], vol_pid['u'], where='post', color=ENCRE_2, linestyle=':', linewidth=1.2)
    ax_z.set_title('Vol sur la mission de la Figure 1 (meilleur modèle, graine médiane)', loc='left', fontsize=11, color=ENCRE)
    ax_z.set_ylabel('z [m]')
    ax_z.legend(frameon=False, fontsize=9, loc='upper right')
    ax_u.set_title('Commande envoyée aux moteurs', loc='left', fontsize=11, color=ENCRE)
    ax_u.set_ylabel('u [-]')
    ax_u.set_xlabel('t [s]')

    # Apprentissage : médiane et étendue sur les graines
    for w in ('base', v):
        x, med, lo, hi = courbe_apprentissage(w)
        ax_a.fill_between(x, lo, hi, color=couleur(w), alpha=0.15, linewidth=0)
        ax_a.plot(x, med, color=couleur(w), linewidth=2.2 if w != 'base' else 1.6)
    if final:
        ax_a.axhline(pid['rmse'], color=ENCRE_2, linestyle=':', linewidth=1.2)
    ax_a.set_yscale('log')
    ax_a.set_yticks([0.1, 0.2, 0.5, 1, 2, 5, 10, 20], ['0.1', '0.2', '0.5', '1', '2', '5', '10', '20'])
    ax_a.minorticks_off()
    ax_a.set_title(f"Apprentissage : RMSE sur la mission ({len(runs[v])} graines)", loc='left', fontsize=11, color=ENCRE)
    ax_a.set_xlabel("pas d'entraînement [milliers]")
    ax_a.set_ylabel('RMSE [m]')
    for ax in (ax_z, ax_u, ax_a):
        habiller(ax)

    # Tableau des chiffres clés (médiane sur les graines) : le meilleur de chaque ligne en gras
    colonnes = ['départ', v] + (['PID'] if final else [])
    valeurs = [[np.median([m[cle] for m in runs['base']]), np.median([m[cle] for m in runs[v]])]
               + ([pid[cle]] if final and cle in pid else [np.nan] if final else []) for cle, _ in METRIQUES]

    def case(w, cle):                       # médiane, puis étendue min - max sur les graines
        x = np.array([m[cle] for m in runs[w]])
        return f"{np.median(x):.3f}\n({x.min():.2f} - {x.max():.2f})"
    texte = [[nom, case('base', cle), case(v, cle)]
             + ([f"{pid[cle]:.3f}" if cle in pid else '-'] if final else []) for cle, nom in METRIQUES]
    ax_t.axis('off')
    table = ax_t.table(cellText=texte, colLabels=[''] + colonnes, cellLoc='center', bbox=[0, 0, 1, 1],
                       colWidths=[0.44, 0.21, 0.21, 0.14] if final else [0.44, 0.28, 0.28])
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    for (r, c), cell in table.get_celld().items():
        cell.set_edgecolor(GRILLE)
        cell.set_facecolor(FOND)
        if c == 0:                          # nom de la métrique
            cell.set_text_props(ha='left', color=ENCRE)
            cell.PAD = 0.03
        elif r == 0:                        # en-tête : couleur de la variante
            cell.set_text_props(weight='bold', color=[GRIS, couleur(v), ENCRE_2][c - 1])
        elif np.nanargmin(valeurs[r - 1]) == c - 1:
            cell.set_text_props(weight='bold', color=ENCRE)
        else:
            cell.set_text_props(color=ENCRE_2)
    ax_t.set_title('Chiffres clés : médiane (min - max), plus bas = mieux', loc='left', fontsize=11, color=ENCRE)

    fig.savefig(os.path.join(dossier_pres, NOMS_FICHIERS[v] + '.png'), dpi=150, facecolor=FOND)
    plt.close(fig)


# --- Tableau des résultats ---
def cellule(v, cle, fmt='{:.3f}'):
    x = np.array([m[cle] for m in runs[v]])
    if len(x) == 1:
        return fmt.format(x[0])
    return f"{fmt.format(np.median(x))} ({fmt.format(x.min())} - {fmt.format(x.max())})"

lignes = ["# Résultats de l'étude d'ablation", "",
          f"Médiane (min - max) sur les graines ; meilleur modèle de chaque entraînement. "
          f"PID : Kp = {gains[0]:.3f}, Ki = {gains[1]:.2f}, Kd = {gains[2]:.3f}.", "",
          "| Variante | Changement | Graines | RMSE mission [m] | Erreur en fin de palier [m] | "
          "Oscillation : abs(vz) en fin de palier [m/s] | z min [m] | "
          "Variation de u par pas | RMSE missions aléatoires [m] | RMSE bruit + biais [m] | "
          "Meilleur modèle à [k pas] | RMSE modèle final [m] |",
          "|---|---|---|---|---|---|---|---|---|---|---|---|"]
for v in variantes:
    lignes.append(f"| **{v}** | {DESCRIPTION[v]} | {len(runs[v])} | {cellule(v, 'rmse')} | {cellule(v, 'erreur_statique')} | "
                  f"{cellule(v, 'vitesse_residuelle')} | {cellule(v, 'z_min', '{:.2f}')} | {cellule(v, 'variation_u')} | "
                  f"{cellule(v, 'rmse_missions_aleatoires')} | {cellule(v, 'rmse_bruit')} | "
                  f"{' / '.join(str(m['pas_meilleur'] // 1000) for m in runs[v])} | "
                  f"{cellule(v, 'rmse_modele_final')} |")
lignes.append(f"| PID | référence linéaire | - | {pid['rmse']:.3f} | {pid['erreur_statique']:.3f} | "
              f"{pid['vitesse_residuelle']:.3f} | {pid['z_min']:.2f} | "
              f"{pid['variation_u']:.3f} | {pid['rmse_missions_aleatoires']:.3f} | {pid['rmse_bruit']:.3f} | - | - |")
with open(os.path.join(HERE, 'resultats.md'), 'w', encoding='utf-8') as f:
    f.write('\n'.join(lignes) + '\n')
print('\n'.join(lignes))
print("\nFigures écrites dans figures/, tableau dans resultats.md")
