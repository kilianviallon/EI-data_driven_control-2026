"""
Modèle linéaire (ARX) du drone : commande u  ->  accélération verticale az.

Données : data_base_cleaning/bdd_{in,out}_clean.csv.gz  (1 ligne = 1 essai de 2000 pas, Ts = 0.05 s)
Sorties : lineaire/figures/*.png   et   lineaire/arx_model.npz (modèle retenu, réutilisable)

Méthode (une fonction par étape) :
  0. Chargement + caractéristique statique u -> az
  1. Découpage train / validation / test PAR ESSAIS (jamais par lignes au hasard)
  2. Régresseurs ARX : phi(k) = [az(k-1)..az(k-na), u(k-nk)..u(k-nk-nb+1), 1]
  3. Moindres carrés
  4. Simulation libre + indicateurs (fit %, RMSE)
  5. Choix des ordres (na, nb, nk) sur la validation
  6. Vérifications : pôles, gain statique, résidus
  7. Comparaison par régime (proche / loin du stationnaire), ARX global vs ARX local
     -> même protocole à appliquer au NN : il suffit de fournir une fonction predict(phi)
  8. Modèle d'état autour du stationnaire (pour un LQ), avec vz et z

Lancement : python lineaire/main.py
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).parent
DATA = HERE.parent / "data_base_cleaning"
FIG = HERE / "figures"
TS = 0.05
SEED = 0

# Bande "proche du vol stationnaire" : |u - u_eq| < BAND
BAND = 0.2
# Sous-ensembles pour la recherche des ordres (plus rapide, largement suffisant)
N_SEL_TRAIN, N_SEL_VAL = 300, 150


# --------------------------------------------------------------------------------------
# Étape 0 : chargement + caractéristique statique
# --------------------------------------------------------------------------------------
def load_data():
    U = np.loadtxt(DATA / "bdd_in_clean.csv.gz", delimiter=",")
    Y = np.loadtxt(DATA / "bdd_out_clean.csv.gz", delimiter=",")
    assert U.shape == Y.shape
    print(f"[0] {U.shape[0]} essais x {U.shape[1]} pas  (durée d'un essai : {U.shape[1] * TS:.0f} s)")
    return U, Y


def static_curve(U, Y, min_len=30):
    """Valeur finale de az sur chaque palier de u assez long -> points (u, az_stationnaire)."""
    us, ys = [], []
    for u, y in zip(U[::5], Y[::5]):
        cuts = np.r_[0, np.flatnonzero(np.diff(u) != 0) + 1, len(u)]
        for a, b in zip(cuts[:-1], cuts[1:]):
            if b - a >= min_len:
                us.append(u[a])
                ys.append(y[b - 5:b].mean())
    us, ys = np.array(us), np.array(ys)
    p = np.polyfit(us, ys, 5)
    grid = np.linspace(-1, 1, 2001)
    u_eq = grid[np.argmin(np.abs(np.polyval(p, grid)))]
    slope_eq = np.polyval(np.polyder(p), u_eq)
    slope_mean = (np.polyval(p, 1) - np.polyval(p, -1)) / 2
    print(f"[0] Vol stationnaire (az = 0) pour u_eq = {u_eq:.3f} ; "
          f"pente locale {slope_eq:.1f} vs pente moyenne {slope_mean:.1f} m/s² par unité de u")
    return us, ys, p, u_eq


# --------------------------------------------------------------------------------------
# Étape 1 : découpage par essais
# --------------------------------------------------------------------------------------
def split(U, Y, frac=(0.6, 0.2, 0.2)):
    idx = np.random.default_rng(SEED).permutation(len(U))
    n_tr, n_va = int(frac[0] * len(U)), int(frac[1] * len(U))
    parts = idx[:n_tr], idx[n_tr:n_tr + n_va], idx[n_tr + n_va:]
    print(f"[1] train {len(parts[0])} / validation {len(parts[1])} / test {len(parts[2])} essais")
    return [(U[p], Y[p]) for p in parts]


# --------------------------------------------------------------------------------------
# Étape 2 : régresseurs
# --------------------------------------------------------------------------------------
def n_init(na, nb, nk):
    """Nombre de pas d'historique nécessaires avant la première prédiction."""
    return max(na, nk + nb - 1)


def regressors(U, Y, na, nb, nk):
    """U, Y : (n_essais, T). Renvoie Phi (n_lignes, na+nb+1) et la cible az(k).

    Construit essai par essai (axe 0), donc aucun régresseur ne mélange deux essais.
    """
    T, n0 = Y.shape[1], n_init(na, nb, nk)
    cols = [Y[:, n0 - i:T - i] for i in range(1, na + 1)]          # az(k-1) ... az(k-na)
    cols += [U[:, n0 - nk - j:T - nk - j] for j in range(nb)]       # u(k-nk) ... u(k-nk-nb+1)
    cols.append(np.ones_like(Y[:, n0:]))                            # constante (absorbe la gravité)
    Phi = np.stack(cols, axis=-1).reshape(-1, na + nb + 1)
    return Phi, Y[:, n0:].reshape(-1)


# --------------------------------------------------------------------------------------
# Étape 3 : moindres carrés
# --------------------------------------------------------------------------------------
def fit_arx(U, Y, na, nb, nk, lam=1e-8, row_mask=None, chunk=100):
    """theta = (Phi^T Phi + lam I)^-1 Phi^T y, accumulé par paquets d'essais (mémoire limitée).

    row_mask(U_chunk) -> booléen (n_essais, T - n0) : lignes à garder (pour le modèle local).
    """
    d = na + nb + 1
    A, b = np.zeros((d, d)), np.zeros(d)
    for s in range(0, len(U), chunk):
        Phi, y = regressors(U[s:s + chunk], Y[s:s + chunk], na, nb, nk)
        if row_mask is not None:
            keep = row_mask(U[s:s + chunk], n_init(na, nb, nk)).reshape(-1)
            Phi, y = Phi[keep], y[keep]
        A += Phi.T @ Phi
        b += Phi.T @ y
    return np.linalg.solve(A + lam * np.eye(d), b)


def arx_predictor(theta):
    """Interface commune ARX / NN : predict(phi) avec phi de forme (n_essais, na+nb+1)."""
    return lambda phi: phi @ theta


# --------------------------------------------------------------------------------------
# Étape 4 : simulation libre + indicateurs
# --------------------------------------------------------------------------------------
def simulate(predict, U, Y, na, nb, nk):
    """Simulation libre sur tout l'essai : initialisée avec les n0 premières vraies valeurs,
    puis le modèle ne réutilise QUE ses propres prédictions (comme dans l'environnement RL)."""
    T, n0 = Y.shape[1], n_init(na, nb, nk)
    Yh = Y.copy()
    ones = np.ones((len(Y), 1))
    for k in range(n0, T):
        phi = np.hstack([Yh[:, k - na:k][:, ::-1],                     # az^(k-1) ... az^(k-na)
                         U[:, k - nk - nb + 1:k - nk + 1][:, ::-1],     # u(k-nk) ... u(k-nk-nb+1)
                         ones])
        Yh[:, k] = predict(phi)
    return Yh


def fit_percent(y, yh):
    return 100 * (1 - np.linalg.norm(y - yh) / np.linalg.norm(y - y.mean()))


def rmse(y, yh):
    return np.sqrt(np.mean((y - yh) ** 2))


def evaluate(predict, U, Y, na, nb, nk):
    n0 = n_init(na, nb, nk)
    Phi, y = regressors(U, Y, na, nb, nk)
    Yh = simulate(predict, U, Y, na, nb, nk)
    return {"fit_1pas": fit_percent(y, predict(Phi)),
            "fit_libre": fit_percent(Y[:, n0:], Yh[:, n0:]),
            "rmse_libre": rmse(Y[:, n0:], Yh[:, n0:])}, Yh


# --------------------------------------------------------------------------------------
# Étape 5 : choix des ordres
# --------------------------------------------------------------------------------------
def select_orders(train, val, na_max=5, nb_max=5, nk_list=(0, 1, 2)):
    (Utr, Ytr), (Uva, Yva) = train, val
    Utr, Ytr, Uva, Yva = Utr[:N_SEL_TRAIN], Ytr[:N_SEL_TRAIN], Uva[:N_SEL_VAL], Yva[:N_SEL_VAL]
    scores = {}
    for nk in nk_list:
        for na in range(1, na_max + 1):
            for nb in range(1, nb_max + 1):
                th = fit_arx(Utr, Ytr, na, nb, nk)
                s, _ = evaluate(arx_predictor(th), Uva, Yva, na, nb, nk)
                scores[(na, nb, nk)] = s["fit_libre"]
    best = max(scores, key=scores.get)
    # Parcimonie : le plus petit modèle à moins de 0.2 point du meilleur
    good = [o for o in scores if scores[o] >= scores[best] - 0.2]
    chosen = min(good, key=lambda o: (o[0] + o[1], o[2]))
    print(f"[5] meilleur fit libre {scores[best]:.2f} % pour (na, nb, nk) = {best} ; "
          f"retenu (plus simple, à < 0.2 pt) : {chosen} -> {scores[chosen]:.2f} %")
    return chosen, scores


# --------------------------------------------------------------------------------------
# Étape 6 : vérifications
# --------------------------------------------------------------------------------------
def check_model(theta, na, nb, U, Y, nk):
    a, b, c = theta[:na], theta[na:na + nb], theta[-1]
    poles = np.roots(np.r_[1, -a])
    gain = b.sum() / (1 - a.sum())
    u_eq = -c / b.sum()
    print(f"[6] pôles : {np.round(poles, 3)}  (|p| max = {np.abs(poles).max():.3f}, < 1 -> stable)")
    print(f"[6] constante de temps dominante ≈ {-TS / np.log(np.abs(poles).max()):.2f} s")
    print(f"[6] gain statique ARX = {gain:.1f} ; u_eq selon l'ARX = {u_eq:.3f}")

    # Résidus à un pas sur quelques essais de validation
    Phi, y = regressors(U[:50], Y[:50], na, nb, nk)
    e = y - Phi @ theta
    uu = U[:50, n_init(na, nb, nk):].reshape(-1)
    lags = np.arange(0, 31)
    acf = [np.corrcoef(e[l:], e[:len(e) - l])[0, 1] if l else 1.0 for l in lags]
    ccf = [np.corrcoef(e[l:], uu[:len(uu) - l])[0, 1] for l in lags]
    return poles, gain, u_eq, lags, np.array(acf), np.array(ccf)


# --------------------------------------------------------------------------------------
# Étape 7 : comparaison par régime
# --------------------------------------------------------------------------------------
def near_hover_mask(u_eq, history=10):
    """Lignes où u est resté dans la bande autour de u_eq pendant les `history` derniers pas."""
    def mask(U, n0):
        inband = np.abs(U - u_eq) < BAND
        ok = np.ones_like(inband)
        for h in range(history):
            ok &= np.roll(inband, h, axis=1)
        ok[:, :history] = False
        return ok[:, n0:]
    return mask


def by_regime(Y, Yh, U, u_eq, n0):
    near = near_hover_mask(u_eq)(U, n0)
    y, yh = Y[:, n0:], Yh[:, n0:]
    return {"proche": rmse(y[near], yh[near]), "loin": rmse(y[~near], yh[~near]),
            "part_proche": near.mean()}


# --------------------------------------------------------------------------------------
# Étape 8 : modèle d'état autour du stationnaire (pour un LQ)
# --------------------------------------------------------------------------------------
def to_state_space(theta, na, nb, nk):
    """x(k) = [az(k)..az(k-na+1), du(k-1)..du(k-m), vz(k), z(k)],  entrée du(k) = u(k) - u_eq.

    x(k+1) = A x(k) + B du(k).  Autour du stationnaire, la constante disparaît.
    """
    assert nk >= 1, "nk = 0 : transmission directe, non géré ici"
    a, b = theta[:na], theta[na:na + nb]
    m = nk + nb - 2                          # plus grand retard de du utilisé : du(k - m)
    n = na + m + 2
    A, B = np.zeros((n, n)), np.zeros(n)
    # az(k+1) = sum a_i az(k+1-i) + sum b_j du(k+1-nk-j)
    A[0, :na] = a
    for j in range(nb):
        lag = nk + j - 1                     # du(k - lag)
        if lag == 0:
            B[0] += b[j]
        else:
            A[0, na + lag - 1] += b[j]
    for i in range(1, na):                   # décalage des az
        A[i, i - 1] = 1
    if m >= 1:                               # décalage des du
        B[na] = 1
        for i in range(1, m):
            A[na + i, na + i - 1] = 1
    iv, iz = na + m, na + m + 1              # intégration (Euler) : vz puis z
    A[iv, iv], A[iv, 0] = 1, TS
    A[iz, iz], A[iz, iv] = 1, TS
    return A, B


# --------------------------------------------------------------------------------------
# Figures
# --------------------------------------------------------------------------------------
def figures(U, Y, us, ys, p, u_eq, scores, order, sims, test, acf, ccf, lags):
    FIG.mkdir(exist_ok=True)
    na, nb, nk = order
    t = np.arange(U.shape[1]) * TS

    # 1. Données + caractéristique statique
    fig, ax = plt.subplots(1, 2, figsize=(13, 4), gridspec_kw={"width_ratios": [2, 1]})
    ax[0].plot(t[:400], Y[0, :400], label="az [m/s²]")
    ax[0].plot(t[:400], 10 * U[0, :400], label="10·u")
    ax[0].set_xlabel("t [s]"); ax[0].legend(); ax[0].set_title("Un essai (20 premières s)")
    g = np.linspace(-1, 1, 200)
    ax[1].plot(us, ys, ".", ms=2, alpha=.4, label="paliers mesurés")
    ax[1].plot(g, np.polyval(p, g), "r", label="polynôme ordre 5")
    ax[1].axvline(u_eq, c="k", ls="--", lw=.8); ax[1].axhline(0, c="k", lw=.5)
    ax[1].axvspan(u_eq - BAND, u_eq + BAND, color="g", alpha=.1, label="bande stationnaire")
    ax[1].set_xlabel("u"); ax[1].set_ylabel("az stationnaire"); ax[1].legend(fontsize=8)
    ax[1].set_title("Caractéristique statique")
    fig.tight_layout(); fig.savefig(FIG / "1_donnees.png", dpi=120); plt.close(fig)

    # 2. Choix des ordres
    fig, ax = plt.subplots(figsize=(6, 4))
    for nb_ in range(1, 6):
        ax.plot(range(1, 6), [scores[(na_, nb_, nk)] for na_ in range(1, 6)], "o-", label=f"nb={nb_}")
    ax.set_xlabel("na"); ax.set_ylabel("fit simulation libre (validation) [%]")
    ax.set_title(f"Choix des ordres (nk = {nk})"); ax.legend(); ax.grid()
    fig.tight_layout(); fig.savefig(FIG / "2_ordres.png", dpi=120); plt.close(fig)

    # 3. Simulation libre sur un essai de test
    Ut, Yt = test
    i, sl = 0, slice(0, 600)
    fig, ax = plt.subplots(2, 1, figsize=(13, 6), sharex=True)
    ax[0].plot(t[sl], Yt[i, sl], "k", lw=1.2, label="mesure")
    for name, Yh in sims.items():
        ax[0].plot(t[sl], Yh[i, sl], lw=1, label=name)
    ax[0].set_ylabel("az [m/s²]"); ax[0].legend(); ax[0].set_title("Simulation libre, essai de test")
    ax[1].plot(t[sl], Ut[i, sl], "C2"); ax[1].axhspan(u_eq - BAND, u_eq + BAND, color="g", alpha=.1)
    ax[1].set_ylabel("u"); ax[1].set_xlabel("t [s]")
    fig.tight_layout(); fig.savefig(FIG / "3_simulation_libre.png", dpi=120); plt.close(fig)

    # 4. Résidus
    fig, ax = plt.subplots(1, 2, figsize=(12, 3.5))
    ax[0].stem(lags, acf); ax[0].set_title("Autocorrélation des résidus"); ax[0].set_xlabel("retard")
    ax[1].stem(lags, ccf); ax[1].set_title("Corrélation résidus / u"); ax[1].set_xlabel("retard")
    fig.tight_layout(); fig.savefig(FIG / "4_residus.png", dpi=120); plt.close(fig)


# --------------------------------------------------------------------------------------
def main():
    U, Y = load_data()
    us, ys, p, u_eq = static_curve(U, Y)
    train, val, test = split(U, Y)

    order, scores = select_orders(train, val)
    na, nb, nk = order
    n0 = n_init(na, nb, nk)

    # Modèle global : toutes les données d'entraînement
    th_glob = fit_arx(*train, na, nb, nk)
    # Modèle local : uniquement les instants proches du vol stationnaire
    th_loc = fit_arx(*train, na, nb, nk, row_mask=near_hover_mask(u_eq))
    print(f"[3] theta global = {np.round(th_glob, 4)}")
    print(f"[3] theta local  = {np.round(th_loc, 4)}")

    poles, gain, u_eq_arx, lags, acf, ccf = check_model(th_glob, na, nb, *val, nk)

    # Évaluation finale, une seule fois, sur le test
    print("\n[7] Résultats sur le TEST (simulation libre sur les essais complets de 100 s)")
    print(f"{'modèle':<12}{'fit 1 pas':>11}{'fit libre':>11}{'RMSE':>8}"
          f"{'RMSE proche':>13}{'RMSE loin':>11}")
    sims = {}
    for name, th in [("ARX global", th_glob), ("ARX local", th_loc)]:
        s, Yh = evaluate(arx_predictor(th), *test, na, nb, nk)
        r = by_regime(test[1], Yh, test[0], u_eq, n0)
        sims[name] = Yh
        print(f"{name:<12}{s['fit_1pas']:>10.2f}%{s['fit_libre']:>10.2f}%{s['rmse_libre']:>8.2f}"
              f"{r['proche']:>13.2f}{r['loin']:>11.2f}")
    print(f"(part des instants 'proches du stationnaire' : {r['part_proche'] * 100:.0f} %)")
    # Pour le NN : sims["NN"] = evaluate(predict_nn, *test, na, nb, nk)[1]  avec la même interface

    A, B = to_state_space(th_glob, na, nb, nk)
    np.savez(HERE / "arx_model.npz", theta_global=th_glob, theta_local=th_loc,
             na=na, nb=nb, nk=nk, u_eq=u_eq_arx, Ts=TS, A=A, B=B)
    print(f"\n[8] Modèle d'état : {A.shape[0]} états (az passés, u passés, vz, z) -> arx_model.npz")

    figures(U, Y, us, ys, p, u_eq, scores, order, sims, test, acf, ccf, lags)
    print(f"Figures écrites dans {FIG}")


if __name__ == "__main__":
    main()
