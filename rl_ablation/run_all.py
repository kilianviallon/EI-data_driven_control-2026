"""
Lance toute l'étude d'ablation : 7 variantes x 3 graines = 21 entraînements, plusieurs en parallèle.

    python run_all.py [entraînements en parallèle (7)] [pas par entraînement (100000)]

Chaque entraînement tourne dans son propre processus (train_variant.py, 1 thread) ; son journal est dans
runs/<variante>_s<graine>/train.log. Un entraînement déjà terminé (metriques.json présent) n'est pas relancé.
À la fin, analyse.py trace les figures et écrit le tableau des résultats.
"""
import os
import sys
import json
import time
import subprocess

from drone_env_ablation import HERE, VARIANTES

GRAINES = [0, 1, 2]
n_parallele = int(sys.argv[1]) if len(sys.argv) > 1 else 7
n_pas = int(sys.argv[2]) if len(sys.argv) > 2 else 100_000

# La graine 0 de chaque variante d'abord : une première comparaison complète arrive au plus tôt
a_faire = [(v, g) for g in GRAINES for v in VARIANTES
           if not os.path.isfile(os.path.join(HERE, 'runs', f'{v}_s{g}', 'metriques.json'))]
n_total = len(a_faire)
env_fils = dict(os.environ, OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', PYTHONIOENCODING='utf-8')


def avancement(v, g):
    """Dernier pas évalué de l'entraînement (lu dans evaluations.json)."""
    try:
        with open(os.path.join(HERE, 'runs', f'{v}_s{g}', 'evaluations.json')) as f:
            return json.load(f)[-1]['pas']
    except (OSError, ValueError, IndexError):
        return 0


print(f"{n_total} entraînements de {n_pas} pas, {n_parallele} en parallèle\n")
t0 = time.time()
en_cours, termines, dernier_point = [], 0, 0.0
while a_faire or en_cours:
    # Démarrer des entraînements tant qu'il y a des places libres
    while a_faire and len(en_cours) < n_parallele:
        v, g = a_faire.pop(0)
        dossier = os.path.join(HERE, 'runs', f'{v}_s{g}')
        os.makedirs(dossier, exist_ok=True)
        log = open(os.path.join(dossier, 'train.log'), 'w')
        p = subprocess.Popen([sys.executable, os.path.join(HERE, 'train_variant.py'), v, str(g), str(n_pas)],
                             cwd=HERE, stdout=log, stderr=subprocess.STDOUT, env=env_fils)
        en_cours.append((v, g, p, log, time.time()))
        print(f"[{time.strftime('%H:%M')}] démarré : {v}_s{g}")

    # Entraînements terminés
    for item in list(en_cours):
        v, g, p, log, debut = item
        if p.poll() is None:
            continue
        en_cours.remove(item)
        log.close()
        termines += 1
        try:
            with open(os.path.join(HERE, 'runs', f'{v}_s{g}', 'metriques.json')) as f:
                m = json.load(f)
            print(f"[{time.strftime('%H:%M')}] fini {termines:2d}/{n_total} : {v}_s{g}  RMSE mission = {m['rmse']:.3f} m"
                  f"  (meilleur à {m['pas_meilleur'] // 1000}k pas)  en {(time.time() - debut) / 60:.1f} min")
        except OSError:
            print(f"[{time.strftime('%H:%M')}] ÉCHEC {v}_s{g} (code {p.returncode}) : voir runs/{v}_s{g}/train.log")

    # Point d'avancement toutes les 2 minutes
    if time.time() - dernier_point > 120 and en_cours:
        dernier_point = time.time()
        etat = '  '.join(f"{v}_s{g} {avancement(v, g) // 1000}k" for v, g, *_ in en_cours)
        print(f"[{time.strftime('%H:%M')}] en cours ({(time.time() - t0) / 60:.0f} min) : {etat}")
    time.sleep(5)

print(f"\nTerminé en {(time.time() - t0) / 60:.0f} min. Analyse des résultats...")
subprocess.run([sys.executable, os.path.join(HERE, 'analyse.py')], cwd=HERE, env=env_fils)
