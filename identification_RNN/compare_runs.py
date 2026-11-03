"""
Comparison of the RNN trainings of model_rnn/lightning_logs (same settings, different random seeds).

For each version: validation RMSE at each epoch (TensorBoard logs), best validation RMSE (the selection
criterion of RNN_drone.py) and RMSE on the test set (last 15 % of the sequences, never seen in training).
Writes model_rnn/comparison_runs.png and prints the table of results.
"""
import os
import json
import glob

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from torch import nn
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

# Work in the folder of this file (data files and checkpoints are next to it)
os.chdir(os.path.dirname(os.path.abspath(__file__)))
logs_dir = 'model_rnn/lightning_logs'

# Test set: same split as RNN_drone.py (last 15 % of the cleaned sequences)
U = pd.read_csv('../data_base_cleaning/bdd_in_clean.csv.gz', header=None).values.astype('float32')
Y = pd.read_csv('../data_base_cleaning/bdd_out_clean.csv.gz', header=None).values.astype('float32')
i_test = int(0.85 * U.shape[0])
U_test, Y_test = U[i_test:], Y[i_test:]

runs = []
for version in sorted(os.listdir(logs_dir), key=lambda v: int(v.split('_')[-1])):
    folder = os.path.join(logs_dir, version)
    with open(os.path.join(folder, 'run_info.json')) as f:
        info = json.load(f)
    norm, training = info['normalization'], info['training']

    # Validation RMSE at each epoch [m/s²] (val_loss is the MSE on normalised data)
    events = EventAccumulator(folder)
    events.Reload()
    val_rmse = np.sqrt([e.value for e in events.Scalars('val_loss')]) * norm['y_scale']

    # Test RMSE of the best checkpoint of this run (same equations as the exported weights)
    ckpt = glob.glob(os.path.join(folder, '*.ckpt'))[0]
    state = torch.load(ckpt, map_location='cpu', weights_only=False)['state_dict']
    hidden = info['model']['hidden_size']
    rnn, head = nn.RNN(1, hidden, batch_first=True), nn.Linear(hidden, 1)
    rnn.load_state_dict({k[len('rnn.'):]: v for k, v in state.items() if k.startswith('rnn.')})
    head.load_state_dict({k[len('head.'):]: v for k, v in state.items() if k.startswith('head.')})
    with torch.no_grad():
        h, _ = rnn(torch.tensor(U_test / norm['u_scale'], dtype=torch.float32).unsqueeze(-1))
        Y_pred = head(h).squeeze(-1).numpy() * norm['y_scale'] + norm['y_offset']

    runs.append(dict(version=version, seed=training.get('seed', '-'), val_rmse=val_rmse,
                     best_epoch=training['best_epoch'], epochs_run=training['epochs_run'],
                     best_val_rmse=float(np.sqrt(training['best_val_loss']) * norm['y_scale']),
                     test_rmse=float(np.sqrt(np.mean((Y_pred - Y_test) ** 2)))))

best = min(runs, key=lambda r: r['best_val_rmse'])         # Selection on the validation set only

# Table of results (Markdown, pasted in the README)
print('| Version | Seed | Best epoch | Validation RMSE [m/s²] | Test RMSE [m/s²] |')
print('|---|---|---|---|---|')
for r in runs:
    b = '**' if r is best else ''
    print(f"| {b}{r['version']}{b} | {r['seed']} | {r['best_epoch']} / {r['epochs_run']} | "
          f"{b}{r['best_val_rmse']:.3f}{b} | {b}{r['test_rmse']:.3f}{b} |")
print(f"Best model (validation): {best['version']}")


# Figure: the best run is highlighted, the other seeds stay in the background
BEST, OTHER = '#2a78d6', '#898781'                          # Highlighted run / other runs
INK, GRID, AXIS, SURFACE = '#52514e', '#e1e0d9', '#c3c2b7', '#fcfcfb'

fig, (ax_c, ax_b) = plt.subplots(1, 2, figsize=(12, 4.8), width_ratios=[1.4, 1], facecolor=SURFACE)
for r in runs:
    epochs = np.arange(1, r['val_rmse'].size + 1)
    if r is best:
        ax_c.plot(epochs, r['val_rmse'], color=BEST, linewidth=2.5, zorder=3, label=f"{r['version']} (meilleur)")
        ax_c.annotate(r['version'], (epochs[-1], r['val_rmse'][-1]), xytext=(6, 0), textcoords='offset points',
                      va='center', fontsize=9, color=INK)
    elif r['seed'] == '-':
        ax_c.plot(epochs, r['val_rmse'], color=OTHER, linewidth=1.5, linestyle='--', label=f"{r['version']} (sans graine)")
    else:
        ax_c.plot(epochs, r['val_rmse'], color=OTHER, linewidth=1.5, alpha=0.6,
                  label='autres graines' if r['seed'] == 1 else None)
ax_c.set_yscale('log')
ax_c.set_yticks([0.1, 0.2, 0.5, 1, 2], ['0.1', '0.2', '0.5', '1', '2'])
ax_c.minorticks_off()
ax_c.set_xlabel('Époque')
ax_c.set_ylabel('RMSE de validation [m/s²]')
ax_c.set_title("Apprentissage : RMSE de validation à chaque époque", loc='left', fontsize=11)
ax_c.legend(frameon=False)

names = [r['version'] for r in runs]
values = [r['best_val_rmse'] for r in runs]
bars = ax_b.bar(names, values, width=0.6, color=[BEST if r is best else OTHER for r in runs])
for bar, r in zip(bars, runs):
    ax_b.annotate(f"{r['best_val_rmse']:.3f}", (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                  xytext=(0, 3), textcoords='offset points', ha='center', fontsize=9, color=INK)
ax_b.set_ylabel('Meilleur RMSE de validation [m/s²]')
ax_b.set_title('Meilleur modèle de chaque entraînement', loc='left', fontsize=11)
ax_b.set_ylim(0, max(values) * 1.15)

for ax in (ax_c, ax_b):
    ax.set_facecolor(SURFACE)
    ax.grid(True, axis='y', which='major', color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ('top', 'right'):
        ax.spines[side].set_visible(False)
    for side in ('left', 'bottom'):
        ax.spines[side].set_color(AXIS)
    ax.tick_params(colors=INK, labelsize=9)
    ax.yaxis.label.set_color(INK)
    ax.xaxis.label.set_color(INK)

fig.suptitle('RNN u → az : 6 entraînements identiques, seule la graine aléatoire change', x=0.01, ha='left',
             fontsize=12, color='#0b0b0b')
plt.tight_layout()
fig.savefig('model_rnn/comparison_runs.png', dpi=150, facecolor=SURFACE)
print('Figure written: model_rnn/comparison_runs.png')
