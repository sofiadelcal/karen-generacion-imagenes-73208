"""Genera los resultados del estudio de dígitos de Karen.

Ejecución: python experimento_digitos.py
"""

import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_digits

from gan_core import train, mmd_rbf


OUT = Path(__file__).parent / "resultados"
OUT.mkdir(exist_ok=True)
rng = np.random.default_rng(204)
digits = load_digits()
data = digits.data.astype(float) / 8.0 - 1.0
train_idx = rng.permutation(len(data))[:1500]
test_idx = np.setdiff1d(np.arange(len(data)), train_idx)
train_data, test_data = data[train_idx], data[test_idx]

reference = test_data[:256]
other_real = train_data[:256]
seed_rows = []
for seed in range(204, 209):
    before, after, losses, _ = train(
        train_data, noise_dim=16, g_widths=[96, 128], d_widths=[128, 64],
        steps=1400, batch=64, seed=seed, lr=0.00035)
    seed_rows.append((seed,
                      mmd_rbf(reference, before, bandwidth=4),
                      mmd_rbf(reference, after, bandwidth=4)))
    if seed == 204:
        initial, final, history = before, after, losses

with (OUT / "metricas_por_semilla.csv").open("w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["semilla", "MMD2_inicial", "MMD2_final"])
    w.writerows(seed_rows)

with (OUT / "perdidas.csv").open("w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["paso", "perdida_discriminador", "perdida_generador"])
    w.writerows(history)

scores = {
    "MMD real vs real": mmd_rbf(reference, other_real, bandwidth=4),
    "MMD antes de entrenar": mmd_rbf(reference, initial, bandwidth=4),
    "MMD tras entrenar": mmd_rbf(reference, final, bandwidth=4),
}
with (OUT / "metricas.csv").open("w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["indicador", "valor"])
    w.writerows(scores.items())

seeds = np.asarray(seed_rows)
with (OUT / "resumen_repeticiones.csv").open("w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["momento", "media_MMD2", "desviacion_entre_semillas", "minimo", "maximo"])
    for name, col in [("inicial", 1), ("final", 2)]:
        values = seeds[:, col]
        w.writerow([name, values.mean(), values.std(ddof=1), values.min(), values.max()])

fig, ax = plt.subplots(figsize=(6.4, 3.5))
for seed, before, after in seed_rows:
    ax.plot([0, 1], [before, after], marker="o", alpha=0.7, label=f"Semilla {int(seed)}")
ax.set(xticks=[0, 1], xticklabels=["Antes", "Después"], ylabel="MMD² frente a reserva real",
       title="Cinco entrenamientos independientes")
ax.grid(alpha=0.2)
ax.legend(fontsize=8, ncol=2)
fig.tight_layout()
fig.savefig(OUT / "repeticiones.png", dpi=200)
plt.close(fig)

fig, axes = plt.subplots(3, 8, figsize=(10, 4.4))
for row, samples in enumerate([reference, initial, final]):
    for col in range(8):
        axes[row, col].imshow(samples[col].reshape(8, 8), cmap="gray_r", vmin=-1, vmax=1)
        axes[row, col].axis("off")
for row, label in enumerate(["Reales", "Inicio", "Paso 1400"]):
    axes[row, 0].set_ylabel(label, fontsize=10)
fig.tight_layout()
fig.savefig(OUT / "mosaico_digitos.png", dpi=200, bbox_inches="tight")
plt.close(fig)

fig, ax = plt.subplots(figsize=(7.2, 3.5))
ax.plot(history[:, 0], history[:, 1], label="Discriminador")
ax.plot(history[:, 0], history[:, 2], label="Generador")
ax.set(xlabel="Actualizaciones adversariales", ylabel="Pérdida", title="Evolución durante el entrenamiento")
ax.grid(alpha=0.25)
ax.legend()
fig.tight_layout()
fig.savefig(OUT / "perdidas.png", dpi=200)
plt.close(fig)

for key, value in scores.items():
    print(f"{key}: {value:.5f}")
print("Semillas:", seed_rows)
