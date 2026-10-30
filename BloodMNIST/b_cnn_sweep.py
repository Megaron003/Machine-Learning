"""
b_cnn_sweep.py -- Item (b): CNN simples e varredura de hiperparametros.

Arquitetura (exatamente o pedido no enunciado):
    Conv2d(3 -> n, k x k, stride 1, padding 'same')  -> ReLU      (i)
    MaxPool2d(2 x 2, stride 2)                                   (ii)
    Flatten -> Linear(n * 14 * 14 -> 8) -> softmax               (iii)
(a softmax e' aplicada implicitamente pela entropia cruzada no treino e
 explicitamente na inferencia, em common.predict_proba).

Varredura em grade completa  n in N_LIST  x  k in K_LIST, com R sementes por ponto.
Criterio: acuracia balanceada no conjunto de VALIDACAO (melhor epoca, early stopping).
Resultados sao gravados incrementalmente (results/b_sweep.jsonl), de modo que a
execucao pode ser interrompida e retomada sem refazer pontos ja avaliados.

Uso:  python b_cnn_sweep.py --data data/bloodmnist.npz
"""
import argparse
import json
import os

import numpy as np
import torch
import torch.nn as nn

import common as C

N_LIST = [2, 4, 8, 16, 32, 64, 128]
K_LIST = [3, 5, 7, 9, 11]


class SimpleCNN(nn.Module):
    def __init__(self, n_kernels=16, k=3, img=28, act="relu", pool="max"):
        super().__init__()
        self.conv = nn.Conv2d(3, n_kernels, k, padding=k // 2)
        self.act = {"relu": nn.ReLU(), "tanh": nn.Tanh()}[act]
        self.pool = nn.MaxPool2d(2) if pool == "max" else nn.AvgPool2d(2)
        self.fc = nn.Linear(n_kernels * (img // 2) ** 2, C.N_CLASSES)

    def forward(self, x):
        return self.fc(torch.flatten(self.pool(self.act(self.conv(x))), 1))


TRAIN_CFG = C.TrainCfg(lr=1e-3, batch_size=128, max_epochs=60, patience=8,
                       weighted_loss=True, scheduler="plateau", weight_decay=1e-4)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/bloodmnist.npz")
    ap.add_argument("--seeds", type=int, default=2)
    ap.add_argument("--n", type=int, nargs="*", default=N_LIST)
    ap.add_argument("--k", type=int, nargs="*", default=K_LIST)
    args = ap.parse_args()
    C.ensure_dirs()
    device = C.get_device()
    data = C.load_bloodmnist(args.data)
    out = f"{C.RESULTS_DIR}/b_sweep.jsonl"
    done = set()
    if os.path.exists(out):
        for line in open(out):
            r = json.loads(line); done.add((r["n"], r["k"], r["seed"]))
    for k in args.k:
        for n in args.n:
            for s in range(args.seeds):
                if (n, k, s) in done:
                    continue
                C.set_seed(s)
                m = SimpleCNN(n, k)
                info = C.train_model(m, data.train, data.val, TRAIN_CFG, device)
                rec = {"n": n, "k": k, "seed": s, "val_bacc": info["best_val_bacc"],
                       "best_epoch": info["best_epoch"], "epochs": info["epochs_run"],
                       "time_s": info["train_time_s"], "params": C.count_params(m)["total"],
                       "final_train_bacc": info["history"]["train_bacc"][info["best_epoch"] - 1]}
                with open(out, "a") as f:
                    f.write(json.dumps(rec) + "\n")
                print(rec, flush=True)
    summarize(out)


def summarize(path):
    """Agrega por (n,k), gera tabela JSON e figuras da progressao."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    recs = [json.loads(l) for l in open(path)]
    ns = sorted({r["n"] for r in recs}); ks = sorted({r["k"] for r in recs})
    M = np.full((len(ks), len(ns)), np.nan); S = np.full_like(M, np.nan)
    TR = np.full_like(M, np.nan); T = np.full_like(M, np.nan); P = np.zeros_like(M)
    for i, k in enumerate(ks):
        for j, n in enumerate(ns):
            v = [r for r in recs if r["n"] == n and r["k"] == k]
            if v:
                M[i, j] = np.mean([r["val_bacc"] for r in v]); S[i, j] = np.std([r["val_bacc"] for r in v])
                TR[i, j] = np.mean([r["final_train_bacc"] for r in v])
                T[i, j] = np.mean([r["time_s"] for r in v]); P[i, j] = v[0]["params"]
    bi, bj = np.unravel_index(np.nanargmax(M), M.shape)
    summary = {"n_list": ns, "k_list": ks, "mean_val_bacc": M.tolist(), "std_val_bacc": S.tolist(),
               "train_bacc": TR.tolist(), "time_s": T.tolist(), "params": P.tolist(),
               "best": {"n": ns[bj], "k": ks[bi], "val_bacc": float(M[bi, bj]), "std": float(S[bi, bj])}}
    C.save_json(summary, f"{C.RESULTS_DIR}/b_summary.json")

    fig, ax = plt.subplots(1, 2, figsize=(10, 3.6))
    for i, k in enumerate(ks):
        ax[0].errorbar(ns, M[i], yerr=S[i], marker="o", ms=4, capsize=2, label=f"k={k}")
    ax[0].set_xscale("log", base=2); ax[0].set_xticks(ns, [str(n) for n in ns])
    ax[0].set_xlabel("Número de kernels $n$"); ax[0].set_ylabel("Acurácia balanceada (validação)")
    ax[0].legend(fontsize=7, ncol=2); ax[0].grid(alpha=.3)
    for j, n in enumerate(ns):
        ax[1].errorbar(ks, M[:, j], yerr=S[:, j], marker="s", ms=4, capsize=2, label=f"n={n}")
    ax[1].set_xticks(ks); ax[1].set_xlabel("Tamanho do kernel $k$ ($k\\times k$)")
    ax[1].set_ylabel("Acurácia balanceada (validação)"); ax[1].legend(fontsize=7, ncol=2); ax[1].grid(alpha=.3)
    fig.tight_layout(); fig.savefig(f"{C.FIG_DIR}/b_progression.png", dpi=200); plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    im = ax.imshow(M, cmap="viridis", aspect="auto")
    for i in range(len(ks)):
        for j in range(len(ns)):
            ax.text(j, i, f"{M[i,j]:.3f}", ha="center", va="center", fontsize=7,
                    color="white" if M[i, j] < np.nanmean(M) else "black")
    ax.set_xticks(range(len(ns)), ns); ax.set_yticks(range(len(ks)), ks)
    ax.set_xlabel("Número de kernels $n$"); ax.set_ylabel("Tamanho do kernel $k$")
    ax.add_patch(plt.Rectangle((bj - .5, bi - .5), 1, 1, fill=False, ec="red", lw=2))
    fig.colorbar(im, ax=ax, label="Acur. balanceada (val.)")
    fig.tight_layout(); fig.savefig(f"{C.FIG_DIR}/b_heatmap.png", dpi=200); plt.close(fig)
    print("melhor:", summary["best"])


if __name__ == "__main__":
    main()
