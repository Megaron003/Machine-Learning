"""
f_dataset_overview.py -- Figuras/tabela descritivas da base BloodMNIST
(amostras por classe e distribuicao das classes nas particoes oficiais).

Uso:  python f_dataset_overview.py --data data/bloodmnist.npz
"""
import argparse

import numpy as np

import common as C


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/bloodmnist.npz")
    args = ap.parse_args()
    C.ensure_dirs()
    raw = C.load_npz(args.data)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    rng = np.random.default_rng(0)
    fig, ax = plt.subplots(3, C.N_CLASSES, figsize=(C.N_CLASSES * 1.2, 3 * 1.3))
    x, y = raw["train"]
    for c in range(C.N_CLASSES):
        idx = rng.choice(np.where(y == c)[0], 3, replace=False)
        for r in range(3):
            ax[r, c].imshow(x[idx[r]]); ax[r, c].axis("off")
        ax[0, c].set_title(C.CLASS_SHORT[c], fontsize=9)
    fig.tight_layout(); fig.savefig(f"{C.FIG_DIR}/f_samples.png", dpi=200); plt.close(fig)

    counts = {s: np.bincount(raw[s][1], minlength=C.N_CLASSES).tolist() for s in raw}
    mean = (raw["train"][0].astype(np.float32) / 255).mean((0, 1, 2)).tolist()
    std = (raw["train"][0].astype(np.float32) / 255).std((0, 1, 2)).tolist()
    C.save_json({"counts": counts, "train_mean": mean, "train_std": std},
                f"{C.RESULTS_DIR}/f_dataset.json")
    print(counts, mean, std)


if __name__ == "__main__":
    main()
