"""
e_compare.py -- Comparacao geral MLP x CNN simples x ResNet-18 (transfer learning).

Le results/{a_mlp,c_cnn,d_transfer}.json e produz:
  * results/compare.json  e  results/compare_table.tex (tabela para o relatorio);
  * results/figs/e_compare.png (desempenho x parametros x tempo de treinamento);
  * results/figs/e_f1_per_class.png (F1 por classe para os tres modelos).

Uso:  python e_compare.py
"""
import numpy as np

import common as C


def main():
    C.ensure_dirs()
    a = C.load_json(f"{C.RESULTS_DIR}/a_mlp.json")
    c = C.load_json(f"{C.RESULTS_DIR}/c_cnn.json")
    d = C.load_json(f"{C.RESULTS_DIR}/d_transfer.json")
    ft = d["FT"]
    bf = int(np.argmax([f["best_val_bacc"] for f in ft["folds"]]))
    rows = [
        {"model": f"MLP (H={a['best_cfg']['hidden']})", "params": a["params"]["total"],
         "acc": a["selected"]["acc"], "bacc": a["selected"]["bacc"], "f1": a["selected"]["f1_macro"],
         "bacc_ms": a["summary_mean_std"]["bacc"], "f1_ms": a["summary_mean_std"]["f1_macro"],
         "train_s": a["selected"]["train_time_s"], "epochs": a["selected"]["epochs"],
         "infer_s": a["test_inference_s"], "f1pc": a["selected"]["f1_per_class"]},
        {"model": f"CNN simples (n={c['n']}, k={c['k']})", "params": c["params"]["total"],
         "acc": c["selected"]["acc"], "bacc": c["selected"]["bacc"], "f1": c["selected"]["f1_macro"],
         "bacc_ms": c["summary_mean_std"]["bacc"], "f1_ms": c["summary_mean_std"]["f1_macro"],
         "train_s": c["selected"]["train_time_s"], "epochs": c["selected"]["epochs"],
         "infer_s": c["test_inference_s"], "f1pc": c["selected"]["f1_per_class"]},
        {"model": "ResNet-18 FT (1 fold)", "params": ft["params"]["total"],
         "acc": ft["test_per_fold"][bf]["acc"], "bacc": ft["test_per_fold"][bf]["bacc"],
         "f1": ft["test_per_fold"][bf]["f1_macro"],
         "bacc_ms": [float(np.mean([m["bacc"] for m in ft["test_per_fold"]])),
                     float(np.std([m["bacc"] for m in ft["test_per_fold"]]))],
         "f1_ms": [float(np.mean([m["f1_macro"] for m in ft["test_per_fold"]])),
                   float(np.std([m["f1_macro"] for m in ft["test_per_fold"]]))],
         "train_s": ft["folds"][bf]["train_time_s"], "epochs": ft["folds"][bf]["epochs_run"],
         "infer_s": ft["test_inference_s_single"], "f1pc": ft["test_per_fold"][bf]["f1_per_class"]},
        {"model": f"ResNet-18 FT (comitê {len(ft['folds'])})", "params": ft["params"]["total"] * len(ft["folds"]),
         "acc": ft["test_ensemble"]["acc"], "bacc": ft["test_ensemble"]["bacc"],
         "f1": ft["test_ensemble"]["f1_macro"], "bacc_ms": None, "f1_ms": None,
         "train_s": float(sum(f["train_time_s"] for f in ft["folds"])),
         "epochs": int(sum(f["epochs_run"] for f in ft["folds"])),
         "infer_s": ft["test_inference_s_single"] * len(ft["folds"]),
         "f1pc": ft["test_ensemble"]["f1_per_class"]},
    ]
    C.save_json(rows, f"{C.RESULTS_DIR}/compare.json")

    def ms(v):
        return "--" if v is None else f"{v[0]:.4f} $\\pm$ {v[1]:.4f}"
    lines = []
    for r in rows:
        lines.append(f"{r['model']} & {r['params']:,} & {r['acc']:.4f} & {r['bacc']:.4f} & {r['f1']:.4f} & "
                     f"{ms(r['bacc_ms'])} & {r['train_s']/60:.1f} & {r['epochs']} & {r['infer_s']:.2f} \\\\"
                     .replace(",", "."))
    open(f"{C.RESULTS_DIR}/compare_table.tex", "w").write("\n".join(lines))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.6))
    cols = ["#4C72B0", "#DD8452", "#55A868", "#C44E52"]
    for r, col in zip(rows, cols):
        ax[0].scatter(r["params"], r["bacc"], s=70, color=col, label=r["model"], zorder=3)
        ax[1].scatter(r["train_s"] / 60, r["bacc"], s=70, color=col, zorder=3)
    ax[0].set_xscale("log"); ax[0].set_xlabel("Número de parâmetros (log)")
    ax[1].set_xscale("log"); ax[1].set_xlabel("Tempo de treinamento [min] (log)")
    for a_ in ax:
        a_.set_ylabel("Acurácia balanceada (teste)"); a_.grid(alpha=.3, which="both")
    ax[0].legend(fontsize=7, loc="lower right")
    fig.tight_layout(); fig.savefig(f"{C.FIG_DIR}/e_compare.png", dpi=200); plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 3.2))
    w = 0.2
    for i, (r, col) in enumerate(zip(rows, cols)):
        ax.bar(np.arange(C.N_CLASSES) + (i - 1.5) * w, r["f1pc"], w, color=col, label=r["model"])
    ax.set_xticks(range(C.N_CLASSES), C.CLASS_SHORT); ax.set_ylim(0.4, 1.0)
    ax.set_ylabel("F1 por classe (teste)"); ax.legend(fontsize=7, ncol=4, loc="lower center")
    ax.grid(alpha=.3, axis="y")
    fig.tight_layout(); fig.savefig(f"{C.FIG_DIR}/e_f1_per_class.png", dpi=200); plt.close(fig)
    for r in rows:
        print({k: v for k, v in r.items() if k != "f1pc"})


if __name__ == "__main__":
    main()
