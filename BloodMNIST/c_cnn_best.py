"""
c_cnn_best.py -- Item (c): retreinamento da melhor CNN simples (escolhida em (b)).

  * Le results/b_summary.json  (melhor (n,k) pela acuracia balanceada de validacao).
  * Retreina com 5 sementes (paciencia maior que na varredura), seleciona o modelo
    de melhor VALIDACAO e o avalia no TESTE: matriz de confusao, acuracia
    balanceada, F1 (macro e ponderada); reporta tambem media +- desvio entre sementes.
  * Seleciona 5 padroes de teste classificados incorretamente -- os erros de MAIOR
    confianca, um por classe verdadeira distinta -- e exibe imagem, classe esperada
    e o vetor de probabilidades estimado pela rede.
  * Visualiza os kernels aprendidos e os mapas de ativacao (apoio a discussao).

Uso:  python c_cnn_best.py --data data/bloodmnist.npz [--n N --k K]
"""
import argparse
import time

import numpy as np
import torch

import common as C
from b_cnn_sweep import SimpleCNN, TRAIN_CFG


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/bloodmnist.npz")
    ap.add_argument("--n", type=int, default=None)
    ap.add_argument("--k", type=int, default=None)
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--reuse", action="store_true",
                    help="nao retreina: carrega results/c_cnn_best.pt e refaz a analise de erros")
    args = ap.parse_args()
    C.ensure_dirs()
    device = C.get_device()
    data = C.load_bloodmnist(args.data)
    if args.reuse:   # reaproveita o modelo salvo; refaz apenas a analise de erros
        out = C.load_json(f"{C.RESULTS_DIR}/c_cnn.json")
        m = SimpleCNN(out["n"], out["k"])
        m.load_state_dict(torch.load(f"{C.RESULTS_DIR}/c_cnn_best.pt")); m.to(device)
        out.update(analyze_errors(C.predict_proba(m, data.test.x, device), data))
        C.save_json(out, f"{C.RESULTS_DIR}/c_cnn.json")
        return
    if args.n is None or args.k is None:
        b = C.load_json(f"{C.RESULTS_DIR}/b_summary.json")["best"]
        args.n, args.k = b["n"], b["k"]
    print(f"CNN simples: n={args.n}, k={args.k}")
    cfg = C.TrainCfg(**{**TRAIN_CFG.__dict__, "max_epochs": 100, "patience": 15})

    finals, best_val = [], -1
    for s in range(args.seeds):
        C.set_seed(1000 + s)
        m = SimpleCNN(args.n, args.k)
        info = C.train_model(m, data.train, data.val, cfg, device)
        p = C.predict_proba(m, data.test.x, device)
        met = C.metrics(data.test.y.numpy(), p.argmax(1))
        finals.append({"seed": 1000 + s, "val_bacc": info["best_val_bacc"], **met,
                       "train_time_s": info["train_time_s"], "epochs": info["epochs_run"],
                       "best_epoch": info["best_epoch"], "history": info["history"]})
        print(f"[s={s}] val {info['best_val_bacc']:.4f} test bacc {met['bacc']:.4f} "
              f"f1 {met['f1_macro']:.4f} t={info['train_time_s']:.1f}s", flush=True)
        if info["best_val_bacc"] > best_val:
            best_val, best_m, best_p, best_rec = info["best_val_bacc"], m, p, finals[-1]

    t0 = time.perf_counter(); C.predict_proba(best_m, data.test.x, device)
    infer = time.perf_counter() - t0

    plot_kernels(best_m, data, f"{C.FIG_DIR}/c_kernels.png")
    summ = {k: [float(np.mean([f[k] for f in finals])), float(np.std([f[k] for f in finals]))]
            for k in ("acc", "bacc", "f1_macro", "f1_weighted", "train_time_s")}
    out = {"n": args.n, "k": args.k, "train_cfg": cfg.__dict__, "finals": finals,
           "summary_mean_std": summ, "selected": best_rec,
           "params": C.count_params(best_m),
           "test_inference_s": infer, "env": C.env_info()}
    out.update(analyze_errors(best_p, data))
    C.save_json(out, f"{C.RESULTS_DIR}/c_cnn.json")
    C.plot_confusion(best_rec["cm"], f"{C.FIG_DIR}/c_cnn_cm.png",
                     f"CNN simples (n={args.n}, k={args.k}) – teste")
    C.plot_history(best_rec["history"], f"{C.FIG_DIR}/c_cnn_hist.png", "CNN simples – curvas de treinamento")
    torch.save(best_m.state_dict(), f"{C.RESULTS_DIR}/c_cnn_best.pt")
    print("resumo:", summ, "| params", out["params"])


def ece(p, y, bins=15):
    """Expected Calibration Error (confianca = max_c p_c), bins de largura igual."""
    conf, pred = p.max(1), p.argmax(1); acc = (pred == y).astype(float)
    edges = np.linspace(0, 1, bins + 1); e = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        msk = (conf > lo) & (conf <= hi)
        if msk.any():
            e += msk.mean() * abs(acc[msk].mean() - conf[msk].mean())
    return float(e)


def analyze_errors(p, data, n_show=5, seed=0):
    """Escolhe `n_show` erros de teste com classes verdadeiras DISTINTAS, sorteados
    (semente fixa) entre todos os erros -- um sorteio evita mostrar apenas os erros
    de confianca saturada (p~1), que nada informam sobre a distribuicao estimada.
    Tambem calcula estatisticas de confianca/calibracao."""
    y = data.test.y.numpy(); yhat = p.argmax(1)
    wrong = np.where(yhat != y)[0]
    rng = np.random.default_rng(seed)
    picked, used = [], set()
    for i in rng.permutation(wrong):
        if int(y[i]) not in used:
            picked.append(int(i)); used.add(int(y[i]))
        if len(picked) == n_show:
            break
    picked.sort(key=lambda i: y[i])
    errors = [{"idx": i, "true": int(y[i]), "pred": int(yhat[i]),
               "proba": [float(v) for v in p[i]]} for i in picked]
    plot_errors(data.raw_test, errors, f"{C.FIG_DIR}/c_errors.png")
    conf = p.max(1)
    stats = {"n_errors": int(len(wrong)),
             "conf_err_median": float(np.median(conf[wrong])),
             "conf_ok_median": float(np.median(conf[yhat == y])),
             "frac_err_conf_gt_0.9": float((conf[wrong] > 0.9).mean()),
             "p_true_on_err_median": float(np.median(p[wrong, y[wrong]])),
             "true_in_top2_on_err": float(np.mean([y[i] in np.argsort(-p[i])[:2] for i in wrong])),
             "ece": ece(p, y)}
    print("erros:", stats)
    return {"errors": errors, "error_stats": stats}


def plot_errors(raw, errors, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(2, len(errors), figsize=(2.6 * len(errors), 5.0),
                           gridspec_kw={"height_ratios": [1, 1.1]}, squeeze=False)
    for j, e in enumerate(errors):
        ax[0, j].imshow(raw[e["idx"]]); ax[0, j].axis("off")
        ax[0, j].set_title(f"#{e['idx']}\nverdadeira: {C.CLASS_SHORT[e['true']]}\n"
                           f"predita: {C.CLASS_SHORT[e['pred']]}", fontsize=8)
        cols = ["tab:green" if c == e["true"] else ("tab:red" if c == e["pred"] else "tab:gray")
                for c in range(C.N_CLASSES)]
        ax[1, j].barh(range(C.N_CLASSES), e["proba"], color=cols)
        ax[1, j].set_yticks(range(C.N_CLASSES), C.CLASS_SHORT, fontsize=7)
        for c, v in enumerate(e["proba"]):
            ax[1, j].text(min(v, 0.62) + 0.02, c, f"{v:.3f}", va="center", fontsize=6.5)
        ax[1, j].invert_yaxis(); ax[1, j].set_xlim(0, 1)
        ax[1, j].tick_params(axis="x", labelsize=7); ax[1, j].set_xlabel("P(classe | x)", fontsize=7)
    fig.tight_layout(); fig.savefig(path, dpi=200); plt.close(fig)


@torch.no_grad()
def plot_kernels(m, data, path, max_k=32):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    W = m.conv.weight.detach().cpu().numpy()          # (n, 3, k, k)
    n = min(W.shape[0], max_k)
    cols = 8; rows = int(np.ceil(n / cols))
    fig, ax = plt.subplots(rows, cols, figsize=(cols * 0.9, rows * 0.9))
    ax = np.atleast_2d(ax)
    for i in range(rows * cols):
        a = ax.flat[i]; a.axis("off")
        if i < n:
            w = W[i].transpose(1, 2, 0)
            w = (w - w.min()) / (w.max() - w.min() + 1e-8)
            a.imshow(w, interpolation="nearest")
    fig.suptitle("Kernels aprendidos (RGB, normalizados por kernel)", fontsize=9)
    fig.tight_layout(); fig.savefig(path, dpi=200); plt.close(fig)


if __name__ == "__main__":
    main()
