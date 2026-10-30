"""
a_mlp.py -- Item (a): MLP com UMA camada intermediaria.

Metodologia (resumo; detalhes no relatorio):
  * Entrada: imagem 28x28x3 achatada (2352 atributos), normalizada por canal
    com media/desvio do conjunto de TREINO.
  * Arquitetura: Linear(2352 -> H) -> BatchNorm1d -> f(.) -> Dropout(p) -> Linear(H -> 8).
    A saida e' o vetor de logits; a softmax esta embutida na entropia cruzada.
  * Selecao de modelo em DUAS etapas, sempre pelo criterio
    "maior acuracia balanceada no conjunto de VALIDACAO oficial":
      Etapa 1: largura H x funcao de ativacao  (demais hiperparametros fixos)
      Etapa 2: com o melhor (H, f), regularizacao (dropout, weight decay) e
               ponderacao da funcao custo por classe.
  * Melhor configuracao: retreinada com 5 sementes; reporta-se media +- desvio
    no TESTE e a matriz de confusao do modelo de melhor validacao.

Uso:  python a_mlp.py --data data/bloodmnist.npz [--quick]
"""
import argparse
import itertools
import time

import numpy as np
import torch
import torch.nn as nn

import common as C


class MLP(nn.Module):
    def __init__(self, d_in=28 * 28 * 3, hidden=256, act="relu", dropout=0.0, bn=True):
        super().__init__()
        acts = {"relu": nn.ReLU(), "tanh": nn.Tanh(), "gelu": nn.GELU()}
        layers = [nn.Flatten(), nn.Linear(d_in, hidden)]
        if bn:
            layers.append(nn.BatchNorm1d(hidden))
        layers += [acts[act], nn.Dropout(dropout), nn.Linear(hidden, C.N_CLASSES)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def run(cfg_model, cfg_train, data, device, seed):
    C.set_seed(seed)
    m = MLP(**cfg_model)
    info = C.train_model(m, data.train, data.val, cfg_train, device)
    return m, info


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/bloodmnist.npz")
    ap.add_argument("--quick", action="store_true", help="grade reduzida (teste rapido)")
    ap.add_argument("--seeds", type=int, default=5)
    args = ap.parse_args()
    C.ensure_dirs()
    device = C.get_device()
    data = C.load_bloodmnist(args.data)
    print("dispositivo:", device, "| treino", tuple(data.train.x.shape))

    base_train = C.TrainCfg(lr=1e-3, batch_size=128, max_epochs=80, patience=12,
                            weighted_loss=True, scheduler="plateau")
    log = {"stage1": [], "stage2": [], "env": C.env_info()}

    # ---------------- Etapa 1: largura x ativacao ----------------
    widths = [64, 128, 256, 512, 1024, 2048] if not args.quick else [64, 256]
    acts = ["relu", "tanh"] if not args.quick else ["relu"]
    for H, act in itertools.product(widths, acts):
        cm = dict(hidden=H, act=act, dropout=0.25)
        _, info = run(cm, base_train, data, device, seed=0)
        rec = {**cm, "val_bacc": info["best_val_bacc"], "epochs": info["epochs_run"],
               "best_epoch": info["best_epoch"], "time_s": info["train_time_s"]}
        log["stage1"].append(rec)
        print("[E1]", rec, flush=True)
    best1 = max(log["stage1"], key=lambda r: r["val_bacc"])

    # ---------------- Etapa 2: regularizacao e ponderacao ----------------
    drops = [0.0, 0.25, 0.5] if not args.quick else [0.25]
    wds = [0.0, 1e-4, 1e-3] if not args.quick else [0.0]
    wls = [True, False]
    for p, wd, wl in itertools.product(drops, wds, wls):
        cm = dict(hidden=best1["hidden"], act=best1["act"], dropout=p)
        ct = C.TrainCfg(**{**base_train.__dict__, "weight_decay": wd, "weighted_loss": wl})
        _, info = run(cm, ct, data, device, seed=0)
        rec = {**cm, "weight_decay": wd, "weighted_loss": wl, "val_bacc": info["best_val_bacc"],
               "epochs": info["epochs_run"], "time_s": info["train_time_s"]}
        log["stage2"].append(rec)
        print("[E2]", rec, flush=True)
    best2 = max(log["stage2"], key=lambda r: r["val_bacc"])

    # ---------------- Avaliacao final com varias sementes ----------------
    cm = dict(hidden=best2["hidden"], act=best2["act"], dropout=best2["dropout"])
    ct = C.TrainCfg(**{**base_train.__dict__, "weight_decay": best2["weight_decay"],
                       "weighted_loss": best2["weighted_loss"]})
    finals = []
    best_model, best_val = None, -1
    for s in range(args.seeds):
        m, info = run(cm, ct, data, device, seed=100 + s)
        p = C.predict_proba(m, data.test.x, device)
        met = C.metrics(data.test.y.numpy(), p.argmax(1))
        finals.append({"seed": 100 + s, "val_bacc": info["best_val_bacc"], **met,
                       "train_time_s": info["train_time_s"], "epochs": info["epochs_run"],
                       "best_epoch": info["best_epoch"], "history": info["history"]})
        print(f"[final s={s}] val {info['best_val_bacc']:.4f} test bacc {met['bacc']:.4f} "
              f"f1 {met['f1_macro']:.4f} t={info['train_time_s']:.1f}s", flush=True)
        if info["best_val_bacc"] > best_val:
            best_val, best_model, best_rec = info["best_val_bacc"], m, finals[-1]

    # tempo de inferencia (lote inteiro de teste)
    t0 = time.perf_counter(); C.predict_proba(best_model, data.test.x, device)
    infer = time.perf_counter() - t0

    summ = {k: [float(np.mean([f[k] for f in finals])), float(np.std([f[k] for f in finals]))]
            for k in ("acc", "bacc", "f1_macro", "f1_weighted", "train_time_s")}
    log.update({"best_cfg": {**cm, "weight_decay": ct.weight_decay,
                             "weighted_loss": ct.weighted_loss, "lr": ct.lr,
                             "batch_size": ct.batch_size},
                "finals": finals, "summary_mean_std": summ, "selected": best_rec,
                "params": C.count_params(best_model), "test_inference_s": infer})
    C.save_json(log, f"{C.RESULTS_DIR}/a_mlp.json")
    C.plot_confusion(best_rec["cm"], f"{C.FIG_DIR}/a_mlp_cm.png",
                     f"MLP (H={cm['hidden']}) – teste")
    C.plot_history(best_rec["history"], f"{C.FIG_DIR}/a_mlp_hist.png", "MLP – curvas de treinamento")
    print("resumo:", summ, "| params", log["params"])


if __name__ == "__main__":
    main()
