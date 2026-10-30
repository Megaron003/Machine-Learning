"""
common.py -- Utilitarios compartilhados da Atividade 3 (IA048, 2s2026).

Contem:
  * carregamento da base BloodMNIST a partir do arquivo .npz oficial (MedMNIST v2);
  * normalizacao por canal (estatisticas estimadas SOMENTE no treino);
  * laco de treinamento generico com early stopping na acuracia balanceada de validacao;
  * metricas (acuracia, acuracia balanceada, F1 macro/ponderada, matriz de confusao);
  * contagem de parametros, cronometragem e graficos.

Todos os scripts (a_mlp.py, b_cnn_sweep.py, c_cnn_best.py, d_transfer.py,
e_compare.py) importam este modulo.  Executar a partir da pasta do projeto:
    python a_mlp.py --data data/bloodmnist.npz
"""
from __future__ import annotations

import copy
import json
import os
import random
import time
from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
                             confusion_matrix, f1_score)

# --------------------------------------------------------------------------
# Constantes da base
# --------------------------------------------------------------------------
CLASS_NAMES_PT = ["Basófilo", "Eosinófilo", "Eritroblasto", "Granulócito imaturo",
                  "Linfócito", "Monócito", "Neutrófilo", "Plaqueta"]
CLASS_SHORT = ["BAS", "EOS", "ERB", "IG", "LIN", "MON", "NEU", "PLT"]
N_CLASSES = 8

RESULTS_DIR = os.environ.get("IA048_RESULTS", "results")
FIG_DIR = os.path.join(RESULTS_DIR, "figs")


def ensure_dirs():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(FIG_DIR, exist_ok=True)


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def set_seed(seed: int):
    """Fixa todas as fontes de aleatoriedade relevantes (reprodutibilidade)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# --------------------------------------------------------------------------
# Dados
# --------------------------------------------------------------------------
@dataclass
class Split:
    x: torch.Tensor   # float32, (N, 3, H, W), ja normalizado
    y: torch.Tensor   # int64, (N,)


@dataclass
class BloodData:
    train: Split
    val: Split
    test: Split
    mean: np.ndarray
    std: np.ndarray
    raw_test: np.ndarray = field(repr=False, default=None)  # uint8 (N,H,W,3) p/ visualizacao


def load_npz(path: str):
    """Le o .npz oficial do MedMNIST (chaves {train,val,test}_{images,labels})."""
    d = np.load(path)
    out = {}
    for s in ("train", "val", "test"):
        x = d[f"{s}_images"]            # (N, H, W, 3) uint8
        y = d[f"{s}_labels"].reshape(-1).astype(np.int64)
        out[s] = (x, y)
    return out


def to_tensor(x_uint8: np.ndarray, mean: np.ndarray, std: np.ndarray) -> torch.Tensor:
    x = x_uint8.astype(np.float32) / 255.0                       # [0,1]
    x = (x - mean[None, None, None, :]) / std[None, None, None, :]
    return torch.from_numpy(np.ascontiguousarray(x.transpose(0, 3, 1, 2)))  # NCHW


def load_bloodmnist(path: str, stats: str = "train") -> BloodData:
    """Carrega a base com particao OFICIAL (treino/validacao/teste).

    stats="train"    -> media/desvio por canal estimados no conjunto de treino
                        (evita vazamento de informacao de val/teste);
    stats="imagenet" -> estatisticas da ImageNet (usadas no transfer learning,
                        pois os pesos pre-treinados esperam essa normalizacao).
    """
    raw = load_npz(path)
    if stats == "imagenet":
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    else:
        xt = raw["train"][0].astype(np.float32) / 255.0
        mean = xt.mean(axis=(0, 1, 2)).astype(np.float32)
        std = xt.std(axis=(0, 1, 2)).astype(np.float32)
    splits = {s: Split(to_tensor(raw[s][0], mean, std), torch.from_numpy(raw[s][1]))
              for s in ("train", "val", "test")}
    return BloodData(splits["train"], splits["val"], splits["test"], mean, std,
                     raw_test=raw["test"][0])


def class_weights(y: torch.Tensor) -> torch.Tensor:
    """Pesos inversamente proporcionais a frequencia: w_c = N / (C * N_c)."""
    counts = torch.bincount(y, minlength=N_CLASSES).float()
    return counts.sum() / (N_CLASSES * counts)


# --------------------------------------------------------------------------
# Metricas
# --------------------------------------------------------------------------
def metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    return {
        "acc": float(accuracy_score(y_true, y_pred)),
        "bacc": float(balanced_accuracy_score(y_true, y_pred)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro")),
        "f1_weighted": float(f1_score(y_true, y_pred, average="weighted")),
        "f1_per_class": f1_score(y_true, y_pred, average=None).tolist(),
        "cm": confusion_matrix(y_true, y_pred, labels=list(range(N_CLASSES))).tolist(),
    }


def count_params(model: nn.Module) -> dict:
    tot = sum(p.numel() for p in model.parameters())
    trn = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"total": int(tot), "trainable": int(trn)}


# --------------------------------------------------------------------------
# Treinamento
# --------------------------------------------------------------------------
@torch.no_grad()
def predict_proba(model: nn.Module, x: torch.Tensor, device, bs: int = 512,
                  transform=None) -> np.ndarray:
    model.eval()
    out = []
    for i in range(0, len(x), bs):
        xb = x[i:i + bs].to(device, non_blocking=True)
        if transform is not None:
            xb = transform(xb, train=False)
        out.append(F.softmax(model(xb), dim=1).cpu())
    return torch.cat(out).numpy()


@dataclass
class TrainCfg:
    lr: float = 1e-3
    weight_decay: float = 0.0
    batch_size: int = 128
    max_epochs: int = 60
    patience: int = 10
    weighted_loss: bool = True
    label_smoothing: float = 0.0
    scheduler: str = "plateau"      # "plateau" | "cosine" | "none"
    optimizer: str = "adam"         # "adam" | "adamw" | "sgd"
    verbose: bool = False


def train_model(model: nn.Module, train: Split, val: Split, cfg: TrainCfg, device,
                transform=None, param_groups=None) -> dict:
    """Treina com mini-lotes, minimizando a entropia cruzada (opcionalmente
    ponderada por classe) e monitora a ACURACIA BALANCEADA de validacao.
    Early stopping com paciencia `cfg.patience`; restaura os melhores pesos.

    `transform(xb, train: bool)` permite redimensionamento/aumentacao em GPU/CPU
    sobre o lote (usado no transfer learning).
    Retorna historico e tempo de treinamento (wall-clock, segundos).
    """
    model.to(device)
    w = class_weights(train.y).to(device) if cfg.weighted_loss else None
    crit = nn.CrossEntropyLoss(weight=w, label_smoothing=cfg.label_smoothing)
    params = param_groups if param_groups is not None else model.parameters()
    if cfg.optimizer == "sgd":
        opt = torch.optim.SGD(params, lr=cfg.lr, momentum=0.9, nesterov=True,
                              weight_decay=cfg.weight_decay)
    elif cfg.optimizer == "adamw":
        opt = torch.optim.AdamW(params, lr=cfg.lr, weight_decay=cfg.weight_decay)
    else:
        opt = torch.optim.Adam(params, lr=cfg.lr, weight_decay=cfg.weight_decay)
    if cfg.scheduler == "plateau":
        sch = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="max", factor=0.5,
                                                         patience=max(2, cfg.patience // 3))
    elif cfg.scheduler == "cosine":
        sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg.max_epochs)
    else:
        sch = None

    n = len(train.y)
    hist = {"train_loss": [], "train_bacc": [], "val_bacc": [], "val_loss": [], "lr": []}
    best, best_state, best_ep, bad = -1.0, None, -1, 0
    t0 = time.perf_counter()
    for ep in range(cfg.max_epochs):
        model.train()
        perm = torch.randperm(n)
        tot_loss, preds, ys = 0.0, [], []
        for i in range(0, n, cfg.batch_size):
            idx = perm[i:i + cfg.batch_size]
            xb = train.x[idx].to(device, non_blocking=True)
            yb = train.y[idx].to(device, non_blocking=True)
            if transform is not None:
                xb = transform(xb, train=True)
            logits = model(xb)
            loss = crit(logits, yb)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            tot_loss += loss.item() * len(idx)
            preds.append(logits.argmax(1).detach().cpu())
            ys.append(yb.cpu())
        tr_bacc = balanced_accuracy_score(torch.cat(ys).numpy(), torch.cat(preds).numpy())
        p_val = predict_proba(model, val.x, device, transform=transform)
        v_bacc = balanced_accuracy_score(val.y.numpy(), p_val.argmax(1))
        v_loss = float(F.nll_loss(torch.log(torch.from_numpy(p_val) + 1e-12), val.y).item())
        hist["train_loss"].append(tot_loss / n)
        hist["train_bacc"].append(float(tr_bacc))
        hist["val_bacc"].append(float(v_bacc))
        hist["val_loss"].append(v_loss)
        hist["lr"].append(opt.param_groups[0]["lr"])
        if cfg.verbose:
            print(f"  ep {ep+1:3d} loss {tot_loss/n:.4f} tr_bacc {tr_bacc:.4f} "
                  f"val_bacc {v_bacc:.4f} val_loss {v_loss:.4f}", flush=True)
        if sch is not None:
            sch.step(v_bacc) if cfg.scheduler == "plateau" else sch.step()
        if v_bacc > best + 1e-4:
            best, best_ep, bad = v_bacc, ep, 0
            best_state = copy.deepcopy({k: v.detach().cpu() for k, v in model.state_dict().items()})
        else:
            bad += 1
            if bad >= cfg.patience:
                break
    train_time = time.perf_counter() - t0
    model.load_state_dict(best_state)
    return {"history": hist, "best_val_bacc": float(best), "best_epoch": best_ep + 1,
            "epochs_run": ep + 1, "train_time_s": train_time,
            "time_per_epoch_s": train_time / (ep + 1)}


# --------------------------------------------------------------------------
# Graficos
# --------------------------------------------------------------------------
def plot_confusion(cm, path: str, title: str = ""):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    cm = np.asarray(cm)
    cmn = cm / cm.sum(axis=1, keepdims=True)
    fig, ax = plt.subplots(figsize=(6.2, 5.4))
    im = ax.imshow(cmn, cmap="Blues", vmin=0, vmax=1)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, f"{cm[i, j]}\n{100*cmn[i, j]:.1f}%", ha="center", va="center",
                    fontsize=6.5, color="white" if cmn[i, j] > 0.55 else "black")
    ax.set_xticks(range(N_CLASSES), CLASS_SHORT)
    ax.set_yticks(range(N_CLASSES), CLASS_SHORT)
    ax.set_xlabel("Classe predita")
    ax.set_ylabel("Classe verdadeira")
    if title:
        ax.set_title(title, fontsize=10)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Fração da linha (recall)")
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def plot_history(hist: dict, path: str, title: str = ""):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ep = np.arange(1, len(hist["train_loss"]) + 1)
    fig, ax = plt.subplots(1, 2, figsize=(9, 3.2))
    ax[0].plot(ep, hist["train_loss"], label="treino")
    ax[0].plot(ep, hist["val_loss"], label="validação")
    ax[0].set_xlabel("Época"); ax[0].set_ylabel("Entropia cruzada"); ax[0].legend()
    ax[1].plot(ep, hist["train_bacc"], label="treino")
    ax[1].plot(ep, hist["val_bacc"], label="validação")
    ax[1].set_xlabel("Época"); ax[1].set_ylabel("Acurácia balanceada"); ax[1].legend()
    for a in ax:
        a.grid(alpha=.3)
    if title:
        fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def save_json(obj, path: str):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def load_json(path: str):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def env_info() -> dict:
    import platform
    return {"torch": torch.__version__, "device": str(get_device()),
            "threads": torch.get_num_threads(), "cpu": platform.processor() or platform.machine(),
            "python": platform.python_version()}
