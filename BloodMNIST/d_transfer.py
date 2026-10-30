"""
d_transfer.py -- Item (d): transfer learning com ResNet-18 pre-treinada na ImageNet-1k.

Modelo-base
  torchvision.models.resnet18; pesos ImageNet-1k carregados de:
    * --weights <arquivo.pth>  (state_dict compativel com torchvision; ex.: pesos
      timm 'resnet18.a1_in1k', que usam os mesmos nomes de camadas), ou
    * torchvision 'IMAGENET1K_V1' (download automatico) se --weights nao for dado.
  Adaptacao da saida: a camada `fc` (512 -> 1000) e' substituida por
  Dropout(0.2) + Linear(512 -> 8), inicializada aleatoriamente.

Pre-processamento
  Imagens 28x28 normalizadas com media/desvio da ImageNet (exigido pelos pesos)
  e interpoladas (bilinear, antialias) para R x R (padrao R=64) -- com 28x28 o
  mapa apos conv1+maxpool teria apenas 7x7 e o ultimo estagio 1x1.
  Aumentacao de dados (somente no treino): flips horizontal/vertical e rotacoes
  de 90 graus -- celulas em esfregaco nao tem orientacao canonica, logo o rotulo
  e' invariante ao grupo diedral D4.

Validacao cruzada (estratificada, K=5) sobre TREINO+VALIDACAO oficiais (13.671 imagens):
  Estrategia FE  (feature extraction): backbone congelado, so a cabeca e' treinada
                 (atributos de 512 dimensoes pre-computados -> custo baixo);
  Estrategia FT  (fine-tuning completo): todas as camadas treinadas, com taxa de
                 aprendizado discriminativa (backbone 10x menor que a cabeca),
                 AdamW + cosseno, early stopping no fold de validacao.
  Para cada estrategia: media +- desvio da acuracia balanceada nos folds de validacao.
  A estrategia vencedora e' avaliada no TESTE: (i) cada modelo de fold
  individualmente e (ii) o comite (media das probabilidades dos K modelos).

Uso:  python d_transfer.py --data data/bloodmnist.npz --weights weights/resnet18.pth
"""
import argparse
import json
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from sklearn.model_selection import StratifiedKFold

import common as C


def build_resnet18(weights_path=None):
    if weights_path:
        m = torchvision.models.resnet18(weights=None)
        sd = torch.load(weights_path, map_location="cpu")
        sd = sd.get("state_dict", sd)
        missing, unexpected = m.load_state_dict(sd, strict=False)
        assert not [k for k in missing if not k.startswith("fc.")], missing
    else:
        m = torchvision.models.resnet18(weights="IMAGENET1K_V1")
    m.fc = nn.Sequential(nn.Dropout(0.2), nn.Linear(512, C.N_CLASSES))
    return m


class Resize:
    """Redimensiona o lote e aplica aumentacao D4 (flips + rot90) no treino."""
    def __init__(self, size=64, augment=True):
        self.size, self.augment = size, augment

    def __call__(self, x, train=False):
        if train and self.augment:
            B = x.shape[0]
            f1 = torch.rand(B, device=x.device) < 0.5
            f2 = torch.rand(B, device=x.device) < 0.5
            x = torch.where(f1[:, None, None, None], x.flip(3), x)
            x = torch.where(f2[:, None, None, None], x.flip(2), x)
            if torch.rand(1).item() < 0.5:            # rot90 no lote (transpoe H,W)
                x = x.transpose(2, 3)
        if x.shape[-1] != self.size:
            x = F.interpolate(x, size=(self.size, self.size), mode="bilinear",
                              align_corners=False, antialias=False)
        return x


@torch.no_grad()
def extract_features(backbone, x, tf, device, bs=256):
    backbone.eval(); feats = []
    for i in range(0, len(x), bs):
        feats.append(backbone(tf(x[i:i + bs].to(device), train=False)).cpu())
    return torch.cat(feats)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/bloodmnist.npz")
    ap.add_argument("--weights", default=None)
    ap.add_argument("--size", type=int, default=64)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--max_folds_run", type=int, default=None, help="limitar folds (teste rapido)")
    args = ap.parse_args()
    C.ensure_dirs()
    device = C.get_device()
    data = C.load_bloodmnist(args.data, stats="imagenet")
    X = torch.cat([data.train.x, data.val.x]); Y = torch.cat([data.train.y, data.val.y])
    skf = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=0)
    folds = list(skf.split(np.zeros(len(Y)), Y.numpy()))
    if args.max_folds_run:
        folds = folds[:args.max_folds_run]
    tf = Resize(args.size, augment=True)
    out_path = f"{C.RESULTS_DIR}/d_transfer.json"
    res = C.load_json(out_path) if os.path.exists(out_path) else {}
    res.update({"args": vars(args), "env": C.env_info()})

    # ======================= Estrategia FE (congelada) =======================
    base = build_resnet18(args.weights); base.fc = nn.Identity(); base.to(device)
    t0 = time.perf_counter()
    Fx = extract_features(base, X, tf, device); Ft = extract_features(base, data.test.x, tf, device)
    t_feat = time.perf_counter() - t0
    fe = {"fold_val_bacc": [], "fold_time_s": [], "feature_time_s": t_feat}
    for fi, (tr, va) in enumerate(folds):
        C.set_seed(fi)
        head = nn.Sequential(nn.Dropout(0.2), nn.Linear(512, C.N_CLASSES))
        mu, sd = Fx[tr].mean(0), Fx[tr].std(0) + 1e-6       # padronizacao dos atributos
        norm = lambda z: (z - mu) / sd
        info = C.train_model(head, C.Split(norm(Fx[tr]), Y[tr]), C.Split(norm(Fx[va]), Y[va]),
                             C.TrainCfg(lr=1e-3, weight_decay=1e-4, max_epochs=200, patience=20,
                                        batch_size=256), device)
        fe["fold_val_bacc"].append(info["best_val_bacc"]); fe["fold_time_s"].append(info["train_time_s"])
        p = C.predict_proba(head, norm(Ft), device)
        fe.setdefault("test_proba", []).append(p.tolist())
        print(f"[FE fold {fi}] val bacc {info['best_val_bacc']:.4f}", flush=True)
    P = np.mean([np.array(p) for p in fe["test_proba"]], 0)
    fe["test_ensemble"] = C.metrics(data.test.y.numpy(), P.argmax(1))
    fe["test_per_fold"] = [C.metrics(data.test.y.numpy(), np.array(p).argmax(1))["bacc"] for p in fe["test_proba"]]
    del fe["test_proba"]
    fe["params_trainable"] = 512 * 8 + 8
    res["FE"] = fe
    C.save_json(res, out_path)
    print("FE: CV", np.mean(fe["fold_val_bacc"]), "teste(comite)", fe["test_ensemble"]["bacc"])

    # ======================= Estrategia FT (fine-tuning) =======================
    ft = res.get("FT", {"folds": []})
    probas = []
    for fi, (tr, va) in enumerate(folds):
        ck = f"{C.RESULTS_DIR}/d_ft_fold{fi}.pt"
        C.set_seed(fi)
        m = build_resnet18(args.weights)
        if os.path.exists(ck) and fi < len(ft["folds"]):      # retomada
            m.load_state_dict(torch.load(ck)); m.to(device)
        else:
            head_p = list(m.fc.parameters()); hid = {id(p) for p in head_p}
            body_p = [p for p in m.parameters() if id(p) not in hid]
            groups = [{"params": body_p, "lr": args.lr / 10}, {"params": head_p, "lr": args.lr}]
            cfg = C.TrainCfg(lr=args.lr, weight_decay=0.05, optimizer="adamw", scheduler="cosine",
                             max_epochs=args.epochs, patience=args.epochs, batch_size=64,
                             weighted_loss=True, label_smoothing=0.05, verbose=True)
            info = C.train_model(m, C.Split(X[tr], Y[tr]), C.Split(X[va], Y[va]), cfg, device,
                                 transform=tf, param_groups=groups)
            torch.save(m.state_dict(), ck)
            ft["folds"].append({k: info[k] for k in ("best_val_bacc", "best_epoch", "epochs_run",
                                                      "train_time_s", "time_per_epoch_s", "history")})
            res["FT"] = ft; C.save_json(res, out_path)
        p = C.predict_proba(m, data.test.x, device, transform=tf, bs=256)
        probas.append(p)
        print(f"[FT fold {fi}] val bacc {ft['folds'][fi]['best_val_bacc']:.4f} "
              f"| teste bacc {C.metrics(data.test.y.numpy(), p.argmax(1))['bacc']:.4f}", flush=True)
    t0 = time.perf_counter(); C.predict_proba(m, data.test.x, device, transform=tf, bs=256)
    ft["test_inference_s_single"] = time.perf_counter() - t0
    y = data.test.y.numpy()
    ft["test_per_fold"] = [C.metrics(y, p.argmax(1)) for p in probas]
    P = np.mean(probas, 0)
    ft["test_ensemble"] = C.metrics(y, P.argmax(1))
    ft["params"] = C.count_params(m)
    ft["cv_val_bacc_mean_std"] = [float(np.mean([f["best_val_bacc"] for f in ft["folds"]])),
                                  float(np.std([f["best_val_bacc"] for f in ft["folds"]]))]
    np.save(f"{C.RESULTS_DIR}/d_ft_test_proba.npy", P)
    res["FT"] = ft
    C.save_json(res, out_path)
    best_fold = int(np.argmax([f["best_val_bacc"] for f in ft["folds"]]))
    C.plot_confusion(ft["test_ensemble"]["cm"], f"{C.FIG_DIR}/d_ft_cm_ensemble.png",
                     f"ResNet-18 FT – comitê de {len(folds)} folds – teste")
    C.plot_confusion(ft["test_per_fold"][best_fold]["cm"], f"{C.FIG_DIR}/d_ft_cm_single.png",
                     f"ResNet-18 FT – fold {best_fold} – teste")
    C.plot_history(ft["folds"][best_fold]["history"], f"{C.FIG_DIR}/d_ft_hist.png",
                   f"ResNet-18 FT – fold {best_fold}")
    print("FT: CV", ft["cv_val_bacc_mean_std"], "teste(comite)", ft["test_ensemble"]["bacc"])


if __name__ == "__main__":
    main()
