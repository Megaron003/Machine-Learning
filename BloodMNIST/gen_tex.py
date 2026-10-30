"""
gen_tex.py -- Gera, a partir dos JSONs em results/, os arquivos LaTeX com
macros numericas e tabelas usados pelo relatorio (report/), eliminando
transcricao manual de numeros.

Uso:  python gen_tex.py   (apos run_all.sh)
"""
import numpy as np

import common as C

R = C.RESULTS_DIR
OUT = "report/gen"


def f4(x):
    return f"{x:.4f}"


def pct(x):
    return f"{100*x:.2f}"


def main():
    import os
    os.makedirs(OUT, exist_ok=True)
    a = C.load_json(f"{R}/a_mlp.json"); b = C.load_json(f"{R}/b_summary.json")
    c = C.load_json(f"{R}/c_cnn.json"); d = C.load_json(f"{R}/d_transfer.json")
    ds = C.load_json(f"{R}/f_dataset.json"); cmp_ = C.load_json(f"{R}/compare.json")
    M = {}

    # ---------- dados ----------
    M["meanR"], M["meanG"], M["meanB"] = [f"{v:.4f}" for v in ds["train_mean"]]
    M["stdR"], M["stdG"], M["stdB"] = [f"{v:.4f}" for v in ds["train_std"]]

    # ---------- (a) ----------
    s = a["selected"]
    M.update({"mlpH": a["best_cfg"]["hidden"], "mlpDrop": a["best_cfg"]["dropout"],
              "mlpWD": a["best_cfg"]["weight_decay"],
              "mlpAcc": pct(s["acc"]), "mlpBacc": pct(s["bacc"]), "mlpFone": pct(s["f1_macro"]),
              "mlpFw": pct(s["f1_weighted"]), "mlpVal": pct(s["val_bacc"]),
              "mlpBaccMS": f"{pct(a['summary_mean_std']['bacc'][0])} \\pm {pct(a['summary_mean_std']['bacc'][1])}",
              "mlpFoneMS": f"{pct(a['summary_mean_std']['f1_macro'][0])} \\pm {pct(a['summary_mean_std']['f1_macro'][1])}",
              "mlpParams": f"{a['params']['total']:,}".replace(",", "THOUS"),
              "mlpTime": f"{s['train_time_s']:.1f}", "mlpEpochs": s["epochs"], "mlpBestEp": s["best_epoch"]})
    rows = []
    for H in sorted({r["hidden"] for r in a["stage1"]}):
        rr = {r["act"]: r for r in a["stage1"] if r["hidden"] == H}
        rows.append(f"{H} & {pct(rr['relu']['val_bacc'])} & {pct(rr['tanh']['val_bacc'])} & "
                    f"{(2352+1)*H + 2*H + (H+1)*8:,} \\\\".replace(",", "THOUS"))
    open(f"{OUT}/tab_mlp_stage1.tex", "w").write("\n".join(rows))
    rows = []
    best2 = max(a["stage2"], key=lambda r: r["val_bacc"])
    for p in sorted({r["dropout"] for r in a["stage2"]}):
        for wd in sorted({r["weight_decay"] for r in a["stage2"]}):
            v = {r["weighted_loss"]: r for r in a["stage2"] if r["dropout"] == p and r["weight_decay"] == wd}
            cells = []
            for wl in (True, False):
                t = pct(v[wl]["val_bacc"])
                cells.append(f"\\textbf{{{t}}}" if v[wl] is best2 else t)
            wds = "0" if wd == 0 else f"$10^{{{int(np.log10(wd))}}}$"
            rows.append(f"{p:.2f} & {wds} & {cells[0]} & {cells[1]} \\\\")
    open(f"{OUT}/tab_mlp_stage2.tex", "w").write("\n".join(rows))
    wins = sum(1 for p in {r["dropout"] for r in a["stage2"]} for wd in {r["weight_decay"] for r in a["stage2"]}
               if [r for r in a["stage2"] if r["dropout"] == p and r["weight_decay"] == wd and r["weighted_loss"]][0]["val_bacc"]
               > [r for r in a["stage2"] if r["dropout"] == p and r["weight_decay"] == wd and not r["weighted_loss"]][0]["val_bacc"])
    M["mlpWeightedWins"] = wins
    M["mlpSeeds"] = "; ".join(f"{pct(f['bacc'])}" for f in a["finals"])

    # ---------- (b) ----------
    Mv = np.array(b["mean_val_bacc"]); Sv = np.array(b["std_val_bacc"]); TR = np.array(b["train_bacc"])
    T = np.array(b["time_s"]); P = np.array(b["params"])
    bi, bj = b["k_list"].index(b["best"]["k"]), b["n_list"].index(b["best"]["n"])
    rows = []
    for i, k in enumerate(b["k_list"]):
        cells = []
        for j in range(len(b["n_list"])):
            t = f"{pct(Mv[i, j])}\\,{{\\scriptsize$\\pm${100*Sv[i,j]:.2f}}}"
            cells.append(f"\\textbf{{{t}}}" if (i, j) == (bi, bj) else t)
        rows.append(f"{k} & " + " & ".join(cells) + " \\\\")
    open(f"{OUT}/tab_cnn_sweep.tex", "w").write("\n".join(rows))
    rows = []
    for i, k in enumerate(b["k_list"]):
        rows.append(f"{k} & " + " & ".join(f"{pct(TR[i, j])}" for j in range(len(b["n_list"]))) + " \\\\")
    open(f"{OUT}/tab_cnn_train.tex", "w").write("\n".join(rows))
    M.update({"bBestN": b["best"]["n"], "bBestK": b["best"]["k"], "bBestVal": pct(b["best"]["val_bacc"]),
              "bBestStd": f"{100*b['best']['std']:.2f}",
              "bTotalTime": f"{sum(__import__('json').loads(l)['time_s'] for l in open(f'{R}/b_sweep.jsonl'))/60:.0f}",
              "bMinVal": pct(np.nanmin(Mv)), "bTimeBest": f"{T[bi, bj]:.0f}"})
    # ganho medio por dobra de n e efeito de k
    M["bKspread"] = f"{100*(Mv[:, -2:].max() - Mv[:, -2:].min()):.2f}"
    M["bNgain"] = f"{100*(Mv[:, 3].mean() - Mv[:, 0].mean()):.2f}"

    # ---------- (c) ----------
    s = c["selected"]; es = c["error_stats"]
    M.update({"cAcc": pct(s["acc"]), "cBacc": pct(s["bacc"]), "cFone": pct(s["f1_macro"]),
              "cFw": pct(s["f1_weighted"]), "cVal": pct(s["val_bacc"]),
              "cBaccMS": f"{pct(c['summary_mean_std']['bacc'][0])} \\pm {pct(c['summary_mean_std']['bacc'][1])}",
              "cFoneMS": f"{pct(c['summary_mean_std']['f1_macro'][0])} \\pm {pct(c['summary_mean_std']['f1_macro'][1])}",
              "cParams": f"{c['params']['total']:,}".replace(",", "THOUS"), "cTime": f"{s['train_time_s']:.1f}",
              "cEpochs": s["epochs"], "cBestEp": s["best_epoch"], "cNerr": es["n_errors"],
              "cConfErr": f"{es['conf_err_median']:.3f}", "cConfOk": f"{es['conf_ok_median']:.4f}",
              "cFracHigh": pct(es["frac_err_conf_gt_0.9"]), "cPtrue": f"{es['p_true_on_err_median']:.3f}",
              "cTopTwo": pct(es["true_in_top2_on_err"]), "cECE": f"{es['ece']:.4f}"})
    rows = []
    for e in c["errors"]:
        pr = e["proba"]
        cells = []
        for k, v in enumerate(pr):
            t = f"{v:.3f}"
            if k == e["pred"]:
                t = f"\\textcolor{{red!70!black}}{{\\textbf{{{t}}}}}"
            elif k == e["true"]:
                t = f"\\textcolor{{green!45!black}}{{\\textbf{{{t}}}}}"
            cells.append(t)
        rows.append(f"{e['idx']} & {C.CLASS_SHORT[e['true']]} & {C.CLASS_SHORT[e['pred']]} & " + " & ".join(cells) + " \\\\")
    open(f"{OUT}/tab_errors.tex", "w").write("\n".join(rows))

    # ---------- (d) ----------
    fe, ft = d["FE"], d["FT"]
    M.update({"feCV": f"{pct(np.mean(fe['fold_val_bacc']))} \\pm {pct(np.std(fe['fold_val_bacc']))}",
              "feTest": pct(fe["test_ensemble"]["bacc"]), "feFone": pct(fe["test_ensemble"]["f1_macro"]),
              "feAcc": pct(fe["test_ensemble"]["acc"]),
              "feTestMS": f"{pct(np.mean(fe['test_per_fold']))} \\pm {pct(np.std(fe['test_per_fold']))}",
              "feFeatTime": f"{fe['feature_time_s']:.0f}", "feHeadTime": f"{np.mean(fe['fold_time_s']):.1f}",
              "ftCV": f"{pct(ft['cv_val_bacc_mean_std'][0])} \\pm {pct(ft['cv_val_bacc_mean_std'][1])}",
              "ftEnsBacc": pct(ft["test_ensemble"]["bacc"]), "ftEnsAcc": pct(ft["test_ensemble"]["acc"]),
              "ftEnsFone": pct(ft["test_ensemble"]["f1_macro"]), "ftEnsFw": pct(ft["test_ensemble"]["f1_weighted"]),
              "ftFoldBaccMS": f"{pct(np.mean([m['bacc'] for m in ft['test_per_fold']]))} \\pm {pct(np.std([m['bacc'] for m in ft['test_per_fold']]))}",
              "ftFoldFoneMS": f"{pct(np.mean([m['f1_macro'] for m in ft['test_per_fold']]))} \\pm {pct(np.std([m['f1_macro'] for m in ft['test_per_fold']]))}",
              "ftFoldAccMS": f"{pct(np.mean([m['acc'] for m in ft['test_per_fold']]))} \\pm {pct(np.std([m['acc'] for m in ft['test_per_fold']]))}",
              "ftParams": f"{ft['params']['total']:,}".replace(",", "THOUS"),
              "ftEpochTime": f"{np.mean([f['time_per_epoch_s'] for f in ft['folds']]):.0f}",
              "ftFoldTime": f"{np.mean([f['train_time_s'] for f in ft['folds']])/60:.1f}",
              "ftTotalTime": f"{np.sum([f['train_time_s'] for f in ft['folds']])/60:.1f}",
              "ftEpochs": d["args"]["epochs"], "ftSize": d["args"]["size"], "ftK": len(ft["folds"])})
    rows = []
    for i, f in enumerate(ft["folds"]):
        m = ft["test_per_fold"][i]
        rows.append(f"{i+1} & {pct(f['best_val_bacc'])} & {f['best_epoch']} & {pct(fe['fold_val_bacc'][i])} & "
                    f"{pct(m['acc'])} & {pct(m['bacc'])} & {pct(m['f1_macro'])} & {f['train_time_s']/60:.1f} \\\\")
    open(f"{OUT}/tab_cv.tex", "w").write("\n".join(rows))

    # ---------- comparacao ----------
    rows = []
    for r in cmp_:
        ms = "--" if r["bacc_ms"] is None else f"${pct(r['bacc_ms'][0])} \\pm {pct(r['bacc_ms'][1])}$"
        params = f"{r['params']:,}".replace(",", "THOUS")
        rows.append(f"{r['model']} & {params} & {pct(r['acc'])} & {pct(r['bacc'])} & {pct(r['f1'])} & "
                    f"{ms} & {r['train_s']/60:.1f} & {r['infer_s']:.2f} \\\\")
    open(f"{OUT}/tab_compare.tex", "w").write("\n".join(rows))
    # F1 por classe
    rows = []
    for k in range(C.N_CLASSES):
        vals = [r["f1pc"][k] for r in cmp_]
        bst = int(np.argmax(vals))
        cells = [f"\\textbf{{{pct(v)}}}" if i == bst else pct(v) for i, v in enumerate(vals)]
        rows.append(f"{C.CLASS_NAMES_PT[k]} ({C.CLASS_SHORT[k]}) & " + " & ".join(cells) + " \\\\")
    open(f"{OUT}/tab_f1pc.tex", "w").write("\n".join(rows))

    with open(f"{OUT}/macros.tex", "w") as fh:
        for k, v in M.items():
            fh.write(f"\\newcommand{{\\{k}}}{{{v}}}\n")
    # convencao brasileira: virgula decimal ({,} funciona em modo texto e matematico)
    # e ponto como separador de milhar
    import glob, re
    for fn in glob.glob(f"{OUT}/*.tex"):
        txt = open(fn).read()
        txt = re.sub(r"(\d)\.(\d)", r"\1{,}\2", txt).replace("THOUS", ".")
        open(fn, "w").write(txt)
    print(f"{len(M)} macros escritas em {OUT}/macros.tex")


if __name__ == "__main__":
    main()
