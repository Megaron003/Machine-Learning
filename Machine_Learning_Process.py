"""
IA048 - Aprendizado de Maquina (2S2026)

Atividade 1  -  Regressao Linear: Predição de Serie Temporal de Trafego Aéreo
Turma A  -  Prof. Levy Boccato

Este script implementa a predição L=1 passo a frente do numero total de voos
(domesticos + internacionais) da base US Airline Traffic Data (2003-2023),
usando um preditor linear ajustado por minimos quadrados irrestrito (OLS).
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ----------------------------------------------------------------------------
# 0. Configuration
# ----------------------------------------------------------------------------
CSV_PATH = "/home/ghost/Mestrado/Machine_Learning/Listas/Lista_I/Data/air traffic.csv"
OUTPUT_DIR = "/home/ghost/Mestrado/Machine_Learning/Listas/Lista_I/Figuras"

L = 1            # horizonte de predicao (fixo, conforme enunciado)
K_MAX = 24       # numero maximo de entradas (lags) testado

# ----------------------------------------------------------------------------
# EVALUATION SPLITS
# ----------------------------------------------------------------------------
N_VAL_YEARS = 3   # 2017-2019 como validacao interna; 2003-2016 como treino
# ----------------------------------------------------------------------------


# ----------------------------------------------------------------------------
# 1. Loading and Prepare Data
# ----------------------------------------------------------------------------
def load_series(csv_path: str) -> pd.Series:
    """Carrega a serie do numero total de voos (Dom + Int), mensal."""
    df = pd.read_csv(csv_path)
    # As colunas numericas vem formatadas com separador de milhar (ex: "43,032,450")
    for col in ["Dom_Flt", "Int_Flt", "Flt"]:
        df[col] = df[col].astype(str).str.replace(",", "", regex=False).astype(int)
    df["Date"] = pd.to_datetime(dict(year=df.Year, month=df.Month, day=1))
    df = df.sort_values("Date").reset_index(drop=True)
    serie = pd.Series(df["Flt"].values, index=df["Date"], name="Flt")
    return serie


def indices_for_range(dates: pd.DatetimeIndex, start: str, end: str) -> np.ndarray:
    mask = (dates >= pd.Timestamp(start)) & (dates <= pd.Timestamp(end))
    return np.where(mask)[0]


# ----------------------------------------------------------------------------
# 2. Interface Construction
# ----------------------------------------------------------------------------
def build_windows(x: np.ndarray, K: int, L: int, target_idx: np.ndarray):
    """
    Monta a matriz de entradas X e o vetor de saidas y para os indices em
    `target_idx` (posicoes n de x(n) que se deseja prever), considerando

        x(n) = [x(n-L), x(n-L-1), ..., x(n-L-K+1)]^T  ->  y = x(n)

    Indices cujo vetor de entrada dependa de amostras anteriores ao inicio
    da serie completa (n-L-K+1 < 0) sao descartados. Isso permite que, por
    exemplo, os primeiros alvos do conjunto de teste usem como entrada
    amostras do fim do conjunto de treinamento/validacao (continuidade da
    serie), evitando perda desnecessaria de amostras nas bordas dos splits.
    """
    X_rows, y_vals, used_idx = [], [], []
    for n in target_idx:
        lo = n - L - K + 1
        hi = n - L
        if lo < 0:
            continue
        X_rows.append(x[lo:hi + 1][::-1])  # x(n-L), x(n-L-1), ..., x(n-L-K+1)
        y_vals.append(x[n])
        used_idx.append(n)
    if len(X_rows) == 0:
        return np.empty((0, K)), np.empty((0,)), np.array([], dtype=int)
    return np.array(X_rows), np.array(y_vals), np.array(used_idx)


def fit_ols(X: np.ndarray, y: np.ndarray):
    """Ajusta y = w^T x + w0 via minimos quadrados irrestrito (solucao fechada)."""
    X_aug = np.hstack([np.ones((X.shape[0], 1)), X])
    theta, *_ = np.linalg.lstsq(X_aug, y, rcond=None)
    return theta[1:], theta[0]  # w, w0


def predict_ols(X: np.ndarray, w: np.ndarray, w0: float) -> np.ndarray:
    return X @ w + w0


# ----------------------------------------------------------------------------
# 3. Metrics
# ----------------------------------------------------------------------------
def rmse(y_true, y_pred):
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mape(y_true, y_pred):
    return float(np.mean(np.abs((y_true - y_pred) / y_true)) * 100)


def r2_score_manual(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return float(1 - ss_res / ss_tot)


# ----------------------------------------------------------------------------
# 4. Routines for K Selection and Evaluation
# ----------------------------------------------------------------------------
def compute_val_curve(x, dates, train_range, val_range, K_max, L):
    """RMSE of validation as a function of K, for K = 1..K_max."""
    train_idx = indices_for_range(dates, *train_range)
    val_idx = indices_for_range(dates, *val_range)
    rmses = []
    for K in range(1, K_max + 1):
        Xtr, ytr, _ = build_windows(x, K, L, train_idx)
        Xval, yval, _ = build_windows(x, K, L, val_idx)
        if len(Xtr) == 0 or len(Xval) == 0:
            rmses.append(np.nan)
            continue
        w, w0 = fit_ols(Xtr, ytr)
        yhat = predict_ols(Xval, w, w0)
        rmses.append(rmse(yval, yhat))
    return np.array(rmses)


def evaluate_split(x, dates, fit_range, test_range, K, L):
    """Ajusta em `fit_range` e avalia em `test_range`, para um K fixo."""
    fit_idx = indices_for_range(dates, *fit_range)
    test_idx = indices_for_range(dates, *test_range)
    Xfit, yfit, _ = build_windows(x, K, L, fit_idx)
    Xtest, ytest, used_test = build_windows(x, K, L, test_idx)
    w, w0 = fit_ols(Xfit, yfit)
    yhat = predict_ols(Xtest, w, w0)
    metrics = {
        "RMSE": rmse(ytest, yhat),
        "MAPE": mape(ytest, yhat),
        "R2": r2_score_manual(ytest, yhat),
    }
    return dates[used_test], ytest, yhat, metrics, (w, w0)


# ----------------------------------------------------------------------------
# 5. Graphics
# ----------------------------------------------------------------------------
def plot_series(dates, x, title, path):
    plt.figure(figsize=(11, 4.5))
    plt.plot(dates, x)
    plt.title(title)
    plt.xlabel("Ano")
    plt.ylabel("Numero total de voos (Dom + Int)")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def plot_val_curve(K_values, rmses, title, path):
    plt.figure(figsize=(8, 4.5))
    plt.plot(K_values, rmses, marker="o")
    best_k = K_values[np.nanargmin(rmses)]
    plt.axvline(best_k, color="red", linestyle="--", alpha=0.6, label=f"K* = {best_k}")
    plt.title(title)
    plt.xlabel("K (numero de entradas)")
    plt.ylabel("RMSE (validacao)")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def plot_prediction(dates, y_true, y_pred, title, path, metrics=None):
    plt.figure(figsize=(11, 4.5))
    plt.plot(dates, y_true, label="Real", marker="o", markersize=3)
    plt.plot(dates, y_pred, label="Predito", marker="x", markersize=3)
    subtitle = ""
    if metrics:
        subtitle = f"  (RMSE={metrics['RMSE']:.0f} | MAPE={metrics['MAPE']:.2f}% | R2={metrics['R2']:.4f})"
    plt.title(title + subtitle)
    plt.xlabel("Data")
    plt.ylabel("Numero total de voos (Dom + Int)")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


# ----------------------------------------------------------------------------
# 6. Main Pipeline
# ----------------------------------------------------------------------------
def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    serie = load_series(CSV_PATH)
    x = serie.values.astype(float)
    dates = serie.index

    # ---------- Item (a): serie completa ----------
    plot_series(dates, x, "Trafego aereo total (EUA) - Jan/2003 a Set/2023",
                os.path.join(OUTPUT_DIR, "a_serie_completa.png"))
    print("=" * 72)
    print(f"Serie carregada: {len(x)} amostras, de {dates[0].date()} a {dates[-1].date()}")
    print("Item (a): grafico salvo em a_serie_completa.png")
    print("Discussao sugerida sobre as tres faixas de comportamento:")
    print(" - Jan/2003-Ago/2008: recuperacao pos-11/09/2001 e crescimento")
    print("   organico do setor aereo ate a vespera da crise financeira.")
    print(" - Set/2008-Dez/2019: colapso do Lehman Brothers (set/2008) e a")
    print("   crise financeira global reduzem drasticamente a demanda; a")
    print("   partir dai, retomada gradual e crescimento ate o pre-pandemia.")
    print(" - Jan/2020-Set/2023: pandemia de COVID-19 (declarada mar/2020)")
    print("   provoca queda abrupta no trafego aereo, seguida de recuperacao")
    print("   lenta e parcial ate o final da amostra.")
    print("=" * 72)

    # ---------- Item (b): treino+val = 2003-2019 / teste = 2020-2023 ----------
    dev_start, dev_end = "2003-01-01", "2019-12-31"
    test_start, test_end = "2020-01-01", "2023-09-30"

    train_end_b = f"{2020 - N_VAL_YEARS - 1}-12-31"
    val_start_b = f"{2020 - N_VAL_YEARS}-01-01"
    train_range_b = (dev_start, train_end_b)
    val_range_b = (val_start_b, dev_end)

    K_values = np.arange(1, K_MAX + 1)
    rmses_b = compute_val_curve(x, dates, train_range_b, val_range_b, K_MAX, L)
    plot_val_curve(K_values, rmses_b, "Item (b1): RMSE de validacao vs K",
                    os.path.join(OUTPUT_DIR, "b1_rmse_vs_K.png"))
    K_best_b = int(K_values[np.nanargmin(rmses_b)])
    print(f"Item (b1): treino={train_range_b} val={val_range_b}")
    print(f"Item (b1): melhor K = {K_best_b} (RMSE validacao = {np.nanmin(rmses_b):.1f})")

    test_dates_b, ytest_b, yhat_b, metrics_b, _ = evaluate_split(
        x, dates, (dev_start, dev_end), (test_start, test_end), K_best_b, L)
    plot_prediction(test_dates_b, ytest_b, yhat_b,
                     f"Item (b2): Teste 2020-2023 (K={K_best_b})",
                     os.path.join(OUTPUT_DIR, "b2_teste_2020_2023.png"), metrics_b)
    print(f"Item (b2): n_teste={len(ytest_b)} RMSE={metrics_b['RMSE']:.1f} "
          f"MAPE={metrics_b['MAPE']:.2f}% R2={metrics_b['R2']:.4f}")

    # Item (b3): apenas 2022-2023
    mask_2223 = test_dates_b >= pd.Timestamp("2022-01-01")
    dates_2223 = test_dates_b[mask_2223]
    y_2223, yhat_2223 = ytest_b[mask_2223], yhat_b[mask_2223]
    metrics_2223 = {
        "RMSE": rmse(y_2223, yhat_2223),
        "MAPE": mape(y_2223, yhat_2223),
        "R2": r2_score_manual(y_2223, yhat_2223),
    }
    plot_prediction(dates_2223, y_2223, yhat_2223,
                     f"Item (b3): Teste 2022-2023 (K={K_best_b})",
                     os.path.join(OUTPUT_DIR, "b3_teste_2022_2023.png"), metrics_2223)
    print(f"Item (b3): n_teste={len(y_2223)} RMSE={metrics_2223['RMSE']:.1f} "
          f"MAPE={metrics_2223['MAPE']:.2f}% R2={metrics_2223['R2']:.4f}")

    # ---------- Item (c): treino=2003-2019 / val=2020-2021 / teste=2022-2023 ----------
    train_c = ("2003-01-01", "2019-12-31")
    val_c = ("2020-01-01", "2021-12-31")
    test_c = ("2022-01-01", "2023-09-30")

    rmses_c = compute_val_curve(x, dates, train_c, val_c, K_MAX, L)
    plot_val_curve(K_values, rmses_c, "Item (c): RMSE de validacao vs K (val=2020-2021)",
                    os.path.join(OUTPUT_DIR, "c_rmse_vs_K.png"))
    K_best_c = int(K_values[np.nanargmin(rmses_c)])
    print(f"Item (c): melhor K = {K_best_c} (RMSE validacao = {np.nanmin(rmses_c):.1f})")

    fit_range_c = (train_c[0], val_c[1])  # treino + validacao combinados: 2003-2021
    test_dates_c, ytest_c, yhat_c, metrics_c, _ = evaluate_split(
        x, dates, fit_range_c, test_c, K_best_c, L)
    plot_prediction(test_dates_c, ytest_c, yhat_c,
                     f"Item (c): Teste 2022-2023 (K={K_best_c})",
                     os.path.join(OUTPUT_DIR, "c_teste_2022_2023.png"), metrics_c)
    print(f"Item (c): n_teste={len(ytest_c)} RMSE={metrics_c['RMSE']:.1f} "
          f"MAPE={metrics_c['MAPE']:.2f}% R2={metrics_c['R2']:.4f}")

    print("=" * 72)
    print("Comparacao (b3) vs (c), ambos avaliados em 2022-2023:")
    print(f"  (b3) K={K_best_b}: RMSE={metrics_2223['RMSE']:.1f} "
          f"MAPE={metrics_2223['MAPE']:.2f}% R2={metrics_2223['R2']:.4f}")
    print(f"  (c)  K={K_best_c}: RMSE={metrics_c['RMSE']:.1f} "
          f"MAPE={metrics_c['MAPE']:.2f}% R2={metrics_c['R2']:.4f}")
    print("=" * 72)


if __name__ == "__main__":
    main()