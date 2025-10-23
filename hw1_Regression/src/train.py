#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
HW1 — Linear Regression with Gradient Descent (手刻GD)
- 從 train.csv 產生 (X, y)
- 可選標準化
- 用 Adagrad 版的 GD 訓練
- 存成 models/linear.npy （包含 w, mu, std, 其他資訊）
用法：
  python hw1/src/train.py --train hw1/data/train.csv --out hw1/models/linear.npy \
                          --lr 0.1 --epochs 4000 --lambda 0.0 --standardize
"""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd


# -------------------------------
# 1) 讀取訓練檔 & 前處理
# -------------------------------
def load_train_csv(path: str) -> pd.DataFrame:
    """
    多數版本需要 big5；若報編碼錯誤可改成 encoding=None 或 'utf-8'.
    資料格式：每月 18 列（18個觀測），每列 20天×24時 = 480 欄。
    前 3 欄通常是 meta（Index, Date, 測站），要丟掉。
    'NR' 視為 0。
    """
    df = pd.read_csv(path, encoding="big5")
    # 丟掉前3欄，只留數值欄
    df = df.iloc[:, 3:].copy()
    # 文字 'NR' -> 0，再轉 float
    df = df.replace("NR", 0).astype(float)
    return df


# -------------------------------
# 2) 組 X, y（9小時 → 第10小時PM2.5）
# -------------------------------
def make_xy_from_months(df: pd.DataFrame, window: int = 9, pm25_row: int = 9):
    """
    df: shape = (12*18, 480)，每 18 列是一個月（18個感測），共12個月。
    產生：
      X: (12*(480-9), 18*9)
      y: (同筆數, 1)  = 第10小時的 PM2.5
    """
    n_features = 18
    hours_per_month = df.shape[1]          # 預期 480
    months = df.shape[0] // n_features     # 預期 12

    X_list, y_list = [], []
    for m in range(months):
        block = df.iloc[m*n_features:(m+1)*n_features, :].to_numpy(dtype=float)  # (18, 480)
        # 逐小時滑窗：取連續9小時做特徵，第10小時PM2.5當答案
        for t in range(hours_per_month - window):
            feat9 = block[:, t:t+window].reshape(-1)       # (18*9,)
            target = block[pm25_row, t+window]             # 第10小時 PM2.5
            X_list.append(feat9)
            y_list.append(target)

    X = np.array(X_list)                    # (N, 162)
    y = np.array(y_list).reshape(-1, 1)     # (N, 1)

    # 基本檢查
    assert X.shape[1] == 18*window, f"X 維度應為 18*{window}，目前 {X.shape[1]}"
    return X, y


# -------------------------------
# 3) 標準化（可選）
# -------------------------------
def standardize(X: np.ndarray):
    mu = X.mean(axis=0, keepdims=True)
    std = X.std(axis=0, keepdims=True)
    std[std < 1e-8] = 1.0
    Xn = (X - mu) / std
    return Xn, mu, std


# -------------------------------
# 4) 手刻 GD（Adagrad）
# -------------------------------
def train_linear_gd(X: np.ndarray, y: np.ndarray,
                    lr: float = 0.1, epochs: int = 4000, lam: float = 0.0,
                    use_adagrad: bool = True, seed: int = 42):
    """
    X: (N, D)  這裡假設已經標準化好（若有）
    y: (N, 1)
    回傳：
      w: (D+1, 1)  # 含 bias
    """
    rng = np.random.default_rng(seed)
    n, d = X.shape
    # 加 bias
    Xb = np.hstack([np.ones((n, 1), dtype=float), X])  # (N, D+1)
    w = np.zeros((d + 1, 1), dtype=float)
    g2 = np.zeros_like(w)
    eps = 1e-8

    for t in range(1, epochs + 1):
        pred = Xb @ w                      # (N,1)
        err = pred - y                     # (N,1)
        # ridge 對 bias 不正則
        grad = (Xb.T @ err) / n + lam * np.vstack([[[0.0]], w[1:]])
        if use_adagrad:
            g2 += grad ** 2
            w -= (lr / (np.sqrt(g2) + eps)) * grad
        else:
            w -= lr * grad

        if t % 500 == 0:
            rmse = float(np.sqrt((err ** 2).mean()))
            print(f"[{t}] rmse={rmse:.4f}")

    return w


# -------------------------------
# 5) 主程式
# -------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", required=True, help="path to train.csv")
    ap.add_argument("--out",   required=True, help="path to save model .npy")
    ap.add_argument("--lr", type=float, default=0.1)
    ap.add_argument("--epochs", type=int, default=4000)
    ap.add_argument("--lambda", dest="lam", type=float, default=0.0)
    ap.add_argument("--standardize", action="store_true",
                    help="enable feature standardization (recommended)")
    args = ap.parse_args()

    # 讀檔 & 組樣本
    df = load_train_csv(args.train)
    X, y = make_xy_from_months(df, window=9, pm25_row=9)

    # 標準化（可選）
    if args.standardize:
        Xn, mu, std = standardize(X)
    else:
        Xn = X
        mu = np.zeros((1, X.shape[1]), dtype=float)
        std = np.ones((1, X.shape[1]), dtype=float)

    # 訓練（手刻 GD / Adagrad）
    w = train_linear_gd(Xn, y, lr=args.lr, epochs=args.epochs, lam=args.lam, use_adagrad=True)

    # 存模型（助教測試時 hw1.sh 會讀你預先訓練好的 .npy）
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(out_path, {
        "w": w,           # (D+1,1) 含 bias
        "mu": mu,         # (1,D)   若沒標準化則全0
        "std": std,       # (1,D)   若沒標準化則全1
        "window": 9,      # 9小時視窗，給 predict 參考
        "target": "PM2.5"
    }, allow_pickle=True)

    print(f"[OK] saved model -> {out_path}")
    print(f"X shape = {X.shape}, y shape = {y.shape}, w shape = {w.shape}")


if __name__ == "__main__":
    main()
