#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
HW1 — Prediction script
讀取：
  - 模型: models/linear.npy（含 w, mu, std, window）
  - 資料: test.csv（每 18 列為一筆，右邊 9 欄是 9 小時數值）
輸出：
  - result/ans.csv（id,value）
用法：
  python hw1/src/predict.py --model hw1/models/linear.npy \
                            --test  hw1/data/test.csv \
                            --out   hw1/result/ans.csv
"""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd


def load_model(path: str) -> dict:
    obj = np.load(path, allow_pickle=True).item()
    # 基本檢查
    assert "w" in obj and "mu" in obj and "std" in obj, "模型檔缺少必要鍵值"
    return obj


def load_test_csv(path: str) -> pd.DataFrame:
    """
    大多數 hw1 的 test.csv 沒有標頭；前 2 欄是 meta（id, 變數名），
    右邊最後 9 欄是 9 小時數值。
    """
    df = pd.read_csv(path, header=None, encoding="big5", na_filter=False)
    # 丟掉 meta 欄，只留數值欄
    if df.shape[1] >= 11:
        df = df.iloc[:, 2:].copy()
    elif df.shape[1] >= 10:
        df = df.iloc[:, 1:].copy()
    df = df.replace("NR", 0).astype(float)
    return df


def build_X_test(df: pd.DataFrame, hours: int = 9) -> np.ndarray:
    num_samples = 240
    feat_per_sample = 18
    assert df.shape[0] == num_samples * feat_per_sample, \
        f"test 列數應為 {num_samples*feat_per_sample}，目前 {df.shape[0]}"
    assert df.shape[1] >= hours, f"test 欄數不足 {hours}（9 小時）"

    X = np.empty((num_samples, feat_per_sample * hours), dtype=float)
    for i in range(num_samples):
        block = df.iloc[i*feat_per_sample:(i+1)*feat_per_sample, -hours:]  # (18, 9)
        X[i, :] = block.to_numpy(dtype=float).reshape(-1)                  # (162,)
    return X


def predict_y(model: dict, X: np.ndarray) -> np.ndarray:
    mu, std, w = model["mu"], model["std"], model["w"]
    # 對齊訓練時的標準化
    Xn = (X - mu) / std
    # 加 bias
    Xb = np.hstack([np.ones((Xn.shape[0], 1), dtype=float), Xn])
    yhat = (Xb @ w).reshape(-1)
    return yhat


def save_submission(y: np.ndarray, out_path: str):
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    ids = [f"id_{i}" for i in range(len(y))]
    pd.DataFrame({"id": ids, "value": y}).to_csv(out_path, index=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="path to models/linear.npy")
    ap.add_argument("--test",  required=True, help="path to data/test.csv")
    ap.add_argument("--out",   required=True, help="path to write ans.csv")
    args = ap.parse_args()

    model = load_model(args.model)
    df = load_test_csv(args.test)
    X = build_X_test(df, hours=int(model.get("window", 9)))
    yhat = predict_y(model, X)
    save_submission(yhat, args.out)

    print(f"[OK] wrote predictions -> {args.out} ; rows={len(yhat)}")


if __name__ == "__main__":
    main()
