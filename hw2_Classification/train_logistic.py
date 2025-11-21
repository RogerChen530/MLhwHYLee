import sys
import numpy as np
from pathlib import Path

RANDOM_SEED = 0
np.random.seed(RANDOM_SEED)


def load_features_labels(x_train_path, y_train_path, x_test_path):
    from pathlib import Path

    def _load(path):
        path = Path(path)
        if path.suffix == ".npy":
            return np.load(path)
        else:
            try:
                return np.loadtxt(path, delimiter=",")
            except Exception:
                return np.loadtxt(path)

    X_train = _load(x_train_path)
    y_train = _load(y_train_path)
    X_test = _load(x_test_path)
    y_train = y_train.reshape(-1)
    return X_train, y_train, X_test


def save_prediction(pred, out_path):
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    ids = np.arange(len(pred), dtype=int)
    pred = (pred > 0.5).astype(int)
    with out_path.open("w") as f:
        f.write("id,label\n")
        for i, p in zip(ids, pred):
            f.write(f"{i},{int(p)}\n")


def sigmoid(z):
    # 避免 overflow
    z = np.clip(z, -30, 30)
    return 1.0 / (1.0 + np.exp(-z))


def logistic_regression_train(X, y, lr=0.1, epochs=200, l2=1e-4):
    n, d = X.shape
    w = np.zeros(d)

    for epoch in range(epochs):
        z = X @ w
        y_hat = sigmoid(z)

        # cross-entropy loss + L2
        eps = 1e-9
        loss = -(y * np.log(y_hat + eps) + (1 - y) * np.log(1 - y_hat + eps)).mean()
        loss += l2 * np.sum(w ** 2)

        # Gradient
        grad = (X.T @ (y_hat - y)) / n + 2 * l2 * w
        # L2 Regularization( 2 * 12 * w )
        # To: 防止w過大、讓模型更保守、避免過擬合

        # Gradient Descent
        w -= lr * grad

        if (epoch + 1) % 20 == 0:
            pred_label = (y_hat > 0.5).astype(int)
            acc = (pred_label == y).mean()
            print(f"[Logistic GD] epoch {epoch+1:3d}, loss={loss:.4f}, acc={acc:.4f}")

    return w


def main():
    if len(sys.argv) != 7:
        print("Usage: python train_logistic.py train.csv test_no_label.csv X_train Y_train X_test output.csv")
        sys.exit(1)

    _, raw_train_path, raw_test_path, x_train_path, y_train_path, x_test_path, out_path = sys.argv

    # 這裡完全不理會 raw_train_path / raw_test_path，因為作業說可以不用所有 argument
    X_train, y_train, X_test = load_features_labels(x_train_path, y_train_path, x_test_path)

    # 標準化（用 train 的 mean/std）
    mean = X_train.mean(axis=0, keepdims=True)
    std = X_train.std(axis=0, keepdims=True) + 1e-8
    X_train_n = (X_train - mean) / std
    X_test_n = (X_test - mean) / std

    # 加 bias term
    X_train_n = np.hstack([np.ones((X_train_n.shape[0], 1)), X_train_n])
    X_test_n = np.hstack([np.ones((X_test_n.shape[0], 1)), X_test_n])

    w = logistic_regression_train(X_train_n, y_train, lr=0.1, epochs=200, l2=1e-4)

    # 存模型（給自己留存用，TA 不一定會用到）
    np.save("model_logistic.npy", {"w": w, "mean": mean, "std": std})

    # 產生預測
    y_test_hat = sigmoid(X_test_n @ w)
    save_prediction(y_test_hat, out_path)


if __name__ == "__main__":
    main()
