#!/usr/bin/env python3
"""
Train a wider MNIST MLP and export weights as raw f32 binary files.

Architecture: 784 → HIDDEN → HIDDEN → 10 (logits)
Default HIDDEN=48 — the largest dimension where JIT still beats Accelerate (1.4×).

Gate 22: Tiled inference engine needs wider hidden layers to exercise
the tiled GEMM infrastructure from Gate 21.

Output files (all little-endian f32) in weights_wide/:
  w1.bin, b1.bin    — Layer 1: 784×HIDDEN, HIDDEN
  w2.bin, b2.bin    — Layer 2: HIDDEN×HIDDEN, HIDDEN
  w3.bin, b3.bin    — Layer 3: HIDDEN×HIDDEN_PAD (padded from HIDDEN×10), HIDDEN_PAD
  test_images.bin   — 16×784 (batch of 16)
  test_images_t.bin — 784×16 (pre-transposed)
  test_labels.bin   — 16 labels
  test_logits.bin   — 16×10 reference logits
  test_hidden1.bin  — 16×HIDDEN reference hidden1
  test_hidden2.bin  — 16×HIDDEN reference hidden2
  config.txt        — HIDDEN dim for Rust to read
"""

import os
import numpy as np

HIDDEN = 48  # Must be multiple of 16 for SME tiling
HIDDEN_PAD = ((HIDDEN + 15) // 16) * 16  # Round up to 16 for output padding
# For output layer: pad 10 → 16 columns, and rows = HIDDEN (already multiple of 16)
OUT_N_PAD = 16  # Pad output columns from 10 → 16

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
WEIGHTS_DIR = os.path.join(SCRIPT_DIR, "weights_wide")
os.makedirs(WEIGHTS_DIR, exist_ok=True)


def save_f32(path, data):
    flat = data.astype(np.float32).flatten()
    with open(path, "wb") as f:
        f.write(flat.tobytes())
    print(f"  {os.path.basename(path)}: {flat.shape[0]} floats ({os.path.getsize(path)} bytes)")


def save_u8(path, data):
    flat = data.astype(np.uint8).flatten()
    with open(path, "wb") as f:
        f.write(flat.tobytes())
    print(f"  {os.path.basename(path)}: {flat.shape[0]} bytes")


def load_mnist():
    try:
        from sklearn.datasets import fetch_openml
        mnist = fetch_openml("mnist_784", version=1, as_frame=False, parser="auto")
        X = mnist.data.astype(np.float32) / 255.0
        y = mnist.target.astype(np.int64)
        return X[:60000], y[:60000], X[60000:], y[60000:]
    except ImportError:
        print("sklearn not available, generating synthetic MNIST-like data...")
        return generate_synthetic()


def generate_synthetic():
    rng = np.random.RandomState(42)
    X_train = rng.randn(60000, 784).astype(np.float32) * 0.3
    y_train = rng.randint(0, 10, 60000).astype(np.int64)
    for i in range(60000):
        X_train[i, y_train[i] * 78:(y_train[i] + 1) * 78] += 1.0
    X_train = np.clip(X_train, 0, 1)
    X_test = rng.randn(10000, 784).astype(np.float32) * 0.3
    y_test = rng.randint(0, 10, 10000).astype(np.int64)
    for i in range(10000):
        X_test[i, y_test[i] * 78:(y_test[i] + 1) * 78] += 1.0
    X_test = np.clip(X_test, 0, 1)
    return X_train, y_train, X_test, y_test


def relu(x):
    return np.maximum(0, x)


def softmax(x):
    e = np.exp(x - x.max(axis=-1, keepdims=True))
    return e / e.sum(axis=-1, keepdims=True)


def train_mlp(X_train, y_train, X_test, y_test):
    rng = np.random.RandomState(42)
    h = HIDDEN

    W1 = rng.randn(784, h).astype(np.float32) * np.sqrt(2.0 / 784)
    b1 = np.zeros(h, dtype=np.float32)
    W2 = rng.randn(h, h).astype(np.float32) * np.sqrt(2.0 / h)
    b2 = np.zeros(h, dtype=np.float32)
    W3 = rng.randn(h, 10).astype(np.float32) * np.sqrt(2.0 / h)
    b3 = np.zeros(10, dtype=np.float32)

    lr = 0.1
    batch_size = 128
    epochs = 30
    n = X_train.shape[0]
    params = 784 * h + h + h * h + h + h * 10 + 10
    print(f"\nTraining: 784→{h}→{h}→10 MLP, {epochs} epochs, lr={lr}, batch={batch_size}")
    print(f"  Parameters: {params}")

    for epoch in range(epochs):
        perm = rng.permutation(n)
        X_shuf, y_shuf = X_train[perm], y_train[perm]

        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            X_b, y_b = X_shuf[start:end], y_shuf[start:end]
            bs = end - start

            z1 = X_b @ W1 + b1
            a1 = relu(z1)
            z2 = a1 @ W2 + b2
            a2 = relu(z2)
            z3 = a2 @ W3 + b3
            probs = softmax(z3)

            dz3 = probs.copy()
            dz3[np.arange(bs), y_b] -= 1
            dz3 /= bs

            dW3 = a2.T @ dz3; db3 = dz3.sum(axis=0)
            da2 = dz3 @ W3.T
            dz2 = da2 * (z2 > 0).astype(np.float32)
            dW2 = a1.T @ dz2; db2 = dz2.sum(axis=0)
            da1 = dz2 @ W2.T
            dz1 = da1 * (z1 > 0).astype(np.float32)
            dW1 = X_b.T @ dz1; db1 = dz1.sum(axis=0)

            W1 -= lr * dW1; b1 -= lr * db1
            W2 -= lr * dW2; b2 -= lr * db2
            W3 -= lr * dW3; b3 -= lr * db3

        if (epoch + 1) % 10 == 0:
            lr *= 0.5

        if (epoch + 1) % 5 == 0 or epoch == 0:
            z1t = X_test @ W1 + b1; a1t = relu(z1t)
            z2t = a1t @ W2 + b2; a2t = relu(z2t)
            z3t = a2t @ W3 + b3
            acc = (z3t.argmax(axis=1) == y_test).mean()
            loss = -np.log(softmax(z3t)[np.arange(len(y_test)), y_test] + 1e-8).mean()
            print(f"  Epoch {epoch+1:3d}: test_acc={acc:.4f}, loss={loss:.4f}")

    z1t = X_test @ W1 + b1; a1t = relu(z1t)
    z2t = a1t @ W2 + b2; a2t = relu(z2t)
    z3t = a2t @ W3 + b3
    final_acc = (z3t.argmax(axis=1) == y_test).mean()
    print(f"\n  Final test accuracy: {final_acc:.4f}")
    return W1, b1, W2, b2, W3, b3, final_acc


def export_weights(W1, b1, W2, b2, W3, b3, X_test, y_test):
    h = HIDDEN
    print(f"\nExporting weights (hidden={h})...")

    # W1: 784×HIDDEN — row-major
    save_f32(os.path.join(WEIGHTS_DIR, "w1.bin"), W1)
    save_f32(os.path.join(WEIGHTS_DIR, "b1.bin"), b1)

    # W2: HIDDEN×HIDDEN — row-major
    save_f32(os.path.join(WEIGHTS_DIR, "w2.bin"), W2)
    save_f32(os.path.join(WEIGHTS_DIR, "b2.bin"), b2)

    # W3: HIDDEN×10 → pad to HIDDEN×OUT_N_PAD (pad columns from 10→16)
    W3_pad = np.zeros((h, OUT_N_PAD), dtype=np.float32)
    W3_pad[:, :10] = W3
    b3_pad = np.zeros(OUT_N_PAD, dtype=np.float32)
    b3_pad[:10] = b3
    save_f32(os.path.join(WEIGHTS_DIR, "w3.bin"), W3_pad)
    save_f32(os.path.join(WEIGHTS_DIR, "b3.bin"), b3_pad)

    # Config file for Rust
    with open(os.path.join(WEIGHTS_DIR, "config.txt"), "w") as f:
        f.write(f"hidden={h}\n")
        f.write(f"out_n_pad={OUT_N_PAD}\n")
    print(f"  config.txt: hidden={h}, out_n_pad={OUT_N_PAD}")

    # Test batch: 16 diverse images
    print("\nExporting test batch (16 images)...")
    selected = []
    for digit in range(10):
        idxs = np.where(y_test == digit)[0]
        if len(idxs) > 0:
            selected.append(idxs[0])
    rng = np.random.RandomState(123)
    while len(selected) < 16:
        idx = rng.randint(0, len(y_test))
        if idx not in selected:
            selected.append(idx)
    selected = selected[:16]

    imgs = X_test[selected]       # (16, 784)
    labels = y_test[selected]     # (16,)

    save_f32(os.path.join(WEIGHTS_DIR, "test_images.bin"), imgs)
    save_u8(os.path.join(WEIGHTS_DIR, "test_labels.bin"), labels)

    # Pre-transposed: [784×16] column-major
    imgs_t = np.ascontiguousarray(imgs.T)
    save_f32(os.path.join(WEIGHTS_DIR, "test_images_t.bin"), imgs_t)

    # Reference forward pass
    z1 = imgs @ W1 + b1; a1 = relu(z1)    # (16, HIDDEN)
    z2 = a1 @ W2 + b2;   a2 = relu(z2)    # (16, HIDDEN)
    z3 = a2 @ W3 + b3                       # (16, 10)

    preds = z3.argmax(axis=1)
    print(f"\n  Labels:      {list(labels)}")
    print(f"  Predictions: {list(preds)}")
    print(f"  Correct:     {(preds == labels).sum()}/16")

    save_f32(os.path.join(WEIGHTS_DIR, "test_logits.bin"), z3)       # (16, 10)
    save_f32(os.path.join(WEIGHTS_DIR, "test_hidden1.bin"), a1)      # (16, HIDDEN)
    save_f32(os.path.join(WEIGHTS_DIR, "test_hidden2.bin"), a2)      # (16, HIDDEN)

    print(f"\nAll files written to {WEIGHTS_DIR}/")


def main():
    print("=" * 60)
    print(f"sme-jit-core Gate 22: Wide MNIST Weight Export (hidden={HIDDEN})")
    print("=" * 60)

    X_train, y_train, X_test, y_test = load_mnist()
    print(f"\nDataset: {X_train.shape[0]} train, {X_test.shape[0]} test")

    W1, b1, W2, b2, W3, b3, acc = train_mlp(X_train, y_train, X_test, y_test)
    export_weights(W1, b1, W2, b2, W3, b3, X_test, y_test)

    print("\n" + "=" * 60)
    print(f"Done. Model accuracy: {acc:.1%} (hidden={HIDDEN})")
    print("=" * 60)


if __name__ == "__main__":
    main()
