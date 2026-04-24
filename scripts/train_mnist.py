#!/usr/bin/env python3
"""
Train a tiny MNIST MLP and export weights as raw f32 binary files.

Architecture: 784 → 16 (ReLU) → 16 (ReLU) → 10 (logits)

The SME JIT kernel computes C[16×16] = A^T × B via FMOPA outer products.
For batch=16 inference, each layer is exactly one 16×16 tile operation:
  - Layer 1: input[16×784] × W1[784×16] → hidden1[16×16], K=49
  - Layer 2: hidden1[16×16] × W2[16×16] → hidden2[16×16], K=1
  - Layer 3: hidden2[16×16] × W3_pad[16×16] → logits[16×16], K=1
    (W3 is 16×10, zero-padded to 16×16; only first 10 columns used)

Weight layout for FMOPA:
  The kernel loads A as K panels of 16 floats (column of the input batch)
  and B as K panels of 16 floats (column of the weight matrix).
  FMOPA ZA0.S, P0/M, Z0.S, Z1.S computes ZA0 += Z0 ⊗ Z1 (outer product).
  
  After K iterations: ZA0[i][j] = Σ_k A[i][k] × B[k][j]
  
  This is standard matmul C = A × B where:
  - A is stored as K contiguous vectors of 16 floats (row-stride = 16)
  - B is stored as K contiguous vectors of 16 floats (row-stride = 16)
  
  So we store weights in "panel" layout: K groups of 16 floats each,
  where group k contains W[k][0..16] (one row of the weight matrix).

Output files (all little-endian f32):
  weights/w1.bin    — 784×16 = 12544 floats (Layer 1 weights)
  weights/b1.bin    — 16 floats (Layer 1 bias)
  weights/w2.bin    — 16×16 = 256 floats (Layer 2 weights)
  weights/b2.bin    — 16 floats (Layer 2 bias)
  weights/w3.bin    — 16×16 = 256 floats (Layer 3 weights, zero-padded)
  weights/b3.bin    — 16 floats (Layer 3 bias, zero-padded)
  weights/test_images.bin  — 16×784 = 12544 floats (16 test images)
  weights/test_labels.bin  — 16 bytes (16 test labels)
  weights/test_logits.bin  — 16×10 = 160 floats (reference logits from Python)
"""

import struct
import os
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
WEIGHTS_DIR = os.path.join(SCRIPT_DIR, "weights")
os.makedirs(WEIGHTS_DIR, exist_ok=True)


def save_f32(path: str, data: np.ndarray):
    """Save a numpy array as raw little-endian f32."""
    flat = data.astype(np.float32).flatten()
    with open(path, "wb") as f:
        f.write(flat.tobytes())
    print(f"  {os.path.basename(path)}: {flat.shape[0]} floats ({os.path.getsize(path)} bytes)")


def save_u8(path: str, data: np.ndarray):
    """Save a numpy array as raw u8."""
    flat = data.astype(np.uint8).flatten()
    with open(path, "wb") as f:
        f.write(flat.tobytes())
    print(f"  {os.path.basename(path)}: {flat.shape[0]} bytes")


def load_mnist():
    """Load MNIST using sklearn (no PyTorch/TF dependency)."""
    try:
        from sklearn.datasets import fetch_openml
        mnist = fetch_openml("mnist_784", version=1, as_frame=False, parser="auto")
        X = mnist.data.astype(np.float32) / 255.0
        y = mnist.target.astype(np.int64)
        return X[:60000], y[:60000], X[60000:], y[60000:]
    except ImportError:
        print("sklearn not available, generating synthetic MNIST-like data...")
        return generate_synthetic_mnist()


def generate_synthetic_mnist():
    """Generate synthetic data if sklearn isn't available."""
    rng = np.random.RandomState(42)
    # Create simple digit-like patterns
    X_train = rng.randn(60000, 784).astype(np.float32) * 0.3
    y_train = rng.randint(0, 10, 60000).astype(np.int64)
    # Add class-specific signal
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


def cross_entropy(probs, labels):
    n = probs.shape[0]
    return -np.log(probs[np.arange(n), labels] + 1e-8).mean()


def train_mlp(X_train, y_train, X_test, y_test):
    """Train a tiny MLP with SGD. Pure numpy, no frameworks."""
    rng = np.random.RandomState(42)
    
    # Xavier initialization
    W1 = rng.randn(784, 16).astype(np.float32) * np.sqrt(2.0 / 784)
    b1 = np.zeros(16, dtype=np.float32)
    W2 = rng.randn(16, 16).astype(np.float32) * np.sqrt(2.0 / 16)
    b2 = np.zeros(16, dtype=np.float32)
    W3 = rng.randn(16, 10).astype(np.float32) * np.sqrt(2.0 / 16)
    b3 = np.zeros(10, dtype=np.float32)
    
    lr = 0.1
    batch_size = 128
    epochs = 30
    n = X_train.shape[0]
    
    print(f"\nTraining: 784→16→16→10 MLP, {epochs} epochs, lr={lr}, batch={batch_size}")
    print(f"  Parameters: {784*16 + 16 + 16*16 + 16 + 16*10 + 10} = {784*16 + 16 + 16*16 + 16 + 16*10 + 10}")
    
    for epoch in range(epochs):
        # Shuffle
        perm = rng.permutation(n)
        X_shuf = X_train[perm]
        y_shuf = y_train[perm]
        
        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            X_batch = X_shuf[start:end]
            y_batch = y_shuf[start:end]
            bs = end - start
            
            # Forward
            z1 = X_batch @ W1 + b1          # (bs, 16)
            a1 = relu(z1)                     # (bs, 16)
            z2 = a1 @ W2 + b2                # (bs, 16)
            a2 = relu(z2)                     # (bs, 16)
            z3 = a2 @ W3 + b3                # (bs, 10)
            probs = softmax(z3)               # (bs, 10)
            
            # Backward (cross-entropy + softmax gradient)
            dz3 = probs.copy()
            dz3[np.arange(bs), y_batch] -= 1
            dz3 /= bs
            
            dW3 = a2.T @ dz3
            db3 = dz3.sum(axis=0)
            da2 = dz3 @ W3.T
            
            dz2 = da2 * (z2 > 0).astype(np.float32)
            dW2 = a1.T @ dz2
            db2 = dz2.sum(axis=0)
            da1 = dz2 @ W2.T
            
            dz1 = da1 * (z1 > 0).astype(np.float32)
            dW1 = X_batch.T @ dz1
            db1 = dz1.sum(axis=0)
            
            # Update
            W1 -= lr * dW1
            b1 -= lr * db1
            W2 -= lr * dW2
            b2 -= lr * db2
            W3 -= lr * dW3
            b3 -= lr * db3
        
        # Decay learning rate
        if (epoch + 1) % 10 == 0:
            lr *= 0.5
        
        # Evaluate
        if (epoch + 1) % 5 == 0 or epoch == 0:
            z1t = X_test @ W1 + b1
            a1t = relu(z1t)
            z2t = a1t @ W2 + b2
            a2t = relu(z2t)
            z3t = a2t @ W3 + b3
            preds = z3t.argmax(axis=1)
            acc = (preds == y_test).mean()
            loss = cross_entropy(softmax(z3t), y_test)
            print(f"  Epoch {epoch+1:3d}: test_acc={acc:.4f}, loss={loss:.4f}")
    
    # Final accuracy
    z1t = X_test @ W1 + b1
    a1t = relu(z1t)
    z2t = a1t @ W2 + b2
    a2t = relu(z2t)
    z3t = a2t @ W3 + b3
    preds = z3t.argmax(axis=1)
    final_acc = (preds == y_test).mean()
    print(f"\n  Final test accuracy: {final_acc:.4f}")
    
    return W1, b1, W2, b2, W3, b3, final_acc


def export_weights(W1, b1, W2, b2, W3, b3, X_test, y_test):
    """Export all weights and test data as raw binary files."""
    print("\nExporting weights...")
    
    # W1: 784×16 — already in the right layout for FMOPA
    # The kernel iterates K=49 times, loading 16 floats from A and 16 from B each time.
    # A = input batch (16×784), B = weights (784×16)
    # FMOPA needs B stored as K=49 panels of 16 floats = rows of W1.
    save_f32(os.path.join(WEIGHTS_DIR, "w1.bin"), W1)  # (784, 16) row-major
    save_f32(os.path.join(WEIGHTS_DIR, "b1.bin"), b1)   # (16,)
    
    # W2: 16×16
    save_f32(os.path.join(WEIGHTS_DIR, "w2.bin"), W2)  # (16, 16) row-major
    save_f32(os.path.join(WEIGHTS_DIR, "b2.bin"), b2)   # (16,)
    
    # W3: 16×10 → zero-pad to 16×16
    W3_pad = np.zeros((16, 16), dtype=np.float32)
    W3_pad[:, :10] = W3
    b3_pad = np.zeros(16, dtype=np.float32)
    b3_pad[:10] = b3
    save_f32(os.path.join(WEIGHTS_DIR, "w3.bin"), W3_pad)  # (16, 16) zero-padded
    save_f32(os.path.join(WEIGHTS_DIR, "b3.bin"), b3_pad)   # (16,) zero-padded
    
    # Test batch: pick 16 diverse images (one per digit where possible)
    print("\nExporting test batch (16 images)...")
    selected_indices = []
    for digit in range(10):
        idxs = np.where(y_test == digit)[0]
        if len(idxs) > 0:
            selected_indices.append(idxs[0])
    # Fill remaining with random
    rng = np.random.RandomState(123)
    while len(selected_indices) < 16:
        idx = rng.randint(0, len(y_test))
        if idx not in selected_indices:
            selected_indices.append(idx)
    selected_indices = selected_indices[:16]
    
    test_images = X_test[selected_indices]  # (16, 784)
    test_labels = y_test[selected_indices]   # (16,)
    
    save_f32(os.path.join(WEIGHTS_DIR, "test_images.bin"), test_images)
    save_u8(os.path.join(WEIGHTS_DIR, "test_labels.bin"), test_labels)
    
    # Compute reference logits for these 16 images
    z1 = test_images @ W1 + b1
    a1 = relu(z1)
    z2 = a1 @ W2 + b2
    a2 = relu(z2)
    z3 = a2 @ W3 + b3  # (16, 10) — unpadded logits
    
    preds = z3.argmax(axis=1)
    print(f"\n  Test labels:      {list(test_labels)}")
    print(f"  Predictions:      {list(preds)}")
    correct = (preds == test_labels).sum()
    print(f"  Correct: {correct}/16")
    
    save_f32(os.path.join(WEIGHTS_DIR, "test_logits.bin"), z3)  # (16, 10)
    
    # Also save intermediate activations for debugging
    save_f32(os.path.join(WEIGHTS_DIR, "test_hidden1.bin"), a1)  # (16, 16) after ReLU
    save_f32(os.path.join(WEIGHTS_DIR, "test_hidden2.bin"), a2)  # (16, 16) after ReLU
    
    print(f"\nAll files written to {WEIGHTS_DIR}/")


def main():
    print("=" * 60)
    print("sme-jit-core Gate 18: MNIST Weight Export")
    print("=" * 60)
    
    X_train, y_train, X_test, y_test = load_mnist()
    print(f"\nDataset: {X_train.shape[0]} train, {X_test.shape[0]} test")
    
    W1, b1, W2, b2, W3, b3, acc = train_mlp(X_train, y_train, X_test, y_test)
    export_weights(W1, b1, W2, b2, W3, b3, X_test, y_test)
    
    print("\n" + "=" * 60)
    print(f"Done. Model accuracy: {acc:.1%}")
    print("=" * 60)


if __name__ == "__main__":
    main()
