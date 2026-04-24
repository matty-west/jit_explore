//! Weight loading for the tiny MNIST inference engine.
//!
//! Loads pre-trained weights from raw f32 binary files exported by
//! `scripts/train_mnist.py`. All files are little-endian f32 arrays.

use std::fs;
use std::path::Path;

/// Load a raw f32 binary file into a Vec<f32>.
pub fn load_f32_bin(path: &Path) -> Result<Vec<f32>, String> {
    let bytes = fs::read(path).map_err(|e| format!("Failed to read {}: {}", path.display(), e))?;
    if bytes.len() % 4 != 0 {
        return Err(format!(
            "{}: file size {} is not a multiple of 4",
            path.display(),
            bytes.len()
        ));
    }
    let floats: Vec<f32> = bytes
        .chunks_exact(4)
        .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
        .collect();
    Ok(floats)
}

/// Load a raw u8 binary file into a Vec<u8>.
pub fn load_u8_bin(path: &Path) -> Result<Vec<u8>, String> {
    fs::read(path).map_err(|e| format!("Failed to read {}: {}", path.display(), e))
}

/// All weights and biases for the 3-layer MLP.
///
/// Architecture: 784 → 16 (ReLU) → 16 (ReLU) → 10 (logits)
///
/// Layer 3 weights are zero-padded from 16×10 to 16×16 so the
/// FMOPA kernel can operate on a full tile.
pub struct MnistWeights {
    /// Layer 1 weights: 784×16 = 12544 floats (row-major)
    pub w1: Vec<f32>,
    /// Layer 1 bias: 16 floats
    pub b1: Vec<f32>,
    /// Layer 2 weights: 16×16 = 256 floats (row-major)
    pub w2: Vec<f32>,
    /// Layer 2 bias: 16 floats
    pub b2: Vec<f32>,
    /// Layer 3 weights: 16×16 = 256 floats (zero-padded from 16×10)
    pub w3: Vec<f32>,
    /// Layer 3 bias: 16 floats (zero-padded from 10)
    pub b3: Vec<f32>,
}

impl MnistWeights {
    /// Load weights from the `scripts/weights/` directory.
    pub fn load(weights_dir: &Path) -> Result<Self, String> {
        let w1 = load_f32_bin(&weights_dir.join("w1.bin"))?;
        let b1 = load_f32_bin(&weights_dir.join("b1.bin"))?;
        let w2 = load_f32_bin(&weights_dir.join("w2.bin"))?;
        let b2 = load_f32_bin(&weights_dir.join("b2.bin"))?;
        let w3 = load_f32_bin(&weights_dir.join("w3.bin"))?;
        let b3 = load_f32_bin(&weights_dir.join("b3.bin"))?;

        // Validate sizes
        if w1.len() != 784 * 16 {
            return Err(format!("w1: expected {} floats, got {}", 784 * 16, w1.len()));
        }
        if b1.len() != 16 {
            return Err(format!("b1: expected 16 floats, got {}", b1.len()));
        }
        if w2.len() != 16 * 16 {
            return Err(format!("w2: expected {} floats, got {}", 16 * 16, w2.len()));
        }
        if b2.len() != 16 {
            return Err(format!("b2: expected 16 floats, got {}", b2.len()));
        }
        if w3.len() != 16 * 16 {
            return Err(format!("w3: expected {} floats, got {}", 16 * 16, w3.len()));
        }
        if b3.len() != 16 {
            return Err(format!("b3: expected 16 floats, got {}", b3.len()));
        }

        Ok(Self { w1, b1, w2, b2, w3, b3 })
    }
}

/// Test data: 16 images with their labels and reference logits.
pub struct MnistTestBatch {
    /// 16 images, each 784 floats (row-major: image[i] = data[i*784..(i+1)*784])
    pub images: Vec<f32>,
    /// Pre-transposed images: 784×16 column-major (Gate 20 — zero-cost input)
    pub images_t: Vec<f32>,
    /// 16 labels (ground-truth digits 0–9)
    pub labels: Vec<u8>,
    /// Reference logits from Python: 16×10 = 160 floats
    pub ref_logits: Vec<f32>,
    /// Reference hidden1 activations: 16×16 = 256 floats
    pub ref_hidden1: Vec<f32>,
    /// Reference hidden2 activations: 16×16 = 256 floats
    pub ref_hidden2: Vec<f32>,
}

/// Weights for a wider MLP (Gate 22): 784 → H → H → 10.
///
/// H (hidden_dim) is read from config.txt. Weights are loaded from
/// `scripts/weights_wide/`. Layer 3 output is padded from H×10 to H×16.
pub struct MnistWeightsWide {
    pub hidden_dim: usize,
    /// Layer 1 weights: 784×H floats (row-major)
    pub w1: Vec<f32>,
    /// Layer 1 bias: H floats
    pub b1: Vec<f32>,
    /// Layer 2 weights: H×H floats (row-major)
    pub w2: Vec<f32>,
    /// Layer 2 bias: H floats
    pub b2: Vec<f32>,
    /// Layer 3 weights: H×16 floats (zero-padded from H×10, row-major)
    pub w3: Vec<f32>,
    /// Layer 3 bias: 16 floats (zero-padded from 10)
    pub b3: Vec<f32>,
}

impl MnistWeightsWide {
    /// Load wide weights from the given directory.
    ///
    /// Reads `config.txt` for hidden dim, then validates all weight shapes.
    pub fn load(weights_dir: &Path) -> Result<Self, String> {
        // Parse config.txt
        let config_path = weights_dir.join("config.txt");
        let config_str = std::fs::read_to_string(&config_path)
            .map_err(|e| format!("Failed to read {}: {}", config_path.display(), e))?;
        let hidden_dim = config_str.lines()
            .find(|l| l.starts_with("hidden="))
            .and_then(|l| l.strip_prefix("hidden="))
            .and_then(|v| v.trim().parse::<usize>().ok())
            .ok_or_else(|| "config.txt missing hidden= line".to_string())?;

        let h = hidden_dim;
        let w1 = load_f32_bin(&weights_dir.join("w1.bin"))?;
        let b1 = load_f32_bin(&weights_dir.join("b1.bin"))?;
        let w2 = load_f32_bin(&weights_dir.join("w2.bin"))?;
        let b2 = load_f32_bin(&weights_dir.join("b2.bin"))?;
        let w3 = load_f32_bin(&weights_dir.join("w3.bin"))?;
        let b3 = load_f32_bin(&weights_dir.join("b3.bin"))?;

        if w1.len() != 784 * h { return Err(format!("w1: expected {} floats, got {}", 784 * h, w1.len())); }
        if b1.len() != h { return Err(format!("b1: expected {} floats, got {}", h, b1.len())); }
        if w2.len() != h * h { return Err(format!("w2: expected {} floats, got {}", h * h, w2.len())); }
        if b2.len() != h { return Err(format!("b2: expected {} floats, got {}", h, b2.len())); }
        if w3.len() != h * 16 { return Err(format!("w3: expected {} floats, got {}", h * 16, w3.len())); }
        if b3.len() != 16 { return Err(format!("b3: expected 16 floats, got {}", b3.len())); }

        Ok(Self { hidden_dim, w1, b1, w2, b2, w3, b3 })
    }
}

/// Test data for wide model: 16 images with labels and reference activations.
pub struct MnistTestBatchWide {
    pub hidden_dim: usize,
    pub images: Vec<f32>,        // 16×784
    pub images_t: Vec<f32>,      // 784×16 (pre-transposed)
    pub labels: Vec<u8>,         // 16
    pub ref_logits: Vec<f32>,    // 16×10
    pub ref_hidden1: Vec<f32>,   // 16×H
    pub ref_hidden2: Vec<f32>,   // 16×H
}

impl MnistTestBatchWide {
    pub fn load(weights_dir: &Path, hidden_dim: usize) -> Result<Self, String> {
        let h = hidden_dim;
        let images = load_f32_bin(&weights_dir.join("test_images.bin"))?;
        let labels = load_u8_bin(&weights_dir.join("test_labels.bin"))?;
        let ref_logits = load_f32_bin(&weights_dir.join("test_logits.bin"))?;
        let ref_hidden1 = load_f32_bin(&weights_dir.join("test_hidden1.bin"))?;
        let ref_hidden2 = load_f32_bin(&weights_dir.join("test_hidden2.bin"))?;

        let images_t = match load_f32_bin(&weights_dir.join("test_images_t.bin")) {
            Ok(t) if t.len() == 784 * 16 => t,
            _ => {
                let mut t = vec![0.0f32; 784 * 16];
                for i in 0..16 { for k in 0..784 { t[k * 16 + i] = images[i * 784 + k]; } }
                t
            }
        };

        if images.len() != 16 * 784 { return Err(format!("test_images: expected {} got {}", 16 * 784, images.len())); }
        if labels.len() != 16 { return Err(format!("test_labels: expected 16 got {}", labels.len())); }
        if ref_logits.len() != 16 * 10 { return Err(format!("test_logits: expected {} got {}", 16 * 10, ref_logits.len())); }
        if ref_hidden1.len() != 16 * h { return Err(format!("test_hidden1: expected {} got {}", 16 * h, ref_hidden1.len())); }
        if ref_hidden2.len() != 16 * h { return Err(format!("test_hidden2: expected {} got {}", 16 * h, ref_hidden2.len())); }

        Ok(Self { hidden_dim: h, images, images_t, labels, ref_logits, ref_hidden1, ref_hidden2 })
    }
}

impl MnistTestBatch {
    /// Load test data from the `scripts/weights/` directory.
    pub fn load(weights_dir: &Path) -> Result<Self, String> {
        let images = load_f32_bin(&weights_dir.join("test_images.bin"))?;
        let labels = load_u8_bin(&weights_dir.join("test_labels.bin"))?;
        let ref_logits = load_f32_bin(&weights_dir.join("test_logits.bin"))?;
        let ref_hidden1 = load_f32_bin(&weights_dir.join("test_hidden1.bin"))?;
        let ref_hidden2 = load_f32_bin(&weights_dir.join("test_hidden2.bin"))?;

        // Load pre-transposed input (Gate 20). Fall back to runtime transpose if missing.
        let images_t = match load_f32_bin(&weights_dir.join("test_images_t.bin")) {
            Ok(t) if t.len() == 784 * 16 => t,
            _ => {
                // Runtime fallback: transpose images ourselves
                let mut t = vec![0.0f32; 784 * 16];
                for i in 0..16 {
                    for k in 0..784 {
                        t[k * 16 + i] = images[i * 784 + k];
                    }
                }
                t
            }
        };

        if images.len() != 16 * 784 {
            return Err(format!("test_images: expected {} floats, got {}", 16 * 784, images.len()));
        }
        if labels.len() != 16 {
            return Err(format!("test_labels: expected 16 bytes, got {}", labels.len()));
        }
        if ref_logits.len() != 16 * 10 {
            return Err(format!("test_logits: expected {} floats, got {}", 16 * 10, ref_logits.len()));
        }
        if ref_hidden1.len() != 16 * 16 {
            return Err(format!("test_hidden1: expected {} floats, got {}", 16 * 16, ref_hidden1.len()));
        }
        if ref_hidden2.len() != 16 * 16 {
            return Err(format!("test_hidden2: expected {} floats, got {}", 16 * 16, ref_hidden2.len()));
        }

        Ok(Self { images, images_t, labels, ref_logits, ref_hidden1, ref_hidden2 })
    }
}
