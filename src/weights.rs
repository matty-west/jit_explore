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
    /// 16 labels (ground-truth digits 0–9)
    pub labels: Vec<u8>,
    /// Reference logits from Python: 16×10 = 160 floats
    pub ref_logits: Vec<f32>,
    /// Reference hidden1 activations: 16×16 = 256 floats
    pub ref_hidden1: Vec<f32>,
    /// Reference hidden2 activations: 16×16 = 256 floats
    pub ref_hidden2: Vec<f32>,
}

impl MnistTestBatch {
    /// Load test data from the `scripts/weights/` directory.
    pub fn load(weights_dir: &Path) -> Result<Self, String> {
        let images = load_f32_bin(&weights_dir.join("test_images.bin"))?;
        let labels = load_u8_bin(&weights_dir.join("test_labels.bin"))?;
        let ref_logits = load_f32_bin(&weights_dir.join("test_logits.bin"))?;
        let ref_hidden1 = load_f32_bin(&weights_dir.join("test_hidden1.bin"))?;
        let ref_hidden2 = load_f32_bin(&weights_dir.join("test_hidden2.bin"))?;

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

        Ok(Self { images, labels, ref_logits, ref_hidden1, ref_hidden2 })
    }
}
