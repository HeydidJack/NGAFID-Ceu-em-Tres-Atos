# Model Configuration Reference

This document details the hyperparameter specifications and configuration requirements for all models in the NGAFID-Ceu-em-Tres-Atos framework. All configurations are passed via a `configs` object (namespace/dotdict) containing the following parameters.

## Global Configuration Parameters

| Parameter | Type | Description | Applicable Models |
|-----------|------|-------------|-------------------|
| `in_dim` | int | Input feature dimension (number of sensor channels). Default: 23 for NGAFID | All |
| `clasnum` | int | Number of output classes. 2 for Anomaly Detection (AD), 19/36 for Fault Classification (FC), 20/37 for Diagnosis | All |
| `learning_rate` | float | Initial learning rate. Typical values: 1e-4 for ConvTok series, 3e-5 for InceptionTimeAttn, 1e-3 for Bi-LSTM | All |
| `batch_size` | int | Training batch size. 32 for deep models, 64/256 for shallow models (MLP/CNN) | All |
| `dropout` | float | Dropout rate for regularization. Default: 0.01 | ConvTok, InceptionTime, MMK Net |
| `patience` | int | Early stopping patience (epochs). Default: 3 | All |
| `train_epochs` | int | Maximum training epochs. Default: 200 | All |
| `use_gpu` | bool | Enable GPU acceleration | All |
| `gpu` | int | GPU device ID | All |

---

## Architecture-Specific Parameters

### 1. MLP (Multi-Layer Perceptron)
Baseline feedforward network for static pattern recognition.

| Parameter | Type | Description |
|-----------|------|-------------|
| `d_model` | int | Hidden layer dimension. Default: 128 |

**Note:** `in_dim` for MLP should be set to `full_len * in_dim` (e.g., 2048 * 23) as the model flattens the temporal dimension.

---

### 2. CNN (Shallow Convolutional Network)
Lightweight 1D convolutional baseline with limited receptive field.

| Parameter | Type | Description |
|-----------|------|-------------|
| `in_len` | int | Input sequence length. Default: 2048 |
| `d_model` | int | Hidden dimension (convolution output channels). Default: 128 |

---

### 3. Bi-LSTM (Bidirectional LSTM)
Recurrent architecture for long-range temporal dependency modeling.

| Parameter | Type | Description |
|-----------|------|-------------|
| `num_layers` | int | Number of stacked Bi-LSTM layers. Default: 3 |
| `d_model` | int | Hidden dimension per direction. Default: 256 |
| `bidirectional` | bool | Enable bidirectional processing. Default: True |

**Architecture Notes:**
- Includes 1×1 conv projection (input_dim → 64) to reduce LSTM parameters
- Uses LayerNorm and Dropout(0.3) in the projection head

---

### 4. InceptionTime
Multi-scale convolutional architecture with parallel kernel branches.

| Parameter | Type | Description |
|-----------|------|-------------|
| `inception_layers` | int | Depth of Inception modules (equivalent to `num_layers`). Default: 4 |
| `filters` | int | Output channels per kernel branch. Default: 128 (AD), 256 (FC) |
| `hidden_dim` | int | Final projection dimension. Default: 2048 |

**Architecture Notes:**
- Residual connections every 3 layers
- Parallel kernels: 10, 20, 40 with bottleneck 1×1 convolutions

---

### 5. MMK Net (Multi-Micro Kernel Network)
Restricted-receptive-field CNN for micro-scale local feature extraction (Fault Analyzer in LMSD).

| Parameter | Type | Description |
|-----------|------|-------------|
| `inception_layers` | int | Number of MMK blocks (equivalent to `num_layers`). Default: 4 |
| `filters` | int | Convolution kernel count per branch. Default: 256 |
| `hidden_dim` | int | Output dimension. Default: 2048 |

**Architecture Notes:**
- Multi-scale micro-kernels: 1, 3, 5 (no pooling to preserve temporal resolution)
- Uses LayerNorm (not BatchNorm) for stability under class imbalance
- Concatenates 3 branches → 3 × `filters` output channels

---

### 6. ConvTokMHSA (Convolutional Tokenizer + Multi-Head Self-Attention)
Global attention architecture for full-sequence context aggregation (Health Analyzer).

| Parameter | Type | Description |
|-----------|------|-------------|
| `L_patch` | int | Patch length for tokenization (sequence segmentation). Default: 4 |
| `token_dim` | int | Token embedding dimension. Default: 128 (AD), 512 (FC) |
| `e_layers` | int | Transformer encoder depth (equivalent to `encoder_layers`). Default: 2 (AD), 4 (FC) |
| `n_heads` | int | Number of attention heads. Default: 4 |
| `d_ff` | int | Feed-forward network dimension. Default: 512 (AD), 1024 (FC) |
| `activation` | str | Activation function: 'gelu' or 'relu'. Default: 'gelu' |
| `output_attention` | bool | Return attention weights for interpretability. Default: False |
| `distil` | bool | Use ConvLayer downsampling between encoder layers. Default: False |

**Architecture Notes:**
- ConvTokenizer: 1D-CNN + statistics (μ, σ) + sinusoidal position encoding
- FullAttention: Global receptive field (no window constraints)
- Convolutional projections for Q, K, V instead of linear

---

### 7. ConvTokSWLA (Sliding Window Local Attention)
Local attention variant with fixed-width sliding window constraint.

| Parameter | Type | Description |
|-----------|------|-------------|
| `viewindow` | int | Half-width of sliding window (w). Default: 4 (AD), 3 (FC) |
| *Other params* | *Same as ConvTokMHSA* | |

**Architecture Notes:**
- Masked attention: Only attends to tokens within `|i-j| ≤ viewindow`
- Fixed window size across all heads
- Suitable for FC tasks requiring local SNR maximization

---

### 8. ConvTokMWLA (Multi-Window Local Attention)
Hierarchical local attention with head-specific window sizes.

| Parameter | Type | Description |
|-----------|------|-------------|
| `viewindow_size` | list[int] | List of half-widths per head. Example: [1, 3, 5, 0] |
| *Other params* | *Same as ConvTokMHSA* | |

**Architecture Notes:**
- Each head `i` uses independent window size `viewindow_size[i]`
- Window size 0 indicates global attention for that head
- Enables multi-scale local feature extraction without increasing depth

---

### 9. ConvTokLPLA (Log-Parse Local Attention)
Hybrid attention combining local windows with log-parse sparse global heads.

| Parameter | Type | Description |
|-----------|------|-------------|
| `viewindow_size` | list[int] | Window sizes per head. Example: [1, 3, 5, 0] (0 for global) |
| *Other params* | *Same as ConvTokMHSA* | |

**Architecture Notes:**
- Implements LogParseProjection: log(1+|x|) mapping + learnable sparse gate
- Sparsemax normalization on global heads to prevent noise re-introduction
- Balances local discrimination with limited global context

---

### 10. LMSD (Long-Mini Scale Diagnostician)
Heterogeneous two-stage cascade architecture (DDF instantiation).

| Parameter | Type | Description |
|-----------|------|-------------|
| `deteclasnum` | int | AD stage output classes. Default: 2 |
| `frclasnum` | int | FC stage output classes. Default: 19 or 36 |
| `NormalIndex` | int | Index of healthy class in final output. Default: 0 |
| `detecprepath` | str | Path to pretrained ConvTokMHSA checkpoint |
| `frprepath` | str | Path to pretrained MMK_Net checkpoint |
| `filters` | int | Must match MMK_Net filters (for feature concatenation) |

**Architecture Notes:**
- **Stage 1 (Long):** ConvTokMHSA with global attention for AD
- **Stage 2 (Micro):** MMK_Net with restricted receptive fields for FC
- **Routing:** Hard-threshold dimensional isolation (negative infinity masking)
- **Training:** Decoupled - stages trained independently, parameters frozen during cascade
- **Inference:** Healthy samples bypass Fault Analyzer; anomalous samples activate MMK_Net

---

## Hyperparameter Quick Reference by Task

### Anomaly Detection (AD)
| Model | num_layers | encoder_layers | hidden_dim | filters | token_dim | viewindow | lr | bs |
|-------|------------|----------------|------------|---------|-----------|-----------|-------|-----|
| Bi-LSTM | 3 | - | 256 | - | - | - | 1e-4 | 32 |
| InceptionTime | 4 | - | 2048 | 256 | - | - | 1e-4 | 32 |
| MMK Net | 4 | - | 2048 | 256 | - | - | 1e-4 | 32 |
| ConvTokMHSA | - | 2 | - | - | 128 | - | 1e-4 | 32 |
| ConvTokSWLA | - | 2 | - | - | 128 | 4 | 1e-4 | 32 |

### Fault Classification (FC)
| Model | num_layers | encoder_layers | hidden_dim | filters | token_dim | viewindow | lr | bs |
|-------|------------|----------------|------------|---------|-----------|-----------|-------|-----|
| InceptionTime | 4 | - | 2048 | 256 | - | - | 1e-4 | 32 |
| MMK Net | 4 | - | 2048 | 256 | - | - | 1e-4 | 32 |
| ConvTokMHSA | - | 4 | - | - | 512 | - | 1e-4 | 32 |
| ConvTokSWLA | - | 4 | - | - | 512 | 4 | 1e-4 | 32 |
| ConvTokMWLA | - | 4 | - | - | 512 | [0,1,3,5] | 1e-4 | 32 |
| ConvTokLPLA | - | 4 | - | - | 512 | [0,1,3,5] | 1e-4 | 32 |

**Key:** `num_layers` applies to Bi-LSTM/InceptionTime/MMK Net; `encoder_layers` applies to ConvTok series.

## Configuration File Example

```python
# configs/example_ad_config.py
class Config:
    # Data
    in_dim = 23
    full_len = 2048
    clasnum = 2
    
    # Model (ConvTokMHSA example)
    L_patch = 4
    token_dim = 128
    e_layers = 2
    n_heads = 4
    d_ff = 512
    dropout = 0.01
    activation = 'gelu'
    
    # Training
    learning_rate = 1e-4
    batch_size = 32
    train_epochs = 200
    patience = 3
    
    # Hardware
    use_gpu = True
    gpu = 0