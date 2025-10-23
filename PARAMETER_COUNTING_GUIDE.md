# Parameter Counting Guide for Transformer Models

This guide explains how to calculate parameters in GPT-style transformer models.

## Table of Contents
1. [Quick Answer](#quick-answer)
2. [Detailed Breakdown](#detailed-breakdown)
3. [General Formulas](#general-formulas)
4. [How to Verify in Code](#how-to-verify-in-code)

---

## Quick Answer

For the **Baby GPT** model (`config/train_shakespeare_char.py`):

| Component | Parameters |
|-----------|------------|
| **Per Transformer Block** | **1,774,464** |
| - Attention | 591,360 (33.3%) |
| - MLP | 1,181,568 (66.6%) |
| - LayerNorm | 1,536 (0.1%) |

**Key Insight:** MLP has ~2× more parameters than Attention!

---

## Detailed Breakdown

### Configuration
```python
n_embd = 384      # Hidden dimension
n_head = 6        # Number of attention heads  
n_layer = 6       # Number of transformer blocks
block_size = 256  # Context length
vocab_size = 65   # Vocabulary size
bias = True       # Include bias terms
```

### 1. CausalSelfAttention

#### a) QKV Projection (`c_attn`)
Combines Query, Key, Value projections into one matrix for efficiency.

```
Input shape:  [batch, seq_len, 384]
Output shape: [batch, seq_len, 1152]  (384 × 3)

Weight: 384 × 1152 = 442,368 parameters
Bias:   1152 = 1,152 parameters
Total:  443,520 parameters
```

**Why 1152?** Because we project to Q, K, V simultaneously: 384 × 3 = 1152

#### b) Output Projection (`c_proj`)
Projects concatenated attention outputs back to hidden dimension.

```
Input shape:  [batch, seq_len, 384]
Output shape: [batch, seq_len, 384]

Weight: 384 × 384 = 147,456 parameters
Bias:   384 = 384 parameters
Total:  147,840 parameters
```

#### Attention Subtotal
```
QKV projection:     443,520
Output projection:  147,840
──────────────────────────
TOTAL:              591,360 parameters
```

---

### 2. MLP (Feed-Forward Network)

The MLP uses a standard architecture: expand → activate → project

#### a) Expansion Layer (`c_fc`)
Expands to 4× hidden dimension (standard in transformers).

```
Input shape:  [batch, seq_len, 384]
Output shape: [batch, seq_len, 1536]  (384 × 4)

Weight: 384 × 1536 = 589,824 parameters
Bias:   1536 = 1,536 parameters
Total:  591,360 parameters
```

#### b) Projection Layer (`c_proj`)
Projects back down to original hidden dimension.

```
Input shape:  [batch, seq_len, 1536]
Output shape: [batch, seq_len, 384]

Weight: 1536 × 384 = 589,824 parameters
Bias:   384 = 384 parameters
Total:  590,208 parameters
```

#### MLP Subtotal
```
Expansion layer:    591,360
Projection layer:   590,208
──────────────────────────
TOTAL:              1,181,568 parameters
```

**Note:** MLP has nearly 2× the parameters of attention!

---

### 3. LayerNorm

Two LayerNorm layers per block: before attention (`ln_1`) and before MLP (`ln_2`).

Each LayerNorm has:
- **Weight:** n_embd parameters (scale parameter)
- **Bias:** n_embd parameters (shift parameter)

```
ln_1: 384 (weight) + 384 (bias) = 768 parameters
ln_2: 384 (weight) + 384 (bias) = 768 parameters
──────────────────────────────────────────────
TOTAL:                            1,536 parameters
```

**Note:** LayerNorm is negligible (~0.1% of block parameters)

---

### Total Per Block

```
Attention:    591,360  (33.3%)
MLP:        1,181,568  (66.6%)
LayerNorm:      1,536  ( 0.1%)
───────────────────────────────
TOTAL:      1,774,464 parameters
```

For 6 layers: **1,774,464 × 6 = 10,646,784 parameters** in transformer blocks alone!

---

## General Formulas

### Notation
- `d` = n_embd (hidden dimension)
- `V` = vocab_size
- `L` = n_layer (number of blocks)
- `T` = block_size (context length)
- `b` = 1 if bias=True, 0 if bias=False
- `m` = MLP expansion ratio (typically 4)

### Per Transformer Block

```
Attention:
  QKV projection:    d × 3d + b×3d = 3d² + 3bd
  Output projection: d × d + b×d   = d² + bd
  ──────────────────────────────────────────
  Attention total:                   4d² + 4bd

MLP:
  Expansion:         d × md + b×md  = md² + mbd
  Projection:        md × d + b×d   = md² + bd
  ──────────────────────────────────────────
  MLP total:                         2md² + (m+1)bd
  (with m=4):                        8d² + 5bd

LayerNorm (×2):
  Total:                             2d + 2bd

──────────────────────────────────────────────
Block Total (m=4):                   12d² + 11bd
```

### Full Model

```
Token embeddings:      V × d
Position embeddings:   T × d
Transformer blocks:    L × (12d² + 11bd)
Final LayerNorm:       d + bd
LM head:               0  (if weight tying)
                       V × d  (if not weight tying)

──────────────────────────────────────────────────────────
Total (with weight tying): V×d + T×d + L×(12d²+11bd) + d + bd
```

### Baby GPT Example

Substituting values (d=384, V=65, L=6, T=256, b=1):

```
Block:  12×384² + 11×384 = 1,769,472 + 4,224 = 1,773,696

Full:   65×384 + 256×384 + 6×1,773,696 + 384 + 384
     =  24,960 + 98,304 + 10,642,176 + 768
     =  10,766,208 parameters (~10.77M)
```

---

## How to Verify in Code

### Method 1: Manual Calculation (No Dependencies)

Use the script I created: `count_parameters.py`

```bash
python3 count_parameters.py
```

This calculates parameters using pure Python (no PyTorch needed).

### Method 2: Count PyTorch Model Parameters

If you have PyTorch installed, use `verify_params.py`:

```bash
python3 verify_params.py
```

Or count manually in Python:

```python
from model import GPT, GPTConfig

# Create model
config = GPTConfig(n_layer=6, n_embd=384, n_head=6, 
                   block_size=256, vocab_size=65, bias=True)
model = GPT(config)

# Count all parameters
total_params = sum(p.numel() for p in model.parameters())
print(f"Total parameters: {total_params:,}")

# Count parameters in first block
block = model.transformer.h[0]
block_params = sum(p.numel() for p in block.parameters())
print(f"Parameters per block: {block_params:,}")

# Detailed breakdown
for name, param in block.named_parameters():
    print(f"{name:40s} {param.shape} → {param.numel():,} params")
```

### Method 3: Quick One-Liner

```python
model.get_num_params()  # Returns total (excludes position embeddings by default)
```

---

## Key Insights

### 1. MLP Dominates Parameter Count
- **MLP: 66.6%** of block parameters
- **Attention: 33.3%** of block parameters
- **LayerNorm: 0.1%** (negligible)

This is because:
- MLP has 4× expansion: `8d²` parameters
- Attention is `4d²` parameters
- Ratio: 8d² / 4d² = 2:1

### 2. Most Parameters Are in Transformer Blocks

For Baby GPT:
- **Transformer blocks:** 10.6M (98.8%)
- **Embeddings:** 123K (1.1%)
- **Final LayerNorm:** 768 (0.01%)

### 3. Scaling Laws

As you increase `n_embd`, parameters grow **quadratically** (∝ d²):

| n_embd | Block Params | % Increase |
|--------|--------------|------------|
| 192    | 443,616      | baseline   |
| 384    | 1,774,464    | 4×         |
| 768    | 7,097,856    | 16×        |

Doubling `n_embd` → 4× more parameters!

### 4. GPT-2 Comparison

| Model | n_layer | n_embd | Params |
|-------|---------|--------|--------|
| **Baby GPT** | 6 | 384 | 10.77M |
| GPT-2 Small | 12 | 768 | 124M |
| GPT-2 Medium | 24 | 1024 | 350M |
| GPT-2 Large | 36 | 1280 | 774M |
| GPT-2 XL | 48 | 1600 | 1558M |

---

## Common Questions

### Q: Why is the QKV projection combined?
**A:** Efficiency! One matrix multiply for Q, K, V together is faster than three separate ones.

### Q: Why does MLP expand to 4×?
**A:** Empirically found to work well. Gives the model capacity to learn complex patterns. Some modern models use different ratios (e.g., 8/3 for some architectures).

### Q: What's the point of weight tying?
**A:** Reduces parameters and improves performance by ensuring input and output token representations are consistent.

### Q: Where do biases matter most?
**A:** Modern models (like this one with GPT-2 settings) often set `bias=False` for slight speedup and fewer parameters. The bias terms are actually quite small compared to weights.

### Q: How many FLOPs per forward pass?
**A:** Approximately `6N` FLOPs per token for a model with `N` parameters (see `model.py:289-303` for MFU calculation).

---

## Exercises

1. **Calculate parameters for GPT-2:** 
   - n_layer=12, n_embd=768, vocab_size=50257, block_size=1024
   - Answer: ~124M parameters

2. **What if we remove bias terms?**
   - Baby GPT with bias=False
   - Answer: Block becomes `12d² = 1,769,472` (saves ~5K params per block)

3. **What if we use MLP ratio of 8?**
   - MLP becomes: `16d² + 9bd`
   - Block becomes: `20d² + 13bd`

4. **Count parameters in attention heads:**
   - Parameters don't change with number of heads!
   - Heads only affect how we split the hidden dimension
   - More heads = smaller head dimension, same total params

---

## References

- Original Transformer: "Attention is All You Need" (Vaswani et al., 2017)
- GPT-2: "Language Models are Unsupervised Multitask Learners" (Radford et al., 2019)
- nanoGPT: https://github.com/karpathy/nanoGPT
- See `model.py` lines 94-106 for Block implementation

---

**Created for learning purposes** • Questions? Check the code comments in `model.py`!
