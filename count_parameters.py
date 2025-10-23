"""
Script to count parameters in a Transformer block
This helps you understand the parameter breakdown
"""

def count_transformer_block_params(n_embd, n_head=None, bias=True, mlp_ratio=4):
    """
    Count parameters in a single transformer block
    
    Args:
        n_embd: Hidden dimension (embedding size)
        n_head: Number of attention heads (not needed for param counting)
        bias: Whether to include bias terms
        mlp_ratio: MLP expansion ratio (typically 4)
    
    Returns:
        Dictionary with parameter counts
    """
    
    # ========================================================================
    # 1. ATTENTION PARAMETERS
    # ========================================================================
    # c_attn: Single linear layer that produces Q, K, V together
    # Shape: [n_embd] -> [3 * n_embd]
    attn_qkv_weight = n_embd * (3 * n_embd)
    attn_qkv_bias = 3 * n_embd if bias else 0
    
    # c_proj: Output projection
    # Shape: [n_embd] -> [n_embd]
    attn_proj_weight = n_embd * n_embd
    attn_proj_bias = n_embd if bias else 0
    
    attention_params = attn_qkv_weight + attn_qkv_bias + attn_proj_weight + attn_proj_bias
    
    # ========================================================================
    # 2. MLP (FEED-FORWARD) PARAMETERS
    # ========================================================================
    mlp_hidden = mlp_ratio * n_embd
    
    # c_fc: First layer (expansion)
    # Shape: [n_embd] -> [mlp_ratio * n_embd]
    mlp_fc_weight = n_embd * mlp_hidden
    mlp_fc_bias = mlp_hidden if bias else 0
    
    # c_proj: Second layer (projection)
    # Shape: [mlp_ratio * n_embd] -> [n_embd]
    mlp_proj_weight = mlp_hidden * n_embd
    mlp_proj_bias = n_embd if bias else 0
    
    mlp_params = mlp_fc_weight + mlp_fc_bias + mlp_proj_weight + mlp_proj_bias
    
    # ========================================================================
    # 3. LAYER NORMALIZATION PARAMETERS
    # ========================================================================
    # Two LayerNorms per block: ln_1 (before attention), ln_2 (before MLP)
    # Each has weight and bias of size n_embd
    ln1_params = n_embd + (n_embd if bias else 0)
    ln2_params = n_embd + (n_embd if bias else 0)
    
    layernorm_params = ln1_params + ln2_params
    
    # ========================================================================
    # TOTAL
    # ========================================================================
    total_params = attention_params + mlp_params + layernorm_params
    
    return {
        'attention': attention_params,
        'mlp': mlp_params,
        'layernorm': layernorm_params,
        'total': total_params,
        'breakdown': {
            'attn_qkv_weight': attn_qkv_weight,
            'attn_qkv_bias': attn_qkv_bias,
            'attn_proj_weight': attn_proj_weight,
            'attn_proj_bias': attn_proj_bias,
            'mlp_fc_weight': mlp_fc_weight,
            'mlp_fc_bias': mlp_fc_bias,
            'mlp_proj_weight': mlp_proj_weight,
            'mlp_proj_bias': mlp_proj_bias,
            'ln1_params': ln1_params,
            'ln2_params': ln2_params,
        }
    }


def count_full_model_params(n_layer, n_embd, vocab_size, block_size, bias=True, 
                            weight_tying=True, mlp_ratio=4):
    """
    Count parameters in the full GPT model
    
    Args:
        n_layer: Number of transformer blocks
        n_embd: Hidden dimension
        vocab_size: Vocabulary size
        block_size: Context length (max sequence length)
        bias: Whether to include bias terms
        weight_tying: Whether embeddings are tied with output layer
        mlp_ratio: MLP expansion ratio
    
    Returns:
        Dictionary with parameter counts
    """
    
    # Embeddings
    token_emb = vocab_size * n_embd
    pos_emb = block_size * n_embd
    
    # Transformer blocks
    block_params = count_transformer_block_params(n_embd, bias=bias, mlp_ratio=mlp_ratio)
    total_block_params = block_params['total'] * n_layer
    
    # Final layer norm
    final_ln = n_embd + (n_embd if bias else 0)
    
    # Output projection (lm_head)
    # With weight tying, this shares weights with token embeddings
    lm_head = 0 if weight_tying else (vocab_size * n_embd)
    
    total = token_emb + pos_emb + total_block_params + final_ln + lm_head
    
    return {
        'token_embeddings': token_emb,
        'position_embeddings': pos_emb,
        'transformer_blocks': total_block_params,
        'blocks_per_layer': block_params['total'],
        'final_layernorm': final_ln,
        'lm_head': lm_head,
        'total': total,
        'weight_tying': weight_tying
    }


# ============================================================================
# FORMULAS (for quick reference)
# ============================================================================
def print_formulas():
    """Print the general formulas for parameter counting"""
    print("=" * 70)
    print("PARAMETER COUNTING FORMULAS")
    print("=" * 70)
    print()
    print("Given:")
    print("  d = n_embd (hidden dimension)")
    print("  V = vocab_size")
    print("  L = n_layer (number of blocks)")
    print("  T = block_size (context length)")
    print("  b = 1 if bias=True, 0 if bias=False")
    print()
    print("Per Transformer Block:")
    print("  Attention:")
    print("    QKV projection:  d × 3d + b×3d")
    print("    Output proj:     d × d + b×d")
    print("    Total:           4d² + 4bd")
    print()
    print("  MLP (with 4x expansion):")
    print("    Expansion:       d × 4d + b×4d")
    print("    Projection:      4d × d + b×d")
    print("    Total:           8d² + 5bd")
    print()
    print("  LayerNorm (×2):")
    print("    Total:           2d + 2bd")
    print()
    print("  Block Total:       12d² + 11bd")
    print()
    print("Full Model:")
    print("  Token embeddings:     V × d")
    print("  Position embeddings:  T × d")
    print("  Transformer blocks:   L × (12d² + 11bd)")
    print("  Final LayerNorm:      d + bd")
    print("  LM head:              0 (if weight tying) or V × d")
    print()
    print("  Total (with weight tying): V×d + T×d + L×(12d² + 11bd) + d + bd")
    print("=" * 70)


if __name__ == "__main__":
    # Example: Baby GPT configuration
    print("BABY GPT EXAMPLE")
    print("=" * 70)
    
    config = {
        'n_layer': 6,
        'n_embd': 384,
        'vocab_size': 65,
        'block_size': 256,
        'bias': True,
        'weight_tying': True
    }
    
    print(f"Configuration: {config}")
    print()
    
    # Count parameters per block
    block_params = count_transformer_block_params(config['n_embd'], bias=config['bias'])
    print(f"Parameters per transformer block: {block_params['total']:,}")
    print(f"  - Attention:   {block_params['attention']:,}")
    print(f"  - MLP:         {block_params['mlp']:,}")
    print(f"  - LayerNorm:   {block_params['layernorm']:,}")
    print()
    
    # Count full model parameters
    full_params = count_full_model_params(**config)
    print(f"Full model parameters: {full_params['total']:,}")
    print(f"  - Token embeddings:     {full_params['token_embeddings']:,}")
    print(f"  - Position embeddings:  {full_params['position_embeddings']:,}")
    print(f"  - Transformer blocks:   {full_params['transformer_blocks']:,}")
    print(f"  - Final LayerNorm:      {full_params['final_layernorm']:,}")
    print(f"  - LM head:              {full_params['lm_head']:,} (weight tying)")
    print()
    
    print("\n")
    print_formulas()
