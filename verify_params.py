"""
Verify parameter counts by actually loading the model
This shows you how to inspect a real PyTorch model
"""

import sys
from model import GPT, GPTConfig

def count_parameters(model, verbose=True):
    """
    Count parameters in a PyTorch model
    
    Args:
        model: PyTorch model
        verbose: Whether to print detailed breakdown
    
    Returns:
        Total number of parameters
    """
    
    if verbose:
        print("=" * 70)
        print("MODEL PARAMETER INSPECTION")
        print("=" * 70)
    
    total_params = 0
    components = {}
    
    for name, param in model.named_parameters():
        num_params = param.numel()
        total_params += num_params
        
        # Group by component
        component = name.split('.')[0] if '.' in name else name
        if component not in components:
            components[component] = 0
        components[component] += num_params
        
        if verbose:
            print(f"{name:50s} {list(param.shape):30s} {num_params:>12,}")
    
    if verbose:
        print("=" * 70)
        print("\nComponent Summary:")
        print("-" * 70)
        for component, count in sorted(components.items()):
            print(f"{component:20s} {count:>12,} ({count/total_params*100:>5.2f}%)")
        print("-" * 70)
        print(f"{'TOTAL':20s} {total_params:>12,}")
        print("=" * 70)
    
    return total_params


def count_block_parameters(model):
    """
    Count parameters in a single transformer block
    """
    # Get the first block
    block = model.transformer.h[0]
    
    print("\n" + "=" * 70)
    print("SINGLE TRANSFORMER BLOCK BREAKDOWN")
    print("=" * 70)
    
    attention_params = 0
    mlp_params = 0
    ln_params = 0
    
    for name, param in block.named_parameters():
        num_params = param.numel()
        print(f"{name:40s} {list(param.shape):30s} {num_params:>10,}")
        
        if 'attn' in name:
            attention_params += num_params
        elif 'mlp' in name:
            mlp_params += num_params
        elif 'ln' in name:
            ln_params += num_params
    
    total_block = attention_params + mlp_params + ln_params
    
    print("-" * 70)
    print(f"{'Attention subtotal:':40s} {attention_params:>10,}")
    print(f"{'MLP subtotal:':40s} {mlp_params:>10,}")
    print(f"{'LayerNorm subtotal:':40s} {ln_params:>10,}")
    print("-" * 70)
    print(f"{'BLOCK TOTAL:':40s} {total_block:>10,}")
    print("=" * 70)
    
    return {
        'attention': attention_params,
        'mlp': mlp_params,
        'layernorm': ln_params,
        'total': total_block
    }


def compare_with_formula(config):
    """
    Compare actual parameter count with formula calculation
    """
    d = config.n_embd
    V = config.vocab_size
    L = config.n_layer
    T = config.block_size
    b = 1  # bias=True
    
    # Per block: 12d² + 11bd
    block_params_formula = 12 * d * d + 11 * b * d
    
    # Full model with weight tying: V×d + T×d + L×(12d² + 11bd) + d + bd
    total_params_formula = V*d + T*d + L*block_params_formula + d + b*d
    
    print("\n" + "=" * 70)
    print("FORMULA VERIFICATION")
    print("=" * 70)
    print(f"Configuration: d={d}, V={V}, L={L}, T={T}, bias=True")
    print()
    print(f"Per block formula: 12d² + 11bd")
    print(f"                 = 12×{d}² + 11×{d}")
    print(f"                 = {block_params_formula:,}")
    print()
    print(f"Full model formula: V×d + T×d + L×(12d² + 11bd) + d + bd")
    print(f"                  = {V}×{d} + {T}×{d} + {L}×{block_params_formula:,} + {d} + {d}")
    print(f"                  = {total_params_formula:,}")
    print("=" * 70)
    
    return block_params_formula, total_params_formula


if __name__ == "__main__":
    # Create Baby GPT model
    config = GPTConfig(
        block_size=256,
        vocab_size=65,  # Shakespeare character-level
        n_layer=6,
        n_head=6,
        n_embd=384,
        dropout=0.2,
        bias=True
    )
    
    print("\nCreating Baby GPT model...")
    model = GPT(config)
    print(f"Model created with config:")
    print(f"  n_layer={config.n_layer}, n_embd={config.n_embd}, n_head={config.n_head}")
    print(f"  block_size={config.block_size}, vocab_size={config.vocab_size}")
    print()
    
    # Count all parameters
    total = count_parameters(model, verbose=True)
    
    # Count parameters in a single block
    block_counts = count_block_parameters(model)
    
    # Compare with formula
    block_formula, total_formula = compare_with_formula(config)
    
    print("\n" + "=" * 70)
    print("VERIFICATION SUMMARY")
    print("=" * 70)
    print(f"Block parameters (actual):  {block_counts['total']:>12,}")
    print(f"Block parameters (formula): {block_formula:>12,}")
    print(f"Match: {block_counts['total'] == block_formula}")
    print()
    print(f"Total parameters (actual):  {total:>12,}")
    print(f"Total parameters (formula): {total_formula:>12,}")
    print(f"Match: {total == total_formula}")
    print("=" * 70)
    
    # Additional insights
    print("\n" + "=" * 70)
    print("KEY INSIGHTS")
    print("=" * 70)
    print(f"1. MLP has ~2× more parameters than Attention")
    print(f"   MLP:       {block_counts['mlp']:,} ({block_counts['mlp']/block_counts['total']*100:.1f}%)")
    print(f"   Attention: {block_counts['attention']:,} ({block_counts['attention']/block_counts['total']*100:.1f}%)")
    print()
    print(f"2. LayerNorm is negligible: {block_counts['layernorm']:,} ({block_counts['layernorm']/block_counts['total']*100:.2f}%)")
    print()
    print(f"3. Most parameters are in transformer blocks:")
    print(f"   {config.n_layer} blocks × {block_counts['total']:,} = {config.n_layer * block_counts['total']:,}")
    print()
    print(f"4. Embeddings are relatively small due to small vocab:")
    print(f"   Token emb:  {config.vocab_size * config.n_embd:,}")
    print(f"   Pos emb:    {config.block_size * config.n_embd:,}")
    print("=" * 70)
