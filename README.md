# DDTree-MLX

**Tree-based speculative decoding for Apple Silicon.** Up to **2.6x faster** than autoregressive generation on Qwen 3.5 27B.

DDTree extends [DFlash](https://github.com/bstnxbt/dflash-mlx) speculative decoding by building a **draft tree** from per-position logits and verifying the entire tree in one forward pass. Instead of betting on a single draft sequence, DDTree explores multiple likely continuations simultaneously, accepting more tokens per verification cycle.

Based on the paper [*Accelerating Speculative Decoding with Block Diffusion Draft Trees*](https://liranringel.github.io/ddtree/DDTree.pdf) by Liran Ringel & Yaniv Romano. This is an independent MLX port for Apple Silicon, with a custom Metal kernel for hybrid model support.

## Performance

Qwen 3.5 27B 4-bit on Mac Studio M3 Ultra 256GB:

| Output Length | Autoregressive | DFlash | DDTree | Speedup vs AR |
|--------------:|---------------:|-------:|-------:|--------------:|
| 1K tokens | 37.8 tok/s | 53.2 tok/s | ~73 tok/s | **1.9x** |
| 2K tokens | 37.8 tok/s | 58.0 tok/s | ~72 tok/s | **1.9x** |
| 4K tokens | 37.4 tok/s | 67.1 tok/s | ~90 tok/s | **2.4x** |
| 8K tokens | 36.7 tok/s | 72.7 tok/s | ~95 tok/s | **2.6x** |
| 16K tokens | 36.2 tok/s | 74.0 tok/s | ~73 tok/s | **2.0x** |

DDTree is **1.24x faster than DFlash on average**, reaching **1.5-1.6x on prompts where the draft model has moderate acceptance** (68-70%). Output is lossless -- every token is verified against the target model.

## How It Works

1. **Draft**: The DFlash block diffusion model generates per-position token probabilities in parallel
2. **Tree Build**: A heap-based algorithm constructs an optimal draft tree from the top-K tokens at each position, maximizing coverage under a node budget
3. **Tree Verify**: All tree nodes are verified through the target model in one forward pass using tree attention masks (ancestor-only visibility) and per-token RoPE positions
4. **Tree Walk**: Greedy walk through the verified tree to find the longest accepted path
5. **Commit**: The accepted path's cache state is installed directly via per-node state capture (zero-cost commit)

### Hybrid Model Support

Qwen 3.5 27B is a hybrid architecture with 48 GatedDeltaNet (recurrent) layers and 16 full attention layers. DDTree handles this with:

- **Attention layers**: Process all tree nodes in parallel via custom tree attention masks
- **Recurrent layers**: A custom Metal kernel performs parent-indexed GatedDelta recurrence, forking state at each branch point so every tree path gets exact logits
- **Tree-aware commit**: Accepted path's recurrent state is installed directly from captured per-node states, eliminating the need for re-forward passes

## Installation

Requires Python 3.11+ and Apple Silicon (M1/M2/M3/M4).

```bash
# Install dflash-mlx (required dependency)
pip install dflash-mlx

# Clone and install ddtree-mlx
git clone https://github.com/humanrouter/ddtree-mlx.git
cd ddtree-mlx
pip install -e .
```

The target model and DFlash drafter will be downloaded automatically on first run from Hugging Face:
- Target: `mlx-community/Qwen3.5-27B-4bit` (~16GB)
- Drafter: `z-lab/Qwen3.5-27B-DFlash` (~3GB)
- Total memory: ~19GB

## Usage

### OpenAI-Compatible Server

```bash
python ddtree_server.py --port 8006
```

Then use any OpenAI-compatible client:

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8006/v1", api_key="unused")
response = client.chat.completions.create(
    model="ddtree",
    messages=[{"role": "user", "content": "Explain TCP vs UDP"}],
    max_tokens=2048,
)
print(response.choices[0].message.content)
```

### Python API

```python
from dflash_mlx.generate import load_runtime_components, get_stop_token_ids
from ddtree_mlx.runtime import generate_ddtree_once

# Load models (downloads from HF on first run)
target_model, tokenizer, draft_model, _ = load_runtime_components(
    model_ref="mlx-community/Qwen3.5-27B-4bit"
)

# Tokenize
prompt_tokens = list(tokenizer.apply_chat_template(
    [{"role": "user", "content": "Write a Python quicksort"}],
    tokenize=True, add_generation_prompt=True, enable_thinking=False,
))

# Generate
result = generate_ddtree_once(
    target_model=target_model,
    draft_model=draft_model,
    tokenizer=tokenizer,
    prompt_tokens=prompt_tokens,
    max_new_tokens=2048,
    tree_budget=4,
    stop_token_ids=get_stop_token_ids(tokenizer),
)

print(tokenizer.decode(result["generated_token_ids"]))
print(f"{result['tokens_per_second']:.1f} tok/s, "
      f"{result['avg_acceptance']:.1f} tokens/cycle, "
      f"{result['fast_path_ratio']:.0%} fast path")
```

### Benchmarking

```bash
python benchmark.py --max-tokens 2048 --budgets 4 --prompts 3
```

## Configuration

| Environment Variable | Default | Description |
|---------------------|---------|-------------|
| `DDTREE_BUDGET` | `4` | Tree node budget (excluding root). Budget 4 is optimal for hybrid models. |
| `DDTREE_TREE_AWARE_LINEAR` | `1` | Enable parent-state forking for recurrent layers (recommended). |
| `DDTREE_TREE_KERNEL` | `1` | Use custom Metal kernel for tree-aware GatedDelta recurrence. |
| `DDTREE_PROFILE_VERIFY` | `0` | Profile linear vs attention layer timing within tree verify. |

## Architecture

```
ddtree_mlx/
  tree.py       # Heap-based tree construction (Algorithm 1 from the paper)
  compile.py    # Converts tree structure to MLX tensors (masks, positions, DFS order)
  verify.py     # Custom forward pass: tree attention + parent-indexed recurrence
  kernels.py    # Metal kernel for tree-aware GatedDelta state update
  cache.py      # Cache management: snapshot, rollback, tree-aware path commit
  runtime.py    # Main generate loop: draft -> build -> verify -> walk -> commit
ddtree_server.py  # OpenAI-compatible FastAPI server
benchmark.py      # Benchmark script (DDTree vs DFlash comparison)
```

## Findings & Insights

See [BENCHMARKS.md](BENCHMARKS.md) for detailed results, including:

- **What worked**: Metal kernel for tree-aware recurrence, zero-cost commit via per-node state capture, eval sync point reduction
- **What didn't work**: Attention-only tree verify (LM head needs all 64 layers), alternative tree shapes (chain, hybrid, root-wide), split prefix/tree attention, adaptive budget controller
- **The fundamental constraint**: On hybrid models (75% recurrent layers), tree verification has limited parallelism. DDTree's advantage comes from better acceptance density -- the tree concentrates budget on the most probable tokens. Pure-attention models (Llama, standard Qwen) would benefit more.

## Citation

```bibtex
@article{ringel2025ddtree,
  title={Accelerating Speculative Decoding with Block Diffusion Draft Trees},
  author={Ringel, Liran and Romano, Yaniv},
  year={2025},
  url={https://liranringel.github.io/ddtree/}
}
```

## License

MIT
