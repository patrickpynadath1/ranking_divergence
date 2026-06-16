# LLM Rank-Wasserstein Evaluation

Method for evaluating language models (including diffusion LMs) by comparing how a
**reference autoregressive LLM** ranks the next token across a real dataset vs. a
generated dataset, then measuring the divergence with an optimal-transport distance.

## Core idea

You don't score the model being evaluated directly. Instead you use a fixed reference
autoregressive LLM `p_θ` as a "ruler": for each real next-token transition, you record
the **rank** that `p_θ` assigns to the actually-observed token. Do this for both the
real data and the generated samples, build two rank histograms, and compare them.

- If the generator matches the data's next-token statistics, the two histograms align → distance ≈ 0.
- If the generator drifts (e.g. over-produces low-rank/unlikely tokens), the histograms diverge → large distance.

## Notation

- Vocabulary `V`, sequences `x̄ = (x_1, ..., x_T)`, `x_j ∈ V`.
- `P_i:(x̄)` = reference LLM's next-token probability vector at position `i`.
- `rank_{p_θ}(token)` = position of `token` when probs are sorted descending (rank 1 = most likely).

## Step 1 — Rank histogram

For a dataset `D` and reference LLM `p_θ`, build a histogram over ranks `j = 1..V`:

```
H_j = Σ_{x̄ ∈ D} Σ_i  1{ rank_{p_θ}(x_{i+1} | x_1..x_i) = j }
```

In words: at every position, find the rank the reference model gives the true next
token, and increment that rank's bin. Then normalize:

```
h_j = H_j / Σ_ℓ H_ℓ        (so Σ_j h_j = 1)
```

Compute this for both `h_data` (from real data) and `h_gen` (from generated samples).

## Step 2 — Cost / ground metric

Transport cost between rank `i` and rank `j` is the log-rank distance:

```
c_ij = | log i − log j |
```

Using `log` emphasizes differences among the **top ranks** (where it matters most)
while staying tractable over a large vocabulary. This makes it a 1-D transport problem
on ordered points `z_j = log j`.

## Step 3 — Closed-form optimal transport distance

Because the cost is a 1-D metric on the log-rank axis, the OT distance is just the
**Wasserstein-1 distance** between the two histograms, which has a closed form via
cumulative discrepancies.

Cumulative discrepancy up to rank `k`:

```
Δ_k = Σ_{j=1}^{k} ( h_data_j − h_gen_j )
```

Distance:

```
d_OT = Σ_{k=1}^{V-1} | Δ_k | · ( log(k+1) − log k )
```

Intuition: `|Δ_k|` is the mass that must cross the boundary between consecutive ranks
`k` and `k+1`; moving it costs `log(k+1) − log k`. Sum over all boundaries.

**Caveats:**
- Both histograms must have the same total mass. If counts differ, normalize first.
- If you can't/won't normalize, use an *unbalanced* OT distance instead.

## Algorithm (from the paper)

```
function RANKHISTOGRAM(D, p_θ):
    H ← zeros(V)
    for sequence x̄ in D:
        for i = 1 .. T-1:
            probs ← p_θ(· | x_1..x_i)
            r ← rank of observed token x_{i+1} under probs   # rank 1 = highest prob
            H[r] ← H[r] + 1
    return H / sum(H)

h_data ← RANKHISTOGRAM(D_data, p_θ)
h_gen  ← RANKHISTOGRAM(D_gen,  p_θ)

Δ ← 0;  d_OT ← 0
for k = 1 .. V-1:
    Δ ← Δ + h_data[k] − h_gen[k]
    d_OT ← d_OT + |Δ| · (log(k+1) − log k)
return d_OT
```

## Reference implementation sketch (Python)

```python
import numpy as np
import torch

def rank_histogram(dataset, model, tokenizer, vocab_size):
    """dataset: iterable of token-id sequences (1D arrays)."""
    H = np.zeros(vocab_size, dtype=np.float64)
    for seq in dataset:
        seq = torch.as_tensor(seq)
        # logits: [T, V]; row i predicts token i+1
        logits = model(seq.unsqueeze(0)).logits[0]
        for i in range(len(seq) - 1):
            true_id = seq[i + 1].item()
            row = logits[i]
            # rank of the true token: how many tokens have strictly higher logit, +1
            rank = int((row > row[true_id]).sum().item()) + 1  # 1-indexed
            H[rank - 1] += 1.0
    total = H.sum()
    return H / total if total > 0 else H

def log_rank_wasserstein(h_data, h_gen):
    """Both inputs: normalized histograms of equal length V."""
    V = len(h_data)
    delta = 0.0
    d_ot = 0.0
    for k in range(1, V):  # boundaries between rank k and k+1 (1-indexed)
        delta += h_data[k - 1] - h_gen[k - 1]
        d_ot += abs(delta) * (np.log(k + 1) - np.log(k))
    return d_ot

# Vectorized equivalent of log_rank_wasserstein:
def log_rank_wasserstein_vec(h_data, h_gen):
    delta = np.cumsum(h_data - h_gen)[:-1]          # Δ_1 .. Δ_{V-1}
    k = np.arange(1, len(h_data))
    weights = np.log(k + 1) - np.log(k)
    return np.sum(np.abs(delta) * weights)
```

## Implementation notes / gotchas

- **The reference LLM is fixed and shared** between the data and generated passes. Use
  the *same* `p_θ` for both histograms — it's the common measuring stick, not the model
  under evaluation.
- **Rank is 1-indexed** (rank 1 = most likely). The closed-form sum uses ranks `1..V-1`
  for the boundaries.
- **Ties in logits:** decide a convention (the `> ` count above is a "strict" rank;
  ties get the same rank). Be consistent across both histograms.
- **Efficiency:** computing exact ranks over full vocab `V` is cheap per step
  (`argsort` or a comparison count). The per-token loop dominates; batch sequences
  through the model and vectorize the rank computation per sequence.
- **Equal sample sizes:** the paper compares an "equally sized generated set." Normalizing
  removes strict size requirements, but matched sizes reduce sampling noise.
- **Diffusion / non-AR generators:** you only need their *generated text*. The reference
  AR model scores both real and generated text; the generator itself never needs to be AR.