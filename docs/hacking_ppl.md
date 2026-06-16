# Naive "gen-PPL hacking" Samplers

Zero-parameter samplers from *Hacking Generative Perplexity* (Franca & Tong, 2026).
They produce **incoherent text by construction** yet score competitive-to-SOTA
**generative perplexity (gen-PPL)** at non-degenerate entropy. Purpose: stress-test
gen-PPL and show it's an unsound quality metric.

## Why they work

gen-PPL rewards *predictability under a frozen AR scorer* (e.g. `gpt2-large`), not
coherence. The samplers exploit two patterns that are predictable but meaningless:

1. **High-frequency tokens** — common tokens incur low average loss under almost any
   scorer in almost any context.
2. **Temporal regularity** — an easily detectable copy, cycle, or template in the
   prefix lets the scorer predict the continuation with high confidence.

Each sampler targets a different mix of (i) and (ii):

| Sampler      | Pattern (i) freq | Pattern (ii) regularity | Output character                  |
|--------------|------------------|-------------------------|-----------------------------------|
| Top-k        | strong           | none                    | bag of common tokens, random order|
| Mirror-k     | strong           | exact copy (period L/2) | random block + verbatim copy      |
| Periodic-k   | strong           | strict cycle (period k) | fixed token loop                  |
| Phrase bank-m| via real phrases | weak/template, aperiodic| stitched real 5-grams, no message |

## Shared building block: restricted empirical marginal

Let `p̂(v) = count_train(v) / |X_train|` be the corpus relative frequency of token `v`,
and let `V_k ⊂ V` be the `k` most frequent token *types*. Renormalize frequencies
within the top-k:

```
p̂_k(v) = p̂(v) / Σ_{u ∈ V_k} p̂(u),   for v ∈ V_k
```

This is the sampling distribution for Top-k and Mirror-k. Periodic-k uses the *ordering*
of the top-k by frequency; Phrase bank uses 5-gram counts instead.

## 1. Top-k

Draw `L` tokens i.i.d. from `p̂_k` and concatenate in draw order. No temporal structure —
pure high-frequency exploitation.

## 2. Mirror-k

Draw `⌊L/2⌋` tokens i.i.d. from `p̂_k` to form the first half `x_1..x_{⌊L/2⌋}`, then set
`x_{⌊L/2⌋+i} = x_i`. The sequence is one random block followed by an **exact copy** of it.

## 3. Periodic-k

Sort the vocabulary by `p̂` descending; take the top-k tokens `v_1..v_k`. Emit them in
fixed order and loop: position `i` holds `v_{((i-1) mod k)+1}`. For `k=4`:
`v1 v2 v3 v4 v1 v2 v3 v4 ...` truncated to `L`. **Deterministic** (no randomness).
This is the strongest gen-PPL hacker — the scorer locks onto the cycle immediately.

## 4. Phrase bank-m

Count every **5-gram** in the training corpus; keep the top-`m` by count to form a bank
`B_m`. To generate, draw 5-grams **uniformly** from `B_m` (deliberately *not* weighted by
frequency) and concatenate until length `≥ L`, then truncate. Output is random and
non-periodic but made of real phrases.

## Reference implementation (Python)

```python
from collections import Counter
import numpy as np

# ---- corpus stats -----------------------------------------------------------
def token_counts(corpus_token_ids):
    """corpus_token_ids: flat iterable of token ids (or chain docs together)."""
    return Counter(corpus_token_ids)

def restricted_marginal(counts, k):
    """Top-k types and their renormalized frequencies p̂_k."""
    top = counts.most_common(k)
    types = np.array([t for t, _ in top])
    freqs = np.array([c for _, c in top], dtype=float)
    return types, freqs / freqs.sum()

# ---- 1. Top-k ---------------------------------------------------------------
def top_k_sampler(types, p_k, L, rng):
    idx = rng.choice(len(types), size=L, p=p_k)
    return types[idx].tolist()

# ---- 2. Mirror-k ------------------------------------------------------------
def mirror_k_sampler(types, p_k, L, rng):
    half = L // 2
    idx = rng.choice(len(types), size=half, p=p_k)
    first = types[idx].tolist()
    seq = list(first)
    for i in range(L - half):          # x_{half+i} = x_i
        seq.append(first[i])
    return seq[:L]

# ---- 3. Periodic-k (deterministic) ------------------------------------------
def periodic_k_sampler(counts, k, L):
    cycle = [t for t, _ in counts.most_common(k)]   # v_1..v_k by freq desc
    return [cycle[i % k] for i in range(L)]

# ---- 4. Phrase bank-m -------------------------------------------------------
def build_phrase_bank(corpus_docs, m, n=5):
    """corpus_docs: iterable of token-id lists (per document)."""
    ng = Counter()
    for doc in corpus_docs:
        for i in range(len(doc) - n + 1):
            ng[tuple(doc[i:i+n])] += 1
    return [g for g, _ in ng.most_common(m)]        # top-m 5-grams

def phrase_bank_sampler(bank, L, rng, n=5):
    seq = []
    while len(seq) < L:
        g = bank[int(rng.integers(len(bank)))]      # UNIFORM, not freq-weighted
        seq.extend(g)
    return seq[:L]
```

## How to use them (evaluation protocol)

The point is to feed sampler output through the *same* pipeline as real models and show
gen-PPL fails while distributional metrics catch them.

- **Scorer:** `gpt2-large`, matching the models you compare against.
- **Sampling settings used in the paper:** 1024 samples; `L=128` for LM1B, `L=1024` for
  OWT; pool statistics by averaging.
- **Report gen-PPL** (eq. 7): `exp( E_s[ (1/(L-1)) Σ_{i=2}^{L} −log p_θ(s_i | s_<i) ] )`.
- **Report empirical unigram entropy** `H_emp = −Σ_v p̂_s(v) log p̂_s(v)` as the
  non-degeneracy guardrail — you want `H` in the same ballpark as real text, not collapsed.
- **Then run distributional metrics** (MAUVE, Energy distance `D_E`, Gradient Moment GM,
  FMTyp-p) plus the `Rep-n` degeneracy diagnostic to show the samplers are correctly
  flagged as bad even though gen-PPL likes them.

## Parameter settings worth reproducing

These are the headline configs (achieve strong/SOTA gen-PPL while incoherent):

```
LM1B  (L=128):
  Top-k   k=32     gen-PPL 75.0   H 2.99
  Mirror  k=5000   gen-PPL 61.3   H 3.84
  Periodic k=64    gen-PPL 29.4   H 4.16   <- beats reference train (56.9)
  Phrase  m=1000   gen-PPL 78.1   H 4.03
  (reference train: gen-PPL 56.9, H 4.33)

OWT   (L=1024):
  Top-k   k=64     gen-PPL 99.9   H 3.78
  Mirror  k=5000   gen-PPL 50.6   H 5.14
  Periodic k=64    gen-PPL  2.10  H 4.16   <- absurdly low
  Periodic k=400   gen-PPL 21.6   H 5.97
  Phrase  m=5000   gen-PPL 60.7   H 4.63
  (reference train: gen-PPL 17.2, H 5.48)
```

Full sweeps (paper Tables 3–6) show the freq/regularity trade-off directly: as `k` or `m`
grows, entropy `H` rises (more diverse) but gen-PPL worsens (less predictable). Periodic-k
is the clearest exploit — small-to-moderate `k` gives very low gen-PPL at decent `H`.

## Implementation gotchas

- **Two different corpora roles.** The marginals / 5-gram bank are computed on the
  *training data* (LM1B or OWT). The scorer for gen-PPL is `gpt2-large`. Keep the
  tokenizer consistent with whichever you're feeding where.
- **Phrase bank uses a uniform draw over the bank**, not frequency-weighted — this is
  intentional and matters for the resulting statistics. Don't "fix" it.
- **Periodic-k is deterministic**; every sample is identical given `k`. That's fine for
  the experiment (you're measuring a fixed degenerate distribution).
- **Mirror odd-length:** the copy fills `L − ⌊L/2⌋` tokens from the start of the first
  half, so the second half can be one token longer than the first.
- **Entropy guardrail:** if you want a sampler to look "non-degenerate," tune `k`/`m` so
  `H_emp` lands near the reference corpus entropy; otherwise the entropy companion metric
  exposes collapse on its own.