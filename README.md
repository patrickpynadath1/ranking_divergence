# Ranking Divergence

Lightweight evaluation utilities for LLM rank-histogram divergences.

The main metric follows `notes.pdf`: given a reference causal language model,
compute the histogram of the rank assigned to each observed next token, then
compare two histograms with the closed-form one-dimensional Wasserstein-1
distance on the log-rank axis.

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from ranking_divergence import rank_wasserstein

tokenizer = AutoTokenizer.from_pretrained("gpt2-large")
tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForCausalLM.from_pretrained("gpt2-large")

result = rank_wasserstein(
    reference_texts=["Human-written reference text."],
    comparison_texts=["Generated text to evaluate."],
    model=model,
    tokenizer=tokenizer,
)
print(result.distance)
```

## Install

```bash
uv pip install -e .
```

For the OpenWebText example:

```bash
uv pip install -e ".[examples]"
```

## Included Metrics

- Rank-Wasserstein divergence over next-token rank histograms
- Generative perplexity under a fixed causal LM scorer
- Empirical unigram entropy
- Rep-n repetition score
- Four zero-parameter samplers from `baselines_to_use.pdf`: Top-k IID,
  Mirror-k, Periodic-k, and Phrase bank-m

## OpenWebText Analysis Pipeline

The full OpenWebText analysis script follows the split/cache/model conventions in
`docs/openwebtext_analysis_plan.md`: `sample_length=1024`, sampler source
`train[:-100000]`, held-out reference `train[-100000:]`, cache
`/home/patrick/.cache/discrete_diffusion/owt`, GPT-2 tokenizer, Hugging Face
`gpt2` generation, and `gpt2-large` scoring.

```bash
uv run python examples/openwebtext_analysis.py \
  --cache-dir /home/patrick/.cache/discrete_diffusion/owt \
  --output-dir outputs/openwebtext_analysis \
  --scorer-model gpt2-large \
  --generator-model gpt2 \
  --num-reference 128 \
  --num-sampler-source 4096 \
  --num-samples 16
```

Each run writes a timestamped directory containing generated sample text, token
IDs, metadata, `metrics.csv`, `metrics.json`, plots, and Markdown/LaTeX tables.
The default run downloads OpenWebText and Hugging Face models if they are not
already cached.

For a quick toy run, the older small example remains available:

```bash
uv run python examples/evaluate_gpt2_openwebtext.py --num-reference 32 --num-samples 16
```
