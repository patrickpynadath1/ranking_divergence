# OpenWebText Analysis Pipeline Plan

## Goal

Build a reproducible OpenWebText analysis pipeline that compares Hugging Face
`gpt2` samples against four zero-parameter samplers from `baselines_to_use.pdf`.
The pipeline should generate and cache samples, compute the evaluation metrics
used in this repository, and produce tables and plots suitable for analysis
notes or a paper draft.

## Data And Runtime Defaults

- Use `sample_length=1024` everywhere: GPT-2 generation, fake sampler outputs,
  generative perplexity truncation/context, rank histograms, and n-gram metrics.
- Do not copy DUO's `small.length: 128`; that length is a bug for this analysis.
- Use DUO's OpenWebText split convention:
  - sampler/training source: `openwebtext`, split `train[:-100000]`
  - held-out reference/evaluation source: `openwebtext`, split `train[-100000:]`
- Use DUO's local cache convention:
  - `scratch_dir=/home/patrick/.cache/discrete_diffusion`
  - OpenWebText cache path: `/home/patrick/.cache/discrete_diffusion/owt`
- Use the `gpt2` tokenizer for OpenWebText and generated samples.
- Set `tokenizer.pad_token = tokenizer.eos_token` for GPT-2-family tokenizers.
- Use `gpt2-large` as the scorer/reference language model for generative
  perplexity and rank-Wasserstein.

## Generators

Evaluate these sample sources with the same `sample_length=1024` and shared
random seed:

- Hugging Face `gpt2` causal LM, sampled with configurable `temperature`,
  `top_p`, `num_samples`, and `max_new_tokens=1024`.
- Top-k IID sampler using OpenWebText train token frequencies.
- Mirror-k sampler using OpenWebText train token frequencies.
- Periodic-k sampler using the top-k OpenWebText train tokens in fixed order.
- Phrase bank-m sampler using the top 5-grams from OpenWebText train text.

Sampler parameters should be CLI-configurable, with defaults matching the OWT
paper-style setup where possible:

- `top_k=64`
- `mirror_k=5000`
- `periodic_k=400`
- `phrase_bank_m=5000`
- `phrase_bank_n=5`

## Metrics

Compute all metrics for every generator/configuration:

- Unigram entropy: average per-sample token entropy, matching the DUO metric
  behavior.
- Generative perplexity: shifted cross-entropy under `gpt2-large`, using valid
  next-token positions only and chunking to the scorer context window as needed.
- Rank-Wasserstein: compute the held-out OpenWebText rank histogram and each
  generated rank histogram under `gpt2-large`, then report the closed-form
  log-rank Wasserstein distance.
- Unique n-grams: report per-sample and corpus-level unique ratios for
  `n=1,2,3,4`.
- Optional Rep-n: include Rep-n for `n=1,2,3` as the repetition counterpart to
  unique n-gram statistics.

## Outputs

Write all analysis artifacts under an output directory such as:

```text
outputs/openwebtext_analysis/<timestamp>/
```

Required artifacts:

- generated sample text for each generator/configuration
- generated token IDs for each generator/configuration
- generator metadata, including seed, `sample_length=1024`, sampler parameters,
  model names, and split/cache paths
- `metrics.csv` with one row per generator/configuration
- `metrics.json` with full provenance and metric values
- plots comparing generators across:
  - unigram entropy
  - generative perplexity
  - rank-Wasserstein
  - unique n-grams
  - gen-PPL vs rank-Wasserstein
  - entropy vs rank-Wasserstein
- paper-ready Markdown or LaTeX tables

## Implementation Shape

- Keep reusable metric and sampler logic in the `ranking_divergence` package.
- Add OpenWebText orchestration as an example or analysis script, not as core
  package behavior.
- Recommended main script path:

```text
examples/openwebtext_analysis.py
```

- The script should expose CLI flags for:
  - `--cache-dir /home/patrick/.cache/discrete_diffusion/owt`
  - `--output-dir outputs/openwebtext_analysis`
  - `--sample-length 1024`
  - `--num-samples`
  - `--num-reference`
  - `--num-sampler-source`
  - `--scorer-model gpt2-large`
  - `--generator-model gpt2`
  - `--batch-size`
  - `--seed`
  - `--temperature`
  - `--top-p`
  - `--top-k`
  - `--mirror-k`
  - `--periodic-k`
  - `--phrase-bank-m`
  - `--phrase-bank-n`
- Add plotting as either:
  - a separate script that reads `metrics.csv`, or
  - a `--plot-only` / `--skip-generation` mode on the main analysis script.
- Do not import DUO directly. Mirror its split, cache, tokenizer, and scorer
  conventions so this repo remains lightweight and portable.

## Testing

- Unit-test OpenWebText split/config helpers by mocking `datasets.load_dataset`;
  do not require real OpenWebText downloads in tests.
- Unit-test each sampler on tiny tokenized text with deterministic seeds.
- Unit-test unique n-gram metrics on hand-computable samples.
- Add a CLI smoke test for argument parsing and output path construction.
- Keep the full GPT-2/OpenWebText run as a manual integration run because it
  requires model and dataset downloads.

## Assumptions

- `sample_length=1024` is required for this OpenWebText analysis.
- DUO's `small.length: 128` is a bug and should not be used.
- `gpt2-large` is the scorer for both generative perplexity and rank-Wasserstein.
- `train[-100000:]` is the held-out OpenWebText reference distribution.
- `/home/patrick/.cache/discrete_diffusion/owt` is the local OpenWebText cache.
- This repository should remain independent of DUO internals.
