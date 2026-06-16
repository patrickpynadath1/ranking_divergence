from __future__ import annotations

from collections import Counter
from typing import Sequence

import torch
import torch.nn.functional as F
from tqdm.auto import tqdm


def _tokenize_texts(texts: str | Sequence[str], tokenizer) -> list[list[int]]:
    samples = [texts] if isinstance(texts, str) else list(texts)
    return [tokenizer.encode(text, add_special_tokens=False) for text in samples]


def empirical_entropy(
    texts: str | Sequence[str],
    tokenizer=None,
    *,
    token_ids: Sequence[Sequence[int]] | None = None,
) -> float:
    """Empirical unigram entropy in nats."""

    if token_ids is None:
        if tokenizer is None:
            raise ValueError("Pass tokenizer or token_ids.")
        token_ids = _tokenize_texts(texts, tokenizer)
    counts: Counter[int] = Counter()
    for ids in token_ids:
        counts.update(ids)
    total = sum(counts.values())
    if total == 0:
        return 0.0
    probs = torch.tensor(list(counts.values()), dtype=torch.float64) / total
    return float(-(probs * probs.log()).sum().item())


def per_sample_unigram_entropy(
    texts: str | Sequence[str],
    tokenizer=None,
    *,
    token_ids: Sequence[Sequence[int]] | None = None,
) -> float:
    """Average per-sample empirical unigram entropy in nats."""

    if token_ids is None:
        if tokenizer is None:
            raise ValueError("Pass tokenizer or token_ids.")
        token_ids = _tokenize_texts(texts, tokenizer)

    values: list[float] = []
    for ids in token_ids:
        counts = Counter(ids)
        total = sum(counts.values())
        if total == 0:
            values.append(0.0)
            continue
        probs = torch.tensor(list(counts.values()), dtype=torch.float64) / total
        values.append(float(-(probs * probs.log()).sum().item()))
    return float(sum(values) / len(values)) if values else 0.0


def unique_ngram_ratios(
    texts: str | Sequence[str],
    tokenizer=None,
    *,
    n: int = 3,
    token_ids: Sequence[Sequence[int]] | None = None,
) -> dict[str, float]:
    """Return per-sample and corpus-level unique n-gram ratios."""

    if n <= 0:
        raise ValueError("n must be positive.")
    if token_ids is None:
        if tokenizer is None:
            raise ValueError("Pass tokenizer or token_ids.")
        token_ids = _tokenize_texts(texts, tokenizer)

    sample_values: list[float] = []
    corpus_grams: Counter[tuple[int, ...]] = Counter()
    corpus_windows = 0
    for ids in token_ids:
        windows = len(ids) - n + 1
        if windows <= 0:
            continue
        grams = [tuple(ids[i : i + n]) for i in range(windows)]
        sample_values.append(len(set(grams)) / windows)
        corpus_grams.update(grams)
        corpus_windows += windows

    return {
        "sample": float(sum(sample_values) / len(sample_values)) if sample_values else 0.0,
        "corpus": float(len(corpus_grams) / corpus_windows) if corpus_windows else 0.0,
    }


def rep_n(
    texts: str | Sequence[str],
    tokenizer=None,
    *,
    n: int = 3,
    token_ids: Sequence[Sequence[int]] | None = None,
) -> float:
    """Corpus average Rep-n from the baseline paper.

    Rep-n is ``1 - distinct n-grams / total n-gram windows`` per sample.
    """

    if n <= 0:
        raise ValueError("n must be positive.")
    if token_ids is None:
        if tokenizer is None:
            raise ValueError("Pass tokenizer or token_ids.")
        token_ids = _tokenize_texts(texts, tokenizer)

    values: list[float] = []
    for ids in token_ids:
        windows = len(ids) - n + 1
        if windows <= 0:
            continue
        grams = {tuple(ids[i : i + n]) for i in range(windows)}
        values.append(1.0 - len(grams) / windows)
    return float(sum(values) / len(values)) if values else 0.0


@torch.inference_mode()
def generative_perplexity(
    texts: str | Sequence[str],
    model,
    tokenizer,
    *,
    batch_size: int = 8,
    max_length: int | None = None,
    device: str | torch.device | None = None,
    show_progress: bool = False,
) -> float:
    """Per-token perplexity of samples under a causal language model."""

    samples = [texts] if isinstance(texts, str) else list(texts)
    if not samples:
        raise ValueError("At least one text sample is required.")
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    if device is None:
        device = next(model.parameters()).device
    device = torch.device(device)
    model = model.to(device)
    model.eval()

    total_loss = 0.0
    total_tokens = 0
    iterator = range(0, len(samples), batch_size)
    if show_progress:
        iterator = tqdm(iterator, desc="gen-ppl")

    for start in iterator:
        encoded = tokenizer(
            samples[start : start + batch_size],
            return_tensors="pt",
            padding=True,
            truncation=max_length is not None,
            max_length=max_length,
            return_attention_mask=True,
        )
        input_ids = encoded["input_ids"].to(device)
        attention_mask = encoded["attention_mask"].to(device)
        if input_ids.shape[1] < 2:
            continue
        logits = model(input_ids=input_ids, attention_mask=attention_mask).logits[:, :-1, :]
        labels = input_ids[:, 1:]
        valid = attention_mask[:, :-1].bool() & attention_mask[:, 1:].bool()
        losses = F.cross_entropy(
            logits.reshape(-1, logits.shape[-1]),
            labels.reshape(-1),
            reduction="none",
        ).reshape_as(labels)
        total_loss += float(losses[valid].sum().item())
        total_tokens += int(valid.sum().item())

    if total_tokens == 0:
        raise ValueError("No valid next-token positions found.")
    return float(torch.exp(torch.tensor(total_loss / total_tokens)).item())


@torch.inference_mode()
def duo_generative_perplexity(
    texts: str | Sequence[str],
    model,
    tokenizer,
    *,
    batch_size: int = 8,
    max_length: int,
    device: str | torch.device | None = None,
    show_progress: bool = False,
) -> float:
    """DUO-style generative perplexity for decoded samples.

    DUO re-tokenizes decoded samples with the scorer tokenizer, truncates/pads
    to ``max_length``, chunks by the scorer context, includes the first EOS in
    the loss, and ignores subsequent EOS positions.
    """

    samples = [texts] if isinstance(texts, str) else list(texts)
    if not samples:
        raise ValueError("At least one text sample is required.")
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    if device is None:
        device = next(model.parameters()).device
    device = torch.device(device)
    model = model.to(device)
    model.eval()

    total_loss = 0.0
    total_tokens = 0
    eval_context_size = min(max_length, int(getattr(model.config, "n_positions", max_length)))
    iterator = range(0, len(samples), batch_size)
    if show_progress:
        iterator = tqdm(iterator, desc="gen-ppl")

    for start in iterator:
        encoded = tokenizer(
            samples[start : start + batch_size],
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
            return_attention_mask=True,
        )
        input_ids = encoded["input_ids"].to(device)
        attention_mask = encoded["attention_mask"].to(device)
        if input_ids.shape[1] < 2:
            continue

        for sample_chunk, mask_chunk in zip(
            torch.split(input_ids, eval_context_size, dim=-1),
            torch.split(attention_mask, eval_context_size, dim=-1),
        ):
            if sample_chunk.shape[1] < 2:
                continue
            logits = model(input_ids=sample_chunk, attention_mask=mask_chunk).logits
            losses = F.cross_entropy(
                logits[:, :-1, :].reshape(-1, logits.shape[-1]),
                sample_chunk[:, 1:].reshape(-1),
                reduction="none",
            ).reshape(sample_chunk[:, 1:].shape)
            first_eos = (sample_chunk == tokenizer.eos_token_id).cumsum(dim=-1) == 1
            token_mask = sample_chunk != tokenizer.eos_token_id
            valid = (first_eos[:, 1:] | token_mask[:, 1:]) & mask_chunk[:, 1:].bool()
            total_loss += float(losses[valid].sum().item())
            total_tokens += int(valid.sum().item())

    if total_tokens == 0:
        raise ValueError("No valid next-token positions found.")
    return float(torch.exp(torch.tensor(total_loss / total_tokens)).item())
