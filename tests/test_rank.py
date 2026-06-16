import math
import sys
import types

import torch
import pytest

from ranking_divergence import (
    MirrorSampler,
    PeriodicSampler,
    PhraseBankSampler,
    RestrictedMarginalSampler,
    empirical_entropy,
    per_sample_unigram_entropy,
    rank_histogram,
    rank_wasserstein_from_histograms,
    rep_n,
    unique_ngram_ratios,
)
from ranking_divergence.data import (
    DUO_OWT_CACHE_DIR,
    OWT_HELDOUT_SPLIT,
    OWT_SAMPLER_SOURCE_SPLIT,
    load_openwebtext_texts,
    openwebtext_split_config,
)


class FakeTokenizer:
    pad_token_id = 0
    eos_token = "<eos>"

    def encode(self, text, add_special_tokens=False):
        del add_special_tokens
        return [int(token) for token in text.split()] if text else []

    def batch_decode(self, samples, skip_special_tokens=True):
        del skip_special_tokens
        return [" ".join(map(str, sample)) for sample in samples]

    def __call__(
        self,
        texts,
        return_tensors="pt",
        padding=True,
        truncation=False,
        max_length=None,
        return_attention_mask=True,
    ):
        del return_tensors, padding, return_attention_mask
        tokenized = [[int(token) for token in text.split()] for text in texts]
        if truncation and max_length is not None:
            tokenized = [tokens[:max_length] for tokens in tokenized]
        width = max(len(tokens) for tokens in tokenized)
        input_ids = []
        attention_mask = []
        for tokens in tokenized:
            pad = width - len(tokens)
            input_ids.append(tokens + [self.pad_token_id] * pad)
            attention_mask.append([1] * len(tokens) + [0] * pad)
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
        }


class FakeCausalLM(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.config = type("Config", (), {"vocab_size": 4})()
        self.register_parameter("dummy", torch.nn.Parameter(torch.zeros(())))

    def forward(self, input_ids, attention_mask=None):
        del attention_mask
        batch_size, sequence_length = input_ids.shape
        logits = torch.zeros(batch_size, sequence_length, self.config.vocab_size)

        # Position 0 predicts observed next token 1 at rank 1.
        logits[:, 0, :] = torch.tensor([0.0, 10.0, 3.0, 1.0])

        # Position 1 predicts observed next token 2 at rank 3.
        logits[:, 1, :] = torch.tensor([9.0, 8.0, 7.0, 1.0])

        return type("CausalLMOutput", (), {"logits": logits})()


def test_rank_wasserstein_identical_histograms_is_zero():
    hist = torch.tensor([3.0, 2.0, 1.0])
    assert rank_wasserstein_from_histograms(hist, hist) == 0.0


def test_rank_wasserstein_matches_closed_form():
    ref = torch.tensor([1.0, 0.0, 0.0])
    cmp = torch.tensor([0.0, 1.0, 0.0])
    assert rank_wasserstein_from_histograms(ref, cmp) == math.log(2.0)


def test_rank_histogram_hand_computable_example():
    raw_histogram = rank_histogram(
        ["0 1 2"],
        FakeCausalLM(),
        FakeTokenizer(),
        normalize=False,
    )
    expected_raw = torch.tensor([1.0, 0.0, 1.0, 0.0], dtype=torch.float64)
    assert torch.equal(raw_histogram, expected_raw)

    normalized_histogram = rank_histogram(
        ["0 1 2"],
        FakeCausalLM(),
        FakeTokenizer(),
        normalize=True,
    )
    expected_normalized = torch.tensor([0.5, 0.0, 0.5, 0.0], dtype=torch.float64)
    assert torch.equal(normalized_histogram, expected_normalized)

    reference = torch.tensor([1.0, 0.0, 0.0, 0.0])
    distance = rank_wasserstein_from_histograms(reference, normalized_histogram, normalize=False)
    assert distance == pytest.approx(0.5 * math.log(3.0))


def test_rep_n_repetition_score():
    assert rep_n("", token_ids=[[1, 2, 1, 2]], n=2) == pytest.approx(1.0 / 3.0)


def test_empirical_entropy_uses_nats():
    value = empirical_entropy("", token_ids=[[1, 1, 2, 2]])
    assert value == pytest.approx(math.log(2.0))


def test_per_sample_unigram_entropy_averages_samples():
    value = per_sample_unigram_entropy("", token_ids=[[1, 1, 2, 2], [3, 3, 3, 3]])
    assert value == pytest.approx(math.log(2.0) / 2.0)


def test_unique_ngram_ratios_are_hand_computable():
    ratios = unique_ngram_ratios("", token_ids=[[1, 2, 1], [1, 2, 3]], n=2)
    assert ratios["sample"] == pytest.approx((1.0 + 1.0) / 2.0)
    assert ratios["corpus"] == pytest.approx(3.0 / 4.0)


def test_mirror_sampler_returns_requested_odd_length():
    base = RestrictedMarginalSampler(
        torch.tensor([1, 2]),
        torch.tensor([0.5, 0.5], dtype=torch.float64),
        FakeTokenizer(),
    )
    sample = MirrorSampler(base).sample_token_ids(num_samples=1, length=5, seed=0)[0]
    assert len(sample) == 5
    assert sample == list(reversed(sample))


def test_samplers_are_deterministic_on_tiny_text():
    texts = ["1 2 2 3 3 3", "4 4 4 4"]
    tokenizer = FakeTokenizer()

    topk = RestrictedMarginalSampler.from_texts(texts, tokenizer, k=2)
    assert topk.sample_token_ids(num_samples=2, length=4, seed=7) == topk.sample_token_ids(
        num_samples=2,
        length=4,
        seed=7,
    )

    periodic = PeriodicSampler.from_texts(texts, tokenizer, k=3)
    assert periodic.sample_token_ids(num_samples=1, length=5) == [[4, 3, 2, 4, 3]]

    phrase = PhraseBankSampler.from_texts(texts, tokenizer, n=2, m=2)
    sample = phrase.sample_token_ids(num_samples=1, length=5, seed=3)[0]
    assert len(sample) == 5


def test_openwebtext_split_config_defaults():
    config = openwebtext_split_config()
    assert config["sampler_source_split"] == OWT_SAMPLER_SOURCE_SPLIT
    assert config["heldout_split"] == OWT_HELDOUT_SPLIT
    assert config["cache_dir"] == DUO_OWT_CACHE_DIR


def test_load_openwebtext_texts_uses_requested_split_and_cache(monkeypatch):
    calls = []

    def load_dataset(*args, **kwargs):
        calls.append((args, kwargs))
        return [{"text": "first"}, {"text": "second"}]

    monkeypatch.setitem(sys.modules, "datasets", types.SimpleNamespace(load_dataset=load_dataset))
    texts = load_openwebtext_texts(split="train[-100000:]", cache_dir="/tmp/owt", limit=1)
    assert texts == ["first"]
    assert calls == [
        (
            ("openwebtext",),
            {
                "split": "train[-100000:]",
                "cache_dir": "/tmp/owt",
                "streaming": False,
                "trust_remote_code": True,
            },
        )
    ]
