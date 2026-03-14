"""Data preprocessing utilities for reasoning-trace training and caching."""

import hashlib
import json
import os
import random
import re
import unicodedata
from dataclasses import dataclass, field
from html import escape
from typing import Any, Dict, List, Optional, Tuple

import torch
from torch.utils.data import Dataset, DataLoader


@dataclass
class ReasoningTrace:
    """Single reasoning trace sample."""
    prompt: str
    response: str
    thinking: str = ""  # <think> block if present
    formatted_text: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)


def _normalize_text(text: str) -> str:
    text = unicodedata.normalize("NFKC", text or "")
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"\u0000", "", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _extract_thinking_and_response(content: str) -> Tuple[str, str]:
    content = _normalize_text(content)
    if "<think>" in content and "</think>" in content:
        think_match = re.search(r"<think>(.*?)</think>", content, flags=re.DOTALL)
        if think_match:
            thinking = _normalize_text(think_match.group(1))
            response = _normalize_text(re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL))
            return thinking, response
    return "", content


def format_reasoning_trace_xml(trace: ReasoningTrace) -> str:
    """Format prompt, reasoning, and answer with explicit XML-style tags."""
    prompt = escape(_normalize_text(trace.prompt))
    reasoning = escape(_normalize_text(trace.thinking))
    answer = escape(_normalize_text(trace.response))

    parts = [
        "<example>",
        f"<prompt>{prompt}</prompt>",
    ]
    if reasoning:
        parts.append(f"<reasoning>{reasoning}</reasoning>")
    parts.append(f"<answer>{answer}</answer>")
    parts.append("</example>")
    return "\n".join(parts)


def is_noisy_trace(
    trace: ReasoningTrace,
    min_prompt_chars: int = 12,
    min_answer_chars: int = 12,
    max_repeat_ratio: float = 0.35,
    max_non_ascii_ratio: float = 0.30,
) -> bool:
    """Heuristic filter for low-quality or unstable reasoning examples."""
    prompt = _normalize_text(trace.prompt)
    response = _normalize_text(trace.response)
    reasoning = _normalize_text(trace.thinking)
    joined = "\n".join(part for part in [prompt, reasoning, response] if part)

    if len(prompt) < min_prompt_chars or len(response) < min_answer_chars:
        return True
    if not joined:
        return True

    lines = [line.strip() for line in joined.splitlines() if line.strip()]
    if lines:
        unique_lines = len(set(lines))
        repeat_ratio = 1.0 - (unique_lines / max(len(lines), 1))
        if repeat_ratio > max_repeat_ratio:
            return True

    non_ascii = sum(1 for ch in joined if ord(ch) > 127)
    if non_ascii / max(len(joined), 1) > max_non_ascii_ratio:
        return True

    bad_patterns = [
        r"lorem ipsum",
        r"http[s]?://",
        r"\b(as an ai|language model)\b",
        r"\bundefined\b",
        r"\bnull\b",
    ]
    lowered = joined.lower()
    if any(re.search(pattern, lowered) for pattern in bad_patterns):
        return True

    token_like = re.findall(r"\S+", joined)
    if token_like:
        duplicate_tokens = 1.0 - (len(set(token_like)) / max(len(token_like), 1))
        if duplicate_tokens > 0.80:
            return True

    return False


def _trace_fingerprint(trace: ReasoningTrace) -> str:
    payload = "\n".join([
        _normalize_text(trace.prompt),
        _normalize_text(trace.thinking),
        _normalize_text(trace.response),
    ])
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


class ReasoningTraceDataset(Dataset):
    """
    PyTorch Dataset for reasoning trace JSONL files.

    Supports the 'messages' format:
        {"messages": [{"role": "system", ...}, {"role": "user", ...}, {"role": "assistant", ...}]}

    The assistant response may contain <think>...</think> blocks
    which are extracted separately for analysis.
    """

    def __init__(
        self,
        path: str,
        tokenizer=None,
        max_length: int = 512,
        max_samples: int = 50,
        split_thinking: bool = True,
        format_xml: bool = True,
        filter_noise: bool = True,
        deduplicate: bool = True,
    ):
        self.path = path
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.split_thinking = split_thinking
        self.format_xml = format_xml
        self.filter_noise = filter_noise
        self.deduplicate = deduplicate
        self.traces: List[ReasoningTrace] = []

        self._load(max_samples)

    def _load(self, max_samples: int):
        """Load and parse JSONL file."""
        if not os.path.exists(self.path):
            print(f"[WARNING] Dataset not found at {self.path}")
            return

        print(f"[INFO] Loading reasoning traces from: {self.path}")
        seen = set()
        with open(self.path, 'r', encoding='utf-8') as f:
            for idx, line in enumerate(f):
                if idx >= max_samples:
                    break
                if not line.strip():
                    continue

                data = json.loads(line)
                trace = self._parse_sample(data, idx)
                if trace is None:
                    continue
                if self.filter_noise and is_noisy_trace(trace):
                    continue
                if self.deduplicate:
                    fingerprint = _trace_fingerprint(trace)
                    if fingerprint in seen:
                        continue
                    seen.add(fingerprint)
                if self.format_xml:
                    trace.formatted_text = format_reasoning_trace_xml(trace)
                else:
                    trace.formatted_text = (
                        f"Prompt: {_normalize_text(trace.prompt)}\n"
                        f"Response: {_normalize_text(trace.response)}"
                    )
                if trace:
                    self.traces.append(trace)

        print(f"[INFO] Loaded {len(self.traces)} reasoning traces.")

    def _parse_sample(self, data: Dict, idx: int) -> Optional[ReasoningTrace]:
        """Parse a single JSONL row into a ReasoningTrace."""
        messages = data.get('messages', [])
        if len(messages) < 2:
            return None

        # Find user and assistant messages
        prompt = ""
        response = ""
        thinking = ""

        for msg in messages:
            role = msg.get('role', '')
            content = _normalize_text(msg.get('content', ''))
            if role == 'user':
                prompt = content
            elif role == 'assistant':
                response = content
                if self.split_thinking:
                    thinking, response = _extract_thinking_and_response(content)

        if not prompt:
            return None

        return ReasoningTrace(
            prompt=_normalize_text(prompt),
            response=_normalize_text(response),
            thinking=thinking,
            metadata={"index": idx}
        )

    def __len__(self) -> int:
        return len(self.traces)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        trace = self.traces[idx]
        text = trace.formatted_text or format_reasoning_trace_xml(trace)

        if self.tokenizer:
            encoded = self.tokenizer(
                text,
                return_tensors="pt",
                padding="max_length",
                truncation=True,
                max_length=self.max_length
            )
            return {
                "input_ids": encoded["input_ids"].squeeze(0),
                "attention_mask": encoded["attention_mask"].squeeze(0),
                "labels": encoded["input_ids"].squeeze(0),
                "prompt": trace.prompt,
                "has_thinking": bool(trace.thinking),
                "text": text,
            }
        return {
            "text": text,
            "prompt": trace.prompt,
            "response": trace.response,
            "thinking": trace.thinking,
        }

    def get_batch_texts(self, batch_size: int = 3) -> List[str]:
        """Get a batch of formatted text strings for direct model input."""
        indices = random.sample(range(len(self.traces)), min(batch_size, len(self.traces)))
        batch = []
        for i in indices:
            t = self.traces[i]
            batch.append(t.formatted_text or format_reasoning_trace_xml(t))
        return batch

    def get_prompts_only(self, batch_size: int = 3) -> List[str]:
        """Get just the prompts for generation-based evaluation."""
        indices = random.sample(range(len(self.traces)), min(batch_size, len(self.traces)))
        return [self.traces[i].prompt for i in indices]

    def get_thinking_traces(self) -> List[str]:
        """Get all extracted thinking traces for analysis."""
        return [t.thinking for t in self.traces if t.thinking]


def get_default_dataset_path() -> str:
    """Resolve the default Claude Opus reasoning dataset path from HF cache."""
    cache_base = os.path.expanduser("~/.cache/huggingface/hub")
    ds_dir = "datasets--TeichAI--claude-4.5-opus-high-reasoning-250x"
    snapshot = "742c86f88b66bf53cb5961a25e4360f5582f4a6e"
    return os.path.join(cache_base, ds_dir, "snapshots", snapshot, "claude-opus-4.5-250x.jsonl")


def get_default_cache_path(output_dir: str = "outputs", seq_length: int = 2048, model_name: str = "qwen") -> str:
    """Default offline cache path keyed by packed sequence length and model."""
    safe_name = re.sub(r"[^a-zA-Z0-9_\-]", "_", model_name.split("/")[-1].lower())
    return os.path.join(output_dir, f"reasoning_traces_cache_{safe_name}_{seq_length}.pt")


def create_dataloader(
    dataset: ReasoningTraceDataset,
    batch_size: int = 2,
    shuffle: bool = True,
) -> DataLoader:
    """Create a PyTorch DataLoader from a ReasoningTraceDataset."""
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,  # Windows-safe
        drop_last=True,
    )


def preprocess_for_causal_lm(
    texts: List[str],
    tokenizer,
    max_length: int = 512,
    device: str = "cpu",
) -> Dict[str, torch.Tensor]:
    """
    Tokenize a batch of texts for causal LM training.

    Returns dict with input_ids, attention_mask, and labels
    ready for model(**output).
    """
    encoded = tokenizer(
        texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_length,
    )

    return {
        "input_ids": encoded["input_ids"].to(device),
        "attention_mask": encoded["attention_mask"].to(device),
        "labels": encoded["input_ids"].to(device),  # Causal LM: labels = input_ids
    }


def pack_tokenized_sequences(
    token_sequences: List[List[int]],
    seq_length: int,
    pad_token_id: int,
) -> Dict[str, torch.Tensor]:
    """Pack variable-length token sequences into dense fixed-length blocks."""
    if seq_length <= 0:
        raise ValueError("seq_length must be positive")

    packed_blocks: List[List[int]] = []
    current: List[int] = []

    for seq in token_sequences:
        if not seq:
            continue
        remaining = list(seq)
        while remaining:
            space_left = seq_length - len(current)
            current.extend(remaining[:space_left])
            remaining = remaining[space_left:]
            if len(current) == seq_length:
                packed_blocks.append(current)
                current = []

    if current:
        current = current + [pad_token_id] * (seq_length - len(current))
        packed_blocks.append(current)

    if not packed_blocks:
        raise ValueError("No token sequences available for packing after preprocessing.")

    input_ids = torch.tensor(packed_blocks, dtype=torch.long)
    attention_mask = (input_ids != pad_token_id).long()
    labels = input_ids.clone()
    labels[attention_mask == 0] = -100

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
    }


def build_reasoning_trace_cache(
    source_path: str,
    tokenizer,
    output_path: str,
    seq_length: int = 2048,
    max_samples: int = 5000,
    split_thinking: bool = True,
    format_xml: bool = True,
    filter_noise: bool = True,
    deduplicate: bool = True,
) -> Dict[str, Any]:
    """Offline preprocessing pipeline: clean, filter, tokenize, pack, and save."""
    dataset = ReasoningTraceDataset(
        source_path,
        tokenizer=None,
        max_length=seq_length,
        max_samples=max_samples,
        split_thinking=split_thinking,
        format_xml=format_xml,
        filter_noise=filter_noise,
        deduplicate=deduplicate,
    )

    texts = [trace.formatted_text for trace in dataset.traces if trace.formatted_text]
    eos_id = tokenizer.eos_token_id
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else eos_id
    if eos_id is None:
        raise ValueError("Tokenizer must define eos_token_id for packed offline caching.")
    if pad_id is None:
        raise ValueError("Tokenizer must define pad_token_id or eos_token_id.")

    token_sequences: List[List[int]] = []
    for text in texts:
        token_ids = tokenizer.encode(text, add_special_tokens=False)
        if token_ids:
            token_sequences.append(token_ids + [eos_id])

    packed = pack_tokenized_sequences(token_sequences, seq_length=seq_length, pad_token_id=pad_id)
    payload = {
        "input_ids": packed["input_ids"],
        "attention_mask": packed["attention_mask"],
        "labels": packed["labels"],
        "metadata": {
            "source_path": source_path,
            "num_examples": len(texts),
            "num_packed_sequences": int(packed["input_ids"].shape[0]),
            "seq_length": seq_length,
            "format_xml": format_xml,
            "filter_noise": filter_noise,
            "deduplicate": deduplicate,
        },
    }

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    torch.save(payload, output_path)
    return payload


class PackedReasoningTraceDataset(Dataset):
    """Dataset backed by an offline packed tensor cache."""

    def __init__(self, cache_path: str):
        self.cache_path = cache_path
        payload = torch.load(cache_path, map_location="cpu")
        self.input_ids = payload["input_ids"]
        self.attention_mask = payload["attention_mask"]
        self.labels = payload["labels"]
        self.metadata = payload.get("metadata", {})

    def __len__(self) -> int:
        return int(self.input_ids.shape[0])

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return {
            "input_ids": self.input_ids[idx],
            "attention_mask": self.attention_mask[idx],
            "labels": self.labels[idx],
        }

    def verify_vocab_size(self, tokenizer_vocab_size: int):
        """Ensure cache token IDs fit within current tokenizer vocab."""
        max_id = self.input_ids.max().item()
        if max_id >= tokenizer_vocab_size:
            raise ValueError(
                f"Cache vocab mismatch: max_id {max_id} >= current vocab {tokenizer_vocab_size}. "
                "Delete the cache file and rebuild."
            )


def load_or_build_reasoning_trace_cache(
    source_path: str,
    tokenizer,
    cache_path: str,
    seq_length: int = 2048,
    max_samples: int = 5000,
) -> PackedReasoningTraceDataset:
    """Load an existing packed dataset cache or build it once offline."""
    if not os.path.exists(cache_path):
        build_reasoning_trace_cache(
            source_path=source_path,
            tokenizer=tokenizer,
            output_path=cache_path,
            seq_length=seq_length,
            max_samples=max_samples,
        )
    ds = PackedReasoningTraceDataset(cache_path)
    # Automatic safety check
    try:
        ds.verify_vocab_size(len(tokenizer))
    except (ValueError, AttributeError):
        print(f"[WARNING] Cache at {cache_path} is incompatible with current tokenizer. Rebuilding...")
        os.remove(cache_path)
        return load_or_build_reasoning_trace_cache(
            source_path, tokenizer, cache_path, seq_length, max_samples
        )
    return ds


def sample_packed_batch(
    dataset: PackedReasoningTraceDataset,
    batch_size: int,
    device: str = "cpu",
) -> Dict[str, torch.Tensor]:
    """Randomly sample a dense packed batch from an offline cache dataset."""
    if len(dataset) == 0:
        raise ValueError("PackedReasoningTraceDataset is empty.")

    k = min(max(1, batch_size), len(dataset))
    indices = random.sample(range(len(dataset)), k)
    batch = [dataset[idx] for idx in indices]

    return {
        "input_ids": torch.stack([item["input_ids"] for item in batch]).to(device),
        "attention_mask": torch.stack([item["attention_mask"] for item in batch]).to(device),
        "labels": torch.stack([item["labels"] for item in batch]).to(device),
    }
