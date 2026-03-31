"""
Standalone smoke demo for ta_lbfgs.topology.hf_interceptor.OnlineTopologyHook.

This script loads a Hugging Face model (default: meta-llama/Llama-3.2-1B-Instruct),
runs one prefill pass through OnlineTopologyHook, prints a concise snapshot summary,
and generates a short response from the same prepared chat inputs.
"""

from __future__ import annotations

import argparse
import time
from typing import Any, Dict, List, Optional

import torch

try:
    from transformers import AutoModelForCausalLM, AutoTokenizer
except Exception as exc:  # pragma: no cover
    raise RuntimeError(
        "transformers is required for this demo. Install with: pip install transformers"
    ) from exc

from ta_lbfgs.topology.hf_interceptor import OnlineTopologyHook


class _DummyOptimizer:
    """Minimal optimizer shim that accepts interceptor snapshots."""

    def __init__(self) -> None:
        self.snapshots: List[Dict[str, Any]] = []

    def ingest_topology_snapshot(self, snapshot: Dict[str, Any]) -> None:
        self.snapshots.append(snapshot)


def _summarize_attention_heads(attentions: Optional[Any]) -> List[str]:
    lines: List[str] = []
    if not attentions:
        return ["attentions: None"]

    lines.append(f"layers_with_attention: {len(attentions)}")
    for layer_idx, layer_attn in enumerate(attentions):
        if layer_attn is None:
            lines.append(f"  layer {layer_idx}: None")
            continue

        shape = tuple(int(x) for x in layer_attn.shape)
        # Common HF shape is [batch, heads, query_len, key_len].
        if len(shape) == 4:
            batch, heads, q_len, k_len = shape
            lines.append(
                f"  layer {layer_idx}: shape={shape}, heads={heads}, q_len={q_len}, k_len={k_len}, batch={batch}"
            )
        else:
            lines.append(f"  layer {layer_idx}: shape={shape}")
    return lines


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run interceptor smoke demo on a HF model.")
    parser.add_argument(
        "--model",
        default="meta-llama/Llama-3.2-1B-Instruct",
        help="Model id or local model path (default: meta-llama/Llama-3.2-1B-Instruct).",
    )
    parser.add_argument(
        "--question",
        default="Who are you?",
        help="User message text used for chat-template prefill.",
    )
    parser.add_argument(
        "--local-files-only",
        action="store_true",
        help="Load model/tokenizer only from local HF cache.",
    )
    parser.add_argument(
        "--warmup-steps",
        type=int,
        default=10,
        help="Warmup steps passed to OnlineTopologyHook.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=40,
        help="Max new tokens for generation after interception.",
    )
    return parser



def main() -> None:
    args = _build_parser().parse_args()
    torch.manual_seed(7)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"device: {device}")
    print(f"loading_model: {args.model}")

    try:
        tokenizer = AutoTokenizer.from_pretrained(
            args.model,
            local_files_only=bool(args.local_files_only),
            trust_remote_code=False,
        )
        model = AutoModelForCausalLM.from_pretrained(
            args.model,
            local_files_only=bool(args.local_files_only),
            trust_remote_code=False,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            attn_implementation="eager",
        )
    except Exception as exc:
        msg = str(exc)
        if "401" in msg or "403" in msg or "gated" in msg.lower() or "access" in msg.lower():
            raise RuntimeError(
                "Failed to load model. This Llama repo is likely gated. "
                "Request access on Hugging Face and login with 'huggingface-cli login', "
                "or pass an accessible local path/model id via --model."
            ) from exc
        raise
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token = tokenizer.eos_token
    model = model.to(device)
    model.eval()

    optimizer = _DummyOptimizer()
    hook = OnlineTopologyHook(model=model, optimizer=optimizer, warmup_steps=int(args.warmup_steps))

    try:
        messages = [
            {"role": "user", "content": args.question},
        ]
        inputs = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        ).to(model.device)

        input_ids = inputs["input_ids"]
        attention_mask = inputs.get("attention_mask")

        with torch.no_grad():
            outputs = hook.on_prompt(input_ids=input_ids, attention_mask=attention_mask)

        # Wait briefly for async worker to ingest the snapshot.
        # For this demo, ingestion is immediate in practice, but we guard anyway.
        for _ in range(20):
            if optimizer.snapshots:
                break
            time.sleep(0.01)

        print("interceptor_run_ok: True")
        print(f"logits_shape: {tuple(int(x) for x in outputs.logits.shape)}")
        print(f"snapshots_received: {len(optimizer.snapshots)}")

        if optimizer.snapshots:
            snap = optimizer.snapshots[-1]
            print(f"snapshot_step: {snap.get('step')}")
            print(f"has_attentions: {snap.get('attn_weights') is not None}")
            print(f"has_hidden_states: {snap.get('hidden_states') is not None}")
            print(f"has_kv_key_norms: {snap.get('kv_key_norms') is not None}")

            for line in _summarize_attention_heads(snap.get("attn_weights")):
                print(line)

        with torch.no_grad():
            gen_outputs = model.generate(**inputs, max_new_tokens=int(args.max_new_tokens))

        generated = tokenizer.decode(gen_outputs[0][inputs["input_ids"].shape[-1] :], skip_special_tokens=True)
        print(f"generated_text: {generated}")

    finally:
        hook.shutdown()


if __name__ == "__main__":
    main()
