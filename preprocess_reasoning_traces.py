import argparse
import os

from transformers import AutoTokenizer

from ta_lbfgs.training.data_preprocessing import (
    build_reasoning_trace_cache,
    get_default_dataset_path,
)


def main():
    parser = argparse.ArgumentParser(
        description="Offline preprocessing for reasoning traces: clean, structure, tokenize, and pack."
    )
    parser.add_argument("--source", type=str, default=get_default_dataset_path())
    parser.add_argument("--tokenizer", type=str, default="Qwen/Qwen2.5-0.5B")
    parser.add_argument("--output", type=str, default=os.path.join("outputs", "reasoning_traces_cache.pt"))
    parser.add_argument("--seq-length", type=int, default=2048)
    parser.add_argument("--max-samples", type=int, default=5000)
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, local_files_only=False)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    payload = build_reasoning_trace_cache(
        source_path=args.source,
        tokenizer=tokenizer,
        output_path=args.output,
        seq_length=args.seq_length,
        max_samples=args.max_samples,
    )

    meta = payload["metadata"]
    print("[SUCCESS] Reasoning trace cache built.")
    print(f"  Source: {meta['source_path']}")
    print(f"  Examples kept: {meta['num_examples']}")
    print(f"  Packed sequences: {meta['num_packed_sequences']}")
    print(f"  Sequence length: {meta['seq_length']}")
    print(f"  Output: {args.output}")


if __name__ == "__main__":
    main()