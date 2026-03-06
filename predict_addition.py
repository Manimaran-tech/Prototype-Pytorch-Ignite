"""
predict_addition.py -- Interactive inference script for integer addition.

Usage:
    python predict_addition.py
    python predict_addition.py --checkpoint ./experiments_addition/checkpoints/best_model.pth

Enter an addition expression like '123+456' and the model will predict the sum.
Type 'quit' or 'exit' to stop.
"""

import argparse
import os
import re
import sys

import yaml
import torch

# Resolve script directory so imports and default paths work from anywhere
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

from trainite.datasets import IntegerAdditionDataset
from trainite.models import DecoderOnlyTransformer


def main():
    default_config = os.path.join(SCRIPT_DIR, "config.yaml")
    default_ckpt = os.path.join(SCRIPT_DIR, "experiments_addition", "checkpoints", "best_model.pth")

    parser = argparse.ArgumentParser(description="Trainite - Interactive Integer Addition")
    parser.add_argument(
        "--checkpoint", type=str, default=default_ckpt,
        help="Path to model checkpoint (.pth)",
    )
    parser.add_argument(
        "--config", type=str, default=default_config,
        help="Path to YAML config",
    )
    parser.add_argument(
        "--device", type=str, default="cuda",
        help="Device (cuda or cpu)",
    )
    args = parser.parse_args()

    # Load config
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    device_str = args.device
    if device_str == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, using CPU.")
        device_str = "cpu"
    device = torch.device(device_str)

    # Build dataset (just for vocab mappings)
    dataset = IntegerAdditionDataset(num_samples=1, min_digits=1, max_digits=3)

    # Build model and load weights
    model = DecoderOnlyTransformer(
        vocab_size=dataset.vocab_size,
        embed_dim=config["model"]["embed_dim"],
        num_heads=config["model"]["num_heads"],
        num_layers=config["model"]["num_layers"],
        dim_feedforward=config["model"]["dim_feedforward"],
        max_seq_len=config["model"]["max_seq_len"],
        pad_token_id=dataset.pad_idx,
    ).to(device)

    model.load_state_dict(
        torch.load(args.checkpoint, map_location=device, weights_only=True)
    )
    model.eval()
    print(f"\nModel loaded from: {args.checkpoint}")
    print(f"Device: {device}\n")
    print("=" * 50)
    print("  TRAINITE - Interactive Integer Addition")
    print("  Enter an addition like '123+456' and press Enter.")
    print("  Type 'quit' or 'exit' to stop.")
    print("=" * 50)

    while True:
        try:
            user_input = input("\n> Enter addition (e.g. 123+456): ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if user_input.lower() in ("quit", "exit", "q"):
            print("Goodbye!")
            break

        if not user_input:
            print("  Please enter a non-empty expression.")
            continue

        # Validate format: number+number
        match = re.match(r"^(\d+)\+(\d+)$", user_input)
        if not match:
            print("  Error: please enter in the format 'number+number' (e.g. 123+456)")
            continue

        num1, num2 = match.group(1), match.group(2)
        expected = str(int(num1) + int(num2))

        # Build input string: "123+456="
        s_input = f"{num1}+{num2}="

        # Validate all characters are in vocab
        if not all(c in dataset.c2i for c in s_input):
            print("  Error: input contains unsupported characters.")
            continue

        # Encode: [SOS] + input chars + [SEP]
        tokens = [dataset.sos_idx] + [dataset.c2i[c] for c in s_input] + [dataset.sep_idx]
        prompt = torch.tensor([tokens], dtype=torch.long, device=device)

        # Generate
        generated = model.generate(
            prompt,
            max_new_tokens=len(expected) + 5,
            eos_token_id=dataset.eos_idx,
        )
        gen_tokens = generated[0].tolist()

        # Decode: extract tokens after [SEP] and before [EOS]
        sep_pos = gen_tokens.index(dataset.sep_idx) if dataset.sep_idx in gen_tokens else -1
        if sep_pos >= 0:
            output_tokens = gen_tokens[sep_pos + 1:]
            if dataset.eos_idx in output_tokens:
                output_tokens = output_tokens[:output_tokens.index(dataset.eos_idx)]
            predicted = "".join(dataset.i2c.get(t, "?") for t in output_tokens)
        else:
            predicted = "<generation failed>"

        match_str = "CORRECT ✅" if predicted == expected else "WRONG ❌"

        print(f"  Input:    {num1} + {num2}")
        print(f"  Expected: {expected}")
        print(f"  Model:    {predicted}  [{match_str}]")


if __name__ == "__main__":
    main()
