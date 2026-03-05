"""
predict.py -- Interactive inference script for the trainite prototype.

Usage:
    python predict.py
    python predict.py --checkpoint ./experiments/checkpoints/final_model.pth

Enter any lowercase string and the model will predict its reversal.
Type 'quit' or 'exit' to stop.
"""

import argparse
import os
import sys

import yaml
import torch

# Resolve script directory so imports and default paths work from anywhere
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

from trainite.datasets import StringReverseDataset
from trainite.models import DecoderOnlyTransformer


def main():
    default_config = os.path.join(SCRIPT_DIR, "config.yaml")
    default_ckpt = os.path.join(SCRIPT_DIR, "experiments", "checkpoints", "best_model.pth")

    parser = argparse.ArgumentParser(description="Trainite - Interactive Inference")
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
    dataset = StringReverseDataset(num_samples=1, min_len=3, max_len=5)

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
    print("  TRAINITE - Interactive String Reversal")
    print("  Type a lowercase string and press Enter.")
    print("  Type 'quit' or 'exit' to stop.")
    print("=" * 50)

    while True:
        try:
            user_input = input("\n> Enter string: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if user_input.lower() in ("quit", "exit", "q"):
            print("Goodbye!")
            break

        if not user_input:
            print("  Please enter a non-empty string.")
            continue

        # Validate: only lowercase letters
        if not all(c in dataset.c2i for c in user_input):
            print("  Error: only lowercase a-z characters are supported.")
            continue

        # Encode: [SOS] + chars + [SEP]
        tokens = [dataset.sos_idx] + [dataset.c2i[c] for c in user_input] + [dataset.sep_idx]
        prompt = torch.tensor([tokens], dtype=torch.long, device=device)

        # Generate
        generated = model.generate(
            prompt,
            max_new_tokens=len(user_input) + 5,
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

        expected = user_input[::-1]
        match = "CORRECT" if predicted == expected else "WRONG"

        print(f"  Input:    {user_input}")
        print(f"  Expected: {expected}")
        print(f"  Model:    {predicted}  [{match}]")


if __name__ == "__main__":
    main()
