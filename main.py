"""
main.py — Trainite Prototype Entry Point

Usage:
    python main.py --config config.yaml

Loads configuration, builds the dataset, model, optimizer, and launches
the Ignite training loop. After training, runs a quick inference demo.
"""

import argparse
import os
import random
import logging

import yaml
import torch
from torch.utils.data import DataLoader, random_split

from trainite.datasets import StringReverseDataset, IntegerAdditionDataset, collate_fn
from trainite.models import DecoderOnlyTransformer
from trainite.trainers import Trainer

logger = logging.getLogger(__name__)


def load_config(config_path: str) -> dict:
    """Load YAML configuration file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def set_seed(seed: int = 42):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main():
    parser = argparse.ArgumentParser(
        description="Trainite — LM Training Toolkit Prototype"
    )
    parser.add_argument(
        "--config", type=str, default="config.yaml",
        help="Path to YAML configuration file",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for reproducibility",
    )
    args = parser.parse_args()

    # ─── Config & Device ───
    config = load_config(args.config)
    set_seed(args.seed)

    device_str = config["training"].get("device", "cpu")
    if device_str == "cuda" and not torch.cuda.is_available():
        logger.warning("CUDA not available, falling back to CPU.")
        device_str = "cpu"
    device = torch.device(device_str)

    output_dir = config["paths"]["output_dir"]
    os.makedirs(output_dir, exist_ok=True)

    # ─── Dataset ───
    task_type = config["dataset"].get("task_type", "string_reverse")
    if task_type == "string_reverse":
        dataset = StringReverseDataset(
            num_samples=config["dataset"]["size"],
            min_len=config["dataset"]["min_len"],
            max_len=config["dataset"]["max_len"],
            seed=args.seed,
        )
    elif task_type == "integer_addition":
        dataset = IntegerAdditionDataset(
            num_samples=config["dataset"]["size"],
            min_digits=config["dataset"]["min_digits"],
            max_digits=config["dataset"]["max_digits"],
            seed=args.seed,
        )
    else:
        raise ValueError(f"Unknown task type: {task_type}")

    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(
        dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(args.seed),
    )

    batch_size = config["dataset"]["batch_size"]
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        collate_fn=collate_fn, num_workers=0,
        pin_memory=(device_str == "cuda"),
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        collate_fn=collate_fn, num_workers=0,
        pin_memory=(device_str == "cuda"),
    )

    # ─── Model ───
    model = DecoderOnlyTransformer(
        vocab_size=dataset.vocab_size,
        embed_dim=config["model"]["embed_dim"],
        num_heads=config["model"]["num_heads"],
        num_layers=config["model"]["num_layers"],
        dim_feedforward=config["model"]["dim_feedforward"],
        max_seq_len=config["model"]["max_seq_len"],
        pad_token_id=dataset.pad_idx,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Model: {total_params:,} parameters")

    # ─── Optimizer & Loss ───
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config["training"]["lr"],
        weight_decay=config["training"]["weight_decay"],
    )
    # loss_fn not needed by trainer (uses masked loss internally), but we pass it for API consistency
    loss_fn = torch.nn.CrossEntropyLoss(ignore_index=dataset.pad_idx)

    # ─── Trainer ───
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        loss_fn=loss_fn,
        device=device,
        output_dir=output_dir,
        pad_token_id=dataset.pad_idx,
    )

    # ─── Run Training ───
    history = trainer.run(
        train_loader=train_loader,
        val_loader=val_loader,
        max_epochs=config["training"]["epochs"],
        log_interval=config["training"].get("log_interval", 10),
    )

    # ─── Inference Demo ───
    logger.info("")
    logger.info("=" * 65)
    logger.info("  INFERENCE DEMO")
    logger.info("=" * 65)

    model.eval()
    
    if task_type == "string_reverse":
        test_strings = ["hello", "world", "ignite", "pytorch", "gsoc", "abc", "trainite"]
        correct = 0
        for s in test_strings:
            # Encode: [SOS] + chars + [SEP]
            tokens = [dataset.sos_idx] + [dataset.c2i[c] for c in s] + [dataset.sep_idx]
            prompt = torch.tensor([tokens], dtype=torch.long, device=device)

            generated = model.generate(
                prompt, max_new_tokens=len(s) + 5, eos_token_id=dataset.eos_idx
            )
            gen_tokens = generated[0].tolist()

            # Decode output (tokens after [SEP])
            sep_pos = gen_tokens.index(dataset.sep_idx) if dataset.sep_idx in gen_tokens else -1
            if sep_pos >= 0:
                output_tokens = gen_tokens[sep_pos + 1:]
                if dataset.eos_idx in output_tokens:
                    output_tokens = output_tokens[:output_tokens.index(dataset.eos_idx)]
                predicted = "".join(dataset.i2c.get(t, "?") for t in output_tokens)
            else:
                predicted = "<failed>"

            expected = s[::-1]
            status = "PASS" if predicted == expected else "FAIL"
            if predicted == expected:
                correct += 1
            logger.info(f"  [{status}]  '{s}' -> expected '{expected}' | got '{predicted}'")

        logger.info(f"\n  Inference Accuracy: {correct}/{len(test_strings)}")
        
    elif task_type == "integer_addition":
        test_pairs = [("12", "34"), ("99", "1"), ("123", "456"), ("9999", "1"), ("12345", "67890")]
        correct = 0
        for num1, num2 in test_pairs:
            s_input = f"{num1}+{num2}="
            expected = str(int(num1) + int(num2))
            
            tokens = [dataset.sos_idx] + [dataset.c2i[c] for c in s_input] + [dataset.sep_idx]
            prompt = torch.tensor([tokens], dtype=torch.long, device=device)

            generated = model.generate(
                prompt, max_new_tokens=len(expected) + 5, eos_token_id=dataset.eos_idx
            )
            gen_tokens = generated[0].tolist()

            sep_pos = gen_tokens.index(dataset.sep_idx) if dataset.sep_idx in gen_tokens else -1
            if sep_pos >= 0:
                output_tokens = gen_tokens[sep_pos + 1:]
                if dataset.eos_idx in output_tokens:
                    output_tokens = output_tokens[:output_tokens.index(dataset.eos_idx)]
                predicted = "".join(dataset.i2c.get(t, "?") for t in output_tokens)
            else:
                predicted = "<failed>"

            status = "PASS" if predicted == expected else "FAIL"
            if predicted == expected:
                correct += 1
            logger.info(f"  [{status}]  '{s_input}' -> expected '{expected}' | got '{predicted}'")

        logger.info(f"\n  Inference Accuracy: {correct}/{len(test_pairs)}")

    logger.info("=" * 65)


if __name__ == "__main__":
    main()
