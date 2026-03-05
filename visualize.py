"""
visualize.py — Generate training curves and inference report.

Usage:
    python visualize.py --output_dir ./experiments

Reads the saved training history and generates publication-quality
matplotlib plots of loss, token accuracy, and sequence accuracy,
plus a table of inference examples.
"""

import argparse
import os

import torch
import matplotlib.pyplot as plt
import matplotlib

matplotlib.use("Agg")  # Non-interactive backend for saving plots


def load_history(output_dir: str) -> dict:
    """Load the training history from a .pt file."""
    path = os.path.join(output_dir, "history.pt")
    if not os.path.exists(path):
        raise FileNotFoundError(f"history.pt not found in {output_dir}")
    return torch.load(path, map_location="cpu", weights_only=True)


def plot_loss(history: dict, save_path: str):
    """Plot training and validation loss curves."""
    epochs = range(1, len(history["train_loss"]) + 1)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(epochs, history["train_loss"], "o-", color="#2196F3", label="Train Loss", linewidth=2, markersize=4)
    ax.plot(epochs, history["val_loss"], "s-", color="#FF5722", label="Val Loss", linewidth=2, markersize=4)

    ax.set_xlabel("Epoch", fontsize=13)
    ax.set_ylabel("Cross-Entropy Loss", fontsize=13)
    ax.set_title("Training & Validation Loss", fontsize=15, fontweight="bold")
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.set_facecolor("#FAFAFA")
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [saved] {save_path}")


def plot_accuracy(history: dict, save_path: str):
    """Plot token and sequence accuracy curves."""
    epochs = range(1, len(history["val_token_acc"]) + 1)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(epochs, history["val_token_acc"], "o-", color="#4CAF50", label="Val Token Accuracy", linewidth=2, markersize=4)
    ax.plot(epochs, history["val_seq_acc"], "D-", color="#9C27B0", label="Val Sequence Accuracy", linewidth=2, markersize=4)

    ax.set_xlabel("Epoch", fontsize=13)
    ax.set_ylabel("Accuracy", fontsize=13)
    ax.set_title("Validation Accuracy (Token & Sequence)", fontsize=15, fontweight="bold")
    ax.legend(fontsize=12)
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3)
    ax.set_facecolor("#FAFAFA")
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [saved] {save_path}")


def plot_combined(history: dict, save_path: str):
    """Create a combined 2-panel figure of loss + accuracy."""
    epochs = range(1, len(history["train_loss"]) + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # Loss panel
    ax1.plot(epochs, history["train_loss"], "o-", color="#2196F3", label="Train Loss", linewidth=2, markersize=3)
    ax1.plot(epochs, history["val_loss"], "s-", color="#FF5722", label="Val Loss", linewidth=2, markersize=3)
    ax1.set_xlabel("Epoch", fontsize=12)
    ax1.set_ylabel("Loss", fontsize=12)
    ax1.set_title("Loss Curves", fontsize=14, fontweight="bold")
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)
    ax1.set_facecolor("#FAFAFA")

    # Accuracy panel
    ax2.plot(epochs, history["val_token_acc"], "o-", color="#4CAF50", label="Token Acc", linewidth=2, markersize=3)
    ax2.plot(epochs, history["val_seq_acc"], "D-", color="#9C27B0", label="Sequence Acc", linewidth=2, markersize=3)
    ax2.set_xlabel("Epoch", fontsize=12)
    ax2.set_ylabel("Accuracy", fontsize=12)
    ax2.set_title("Validation Accuracy", fontsize=14, fontweight="bold")
    ax2.legend(fontsize=11)
    ax2.set_ylim(0, 1.05)
    ax2.grid(True, alpha=0.3)
    ax2.set_facecolor("#FAFAFA")

    fig.suptitle("Trainite — String Reversal Training Report", fontsize=16, fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [saved] {save_path}")


def run_inference_report(output_dir: str, device_str: str = "cpu"):
    """
    Load the best checkpoint and run inference on test strings,
    printing a clean report.
    """
    import yaml
    from trainite.datasets import StringReverseDataset
    from trainite.models import DecoderOnlyTransformer

    # Load config
    config_path = os.path.join(os.path.dirname(output_dir), "config.yaml")
    if not os.path.exists(config_path):
        config_path = "config.yaml"
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    device = torch.device(device_str)

    # Rebuild dataset (just for vocab mapping)
    dataset = StringReverseDataset(
        num_samples=10,
        min_len=config["dataset"]["min_len"],
        max_len=config["dataset"]["max_len"],
    )

    # Build model
    model = DecoderOnlyTransformer(
        vocab_size=dataset.vocab_size,
        embed_dim=config["model"]["embed_dim"],
        num_heads=config["model"]["num_heads"],
        num_layers=config["model"]["num_layers"],
        dim_feedforward=config["model"]["dim_feedforward"],
        max_seq_len=config["model"]["max_seq_len"],
        pad_token_id=dataset.pad_idx,
    ).to(device)

    # Load best checkpoint
    ckpt_dir = os.path.join(output_dir, "checkpoints")
    final_model = os.path.join(ckpt_dir, "final_model.pth")
    if os.path.exists(final_model):
        model.load_state_dict(torch.load(final_model, map_location=device, weights_only=True))
        print(f"  Loaded checkpoint: {final_model}")
    else:
        print(f"  WARNING: No checkpoint found at {final_model}")
        return

    model.eval()

    test_strings = [
        "hello", "world", "ignite", "pytorch", "gsoc",
        "trainite", "abcdef", "reverse", "python", "transformer",
    ]

    print("\n" + "=" * 65)
    print("  INFERENCE REPORT")
    print("=" * 65)
    print(f"  {'Input':<15} {'Expected':<15} {'Predicted':<15} {'Status'}")
    print("  " + "-" * 60)

    correct = 0
    for s in test_strings:
        tokens = [dataset.sos_idx] + [dataset.c2i.get(c, dataset.pad_idx) for c in s] + [dataset.sep_idx]
        prompt = torch.tensor([tokens], dtype=torch.long, device=device)

        generated = model.generate(prompt, max_new_tokens=len(s) + 5, eos_token_id=dataset.eos_idx)
        gen_tokens = generated[0].tolist()

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
        print(f"  {s:<15} {expected:<15} {predicted:<15} {status}")

    total = len(test_strings)
    print("  " + "-" * 60)
    print(f"  Accuracy: {correct}/{total} ({100 * correct / total:.1f}%)")
    print("=" * 65)


def main():
    parser = argparse.ArgumentParser(description="Trainite Visualization")
    parser.add_argument("--output_dir", type=str, default="./experiments", help="Experiment output dir")
    parser.add_argument("--device", type=str, default="cpu", help="Device for inference")
    args = parser.parse_args()

    print("\n[*] Trainite -- Generating Training Visualizations\n")

    # Plots
    history = load_history(args.output_dir)

    plots_dir = os.path.join(args.output_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    plot_loss(history, os.path.join(plots_dir, "loss_curves.png"))
    plot_accuracy(history, os.path.join(plots_dir, "accuracy_curves.png"))
    plot_combined(history, os.path.join(plots_dir, "training_report.png"))

    # Inference
    run_inference_report(args.output_dir, device_str=args.device)


if __name__ == "__main__":
    main()
