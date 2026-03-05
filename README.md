# Trainite Prototype — LM Training Toolkit

A **config-driven**, modular training toolkit prototype built with **PyTorch** and **PyTorch-Ignite**, demonstrating a decoder-only Transformer trained autoregressively on a string reversal task.

> **GSoC 2026 — PyTorch-Ignite**
> This prototype demonstrates the architecture and capabilities planned for the `trainite` Language Model Training Toolkit.

---

## Project Structure

```
trainite_prototype/
├── config.yaml                          # All hyperparameters & paths
├── main.py                              # CLI entrypoint
├── visualize.py                         # Training curves & inference report
├── requirements.txt                     # Dependencies
├── README.md
└── trainite/
    ├── __init__.py
    ├── datasets/
    │   ├── __init__.py
    │   └── string_reverse.py            # In-memory deterministic dataset
    ├── models/
    │   ├── __init__.py
    │   └── transformer.py               # GPT-style decoder-only Transformer
    └── trainers/
        ├── __init__.py
        └── ignite_trainer.py            # Ignite Engine + handlers
```

---

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Train

```bash
python main.py --config config.yaml
```

Training outputs (logs, checkpoints, TensorBoard data) are saved to `./experiments/`.

### 3. Visualize Results

```bash
python visualize.py --output_dir ./experiments
```

Generates:
- `experiments/plots/loss_curves.png` — Training & validation loss
- `experiments/plots/accuracy_curves.png` — Token & sequence accuracy
- `experiments/plots/training_report.png` — Combined 2-panel figure
- A formatted inference table in the terminal

### 4. TensorBoard

```bash
tensorboard --logdir ./experiments/tb_logs
```

---

## Architecture

### Model: Decoder-Only Transformer

A GPT-style language model using `nn.TransformerEncoder` with causal masking (architecturally identical to a decoder without cross-attention):

| Component           | Detail                                     |
|---------------------|--------------------------------------------|
| Embedding           | `nn.Embedding` with padding support        |
| Positional Encoding | Standard sinusoidal (no learned PE)        |
| Transformer Layers  | `nn.TransformerEncoderLayer` (pre-norm)    |
| Output              | Linear projection → vocab logits           |
| Generation          | Autoregressive with greedy decoding        |

### Task: Conditional String Reversal

The model learns to reverse strings by conditioning on the input:

```
Input:  [SOS] a b c d e [SEP]
Target: a b c d e [SEP] e d c b a [EOS]
```

- **Training**: Standard next-token prediction with cross-entropy loss (padding tokens ignored).
- **Inference**: Given `[SOS] + input + [SEP]`, the model autoregressively generates the reversed string until `[EOS]`.

### Training Loop: PyTorch-Ignite

The `Trainer` class wraps a PyTorch-Ignite `Engine` with production-quality handlers:

- **`ModelCheckpoint`** — Saves best model by validation loss
- **`EarlyStopping`** — Halts training if val loss plateaus (patience=10)
- **`TensorboardLogger`** — Logs per-iteration loss & accuracy
- **`ProgressBar`** — Real-time tqdm progress in the terminal
- **Training history** — Saved as `history.pt` for offline visualization

### Metrics Tracked

| Metric             | Description                                     |
|--------------------|-------------------------------------------------|
| Train Loss         | Cross-entropy loss per epoch                     |
| Val Loss           | Cross-entropy loss on held-out validation set    |
| Token Accuracy     | % of non-padding tokens correctly predicted      |
| Sequence Accuracy  | % of sequences where _all_ tokens are correct   |

---

## Configuration

All hyperparameters are controlled via `config.yaml`:

```yaml
model:
  vocab_size: 30
  embed_dim: 128
  num_heads: 4
  num_layers: 4
  dim_feedforward: 512
  max_seq_len: 64

dataset:
  size: 20000
  min_len: 5
  max_len: 25
  batch_size: 256

training:
  epochs: 50
  lr: 0.001
  weight_decay: 0.01
  log_interval: 10
  device: "cuda"

paths:
  output_dir: "./experiments"
```

---

## Design Decisions

1. **`TransformerEncoder` over `TransformerDecoder`**: A decoder-only LM is just self-attention + FFN with causal masking. Using `nn.TransformerEncoder` with `is_causal=True` avoids the need for dummy encoder memory, making the code cleaner and more performant.

2. **Pre-Norm (`norm_first=True`)**: More stable training for small models compared to post-norm.

3. **In-Memory Dataset**: All data is generated deterministically at startup and stored in RAM. No file I/O during training.

4. **Gradient Clipping**: `clip_grad_norm_(max_norm=1.0)` prevents gradient explosions during early training with small transformers.

---

## Author

**Manimaran** — GSoC 2026 Applicant, PyTorch-Ignite

---

## License

MIT
