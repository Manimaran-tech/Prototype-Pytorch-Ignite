"""
Ignite-based Trainer and Evaluator for the trainite prototype.

Wraps PyTorch-Ignite Engine with production-quality handlers:
- ModelCheckpoint (saves best model by validation loss)
- EarlyStopping (stops if val loss does not improve)
- TensorboardLogger (tracks loss + accuracy curves)
- ProgressBar (terminal output during training)

The trainer uses masked loss — only output tokens (after [SEP]) contribute
to the loss, so the model focuses entirely on learning the reversal.
"""

import os
import logging

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from ignite.engine import Engine, Events
from ignite.metrics import RunningAverage
from ignite.handlers import ModelCheckpoint, EarlyStopping
from ignite.contrib.handlers import TensorboardLogger
from ignite.contrib.handlers.tqdm_logger import ProgressBar

logger = logging.getLogger(__name__)


class Trainer:
    """
    Configurable Ignite-based Trainer with masked loss support.

    The loss_mask ensures gradient flows only from the reversed output
    portion of the sequence, giving the model a focused learning signal.
    """

    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        loss_fn: nn.Module,
        device: torch.device,
        output_dir: str = "./experiments",
        pad_token_id: int = 0,
    ):
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.device = device
        self.output_dir = output_dir
        self.pad_token_id = pad_token_id

        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(os.path.join(output_dir, "checkpoints"), exist_ok=True)

        self._setup_logging()

        # Create Ignite engines
        self.trainer_engine = Engine(self._train_step)
        self.evaluator_engine = Engine(self._eval_step)

        # Running average of train loss on progress bar
        RunningAverage(output_transform=lambda x: x["loss"]).attach(
            self.trainer_engine, "running_loss"
        )

        # Attach progress bar
        pbar = ProgressBar(persist=True)
        pbar.attach(
            self.trainer_engine,
            output_transform=lambda x: {"loss": x["loss"]},
        )

        # Training history for visualization
        self.history = {
            "train_loss": [],
            "val_loss": [],
            "train_token_acc": [],
            "val_token_acc": [],
            "val_seq_acc": [],
        }

    def _setup_logging(self):
        """Configure file + console logging."""
        log_path = os.path.join(self.output_dir, "training.log")
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s [%(levelname)s] %(message)s",
            handlers=[
                logging.FileHandler(log_path, mode="w"),
                logging.StreamHandler(),
            ],
        )

    def _compute_masked_loss(
        self, logits: torch.Tensor, targets: torch.Tensor, loss_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute cross-entropy loss only on masked (output) tokens.

        Args:
            logits: (B, S, V)
            targets: (B, S)
            loss_mask: (B, S) — 1 where loss should be computed
        """
        # Flatten: (B*S, V) and (B*S,)
        B, S, V = logits.shape
        flat_logits = logits.reshape(B * S, V)
        flat_targets = targets.reshape(B * S)
        flat_mask = loss_mask.reshape(B * S)

        # Compute per-element loss (no reduction)
        per_token_loss = torch.nn.functional.cross_entropy(
            flat_logits, flat_targets, reduction="none", ignore_index=self.pad_token_id
        )

        # Apply mask and average
        masked_loss = (per_token_loss * flat_mask).sum() / flat_mask.sum().clamp(min=1)
        return masked_loss

    def _train_step(self, engine: Engine, batch: dict) -> dict:
        """Single training step with masked loss."""
        self.model.train()
        x = batch["x"].to(self.device)
        y = batch["y"].to(self.device)
        loss_mask = batch["loss_mask"].to(self.device)

        self.optimizer.zero_grad()

        logits = self.model(x)  # (B, S, V)
        loss = self._compute_masked_loss(logits, y, loss_mask)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.optimizer.step()

        # Token accuracy on masked positions only
        preds = logits.argmax(dim=-1)
        mask_bool = loss_mask.bool()
        correct = ((preds == y) & mask_bool).sum().item()
        total = mask_bool.sum().item()
        token_acc = correct / total if total > 0 else 0.0

        return {"loss": loss.item(), "token_acc": token_acc}

    @torch.no_grad()
    def _eval_step(self, engine: Engine, batch: dict) -> dict:
        """Single evaluation step with masked loss."""
        self.model.eval()
        x = batch["x"].to(self.device)
        y = batch["y"].to(self.device)
        loss_mask = batch["loss_mask"].to(self.device)

        logits = self.model(x)
        loss = self._compute_masked_loss(logits, y, loss_mask)

        preds = logits.argmax(dim=-1)
        mask_bool = loss_mask.bool()
        correct = ((preds == y) & mask_bool).sum().item()
        total = mask_bool.sum().item()
        token_acc = correct / total if total > 0 else 0.0

        # Sequence accuracy: all masked tokens must be correct
        B = x.size(0)
        seq_correct = 0
        for i in range(B):
            m = mask_bool[i]
            if m.any() and torch.all(preds[i][m] == y[i][m]):
                seq_correct += 1
        seq_acc = seq_correct / B

        return {"loss": loss.item(), "token_acc": token_acc, "seq_acc": seq_acc}

    def _run_validation(self, val_loader: DataLoader) -> dict:
        """Run full validation pass and return averaged metrics."""
        self.model.eval()
        total_loss, total_tok_acc, total_seq_acc = 0.0, 0.0, 0.0
        n_batches = 0

        with torch.no_grad():
            for batch in val_loader:
                result = self._eval_step(None, batch)
                total_loss += result["loss"]
                total_tok_acc += result["token_acc"]
                total_seq_acc += result["seq_acc"]
                n_batches += 1

        return {
            "val_loss": total_loss / n_batches,
            "val_token_acc": total_tok_acc / n_batches,
            "val_seq_acc": total_seq_acc / n_batches,
        }

    def run(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        max_epochs: int = 50,
        log_interval: int = 10,
    ) -> dict:
        """
        Run the full training loop.

        Args:
            train_loader: Training data loader.
            val_loader: Validation data loader.
            max_epochs: Number of epochs.
            log_interval: Logging frequency within epochs.

        Returns:
            Training history dictionary.
        """
        # Best val loss tracker for checkpoint & early stopping
        best_val_loss = float("inf")
        patience_counter = 0
        patience = 10

        @self.trainer_engine.on(Events.EPOCH_COMPLETED)
        def log_and_validate(engine):
            nonlocal best_val_loss, patience_counter

            epoch = engine.state.epoch
            train_loss = engine.state.metrics.get("running_loss", 0.0)
            train_acc = engine.state.output.get("token_acc", 0.0) if engine.state.output else 0.0

            # Full validation pass
            val_metrics = self._run_validation(val_loader)

            # Record history
            self.history["train_loss"].append(train_loss)
            self.history["val_loss"].append(val_metrics["val_loss"])
            self.history["train_token_acc"].append(train_acc)
            self.history["val_token_acc"].append(val_metrics["val_token_acc"])
            self.history["val_seq_acc"].append(val_metrics["val_seq_acc"])

            logger.info(
                f"Epoch {epoch:03d} | "
                f"Train Loss: {train_loss:.4f} | "
                f"Val Loss: {val_metrics['val_loss']:.4f} | "
                f"Val Token Acc: {val_metrics['val_token_acc']:.4f} | "
                f"Val Seq Acc: {val_metrics['val_seq_acc']:.4f}"
            )

            # Checkpoint: save best model
            if val_metrics["val_loss"] < best_val_loss:
                best_val_loss = val_metrics["val_loss"]
                patience_counter = 0
                ckpt_path = os.path.join(
                    self.output_dir, "checkpoints", "best_model.pth"
                )
                torch.save(self.model.state_dict(), ckpt_path)
                logger.info(f"  ↳ New best model saved (val_loss={best_val_loss:.4f})")
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    logger.info(f"  ↳ Early stopping triggered (patience={patience})")
                    self.trainer_engine.terminate()

        # TensorBoard logging
        tb_logger = TensorboardLogger(log_dir=os.path.join(self.output_dir, "tb_logs"))
        tb_logger.attach_output_handler(
            self.trainer_engine,
            event_name=Events.ITERATION_COMPLETED(every=log_interval),
            tag="training",
            output_transform=lambda x: {
                "batch_loss": x["loss"],
                "batch_token_acc": x["token_acc"],
            },
        )

        logger.info(f"Starting training for {max_epochs} epochs...")
        logger.info(f"Output directory: {self.output_dir}")
        logger.info(f"Device: {self.device}")
        logger.info(
            f"Model params: {sum(p.numel() for p in self.model.parameters()):,}"
        )

        # Run!
        self.trainer_engine.run(train_loader, max_epochs=max_epochs)

        tb_logger.close()

        # Save training history
        torch.save(self.history, os.path.join(self.output_dir, "history.pt"))

        # Save final model
        final_path = os.path.join(self.output_dir, "checkpoints", "final_model.pth")
        torch.save(self.model.state_dict(), final_path)

        logger.info("Training complete!")
        if self.history["val_seq_acc"]:
            logger.info(
                f"Best Val Loss: {min(self.history['val_loss']):.4f} | "
                f"Best Val Seq Acc: {max(self.history['val_seq_acc']):.4f}"
            )

        return self.history
