"""
String Reversal Dataset — In-memory deterministic dataset.

Generates random strings and their reversed counterparts, formatted for
conditional autoregressive generation:

    Full sequence: [SOS] a b c d e [SEP] e d c b a [EOS]
    Input (x):     [SOS] a b c d e [SEP] e d c b a
    Target (y):    a b c d e [SEP] e d c b a [EOS]
    Loss Mask:     0 0 0 0 0 0 1   1 1 1 1 1 1

Loss is computed ONLY on the reversed portion (after [SEP]) so the model
learns the reversal mapping, not just copying the input.
"""

import random
import string

import torch
from torch.utils.data import Dataset


class StringReverseDataset(Dataset):
    """Deterministic in-memory string reversal dataset."""

    # Special tokens
    PAD_TOKEN = "[PAD]"
    SOS_TOKEN = "[SOS]"
    EOS_TOKEN = "[EOS]"
    SEP_TOKEN = "[SEP]"

    def __init__(self, num_samples: int, min_len: int, max_len: int, seed: int = 42):
        self.num_samples = num_samples
        self.min_len = min_len
        self.max_len = max_len

        # Build vocabulary
        self.chars = list(string.ascii_lowercase)

        self.pad_idx = 0
        self.sos_idx = 1
        self.eos_idx = 2
        self.sep_idx = 3

        self.c2i = {c: i + 4 for i, c in enumerate(self.chars)}
        self.i2c = {i + 4: c for i, c in enumerate(self.chars)}
        self.i2c[self.pad_idx] = self.PAD_TOKEN
        self.i2c[self.sos_idx] = self.SOS_TOKEN
        self.i2c[self.eos_idx] = self.EOS_TOKEN
        self.i2c[self.sep_idx] = self.SEP_TOKEN

        self.vocab_size = len(self.chars) + 4  # 26 letters + 4 special

        # Generate data deterministically
        rng = random.Random(seed)
        self.data = []
        for _ in range(num_samples):
            length = rng.randint(min_len, max_len)
            s = "".join(rng.choices(self.chars, k=length))
            self.data.append(s)

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> dict:
        """
        Returns:
            dict with keys:
                - "tokens": Full token sequence [SOS] src [SEP] tgt [EOS]
                - "loss_mask_start": Index where loss computation should begin
                  (the token right after [SEP])
        """
        s = self.data[idx]
        rev_s = s[::-1]

        src_tokens = [self.c2i[c] for c in s]
        tgt_tokens = [self.c2i[c] for c in rev_s]

        # Full sequence: [SOS] + src + [SEP] + tgt + [EOS]
        full_seq = [self.sos_idx] + src_tokens + [self.sep_idx] + tgt_tokens + [self.eos_idx]

        # The loss mask starts right after [SEP] position in the target
        # In the shifted target (y = full_seq[1:]), the [SEP] token is at index len(src_tokens)
        # So loss starts at index len(src_tokens) + 1 in y
        # But it's easier to track the position of SEP in full_seq: 1 + len(src_tokens)
        # In y (target), SEP is at position len(src_tokens), so loss starts at len(src_tokens)
        loss_mask_start = len(src_tokens)  # Position in target where loss begins (at SEP)

        return {
            "tokens": torch.tensor(full_seq, dtype=torch.long),
            "loss_mask_start": loss_mask_start,
        }

    def decode(self, token_ids: list[int]) -> str:
        """Decode a list of token IDs back to a string."""
        result = []
        for t in token_ids:
            if t == self.eos_idx:
                break
            if t in (self.pad_idx, self.sos_idx, self.sep_idx):
                continue
            result.append(self.i2c.get(t, "?"))
        return "".join(result)


def collate_fn(batch: list[dict]) -> dict:
    """
    Collate function that pads sequences and builds a loss mask.

    Returns:
        dict with:
            - "x": Input tokens (B, S) — all tokens except last
            - "y": Target tokens (B, S) — all tokens except first
            - "loss_mask": Binary mask (B, S) — 1 where loss should be computed
    """
    tokens_list = [item["tokens"] for item in batch]
    loss_starts = [item["loss_mask_start"] for item in batch]

    max_len = max(t.size(0) for t in tokens_list)

    padded_tokens = []
    loss_masks = []

    for tokens, ls in zip(tokens_list, loss_starts):
        seq_len = tokens.size(0)
        pad_len = max_len - seq_len

        padded = torch.nn.functional.pad(tokens, (0, pad_len), value=0)
        padded_tokens.append(padded)

        # Build loss mask for the TARGET (shifted by 1)
        # Target is full_seq[1:], so target length is max_len - 1
        mask = torch.zeros(max_len - 1, dtype=torch.float32)
        # Loss starts at position `ls` in the target and goes until seq_len - 2 (last real target token)
        target_end = seq_len - 1  # last index in target that has a real token
        mask[ls:target_end] = 1.0
        loss_masks.append(mask)

    padded_tokens = torch.stack(padded_tokens)
    loss_masks = torch.stack(loss_masks)

    x = padded_tokens[:, :-1]  # Input: everything except last
    y = padded_tokens[:, 1:]   # Target: everything except first

    return {"x": x, "y": y, "loss_mask": loss_masks}
