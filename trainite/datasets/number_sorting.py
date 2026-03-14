"""
Number Sorting Dataset — In-memory deterministic dataset.

Generates random sequences of digits and their sorted counterpart.
Example:
    Input (x):     [SOS] 9 2 7 1 5 [SEP] 1 2 5 7 9
    Target (y):    9 2 7 1 5 [SEP] 1 2 5 7 9 [EOS]
    Loss Mask:     0 0 0 0 0 0 0     1 1 1 1 1 1

Loss is computed ONLY on the sorted string portion (after [SEP]).
"""

import random
import torch
from torch.utils.data import Dataset


class NumberSortingDataset(Dataset):
    """Deterministic in-memory number sorting dataset."""

    PAD_TOKEN = "[PAD]"
    SOS_TOKEN = "[SOS]"
    EOS_TOKEN = "[EOS]"
    SEP_TOKEN = "[SEP]"

    def __init__(self, num_samples: int, min_len: int, max_len: int, seed: int = 42):
        self.num_samples = num_samples
        self.min_len = min_len
        self.max_len = max_len

        # Build vocabulary: Digits 0-9
        self.chars = list("0123456789, ")

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

        self.vocab_size = len(self.chars) + 4

        # Generate data deterministically
        rng = random.Random(seed)
        self.data = []
        for _ in range(num_samples):
            length = rng.randint(min_len, max_len)
            sequence = []
            for _ in range(length):
                # Ensure a balanced mix of 1-digit, 2-digit, and 3-digit numbers
                # otherwise random.randint(0, 999) will be 90% 3-digit numbers
                p = rng.random()
                if p < 0.2:
                    val = rng.randint(0, 9)
                elif p < 0.5:
                    val = rng.randint(10, 99)
                else:
                    val = rng.randint(100, 999)
                sequence.append(val)
            
            sorted_seq = sorted(sequence)
            
            # Format as comma separated strings for easy reading
            seq_str = ",".join(map(str, sequence))
            sorted_str = ",".join(map(str, sorted_seq))
            
            self.data.append((seq_str, sorted_str))

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> dict:
        seq_str, sorted_str = self.data[idx]

        src_tokens = [self.c2i[c] for c in seq_str]
        tgt_tokens = [self.c2i[c] for c in sorted_str]

        # Full sequence: [SOS] + src + [SEP] + tgt + [EOS]
        full_seq = [self.sos_idx] + src_tokens + [self.sep_idx] + tgt_tokens + [self.eos_idx]

        loss_mask_start = len(src_tokens)

        return {
            "tokens": torch.tensor(full_seq, dtype=torch.long),
            "loss_mask_start": loss_mask_start,
        }

    def decode(self, token_ids: list[int]) -> str:
        result = []
        for t in token_ids:
            if t == self.eos_idx:
                break
            if t in (self.pad_idx, self.sos_idx, self.sep_idx):
                continue
            result.append(self.i2c.get(t, "?"))
        return "".join(result)
