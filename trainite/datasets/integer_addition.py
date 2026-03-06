"""
Integer Addition Dataset — In-memory deterministic dataset.

Generates random integer additions, formatted for autoregressive generation:

    Full sequence: [SOS] 1 2 + 3 4 = [SEP] 4 6 [EOS]
    Input (x):     [SOS] 1 2 + 3 4 = [SEP] 4 6
    Target (y):    1 2 + 3 4 = [SEP] 4 6 [EOS]
    Loss Mask:     0 0 0 0 0 0 0     1 1 1

Loss is computed ONLY on the sum portion (after [SEP]).
"""

import random
import torch
from torch.utils.data import Dataset


class IntegerAdditionDataset(Dataset):
    """Deterministic in-memory integer addition dataset."""

    PAD_TOKEN = "[PAD]"
    SOS_TOKEN = "[SOS]"
    EOS_TOKEN = "[EOS]"
    SEP_TOKEN = "[SEP]"

    def __init__(self, num_samples: int, min_digits: int, max_digits: int, seed: int = 42):
        self.num_samples = num_samples
        self.min_digits = min_digits
        self.max_digits = max_digits

        # Build vocabulary
        self.chars = list("0123456789+=")

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
            len1 = rng.randint(min_digits, max_digits)
            len2 = rng.randint(min_digits, max_digits)
            
            # Avoid leading zero if length > 1
            num1_str = str(rng.randint(10**(len1-1) if len1 > 1 else 0, 10**len1 - 1))
            num2_str = str(rng.randint(10**(len2-1) if len2 > 1 else 0, 10**len2 - 1))
            
            self.data.append((num1_str, num2_str))

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> dict:
        num1_str, num2_str = self.data[idx]
        sum_str = str(int(num1_str) + int(num2_str))

        s = f"{num1_str}+{num2_str}="
        tgt_s = sum_str

        src_tokens = [self.c2i[c] for c in s]
        tgt_tokens = [self.c2i[c] for c in tgt_s]

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
