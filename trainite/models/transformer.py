"""
Decoder-only Transformer for sequence-to-sequence tasks.

Architecture: A standard GPT-style model using nn.TransformerEncoder with
causal masking. This is the canonical way to implement a decoder-only
language model in native PyTorch — an encoder stack with causal self-attention
is architecturally identical to a decoder without cross-attention.
"""

import math
import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    """Standard sinusoidal positional encoding."""

    def __init__(self, d_model: int, max_len: int = 512, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)  # (max_len, 1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)
        )  # (d_model/2,)

        pe = torch.zeros(1, max_len, d_model)  # (1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)

        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape (batch_size, seq_len, d_model)
        """
        x = x + self.pe[:, : x.size(1)]
        return self.dropout(x)


class DecoderOnlyTransformer(nn.Module):
    """
    A GPT-style decoder-only transformer.

    Uses nn.TransformerEncoder with a causal attention mask,
    which is architecturally equivalent to a decoder-only model
    (self-attention + feedforward, no cross-attention).
    """

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int = 128,
        num_heads: int = 4,
        num_layers: int = 4,
        dim_feedforward: int = 512,
        max_seq_len: int = 128,
        dropout: float = 0.1,
        pad_token_id: int = 0,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.pad_token_id = pad_token_id
        self.max_seq_len = max_seq_len

        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_token_id)
        self.pos_encoder = PositionalEncoding(embed_dim, max_len=max_seq_len, dropout=dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            norm_first=True,  # Pre-norm for more stable training
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
            enable_nested_tensor=False,
        )

        self.layer_norm = nn.LayerNorm(embed_dim)
        self.output_projection = nn.Linear(embed_dim, vocab_size)

        self._init_weights()

    def _init_weights(self):
        """Initialize weights with Xavier uniform."""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, src: torch.Tensor) -> torch.Tensor:
        """
        Args:
            src: Input token IDs, shape (batch_size, seq_len)

        Returns:
            Logits of shape (batch_size, seq_len, vocab_size)
        """
        seq_len = src.size(1)

        # Causal mask: upper triangular -inf mask
        causal_mask = nn.Transformer.generate_square_subsequent_mask(
            seq_len, device=src.device
        )

        # Padding mask: True where padding
        padding_mask = src == self.pad_token_id

        # Embedding + positional encoding
        x = self.embedding(src) * math.sqrt(self.embed_dim)
        x = self.pos_encoder(x)

        # Transformer forward with causal + padding masks
        x = self.transformer(
            x,
            mask=causal_mask,
            src_key_padding_mask=padding_mask,
            is_causal=True,
        )

        x = self.layer_norm(x)
        logits = self.output_projection(x)
        return logits

    @torch.no_grad()
    def generate(
        self,
        prompt: torch.Tensor,
        max_new_tokens: int = 50,
        eos_token_id: int = 2,
    ) -> torch.Tensor:
        """
        Autoregressively generate tokens given a prompt.

        Args:
            prompt: Tensor of shape (1, prompt_len) with token IDs.
            max_new_tokens: Maximum number of new tokens to generate.
            eos_token_id: Token ID to stop generation.

        Returns:
            Generated sequence tensor of shape (1, generated_len).
        """
        self.eval()
        generated = prompt.clone()

        for _ in range(max_new_tokens):
            # Truncate to max_seq_len if needed
            input_seq = generated[:, -self.max_seq_len :]

            logits = self.forward(input_seq)
            next_token_logits = logits[:, -1, :]  # Get logits for the last position
            next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)

            generated = torch.cat([generated, next_token], dim=1)

            if next_token.item() == eos_token_id:
                break

        return generated
