from torch import nn
import torch
import math

from src.models.positional_encoding import PositionalEncoding

PAD_IDX = 0
SOS_IDX = 1
EOS_IDX = 2


class TextDecoder(nn.Module):

    def __init__(
        self,
        vocab_size,
        embed_dim=512,
        num_heads=8,
        num_layers=4,
        dropout=0.1
    ):

        super().__init__()

        self.embed_dim = embed_dim

        # Token embedding
        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embed_dim,
            padding_idx=PAD_IDX
        )

        # Positional encoding
        self.position = PositionalEncoding(embed_dim)

        # Dropout after embedding + position
        self.dropout = nn.Dropout(dropout)

        # Transformer decoder layer
        layer = nn.TransformerDecoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dropout=dropout,
            batch_first=True
        )

        # Multi-layer decoder
        self.decoder = nn.TransformerDecoder(
            decoder_layer=layer,
            num_layers=num_layers
        )

        # Final vocab projection
        self.fc = nn.Linear(
            embed_dim,
            vocab_size
        )

    def generate_causal_mask(self, size, device):

        """
        Prevent decoder from seeing future tokens
        """

        mask = torch.triu(
            torch.full((size, size), float("-inf")),
            diagonal=1
        )

        return mask.to(device)

    def forward(
        self,
        tgt_ids,
        memory,
        memory_padding_mask=None
    ):

        """
        Args:
            tgt_ids:
                [B, T]

            memory:
                [B, S, D]

            memory_padding_mask:
                [B, S]
        """

        tgt_padding_mask = (tgt_ids == PAD_IDX)

        tgt = self.embedding(tgt_ids)

        # Transformer embedding scaling
        tgt = tgt * math.sqrt(self.embed_dim)

        tgt = self.position(tgt)

        tgt = self.dropout(tgt)

        tgt_mask = self.generate_causal_mask(
            size=tgt.size(1),
            device=tgt.device
        )

        out = self.decoder(
            tgt=tgt,
            memory=memory,

            tgt_mask=tgt_mask,

            tgt_key_padding_mask=tgt_padding_mask,

            memory_key_padding_mask=memory_padding_mask
        )

        # -------------------------
        # Vocabulary projection
        # -------------------------
        out = self.fc(out)

        return out