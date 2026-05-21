import torch
import torch.nn as nn


class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 512):
        super().__init__()
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-torch.log(torch.tensor(10000.0)) / d_model)
        )
        encoding = torch.zeros(max_len, d_model)
        encoding[:, 0::2] = torch.sin(position * div_term)
        encoding[:, 1::2] = torch.cos(position * div_term[: encoding[:, 1::2].shape[1]])
        self.register_buffer("encoding", encoding.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.encoding[:, : x.shape[1]]


class TemporalTransformerClassifier(nn.Module):
    """Small Transformer encoder for temporal classification."""

    def __init__(
        self,
        input_dim: int,
        d_model: int = 128,
        num_heads: int = 4,
        num_layers: int = 1,
        dropout: float = 0.3,
        num_classes: int = 2,
    ):
        super().__init__()
        self.input_projection = nn.Linear(input_dim, d_model)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        self.position_encoding = SinusoidalPositionalEncoding(d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.classifier = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Dropout(dropout),
            nn.Linear(d_model, num_classes),
        )

    def forward(
        self,
        x: torch.Tensor,
        lengths: torch.Tensor | None = None,
    ) -> torch.Tensor:
        x = self.input_projection(x)
        batch_size = x.shape[0]
        cls = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat([cls, x], dim=1)
        x = self.position_encoding(x)

        key_padding_mask = None
        if lengths is not None:
            seq_len = x.shape[1]
            positions = torch.arange(seq_len, device=x.device).unsqueeze(0)
            valid_lengths = lengths.to(x.device).unsqueeze(1) + 1
            key_padding_mask = positions >= valid_lengths
            key_padding_mask[:, 0] = False

        encoded = self.encoder(x, src_key_padding_mask=key_padding_mask)
        return self.classifier(encoded[:, 0])

