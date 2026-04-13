import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Optional

from .norm import SlicedGroupNorm


class HeadLocalFFN(nn.Module):
    """Block-diagonal FFN where each attention head has its own independent FFN subspace.
    
    Instead of a single Linear(d_model, dim_feedforward), each head gets an independent
    Linear(head_dim, ffn_per_head). Uses einsum for BLAS-efficient batched matmul —
    equivalent to block-diagonal weights without explicit loops.
    
    This eliminates gradient interference across head configurations during elastic
    width training: when slicing to k heads, heads 0..k-1 produce identical outputs
    regardless of whether heads k..nhead-1 exist.
    """

    def __init__(self, d_model: int, dim_feedforward: int, nhead: int,
                 dropout: float = 0.1, activation: str = "relu"):
        super().__init__()
        assert d_model % nhead == 0, "d_model must be divisible by nhead"
        assert dim_feedforward % nhead == 0, "dim_feedforward must be divisible by nhead"

        self.nhead = nhead
        self.head_dim = d_model // nhead
        self.ffn_per_head = dim_feedforward // nhead

        # Block-diagonal weights: [nhead, ffn_per_head, head_dim] and [nhead, head_dim, ffn_per_head]
        self.w1 = nn.Parameter(torch.empty(nhead, self.ffn_per_head, self.head_dim))
        self.b1 = nn.Parameter(torch.zeros(nhead, self.ffn_per_head))
        self.w2 = nn.Parameter(torch.empty(nhead, self.head_dim, self.ffn_per_head))
        self.b2 = nn.Parameter(torch.zeros(nhead, self.head_dim))

        self.norm = SlicedGroupNorm(dim_feedforward, number_slice=nhead)
        self.dropout = nn.Dropout(dropout)

        if activation == "relu":
            self.activation = F.relu
        elif activation == "gelu":
            self.activation = F.gelu
        else:
            raise ValueError(f"Unsupported activation: {activation}")

        self._reset_parameters()

    def _reset_parameters(self):
        for w in [self.w1, self.w2]:
            nn.init.xavier_uniform_(w)

    def forward(self, x: Tensor, effective_heads: Optional[int] = None) -> Tensor:
        """
        Args:
            x: [..., k * head_dim] where k = effective_heads or nhead
            effective_heads: number of active heads (None = all heads)
        Returns:
            Tensor of same shape as input: [..., k * head_dim]
        """
        k = effective_heads if effective_heads is not None else self.nhead

        # [..., k * head_dim] -> [..., k, head_dim]
        x_heads = x.unflatten(-1, (k, self.head_dim))

        # Up-project each head independently: h[...,k,f] = sum_d x[...,k,d] * w1[k,f,d]
        # 'd' (head_dim) is summed over, 'k' (head index) is batched — no cross-head mixing
        # x_heads[...,k,1,d] @ w1[k,d,f] -> [...,k,1,f] -> [...,k,f]
        h = (x_heads.unsqueeze(-2) @ self.w1[:k].permute(0, 2, 1)).squeeze(-2) + self.b1[:k]

        # Flatten for group norm, activate, norm, dropout
        h = h.flatten(-2)                                     # [..., k * ffn_per_head]
        h = self.activation(h)
        h = self.norm(h, effective_embed_dim=k * self.ffn_per_head)
        h = self.dropout(h)

        # Unflatten back to per-head
        h = h.unflatten(-1, (k, self.ffn_per_head))           # [..., k, ffn_per_head]

        # Down-project each head independently: out[...,k,d] = sum_f h[...,k,f] * w2[k,d,f]
        # 'f' (ffn_per_head) is summed over, 'k' (head index) is batched — no cross-head mixing
        # h[...,k,1,f] @ w2.permute[k,f,d] -> [...,k,1,d] -> [...,k,d]
        out = (h.unsqueeze(-2) @ self.w2[:k].permute(0, 2, 1)).squeeze(-2) + self.b2[:k]

        return out.flatten(-2)                                 # [..., k * head_dim]

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        """Compatibility: reshape standard FFN weights (linear1/linear2) into head-local format."""
        # Check if loading from a standard SlicedLinear checkpoint
        l1_weight_key = prefix.replace('ffn.', 'linear1.') + 'weight'
        l1_bias_key = prefix.replace('ffn.', 'linear1.') + 'bias'
        l2_weight_key = prefix.replace('ffn.', 'linear2.') + 'weight'
        l2_bias_key = prefix.replace('ffn.', 'linear2.') + 'bias'

        # If standard linear1/linear2 keys exist, reshape them
        if l1_weight_key in state_dict:
            # linear1.weight: [dim_feedforward, d_model] -> [nhead, ffn_per_head, head_dim]
            w1_flat = state_dict.pop(l1_weight_key)
            w1_reshaped = w1_flat.view(self.nhead, self.ffn_per_head, self.nhead, self.head_dim)
            # Take block-diagonal: each head's ffn_per_head rows, each head's head_dim cols
            w1_diag = torch.stack([w1_reshaped[i, :, i, :] for i in range(self.nhead)])
            state_dict[prefix + 'w1'] = w1_diag

        if l1_bias_key in state_dict:
            b1_flat = state_dict.pop(l1_bias_key)
            state_dict[prefix + 'b1'] = b1_flat.view(self.nhead, self.ffn_per_head)

        if l2_weight_key in state_dict:
            # linear2.weight: [d_model, dim_feedforward] -> [nhead, head_dim, ffn_per_head]
            w2_flat = state_dict.pop(l2_weight_key)
            w2_reshaped = w2_flat.view(self.nhead, self.head_dim, self.nhead, self.ffn_per_head)
            w2_diag = torch.stack([w2_reshaped[i, :, i, :] for i in range(self.nhead)])
            state_dict[prefix + 'w2'] = w2_diag

        if l2_bias_key in state_dict:
            b2_flat = state_dict.pop(l2_bias_key)
            state_dict[prefix + 'b2'] = b2_flat.view(self.nhead, self.head_dim)

        super()._load_from_state_dict(state_dict, prefix, local_metadata, strict,
                                      missing_keys, unexpected_keys, error_msgs)
