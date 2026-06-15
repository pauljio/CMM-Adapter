# Adapter patch for SwinTransformer backbone
# 仅供代码合并参考
import torch
import torch.nn as nn

class SelfAdapter(nn.Module):
    def __init__(self, d_model, d_mid, dropout=0.1):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.down_proj = nn.Linear(d_model, d_mid)
        self.up_proj = nn.Linear(d_mid, d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.GELU()

    def forward(self, x):
        x = self.norm(x)
        x = self.down_proj(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.up_proj(x)
        return x
