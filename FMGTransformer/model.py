import torch
from torch import nn
import torch.nn.functional as F
from einops import rearrange, repeat

# Numerical Embedder（数值特征的映射器）
class NumericalEmbedder(nn.Module):
    def __init__(self, dim, num_numerical_types):
        super().__init__()
        self.weights = nn.Parameter(torch.randn(num_numerical_types, dim))
        self.biases = nn.Parameter(torch.randn(num_numerical_types, dim))

    def forward(self, x):
        x = rearrange(x, 'b n -> b n 1')
        return x * self.weights + self.biases

# Transformer 编码器
class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, attn_dropout, ff_dropout):
        super().__init__()
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=dim,
                nhead=heads,
                dim_feedforward=dim * 4,
                dropout=attn_dropout,
                activation='gelu',
                batch_first=True
            )
            for _ in range(depth)
        ])

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

import torch
from torch import nn
import torch.nn.functional as F
from einops import repeat


# 多尺度 CNN 模块
class MultiScaleCNN(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv1 = nn.Conv1d(dim, dim, kernel_size=1, padding=0)
        self.conv3 = nn.Conv1d(dim, dim, kernel_size=3, padding=1)
        self.conv5 = nn.Conv1d(dim, dim, kernel_size=5, padding=2)
        self.pool = nn.AdaptiveMaxPool1d(1)

    def forward(self, x):  # (B, D, N)
        x1 = F.relu(self.conv1(x))
        x3 = F.relu(self.conv3(x))
        x5 = F.relu(self.conv5(x))
        x_cat = torch.cat([x1, x3, x5], dim=1)  # (B, 3D, N)
        return self.pool(x_cat).squeeze(-1)     # (B, 3D)

# Gated Feature Attention
class GatedAttention(nn.Module):
    def __init__(self, dim_in, dim_out):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(dim_in, dim_out),
            nn.ReLU(),
            nn.Linear(dim_out, dim_out),
            nn.Sigmoid()
        )

    def forward(self, x):
        gate = self.gate(x)
        return x * gate

# 主模型
class FMGTransformer(nn.Module):
    def __init__(
        self,
        categories,
        num_continuous,
        dim=64,
        depth=4,
        heads=8,
        dim_head=16,
        dim_out=1,
        num_special_tokens=1,
        attn_dropout=0.1,
        ff_dropout=0.1,
    ):
        super().__init__()
        self.num_categories = len(categories)
        self.num_unique_categories = sum(categories)
        self.num_continuous = num_continuous

        total_tokens = self.num_unique_categories + num_special_tokens
        categories_offset = torch.tensor(
            [num_special_tokens] + list(torch.cumsum(torch.tensor(categories[:-1]), dim=0)))
        self.register_buffer('categories_offset', categories_offset)

        self.categorical_embeds = nn.Embedding(total_tokens, dim)
        self.numerical_embedder = NumericalEmbedder(dim, num_continuous)

        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))

        self.transformer = Transformer(
            dim=dim,
            depth=depth,
            heads=heads,
            dim_head=dim_head,
            attn_dropout=attn_dropout,
            ff_dropout=ff_dropout
        )

        self.multi_scale_cnn = MultiScaleCNN(dim)
        self.reduce = nn.Linear(896, 256)
        self.attn_gate = GatedAttention(dim * 2, dim * 2)

        self.to_logits = nn.Sequential(
            nn.LayerNorm(dim * 2),
            nn.ReLU(),
            nn.Linear(dim * 2, dim_out)
        )

    def forward(self, x_categ, x_numer):
        x_categ = x_categ + self.categories_offset
        x_categ = self.categorical_embeds(x_categ)

        x_numer = self.numerical_embedder(x_numer)

        x = torch.cat([x_categ, x_numer], dim=1)

        b = x.shape[0]
        cls_tokens = repeat(self.cls_token, '1 1 d -> b 1 d', b=b)
        x = torch.cat([cls_tokens, x], dim=1)
        x = self.transformer(x)

        x_cls = x[:, 0]           # (B, D)
        x_all_tokens = x.permute(0, 2, 1)  # (B, D, N+1)
        x_feature_tokens = x[:, 1:].permute(0, 2, 1)  # (B, D, N)
        x_cnn1 = self.multi_scale_cnn(x_all_tokens)  # 包含CLS
        x_cnn2 = self.multi_scale_cnn(x_feature_tokens)  # 不含CLS
        x_cnn_combined = torch.cat([x_cnn1, x_cnn2], dim=-1)  # (B, 2 * CNN_dim)
        x_all = torch.cat([x_cls, x_cnn_combined], dim=-1)  # (32, 896)
        #print(x_cls.shape, x_cnn1.shape, x_cnn2.shape)

        #x_tokens = x[:, 1:].permute(0, 2, 1)  # (B, D, N)
        #x_cnn = self.multi_scale_cnn(x_tokens)
        #x_all = torch.cat([x_cls, x_cnn], dim=-1)  # (B, 256)

        x_reduced = self.reduce(x_all)  # (B, 128)
        x_attn = self.attn_gate(x_reduced)

        return self.to_logits(x_attn)


