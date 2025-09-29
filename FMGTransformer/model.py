
import torch
from torch import nn
import torch.nn.functional as F
from einops import rearrange, repeat

class NumericalEmbedder(nn.Module):
    def __init__(self, dim, num_numerical_types: int, use_mlp: bool = False):
        super().__init__()
        self.weights = nn.Parameter(torch.empty(num_numerical_types, dim))
        self.biases  = nn.Parameter(torch.empty(num_numerical_types, dim))
        nn.init.ones_(self.weights)
        nn.init.zeros_(self.biases)

        self.use_mlp = use_mlp
        if use_mlp:
            self.mlp = nn.Sequential(
                nn.Linear(dim, dim),
                nn.GELU(),
                nn.Linear(dim, dim)
            )

            nn.init.zeros_(self.mlp[-1].weight)
            nn.init.zeros_(self.mlp[-1].bias)
        else:
            self.mlp = None

    def forward(self, x):
        e = rearrange(x, 'b n -> b n 1').to(dtype=self.weights.dtype) * self.weights + self.biases
        if self.mlp is not None:
            e = e + self.mlp(e)
        return e


class TokenDropout(nn.Module):
    def __init__(self, p: float = 0.1):
        super().__init__()
        self.p = float(p)
    def forward(self, x):
        if not self.training or self.p <= 0:
            return x
        B, N, _ = x.shape
        mask = torch.empty(B, N, 1, device=x.device).bernoulli_(1 - self.p)
        return x * mask


class Transformer(nn.Module):
    def __init__(self, dim: int, depth: int, heads: int, attn_dropout: float = 0.1):
        super().__init__()
        enc_layer = nn.TransformerEncoderLayer(
            d_model=dim, nhead=heads, dim_feedforward=dim * 4,
            dropout=attn_dropout, activation='gelu', batch_first=True
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=depth)
    def forward(self, x):
        return self.encoder(x)


class MultiScaleCNN(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.conv1 = nn.Conv1d(dim, dim, kernel_size=1, padding=0)
        self.conv3 = nn.Conv1d(dim, dim, kernel_size=3, padding=1)
        self.conv5 = nn.Conv1d(dim, dim, kernel_size=5, padding=2)
        self.pool = nn.AdaptiveMaxPool1d(1)
    def forward(self, x):
        x1 = F.relu(self.conv1(x))
        x3 = F.relu(self.conv3(x))
        x5 = F.relu(self.conv5(x))
        x_cat = torch.cat([x1, x3, x5], dim=1)
        return self.pool(x_cat).squeeze(-1)

# ---------- 注意力池化（可选，替代CNN） ----------
class AttnPool(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.score = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, 1))
    def forward(self, tokens):
        w = self.score(tokens)
        a = torch.softmax(w, dim=1)
        return (tokens * a).sum(dim=1)


class GatedAttention(nn.Module):
    def __init__(self, dim_in: int, dim_out: int):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(dim_in, dim_out),
            nn.ReLU(),
            nn.Linear(dim_out, dim_out),
            nn.Sigmoid()
        )
    def forward(self, x):
        return x * self.gate(x)


class DropPath(nn.Module):
    def __init__(self, p: float = 0.0):
        super().__init__()
        self.p = float(p)
    def forward(self, x):
        if not self.training or self.p <= 0.0:
            return x
        keep = 1.0 - self.p
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        mask = x.new_empty(shape).bernoulli_(keep)
        return x * mask / keep


class FTTransformer_MultiScaleGated(nn.Module):
    def __init__(
        self,
        categories,
        num_continuous: int,
        dim: int = 128,
        depth: int = 2,
        heads: int = 4,
        dim_out: int = 1,
        num_special_tokens: int = 1,
        attn_dropout: float = 0.1,
        embed_dropout: float = 0.1,
        token_dropout: float = 0.0,
        use_transformer: bool = True,
        use_cnn: bool = True,
        use_gate: bool = True,
        use_fusion: bool = True,

        pool: str = "attn",
        use_field_embed: bool = True,
        numerical_mlp: bool = False,
        head_dropout: float = 0.0,
        # —— 新增：残差增强相关开关 ——
        fuser_residual: bool = True,
        head_residual: bool = True,
        drop_path: float = 0.0
    ):
        super().__init__()
        self.num_categories = len(categories)
        self.num_unique_categories = int(sum(categories))
        self.num_continuous = int(num_continuous)

        self.use_transformer = use_transformer
        self.use_cnn = use_cnn
        self.use_gate = use_gate
        self.use_fusion = use_fusion
        self.pool = pool
        self.use_field_embed = use_field_embed

        if self.num_categories > 0:
            cs = torch.tensor(categories[:-1], dtype=torch.long) if self.num_categories > 1 else torch.tensor([], dtype=torch.long)
            offsets = [num_special_tokens] + (torch.cumsum(cs, dim=0).tolist() if cs.numel() > 0 else [])
            categories_offset = torch.tensor(offsets, dtype=torch.long)
        else:
            categories_offset = torch.tensor([], dtype=torch.long)
        self.register_buffer('categories_offset', categories_offset)


        total_tokens = self.num_unique_categories + num_special_tokens
        self.categorical_embeds = nn.Embedding(total_tokens, dim)
        self.numerical_embedder = NumericalEmbedder(dim, num_continuous, use_mlp=numerical_mlp)


        self.num_fields = self.num_categories + self.num_continuous
        if use_field_embed and self.num_fields > 0:
            self.field_embed = nn.Embedding(self.num_fields, dim)
            self.register_buffer('field_idx_cat', torch.arange(self.num_categories, dtype=torch.long))
            self.register_buffer('field_idx_num', torch.arange(self.num_continuous, dtype=torch.long) + self.num_categories)
        else:
            self.field_embed = None


        self.embed_scale = dim ** -0.5
        self.embed_dropout = nn.Dropout(embed_dropout)
        self.token_dropout = TokenDropout(token_dropout)
        self.pre_ln = nn.LayerNorm(dim)


        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))


        self.transformer = Transformer(dim=dim, depth=depth, heads=heads, attn_dropout=attn_dropout)

        pool_out_dim = 0
        if pool == "cnn":
            self.multi_scale_cnn = MultiScaleCNN(dim)
            pool_out_dim = dim * 3
        elif pool in ("attn", "mean"):
            self.attn_pool = AttnPool(dim) if pool == "attn" else None
            pool_out_dim = dim
        else:
            pool_out_dim = 0


        fused_in = dim + pool_out_dim
        fused_out = dim * 2

        self.reduce = nn.Linear(fused_in, fused_out)
        self.shortcut_reduce = nn.Linear(fused_in, fused_out, bias=False)
        self.fuse_ln = nn.LayerNorm(fused_out)
        self.fuser_residual = fuser_residual
        self.drop_path1 = DropPath(drop_path)

        self.attn_gate = GatedAttention(fused_out, fused_out) if use_gate else nn.Identity()

        self.head_residual = head_residual
        hidden = fused_out
        self.head_pre_ln = nn.LayerNorm(hidden)
        self.head_ff = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Dropout(head_dropout),
            nn.Linear(hidden, hidden)
        )
        self.drop_path2 = DropPath(drop_path)

        self.out = nn.Linear(hidden, dim_out)

    def forward(self, x_categ, x_numer):
        dev = self.categorical_embeds.weight.device

        if x_categ.numel() > 0:
            x_categ = x_categ.to(dev, non_blocking=True).long()
        x_numer = x_numer.to(dev, non_blocking=True).float()

        if self.num_categories > 0 and x_categ.numel() > 0:
            x_cat = self.categorical_embeds(x_categ + self.categories_offset.to(dev))
        else:
            B = x_numer.size(0)
            x_cat = x_numer.new_zeros(B, 0, self.categorical_embeds.embedding_dim)

        x_num = self.numerical_embedder(x_numer)


        if self.use_field_embed and self.num_fields > 0:
            B = x_num.size(0)
            if x_cat.size(1) > 0:
                fe_cat = self.field_embed(self.field_idx_cat.to(dev)).unsqueeze(0).expand(B, -1, -1)
            else:
                fe_cat = x_num.new_zeros(B, 0, x_num.size(-1))
            fe_num = self.field_embed(self.field_idx_num.to(dev)).unsqueeze(0).expand(B, -1, -1)
            x_cat = x_cat + fe_cat
            x_num = x_num + fe_num

        if self.use_fusion:
            x = torch.cat([x_cat, x_num], dim=1)
        elif self.num_continuous > 0:
            x = x_num
        else:
            x = x_cat


        x = x * self.embed_scale
        x = self.embed_dropout(x)
        x = self.token_dropout(x)
        x = self.pre_ln(x)


        b = x.size(0)
        cls_tokens = repeat(self.cls_token, '1 1 d -> b 1 d', b=b)
        x = torch.cat([cls_tokens, x], dim=1)
        if self.use_transformer:
            x = self.transformer(x)

        x_cls = x[:, 0]
        tokens = x[:, 1:]

        if self.pool == "cnn" and tokens.numel() > 0:
            x_pool = self.multi_scale_cnn(tokens.permute(0, 2, 1))
        elif self.pool == "attn" and tokens.numel() > 0:
            x_pool = self.attn_pool(tokens)
        elif self.pool == "mean" and tokens.numel() > 0:
            x_pool = tokens.mean(dim=1)
        else:
            x_pool = x_cls.new_zeros(x_cls.size(0), 0)

        x_all = torch.cat([x_cls, x_pool], dim=-1)
        y_main = self.reduce(x_all)
        if self.fuser_residual:
            y = self.drop_path1(y_main) + self.shortcut_reduce(x_all)
        else:
            y = y_main
        y = self.fuse_ln(y)

        # ---------- 门控 ----------
        y = self.attn_gate(y)

        # ---------- Head 残差 ----------
        if self.head_residual:
            h = self.head_pre_ln(y)
            h = self.head_ff(h)
            y = self.drop_path2(h) + y
            return self.out(y)
        else:
            return self.out(self.head_pre_ln(y))
