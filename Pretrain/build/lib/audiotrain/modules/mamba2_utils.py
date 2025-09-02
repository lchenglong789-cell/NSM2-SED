import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from einops import rearrange


def drop_path(x, drop_prob: float = 0., training: bool = False):
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output


class DropPath(nn.Module):
    """
    Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """

    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


def SSD_Attention(x, dt, A, B, C, D, H=None, W=None):
    batch, seqlen, head, dim = x.shape  # (B, L, H, D)
    dstate = B.shape[2]
    V = x.permute(0, 2, 1, 3)  # (B, H, L, D)
    dt = dt.permute(0, 2, 1)  # (B, H, L)
    dA = dt.unsqueeze(-1) * A.view(1, -1, 1, 1).repeat(batch, 1, seqlen, 1)  #
    dA = -dA

    V_scaled = V * dA
    K = B.view(batch, 1, seqlen, dstate)  # (B, 1, L, D)

    KV = K.transpose(-2, -1) @ V_scaled  # (B, H, dstate, D)
    Q = C.view(batch, 1, seqlen, dstate)  # .repeat(1, head, 1, 1)
    x = Q @ KV  # (B, H, L, D)
    x = x + V * D.view(1, -1, 1, 1).repeat(batch, 1, seqlen, 1)
    x = x.permute(0, 2, 1, 3).contiguous()  # (B, L, H, D)
    return x, KV


class Mamba2(nn.Module):  # Mamba2 + attention
    def __init__(
            self,
            d_model,
            d_conv=3,  # default to 3 for 2D
            # expand=2,
            head_dim=64,  # default to 64
            ngroups=1,
            A_init_range=(1, 16),
            dt_min=0.001,
            dt_max=0.1,
            dt_init_floor=1e-4,
            dt_limit=(0.0, float("inf")),
            learnable_init_states=False,
            activation="silu",  # default to silu
            bias=False,
            conv_bias=True,
            d_state=64,
    ):
        super().__init__()
        self.d_model = d_model
        self.d_conv = d_conv
        # self.expand = expand
        # self.d_inner = int(self.expand * self.d_model)
        self.d_inner = self.d_model
        self.head_dim = head_dim
        self.d_state = d_state
        if ngroups == -1:
            ngroups = self.d_inner // self.head_dim  # equivalent to multi-head attention
        self.ngroups = ngroups
        assert self.d_inner % self.head_dim == 0
        self.nheads = self.d_inner // self.head_dim
        self.dt_limit = dt_limit
        self.learnable_init_states = learnable_init_states
        self.activation = activation
        d_in_proj = 2 * self.d_inner + 2 * self.ngroups * self.d_state + self.nheads
        self.in_proj = nn.Linear(self.d_model, int(d_in_proj), bias=bias)  #

        conv_dim = self.d_inner + 2 * self.ngroups * self.d_state
        self.conv2d = nn.Conv2d(
            in_channels=conv_dim,
            out_channels=conv_dim,
            groups=conv_dim,
            bias=conv_bias,
            kernel_size=d_conv,
            padding=(d_conv - 1) // 2,
        )

        self.act = nn.SiLU()

        # Initialize log dt bias
        dt = torch.exp(torch.rand(self.nheads) * (math.log(dt_max) - math.log(dt_min)) + math.log(dt_min))
        dt = torch.clamp(dt, min=dt_init_floor)
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        self.dt_bias = nn.Parameter(inv_dt)
        self.dt_bias._no_weight_decay = True

        # A parameter
        assert 0 < A_init_range[0] <= A_init_range[1]
        A = torch.empty(self.nheads, dtype=torch.float32).uniform_(*A_init_range)
        A_log = torch.log(A)
        self.A_log = nn.Parameter(A_log)
        # self.register_buffer("A_log", torch.zeros(self.nheads, dtype=torch.float32, device=device), persistent=True)
        self.A_log._no_weight_decay = True

        # D "skip" parameter
        self.D = nn.Parameter(torch.ones(self.nheads))
        self.D._no_weight_decay = True

        # modified from RMSNormGated to layer norm
        self.norm = nn.LayerNorm(self.d_inner)
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias)

    def forward(self, u, H, W, seq_idx=None):
        batch, seqlen, head = u.shape

        zxbcdt = self.in_proj(u)  # (B, L, d_in_proj)
        A = -torch.exp(self.A_log)  # (nheads) or (d_inner, d_state)

        z, xBC, dt = torch.split(
            zxbcdt, [self.d_inner, self.d_inner + 2 * self.ngroups * self.d_state, self.nheads], dim=-1
        )
        dt = F.softplus(dt + self.dt_bias)  # (B, L, nheads)

        # 2D Convolution
        xBC = xBC.view(batch, H, W, -1).permute(0, 3, 1, 2).contiguous()
        xBC = self.act(self.conv2d(xBC))
        xBC = xBC.permute(0, 2, 3, 1).view(batch, H * W, -1).contiguous()

        # Split into 3 main branches: X, B, C
        # These correspond to V, K, Q respectively in the SSM/attention duality
        x, B, C = torch.split(xBC, [self.d_inner, self.ngroups * self.d_state, self.ngroups * self.d_state], dim=-1)

        y, attn = SSD_Attention(
            rearrange(x, "b l (h p) -> b l h p", p=self.head_dim), dt, A, B, C, self.D, H, W
        )
        y = rearrange(y, "b l h p -> b l (h p)")

        y = self.norm(y)
        y = y * z
        out = self.out_proj(y)
        return out, attn


class VMamba2Block(nn.Module):
    def __init__(self, dim, input_resolution, num_heads, mlp_ratio=4., qkv_bias=True, drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm, ssd_expansion=1, ssd_ngroups=1, ssd_chunk_size=256,
                 linear_attn_duality=False, d_state=64, **kwargs):
        super().__init__()
        self.input_resolution = input_resolution

        self.cpe1 = nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim)
        self.norm1 = norm_layer(dim)

        self.attn = Mamba2(d_model=dim,
                           # expand=ssd_expansion,
                           # head_dim=dim*ssd_expansion//num_heads,
                           head_dim=dim // num_heads,
                           ngroups=ssd_ngroups,
                           d_state=d_state)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.cpe2 = nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim)
        self.norm2 = norm_layer(dim)
        self.mlp = Mlp(in_features=dim, hidden_features=int(dim * mlp_ratio), act_layer=act_layer, drop=drop)

    def forward(self, x, H=None, W=None, return_attention=False):
        B, L, C = x.shape
        if H is None and W is None:
            H, W = self.input_resolution
            assert H * W == L, "input feature has wrong size"

        x = x + self.cpe1(x.reshape(B, H, W, C).permute(0, 3, 1, 2)).flatten(2).permute(0, 2, 1)
        shortcut = x

        x = self.norm1(x)
        x, attn = self.attn(x, H, W)
        x = shortcut + self.drop_path(x)
        x = x + self.cpe2(x.reshape(B, H, W, C).permute(0, 3, 1, 2)).flatten(2).permute(0, 2, 1)

        # FFN
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        if return_attention:
            return x, attn
        else:
            return x
