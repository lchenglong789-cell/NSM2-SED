import torch
from torch import Tensor, nn
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


def silu(x):
    return x * F.sigmoid(x)


def segsum(x: Tensor) -> Tensor:
    T = x.size(-1)
    device = x.device
    x = x[..., None].repeat(1, 1, 1, 1, T)
    mask = torch.tril(torch.ones(T, T, dtype=torch.bool, device=device), diagonal=-1)
    x = x.masked_fill(~mask, 0)
    x_segsum = torch.cumsum(x, dim=-2)
    mask = torch.tril(torch.ones(T, T, dtype=torch.bool, device=device), diagonal=0)
    x_segsum = x_segsum.masked_fill(~mask, -torch.inf)
    return x_segsum

def SSD(x, A, B, C, chunk_size=1):
    chunk_size = chunk_size
    # if x.shape[1] % chunk_size == 0:
    #
    x = x.reshape(x.shape[0], x.shape[1] // chunk_size, chunk_size, x.shape[2], x.shape[3], )
    B = B.reshape(B.shape[0], B.shape[1] // chunk_size, chunk_size, B.shape[2], B.shape[3], )
    C = C.reshape(C.shape[0], C.shape[1] // chunk_size, chunk_size, C.shape[2], C.shape[3], )
    A = A.reshape(A.shape[0], A.shape[1] // chunk_size, chunk_size, A.shape[2])
    A = A.permute(0, 3, 1, 2)
    A_cumsum = torch.cumsum(A, dim=-1)

    # 1. Compute the output for each intra-chunk (diagonal blocks)
    L = torch.exp(segsum(A))
    Y_diag = torch.einsum("bclhn, bcshn, bhcls, bcshp -> bclhp", C, B, L, x)

    # 2. Compute the state for each intra-chunk
    decay_states = torch.exp(A_cumsum[:, :, :, -1:] - A_cumsum)
    states = torch.einsum("bclhn, bhcl, bclhp -> bchpn", B, decay_states, x)

    # 3. Compute the inter-chunk SSM recurrence; produces correct SSM states at chunk boundaries
    initial_states = torch.zeros_like(states[:, :1])
    states = torch.cat([initial_states, states], dim=1)
    decay_chunk = torch.exp(segsum(F.pad(A_cumsum[:, :, :, -1], (1, 0))))[0]
    new_states = torch.einsum("bhzc, bchpn -> bzhpn", decay_chunk, states)
    states = new_states[:, :-1]

    # 4. Compute state -> output conversion per chunk
    # (left term of low-rank factorization of off-diagonal blocks; C terms)
    state_decay_out = torch.exp(A_cumsum)
    Y_off = torch.einsum("bclhn, bchpn, bhcl -> bclhp", C, states, state_decay_out)
    # Add output of intra-chunk and inter-chunk terms (diagonal and off-diagonal blocks)
    # Y = rearrange(Y_diag + Y_off, "b c l h p -> b (c l) h p")
    Y = Y_diag + Y_off
    Y = Y.reshape(Y.shape[0], Y.shape[1] * Y.shape[2], Y.shape[3], Y.shape[4], )
    return Y


class RMSNorm(nn.Module):
    def __init__(self, d: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d))

    def forward(self, x, z):
        x = x * silu(z)
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight


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

        conv_dim = self.d_inner + 2 * d_state
        self.conv1d = nn.Conv1d(conv_dim, conv_dim, d_conv, groups=conv_dim, padding=d_conv - 1, )

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
        # self.norm = nn.LayerNorm(self.d_inner)
        self.RMSNorm = RMSNorm(self.d_inner, )
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias)

    def forward(self, u, seq_idx=None):
        batch, seqlen, head = u.shape

        zxbcdt = self.in_proj(u)  # (B, L, d_in_proj)
        A = -torch.exp(self.A_log)  # (nheads) or (d_inner, d_state)

        z, xBC, dt = torch.split(
            zxbcdt, [self.d_inner, self.d_inner + 2 * self.ngroups * self.d_state, self.nheads], dim=-1
        )
        dt = F.softplus(dt + self.dt_bias)  # (B, L, nheads)

        xBC = silu(self.conv1d(xBC.transpose(1, 2)).transpose(1, 2)[:, : u.shape[1], :])
        x, B, C = torch.split(xBC, [self.d_inner, self.d_state, self.d_state], dim=-1)

        _b, _l, _hp = x.shape
        _h = _hp // self.head_dim
        _p = self.head_dim
        x = x.reshape(_b, _l, _h, _p)

        y = SSD(x * dt.unsqueeze(-1),
                A * dt,
                B.unsqueeze(2),
                C.unsqueeze(2),
                chunk_size=1)
        y = y + x * self.D.unsqueeze(-1)

        _b, _l, _h, _p = y.shape
        y = y.reshape(_b, _l, _h * _p)

        y = self.RMSNorm(y, z)
        y = self.out_proj(y)

        return y

class VMamba2D(nn.Module):
    """
    2D Four-directional Mamba2 block for input (B, H, W, D)
    Four directions: left->right, right->left, top->bottom, bottom->top
    """
    def __init__(self, dim, num_heads, ssd_ngroups=1, d_state=64, dropout=0.1):
        super().__init__()
        self.dim = dim
        self.mamba2_lr = Mamba2(d_model=dim, head_dim=dim // num_heads, ngroups=ssd_ngroups, d_state=d_state)
        self.mamba2_rl = Mamba2(d_model=dim, head_dim=dim // num_heads, ngroups=ssd_ngroups, d_state=d_state)
        self.mamba2_tb = Mamba2(d_model=dim, head_dim=dim // num_heads, ngroups=ssd_ngroups, d_state=d_state)
        self.mamba2_bt = Mamba2(d_model=dim, head_dim=dim // num_heads, ngroups=ssd_ngroups, d_state=d_state)

        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(dim)

    def forward(self, x):
        """
        x: (B, H, W, D)
        """
        B, H, W, D = x.shape

        # Left -> Right
        x_lr = self.mamba2_lr(x.reshape(B*H, W, D))  # treat each row as a sequence
        x_lr = x_lr.reshape(B, H, W, D)

        # Right -> Left
        x_rl = self.mamba2_rl(x.flip(2).reshape(B*H, W, D))
        x_rl = x_rl.reshape(B, H, W, D).flip(2)

        # Top -> Bottom
        x_tb = self.mamba2_tb(x.permute(0, 2, 1, 3).reshape(B*W, H, D))
        x_tb = x_tb.reshape(B, W, H, D).permute(0, 2, 1, 3)

        # Bottom -> Top
        x_bt = self.mamba2_bt(x.flip(1).permute(0, 2, 1, 3).reshape(B*W, H, D))
        x_bt = x_bt.reshape(B, W, H, D).permute(0, 2, 1, 3).flip(1)

        # Sum all directions
        y = x_lr + x_rl + x_tb + x_bt

        # Residual + dropout + layernorm
        y = self.dropout(y)
        y = self.layer_norm(x + y)

        return y


class VMamba2DBlock(nn.Module):
    def __init__(self, dim, num_heads, num_layers=1, ssd_ngroups=1, d_state=64, dropout=0.1, norm_layer=nn.LayerNorm):
        super().__init__()
        self.layers = nn.ModuleList([
            VMamba2D(dim=dim, num_heads=num_heads, ssd_ngroups=ssd_ngroups, d_state=d_state, dropout=dropout)
            for _ in range(num_layers)
        ])
        self.norm = norm_layer(dim)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return self.norm(x)





# class VMamba(nn.Module):
#     """
#     Four-directional Mamba2 block:
#     - forward/backward along sequence
#     - optionally top/down or other axes for 2D-like input
#     """
#     def __init__(self, dim, num_heads, ssd_ngroups=1, d_state=64, dropout: float = 0.1):
#         super().__init__()
#         # four Mamba2 layers for four directions
#         self.mamba2_fwd = Mamba2(d_model=dim, head_dim=dim // num_heads, ngroups=ssd_ngroups, d_state=d_state)
#         self.mamba2_bwd = Mamba2(d_model=dim, head_dim=dim // num_heads, ngroups=ssd_ngroups, d_state=d_state)
#         self.mamba2_down = Mamba2(d_model=dim, head_dim=dim // num_heads, ngroups=ssd_ngroups, d_state=d_state)
#         self.mamba2_up = Mamba2(d_model=dim, head_dim=dim // num_heads, ngroups=ssd_ngroups, d_state=d_state)
#
#         self.dropout = nn.Dropout(dropout)
#         self.layer_norm = nn.LayerNorm(dim)
#
#     def forward(self, x):
#         """
#         x: (B, L, D)
#         """
#         B, L, D = x.shape
#
#         # forward/backward along sequence
#         x_fwd = self.mamba2_fwd(x)
#         x_bwd = self.mamba2_bwd(x.flip(1)).flip(1)
#
#         # simulate another axis by reshaping: e.g., treat batch as pseudo-spatial
#         # for 2D sequences, x2d = x.reshape(B1, H, W, D)
#         x_down = self.mamba2_down(x)  # can reshape for vertical scan
#         x_up = self.mamba2_up(x.flip(1)).flip(1)
#
#         # sum all four directions
#         y = x_fwd + x_bwd + x_down + x_up
#
#         # residual + layer norm
#         y = self.dropout(y)
#         y = self.layer_norm(x + y)
#         return y


# class VMambaBlock(nn.Module):
#     def __init__(self, dim, num_heads, num_layers=1, ssd_ngroups=1, d_state=64, dropout=0.1, norm_layer=nn.LayerNorm):
#         super().__init__()
#         self.layers = nn.ModuleList([
#             VMamba(dim=dim, num_heads=num_heads, ssd_ngroups=ssd_ngroups, d_state=d_state, dropout=dropout)
#             for _ in range(num_layers)
#         ])
#         self.norm = norm_layer(dim)
#
#     def forward(self, x):
#         for layer in self.layers:
#             x = layer(x)
#         return self.norm(x)