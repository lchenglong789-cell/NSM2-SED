import torch
from fvcore.nn import FlopCountAnalysis, parameter_count_table

from CRNN_amamba2 import CRNN
# from CRNN_mamba2 import CRNN
# from CRNN_vim import CRNN
# from CRNN_VMamba import CRNN
# from CRNN_transformer import CRNN

model = CRNN(
    nclass=10,
    embedding_size=768,
    amamba2_dropout=0.0,
)
model.eval()

# 固定尺寸
B = 1
F = 128
T = 156

x = torch.randn(B, F, T)
pretrain_x = torch.randn(B, 64, 1001)

# ---------------- Params ----------------
def count_params(model):
    return sum(p.numel() for p in model.parameters()) / 1e6

print(f"Params: {count_params(model):.2f} M")

# ---------------- FLOPs ----------------
# 确保 adaptive pooling 使用常数
with torch.no_grad():
    flops = FlopCountAnalysis(model, (x, pretrain_x))
    print(f"FLOPs (community): {flops.total() / 1e9:.2f} G")

