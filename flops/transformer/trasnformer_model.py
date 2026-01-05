import os
import torch
from audio_transformer import FrameASTModel

class ATST(torch.nn.Module):
    def __init__(self, atst_path, *args, atst_dropout=0.0, **kwargs, ) -> None:
        super().__init__(*args, **kwargs)
        self.atst = FrameASTModel(atst_dropout=atst_dropout)
        self.fake_length = torch.tensor([1001])
        self.cls_embed = None

    def set_cls_embed(self, cls_embed):
        self.cls_embed = cls_embed

    def forward(self, atst_feat, other_emb=None):
        atst_feat = atst_feat.unsqueeze(1)
        atst_x = self.atst.get_intermediate_layers(
            atst_feat,
            self.fake_length.to(atst_feat).repeat(len(atst_feat)),
            1,
            scene=False,
            other_emb=other_emb,
        )
        atst_x = atst_x.transpose(1, 2)
        return atst_x



from thop import profile
import time

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ATST(atst_path=None, atst_dropout=0.0).to(device)
model.eval()

x = torch.randn(1, 64, 1001).to(device)


# ----------------- FLOPs and Params -----------------
flops, params = profile(model, inputs=(x,))
print(f"FLOPs: {flops/1e9:.2f}G")
print(f"Params: {params/1e6:.2f}M")


# ----------------- GPU inference latency (ms) -----------------
with torch.no_grad():
    for _ in range(10):
        _ = model(x)

n_runs = 100
torch.cuda.synchronize()
start_time = time.time()
with torch.no_grad():
    for _ in range(n_runs):
        _ = model(x)
        torch.cuda.synchronize()
end_time = time.time()

latency_ms = (end_time - start_time) / n_runs * 1000
print(f"GPU Inference Latency: {latency_ms:.2f} ms")


# ----------------- GPU inference latency (ms) -----------------
torch.cuda.empty_cache()
torch.cuda.reset_peak_memory_stats(device)

with torch.no_grad():
    _ = model(x)
    torch.cuda.synchronize()

peak_memory = torch.cuda.max_memory_allocated(device) / 1024**2  # MB
print(f"GPU Memory Occupancy: {peak_memory:.2f} MB")