import torch
import time
import statistics
# import psutil
import os
from torch.profiler import profile, ProfilerActivity

# from CRNN_amamba2 import CRNN
# from CRNN_mamba2 import CRNN
# from CRNN_vim import CRNN
from CRNN_VMamba import CRNN
# from CRNN_transformer import CRNN

# ---------------- model ----------------
model = CRNN(
    nclass=10,
    embedding_size=768,
    amamba2_dropout=0.0,
)
model.eval()


# ---------------- input ----------------
B = 1
x = torch.randn(B, 128, 156)        # (B, F, T)
pretrain_x = torch.randn(B, 64, 1001)


def count_params(model):
    return sum(p.numel() for p in model.parameters()) / 1e6

print(f"Params: {count_params(model):.2f} M")


# ---------------- FLOPs ----------------
with profile(activities=[ProfilerActivity.CPU], with_flops=True) as prof:
    with torch.no_grad():
        _ = model(x, pretrain_x)

total_flops = sum(evt.flops for evt in prof.key_averages() if evt.flops)
print(f"FLOPs: {total_flops / 1e9:.2f} G")


# ================== GPU Inference Speed ==================
device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device)
x_cuda = x.to(device)
pretrain_x_cuda = pretrain_x.to(device)


# ---------- warm-up ----------
with torch.no_grad():
    for _ in range(10):
        _ = model(x_cuda, pretrain_x_cuda)
torch.cuda.synchronize()


# ---------- latency statistics ----------
latencies = []
iters = 100

with torch.no_grad():
    for _ in range(iters):
        torch.cuda.synchronize()
        t0 = time.time()

        _ = model(x_cuda, pretrain_x_cuda)

        torch.cuda.synchronize()
        t1 = time.time()
        latencies.append(t1 - t0)

# ---------- statistics ----------
mean_latency = statistics.mean(latencies) * 1000   # ms
p50_latency = statistics.median(latencies) * 1000
p90_latency = sorted(latencies)[int(0.9 * len(latencies))] * 1000

# throughput (batch size = B)
throughput = B / (statistics.mean(latencies))

print(f"Mean Latency: {mean_latency:.2f} ms")
print(f"P50 Latency:  {p50_latency:.2f} ms")
print(f"P90 Latency:  {p90_latency:.2f} ms")
print(f"Throughput:   {throughput:.2f} samples/sec")

# ---------------- peak memory ----------------
torch.cuda.reset_peak_memory_stats()
with torch.no_grad():
    _ = model(x_cuda, pretrain_x_cuda)
torch.cuda.synchronize()

mem = torch.cuda.max_memory_allocated() / 1024**2
print(f"GPU Peak Memory: {mem:.1f} MB")




# latency
start = time.time()
with torch.no_grad():
    for _ in range(100):
        _ = model(x_cuda, pretrain_x_cuda)
torch.cuda.synchronize()
end = time.time()

latency_gpu = (end - start) / 100 * 1000
print(f"GPU Latency: {latency_gpu:.2f} ms")

# # peak memory
# torch.cuda.reset_peak_memory_stats()
# with torch.no_grad():
#     _ = model(x_cuda, pretrain_x_cuda)
# mem = torch.cuda.max_memory_allocated() / 1024**2
# print(f"GPU Peak Memory: {mem:.1f} MB")

