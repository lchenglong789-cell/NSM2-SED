import os
import torch
from .vim_module import FrameVimModel


class VimMamba2(torch.nn.Module):
    def __init__(self, mamba2_path, *args, amamba2_dropout=0.0, **kwargs, ) -> None:
        super().__init__(*args, **kwargs)
        self.mamba2 = FrameVimModel(mamba2_dropout=amamba2_dropout)
        # self.load_mamba2(mamba2_path)
        self.fake_length = torch.tensor([1001])
        self.cls_embed = None

    def set_cls_embed(self, cls_embed):
        self.cls_embed = cls_embed

    def forward(self, mamba2_feat, other_emb=None):
        mamba2_feat = mamba2_feat.unsqueeze(1)
        mamba2_x = self.mamba2.get_intermediate_layers(
            mamba2_feat,
            self.fake_length.to(mamba2_feat).repeat(len(mamba2_feat)),
            1,
            scene=False,
            other_emb=other_emb,
        )
        mamba2_x = mamba2_x.transpose(1, 2)
        return mamba2_x


def main():
    # 创建模型
    model = VimMamba2(mamba2_path=None, amamba2_dropout=0.0)
    model.eval()
    # 输入：8 batch，64 dim，1001 长度
    x = torch.randn(8, 64, 1001)
    # 前向
    with torch.no_grad():
        out = model(x)

    print("输入 shape:", x.shape)
    print("输出 shape:", out.shape)

if __name__ == "__main__":
    main()