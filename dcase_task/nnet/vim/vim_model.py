import os
import torch
from .vim_module import FrameVimModel


class VimMamba2(torch.nn.Module):
    def __init__(self, mamba2_path, *args, vim_dropout=0.0, **kwargs, ) -> None:
        super().__init__(*args, **kwargs)
        self.mamba2 = FrameVimModel(mamba2_dropout=vim_dropout)
        self.load_mamba2(mamba2_path)
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


    def load_vim(self, path=None):
        if path is None:
            pre_path = "../nnet/ckpts/last.ckpt"
            assert os.path.exists(pre_path), "Please make sure you have a default path to load vim. Please change this path to the vim_as2M.ckpt that you downloaded."
            path = pre_path    # Change path to the vim_as2M.ckpt the downloaded checkpoint from the home page.
        state_dict = torch.load(path, map_location="cpu")["state_dict"]
        vim_state_dict = {}
        for k, v in state_dict.items():
            if "model.teacher.encoder." in k:
                print("model.teacher.encoder")
                if "encoder.norm." in k:
                    new_k = k.replace("model.teacher.encoder.norm", "norm_frame")
                elif "cls_token" in k:
                    continue
                else:
                    new_k = k.replace("model.teacher.encoder.", "")
                vim_state_dict[new_k] = v
            # C2F
            if "encoder.encoder.frame_encoder." in k:
                print("encoder.encoder.frame_encoder")
                new_k = k.replace("encoder.encoder.frame_encoder.", "")
                vim_state_dict[new_k] = v
                continue
            if "encoder.encoder.teacher_module." in k:
                continue
            # Frame
            if "encoder.encoder." in k:  # This is
                print("encoder.encoder")
                new_k = k.replace("encoder.encoder.", "")
                vim_state_dict[new_k] = v

        self.vim.load_state_dict(vim_state_dict, strict=True)
        for n, param in self.vim.named_parameters():
            param.requires_grad = False
        # state_dict = torch.load(path, map_location="cpu")["sed_teacher"]
        # vim_state_dict = {}
        # for k, v in state_dict.items():
        #     if ("total_ops" in k) or ("total_params" in k):
        #         continue
        #     if "vim_frame.vim." in k:
        #         k = k.replace("vim_frame.vim.", "")
        #         vim_state_dict[k] = v
        # self.vim.load_state_dict(vim_state_dict, strict=True)
        # for n, param in vim.named_parameters():
        #     param.requires_grad = False


# def main():
#     # 创建模型
#     model = VimMamba2(mamba2_path=None, vim_dropout=0.0)
#     model.eval()
#     # 输入：8 batch，64 dim，1001 长度
#     x = torch.randn(8, 64, 1001)
#     # 前向
#     with torch.no_grad():
#         out = model(x)
# 
#     print("输入 shape:", x.shape)
#     print("输出 shape:", out.shape)
# 
# if __name__ == "__main__":
#     main()
