import os
import torch
from .audio_mamba2 import FrameASMModel


class AMamba2(torch.nn.Module):
    def __init__(self, amamba2_path, *args, amamba2_dropout=0.0, **kwargs, ) -> None:
        super().__init__(*args, **kwargs)
        self.amamba2 = FrameASMModel(amamba2_dropout=amamba2_dropout)
        self.load_amamba2(amamba2_path)
        self.fake_length = torch.tensor([1001])
        self.cls_embed = None

    def set_cls_embed(self, cls_embed):
        self.cls_embed = cls_embed

    def forward(self, amamba2_feat, other_emb=None):
        amamba2_feat = amamba2_feat.unsqueeze(1)
        amamba2_x = self.amamba2.get_intermediate_layers(
            amamba2_feat,
            self.fake_length.to(amamba2_feat).repeat(len(amamba2_feat)),
            1,
            scene=False,
            other_emb=other_emb,
        )
        amamba2_x = amamba2_x.transpose(1, 2)
        return amamba2_x

    def load_amamba2(self, path=None):
        if path is None:
            pre_path = "../ckpts/amamba2_as2M.ckpt"
            assert os.path.exists(pre_path), "Please make sure you have a default path to load AMamba2. Please change this path to the amamba2_as2M.ckpt that you downloaded."
            path = pre_path    # Change path to the amamba2_as2M.ckpt the downloaded checkpoint from the home page.
        state_dict = torch.load(path, map_location="cpu")["state_dict"]
        amamba2_state_dict = {}
        for k, v in state_dict.items():
            if "model.teacher.encoder." in k:
                print("model.teacher.encoder")
                if "encoder.norm." in k:
                    new_k = k.replace("model.teacher.encoder.norm", "norm_frame")
                elif "cls_token" in k:
                    continue
                else:
                    new_k = k.replace("model.teacher.encoder.", "")
                amamba2_state_dict[new_k] = v
            # C2F
            if "encoder.encoder.frame_encoder." in k:
                print("encoder.encoder.frame_encoder")
                new_k = k.replace("encoder.encoder.frame_encoder.", "")
                amamba2_state_dict[new_k] = v
                continue
            if "encoder.encoder.teacher_module." in k:
                continue
            # Frame
            if "encoder.encoder." in k:  # This is
                print("encoder.encoder")
                new_k = k.replace("encoder.encoder.", "")
                amamba2_state_dict[new_k] = v

        self.amamba2.load_state_dict(amamba2_state_dict, strict=True)
        for n, param in self.amamba2.named_parameters():
            param.requires_grad = False
        # state_dict = torch.load(path, map_location="cpu")["sed_teacher"]
        # amamba2_state_dict = {}
        # for k, v in state_dict.items():
        #     if ("total_ops" in k) or ("total_params" in k):
        #         continue
        #     if "amamba2_frame.amamba2." in k:
        #         k = k.replace("amamba2_frame.amamba2.", "")
        #         amamba2_state_dict[k] = v
        # self.amamba2.load_state_dict(amamba2_state_dict, strict=True)
        # for n, param in amamba2.named_parameters():
        #     param.requires_grad = False