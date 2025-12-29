import os
import torch
from .audio_transformer import FrameASTModel

class trans(torch.nn.Module):
    def __init__(self, trans_path, *args, trans_dropout=0.0, **kwargs, ) -> None:
        super().__init__(*args, **kwargs)
        self.trans = FrameASTModel(trans_dropout=trans_dropout)
        self.load_trans(trans_path)
        self.fake_length = torch.tensor([1001])
        self.cls_embed = None

    def set_cls_embed(self, cls_embed):
        self.cls_embed = cls_embed

    def forward(self, trans_feat, other_emb=None):
        trans_feat = trans_feat.unsqueeze(1)
        trans_x = self.trans.get_intermediate_layers(
            trans_feat,
            self.fake_length.to(trans_feat).repeat(len(trans_feat)),
            1,
            scene=False,
            other_emb=other_emb,
        )
        trans_x = trans_x.transpose(1, 2)
        return trans_x


    def load_trans(self, path=None):
        if path is None:
            pre_path = "../nnet/ckpts/last.ckpt"
            assert os.path.exists(pre_path), "Please make sure you have a default path to load trans. Please change this path to the trans_as2M.ckpt that you downloaded."
            path = pre_path    # Change path to the trans_as2M.ckpt the downloaded checkpoint from the home page.
        state_dict = torch.load(path, map_location="cpu")["state_dict"]
        trans_state_dict = {}
        for k, v in state_dict.items():
            if "model.teacher.encoder." in k:
                if "encoder.norm." in k:
                    new_k = k.replace("model.teacher.encoder.norm", "norm_frame")
                elif "cls_token" in k:
                    continue
                else:
                    new_k = k.replace("model.teacher.encoder.", "")
                trans_state_dict[new_k] = v
            # C2F
            if "encoder.encoder.frame_encoder." in k:
                new_k = k.replace("encoder.encoder.frame_encoder.", "")
                trans_state_dict[new_k] = v
                continue
            if "encoder.encoder.teacher_module." in k:
                continue
            # trans-Frame
            if "encoder.encoder." in k:
                new_k = k.replace("encoder.encoder.", "")
                trans_state_dict[new_k] = v

        self.trans.load_state_dict(trans_state_dict, strict=True)
        for n, param in self.trans.named_parameters():
            param.requires_grad = False
        # state_dict = torch.load(path, map_location="cpu")["sed_teacher"]
        # trans_state_dict = {}
        # for k, v in state_dict.items():
        #     if ("total_ops" in k) or ("total_params" in k):
        #         continue
        #     if "trans_frame.trans." in k:
        #         k = k.replace("trans_frame.trans.", "")
        #         trans_state_dict[k] = v
        # self.trans.load_state_dict(trans_state_dict, strict=True)
        # for n, param in self.trans.named_parameters():
        #     param.requires_grad = False
