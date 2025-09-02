import torch
import pytorch_lightning as pl

from audiotrain.methods.frame.downstream.comparison_models.frame_amamba2 import FrameASM_base, FrameAMamba2LightningModule
from audiotrain.methods.frame.downstream.utils import load_pretrained_weights
from audiotrain.methods.frame.downstream.mamba2 import FreezingTransform


class FrameAMamba2PredModule(pl.LightningModule):
    """This module has been modified for frame-level prediction"""

    def __init__(self, pretrained_ckpt_path, dataset_name="as_strong", **kwargs):
        super().__init__()
        self.encoder = get_frame_amamba2(pretrained_ckpt_path, **kwargs)
        self.embed_dim = self.encoder.embed_dim
        self.transform = FreezingTransform(max_len=10)
        self.last_layer = dataset_name != "as_strong"

    def forward(self, batch):
        (x, length), y = batch
        x = x.unsqueeze(1)
        x = self.encoder.get_intermediate_layers(
            x,
            length,
            1,
            scene=False
        )

        return x, y

    def finetune_mode(self):
        if self.last_layer:
            self.freeze()
            # Unfreeze last tfm block
            for i, layer in enumerate(self.encoder.blocks):
                if i == len(self.encoder.blocks) - 1:
                    for n, p in layer.named_parameters():
                        p.requires_grad = True
            # Unfreeze last norm layer
            for n, p in self.encoder.norm_frame.named_parameters():
                p.requires_grad = True
        else:
            for n, p in self.encoder.named_parameters():
                if "mask_embed" in n:
                    p.requires_grad = False
                else:
                    p.requires_grad = True

    def finetune_mannual_train(self):
        if self.last_layer:
            for i, layer in enumerate(self.encoder.blocks):
                if i == len(self.encoder.blocks) - 1:
                    layer.train()
            self.encoder.norm_frame.train()
        else:
            self.encoder.train()


def get_frame_amamba2(pretrained_ckpt_path, **kwargs):
    # get pretrained encoder
    print("Loading frame-amamba2 model:")
    s = torch.load(pretrained_ckpt_path, map_location="cpu")
    param_names = list(s['state_dict'].keys())
    if not ('model' in param_names[0]):
        pretrained_model = FrameAMamba2LightningModule(**kwargs)
        renamed_state_dict = {}
        for k, v in s['state_dict'].items():
            if "encoder.encoder." in k:
                renamed_state_dict[k.replace("encoder.encoder.", "")] = v
        pretrained_model.model.teacher.encoder.load_state_dict(renamed_state_dict)
    else:
        print("Loading from checkpoint")
        pretrained_model = FrameAMamba2LightningModule.load_from_checkpoint(pretrained_ckpt_path, **kwargs)
    pretrained_encoder = pretrained_model.model.teacher.encoder
    pretrained_encoder.hyper_param = s['hyper_parameters']
    return pretrained_encoder
