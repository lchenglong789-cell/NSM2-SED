import torch.nn as nn
import torch

from VMamba.VMamba_model import VMamba2
from RNN import BidirectionalGRU
from CNN import CNN
# from .SSF_CNN import CNN


class CRNN(nn.Module):
    def __init__(
            self,
            unfreeze_vmamba_layer=0,
            n_in_channel=1,
            nclass=10,
            activation="glu",
            dropout=0.5,
            rnn_type="BGRU",
            n_RNN_cell=128,
            n_layers_RNN=2,
            dropout_recurrent=0,
            embedding_size=768,
            model_init=None,
            vmamba_init=None,
            vmamba_dropout=0.0,
            mode=None,
            **kwargs,
    ):
        super(CRNN, self).__init__()

        self.n_in_channel = n_in_channel
        self.vmamba_dropout = vmamba_dropout
        n_in_cnn = n_in_channel
        self.cnn = CNN(
            n_in_channel=1, activation=activation, conv_dropout=dropout, **kwargs
        )

        if rnn_type == "BGRU":
            nb_in = self.cnn.nb_filters[-1]
            nb_in = nb_in * n_in_channel
            self.rnn = BidirectionalGRU(
                n_in=nb_in,
                n_hidden=n_RNN_cell,
                dropout=dropout_recurrent,
                num_layers=n_layers_RNN,
            )
        else:
            NotImplementedError("Only BGRU supported for CRNN for now")

        self.dropout = nn.Dropout(dropout)
        self.dense = nn.Linear(n_RNN_cell * 2, nclass)
        self.sigmoid = nn.Sigmoid()

        self.dense_softmax = nn.Linear(n_RNN_cell * 2, nclass)
        self.softmax = nn.Softmax(dim=-1)

        self.cat_tf = torch.nn.Linear(896, nb_in)

        self.init_vmamba(vmamba_init)
        self.init_model(model_init, mode=mode)

        self.unfreeze_vmamba_layer = unfreeze_vmamba_layer

    def init_vmamba(self, path=None):
        if path is None:
            self.vmamba_frame = VMamba2(None, vmamba_dropout=self.vmamba_dropout)
        else:
            self.vmamba_frame = VMamba2(path, vmamba_dropout=self.vmamba_dropout)

            print("Loading vmamba from:", path)
        self.vmamba_frame.eval()
        for param in self.vmamba_frame.parameters():
            param.detach_()

    def init_model(self, path, mode=None):
        if path is None:
            pass
        else:
            if mode == "teacher":
                print("Loading teacher from:", path)
                state_dict = torch.load(path, map_location="cpu")["sed_teacher"]
            else:
                print("Loading student from:", path)
                state_dict = torch.load(path, map_location="cpu")["sed_student"]
            self.load_state_dict(state_dict, strict=True)
            print("Model loaded")

    def forward(self, x, pretrain_x, pad_mask=None, embeddings=None):
        x = x.transpose(1, 2).unsqueeze(1)
        # conv featuress
        x = self.cnn(x)
        bs, chan, frames, freq = x.size()
        # x = x.squeeze(-1)
        # x = x.permute(0, 2, 1)  # [bs, frames, chan] torch.Size([96, 156, 128])
        # print("cnn Ok", x.shape)

        B, C, T, K = x.shape
        x = x.reshape(B, T, C * K)
        # print("cnn Ok", x.shape)

        # rnn features
        embeddings = self.vmamba_frame(pretrain_x)
        # print("embeddings Ok", embeddings.shape)
        if embeddings.shape[-1] != x.shape[1]:
            embeddings = torch.nn.functional.adaptive_avg_pool1d(embeddings, x.shape[1]).transpose(1, 2)
        else:
            embeddings = embeddings.transpose(1, 2)
        x = self.cat_tf(torch.cat((x, embeddings), -1))

        x = self.rnn(x)

        x = self.dropout(x)
        strong = self.dense(x)  # [bs, frames, nclass]
        strong = self.sigmoid(strong)
        sof = self.dense_softmax(x)  # [bs, frames, nclass]
        sof = self.softmax(sof)
        sof = torch.clamp(sof, min=1e-7, max=1)
        weak = (strong * sof).sum(1) / sof.sum(1)  # [bs, nclass]

        return strong.transpose(1, 2), weak

    def train(self, mode=True):
        """
        Override the default train() to freeze the BN parameters
        """
        super(CRNN, self).train(mode)

        # we comments this part because there are only two stage in the fine-tuning;
        # either eval the Frame or train the whole model
        # if mode:
        #     self.vmamba_frame.eval()
        #     if self.unfreeze_vmamba_layer > 0:
        #         self.vmamba_frame.vmamba.norm_frame.train()
        #         if self.unfreeze_vmamba_layer == 14:
        #             self.vmamba_frame.train()
        #         else:
        #             unfreeze_blocks = self.unfreeze_vmamba_layer - 1
        #             for i in range(unfreeze_blocks):
        #                 block_idx = 11 - i
        #                 self.vmamba_frame.vmamba.blocks[block_idx].train()
