import torch.nn as nn
import torch


class GCN(nn.Module):
    def __init__(self, input_channels, kernel_size):  # kernel_size = 3/5/7
        super(GCN, self).__init__()
        padding = (kernel_size - 1) // 2
        self.conv1 = nn.Conv2d(input_channels, input_channels, kernel_size=(1, kernel_size), padding=(0, padding))
        self.conv2 = nn.Conv2d(input_channels, input_channels, kernel_size=(kernel_size, 1), padding=(padding, 0))
        self.bn = nn.BatchNorm2d(input_channels)
        self.relu = nn.ReLU()

    def forward(self, input):
        output = self.conv2(self.conv1(input)) + self.conv1(self.conv2(input))
        # output = torch.cat((output1, output2), dim=1)
        output = self.relu(self.bn(output))
        # print('output', output.shape)
        return output


class SSF_module(nn.Module):
    def __init__(self, input_channels=1, height=626, width=128):
        super(SSF_module, self).__init__()

        self.gcn1 = GCN(input_channels, kernel_size=3)
        self.gcn2 = GCN(input_channels * 2, kernel_size=5)
        self.gcn3 = GCN(input_channels * 4, kernel_size=7)

        # self.gap = nn.AdaptiveAvgPool2d((height, width))
        self.conv1 = nn.Conv2d(input_channels, input_channels, kernel_size=1)
        self.conv2 = nn.Conv2d(input_channels * 2, input_channels * 2, kernel_size=1)
        self.conv3 = nn.Conv2d(input_channels * 4, input_channels * 4, kernel_size=1)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        # x = x.unsqueeze(1)
        B, C, H, W = x.shape
        gcn1 = self.gcn1(x)  # [16, 1, 64, 1001]
        gcn2 = self.gcn2(torch.cat((x, gcn1), dim=1))  # [16, 2, 64, 1001]
        gcn3 = self.gcn3(torch.cat((x, gcn1, gcn2), dim=1))  # [16, 4, 64, 1001]
        gcn = torch.cat((gcn1, gcn2, gcn3), dim=1)  # [16, 7, 64, 1001]

        gap1 = self.conv1(gcn1)  # [16, 1, 64, 1001]
        gap2 = self.conv2(gcn2)  # [16, 2, 64, 1001]
        gap3 = self.conv3(gcn3)  # [16, 4, 64, 1001]
        gap = torch.cat((gap1, gap2, gap3), dim=1)  # [16, 7, 64, 1001]
        gap = self.softmax(gap)

        output = torch.mul(gcn, gap)
        # print("Selective Feature Fusion Module", output.shape)
        return output


class GLU(nn.Module):
    def __init__(self, input_num):
        super(GLU, self).__init__()
        self.sigmoid = nn.Sigmoid()
        self.linear = nn.Linear(input_num, input_num)

    def forward(self, x):
        lin = self.linear(x.permute(0, 2, 3, 1))
        lin = lin.permute(0, 3, 1, 2)
        sig = self.sigmoid(x)
        res = lin * sig
        return res


class ContextGating(nn.Module):
    def __init__(self, input_num):
        super(ContextGating, self).__init__()
        self.sigmoid = nn.Sigmoid()
        self.linear = nn.Linear(input_num, input_num)

    def forward(self, x):
        lin = self.linear(x.permute(0, 2, 3, 1))
        lin = lin.permute(0, 3, 1, 2)
        sig = self.sigmoid(lin)
        res = x * sig
        return res


class CNN(nn.Module):
    def __init__(
            self,
            n_in_channel,
            activation="Relu",
            conv_dropout=0,
            kernel_size=[3, 3, 3],
            padding=[1, 1, 1],
            stride=[1, 1, 1],
            nb_filters=[64, 64, 64],
            pooling=[(1, 4), (1, 4), (1, 4)],
            normalization="batch",
            **transformer_kwargs
    ):
        """
            Initialization of CNN network s

        Args:
            n_in_channel: int, number of input channel
            activation: str, activation function
            conv_dropout: float, dropout
            kernel_size: kernel size
            padding: padding
            stride: list, stride
            nb_filters: number of filters
            pooling: list of tuples, time and frequency pooling
            normalization: choose between "batch" for BatchNormalization and "layer" for LayerNormalization.
        """
        super(CNN, self).__init__()

        self.nb_filters = nb_filters
        cnn = nn.Sequential()

        def conv(i, normalization="batch", dropout=None, activ="relu"):
            nIn = n_in_channel if i == 0 else nb_filters[i - 1]
            nOut = nb_filters[i]
            cnn.add_module(
                "conv{0}".format(i),
                nn.Conv2d(nIn, nOut, kernel_size[i], stride[i], padding[i]),
            )
            if normalization == "batch":
                cnn.add_module(
                    "batchnorm{0}".format(i),
                    nn.BatchNorm2d(nOut, eps=0.001, momentum=0.99),
                )
            elif normalization == "layer":
                cnn.add_module("layernorm{0}".format(i), nn.GroupNorm(1, nOut))

            if activ.lower() == "leakyrelu":
                cnn.add_module("relu{0}".format(i), nn.LeakyReLU(0.2))
            elif activ.lower() == "relu":
                cnn.add_module("relu{0}".format(i), nn.ReLU())
            elif activ.lower() == "glu":
                cnn.add_module("glu{0}".format(i), GLU(nOut))
            elif activ.lower() == "cg":
                cnn.add_module("cg{0}".format(i), ContextGating(nOut))

            if dropout is not None:
                cnn.add_module("dropout{0}".format(i), nn.Dropout(dropout))

        # 128x862x64
        for i in range(len(nb_filters)):
            conv(i, normalization=normalization, dropout=conv_dropout, activ=activation)
            cnn.add_module(
                "pooling{0}".format(i), nn.AvgPool2d(pooling[i])
            )  # bs x tframe x mels

        self.ssf = SSF_module()
        self.cnn = cnn

    def forward(self, x):
        """
        Forward step of the CNN module

        Args:
            x (Tensor): input batch of size (batch_size, n_channels, n_frames, n_freq)

        Returns:
            Tensor: batch embedded
        """
        # conv features
        x = self.ssf(x)
        x = self.cnn(x)
        return x


if __name__ == "__main__":
    input_x = torch.randn(16, 1, 626, 128)

    model = CNN(n_in_channel=7, activation='cg')

    output_y = model(input_x)
    print('input_x:', input_x.shape)
    print('output_y:', output_y.shape)
