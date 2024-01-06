import torch
from torch import nn


"""
This implementation is partially based on Kidger's code for 'Neural Controlled Differential Equations for Irregular Time Series',
See https://github.com/patrick-kidger/NeuralCDE
"""


class CDEFunc(nn.Module):               # Neural CDE for Image Data
    def __init__(self, channels=3):
        super().__init__()

        self.down1 = self.conv_block(in_channels=channels, out_channels=4)
        self.down2 = self.conv_block(in_channels=4, out_channels=8)
        self.down3 = self.conv_block(in_channels=8, out_channels=16)

        self.conv_trans1 = nn.ConvTranspose2d(16, 8, kernel_size=2, stride=2, padding=0)
        self.up1 = self.conv_block(16, 8)
        self.conv_trans2 = nn.ConvTranspose2d(8, 4, kernel_size=2, stride=2, padding=0)
        self.up2 = self.conv_block(8, 4)
        self.readout = nn.Conv2d(4, 1, kernel_size=3, padding="same")

    def conv_block(self, in_channels, out_channels, kernel_size=3):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, padding="same"),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size, padding="same"),
            nn.ReLU()
        )

    def get_size(self):
        # returns the number of parameters in the network
        return sum(p.numel() for p in self.parameters())

    def forward(self, x):
        # encoder
        res1 = self.down1(x)
        x = nn.MaxPool2d(kernel_size=2)(res1)
        res2 = self.down2(x)
        x = nn.MaxPool2d(kernel_size=2)(res2)
        x = self.down3(x)

        # decoder
        x = self.conv_trans1(x)
        x = self.up1(torch.cat([res2, x], dim=1))
        x = self.conv_trans2(x)
        x = self.up2(torch.cat([res1, x], dim=1))
        return nn.Tanh(self.readout(x))           # tanh to bound the output


class NeuralCDE(nn.Module):
    def __init__(self, cdefunc):
        super(NeuralCDE, self).__init__()
        self.func = cdefunc

    def forward(self, t, x, spline):
        """
        :param x: ???
        :param spline: interpolating spline of data
        :return:
        """
        return self.func(x) @ spline.eval(t)
