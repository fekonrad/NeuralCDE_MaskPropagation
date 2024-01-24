import torch
from torch import nn
from torchdiffeq import odeint

class ConvCDEFunc(nn.Module):
    def __init__(self, in_channels, hidden_channels, kernel_size=3):
        super().__init__()
        self.func = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, kernel_size, padding="same"),
            nn.ReLU(),
            nn.Conv2d(hidden_channels, in_channels, kernel_size, padding="same"),
            nn.Tanh()  # bound output
        )

    def forward(self, x):
        return self.func(x)


class VectorField(nn.Module):
    def __init__(self, spline, func):
        super().__init__()
        self.spline = spline
        self.func = func

    def forward(self, t, x):
        # Since the ODEFunc in 'CT_UNet' will be convolutional, we can't make 'func' return a matrix,
        # hence we'll use '*' instead of '@'
        # interpret this as 'func' returning a diagonal matrix instead of a general matrix
        return self.func(x) * self.spline.deriv(t)


def cdeint(spline, func, z0, times):
    """
      spline: Interpolating spline representing the control path, as in CubicSplineInt
      func: nn.Module representing the ODE function, as in ConvCDEFunc
      z0: Initial state for CDE, shape must match with the input of 'func' and output dimension of 'spline'
      times: 1-D tensor of increasing time points at which the solution is computed
    """
    vector_field = VectorField(spline, func)
    return odeint(vector_field, z0, times)  # uses torchdiffeq's 'odeint' method   (https://github.com/rtqichen/torchdiffeq)


class CT_UNet(nn.Module):
    def __init__(self, num_channels=1):
        super().__init__()

        # encoder
        self.down1 = self.conv_block(num_channels, 64)
        self.down2 = self.conv_block(64, 128)
        self.down3 = self.conv_block(128, 256)
        self.down4 = self.conv_block(256, 512)

        # neural CDE Func
        cde_func = ConvCDEFunc(in_channels=512, hidden_channels=1024)

        # decoder
        self.up1 = self.conv_block(512, 256)
        self.up2 = self.conv_block(256, 128)
        self.up3 = self.conv_block(128, 64)
        self.up4 = self.conv_block(64, num_channels)
        self.readout = nn.Conv2d(num_channels, 1, kernel_size=1, padding="same")

    def conv_block(self, in_channels, out_channels, kernel_size=3):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, padding="same"),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size, padding="same"),
            nn.ReLU()
        )

    def forward(self, x, times):
        """
          - x : Tensor of shape (batch_size, frames, channels, height, width)
          - times : 1-D Tensor with increasing values; denotes time points at which the CDE solution is computed
        """