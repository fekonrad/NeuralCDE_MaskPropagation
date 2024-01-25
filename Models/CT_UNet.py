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
    self.cde_func = ConvCDEFunc(in_channels=512, hidden_channels=1024)

    # decoder
    self.up1 = self.conv_block(512, 256)
    self.up2 = self.conv_block(256, 128)
    self.up3 = self.conv_block(128, 64)
    self.up4 = self.conv_block(64, num_channels)
    self.readout = nn.Conv2d(num_channels, 1, kernel_size=1, padding="same")

    # mask encoder and decoder 
    self.mask_encoder = self.__create_mask_encoder__()

  def conv_block(self,in_channels, out_channels, kernel_size=3):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size, padding="same"),
        nn.ReLU(),
        nn.Conv2d(out_channels, out_channels, kernel_size, padding="same"),
        nn.ReLU()
    )

  def __create_mask_encoder__(self): 
    encoder = nn.Sequential(
        self.conv_block(1, 64), 
        self.conv_block(64, 128), 
        self.conv_block(128, 256) 
    )
    return encoder 

  def get_size(self):
    return sum(p.numel() for p in self.parameters())

  def forward(self, x, times, y0):
    """
      - x : Tensor of shape (batch_size, frames, channels, height, width)
      - times : 1-D Tensor with increasing values; denotes time points at which the CDE solution is computed
      - y0 : Tensor of shape (1, height, width) representing initial mask
    """

    # TODO: Need to reshape 'x' so that it's shape is compatible with 'nn.Conv2d' !
    # (batch_size, frames, channels, height, width) --> (batch_size*frames, channels, height, width)

    # Step 1: Encode sequence and initial mask into the latent space, keep residuals 
    z0 = self.mask_encoder(y0) 

    res1 = self.down1(x)
    x = nn.MaxPool2d(2)(res1)
    res2 = self.down2(x)
    x = nn.MaxPool2d(2)(res2)
    res3 = self.down3(x)
    x = nn.MaxPool2d(2)(res3)
    res4 = self.down4(x)
    x = nn.MaxPool2d(2)(res4)

    # Step 2: Compute Spline and Solve CDE in the Latent Space at specified time points 
    spline = CubicSplineInt(times, x)
    z = cde_solve(spline, self.cde_func, z0, times)

    # Step 3: Decode CDE Solution back into the ambient space, add residuals 
    x = nn.Upsample2d(2)(z)
    x = self.up1(torch.cat([x,res4]), dim=1)
    x = nn.Upsample2d(2)(x)
    x = self.up2(torch.cat([x,res3]), dim=1)    
    x = nn.Upsample2d(2)(x)
    x = self.up3(torch.cat([x,res2]), dim=1)    
    x = nn.Upsample2d(2)(x)
    x = self.up4(torch.cat([x,res1]), dim=1)
    return x 
