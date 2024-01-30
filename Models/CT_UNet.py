import torch
from torch import nn
from utils.cde_solver import cde_solve
from utils.splines import CubicSplineInt


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


class CT_UNet(nn.Module):
    def __init__(self, num_channels=1, hidden_channels=32, atol=1e-4, rtol=1e-3, method=None):
        super().__init__()
        self.hidden_channels = hidden_channels
        self.atol = atol
        self.rtol = rtol
        self.method = method

        # encoder
        self.down1 = self.conv_block(num_channels, hidden_channels)
        self.down2 = self.conv_block(hidden_channels, 2 * hidden_channels)
        self.down3 = self.conv_block(2 * hidden_channels, 4 * hidden_channels)
        self.down4 = self.conv_block(4 * hidden_channels, 8 * hidden_channels)

        # neural CDE Func
        self.cde_func = ConvCDEFunc(in_channels=8 * hidden_channels, hidden_channels=16 * hidden_channels)

        # decoder
        self.up1 = self.conv_block(16 * hidden_channels, 8 * hidden_channels)
        self.conv_trans1 = nn.ConvTranspose2d(8 * hidden_channels, 4 * hidden_channels, kernel_size=2, stride=2)
        self.up2 = self.conv_block(8 * hidden_channels, 4 * hidden_channels)
        self.conv_trans2 = nn.ConvTranspose2d(4 * hidden_channels, 2 * hidden_channels, kernel_size=2, stride=2)
        self.up3 = self.conv_block(4 * hidden_channels, 2 * hidden_channels)
        self.conv_trans3 = nn.ConvTranspose2d(2 * hidden_channels, hidden_channels, kernel_size=2, stride=2)
        self.up4 = self.conv_block(2 * hidden_channels, hidden_channels)
        self.readout = nn.Conv2d(hidden_channels, 1, kernel_size=1, padding="same")

        # mask encoder and decoder
        self.mask_encoder = self.__create_mask_encoder__()

    def conv_block(self, in_channels, out_channels, kernel_size=3):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, padding="same"),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size, padding="same"),
            nn.ReLU()
        )

    def __create_mask_encoder__(self):
        encoder = nn.Sequential(
            self.conv_block(1, self.hidden_channels),
            nn.MaxPool2d(2),
            self.conv_block(self.hidden_channels, 2 * self.hidden_channels),
            nn.MaxPool2d(2),
            self.conv_block(2 * self.hidden_channels, 4 * self.hidden_channels),
            nn.MaxPool2d(2),
            self.conv_block(4 * self.hidden_channels, 8 * self.hidden_channels),
            nn.MaxPool2d(2),
        )
        return encoder

    def get_size(self):
        return sum(p.numel() for p in self.parameters())

    def forward(self, x, times, y0):
        """
          - x : Tensor of shape (batch_size, frames, channels, height, width)
          - times : Tensor of shape (batch_size, frames) with increasing values in each row; denotes time points at which the CDE solution is computed
          - y0 : Tensor of shape (batch_size, num_channels, height, width) representing initial mask
        """
        # Step 1: Encode sequence and initial mask into the latent space, keep residuals
        b, f, c, h, w = tuple(x.shape)
        z0 = self.mask_encoder(y0)

        x = x.view(-1, c, h, w)  # reshape tensor to make it compatible with Conv2d layers
        res1 = self.down1(x)
        x = nn.MaxPool2d(2)(res1)
        res2 = self.down2(x)
        x = nn.MaxPool2d(2)(res2)
        res3 = self.down3(x)
        x = nn.MaxPool2d(2)(res3)
        res4 = self.down4(x)
        x = nn.MaxPool2d(2)(res4)

        # Step 2: Compute Spline and Solve CDE in the Latent Space at specified time points
        _, latent_c, latent_h, latent_w = x.shape
        x = x.view(b, f, latent_c, latent_h, latent_w)  # convert back to (b,f,c,h,w) shape
        z = torch.empty(b, f, latent_c, latent_h, latent_w)

        for index, video in enumerate(x):
            t = times[index]
            spline = CubicSplineInt(t, video)
            z[index] = cde_solve(spline, self.cde_func, z0[index], t, atol=self.atol, rtol=self.rtol, method=self.method)

        # Step 3: Decode CDE Solution back into the ambient space, add residuals
        x = z.view(-1, latent_c, latent_h, latent_w)  # need to reshape again for Conv2d

        x = nn.Upsample(scale_factor=2)(x)
        x = self.up1(torch.cat([x, res4], dim=1))
        x = self.conv_trans1(x)
        x = self.up2(torch.cat([x, res3], dim=1))
        x = self.conv_trans2(x)
        x = self.up3(torch.cat([x, res2], dim=1))
        x = self.conv_trans3(x)
        x = self.up4(torch.cat([x, res1], dim=1))
        return nn.Sigmoid()(self.readout(x)).view(b, f, 1, h, w)
