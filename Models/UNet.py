import torch
from torch import nn


class UNet(nn.Module):
    def __init__(self, num_channels=64):
        """
        :param num_channels:
        :param F_int: Number of channels for the 1d convolution in the attention gate
        """
        super().__init__()  # call constructor of base class nn.Module

        # encoder block (without down-sampling)
        self.first = self.first_block(3,  num_channels)
        self.down1 = self.conv_block(num_channels, 2*num_channels)
        self.down2 = self.conv_block(2*num_channels, 4*num_channels)
        self.down3 = self.conv_block(4*num_channels, 8*num_channels)
        self.down4 = self.conv_block(8*num_channels, 16*num_channels)

        # decoder block
        self.attention_gate1 = AttentionGate(16*num_channels, 8*num_channels, 8*num_channels)
        self.conv_trans1 = nn.ConvTranspose2d(16*num_channels, 8*num_channels, kernel_size=2, stride=2, padding=0)
        self.up1 = self.conv_block(16*num_channels, 8*num_channels)
        self.attention_gate2 = AttentionGate(8*num_channels, 4*num_channels, 4*num_channels)
        self.conv_trans2 = nn.ConvTranspose2d(8*num_channels, 4*num_channels, kernel_size=2, stride=2, padding=0)
        self.up2 = self.conv_block(8*num_channels, 4*num_channels)
        self.attention_gate3 = AttentionGate(4*num_channels, 2*num_channels, 2*num_channels)
        self.conv_trans3 = nn.ConvTranspose2d(4*num_channels, 2*num_channels, kernel_size=2, stride=2, padding=0)
        self.up3 = self.conv_block(4*num_channels, 2*num_channels)
        self.attention_gate4 = AttentionGate(2*num_channels, num_channels, num_channels)
        self.conv_trans4 = nn.ConvTranspose2d(2*num_channels, num_channels, kernel_size=2, stride=2, padding=0)
        self.up4 = self.conv_block(2*num_channels, num_channels)
        self.last = nn.Conv2d(num_channels, 1, kernel_size=1, padding="same")

    def first_block(self, in_channels, out_channels, kernel_size=3):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, padding="same"),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size, padding="same"),
            nn.ReLU()
        )

    def conv_block(self,in_channels, out_channels, kernel_size=3):
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
        """ Encoder Block with residuals """
        res1 = self.first(x)
        x = nn.MaxPool2d(2)(res1)
        res2 = self.down1(x)
        x = nn.MaxPool2d(2)(res2)
        res3 = self.down2(x)
        x = nn.MaxPool2d(2)(res3)
        res4 = self.down3(x)
        x = nn.MaxPool2d(2)(res4)
        x = self.down4(x)

        """ Decoder Block """
        res4 = self.attention_gate1(x, res4)
        x = self.conv_trans1(x)
        x = self.up1(torch.cat((res4, x), dim=1))
        res3 = self.attention_gate2(x, res3)
        x = self.conv_trans2(x)
        x = self.up2(torch.cat((res3, x), dim=1))
        res2 = self.attention_gate3(x, res2)
        x = self.conv_trans3(x)
        x = self.up3(torch.cat((res2, x), dim=1))
        res1 = self.attention_gate4(res1)
        x = self.conv_trans4(x)
        x = self.up4(torch.cat((res1, x), dim=1))
        x = self.last(x)
        return nn.Sigmoid()(x)


class AttentionGate(nn.Module):
    def __init__(self, in_channels, gate_channels, F_int):
        super().__init__()
        self.W_g = nn.Conv2d(in_channels=gate_channels, out_channels=F_int, kernel_size=1)
        self.W_x = nn.Conv2d(in_channels=in_channels, out_channels=F_int, kernel_size=1)
        self.psi = nn.Conv2d(in_channels=F_int, out_channels=1, kernel_size=1)

    def forward(self, input_, gating_signal):
        gating_signal = self.W_g(gating_signal)
        x = self.W_x(input_)
        x = nn.ReLU()(x + gating_signal)
        x = self.psi(x)
        alpha = nn.Sigmoid()(x)
        return alpha*x


class DiceLoss(nn.Module):
    def __init__(self, smooth=1.0):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, predictions, targets):
        predictions = predictions.view(-1)
        targets = targets.view(-1)

        intersection = (predictions * targets).sum()
        union = predictions.sum() + targets.sum()

        dice_coefficient = (2.0 * intersection + self.smooth) / (union + self.smooth)
        dice_loss = 1.0 - dice_coefficient

        return dice_loss


class EarlyStopping:
    """
    Implementation based on https://stackoverflow.com/questions/71998978/early-stopping-in-pytorch
    """
    def __init__(self, patience=1, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = float('inf')

    def early_stop(self, validation_loss):
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True 
        return False

