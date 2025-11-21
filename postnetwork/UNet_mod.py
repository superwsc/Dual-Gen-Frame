
""" Parts of the U-Net model """

import torch
import torch.nn as nn
import torch.nn.functional as F
import os
##########################################################################
class FilmBlock(nn.Module):  #input shape: n, c, h, w
    """Cross-gating MLP block."""
    def __init__(self, cin_x, cin_y, x_out_channels):
        super().__init__()
        self.cin_y = cin_y
        self.cin_x = cin_x
        self.x_out_channels = x_out_channels
        self.Conv_0 = nn.Conv2d(self.cin_x, self.x_out_channels, kernel_size=(3,3), stride=1, padding = 1)
        #self.Conv_1 = nn.Conv2d(self.cin_y, self.x_out_channels, kernel_size=(1,1))
        self.Conv_1 = nn.Linear(self.cin_y, self.x_out_channels)
        self.LayerNorm_x = nn.LayerNorm([x_out_channels])
        self.in_project_x = nn.Linear(self.x_out_channels, self.x_out_channels)
        self.gelu1 = nn.GELU()
        self.LayerNorm_y = nn.LayerNorm([x_out_channels])
        self.in_project_y = nn.Linear(self.x_out_channels, self.x_out_channels, bias=True)     # for feature vector, there's no need to get spatial projection
        self.w_project_y = nn.Linear(self.x_out_channels, self.x_out_channels, bias=True)
        self.b_project_y = nn.Linear(self.x_out_channels, self.x_out_channels, bias=True)
        self.gelu2 = nn.GELU()
        self.out_project_x = nn.Linear(self.x_out_channels, self.x_out_channels, bias=True)
    def forward(self, x,y):     #shape: N,C,H,W
        x = self.Conv_0(x)
        y = self.Conv_1(y)
        y = y.unsqueeze(2).unsqueeze(3)
        y = y.expand_as(x)
        assert y.shape == x.shape
        shortcut_x = x
        # Get gating weights from X
        x = x.permute(0,2,3,1).contiguous()
        x = self.LayerNorm_x(x)
        x = self.in_project_x(x)
        x = self.gelu1(x)
        # Get gating weights from Y
        y = y.permute(0,2,3,1).contiguous()
        y = self.LayerNorm_y(y)
        y = self.in_project_y(y)
        y = self.gelu2(y)
        y_weight = self.w_project_y(y)
        y_bias = self.b_project_y(y)
        #no spatial projection for y
        # Apply cross gating
        x = x * y_weight + y_bias  # gating x using y
        x = self.out_project_x(x)
        x = x.permute(0,3,1,2).contiguous()

        x = x + shortcut_x
        return x


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            #nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            #nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels, hyper=False):
        super().__init__()
        self.hyper = hyper
        if hyper==True:
            self.hypernet = nn.Sequential(nn.Linear(64, in_channels*3),
                                              nn.ReLU(),
                                              nn.Linear(in_channels*3, in_channels*9))
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x, vec):
        if self.hyper==True:
            weight = self.hypernet(vec).reshape(-1, 1, 3, 3)
            B, C, H, W = x.shape
            x = F.conv2d(x.view(1, -1, H, W), weight, groups=B*C, padding=1)
            x = x.view(B, -1, H, W)
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])

        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


# ================================================ UNet model ================================================ #
""" Full assembly of the parts to form the complete network """
class UNet(nn.Module):
    def __init__(self, n_channels=3, n_classes=3, bilinear=False):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.dim = 64   #64
        self.feature_chans = 64     #channel of feature vector

        self.inc = (DoubleConv(n_channels, self.dim))
        self.film1 = FilmBlock(self.dim, self.feature_chans, self.dim)
        self.down1 = (Down(self.dim, self.dim * 2, hyper=True))
        self.film2 = FilmBlock(self.dim * 2, self.feature_chans, self.dim * 2)
        self.down2 = (Down(self.dim * 2, self.dim * 4, hyper=False))
        self.film3 = FilmBlock(self.dim * 4, self.feature_chans, self.dim * 4)
        self.down3 = (Down(self.dim * 4, self.dim * 8, hyper=False))
        factor = 2 if bilinear else 1
        self.film4 = FilmBlock(self.dim * 8, self.feature_chans, self.dim * 8)
        self.down4 = (Down(self.dim * 8, self.dim * 16 // factor, hyper=False))
        self.up1 = (Up(self.dim * 16, self.dim * 8 // factor, bilinear))
        self.up2 = (Up(self.dim * 8, self.dim * 4 // factor, bilinear))
        self.up3 = (Up(self.dim * 4, self.dim * 2 // factor, bilinear))
        self.up4 = (Up(self.dim * 2, self.dim, bilinear))
        self.outc = (OutConv(self.dim, n_classes))

    def forward(self, x, feature_vec):
        vec = feature_vec
        #feature_vec = feature_vec.unsqueeze(2).unsqueeze(3)
        x1 = self.inc(x)

        x1 = self.film1(x1, feature_vec)
        x2 = self.down1(x1, vec)

        x2 = self.film2(x2, feature_vec)
        x3 = self.down2(x2, vec)

        x3 = self.film3(x3, feature_vec)

        x = self.up3(x3, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits


if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = '1'
    from thop import profile
    torch.backends.cudnn.enabled = False
    input = torch.ones(1, 32, 256, 256, dtype=torch.float, requires_grad=False).cuda()
    feature_vec = torch.ones(1, 64, dtype=torch.float, requires_grad=False).cuda()

    model = UNet(n_channels = 32).cuda()

    out = model(input, feature_vec)
    flops, params = profile(model, inputs=(input,feature_vec,))

    print('input shape:', input.shape)
    print('parameters:', params/1e6, 'M')
    print('flops', flops/1e9 , 'G')
    print('output shape', out.shape)