import torch
from torch import nn
import torch.nn.functional as F
import einops
import numpy as np
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
import math

class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
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

class Conv(nn.Module):
    def __init__(self, nIn, nOut, kSize, stride, padding, dilation=(1, 1), groups=1, bn_acti=False, bias=False):
        super().__init__()

        self.bn_acti = bn_acti

        self.conv = nn.Conv2d(nIn, nOut, kernel_size=kSize,
                              stride=stride, padding=padding,
                              dilation=dilation, groups=groups, bias=bias)

        if self.bn_acti:
            self.bn_relu = nn.ReLU(nOut)

    def forward(self, input):
        output = self.conv(input)

        if self.bn_acti:
            output = self.bn_relu(output)

        return output

class UNet_2D(nn.Module):
    def __init__(self, n_channels, n_classes, basic_chans = 16, bilinear=False):
        super(UNet_2D, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        self.basic_channel = basic_chans

        self.inc = DoubleConv(n_channels, self.basic_channel*4)
        self.down1 = Down(self.basic_channel * 4, self.basic_channel*8)
        self.down2 = Down(self.basic_channel * 8, self.basic_channel*16)
        self.down3 = Down(self.basic_channel * 16, self.basic_channel * 32)
        factor = 2 if bilinear else 1
        self.down4 = Down(self.basic_channel * 32, self.basic_channel * 64 // factor)
        self.up1 = Up(self.basic_channel * 64, self.basic_channel * 32 // factor, bilinear)
        self.up2 = Up(self.basic_channel * 32, self.basic_channel * 16 // factor, bilinear)
        self.up3 = Up(self.basic_channel * 16, self.basic_channel * 8 // factor, bilinear)
        self.up4 = Up(self.basic_channel * 8, self.basic_channel * 4, bilinear)
        self.outc = OutConv(self.basic_channel * 4, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        out = self.outc(x)
        return out

def block_images_einops(x, patch_size):
  """Image to patches."""
  batch, height, width, channels = x.shape
  grid_height = height // patch_size[0]
  grid_width = width // patch_size[1]
  x = einops.rearrange(
      x, "n (gh fh) (gw fw) c -> n (gh gw) (fh fw) c", # 现将基于 fh,fw 窗口 大小图片拉成一维，共有 ghxgw 个
      gh=grid_height, gw=grid_width, fh=patch_size[0], fw=patch_size[1]) # 将维度变为 n (gh gw) (fh fw) c
  return x


def unblock_images_einops(x, grid_size, patch_size):
  """patches to images."""
  x = einops.rearrange(
      x, "n (gh gw) (fh fw) c -> n (gh fh) (gw fw) c",
      gh=grid_size[0], gw=grid_size[1], fh=patch_size[0], fw=patch_size[1])
  return x


# MFI
class GetSpatialGatingWeights_2D_Multi_Scale_Cascade_Grid(nn.Module): # 将输入 channel -> channel/2 ，u计算grid/v计算block 再concat
    """Get gating weights for cross-gating MLP block."""
    def __init__(self, nIn,Nout, H_size=128, W_size=128, input_proj_factor=2, dropout_rate=0.0, use_bias=True, train_size=512):
        super(GetSpatialGatingWeights_2D_Multi_Scale_Cascade_Grid, self).__init__()
        self.H = H_size
        self.W = W_size
        self.IN = nIn
        self.OUT = Nout
        if train_size == 512:
            self.grid_size = [[16, 16], [8, 8], [2, 2]]
        else:
            self.grid_size = [[6, 6], [3, 3], [2, 2]]

        self.block_size = [[int(H_size / l[0]), int(W_size / l[1])] for l in self.grid_size]
        self.input_proj_factor = input_proj_factor # 控制将输入 映射到多维，达到扩大channel 的目的.
        self.dropout_rate = dropout_rate
        self.use_bias = use_bias
        self.dropout = nn.Dropout(self.dropout_rate)
        self.LayerNorm = nn.LayerNorm(self.IN)
        self.Linear_end = nn.Linear(self.IN,self.OUT)
        self.Gelu = nn.GELU()
        self.Linear_grid_MLP_1 = nn.Linear((self.grid_size[0][0] * self.grid_size[0][1]),
                                           (self.grid_size[0][0] * self.grid_size[0][1]), bias=use_bias)

        self.Linear_Block_MLP_1 = nn.Linear((self.block_size[0][0] * self.block_size[0][1]),
                                            (self.block_size[0][0] * self.block_size[0][1]), bias=use_bias)

        self.Linear_grid_MLP_2 = nn.Linear((self.grid_size[1][0] * self.grid_size[1][1]),
                                           (self.grid_size[1][0] * self.grid_size[1][1]), bias=use_bias)

        self.Linear_Block_MLP_2 = nn.Linear((self.block_size[1][0] * self.block_size[1][1]),
                                            (self.block_size[1][0] * self.block_size[1][1]), bias=use_bias)

        self.Linear_grid_MLP_3 = nn.Linear((self.grid_size[2][0] * self.grid_size[2][1]),
                                           (self.grid_size[2][0] * self.grid_size[2][1]), bias=use_bias)

        self.Linear_Block_MLP_3 = nn.Linear((self.block_size[2][0] * self.block_size[2][1]),
                                            (self.block_size[2][0] * self.block_size[2][1]), bias=use_bias)
        self.conv = nn.Conv2d(self.IN * 2, self.OUT, kernel_size=3, padding=1, bias=False)

    def forward(self, x): # 去掉原 deterministic drop 后 未加 mask。
        n, h, w,num_channels = x.shape
        # n x h x w x c
        # input projection
        x = self.LayerNorm(x.float()) # 没有 float 报错
        x = self.Gelu(x)

        # grid 和 block 的大小都根据 给定的 grid_size or block_size 自动匹配另一个大小。即 grid_size 给定，自动计算 block_size
        # Get grid MLP weights
        gh1, gw1 = self.grid_size[0]
        fh1, fw1 = h // gh1, w // gw1
        u1 = block_images_einops(x, patch_size=(fh1, fw1)) # 得到 B (gh gw) (fh fw) c 即：ghxgw 个 fhxfw.
        # 此函数只需要 fh,fw 得到 gh 和gw 是方便 unblock_images_einops 使用
        u1 = u1.permute(0,3,2,1)

        u1 = self.Linear_grid_MLP_1(u1)
        u1 = u1.permute(0,3,2,1)
        u1 = unblock_images_einops(u1, grid_size=(gh1, gw1), patch_size=(fh1, fw1))

        fh1, fw1 = self.block_size[0]
        gh1, gw1 = h // fh1, w // fw1
        v1 = block_images_einops(x, patch_size=(fh1, fw1))
        v1 = v1.permute(0, 1, 3, 2)
        v1 = self.Linear_Block_MLP_1(v1)
        v1 = v1.permute(0, 1, 3, 2)
        v1 = unblock_images_einops(v1, grid_size=(gh1, gw1), patch_size=(fh1, fw1))

        gh2, gw2 = self.grid_size[1]
        fh2, fw2 = h // gh2, w // gw2
        u2 = block_images_einops(u1, patch_size=(fh2, fw2))  # 得到 B (gh gw) (fh fw) c 即：ghxgw 个 fhxfw.
        # 此函数只需要 fh,fw 得到 gh 和gw 是方便 unblock_images_einops 使用
        u2 = u2.permute(0, 3, 2, 1)

        u2 = self.Linear_grid_MLP_2(u2)
        u2 = u2.permute(0, 3, 2, 1)
        u2 = unblock_images_einops(u2, grid_size=(gh2, gw2), patch_size=(fh2, fw2))

        fh2, fw2 = self.block_size[1]
        gh2, gw2 = h // fh2, w // fw2
        v2 = block_images_einops(v1, patch_size=(fh2, fw2))
        v2 = v2.permute(0, 1, 3, 2)
        v2 = self.Linear_Block_MLP_2(v2)
        v2 = v2.permute(0, 1, 3, 2)
        v2 = unblock_images_einops(v2, grid_size=(gh2, gw2), patch_size=(fh2, fw2))

        gh3, gw3 = self.grid_size[2]
        fh3, fw3 = h // gh3, w // gw3
        u3 = block_images_einops(u2, patch_size=(fh3, fw3))  # 得到 B (gh gw) (fh fw) c 即：ghxgw 个 fhxfw.
        # 此函数只需要 fh,fw 得到 gh 和gw 是方便 unblock_images_einops 使用
        u3 = u3.permute(0, 3, 2, 1)

        u3 = self.Linear_grid_MLP_3(u3)
        u3 = u3.permute(0, 3, 2, 1)
        u3 = unblock_images_einops(u3, grid_size=(gh3, gw3), patch_size=(fh3, fw3))

        fh3, fw3 = self.block_size[2]
        gh3, gw3 = h // fh3, w // fw3
        v3 = block_images_einops(v2, patch_size=(fh3, fw3))
        v3 = v3.permute(0, 1, 3, 2)
        v3 = self.Linear_Block_MLP_3(v3)
        v3 = v3.permute(0, 1, 3, 2)
        v3 = unblock_images_einops(v3, grid_size=(gh3, gw3), patch_size=(fh3, fw3))

        vx = self.Linear_end(v3)
        vx = self.dropout(vx)

        ux = self.Linear_end(u3)
        ux = self.dropout(ux)

        x = torch.cat([vx, ux], dim=3)
        x = self.conv(x.permute(0, 3, 1, 2))
        x = x.permute(0, 2, 3, 1)

        return x # 不改变维度。# n h w c


class conv_T_y_2_x(nn.Module):
    """ Unified y Dimensional to x """
    def __init__(self,y_nIn,x_nOut):
        super(conv_T_y_2_x, self).__init__()
        self.x_c = x_nOut
        self.y_c = y_nIn
        self.convT = nn.ConvTranspose2d(in_channels=self.y_c, out_channels=self.x_c, kernel_size=(3,3),
                                        stride=(2, 2))
    def forward(self,x,y):
        # 考虑通用性，先将维度变为一致，在采样到相同大小
        y = self.convT(y)
        _, _, h, w, = x.shape
        y = nn.Upsample(size=(h, w), mode='bilinear', align_corners=True)(y)
        return y

#TODO CGB
class CrossGatingBlock(nn.Module):
    """Cross-gating MLP block."""

    def __init__(self, x_in, y_in, out_features, patch_size, block_size, grid_size,
                 dropout_rate=0.0, input_proj_factor=2, upsample_y=True, use_bias=True, train_size=512):

        super(CrossGatingBlock, self).__init__()
        self.IN_x = x_in
        self.IN_y = y_in
        self._h = patch_size[0]
        self._w = patch_size[1]
        self.features = out_features
        self.block_size = block_size
        self.grid_size = grid_size
        self.dropout_rate = dropout_rate
        self.input_proj_factor = input_proj_factor
        self.upsample_y = upsample_y
        self.use_bias = use_bias
        self.Conv1X1_x = nn.Conv2d(self.IN_x, self.features,(1,1))
        self.Conv1X1_y = nn.Conv2d(self.IN_x, self.features,(1,1))
        self.LayerNorm_x = nn.LayerNorm(self.features)
        self.LayerNorm_y = nn.LayerNorm(self.features)
        self.Linear_x = nn.Linear(self.features, self.features,bias=use_bias)
        self.Linear_y = nn.Linear(self.features, self.features,bias=use_bias)
        self.Gelu_x = nn.GELU()
        self.Gelu_y = nn.GELU()
        self.Linear_x_end = nn.Linear(self.features, self.features,bias=use_bias)
        self.Linear_y_end = nn.Linear(self.features, self.features,bias=use_bias)
        self.dropout_x = nn.Dropout(self.dropout_rate)
        self.dropout_y = nn.Dropout(self.dropout_rate)

        self.ConvT = conv_T_y_2_x(self.IN_y, self.IN_x)
        self.fun_gx = GetSpatialGatingWeights_2D_Multi_Scale_Cascade_Grid(nIn=self.features, Nout=self.features, H_size=self._h, W_size=self._w,
                                                 input_proj_factor=2, dropout_rate=dropout_rate, use_bias=True, train_size=train_size)

        self.fun_gy = GetSpatialGatingWeights_2D_Multi_Scale_Cascade_Grid(nIn=self.features, Nout=self.features, H_size=self._h, W_size=self._w,
                                                 input_proj_factor=2, dropout_rate=dropout_rate, use_bias=True, train_size=train_size)

    def forward(self, x, y):
    # Upscale Y signal, y is the gating signal.
        if self.upsample_y:
            # 将 y 的维度调整为与 x 相同大小
            y = self.ConvT(x,y) # nn.ConvTranspose 反卷积

        x = self.Conv1X1_x(x)
        y = self.Conv1X1_y(y)
        assert y.shape == x.shape
        x = x.permute(0, 2, 3, 1)  # n x h x w x c
        y = y.permute(0, 2, 3, 1)
        shortcut_x = x
        shortcut_y = y
        # Get gating weights from X
        x = self.LayerNorm_x(x)
        x = self.Linear_x(x)
        x = self.Gelu_x(x)

        gx = self.fun_gx(x)
        # n x h x w x c
        # Get gating weights from Y
        y = self.LayerNorm_y(y)
        y = self.Linear_y(y)
        y = self.Gelu_y(y)

        gy = self.fun_gy(y)

        y = y * gx
        y = self.Linear_y_end(y)
        y = self.dropout_y(y)
        y = y + shortcut_y
        x = x * gy  # gating x using y
        x = self.Linear_y_end(x)
        x = self.dropout_x(x)
        x = x + y + shortcut_x  # get all aggregated signals # 注意此处的 x 融合了来自y 的信息。
        x = x.permute(0, 3, 1, 2)  # n x h x w x c --> n x c x h x w
        y = y.permute(0, 3, 1, 2)
        return x, y


class FusionUp(nn.Module):
    """Upscaling then double conv"""

    def  __init__(self, in_channels, out_channels, bilinear=True):
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


class UNet_MLP(nn.Module):
    def __init__(self, n_channels, n_classes, basic_chans = 16, bilinear=True):
        super(UNet_MLP, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        self.basic_channel = basic_chans

        self.inc = DoubleConv(n_channels, self.basic_channel*4)
        self.down1 = Down(self.basic_channel * 4, self.basic_channel * 8)
        self.down2 = Down(self.basic_channel * 8, self.basic_channel * 16)
        self.down3 = Down(self.basic_channel * 16, self.basic_channel * 32)
        factor = 2 if bilinear else 1
        self.down4 = Down(self.basic_channel * 32, self.basic_channel * 64 // factor)

        resolution = 512
        res1 = resolution / 8
        res2 = res1 / 2
        res3 = res2 / 2

        #Multi-scale MLP
        self.up1 = CrossGatingBlock(self.basic_channel * 16, self.basic_channel * 16, self.basic_channel * 16, [res1, res1], [8, 8], [4, 4], 0.1, upsample_y=False,
                                    train_size=resolution)
        self.up2 = CrossGatingBlock(16, 16, 16, [res2, res2], [4, 4], [4, 4], 0.1, upsample_y=False,
                                    train_size=resolution)
        self.up3 = CrossGatingBlock(16, 16, 16, [res3, res3], [2, 2], [2, 2], 0.1, upsample_y=False,
                                    train_size=resolution)
        self.up4 = CrossGatingBlock(16, 16, 16, [res3, res3], [2, 2], [2, 2], 0.1, upsample_y=False,
                                    train_size=resolution)

        self.fuseup1 = FusionUp(self.basic_channel * 64, self.basic_channel * 32 // factor, bilinear)
        self.fuseup2 = FusionUp(self.basic_channel * 32, self.basic_channel * 16 // factor, bilinear)
        self.fuseup3 = FusionUp(self.basic_channel * 16, self.basic_channel * 8 // factor, bilinear)
        self.fuseup4 = FusionUp(self.basic_channel * 8, self.basic_channel * 4, bilinear)
        self.outc = OutConv(self.basic_channel * 4, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        x5_1, x5_2 = torch.chunk(x5, 2, dim=1)
        x5_out1, x5_out2 = self.up1(x5_1, x5_2)  # 32 --> 32
        mfi_out_1 = torch.cat([x5_out1, x5_out2], dim=1)
        x = self.fuseup1(mfi_out_1, x4)

        x = self.fuseup2(x, x3)
        x = self.fuseup3(x, x2)
        x = self.fuseup4(x, x1)
        out = self.outc(x)
        return out


# global mlp
class Multi_Scale_globalSpatialGatingWeights(nn.Module): # 将输入 channel -> channel/2 ，u计算grid/v计算block 再concat
    """Get gating weights for cross-gating MLP block."""
    def __init__(self, nIn,Nout, H_size=128, W_size=128, input_proj_factor=2, dropout_rate=0.0, use_bias=True, train_size=512):
        super(Multi_Scale_globalSpatialGatingWeights, self).__init__()
        self.H = H_size
        self.W = W_size
        self.IN = nIn
        self.OUT = Nout
        if train_size == 512:
            self.grid_size = [[16, 16], [8, 8], [2, 2]]
        else:
            self.grid_size = [[6, 6], [3, 3], [2, 2]]

        self.block_size = [[int(H_size / l[0]), int(W_size / l[1])] for l in self.grid_size]
        self.input_proj_factor = input_proj_factor # 控制将输入 映射到多维，达到扩大channel 的目的.
        self.dropout_rate = dropout_rate
        self.use_bias = use_bias
        self.dropout = nn.Dropout(self.dropout_rate)
        self.LayerNorm = nn.LayerNorm(self.IN)
        self.Linear_end = nn.Linear(self.IN,self.OUT)
        self.Gelu = nn.GELU()
        self.Linear_grid_MLP_1 = nn.Linear((self.grid_size[0][0] * self.grid_size[0][1]),
                                           (self.grid_size[0][0] * self.grid_size[0][1]), bias=use_bias)

        self.Linear_grid_MLP_2 = nn.Linear((self.grid_size[1][0] * self.grid_size[1][1]),
                                           (self.grid_size[1][0] * self.grid_size[1][1]), bias=use_bias)

        self.Linear_grid_MLP_3 = nn.Linear((self.grid_size[2][0] * self.grid_size[2][1]),
                                           (self.grid_size[2][0] * self.grid_size[2][1]), bias=use_bias)

    def forward(self, x): # 去掉原 deterministic drop 后 未加 mask。
        n, h, w,num_channels = x.shape
        # n x h x w x c
        # input projection
        x = self.LayerNorm(x.float()) # 没有 float 报错
        x = self.Gelu(x)

        # grid 和 block 的大小都根据 给定的 grid_size or block_size 自动匹配另一个大小。即 grid_size 给定，自动计算 block_size
        # Get grid MLP weights
        gh1, gw1 = self.grid_size[0]
        fh1, fw1 = h // gh1, w // gw1
        u1 = block_images_einops(x, patch_size=(fh1, fw1)) # 得到 B (gh gw) (fh fw) c 即：ghxgw 个 fhxfw.
        # 此函数只需要 fh,fw 得到 gh 和gw 是方便 unblock_images_einops 使用
        u1 = u1.permute(0,3,2,1)

        u1 = self.Linear_grid_MLP_1(u1)
        u1 = u1.permute(0,3,2,1)
        u1 = unblock_images_einops(u1, grid_size=(gh1, gw1), patch_size=(fh1, fw1))

        gh2, gw2 = self.grid_size[1]
        fh2, fw2 = h // gh2, w // gw2
        u2 = block_images_einops(u1, patch_size=(fh2, fw2))  # 得到 B (gh gw) (fh fw) c 即：ghxgw 个 fhxfw.
        # 此函数只需要 fh,fw 得到 gh 和gw 是方便 unblock_images_einops 使用
        u2 = u2.permute(0, 3, 2, 1)

        u2 = self.Linear_grid_MLP_2(u2)
        u2 = u2.permute(0, 3, 2, 1)
        u2 = unblock_images_einops(u2, grid_size=(gh2, gw2), patch_size=(fh2, fw2))

        gh3, gw3 = self.grid_size[2]
        fh3, fw3 = h // gh3, w // gw3
        u3 = block_images_einops(u2, patch_size=(fh3, fw3))  # 得到 B (gh gw) (fh fw) c 即：ghxgw 个 fhxfw.
        # 此函数只需要 fh,fw 得到 gh 和gw 是方便 unblock_images_einops 使用
        u3 = u3.permute(0, 3, 2, 1)

        u3 = self.Linear_grid_MLP_3(u3)
        u3 = u3.permute(0, 3, 2, 1)
        u3 = unblock_images_einops(u3, grid_size=(gh3, gw3), patch_size=(fh3, fw3))

        ux = self.Linear_end(u3)
        x = self.dropout(ux)

        return x # 不改变维度。# n h w c

# block mlp(local information extraction)
class Multi_Scale_bolckSpatialGatingWeights(nn.Module): # 将输入 channel -> channel/2 ，u计算grid/v计算block 再concat
    """Get gating weights for cross-gating MLP block."""
    def __init__(self, nIn,Nout, H_size=128, W_size=128, input_proj_factor=2, dropout_rate=0.0, use_bias=True, train_size=512):
        super(Multi_Scale_bolckSpatialGatingWeights, self).__init__()
        self.H = H_size
        self.W = W_size
        self.IN = nIn
        self.OUT = Nout
        if train_size == 512:
            self.grid_size = [[16, 16], [8, 8], [2, 2]]
        else:
            self.grid_size = [[6, 6], [3, 3], [2, 2]]

        self.block_size = [[int(H_size / l[0]), int(W_size / l[1])] for l in self.grid_size]
        self.input_proj_factor = input_proj_factor # 控制将输入 映射到多维，达到扩大channel 的目的.
        self.dropout_rate = dropout_rate
        self.use_bias = use_bias
        self.dropout = nn.Dropout(self.dropout_rate)
        self.LayerNorm = nn.LayerNorm(self.IN)
        self.Linear_end = nn.Linear(self.IN,self.OUT)
        self.Gelu = nn.GELU()
        self.Linear_Block_MLP_1 = nn.Linear((self.block_size[0][0] * self.block_size[0][1]),
                                            (self.block_size[0][0] * self.block_size[0][1]), bias=use_bias)

        self.Linear_Block_MLP_2 = nn.Linear((self.block_size[1][0] * self.block_size[1][1]),
                                            (self.block_size[1][0] * self.block_size[1][1]), bias=use_bias)

        self.Linear_Block_MLP_3 = nn.Linear((self.block_size[2][0] * self.block_size[2][1]),
                                            (self.block_size[2][0] * self.block_size[2][1]), bias=use_bias)
        self.conv = nn.Conv2d(self.IN * 2, self.OUT, kernel_size=3, padding=1, bias=False)

    def forward(self, x): # 去掉原 deterministic drop 后 未加 mask。
        n, h, w,num_channels = x.shape
        # n x h x w x c
        # input projection
        x = self.LayerNorm(x.float()) # 没有 float 报错
        x = self.Gelu(x)

        # grid 和 block 的大小都根据 给定的 grid_size or block_size 自动匹配另一个大小。即 grid_size 给定，自动计算 block_size
        # Get grid MLP weights
        fh1, fw1 = self.block_size[0]
        gh1, gw1 = h // fh1, w // fw1
        v1 = block_images_einops(x, patch_size=(fh1, fw1))
        v1 = v1.permute(0, 1, 3, 2)
        v1 = self.Linear_Block_MLP_1(v1)
        v1 = v1.permute(0, 1, 3, 2)
        v1 = unblock_images_einops(v1, grid_size=(gh1, gw1), patch_size=(fh1, fw1))

        fh2, fw2 = self.block_size[1]
        gh2, gw2 = h // fh2, w // fw2
        v2 = block_images_einops(v1, patch_size=(fh2, fw2))
        v2 = v2.permute(0, 1, 3, 2)
        v2 = self.Linear_Block_MLP_2(v2)
        v2 = v2.permute(0, 1, 3, 2)
        v2 = unblock_images_einops(v2, grid_size=(gh2, gw2), patch_size=(fh2, fw2))

        fh3, fw3 = self.block_size[2]
        gh3, gw3 = h // fh3, w // fw3
        v3 = block_images_einops(v2, patch_size=(fh3, fw3))
        v3 = v3.permute(0, 1, 3, 2)
        v3 = self.Linear_Block_MLP_3(v3)
        v3 = v3.permute(0, 1, 3, 2)
        v3 = unblock_images_einops(v3, grid_size=(gh3, gw3), patch_size=(fh3, fw3))

        vx = self.Linear_end(v3)
        x = self.dropout(vx)
        return x # 不改变维度。# n h w c

class self_attn(nn.Module):
    def __init__(self, in_channels, mode='hw'):
        super(self_attn, self).__init__()

        self.mode = mode

        self.query_conv = nn.Conv2d(in_channels, in_channels // 8, kernel_size=(1, 1),stride=1, padding=0,
                              dilation=(1, 1), groups=1, bias= False)
        self.key_conv = nn.Conv2d(in_channels, in_channels // 8, kernel_size=(1, 1),stride=1, padding=0,
                              dilation=(1, 1), groups=1, bias= False)
        self.value_conv = nn.Conv2d(in_channels, in_channels, kernel_size=(1, 1),stride=1, padding=0,
                              dilation=(1, 1), groups=1, bias= False)
        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Sigmoid()

    def forward(self, x):
        batch_size, channel, height, width = x.size()

        axis = 1
        if 'h' in self.mode:
            axis *= height
        if 'w' in self.mode:
            axis *= width

        view = (batch_size, -1, axis)

        projected_query = self.query_conv(x).contiguous().view(*view).permute(0, 2, 1)
        projected_key = self.key_conv(x).contiguous().view(*view)

        attention_map = torch.bmm(projected_query, projected_key)
        attention = self.softmax(attention_map)
        projected_value = self.value_conv(x).contiguous().view(*view)

        out = torch.bmm(projected_value, attention.permute(0, 2, 1))
        out = out.contiguous().view(batch_size, channel, height, width)

        out = self.gamma * out + x
        return out


class Axias_MHSA_block(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(Axias_MHSA_block, self).__init__()
        self.conv0 = nn.Conv2d(in_channel, out_channel, kernel_size=(1, 1),stride=1, padding=0)
        self.conv1 = nn.Conv2d(out_channel, out_channel, kernel_size=(3, 3),stride = 1, padding=1)
        self.Hattn = self_attn(out_channel, mode='h')
        self.Wattn = self_attn(out_channel, mode='w')

    def forward(self, x):
        x = self.conv0(x)
        x = self.conv1(x)

        Hx = self.Hattn(x)
        Wx = self.Wattn(Hx)

        return Wx


# Adaptive
class AdaptiveCrossGatingBlock(nn.Module):
    """Cross-gating MLP block."""

    def __init__(self, x_in, y_in, out_features, grid_size,  dropout_rate=0.0,
                 input_proj_factor=2, upsample_y=True, use_bias=True, train_size=512):

        super(AdaptiveCrossGatingBlock, self).__init__()
        self.IN_x = x_in
        self.IN_y = y_in
        self._h = grid_size[0]
        self._w = grid_size[1]
        self.features = out_features
        self.grid_size = grid_size
        self.dropout_rate = dropout_rate
        self.input_proj_factor = input_proj_factor
        self.upsample_y = upsample_y
        self.use_bias = use_bias
        self.Conv1X1_x = nn.Conv2d(self.IN_x, self.features,(1,1))
        self.Conv1X1_y = nn.Conv2d(self.IN_x, self.features,(1,1))
        self.LayerNorm_x = nn.LayerNorm(self.features)
        self.LayerNorm_y = nn.LayerNorm(self.features)
        self.Linear_x = nn.Linear(self.features, self.features,bias=use_bias)
        self.Linear_y = nn.Linear(self.features, self.features,bias=use_bias)
        self.Gelu_x = nn.GELU()
        self.Gelu_y = nn.GELU()
        self.Linear_x_end = nn.Linear(self.features, self.features,bias=use_bias)
        self.Linear_y_end = nn.Linear(self.features, self.features,bias=use_bias)
        self.dropout_x = nn.Dropout(self.dropout_rate)
        self.dropout_y = nn.Dropout(self.dropout_rate)


        self.ConvT = conv_T_y_2_x(self.IN_y, self.IN_x)
        self.fun_gx = Multi_Scale_globalSpatialGatingWeights(nIn=self.features, Nout=self.features, H_size=self._h, W_size=self._w,
                                                 input_proj_factor=2, dropout_rate=dropout_rate, use_bias=True, train_size=train_size)

        self.fun_gy = Multi_Scale_bolckSpatialGatingWeights(nIn=self.features, Nout=self.features, H_size=self._h, W_size=self._w,
                                                 input_proj_factor=2, dropout_rate=dropout_rate, use_bias=True, train_size=train_size)

        self.r = 1                                  #step length  步长，默认为1
        self.M = 2                                  #分支数
        d = max(int(self.features / self.r), 32)    # 计算向量Z 的长度d
        self.global_pool = nn.AdaptiveAvgPool2d(1)  # 自适应pool到指定维度    这里指定为1，实现 GAP
        self.fc1 = nn.Sequential(nn.Conv2d(self.features, d, 1, bias=False),
                                 nn.BatchNorm2d(d),
                                 nn.ReLU(inplace=True))  # 降维
        self.fc2 = nn.Conv2d(d, self.features * self.M, 1, 1, bias=False)  # 升维
        self.softmax = nn.Softmax(dim=1)  # 指定dim=1  使得两个全连接层对应位置进行softmax,保证 对应位置a+b+..=1

    def forward(self, x, y):
    # Upscale Y signal, y is the gating signal.
        if self.upsample_y:
            # 将 y 的维度调整为与 x 相同大小
            y = self.ConvT(x,y) # nn.ConvTranspose 反卷积

        x = self.Conv1X1_x(x)
        y = self.Conv1X1_y(y)
        assert y.shape == x.shape

        shortcut_x = x
        shortcut_y = y
        x = x.permute(0, 2, 3, 1)  # n x h x w x c
        y = y.permute(0, 2, 3, 1)

        # Get gating weights from X
        x = self.LayerNorm_x(x)
        x = self.Linear_x(x)
        x = self.Gelu_x(x)

        gx = self.fun_gx(x)
        # n x h x w x c
        # Get gating weights from Y
        y = self.LayerNorm_y(y)
        y = self.Linear_y(y)
        y = self.Gelu_y(y)

        gy = self.fun_gy(y)

        # x = x.permute(0, 3, 1, 2)  # n x h x w x c --> n x c x h x w
        # y = y.permute(0, 3, 1, 2)

        gx = gx.permute(0, 3, 1, 2)  # n x h x w x c --> n x c x h x w
        gy = gy.permute(0, 3, 1, 2)

        fea_U = gx+gy # 逐元素相加生成 混合特征U
        fea_s = self.global_pool(fea_U)
        fea_z = self.fc1(fea_s)
        fea_z = self.fc2(fea_z)
        fea_z = fea_z.view(fea_z.shape[0], 2, -1, fea_z.shape[-1])

        attention_vectors = self.softmax(fea_z)
        attention_vectors1, attention_vectors2 = torch.split(attention_vectors, 1, dim=1)

        attention_vectors1 = attention_vectors1.reshape(attention_vectors1.shape[0], self.features, -1,
                                                    attention_vectors1.shape[-1])
        attention_vectors2 = attention_vectors2.reshape(attention_vectors2.shape[0], self.features, -1,
                                                    attention_vectors2.shape[-1])
        out_x = attention_vectors1 * gx
        out_y = attention_vectors2 * gy
        x = out_x + shortcut_x
        y = out_y + shortcut_y

        # y = y * gx
        # y = self.Linear_y_end(y)
        # y = self.dropout_y(y)
        # y = y + shortcut_y
        # x = x * gy  # gating x using y
        # x = self.Linear_y_end(x)
        # x = self.dropout_x(x)
        # x = x + y + shortcut_x  # get all aggregated signals # 注意此处的 x 融合了来自y 的信息。
        # x = x.permute(0, 3, 1, 2)  # n x h x w x c --> n x c x h x w
        # y = y.permute(0, 3, 1, 2)
        return x, y

class UNet_MLP_Reduce(nn.Module):
    def __init__(self, n_channels, n_classes, basic_chans = 16, bilinear=False):
        super(UNet_MLP_Reduce, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        self.basic_channel = basic_chans

        self.inc = DoubleConv(n_channels, self.basic_channel*4)
        self.down1 = Down(self.basic_channel * 4, self.basic_channel * 8)
        self.down2 = Down(self.basic_channel * 8, self.basic_channel * 16)
        self.down3 = Down(self.basic_channel * 16, self.basic_channel * 32)
        factor = 2 if bilinear else 1
        self.down4 = Down(self.basic_channel * 32, self.basic_channel * 64 // factor)

        resolution = 512
        res1 = resolution / 8
        res2 = resolution / 8
        res3 = resolution / 8
        res4 = resolution / 8

        #Multi-scale MLP
        self.agb1 = AdaptiveCrossGatingBlock(self.basic_channel * 32, self.basic_channel * 32, self.basic_channel * 32,
                                             [res4, res4], 0.1, train_size=resolution)
        self.agb2 = AdaptiveCrossGatingBlock(self.basic_channel * 16, self.basic_channel * 16, self.basic_channel * 16,
                                             [res3, res3], 0.1, train_size=resolution)
        self.agb3 = AdaptiveCrossGatingBlock(self.basic_channel * 8, self.basic_channel * 8, self.basic_channel * 8,
                                             [res2, res2], 0.1, train_size=resolution)
        self.agb4 = AdaptiveCrossGatingBlock(self.basic_channel * 4, self.basic_channel * 4, self.basic_channel * 4,
                                             [res1, res1], 0.1, train_size=resolution)

        self.fuseup1 = FusionUp(self.basic_channel * 64, self.basic_channel * 32 // factor, bilinear)
        self.fuseup2 = FusionUp(self.basic_channel * 32, self.basic_channel * 16 // factor, bilinear)
        self.fuseup3 = FusionUp(self.basic_channel * 16, self.basic_channel * 8 // factor, bilinear)
        self.fuseup4 = FusionUp(self.basic_channel * 8, self.basic_channel * 4, bilinear)
        self.outc = OutConv(self.basic_channel * 4, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        x5_1, x5_2 = torch.chunk(x5, 2, dim=1)
        x5_out1, x5_out2 = self.agb1(x5_1, x5_2)  # 32 --> 32
        mfi_out_1 = torch.cat([x5_out1, x5_out2], dim=1)
        x = self.fuseup1(mfi_out_1, x4)

        x4_1, x4_2 = torch.chunk(x, 2, dim=1)
        x4_out1, x4_out2 = self.agb2(x4_1, x4_2)  # 32 --> 32
        mfi_out_2 = torch.cat([x4_out1, x4_out2], dim=1)
        x = self.fuseup2(mfi_out_2, x3)

        x3_1, x3_2 = torch.chunk(x, 2, dim=1)
        x3_out1, x3_out2 = self.agb3(x3_1, x3_2)  # 32 --> 32
        mfi_out_3 = torch.cat([x3_out1, x3_out2], dim=1)
        x = self.fuseup3(mfi_out_3, x2)

        x2_1, x2_2 = torch.chunk(x, 2, dim=1)
        x2_out1, x2_out2 = self.agb4(x2_1, x2_2)  # 32 --> 32
        mfi_out_4 = torch.cat([x2_out1, x2_out2], dim=1)
        x = self.fuseup4(mfi_out_4, x1)

        # x = self.fuseup2(x, x3)
        # x = self.fuseup3(x, x2)
        # x = self.fuseup4(x, x1)
        out = self.outc(x)
        return out

class SingleConvBlock(nn.Module):
    def __init__(self, in_features, out_features, stride,
                 use_bs=True
                 ):
        super(SingleConvBlock, self).__init__()
        self.use_bn = use_bs
        self.conv = nn.Conv2d(in_features, out_features, 1, stride=stride,
                              bias=True)
        self.bn = nn.BatchNorm2d(out_features)

    def forward(self, x):
        x = self.conv(x)
        if self.use_bn:
            x = self.bn(x)
        return x

class UNet_MLP_mask(nn.Module):
    def __init__(self, n_channels, n_classes, basic_chans = 16, bilinear=False):
        super(UNet_MLP_mask, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        self.basic_channel = basic_chans

        self.inc = DoubleConv(n_channels, self.basic_channel*4)
        self.down1 = Down(self.basic_channel * 4, self.basic_channel * 8)
        self.down2 = Down(self.basic_channel * 8, self.basic_channel * 16)
        self.down3 = Down(self.basic_channel * 16, self.basic_channel * 32)
        factor = 2 if bilinear else 1
        self.down4 = Down(self.basic_channel * 32, self.basic_channel * 64 // factor)

        # multi-scale information processing
        self.ms_x5 = SingleConvBlock(self.basic_channel * 64, self.basic_channel * 4,1)
        self.ms_x4 = SingleConvBlock(self.basic_channel * 32, self.basic_channel * 4, 1)
        self.ms_x3 = SingleConvBlock(self.basic_channel * 16, self.basic_channel * 4, 1)
        self.ms_x2 = SingleConvBlock(self.basic_channel * 8, self.basic_channel * 4, 1)

        self.up_x5 = nn.Upsample(scale_factor=16, mode='bilinear', align_corners=True)
        self.up_x4 = nn.Upsample(scale_factor=8, mode='bilinear', align_corners=True)
        self.up_x3 = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)
        self.up_x2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.block_cat = SingleConvBlock(self.basic_channel*20, self.basic_channel*4, stride=1, use_bs=False)  # hed fusion method
        self.block_conv = DoubleConv(self.basic_channel * 4, self.basic_channel * 4)
        self.pre_mask = SingleConvBlock(self.basic_channel * 4, 1, 1)
        self.down_x5 = nn.MaxPool2d(16)

        self.cor_feature_x5 = Conv(self.basic_channel * 64, self.basic_channel * 32, 3, 1, 1, bn_acti=True)
        self.cor_mask_x5 = Conv(self.basic_channel * 64, self.basic_channel * 32, 3, 1, 1, bn_acti=True)
        self.cor_x5 = Conv(self.basic_channel * 64, self.basic_channel * 64, 3, 1, 1, bn_acti=True)

        resolution = 512
        res1 = resolution / 8
        res2 = resolution / 8
        res3 = resolution / 8
        res4 = resolution / 8

        #Multi-scale MLP
        self.agb1 = AdaptiveCrossGatingBlock(self.basic_channel * 16, self.basic_channel * 16, self.basic_channel * 16,
                                       [res4, res4], 0.1,train_size=resolution)
        self.agb2 = AdaptiveCrossGatingBlock(self.basic_channel * 8, self.basic_channel * 8, self.basic_channel * 8,
                                             [res3, res3], 0.1, train_size=resolution)
        self.agb3 = AdaptiveCrossGatingBlock(self.basic_channel * 4, self.basic_channel * 4, self.basic_channel * 4,
                                             [res2, res2], 0.1, train_size=resolution)

        self.agb4 = AdaptiveCrossGatingBlock(self.basic_channel * 2, self.basic_channel * 2, self.basic_channel * 2,
                                             [res1, res1], 0.1, train_size=resolution)

        self.fuseup1 = FusionUp(self.basic_channel * 64, self.basic_channel * 32 // factor, bilinear)
        self.fuseup2 = FusionUp(self.basic_channel * 32, self.basic_channel * 16 // factor, bilinear)
        self.fuseup3 = FusionUp(self.basic_channel * 16, self.basic_channel * 8 // factor, bilinear)
        self.fuseup4 = FusionUp(self.basic_channel * 8, self.basic_channel * 4, bilinear)
        self.outc = OutConv(self.basic_channel * 4, n_classes)

    def forward(self, x):
        x1 = self.inc(x)

        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        fuse_x5 = self.ms_x5(x5)
        fuse_x5 = self.up_x5(fuse_x5)
        fuse_x4 = self.ms_x4(x4)
        fuse_x4 = self.up_x4(fuse_x4)
        fuse_x3 = self.ms_x3(x3)
        fuse_x3 = self.up_x3(fuse_x3)
        fuse_x2 = self.ms_x2(x2)
        fuse_x2 = self.up_x2(fuse_x2)
        fuse_x = [x1, fuse_x2, fuse_x3, fuse_x4, fuse_x5]

        block_cat = torch.cat(fuse_x, dim=1)  # Bx6xHxW
        block_cat = self.block_cat(block_cat)  # Bx1xHxW

        mul_x4 = fuse_x4 * torch.sigmoid(fuse_x5)
        mul_x3 = fuse_x3 * torch.sigmoid(mul_x4)
        mul_x2 = fuse_x2 * torch.sigmoid(mul_x3)
        mul_x1 = x1 * torch.sigmoid(mul_x2)
        fuse_conv = self.block_conv(block_cat + mul_x1)
        pre_mask = self.pre_mask(fuse_conv)
        mask_x5 = self.down_x5(pre_mask)
        mask_x5 = torch.sigmoid(mask_x5)

        ''''''
        feature_x5 = mask_x5.mul(x5)
        crossmask_x5 = (1-mask_x5).mul(x5)
        cor_feature_x5 = self.cor_feature_x5(feature_x5)
        cor_mask_x5 = self.cor_mask_x5(crossmask_x5)
        cor_x5 = cor_mask_x5.mul(cor_feature_x5)
        feature_x5_reduce = self.cor_feature_x5(feature_x5)
        x5 = self.cor_x5(torch.cat([cor_x5, feature_x5_reduce], dim=1))

        x4_1, x4_2 = torch.chunk(x4, 2, dim=1)
        x4_out1, x4_out2 = self.agb1(x4_1, x4_2)  # 32 --> 32
        agb_out_4 = torch.cat([x4_out1, x4_out2], dim=1)
        x4 = agb_out_4 + x4
        x4 = self.fuseup1(x5, x4)

        x3_1, x3_2 = torch.chunk(x3, 2, dim=1)
        x3_out1, x3_out2 = self.agb2(x3_1, x3_2)  # 32 --> 32
        agb_out_3 = torch.cat([x3_out1, x3_out2], dim=1)
        x3 = agb_out_3 + x3
        x3 = self.fuseup2(x4, x3)

        x2_1, x2_2 = torch.chunk(x2, 2, dim=1)
        x2_out1, x2_out2 = self.agb3(x2_1, x2_2)  # 32 --> 32
        agb_out_2 = torch.cat([x2_out1, x2_out2], dim=1)
        x2 = agb_out_2 + x2
        x2 = self.fuseup3(x3, x2)

        x1_1, x1_2 = torch.chunk(x1, 2, dim=1)
        x1_out1, x1_out2 = self.agb4(x1_1, x1_2)  # 32 --> 32
        agb_out_1 = torch.cat([x1_out1, x1_out2], dim=1)
        x1 = agb_out_1 + x1
        x = self.fuseup4(x2, x1)

        out = self.outc(x)
        return out


class UNet_MLP_mask_all(nn.Module):
    def __init__(self, n_channels, n_classes, basic_chans=16, bilinear=False):
        super(UNet_MLP_mask_all, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        self.basic_channel = basic_chans

        self.inc = DoubleConv(n_channels, self.basic_channel * 4)
        self.down1 = Down(self.basic_channel * 4, self.basic_channel * 8)
        self.down2 = Down(self.basic_channel * 8, self.basic_channel * 16)
        self.down3 = Down(self.basic_channel * 16, self.basic_channel * 32)
        factor = 2 if bilinear else 1
        self.down4 = Down(self.basic_channel * 32, self.basic_channel * 64 // factor)

        # multi-scale information processing
        self.ms_x5 = SingleConvBlock(self.basic_channel * 64, self.basic_channel * 4, 1)
        self.ms_x4 = SingleConvBlock(self.basic_channel * 32, self.basic_channel * 4, 1)
        self.ms_x3 = SingleConvBlock(self.basic_channel * 16, self.basic_channel * 4, 1)
        self.ms_x2 = SingleConvBlock(self.basic_channel * 8, self.basic_channel * 4, 1)

        self.up_x5 = nn.Upsample(scale_factor=16, mode='bilinear', align_corners=True)
        self.up_x4 = nn.Upsample(scale_factor=8, mode='bilinear', align_corners=True)
        self.up_x3 = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)
        self.up_x2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.block_cat = SingleConvBlock(self.basic_channel * 20, self.basic_channel * 4, stride=1,
                                         use_bs=False)  # hed fusion method
        self.block_conv = DoubleConv(self.basic_channel * 4, self.basic_channel * 4)
        self.pre_mask = SingleConvBlock(self.basic_channel * 4, 1, 1)
        self.down_x5 = nn.MaxPool2d(16)

        self.cor_feature_x5 = Conv(self.basic_channel * 64, self.basic_channel * 32, 3, 1, 1, bn_acti=True)
        self.cor_mask_x5 = Conv(self.basic_channel * 64, self.basic_channel * 32, 3, 1, 1, bn_acti=True)
        self.cor_x5 = Conv(self.basic_channel * 64, self.basic_channel * 64, 3, 1, 1, bn_acti=True)

        self.pre_mask_x3 = SingleConvBlock(self.basic_channel * 4, 1, 1)
        self.up_mask_x3 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.pre_mask_x3 = SingleConvBlock(self.basic_channel * 32, 1, 1)
        self.cor_feature_x3 = Conv(self.basic_channel * 16, self.basic_channel * 16, 3, 1, 1, bn_acti=True)
        self.cor_mask_x3 = Conv(self.basic_channel * 16, self.basic_channel * 16, 3, 1, 1, bn_acti=True)
        self.cor_x3 = Conv(self.basic_channel * 32, self.basic_channel * 16, 3, 1, 1, bn_acti=True)


        self.pre_mask_x1 = SingleConvBlock(self.basic_channel * 4, 1, 1)
        self.up_mask_x1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.pre_mask_x1 = SingleConvBlock(self.basic_channel * 8, 1, 1)
        self.cor_feature_x1 = Conv(self.basic_channel * 4, self.basic_channel * 4, 3, 1, 1, bn_acti=True)
        self.cor_mask_x1 = Conv(self.basic_channel * 4, self.basic_channel * 4, 3, 1, 1, bn_acti=True)
        self.cor_x1 = Conv(self.basic_channel * 8, self.basic_channel * 4, 3, 1, 1, bn_acti=True)

        resolution = 512
        res1 = resolution / 8
        res2 = resolution / 8
        res3 = resolution / 8
        res4 = resolution / 8

        # Multi-scale MLP
        self.agb1 = AdaptiveCrossGatingBlock(self.basic_channel * 16, self.basic_channel * 16, self.basic_channel * 16,
                                             [res4, res4], 0.1, train_size=resolution)
        self.agb2 = AdaptiveCrossGatingBlock(self.basic_channel * 8, self.basic_channel * 8, self.basic_channel * 8,
                                             [res3, res3], 0.1, train_size=resolution)
        self.agb3 = AdaptiveCrossGatingBlock(self.basic_channel * 4, self.basic_channel * 4, self.basic_channel * 4,
                                             [res2, res2], 0.1, train_size=resolution)

        self.agb4 = AdaptiveCrossGatingBlock(self.basic_channel * 2, self.basic_channel * 2, self.basic_channel * 2,
                                             [res1, res1], 0.1, train_size=resolution)

        self.fuseup1 = FusionUp(self.basic_channel * 64, self.basic_channel * 32 // factor, bilinear)
        self.fuseup2 = FusionUp(self.basic_channel * 32, self.basic_channel * 16 // factor, bilinear)
        self.fuseup3 = FusionUp(self.basic_channel * 16, self.basic_channel * 8 // factor, bilinear)
        self.fuseup4 = FusionUp(self.basic_channel * 8, self.basic_channel * 4, bilinear)
        self.outc = OutConv(self.basic_channel * 4, n_classes)

    def forward(self, x):
        x1 = self.inc(x)

        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        fuse_x5 = self.ms_x5(x5)
        fuse_x5 = self.up_x5(fuse_x5)
        fuse_x4 = self.ms_x4(x4)
        fuse_x4 = self.up_x4(fuse_x4)
        fuse_x3 = self.ms_x3(x3)
        fuse_x3 = self.up_x3(fuse_x3)
        fuse_x2 = self.ms_x2(x2)
        fuse_x2 = self.up_x2(fuse_x2)
        fuse_x = [x1, fuse_x2, fuse_x3, fuse_x4, fuse_x5]

        block_cat = torch.cat(fuse_x, dim=1)  # Bx6xHxW
        block_cat = self.block_cat(block_cat)  # Bx1xHxW

        mul_x4 = fuse_x4 * torch.sigmoid(fuse_x5)
        mul_x3 = fuse_x3 * torch.sigmoid(mul_x4)
        mul_x2 = fuse_x2 * torch.sigmoid(mul_x3)
        mul_x1 = x1 * torch.sigmoid(mul_x2)
        fuse_conv = self.block_conv(block_cat + mul_x1)
        pre_mask = self.pre_mask(fuse_conv)
        mask_x5 = self.down_x5(pre_mask)
        mask_x5 = torch.sigmoid(mask_x5)

        '''Cross relation context for the fifth layer'''
        feature_x5 = mask_x5.mul(x5)
        crossmask_x5 = (1 - mask_x5).mul(x5)
        cor_feature_x5 = self.cor_feature_x5(feature_x5)
        cor_mask_x5 = self.cor_mask_x5(crossmask_x5)
        cor_x5 = cor_mask_x5.mul(cor_feature_x5)
        feature_x5_reduce = self.cor_feature_x5(feature_x5)
        x5 = self.cor_x5(torch.cat([cor_x5, feature_x5_reduce], dim=1))

        x4_1, x4_2 = torch.chunk(x4, 2, dim=1)
        x4_out1, x4_out2 = self.agb1(x4_1, x4_2)  # 32 --> 32
        agb_out_4 = torch.cat([x4_out1, x4_out2], dim=1)
        x4 = agb_out_4 + x4
        x4 = self.fuseup1(x5, x4)

        x3_1, x3_2 = torch.chunk(x3, 2, dim=1)
        x3_out1, x3_out2 = self.agb2(x3_1, x3_2)  # 32 --> 32
        agb_out_3 = torch.cat([x3_out1, x3_out2], dim=1)
        x3 = agb_out_3 + x3

        '''Cross relation context for the third layer'''
        mask_x3 = self.up_mask_x3(x4)             # upsampling operator
        mask_x3 = torch.sigmoid(mask_x3)
        mask_x3 = self.pre_mask_x3(mask_x3)       # mask

        feature_x3 = mask_x3.mul(x3)
        crossmask_x3 = (1 - mask_x3).mul(x3)
        cor_feature_x3 = self.cor_feature_x3(feature_x3)
        cor_mask_x3 = self.cor_mask_x3(crossmask_x3)
        cor_x3 = cor_mask_x3.mul(cor_feature_x3)
        feature_x3_reduce = self.cor_feature_x3(feature_x3)
        x3 = self.cor_x3(torch.cat([cor_x3, feature_x3_reduce], dim=1))
        x3 = self.fuseup2(x4, x3)

        x2_1, x2_2 = torch.chunk(x2, 2, dim=1)
        x2_out1, x2_out2 = self.agb3(x2_1, x2_2)  # 32 --> 32
        agb_out_2 = torch.cat([x2_out1, x2_out2], dim=1)
        x2 = agb_out_2 + x2
        x2 = self.fuseup3(x3, x2)

        x1_1, x1_2 = torch.chunk(x1, 2, dim=1)
        x1_out1, x1_out2 = self.agb4(x1_1, x1_2)  # 32 --> 32
        agb_out_1 = torch.cat([x1_out1, x1_out2], dim=1)
        x1 = agb_out_1 + x1

        '''Cross relation context for the first layer'''
        mask_x1 = self.up_mask_x1(x2)             # upsampling operator
        mask_x1 = torch.sigmoid(mask_x1)
        mask_x1 = self.pre_mask_x1(mask_x1)       # mask

        feature_x1 = mask_x1.mul(x1)
        crossmask_x1 = (1 - mask_x1).mul(x1)
        cor_feature_x1 = self.cor_feature_x1(feature_x1)
        cor_mask_x1 = self.cor_mask_x1(crossmask_x1)
        cor_x1 = cor_mask_x1.mul(cor_feature_x1)
        feature_x1_reduce = self.cor_feature_x1(feature_x1)
        x1= self.cor_x1(torch.cat([cor_x1, feature_x1_reduce], dim=1))

        x = self.fuseup4(x2, x1)

        out = self.outc(x)
        return out





