from __future__ import print_function, division
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
import torch
# from nnunet.network_architecture.neural_network import SegmentationNetwork
import math
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
# from ResNetV2 import ResNet as ResNetV2
from models.Ours_MLP.ResNetV2 import ResNet as ResNetV2
from models.Ours_MLP.ResNetV2 import ResNetlite


def np2th(weights, conv=False):
    """Possibly convert HWIO to OIHW."""
    if conv:
        weights = weights.transpose([3, 2, 0, 1])
    return torch.from_numpy(weights)


class StdConv2d(nn.Conv2d):

    def forward(self, x):
        w = self.weight
        v, m = torch.var_mean(w, dim=[1, 2, 3], keepdim=True, unbiased=False)
        w = (w - m) / torch.sqrt(v + 1e-5)
        return F.conv2d(x, w, self.bias, self.stride, self.padding,
                        self.dilation, self.groups)


def conv3x3(cin, cout, stride=1, groups=1, bias=False):
    return StdConv2d(cin, cout, kernel_size=3, stride=stride,
                     padding=1, bias=bias, groups=groups)


def conv1x1(cin, cout, stride=1, bias=False):
    return StdConv2d(cin, cout, kernel_size=1, stride=stride,
                     padding=0, bias=bias)


class PreActBottleneck(nn.Module):
    """Pre-activation (v2) bottleneck block.
    """

    def __init__(self, cin, cout=None, cmid=None, stride=1):
        super().__init__()
        cout = cout or cin
        cmid = cmid or cout//4

        self.gn1 = nn.GroupNorm(32, cmid, eps=1e-6)
        self.conv1 = conv1x1(cin, cmid, bias=False)
        self.gn2 = nn.GroupNorm(32, cmid, eps=1e-6)
        self.conv2 = conv3x3(cmid, cmid, stride, bias=False)  # Original code has it on conv1!!
        self.gn3 = nn.GroupNorm(32, cout, eps=1e-6)
        self.conv3 = conv1x1(cmid, cout, bias=False)
        self.relu = nn.ReLU(inplace=True)

        if (stride != 1 or cin != cout):
            # Projection also with pre-activation according to paper.
            self.downsample = conv1x1(cin, cout, stride, bias=False)
            self.gn_proj = nn.GroupNorm(cout, cout)

    def forward(self, x):

        # Residual branch
        residual = x
        if hasattr(self, 'downsample'):
            residual = self.downsample(x)
            residual = self.gn_proj(residual)

        # Unit's branch
        y = self.relu(self.gn1(self.conv1(x)))
        y = self.relu(self.gn2(self.conv2(y)))
        y = self.gn3(self.conv3(y))

        y = self.relu(residual + y)
        return y


class up_conv(nn.Module):
    """
    Up Convolution Block
    """
    def __init__(self, in_ch, out_ch, scfactor=2):
        super(up_conv, self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=scfactor, mode='bilinear'),
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.up(x)
        return x

class conv_block(nn.Module):
    """
    Convolution Block
    """

    def __init__(self, in_ch, out_ch):
        super(conv_block, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True))

    def forward(self, x):
        x = self.conv(x)
        return x


class BN_Conv2d(nn.Module):
    """
    BN_CONV, default activation is ReLU
    """

    def __init__(self, in_channels: object, out_channels: object, kernel_size: object, stride: object, padding: object,
                 dilation=1, groups=1, bias=False, activation=nn.ReLU(inplace=True)) -> object:
        super(BN_Conv2d, self).__init__()
        layers = [nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride,
                            padding=padding, dilation=dilation, groups=groups, bias=bias),
                  nn.BatchNorm2d(out_channels)]
        if activation is not None:
            layers.append(activation)
        self.seq = nn.Sequential(*layers)

    def forward(self, x):
        return self.seq(x)


class Multiscaleconv_small(nn.Module):
    """
    Convolution Block
    """
    def __init__(self, in_ch, out_ch):
        super(Multiscaleconv_small, self).__init__()

        self.conv1 = BN_Conv2d(in_ch, out_ch, 1, 1, 0, bias=False)
        self.conv3 = nn.Sequential(
            BN_Conv2d(in_ch, out_ch//2, 3, 1, 1, bias=False),
            BN_Conv2d(out_ch//2, out_ch, 1, 1, 0, bias=False),
        )
        self.conv5 = nn.Sequential(
            BN_Conv2d(in_ch, out_ch//2, 3, 1, 1, bias=False),
            BN_Conv2d(out_ch//2, out_ch//2, 3, 1, 1, bias=False),
            BN_Conv2d(out_ch//2, out_ch, 1, 1, 0, bias=False)
        )
    def forward(self, x):
        x = self.conv3(x)+ self.conv5(x)+self.conv1(x)
        return x

class DWConv(nn.Module):
    def __init__(self, dim=768):
        super(DWConv, self).__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim)

    def forward(self, x, H, W):
        B, N, C = x.shape
        x = x.transpose(1, 2).view(B, C, H, W)
        x = self.dwconv(x)
        x = x.flatten(2).transpose(1, 2)

        return x

class MLPBlock(nn.Module):
    """ Image to Patch Embedding
    """
    def __init__(self, in_dim, out_dim=768, mlp_ratio=4., drop = 0.4, img_size=224, patch_size=7, stride=4):
        super().__init__()

        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        self.drop = drop
        self.shift_size = 5
        self.pad = self.shift_size // 2

        self.H, self.W = img_size[0] // patch_size[0], img_size[1] // patch_size[1]
        self.num_patches = self.H * self.W
        self.proj = nn.Conv2d(in_dim, out_dim, kernel_size=patch_size, stride=stride,
                              padding=(patch_size[0] // 2, patch_size[1] // 2))
        self.norm = nn.BatchNorm2d(out_dim)
        self.fc1 = nn.Linear(in_dim, in_dim)
        self.sublayernorm = nn.LayerNorm(in_dim)
        self.drop = nn.Dropout(drop)
        self.fc2 = nn.Linear(out_dim, out_dim * out_dim)
        self.fc3 = nn.Linear(out_dim, out_dim)
        self.act = nn.GELU()

        self.dwconv = BN_Conv2d(out_dim, out_dim, 3, 1, 1, bias=False)
        self.drop_path = nn.Identity()
        self.norm2 = nn.LayerNorm(out_dim)
        mlp_hidden_dim = int(out_dim * mlp_ratio)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x):
        x = self.proj(x)
        B, C, H, W = x.shape
        # x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)
        y, z = torch.split(x, C // 2, dim= 1)

        xn = F.pad(y, (self.pad, self.pad, self.pad, self.pad), "constant", 0)
        xs = torch.chunk(xn, self.shift_size, 1)  # torch.chunk(tensor, chunk_num, dim) 将tensor按dim（行或列）分割成chunk_num个tensor块，返回的是一个元组。

        x1_shift = [torch.roll(x_c, shift, 2) for x_c, shift in zip(xs, range(-self.pad, self.pad + 1))]
        x1_cat = torch.cat(x1_shift, 1)
        x1_cat = torch.narrow(x1_cat, 2, self.pad, H)  # 个函数是返回tensor的第dim维切片start: start+length的数, 针对例子，
        x1_s = torch.narrow(x1_cat, 3, self.pad, W)
        # x1_s = x1_s.reshape(B, C//2, H * W).contiguous() # 8, 64, 1024
        x_shift_r = self.fc1(x1_s.permute(0, 2, 3, 1))
        # x_shift_r = x_shift_r.permute(0, 3, 1, 2)
        x_shift_r = self.sublayernorm(x_shift_r)
        x_shift_r = self.drop(x_shift_r)

        xn = F.pad(z, (self.pad, self.pad, self.pad, self.pad), "constant", 0)
        xs = torch.chunk(xn, self.shift_size, 1)  # torch.chunk(tensor, chunk_num, dim) 将tensor按dim（行或列）分割成chunk_num个tensor块，返回的是一个元组。
        x2_shift = [torch.roll(x_c, shift, 3) for x_c, shift in zip(xs, range(-self.pad, self.pad + 1))]
        x2_cat = torch.cat(x2_shift, 1)
        x2_cat = torch.narrow(x2_cat, 2, self.pad, H)  # 个函数是返回tensor的第dim维切片start: start+length的数, 针对例子，
        x2_s = torch.narrow(x2_cat, 3, self.pad, W)
        # x2_s = x2_s.reshape(B, C // 2, H * W).contiguous() # 8, 64, 1024
        # x_shift_c = x2_s.transpose(1, 2)
        x_shift_c = self.fc1(x2_s.permute(0, 2, 3, 1))
        # x_shift_c = x_shift_c.permute(0, 3, 1, 2)
        x_shift_c = self.sublayernorm(x_shift_c)
        x_shift_c = self.drop(x_shift_c)

        y_cat = torch.cat((x_shift_r.permute(0, 3, 1, 2),x_shift_c.permute(0, 3, 1, 2)),1)
        y_cat = self.fc2(y_cat.permute(0, 2, 3, 1))
        y_cat = self.act(y_cat)
        y_cat = self.drop(y_cat) # 8,
        y_cat = y_cat.view(-1, C, C)

        dy = self.fc3(x.permute(0, 2, 3, 1))
        dy = dy.reshape(-1, C).unsqueeze(1)

        x_dy = torch.bmm(dy, y_cat)
        x_dy = x_dy.view(B, H, W, C).permute(0, 3, 1, 2)
        x = self.dwconv(x_dy) # 8, 1024, 64

        return x

class LightMLPBlock(nn.Module):
    """ Image to Patch Embedding
    """
    def __init__(self, in_dim, out_dim=768, mlp_ratio=4., drop = 0.4, img_size=224, patch_size=7, stride=4):
        super().__init__()

        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        self.drop = drop
        self.shift_size = 5
        self.pad = self.shift_size // 2

        self.H, self.W = img_size[0] // patch_size[0], img_size[1] // patch_size[1]
        self.num_patches = self.H * self.W
        self.proj = nn.Conv2d(in_dim, out_dim, kernel_size=patch_size, stride=stride,
                              padding=(patch_size[0] // 2, patch_size[1] // 2))
        self.norm = nn.BatchNorm2d(out_dim)
        self.fc1 = nn.Linear(in_dim, in_dim)
        self.sublayernorm = nn.LayerNorm(in_dim)
        self.drop = nn.Dropout(drop)
        self.fc2 = nn.Linear(out_dim, out_dim * in_dim // 2)
        self.fc3 = nn.Linear(out_dim, in_dim // 2)
        self.act = nn.GELU()
        self.dwconv = BN_Conv2d(out_dim, out_dim, 3, 1, 1, bias=False)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x):
        x = self.proj(x)
        B, C, H, W = x.shape
        # x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)
        y, z = torch.split(x, C // 2, dim= 1)

        xn = F.pad(y, (self.pad, self.pad, self.pad, self.pad), "constant", 0)
        xs = torch.chunk(xn, self.shift_size, 1)  # torch.chunk(tensor, chunk_num, dim) 将tensor按dim（行或列）分割成chunk_num个tensor块，返回的是一个元组。

        x1_shift = [torch.roll(x_c, shift, 2) for x_c, shift in zip(xs, range(-self.pad, self.pad + 1))]
        x1_cat = torch.cat(x1_shift, 1)
        x1_cat = torch.narrow(x1_cat, 2, self.pad, H)  # 个函数是返回tensor的第dim维切片start: start+length的数, 针对例子，
        x1_s = torch.narrow(x1_cat, 3, self.pad, W)
        x_shift_r = self.fc1(x1_s.permute(0, 2, 3, 1))
        x_shift_r = self.sublayernorm(x_shift_r)
        x_shift_r = self.drop(x_shift_r)

        xn = F.pad(z, (self.pad, self.pad, self.pad, self.pad), "constant", 0)
        xs = torch.chunk(xn, self.shift_size, 1)  # torch.chunk(tensor, chunk_num, dim) 将tensor按dim（行或列）分割成chunk_num个tensor块，返回的是一个元组。
        x2_shift = [torch.roll(x_c, shift, 3) for x_c, shift in zip(xs, range(-self.pad, self.pad + 1))]
        x2_cat = torch.cat(x2_shift, 1)
        x2_cat = torch.narrow(x2_cat, 2, self.pad, H)  # 个函数是返回tensor的第dim维切片start: start+length的数, 针对例子，
        x2_s = torch.narrow(x2_cat, 3, self.pad, W)
        x_shift_c = self.fc1(x2_s.permute(0, 2, 3, 1))
        x_shift_c = self.sublayernorm(x_shift_c)
        x_shift_c = self.drop(x_shift_c)

        y_cat = torch.cat((x_shift_r.permute(0, 3, 1, 2),x_shift_c.permute(0, 3, 1, 2)),1)
        y_cat = self.fc2(y_cat.permute(0, 2, 3, 1))
        y_cat = self.act(y_cat)
        y_cat = self.drop(y_cat) # 8,
        y_cat = y_cat.view(-1, C // 4, C)

        dy = self.fc3(x.permute(0, 2, 3, 1))
        dy = dy.reshape(-1, C // 4).unsqueeze(1)

        x_dy = torch.bmm(dy, y_cat)
        x_dy = x_dy.view(B, H, W, C).permute(0, 3, 1, 2)
        x = self.dwconv(x_dy) # 8, 1024, 64

        return x


class LightMLPBlockv2(nn.Module):
    """ Image to Patch Embedding
    """
    def __init__(self, in_dim, out_dim=768, mlp_ratio=4., drop = 0.4, img_size=224, patch_size=7, stride=1):
        super().__init__()

        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        self.drop = drop
        self.shift_size = 5
        self.pad = self.shift_size // 2

        self.H, self.W = img_size[0] // patch_size[0], img_size[1] // patch_size[1]
        self.num_patches = self.H * self.W
        self.proj = nn.Conv2d(in_dim, out_dim, kernel_size=patch_size, stride=stride,
                              padding=(patch_size[0] // 2, patch_size[1] // 2))
        self.norm = nn.BatchNorm2d(out_dim)
        self.fc1 = nn.Linear(out_dim//2, out_dim//2)
        self.sublayernorm = nn.LayerNorm(out_dim//2)
        self.drop = nn.Dropout(drop)
        self.fc2 = nn.Linear(out_dim, out_dim * out_dim // 4)
        self.fc3 = nn.Linear(out_dim, out_dim // 4)
        self.norm2 = nn.LayerNorm(out_dim)
        self.act = nn.GELU()
        self.dwconv = BN_Conv2d(out_dim, out_dim, 3, 1, 1, bias=False)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x):
        x = self.proj(x)
        B, C, H, W = x.shape
        # x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)
        y, z = torch.split(x, C // 2, dim= 1)

        xn = F.pad(y, (self.pad, self.pad, self.pad, self.pad), "constant", 0)
        xs = torch.chunk(xn, self.shift_size, 1)

        x1_shift = [torch.roll(x_c, shift, 2) for x_c, shift in zip(xs, range(-self.pad, self.pad + 1))]
        x1_cat = torch.cat(x1_shift, 1)
        x1_cat = torch.narrow(x1_cat, 2, self.pad, H)
        x1_s = torch.narrow(x1_cat, 3, self.pad, W)
        x_shift_r = self.fc1(x1_s.permute(0, 2, 3, 1))
        x_shift_r = self.sublayernorm(x_shift_r)
        x_shift_r = self.drop(x_shift_r)

        xn = F.pad(z, (self.pad, self.pad, self.pad, self.pad), "constant", 0)
        xs = torch.chunk(xn, self.shift_size, 1)
        x2_shift = [torch.roll(x_c, shift, 3) for x_c, shift in zip(xs, range(-self.pad, self.pad + 1))]
        x2_cat = torch.cat(x2_shift, 1)
        x2_cat = torch.narrow(x2_cat, 2, self.pad, H)
        x2_s = torch.narrow(x2_cat, 3, self.pad, W)
        x_shift_c = self.fc1(x2_s.permute(0, 2, 3, 1))
        x_shift_c = self.sublayernorm(x_shift_c)
        x_shift_c = self.drop(x_shift_c)

        y_cat = torch.cat((x_shift_r.permute(0, 3, 1, 2),x_shift_c.permute(0, 3, 1, 2)),1)
        y_cat = self.fc2(y_cat.permute(0, 2, 3, 1))
        y_cat = self.act(y_cat)
        y_cat = self.drop(y_cat) # 8,
        y_cat = y_cat.view(-1, C // 4, C)

        dy = self.fc3(x.permute(0, 2, 3, 1))
        dy = dy.reshape(-1, C // 4).unsqueeze(1)

        x_dy = torch.bmm(dy, y_cat)
        x_dy = self.norm2(x_dy)
        x_dy = self.act(x_dy)
        x_dy = x_dy.view(B, H, W, C).permute(0, 3, 1, 2)
        x = self.dwconv(x_dy) # 8, 1024, 64

        return x

class LightMLPBlockv4(nn.Module):
    """ Image to Patch Embedding
    """
    def __init__(self, in_dim, out_dim=768, mlp_ratio=4., drop = 0.4, img_size=224, patch_size=7, stride=1):
        super().__init__()

        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        self.drop = drop
        self.shift_size = 5
        self.pad = self.shift_size // 2

        self.H, self.W = img_size[0] // patch_size[0], img_size[1] // patch_size[1]
        self.num_patches = self.H * self.W
        self.proj = nn.Conv2d(in_dim, out_dim, kernel_size=patch_size, stride=stride,
                              padding=(patch_size[0] // 2, patch_size[1] // 2))
        self.norm = nn.BatchNorm2d(out_dim)
        self.fc1 = nn.Linear(out_dim//2, out_dim//2)
        self.sublayernorm = nn.LayerNorm(out_dim//2)
        self.drop = nn.Dropout(drop)
        self.fc2 = nn.Linear(out_dim, out_dim * out_dim // 8)
        self.fc3 = nn.Linear(out_dim, out_dim // 8)
        self.norm2 = nn.LayerNorm(out_dim)
        self.act = nn.GELU()
        self.dwconv = BN_Conv2d(out_dim, out_dim, 3, 1, 1, bias=False)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x):
        x = self.proj(x)
        B, C, H, W = x.shape
        # x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)
        y, z = torch.split(x, C // 2, dim= 1)

        xn = F.pad(y, (self.pad, self.pad, self.pad, self.pad), "constant", 0)
        xs = torch.chunk(xn, self.shift_size, 1)

        x1_shift = [torch.roll(x_c, shift, 2) for x_c, shift in zip(xs, range(-self.pad, self.pad + 1))]
        x1_cat = torch.cat(x1_shift, 1)
        x1_cat = torch.narrow(x1_cat, 2, self.pad, H)
        x1_s = torch.narrow(x1_cat, 3, self.pad, W)
        x_shift_r = self.fc1(x1_s.permute(0, 2, 3, 1))
        x_shift_r = self.sublayernorm(x_shift_r)
        x_shift_r = self.drop(x_shift_r)

        xn = F.pad(z, (self.pad, self.pad, self.pad, self.pad), "constant", 0)
        xs = torch.chunk(xn, self.shift_size, 1)
        x2_shift = [torch.roll(x_c, shift, 3) for x_c, shift in zip(xs, range(-self.pad, self.pad + 1))]
        x2_cat = torch.cat(x2_shift, 1)
        x2_cat = torch.narrow(x2_cat, 2, self.pad, H)
        x2_s = torch.narrow(x2_cat, 3, self.pad, W)
        x_shift_c = self.fc1(x2_s.permute(0, 2, 3, 1))
        x_shift_c = self.sublayernorm(x_shift_c)
        x_shift_c = self.drop(x_shift_c)

        y_cat = torch.cat((x_shift_r.permute(0, 3, 1, 2),x_shift_c.permute(0, 3, 1, 2)),1)
        y_cat = self.fc2(y_cat.permute(0, 2, 3, 1))
        y_cat = self.act(y_cat)
        y_cat = self.drop(y_cat) # 8,
        y_cat = y_cat.view(-1, C // 8, C)

        dy = self.fc3(x.permute(0, 2, 3, 1))
        dy = dy.reshape(-1, C // 8).unsqueeze(1)

        x_dy = torch.bmm(dy, y_cat)
        x_dy = self.norm2(x_dy)
        x_dy = self.act(x_dy)
        x_dy = x_dy.view(B, H, W, C).permute(0, 3, 1, 2)
        x = self.dwconv(x_dy) # 8, 1024, 64

        return x

class LightMLPBlockv5(nn.Module):
    """ Image to Patch Embedding
    """
    def __init__(self, in_dim, out_dim=768, mlp_ratio=4., drop = 0.4, img_size=224, patch_size=7, stride=1):
        super().__init__()

        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        self.drop = drop
        self.shift_size = 5
        self.pad = self.shift_size // 2

        self.H, self.W = img_size[0] // patch_size[0], img_size[1] // patch_size[1]
        self.num_patches = self.H * self.W
        self.proj = nn.Conv2d(in_dim, out_dim, kernel_size=patch_size, stride=stride,
                              padding=(patch_size[0] // 2, patch_size[1] // 2))
        self.norm = nn.BatchNorm2d(out_dim)
        self.fc1 = nn.Linear(out_dim//2, out_dim//2)
        self.sublayernorm = nn.LayerNorm(out_dim//2)
        self.drop = nn.Dropout(drop)
        self.fc2 = nn.Linear(out_dim, out_dim * out_dim // 16)
        self.fc3 = nn.Linear(out_dim, out_dim // 16)
        self.norm2 = nn.LayerNorm(out_dim)
        self.act = nn.GELU()
        self.dwconv = BN_Conv2d(out_dim, out_dim, 3, 1, 1, bias=False)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x):
        x = self.proj(x)
        B, C, H, W = x.shape
        # x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)
        y, z = torch.split(x, C // 2, dim= 1)

        xn = F.pad(y, (self.pad, self.pad, self.pad, self.pad), "constant", 0)
        xs = torch.chunk(xn, self.shift_size, 1)

        x1_shift = [torch.roll(x_c, shift, 2) for x_c, shift in zip(xs, range(-self.pad, self.pad + 1))]
        x1_cat = torch.cat(x1_shift, 1)
        x1_cat = torch.narrow(x1_cat, 2, self.pad, H)
        x1_s = torch.narrow(x1_cat, 3, self.pad, W)
        x_shift_r = self.fc1(x1_s.permute(0, 2, 3, 1))
        x_shift_r = self.sublayernorm(x_shift_r)
        x_shift_r = self.drop(x_shift_r)

        xn = F.pad(z, (self.pad, self.pad, self.pad, self.pad), "constant", 0)
        xs = torch.chunk(xn, self.shift_size, 1)
        x2_shift = [torch.roll(x_c, shift, 3) for x_c, shift in zip(xs, range(-self.pad, self.pad + 1))]
        x2_cat = torch.cat(x2_shift, 1)
        x2_cat = torch.narrow(x2_cat, 2, self.pad, H)
        x2_s = torch.narrow(x2_cat, 3, self.pad, W)
        x_shift_c = self.fc1(x2_s.permute(0, 2, 3, 1))
        x_shift_c = self.sublayernorm(x_shift_c)
        x_shift_c = self.drop(x_shift_c)

        y_cat = torch.cat((x_shift_r.permute(0, 3, 1, 2),x_shift_c.permute(0, 3, 1, 2)),1)
        y_cat = self.fc2(y_cat.permute(0, 2, 3, 1))
        y_cat = self.act(y_cat)
        y_cat = self.drop(y_cat) # 8,
        y_cat = y_cat.view(-1, C // 16, C)

        dy = self.fc3(x.permute(0, 2, 3, 1))
        dy = dy.reshape(-1, C // 16).unsqueeze(1)

        x_dy = torch.bmm(dy, y_cat)
        x_dy = self.norm2(x_dy)
        x_dy = self.act(x_dy)
        x_dy = x_dy.view(B, H, W, C).permute(0, 3, 1, 2)
        x = self.dwconv(x_dy) # 8, 1024, 64

        return x


class shiftMLPBlock(nn.Module):
    """ Image to Patch Embedding
    """
    def __init__(self, in_dim, out_dim=768, mlp_ratio=4., drop = 0.4, img_size=224, patch_size=7, stride=1):
        super().__init__()

        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        self.drop = drop
        self.shift_size = 5
        self.pad = self.shift_size // 2

        self.H, self.W = img_size[0] // patch_size[0], img_size[1] // patch_size[1]
        self.num_patches = self.H * self.W
        self.proj = nn.Conv2d(in_dim, out_dim, kernel_size=patch_size, stride=stride,
                              padding=(patch_size[0] // 2, patch_size[1] // 2))
        self.norm = nn.BatchNorm2d(out_dim)
        self.fc1 = nn.Linear(out_dim//2, out_dim//2)
        self.sublayernorm = nn.LayerNorm(out_dim//2)
        self.drop = nn.Dropout(drop)
        self.act = nn.GELU()
        self.dwconv = BN_Conv2d(out_dim, out_dim, 3, 1, 1, bias=False)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x):
        x = self.proj(x)
        B, C, H, W = x.shape
        # x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)
        y, z = torch.split(x, C // 2, dim= 1)

        xn = F.pad(y, (self.pad, self.pad, self.pad, self.pad), "constant", 0)
        xs = torch.chunk(xn, self.shift_size, 1)

        x1_shift = [torch.roll(x_c, shift, 2) for x_c, shift in zip(xs, range(-self.pad, self.pad + 1))]
        x1_cat = torch.cat(x1_shift, 1)
        x1_cat = torch.narrow(x1_cat, 2, self.pad, H)
        x1_s = torch.narrow(x1_cat, 3, self.pad, W)
        x_shift_r = self.fc1(x1_s.permute(0, 2, 3, 1))
        x_shift_r = self.sublayernorm(x_shift_r)
        x_shift_r = self.drop(x_shift_r)

        xn = F.pad(z, (self.pad, self.pad, self.pad, self.pad), "constant", 0)
        xs = torch.chunk(xn, self.shift_size, 1)
        x2_shift = [torch.roll(x_c, shift, 3) for x_c, shift in zip(xs, range(-self.pad, self.pad + 1))]
        x2_cat = torch.cat(x2_shift, 1)
        x2_cat = torch.narrow(x2_cat, 2, self.pad, H)
        x2_s = torch.narrow(x2_cat, 3, self.pad, W)
        x_shift_c = self.fc1(x2_s.permute(0, 2, 3, 1))
        x_shift_c = self.sublayernorm(x_shift_c)
        x_shift_c = self.drop(x_shift_c)

        y_cat = torch.cat((x_shift_r.permute(0, 3, 1, 2),x_shift_c.permute(0, 3, 1, 2)),1)

        x = self.dwconv(y_cat) # 8, 1024, 64

        return x

class DynamicMLPBlock(nn.Module):
    """ Image to Patch Embedding
    """
    def __init__(self, in_dim, out_dim=768, mlp_ratio=4., drop = 0.4, img_size=224, patch_size=7, stride=4):
        super().__init__()

        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        self.drop = drop
        self.shift_size = 5
        self.pad = self.shift_size // 2

        self.H, self.W = img_size[0] // patch_size[0], img_size[1] // patch_size[1]
        self.num_patches = self.H * self.W
        self.proj = nn.Conv2d(in_dim, out_dim, kernel_size=patch_size, stride=stride,
                              padding=(patch_size[0] // 2, patch_size[1] // 2))
        self.norm = nn.BatchNorm2d(out_dim)
        self.fc1 = nn.Linear(in_dim, in_dim)
        self.sublayernorm = nn.LayerNorm(in_dim)
        self.drop = nn.Dropout(drop)
        self.fc2 = nn.Linear(out_dim, out_dim * out_dim // 4)
        self.fc3 = nn.Linear(out_dim, out_dim // 4)
        self.act = nn.GELU()

        self.dwconv = BN_Conv2d(out_dim, out_dim, 3, 1, 1, bias=False)
        self.drop_path = nn.Identity()
        self.norm2 = nn.LayerNorm(out_dim)
        mlp_hidden_dim = int(out_dim * mlp_ratio)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x):
        x = self.proj(x)
        B, C, H, W = x.shape
        # x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)
        y, z = torch.split(x, C // 2, dim= 1)

        x_shift_r = self.fc1(y.permute(0, 2, 3, 1))
        # x_shift_r = x_shift_r.permute(0, 3, 1, 2)
        x_shift_r = self.sublayernorm(x_shift_r)
        x_shift_r = self.drop(x_shift_r)

        # x_shift_c = x2_s.transpose(1, 2)
        x_shift_c = self.fc1(z.permute(0, 2, 3, 1))
        # x_shift_c = x_shift_c.permute(0, 3, 1, 2)
        x_shift_c = self.sublayernorm(x_shift_c)
        x_shift_c = self.drop(x_shift_c)

        y_cat = torch.cat((x_shift_r.permute(0, 3, 1, 2),x_shift_c.permute(0, 3, 1, 2)),1)
        y_cat = self.fc2(y_cat.permute(0, 2, 3, 1))
        y_cat = self.act(y_cat)
        y_cat = self.drop(y_cat) # 8,
        y_cat = y_cat.view(-1, C // 4, C)

        dy = self.fc3(x.permute(0, 2, 3, 1))
        dy = dy.reshape(-1, C // 4).unsqueeze(1)

        x_dy = torch.bmm(dy, y_cat)
        x_dy = x_dy.view(B, H, W, C).permute(0, 3, 1, 2)
        x = self.dwconv(x_dy) # 8, 1024, 64

        return x

class GlobalPerceptron(nn.Module):

    def __init__(self, input_channels, internal_neurons):
        super(GlobalPerceptron, self).__init__()
        self.fc1 = nn.Conv2d(in_channels=input_channels, out_channels=internal_neurons, kernel_size=1, stride=1, bias=True)
        self.fc2 = nn.Conv2d(in_channels=internal_neurons, out_channels=input_channels, kernel_size=1, stride=1, bias=True)
        self.input_channels = input_channels

    def forward(self, inputs):
        x = F.adaptive_avg_pool2d(inputs, output_size=(1, 1))
        x = self.fc1(x)
        x = F.relu(x, inplace=True)
        x = self.fc2(x)
        x = F.sigmoid(x)
        x = x.view(-1, self.input_channels, 1, 1)
        return x

class AdaAttention_Block(nn.Module):
    """
    Adapetive Channel Convolution Block
    """
    def __init__(self, up_ch, low_ch):
        super(AdaAttention_Block, self).__init__()

        globalperceptron_reduce = 2

        self.conv1 = BN_Conv2d(up_ch, low_ch, 1, 1, 0, bias=False)
        self.conv21 = BN_Conv2d(low_ch, up_ch, 1, 1, 0, bias=False)

        self.ch_conv3 = nn.Sequential(
            BN_Conv2d(low_ch, low_ch // 2, 3, 1, 1, bias=False),
            BN_Conv2d(low_ch // 2, low_ch, 1, 1, 0, bias=False),
        )

        self.res_conv3 = BN_Conv2d(low_ch, up_ch, 1, 1, 0, bias=False)

        self.dial3 = nn.Sequential(
            BN_Conv2d(up_ch, up_ch // 2, 3, 1, 2, bias=False,dilation= 2),
            # BN_Conv2d(up_ch, low_ch // 2, 3, 1, 1, bias=False),
            # BN_Conv2d(low_ch // 2, low_ch // 2, 3, 1, 1, bias=False),
            BN_Conv2d(up_ch // 2, low_ch, 1, 1, 0, bias=False),
        )

        self.relu = nn.ReLU(inplace=True)
        self.delta = nn.Sigmoid()
        self.gp = GlobalPerceptron(input_channels=low_ch, internal_neurons=low_ch // globalperceptron_reduce)

    def forward(self, up, low):

        # global attention block
        c3 = self.ch_conv3(low)
        d3 = self.dial3(up)
        cd3 = c3 + d3
        gp = self.gp(cd3)
        out_layer1 = c3 * gp + d3 * gp  # channel -- low channel

        res_low = self.res_conv3(low)

        # local attention block
        # layer2 = self.conv21(res_low)
        in_layer2 = self.conv21(out_layer1)
        out = res_low + in_layer2
        relu = self.relu(out)
        delta =  self.delta(relu)
        y = delta * res_low + (1-delta) * in_layer2

        up1 = self.conv1(up)
        out = torch.cat([up1, y], 1)

        return out


class HAM_Block(nn.Module):
    """
    Adapetive Channel Convolution Block
    """
    def __init__(self, up_ch, low_ch):
        super(HAM_Block, self).__init__()

        globalperceptron_reduce = 2

        self.conv1 = BN_Conv2d(up_ch, up_ch, 1, 1, 0, bias=False)
        self.conv21 = BN_Conv2d(low_ch, low_ch, 1, 1, 0, bias=False)

        self.ch_conv3 = nn.Sequential(
            BN_Conv2d(low_ch, low_ch // 2, 3, 1, 1, bias=False),
            BN_Conv2d(low_ch // 2, low_ch, 1, 1, 0, bias=False),
        )

        self.res_conv3 = BN_Conv2d(low_ch, low_ch, 3, 1, 1, bias=False)

        self.dial3 = nn.Sequential(
            BN_Conv2d(up_ch, up_ch // 2, 3, 1, 2, bias=False,dilation= 2),
            BN_Conv2d(up_ch // 2, low_ch, 1, 1, 0, bias=False),
        )

        self.relu = nn.ReLU(inplace=True)
        self.delta = nn.Sigmoid()
        self.gp = GlobalPerceptron(input_channels=low_ch, internal_neurons=low_ch // globalperceptron_reduce)

    def forward(self, up, low):

        # global attention block
        c3 = self.ch_conv3(low)
        d3 = self.dial3(up)
        cd3 =  c3 + d3
        gp = self.gp(cd3)
        out_layer1 = c3 * gp + d3 * (1-gp)  # channel -- low channel

        res_low = self.res_conv3(low)

        # local attention block
        # layer2 = self.conv21(res_low)
        in_layer2 = self.conv21(out_layer1)
        out = res_low + in_layer2
        relu = self.relu(out)
        delta =  self.delta(relu)
        y = delta * res_low + (1-delta) * in_layer2

        up1 = self.conv1(up)
        out = torch.cat([up1, y], 1)

        return out



class Attention_block(nn.Module):
    '''
      Attention block/mechanism
    '''
    def __init__(self, in_dims, out_dims):
        super(Attention_block, self).__init__()

        self.conv_x = nn.Conv2d(out_dims, out_dims, kernel_size=3, stride=1, padding=1, bias=True)
        #
        self.pool_x = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv_g = nn.Conv2d(in_dims, out_dims, kernel_size=3, stride=1, padding=1, bias=True)
        self.relu = nn.ReLU(inplace=True)

        self.Up = nn.Upsample(scale_factor=2, mode='bilinear')
        self.psi = nn.Conv2d(out_dims, 1, kernel_size=3, stride=1, padding=1, bias=True)

    def forward(self, x, gating, drop_rate=0.25):
        B, C, H, W = x.shape
        shape_x = x.shape

        theta_x = self.conv_x(x)
        theta_x = self.pool_x(theta_x)

        # gating signal ""
        phi_g = self.conv_g(gating)

        # Add components
        concat_xg = torch.add(phi_g, theta_x)
        act_xg = self.relu(concat_xg)

        # Apply convolution
        psi = self.psi(act_xg)

        # Apply sigmoid activation
        sigmoid_xg = torch.sigmoid(psi)

        # UpSample and resample to correct size
        upsample_psi = self.Up(sigmoid_xg)

        y = torch.matmul(x,upsample_psi)

        return y

class Attention_block_simple(nn.Module):
    '''
      Attention block/mechanism
    '''
    def __init__(self, in_dims, out_dims,reduction=16):
        super(Attention_block_simple, self).__init__()

        self.conv_x = nn.Conv2d(out_dims, out_dims, kernel_size=3, stride=1, padding=1, bias=True)

        self.pool_x = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv_g = nn.Conv2d(in_dims//2, out_dims, kernel_size=3, stride=1, padding=1, bias=True)
        self.up_g = nn.Upsample(scale_factor=2, mode='bilinear')
        self.relu = nn.ReLU(inplace=True)

        self.Up = nn.Upsample(scale_factor=2, mode='bilinear')
        self.psi = nn.Conv2d(out_dims, 1, kernel_size=3, stride=1, padding=1, bias=True)

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.norm = nn.BatchNorm2d(in_dims)
        self.fc = nn.Sequential(
            nn.Linear(in_dims//2, in_dims // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(in_dims// reduction, out_dims, bias=False),
            nn.Sigmoid()
        )


    def forward(self, x, gating, drop_rate=0.25):
        b, c, h, w = gating.shape
        B, C, H, W = x.shape
        theta_x = self.conv_x(x)

        gating = self.norm(gating)
        gating_1, gating_2 = torch.split(gating, c // 2, dim=1)

        # gating signal ""
        phi_g = self.conv_g(gating_1)
        phi_g = self.up_g(phi_g)

        # Add components
        concat_xg = torch.add(phi_g, theta_x)
        act_xg = self.relu(concat_xg)

        # Apply convolution
        psi = self.psi(act_xg)

        # Apply sigmoid activation
        sigmoid_xg = torch.sigmoid(psi)

        # UpSample and resample to correct size
        # upsample_psi = self.Up(sigmoid_xg)

        # upsample_psi = upsample_psi.view(-1, H, W)
        x = torch.matmul(x, sigmoid_xg)

        # SElayer
        y = self.avg_pool(gating_2)
        y = y.view(b, c//2)
        y = self.fc(y)
        y= y.view(B, C, 1, 1)

        return x * y.expand_as(x)

        # return y



class Attention_SE2(nn.Module):
    '''
      Attention block/mechanism
    '''
    def __init__(self, in_dims, out_dims,reduction=16):
        super(Attention_SE2, self).__init__()

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.norm = nn.BatchNorm2d(out_dims)
        self.fc = nn.Sequential(
            nn.Linear(out_dims, out_dims // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(out_dims// reduction, out_dims, bias=False),
            nn.Sigmoid()
        )
        self.ch_conv3 = nn.Sequential(
            BN_Conv2d(out_dims, out_dims // 2, 3, 1, 1, bias=False),
            BN_Conv2d(out_dims // 2, out_dims, 1, 1, 0, bias=False),
        )

        self.dial3 = nn.Sequential(
            BN_Conv2d(in_dims, in_dims // 2, 3, 1, 2, bias=False, dilation=2),
            BN_Conv2d(in_dims // 2, out_dims, 1, 1, 0, bias=False),
        )

    def forward(self,gating, x, drop_rate=0.25):
        b, c, h, w = gating.shape
        B, C, H, W = x.shape

        # gating = self.norm(gating)

        c3 = self.ch_conv3(x)
        d3 = self.dial3(gating)
        cd3 =  c3 + d3
        cd3 = self.norm(cd3)
        cd3 = self.avg_pool(cd3)
        cd3 = cd3.view(B, C)
        # cd3  = self.fc(cd3)
        gp = self.fc(cd3)
        gp = gp.view(B, C, 1, 1)
        out_layer1 = x * gp.expand_as(x)# channel -- low channel

        print(out_layer1.size())

        return out_layer1



# # Up operator Jiangxiong Fang
class Attention_block_Up(nn.Module):
    '''
      Attention block/mechanism
    '''
    def __init__(self, in_dims, out_dims):
        super(Attention_block_Up, self).__init__()

        self.conv_x = nn.Conv2d(out_dims, out_dims, kernel_size=3, stride=1, padding=1, bias=True)
        self.up_g = nn.Upsample(scale_factor=2, mode='bilinear')
        self.conv_g = nn.Conv2d(in_dims, out_dims, kernel_size=3, stride=1, padding=1, bias=True)
        self.relu = nn.ReLU(inplace=True)

        self.Up = nn.Upsample(scale_factor=2, mode='bilinear')
        self.psi = nn.Conv2d(out_dims, 1, kernel_size=3, stride=1, padding=1, bias=True)

    def forward(self, x, gating, drop_rate=0.25):

        theta_x = self.conv_x(x)
        # theta_x = self.pool_x(theta_x)

        # gating signal ""
        phi_g = self.conv_g(gating)
        phi_g = self.up_g(phi_g)

        # Add components
        concat_xg = torch.add(phi_g, theta_x)
        act_xg = self.relu(concat_xg)

        # Apply convolution
        psi = self.psi(act_xg)

        # Apply sigmoid activation
        sigmoid_xg = torch.sigmoid(psi)

        # upsample_psi = self.Up(sigmoid_xg)

        # upsample_psi = upsample_psi.view(-1, H, W)
        y = torch.matmul(x,sigmoid_xg)
        return y


class DoubleMLPUNet(nn.Module):
    """
    UNet - Unet with down-upsampling operation
    Paper : https://arxiv.org/abs/1505.04597
    """

    def __init__(self, in_ch=1, out_ch=3):
        super(DoubleMLPUNet, self).__init__()

        n1 = 1

        self.in_chanel = in_ch
        self.out_chanel = out_ch

        filters = [16, 32, 64, 128, 256, 512]
        embed_dims = [64, 128, 256, 512]

        drop_rate = 0.2
        img_size = 512

        self.up =  nn.Upsample(scale_factor=2)
        self.down = nn.MaxPool2d(kernel_size=2, stride=2)

        self.MConv0 = Multiscaleconv_small(in_ch, filters[0])
        self.MConv1 = Multiscaleconv_small(filters[0], filters[1])
        self.MConv2 = Multiscaleconv_small(filters[1], filters[2])
        self.MConv3 = Multiscaleconv_small(filters[2], filters[3])
        self.MConv4 = Multiscaleconv_small(filters[3], filters[4])

        self.Maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.mlp_block1 = MLPBlock(in_dim=embed_dims[0], out_dim=embed_dims[1], mlp_ratio=1, drop=drop_rate,
                                   img_size=img_size // 4, patch_size=3, stride=2)
        self.mlp_block2 = MLPBlock(in_dim=embed_dims[1], out_dim=embed_dims[2], mlp_ratio=1, drop=drop_rate,
                                   img_size=img_size // 8, patch_size=3, stride=2)


        self.Up_conv5 = conv_block(filters[4], filters[4])
        self.Up5 = up_conv(filters[4], filters[4], 2)

        self.Up4 = up_conv(filters[3], filters[3], 2)
        self.Up_conv4 = conv_block(filters[4]+filters[3], filters[3])

        self.Up3 = up_conv(filters[2], filters[2], 2)
        self.Up_conv3 = conv_block(filters[3]+filters[2], filters[2])

        self.Up2 = up_conv(filters[1], filters[1], 2)
        self.Up_conv2 = conv_block(+filters[2]+filters[1], filters[1])

        self.Up1 = up_conv(filters[1], filters[0], 2)
        self.Up_conv1 = conv_block(filters[1]+filters[0], filters[0])

        self.Conv = nn.Conv2d(filters[0], out_ch, kernel_size=1, stride=1, padding=0)
        self.do_ds = True

        # self.active = torch.nn.Sigmoid()self.MConv0(x)

    def forward(self, x):
        B = x.shape[0]

        e1 = self.MConv0(x)          #16 * 512 * 512
        pool1 = self.Maxpool1(e1)    #16*256*256

        e2 = self.MConv1(pool1)
        pool2 = self.Maxpool2(e2)    #32*128*128

        e3 = self.MConv2(pool2)
        # pool3 = self.Maxpool3(e3)     #64*64*64

        # e4 = self.MConv3(pool3)
        e4 = self.mlp_block1(e3) #4, 128,64,64

        e5 = self.mlp_block2(e4)

        # pool4 = self.Maxpool4(e4)     #128*32*32
        # e5 = self.MConv4(pool4)                #256*32*32

        d5 = self.Up_conv5(e5)              #input: 256*32*32,  output: 256*32*32

        up4 = self.Up5(d5)                  #input: 256*32*32,  output: 256*64*64
        d4 = torch.cat((e4, up4), dim=1)    #input: 384*64*64
        d4 = self.Up_conv4(d4)              #input: 384*64*64,  output: 128*64*64

        up3 = self.Up4(d4)                  #input: 128*64*64,  output: 128*128*128
        d3 = torch.cat((up3, e3), dim=1)    #input: 192*128*128
        d3 = self.Up_conv3(d3)              #input: 192*128*128, output: 64*128*128

        up2 = self.Up3(d3)                  #input: 64*128*128,  output: 64*256*256
        d2 = torch.cat((up2, e2), dim=1)    #input: 96*256*256
        d2 = self.Up_conv2(d2)              #input: 96*256*256, output: 32*256*256

        up1 = self.Up2(d2)                  #input: 32*256*256,  output: 128*512*512
        d1 = torch.cat((up1, e1), dim=1)    #input: 48*512*512
        d1 = self.Up_conv1(d1)              #input: 48*512*512, output: 64*128*128

        out = self.Conv(d1)

        return out


##### The fifth layer is inserted into the attention block
class DoubleMLPUNet_Att_Layer5(nn.Module):
    """
    UNet - Unet with down-upsampling operation
    Paper : https://arxiv.org/abs/1505.04597
    """

    def __init__(self, in_ch=1, out_ch=3, type = 0):
        super(DoubleMLPUNet_Att_Layer5, self).__init__()

        n1 = 1
        self.type = type

        self.in_chanel = in_ch
        self.out_chanel = out_ch

        filters = [16, 32, 64, 128, 256, 512]

        embed_dims = [64, 128, 256, 512]

        drop_rate = 0.2
        img_size = 512

        self.up =  nn.Upsample(scale_factor=2)
        self.down = nn.MaxPool2d(kernel_size=2, stride=2)

        self.MConv0 = Multiscaleconv_small(in_ch, filters[0])
        self.MConv1 = Multiscaleconv_small(filters[0], filters[1])
        self.MConv2 = Multiscaleconv_small(filters[1], filters[2])
        self.MConv3 = Multiscaleconv_small(filters[2], filters[3])
        self.MConv4 = Multiscaleconv_small(filters[3], filters[4])

        self.Maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.mlp_block1 = LightMLPBlock(in_dim=embed_dims[0], out_dim=embed_dims[1], mlp_ratio=1, drop=drop_rate,
                                   img_size=img_size // 4, patch_size=3, stride=2)
        self.mlp_block2 = LightMLPBlock(in_dim=embed_dims[1], out_dim=embed_dims[2], mlp_ratio=1, drop=drop_rate,
                                   img_size=img_size // 8, patch_size=3, stride=2)

        self.att_block1 = Attention_block(in_dims=filters[4],out_dims=filters[3])
        self.att_block2 = Attention_block(in_dims=filters[3], out_dims=filters[2])
        self.att_block3 = Attention_block(in_dims=filters[2], out_dims=filters[1])
        self.att_block4 = Attention_block(in_dims=filters[1], out_dims=filters[0])

        self.Up_conv5 = conv_block(filters[4], filters[4])
        self.Up5 = up_conv(filters[4], filters[4], 2)

        self.Up4 = up_conv(filters[3], filters[3], 2)
        self.Up_conv4 = conv_block(filters[4] + filters[3], filters[3])

        self.Up3 = up_conv(filters[2], filters[2], 2)
        self.Up_conv3 = conv_block(filters[3] + filters[2], filters[2])

        self.Up2 = up_conv(filters[1], filters[1], 2)
        self.Up_conv2 = conv_block(+filters[2] + filters[1], filters[1])

        self.Up1 = up_conv(filters[1], filters[0], 2)
        self.Up_conv1 = conv_block(filters[1] + filters[0], filters[0])

        self.Conv = nn.Conv2d(filters[0], out_ch, kernel_size=1, stride=1, padding=0)
        self.do_ds = True

        # self.active = torch.nn.Sigmoid()self.MConv0(x)

    def forward(self, x):
        B = x.shape[0]

        e1 = self.MConv0(x)  # 16 * 512 * 512
        pool1 = self.Maxpool1(e1)  # 16*256*256

        e2 = self.MConv1(pool1)
        pool2 = self.Maxpool2(e2)  # 32*128*128

        e3 = self.MConv2(pool2)
        # pool3 = self.Maxpool3(e3)     #64*64*64

        # e4 = self.MConv3(pool3)
        e4 = self.mlp_block1(e3)        # 4, 128,64,64

        e5 = self.mlp_block2(e4)

        # pool4 = self.Maxpool4(e4)     #128*32*32
        # e5 = self.MConv4(pool4)                #256*32*32

        if self.type == 5:
            d5 = self.Up_conv5(e5)  # input: 256*32*32,  output: 256*32*32
            att_5 = self.att_block1(e4, d5)
            up4 = self.Up5(d5)                  #input: 256*32*32,  output: 256*64*64
            d4 = torch.cat((att_5, up4), dim=1)  # input: 384*64*64
            d4 = self.Up_conv4(d4)  # input: 384*64*64,  output: 128*64*64
            # att_4 = self.att_b(e3, d4)
            up3 = self.Up4(d4)  # input: 128*64*64,  output: 128*128*128
            d3 = torch.cat((e3, up3), dim=1)  # input: 192*128*128
        elif self.type == 4:
            d5 = self.Up_conv5(e5)  # input: 256*32*32,  output: 256*32*32
            att_5 = self.att_block1(e4, d5)
            up4 = self.Up5(d5)  # input: 256*32*32,  output: 256*64*64
            d4 = torch.cat((att_5, up4), dim=1)  # input: 384*64*64

            d4 = self.Up_conv4(d4)  # input: 384*64*64,  output: 128*64*64
            att_4 = self.att_block2(e3, d4)
            up3 = self.Up4(d4)  # input: 128*64*64,  output: 128*128*128
            d3 = torch.cat((att_4, up3), dim=1)  # input: 192*128*128

        else:
            d5 = self.Up_conv5(e5)  # input: 256*32*32,  output: 256*32*32
            #        att_5 = self.att_block1(e4, d5)
            up4 = self.Up5(d5)  # input: 256*32*32,  output: 256*64*64
            d4 = torch.cat((e4, up4), dim=1)  # input: 384*64*64
            d4 = self.Up_conv4(d4)  # input: 384*64*64,  output: 128*64*64
            # att_4 = self.att_b(e3, d4)
            up3 = self.Up4(d4)  # input: 128*64*64,  output: 128*128*128
            d3 = torch.cat((e3, up3), dim=1)  # input: 192*128*128

        d3 = self.Up_conv3(d3)  # input: 192*128*128, output: 64*128*128
        # att_3 = self.att_block3(e2, d3)
        up2 = self.Up3(d3)  # input: 64*128*128,  output: 64*256*256
        d2 = torch.cat((e2, up2), dim=1)  # input: 96*256*256

        d2 = self.Up_conv2(d2)  # input: 96*256*256, output: 32*256*256
        # att_2 = self.att_block4(e1, d2)
        up1 = self.Up2(d2)  # input: 32*256*256,  output: 128*512*512
        d1 = torch.cat((e1, up1), dim=1)  # input: 48*512*512

        d1 = self.Up_conv1(d1)  # input: 48*512*512, output: 64*128*128
        out = self.Conv(d1)

        return out


##### The fifth layer is inserted into the attention block
class DoubleMLPUNet_DownAtt(nn.Module):
    """
    UNet - Unet with down-upsampling operation
    Paper : https://arxiv.org/abs/1505.04597
    """

    def __init__(self, in_ch=1, out_ch=3):
        super(DoubleMLPUNet_DownAtt, self).__init__()

        n1 = 1

        self.in_chanel = in_ch
        self.out_chanel = out_ch

        filters = [16, 32, 64, 128, 256, 512]

        embed_dims = [64, 128, 256, 512]

        drop_rate = 0.2
        img_size = 512

        self.up =  nn.Upsample(scale_factor=2)
        self.down = nn.MaxPool2d(kernel_size=2, stride=2)

        self.MConv0 = Multiscaleconv_small(in_ch, filters[0])
        self.MConv1 = Multiscaleconv_small(filters[0], filters[1])
        self.MConv2 = Multiscaleconv_small(filters[1], filters[2])
        self.MConv3 = Multiscaleconv_small(filters[2], filters[3])
        self.MConv4 = Multiscaleconv_small(filters[3], filters[4])

        self.Maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.mlp_block1 = LightMLPBlock(in_dim=embed_dims[0], out_dim=embed_dims[1], mlp_ratio=1, drop=drop_rate,
                                   img_size=img_size // 4, patch_size=3, stride=2)
        self.mlp_block2 = LightMLPBlock(in_dim=embed_dims[1], out_dim=embed_dims[2], mlp_ratio=1, drop=drop_rate,
                                   img_size=img_size // 8, patch_size=3, stride=2)

        self.att_block1 = Attention_block(in_dims=filters[4],out_dims=filters[3])
        self.att_block2 = Attention_block(in_dims=filters[3], out_dims=filters[2])
        self.att_block3 = Attention_block(in_dims=filters[2], out_dims=filters[1])
        self.att_block4 = Attention_block(in_dims=filters[1], out_dims=filters[0])

        self.Up_conv5 = conv_block(filters[4], filters[4])
        self.Up5 = up_conv(filters[4], filters[4], 2)

        self.Up4 = up_conv(filters[3], filters[3], 2)
        self.Up_conv4 = conv_block(filters[4] + filters[3], filters[3])

        self.Up3 = up_conv(filters[2], filters[2], 2)
        self.Up_conv3 = conv_block(filters[3] + filters[2], filters[2])

        self.Up2 = up_conv(filters[1], filters[1], 2)
        self.Up_conv2 = conv_block(filters[2] + filters[1], filters[1])

        self.Up1 = up_conv(filters[1], filters[0], 2)
        self.Up_conv1 = conv_block(filters[1] + filters[0], filters[0])

        self.Conv = nn.Conv2d(filters[0], out_ch, kernel_size=1, stride=1, padding=0)
        self.do_ds = True

        # self.active = torch.nn.Sigmoid()self.MConv0(x)

    def forward(self, x):
        B = x.shape[0]

        e1 = self.MConv0(x)  # 16 * 512 * 512
        pool1 = self.Maxpool1(e1)  # 16*256*256

        e2 = self.MConv1(pool1)
        pool2 = self.Maxpool2(e2)  # 32*128*128

        e3 = self.MConv2(pool2)
        # pool3 = self.Maxpool3(e3)     #64*64*64

        # e4 = self.MConv3(pool3)
        e4 = self.mlp_block1(e3)        # 4, 128,64,64

        e5 = self.mlp_block2(e4)

        d5 = self.Up_conv5(e5)  # input: 256*32*32,  output: 256*32*32
        att_5 = self.att_block1(e4, d5)
        up4 = self.Up5(d5)                  #input: 256*32*32,  output: 256*64*64
        d4 = torch.cat((att_5, up4), dim=1)  # input: 384*64*64

        d4 = self.Up_conv4(d4)  # input: 384*64*64,  output: 128*64*64
        att_4 = self.att_block2(e3, d4)
        up3 = self.Up4(d4)  # input: 128*64*64,  output: 128*128*128
        d3 = torch.cat((att_4, up3), dim=1)  # input: 192*128*128

        d3 = self.Up_conv3(d3)  # input: 192*128*128, output: 64*128*128
        att_3 = self.att_block3(e2, d3)
        up2 = self.Up3(d3)  # input: 64*128*128,  output: 64*256*256
        d2 = torch.cat((up2, att_3), dim=1)  # input: 96*256*256

        d2 = self.Up_conv2(d2)  # input: 96*256*256, output: 32*256*256
        att_2 = self.att_block4(e1, d2)
        up1 = self.Up2(d2)  # input: 32*256*256,  output: 128*512*512
        d1 = torch.cat((up1, att_2), dim=1)  # input: 48*512*512

        d1 = self.Up_conv1(d1)  # input: 48*512*512, output: 64*128*128
        out = self.Conv(d1)

        return out

##### The fifth layer is inserted into the attention block
class DoubleMLPUNet_UpAtt(nn.Module):
    """
    UNet - Unet with down-upsampling operation
    Paper : https://arxiv.org/abs/1505.04597
    """

    def __init__(self, in_ch=1, out_ch=3):
        super(DoubleMLPUNet_UpAtt, self).__init__()

        n1 = 1

        self.in_chanel = in_ch
        self.out_chanel = out_ch

        filters = [16, 32, 64, 128, 256, 512]

        embed_dims = [64, 128, 256, 512]

        drop_rate = 0.2
        img_size = 512

        self.up =  nn.Upsample(scale_factor=2)
        self.down = nn.MaxPool2d(kernel_size=2, stride=2)

        self.MConv0 = Multiscaleconv_small(in_ch, filters[0])
        self.MConv1 = Multiscaleconv_small(filters[0], filters[1])
        self.MConv2 = Multiscaleconv_small(filters[1], filters[2])
        self.MConv3 = Multiscaleconv_small(filters[2], filters[3])
        self.MConv4 = Multiscaleconv_small(filters[3], filters[4])

        self.Maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.mlp_block1 = LightMLPBlock(in_dim=embed_dims[0], out_dim=embed_dims[1], mlp_ratio=1, drop=drop_rate,
                                   img_size=img_size // 4, patch_size=3, stride=2)
        self.mlp_block2 = LightMLPBlock(in_dim=embed_dims[1], out_dim=embed_dims[2], mlp_ratio=1, drop=drop_rate,
                                   img_size=img_size // 8, patch_size=3, stride=2)

        self.att_block1 = Attention_block_Up(in_dims=filters[4],out_dims=filters[3])
        self.att_block2 = Attention_block_Up(in_dims=filters[3], out_dims=filters[2])
        self.att_block3 = Attention_block_Up(in_dims=filters[2], out_dims=filters[1])
        self.att_block4 = Attention_block_Up(in_dims=filters[1], out_dims=filters[0])

        self.Up_conv5 = conv_block(filters[4], filters[4])
        self.Up5 = up_conv(filters[4], filters[4], 2)

        self.Up4 = up_conv(filters[3], filters[3], 2)
        self.Up_conv4 = conv_block(filters[4] + filters[3], filters[3])

        self.Up3 = up_conv(filters[2], filters[2], 2)
        self.Up_conv3 = conv_block(filters[3] + filters[2], filters[2])

        self.Up2 = up_conv(filters[1], filters[1], 2)
        self.Up_conv2 = conv_block(filters[2] + filters[1], filters[1])

        self.Up1 = up_conv(filters[1], filters[0], 2)
        self.Up_conv1 = conv_block(filters[1] + filters[0], filters[0])

        self.Conv = nn.Conv2d(filters[0], out_ch, kernel_size=1, stride=1, padding=0)
        self.do_ds = True

        # self.active = torch.nn.Sigmoid()self.MConv0(x)

    def forward(self, x):
        B = x.shape[0]

        e1 = self.MConv0(x)  # 16 * 512 * 512
        pool1 = self.Maxpool1(e1)  # 16*256*256

        e2 = self.MConv1(pool1)
        pool2 = self.Maxpool2(e2)  # 32*128*128

        e3 = self.MConv2(pool2)
        # pool3 = self.Maxpool3(e3)     #64*64*64

        # e4 = self.MConv3(pool3)
        e4 = self.mlp_block1(e3)        # 4, 128,64,64

        e5 = self.mlp_block2(e4)

        d5 = self.Up_conv5(e5)  # input: 256*32*32,  output: 256*32*32
        att_5 = self.att_block1(e4, d5)
        up4 = self.Up5(d5)                  #input: 256*32*32,  output: 256*64*64
        d4 = torch.cat((att_5, up4), dim=1)  # input: 384*64*64

        d4 = self.Up_conv4(d4)  # input: 384*64*64,  output: 128*64*64
        att_4 = self.att_block2(e3, d4)
        up3 = self.Up4(d4)  # input: 128*64*64,  output: 128*128*128
        d3 = torch.cat((att_4, up3), dim=1)  # input: 192*128*128

        d3 = self.Up_conv3(d3)  # input: 192*128*128, output: 64*128*128
        att_3 = self.att_block3(e2, d3)
        up2 = self.Up3(d3)  # input: 64*128*128,  output: 64*256*256
        d2 = torch.cat((up2, att_3), dim=1)  # input: 96*256*256

        d2 = self.Up_conv2(d2)  # input: 96*256*256, output: 32*256*256
        att_2 = self.att_block4(e1, d2)
        up1 = self.Up2(d2)  # input: 32*256*256,  output: 128*512*512
        d1 = torch.cat((up1, att_2), dim=1)  # input: 48*512*512

        d1 = self.Up_conv1(d1)  # input: 48*512*512, output: 64*128*128
        out = self.Conv(d1)

        return out


class DoubleMLPUNet_SimpleAtt(nn.Module):
    """
    UNet - Unet with down-upsampling operation
    Paper : https://arxiv.org/abs/1505.04597
    """

    def __init__(self, in_ch=1, out_ch=3):
        super(DoubleMLPUNet_SimpleAtt, self).__init__()

        n1 = 1

        self.in_chanel = in_ch
        self.out_chanel = out_ch

        filters = [16, 32, 64, 128, 256, 512]

        embed_dims = [64, 128, 256, 512]

        drop_rate = 0.2
        img_size = 512

        self.up =  nn.Upsample(scale_factor=2)
        self.down = nn.MaxPool2d(kernel_size=2, stride=2)

        self.MConv0 = Multiscaleconv_small(3*in_ch, filters[0])
        self.MConv1 = Multiscaleconv_small(filters[0], filters[1])
        self.MConv2 = Multiscaleconv_small(filters[1], filters[2])
        self.MConv3 = Multiscaleconv_small(filters[2], filters[3])
        self.MConv4 = Multiscaleconv_small(filters[3], filters[4])

        self.Maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.mlp_block1 = LightMLPBlock(in_dim=embed_dims[0], out_dim=embed_dims[1], mlp_ratio=1, drop=drop_rate,
                                   img_size=img_size // 4, patch_size=3, stride=2)
        self.mlp_block2 = LightMLPBlock(in_dim=embed_dims[1], out_dim=embed_dims[2], mlp_ratio=1, drop=drop_rate,
                                   img_size=img_size // 8, patch_size=3, stride=2)

        self.att_block1 = Attention_block(in_dims=filters[4],out_dims=filters[3])
        self.att_block2 = Attention_block(in_dims=filters[3], out_dims=filters[2])
        self.att_block3 = Attention_block(in_dims=filters[2], out_dims=filters[1])
        self.att_block4 = Attention_block(in_dims=filters[1], out_dims=filters[0])

        self.Up_conv5 = conv_block(filters[4], filters[4])
        self.Up5 = up_conv(filters[4], filters[4], 2)

        self.Up4 = up_conv(filters[3], filters[3], 2)
        self.Up_conv4 = conv_block(filters[4] + filters[3], filters[3])

        self.Up3 = up_conv(filters[2], filters[2], 2)
        self.Up_conv3 = conv_block(filters[3] + filters[2], filters[2])

        self.Up2 = up_conv(filters[1], filters[1], 2)
        self.Up_conv2 = conv_block(+filters[2] + filters[1], filters[1])

        self.Up1 = up_conv(filters[1], filters[0], 2)
        self.Up_conv1 = conv_block(filters[1] + filters[0], filters[0])

        self.Conv = nn.Conv2d(filters[0], out_ch, kernel_size=1, stride=1, padding=0)
        self.do_ds = True

        # self.active = torch.nn.Sigmoid()self.MConv0(x)

    def forward(self, x):
        B = x.shape[0]
        if x.size()[1] == 1:
            x = x.repeat(1,3,1,1)

        e1 = self.MConv0(x)  # 16 * 512 * 512
        pool1 = self.Maxpool1(e1)  # 16*256*256

        e2 = self.MConv1(pool1)
        pool2 = self.Maxpool2(e2)  # 32*128*128

        e3 = self.MConv2(pool2)
        # pool3 = self.Maxpool3(e3)     #64*64*64

        # e4 = self.MConv3(pool3)
        e4 = self.mlp_block1(e3)  # 4, 128,64,64

        e5 = self.mlp_block2(e4)

        # pool4 = self.Maxpool4(e4)     #128*32*32
        # e5 = self.MConv4(pool4)                #256*32*32

        d5 = self.Up_conv5(e5)  # input: 256*32*32,  output: 256*32*32

        att_5 = self.att_block1(e4, d5)
        up4 = self.Up5(d5)                  #input: 256*32*32,  output: 256*64*64
        d4 = torch.cat((up4, att_5), dim=1)  # input: 384*64*64
        d4 = self.Up_conv4(d4)  # input: 384*64*64,  output: 128*64*64

        # att_4 = self.att_block2(e3, d4)
        up3 = self.Up4(d4)  # input: 128*64*64,  output: 128*128*128
        # att_4 = self.att_block2(att_4, up3)
        d3 = torch.cat((e3, up3), dim=1)  # input: 192*128*128
        d3 = self.Up_conv3(d3)  # input: 192*128*128, output: 64*128*128

        # att_3 = self.att_block3(e2, d3)
        up2 = self.Up3(d3)  # input: 64*128*128,  output: 64*256*256
        d2 = torch.cat((e2, up2), dim=1)  # input: 96*256*256
        d2 = self.Up_conv2(d2)  # input: 96*256*256, output: 32*256*256

        # att_2 = self.att_block4(e1, d2)
        up1 = self.Up2(d2)  # input: 32*256*256,  output: 128*512*512
        d1 = torch.cat((e1, up1), dim=1)  # input: 48*512*512
        d1 = self.Up_conv1(d1)  # input: 48*512*512, output: 64*128*128

        out = self.Conv(d1)

        return out

class DoubleMLPUNet_SimpleAtt_V2(nn.Module):
    """
    UNet - Unet with down-upsampling operation
    Paper : https://arxiv.org/abs/1505.04597
    """

    def __init__(self, in_ch=1, out_ch=3):
        super(DoubleMLPUNet_SimpleAtt_V2, self).__init__()

        n1 = 1

        self.in_chanel = in_ch
        self.out_chanel = out_ch

        filters = [32,64, 128, 256, 512, 1024]

        embed_dims = [32,64, 128, 256, 512,1024]

        drop_rate = 0.2
        img_size = 512

        self.up =  nn.Upsample(scale_factor=2)
        self.down = nn.MaxPool2d(kernel_size=2, stride=0.5)

        self.hybrid_model = ResNetV2(block_units=(3, 4, 7), width_factor=0.5)

        self.mlp_block1 = LightMLPBlockv2(in_dim=embed_dims[3], out_dim=embed_dims[3], mlp_ratio=1, drop=drop_rate,
                                   img_size=img_size // 16, patch_size=3, stride=2)
        self.mlp_block2 = LightMLPBlockv2(in_dim=embed_dims[3], out_dim=embed_dims[3], mlp_ratio=1, drop=drop_rate,
                                   img_size=img_size // 16, patch_size=3, stride=2)

        self.att_block1 = Attention_block(in_dims=filters[3],out_dims=filters[3])
        self.att_block2 = Attention_block(in_dims=filters[3], out_dims=filters[2])
        self.att_block3 = Attention_block(in_dims=filters[2], out_dims=filters[0])
        # self.att_block4 = Attention_block(in_dims=filters[1], out_dims=filters[0])

        self.Up_conv5 = conv_block(filters[3], filters[3])

        self.Up5 = up_conv(filters[3], filters[3], 2)
        self.Up_conv4 = conv_block(filters[3] + filters[3], filters[3])
        # self.Up_conv4 = conv_block(filters[4], filters[3])

        self.Up4 = up_conv(filters[3], filters[3], 2)
        self.Up_conv3 = conv_block(filters[3] + filters[2], filters[2])
        # self.Up_conv3 = conv_block(filters[3], filters[2])

        self.Up3 = up_conv(filters[2], filters[2], 2)
        self.Up_conv2 = conv_block(filters[2] + filters[0], filters[1])


        self.Up2 = up_conv(filters[1], filters[1], 2)
        self.Up_conv1 = conv_block(filters[1], filters[0])

        # self.Up1 = up_conv(filters[1], filters[0], 2)
        self.Conv = nn.Conv2d(filters[0], out_ch, kernel_size=1, stride=1, padding=0)
        self.conv = nn.Conv2d(filters[4], filters[3], kernel_size=1, stride=1,
                              padding=0,)
        self.do_ds = True

        # self.active = torch.nn.Sigmoid()self.MConv0(x)

    def forward(self, x):
        B = x.shape[0]
        if x.size()[1] == 1:
            x = x.repeat(1,3,1,1)

        x, features = self.hybrid_model(x)

        x = self.conv(x)

        # e4 = self.MConv3(pool3)
        e4 = self.mlp_block1(x)  # 4, 128,64,64

        e5 = self.mlp_block2(e4)


        d5 = self.Up_conv5(e5)  # input: 1024*40*40,  output: 1024*40*40

        att_5 = self.att_block1(features[0], d5)
        up4 = self.Up5(d5)                  #input: 512*32*32,  output: 256*64*64
        att_5 = self.att_block1(features[0], d5)
        d4 = torch.cat((att_5, up4), dim=1)  # input: 384*64*64
        d4 = self.Up_conv4(d4)  # input: 384*64*64,  output: 128*64*64

        att_4 = self.att_block2(features[1], d4)
        up3 = self.Up4(d4)  # input: 128*64*64,  output: 128*128*128
        d3 = torch.cat((features[1], up3), dim=1)  # input: 192*128*128
        d3 = self.Up_conv3(d3)  # input: 192*128*128, output: 64*128*128
        # d3 = self.Up_conv3(up3)

        # att_3 = self.att_block3(features[2], d3)
        up2 = self.Up3(d3)  # input: 64*128*128,  output: 64*256*256
        d2 = torch.cat((features[2], up2), dim=1)  # input: 96*256*256
        d2 = self.Up_conv2(d2)  # input: 96*256*256, output: 32*256*256

        # att_2 = self.att_block4(e1, d2)
        up1 = self.Up2(d2)  # input: 32*256*256,  output: 128*512*512
        # d1 = torch.cat((features[3], up1), dim=1)  # input: 48*512*512
        d1 = self.Up_conv1(up1)  # input: 48*512*512, output: 64*128*128

        out = self.Conv(d1)

        return out

class DoubleMLPUNet_AttSEup(nn.Module):
    """
    UNet - Unet with down-upsampling operation
    Paper : https://arxiv.org/abs/1505.04597
    """
    def __init__(self, in_ch=1, out_ch=2):
        super(DoubleMLPUNet_AttSEup, self).__init__()

        n1 = 1

        self.in_chanel = in_ch
        self.out_chanel = out_ch

        filters = [32, 64, 128, 256, 512, 1024]

        embed_dims = [ 64, 128, 256, 512, 1024]

        drop_rate = 0.2
        img_size = 512

        self.up =  nn.Upsample(scale_factor=2)
        self.down = nn.MaxPool2d(kernel_size=2, stride=0.5)

        self.hybrid_model = ResNetV2(block_units=(3, 4, 9), width_factor=0.5)

        self.mlp_block1 = LightMLPBlockv2(in_dim=embed_dims[3], out_dim=embed_dims[2], mlp_ratio=1, drop=drop_rate,
                                   img_size=img_size // 16, patch_size=3, stride=1)
        self.mlp_block2 = LightMLPBlockv2(in_dim=embed_dims[2], out_dim=embed_dims[3], mlp_ratio=1, drop=drop_rate,
                                   img_size=img_size // 16, patch_size=3, stride=1)

        self.att_block1 = Attention_block_simple(in_dims=filters[4],out_dims=filters[3])
        self.att_block2 = Attention_block_simple(in_dims=filters[3], out_dims=filters[2])
        self.att_block3 = Attention_block_simple(in_dims=filters[2], out_dims=filters[0])
        # self.att_block4 = Attention_block_up(in_dims=filters[0], out_dims=filters[0])


        self.Up_conv5 = conv_block(filters[4], filters[4])

        self.Up5 = up_conv(filters[4], filters[4], 2)
        self.Up_conv4 = conv_block(filters[4]+filters[3], filters[3])

        self.Up4 = up_conv(filters[3], filters[3], 2)
        self.Up_conv3 = conv_block(filters[3]+filters[2], filters[2])

        self.Up3 = up_conv(filters[2], filters[2], 2)
        self.Up_conv2 = conv_block(filters[2]+filters[0] , filters[1])

        self.Up2 = up_conv(filters[1], filters[1], 2)
        self.Up_conv1 = conv_block(filters[1], filters[0])

        self.Conv = nn.Conv2d(filters[0], out_ch, kernel_size=1, stride=1, padding=0)
        self.conv = nn.Conv2d(filters[4], filters[3], kernel_size=1, stride=1,
                              padding=0,)
        self.do_ds = True

        # self.active = torch.nn.Sigmoid()self.MConv0(x)

    def forward(self, x):
        if x.size()[1] == 1:
            x = x.repeat(1,3,1,1)

        x, features = self.hybrid_model(x)

        # x = self.conv(x)

        e4 = self.mlp_block1(x)  # 4, 128,64,64
        e5 = self.mlp_block2(e4)

        d5 = self.Up_conv5(e5)  # input: 1024*40*40,  output: 1024*40*40

        att_5 = self.att_block1(features[0], d5)
        up4 = self.Up5(d5)
        d4 = torch.cat((att_5, up4), dim=1)  # input: 384*64*64
        # d4 = torch.cat((features[0], up4), dim=1)
        d4 = self.Up_conv4(d4)  # input: 384*64*64,  output: 128*64*64

        att_4 = self.att_block2(features[1], d4)
        up3 = self.Up4(d4)  # input: 128*64*64,  output: 128*128*128
        d3 = torch.cat((att_4, up3), dim=1)  # input: 192*128*128
        # d3 = torch.cat((features[1], up3), dim=1)
        d3 = self.Up_conv3(d3)  # input: 192*128*128, output: 64*128*128

        att_3 = self.att_block3(features[2], d3)
        up2 = self.Up3(d3)  # input: 64*128*128,  output: 64*256*256
        d2 = torch.cat((att_3, up2), dim=1)  # input: 96*256*256
        # d2 = torch.cat((features[2], up2), dim=1)
        d2 = self.Up_conv2(d2)  # input: 96*256*256, output: 32*256*256

        up1 = self.Up2(d2)  # input: 32*256*256,  output: 128*512*512
        # d1 = torch.cat((att_2, up1), dim=1)  # input: 48*512*512
        d1 = self.Up_conv1(up1)  # input: 48*512*512, output: 64*128*128

        out = self.Conv(d1)

        return out

class DoubleshiftMLPUNet(nn.Module):
    """
    UNet - Unet with down-upsampling operation
    Paper : https://arxiv.org/abs/1505.04597
    """
    def __init__(self, in_ch=1, out_ch=2):
        super(DoubleshiftMLPUNet, self).__init__()

        n1 = 1

        self.in_chanel = in_ch
        self.out_chanel = out_ch

        filters = [ 64, 128, 256, 512, 1024]

        embed_dims = [64, 128, 256, 512, 1024]

        drop_rate = 0.2
        img_size = 512

        self.up =  nn.Upsample(scale_factor=2)
        self.down = nn.MaxPool2d(kernel_size=2, stride=1)

        self.hybrid_model = ResNetV2(block_units=(3, 4, 9), width_factor=1)

        self.mlp_block1 = shiftMLPBlock(in_dim=embed_dims[4], out_dim=embed_dims[2], mlp_ratio=1, drop=drop_rate,
                                   img_size=img_size // 16, patch_size=3, stride=1)
        self.mlp_block2 = shiftMLPBlock(in_dim=embed_dims[2], out_dim=embed_dims[4], mlp_ratio=1, drop=drop_rate,
                                   img_size=img_size // 16, patch_size=3, stride=1)

        # self.att_block1 = Attention_SE2(filters[4],filters[3])
        # self.att_block2 = Attention_SE2(filters[3], filters[2])
        # self.att_block3 = Attention_SE2(filters[2], filters[0])
        # self.att_block4 = Attention_block_up(in_dims=filters[0], out_dims=filters[0])


        self.Up_conv5 = conv_block(filters[4], filters[4])

        self.Up5 = up_conv(filters[4], filters[4], 2)
        self.Up_conv4 = conv_block(filters[4]+filters[3], filters[3])

        self.Up4 = up_conv(filters[3], filters[3], 2)
        self.Up_conv3 = conv_block(filters[3]+filters[2], filters[2])

        self.Up3 = up_conv(filters[2], filters[2], 2)
        self.Up_conv2 = conv_block(filters[2]+filters[0] , filters[1])

        self.Up2 = up_conv(filters[1], filters[1], 2)
        self.Up_conv1 = conv_block(filters[1], filters[0])

        self.Conv = nn.Conv2d(filters[0], out_ch, kernel_size=1, stride=1, padding=0)
        self.conv = nn.Conv2d(filters[4], filters[3], kernel_size=1, stride=1,
                              padding=0,)
        self.do_ds = True

        # self.active = torch.nn.Sigmoid()self.MConv0(x)

    def forward(self, x):
        if x.size()[1] == 1:
            x = x.repeat(1,3,1,1)

        x, features = self.hybrid_model(x)

        e4 = self.mlp_block1(x)  # 4, 128,64,64
        e5 = self.mlp_block2(e4)

        d5 = self.Up_conv5(e5)  # input: 1024*40*40,  output: 1024*40*40

        # att_5 = self.att_block1(features[0], d5)
        up4 = self.Up5(d5)
        # d4 = self.att_block1(up4, features[0])
        # d4 = torch.cat((d4, up4), dim=1)  # input: 384*64*64
        d4 = torch.cat((features[0], up4), dim=1)
        d4 = self.Up_conv4(d4)  # input: 384*64*64,  output: 128*64*64

        # att_4 = self.att_block2(features[1], d4)
        up3 = self.Up4(d4)  # input: 128*64*64,  output: 128*128*128
        # d3 = self.att_block2(up3, features[1])
        # d3 = torch.cat((d3, up3), dim=1)  # input: 192*128*128
        d3 = torch.cat((features[1], up3), dim=1)
        d3 = self.Up_conv3(d3)  # input: 192*128*128, output: 64*128*128

        # att_3 = self.att_block3(features[2], d3)
        up2 = self.Up3(d3)  # input: 64*128*128,  output: 64*256*256
        # d2 = self.att_block3(up2, features[2])
        # d2 = torch.cat((d2, up2), dim=1)  # input: 96*256*256
        d2 = torch.cat((features[2], up2), dim=1)
        d2 = self.Up_conv2(d2)  # input: 96*256*256, output: 32*256*256

        up1 = self.Up2(d2)  # input: 32*256*256,  output: 128*512*512
        # d1 = torch.cat((att_2, up1), dim=1)  # input: 48*512*512
        d1 = self.Up_conv1(up1)  # input: 48*512*512, output: 64*128*128

        out = self.Conv(d1)

        return out

class DoubleMLPUNetv2(nn.Module):
    """
    UNet - Unet with down-upsampling operation
    Paper : https://arxiv.org/abs/1505.04597
    """
    def __init__(self, in_ch=1, out_ch=2):
        super(DoubleMLPUNetv2, self).__init__()

        n1 = 1

        self.in_chanel = in_ch
        self.out_chanel = out_ch

        filters = [32, 64, 128, 256, 512, 1024]

        embed_dims = [64, 128, 256, 512, 1024]

        drop_rate = 0.2
        img_size = 512

        self.up =  nn.Upsample(scale_factor=2)
        self.down = nn.MaxPool2d(kernel_size=2, stride=1)

        self.hybrid_model = ResNetV2(block_units=(3, 4, 9), width_factor=0.5)

        self.mlp_block1 = LightMLPBlockv2(in_dim=embed_dims[3], out_dim=embed_dims[2], mlp_ratio=1, drop=drop_rate,
                                   img_size=img_size // 16, patch_size=3, stride=1)
        self.mlp_block2 = LightMLPBlockv2(in_dim=embed_dims[2], out_dim=embed_dims[3], mlp_ratio=1, drop=drop_rate,
                                          img_size=img_size // 16, patch_size=3, stride=1)


        self.Up_conv5 = conv_block(filters[4], filters[4])

        self.Up5 = up_conv(filters[4], filters[4], 2)
        self.Up_conv4 = conv_block(filters[4]+filters[3], filters[3])

        self.Up4 = up_conv(filters[3], filters[3], 2)
        self.Up_conv3 = conv_block(filters[3]+filters[2], filters[2])

        self.Up3 = up_conv(filters[2], filters[2], 2)
        self.Up_conv2 = conv_block(filters[2]+filters[0] , filters[1])

        self.Up2 = up_conv(filters[1], filters[1], 2)
        self.Up_conv1 = conv_block(filters[1], filters[0])

        self.Conv = nn.Conv2d(filters[0], out_ch, kernel_size=1, stride=1, padding=0)
        self.conv = nn.Conv2d(filters[4], filters[3], kernel_size=1, stride=1,
                              padding=0,)
        self.do_ds = True

        # self.active = torch.nn.Sigmoid()self.MConv0(x)

    def forward(self, x):
        if x.size()[1] == 1:
            x = x.repeat(1,3,1,1)

        x, features = self.hybrid_model(x)

        e4 = self.mlp_block1(x)  # 4, 128,64,64
        e5 = self.mlp_block2(e4)

        d5 = self.Up_conv5(e5)  # input: 1024*40*40,  output: 1024*40*40

        # att_5 = self.att_block1(features[0], d5)
        up4 = self.Up5(d5)
        # d4 = self.att_block1(up4, features[0])
        # d4 = torch.cat((d4, up4), dim=1)  # input: 384*64*64
        d4 = torch.cat((features[0], up4), dim=1)
        d4 = self.Up_conv4(d4)  # input: 384*64*64,  output: 128*64*64

        # att_4 = self.att_block2(features[1], d4)
        up3 = self.Up4(d4)  # input: 128*64*64,  output: 128*128*128
        # d3 = self.att_block2(up3, features[1])
        # d3 = torch.cat((d3, up3), dim=1)  # input: 192*128*128
        d3 = torch.cat((features[1], up3), dim=1)
        d3 = self.Up_conv3(d3)  # input: 192*128*128, output: 64*128*128

        # att_3 = self.att_block3(features[2], d3)
        up2 = self.Up3(d3)  # input: 64*128*128,  output: 64*256*256
        # d2 = self.att_block3(up2, features[2])
        # d2 = torch.cat((d2, up2), dim=1)  # input: 96*256*256
        d2 = torch.cat((features[2], up2), dim=1)
        d2 = self.Up_conv2(d2)  # input: 96*256*256, output: 32*256*256

        up1 = self.Up2(d2)  # input: 32*256*256,  output: 128*512*512
        # d1 = torch.cat((att_2, up1), dim=1)  # input: 48*512*512
        d1 = self.Up_conv1(up1)  # input: 48*512*512, output: 64*128*128

        out = self.Conv(d1)

        return out

class DoubleMLPUNetv2large(nn.Module):
    """
    UNet - Unet with down-upsampling operation
    Paper : https://arxiv.org/abs/1505.04597
    """
    def __init__(self, in_ch=1, out_ch=2):
        super(DoubleMLPUNetv2large, self).__init__()

        n1 = 1

        self.in_chanel = in_ch
        self.out_chanel = out_ch

        filters = [64, 128, 256, 512, 1024]

        embed_dims = [128, 256, 512, 1024]

        drop_rate = 0.2
        img_size = 512

        self.up =  nn.Upsample(scale_factor=2)
        self.down = nn.MaxPool2d(kernel_size=2, stride=1)

        self.hybrid_model = ResNetV2(block_units=(3, 4, 9), width_factor=1)

        self.mlp_block1 = LightMLPBlockv2(in_dim=embed_dims[3], out_dim=embed_dims[2], mlp_ratio=1, drop=drop_rate,
                                   img_size=img_size // 16, patch_size=3, stride=1)
        self.mlp_block2 = LightMLPBlockv2(in_dim=embed_dims[2], out_dim=embed_dims[3], mlp_ratio=1, drop=drop_rate,
                                          img_size=img_size // 16, patch_size=3, stride=1)



        self.Up_conv5 = conv_block(filters[4], filters[4])

        self.Up5 = up_conv(filters[4], filters[4], 2)
        self.Up_conv4 = conv_block(filters[4]+filters[3], filters[3])

        self.Up4 = up_conv(filters[3], filters[3], 2)
        self.Up_conv3 = conv_block(filters[3]+filters[2], filters[2])

        self.Up3 = up_conv(filters[2], filters[2], 2)
        self.Up_conv2 = conv_block(filters[2]+filters[0] , filters[1])

        self.Up2 = up_conv(filters[1], filters[1], 2)
        self.Up_conv1 = conv_block(filters[1], filters[0])

        self.Conv = nn.Conv2d(filters[0], out_ch, kernel_size=1, stride=1, padding=0)
        self.conv = nn.Conv2d(filters[4], filters[3], kernel_size=1, stride=1,
                              padding=0,)
        self.do_ds = True

        # self.active = torch.nn.Sigmoid()self.MConv0(x)

    def forward(self, x):
        if x.size()[1] == 1:
            x = x.repeat(1,3,1,1)

        x, features = self.hybrid_model(x)

        e4 = self.mlp_block1(x)  # 4, 128,64,64
        e5 = self.mlp_block2(e4)

        d5 = self.Up_conv5(e5)  # input: 1024*40*40,  output: 1024*40*40


        up4 = self.Up5(d5)
        d4 = torch.cat((features[0], up4), dim=1)
        d4 = self.Up_conv4(d4)  # input: 384*64*64,  output: 128*64*64


        up3 = self.Up4(d4)  # input: 128*64*64,  output: 128*128*128
        d3 = torch.cat((features[1], up3), dim=1)
        d3 = self.Up_conv3(d3)  # input: 192*128*128, output: 64*128*128

        up2 = self.Up3(d3)  # input: 64*128*128,  output: 64*256*256
        d2 = torch.cat((features[2], up2), dim=1)
        d2 = self.Up_conv2(d2)  # input: 96*256*256, output: 32*256*256

        up1 = self.Up2(d2)  # input: 32*256*256,  output: 128*512*512
        d1 = self.Up_conv1(up1)  # input: 48*512*512, output: 64*128*128

        out = self.Conv(d1)

        return out


class Double3MLPUNet(nn.Module):
    """
    UNet - Unet with down-upsampling operation
    Paper : https://arxiv.org/abs/1505.04597
    """
    def __init__(self, in_ch=1, out_ch=2):
        super(Double3MLPUNet, self).__init__()

        n1 = 1

        self.in_chanel = in_ch
        self.out_chanel = out_ch

        filters = [32, 64, 128, 256, 512, 1024]

        embed_dims = [64, 128, 256, 512, 1024]

        drop_rate = 0.2
        img_size = 512

        self.up =  nn.Upsample(scale_factor=2)
        self.down = nn.MaxPool2d(kernel_size=2, stride=1)

        self.hybrid_model = ResNetV2(block_units=(3, 4, 9), width_factor=1)

        self.mlp_block1 = LightMLPBlockv2(in_dim=embed_dims[3], out_dim=embed_dims[2], mlp_ratio=1, drop=drop_rate,
                                   img_size=img_size // 16, patch_size=3, stride=1)
        self.mlp_block2 = LightMLPBlockv2(in_dim=embed_dims[2], out_dim=embed_dims[2], mlp_ratio=1, drop=drop_rate,
                                   img_size=img_size // 16, patch_size=3, stride=1)
        self.mlp_block3 = LightMLPBlockv2(in_dim=embed_dims[2], out_dim=embed_dims[3], mlp_ratio=1, drop=drop_rate,
                                          img_size=img_size // 16, patch_size=3, stride=1)


        self.Up_conv5 = conv_block(filters[4], filters[4])

        self.Up5 = up_conv(filters[4], filters[4], 2)
        self.Up_conv4 = conv_block(filters[4]+filters[3], filters[3])

        self.Up4 = up_conv(filters[3], filters[3], 2)
        self.Up_conv3 = conv_block(filters[3]+filters[2], filters[2])

        self.Up3 = up_conv(filters[2], filters[2], 2)
        self.Up_conv2 = conv_block(filters[2]+filters[0] , filters[1])

        self.Up2 = up_conv(filters[1], filters[1], 2)
        self.Up_conv1 = conv_block(filters[1], filters[0])

        self.Conv = nn.Conv2d(filters[0], out_ch, kernel_size=1, stride=1, padding=0)
        self.conv = nn.Conv2d(filters[4], filters[3], kernel_size=1, stride=1,
                              padding=0,)
        self.do_ds = True

        # self.active = torch.nn.Sigmoid()self.MConv0(x)

    def forward(self, x):
        if x.size()[1] == 1:
            x = x.repeat(1,3,1,1)

        x, features = self.hybrid_model(x)

        e4 = self.mlp_block1(x)  # 4, 128,64,64
        e5 = self.mlp_block2(e4)
        e5 = self.mlp_block3(e5)

        d5 = self.Up_conv5(e5)  # input: 1024*40*40,  output: 1024*40*40

        # att_5 = self.att_block1(features[0], d5)
        up4 = self.Up5(d5)
        # d4 = self.att_block1(up4, features[0])
        # d4 = torch.cat((d4, up4), dim=1)  # input: 384*64*64
        d4 = torch.cat((features[0], up4), dim=1)
        d4 = self.Up_conv4(d4)  # input: 384*64*64,  output: 128*64*64

        # att_4 = self.att_block2(features[1], d4)
        up3 = self.Up4(d4)  # input: 128*64*64,  output: 128*128*128
        # d3 = self.att_block2(up3, features[1])
        # d3 = torch.cat((d3, up3), dim=1)  # input: 192*128*128
        d3 = torch.cat((features[1], up3), dim=1)
        d3 = self.Up_conv3(d3)  # input: 192*128*128, output: 64*128*128

        # att_3 = self.att_block3(features[2], d3)
        up2 = self.Up3(d3)  # input: 64*128*128,  output: 64*256*256
        # d2 = self.att_block3(up2, features[2])
        # d2 = torch.cat((d2, up2), dim=1)  # input: 96*256*256
        d2 = torch.cat((features[2], up2), dim=1)
        d2 = self.Up_conv2(d2)  # input: 96*256*256, output: 32*256*256

        up1 = self.Up2(d2)  # input: 32*256*256,  output: 128*512*512
        # d1 = torch.cat((att_2, up1), dim=1)  # input: 48*512*512
        d1 = self.Up_conv1(up1)  # input: 48*512*512, output: 64*128*128

        out = self.Conv(d1)

        return out

class RegDoubleMLPUNet(nn.Module):
    """
    UNet - Unet with down-upsampling operation
    Paper : https://arxiv.org/abs/1505.04597
    """
    def __init__(self, in_ch=1, out_ch=2):
        super(RegDoubleMLPUNet, self).__init__()

        self.in_chanel = in_ch
        self.out_chanel = out_ch

        filters = [32, 64, 128, 256, 512, 1024]

        embed_dims = [64, 128, 256, 512, 1024]

        drop_rate = 0.2
        img_size = 512

        self.hybrid_model = ResNetV2(block_units=(3, 4, 9), width_factor=0.5)

        self.mlp_block1 = LightMLPBlockv4(in_dim=embed_dims[3], out_dim=embed_dims[2], mlp_ratio=1, drop=drop_rate,
                                          img_size=img_size // 16, patch_size=3, stride=1)

        self.mlp_block2 = LightMLPBlockv4(in_dim=embed_dims[2], out_dim=embed_dims[3], mlp_ratio=1, drop=drop_rate,
                                          img_size=img_size // 16, patch_size=3, stride=1)


        self.conv5 = conv_block(filters[4], filters[4])
        self.Up5 = up_conv(filters[4], filters[3], 2)

        self.Up4_first = up_conv(filters[4], filters[2], 4)
        self.Up4_second = up_conv(filters[3], filters[2], 2)
        self.Up4_layer = up_conv(filters[4], filters[2], 2)

        self.Up3_first = up_conv(filters[4], filters[0], 8)
        self.Up3_second = up_conv(filters[3], filters[0], 4)
        self.Up3_third = up_conv(filters[2], filters[0], 2)
        self.Up3_layer = up_conv(filters[3]+filters[2], filters[2], 2)

        self.Up2 = up_conv(filters[2]+filters[1], filters[1], 2)
        self.Up_conv1 = conv_block(filters[1], filters[0])

        self.Conv = nn.Conv2d(filters[0], out_ch, kernel_size=1, stride=1, padding=0)
        self.conv = nn.Conv2d(filters[4], filters[3], kernel_size=1, stride=1,
                              padding=0,)
        self.do_ds = True

        # self.active = torch.nn.Sigmoid()self.MConv0(x)

    def forward(self, x):
        if x.size()[1] == 1:
            x = x.repeat(1,3,1,1)

        x, features = self.hybrid_model(x)

        e4 = self.mlp_block1(x)  # 512, 32, 32
        e5 = self.mlp_block2(e4) # 512, 32, 32


        d5 = self.conv5(e5)  # 512, 32, 32

        layer4_up = self.Up5(d5)                  #input: 512, 32, 32,  output: 256*64*64
        f4 = torch.sigmoid(layer4_up)*features[0]          # 256*64*64
        layer4 = torch.cat((features[0],f4), dim=1)        #output: 512*64*64

        #layer 3
        layer3_up_first = self.Up4_first(e5)                 #input: 512, 32, 32,  output: 128,128,128
        f3_first = torch.sigmoid(layer3_up_first)*features[1]          #input: 128,128,128
        layer3_up_second = self.Up4_second(f4)                # input: 256*64*64,  output: 128*128*128
        f3_second = torch.sigmoid(layer3_up_second) * f3_first    # input: 128*128*128
        f3_layer = self.Up4_layer(layer4)                     # input: 512*64*64, 128*128*128
        layer3 = torch.cat((features[1], f3_second,f3_layer,), dim=1)     # output: 512*128*128

        # layer 2
        layer2_up_first = self.Up3_first(e5)                         # input: 512, 32, 32 output: 32, 256, 256
        f2_first = torch.sigmoid(layer2_up_first) * features[2]               # input: 32, 256, 256
        layer2_up_second = self.Up3_second(f4)                       # input: 128*128*128,  output:  32, 256, 256
        f2_second = torch.sigmoid(layer2_up_second) * f2_first       # input:  32, 256, 256
        layer2_up_third = self.Up3_third(f3_second)                  # input: 128*128*128,  output: 32, 256, 256
        f2_third = torch.sigmoid(layer2_up_third) * f2_second        # input: 32, 256, 256
        f2_layer = self.Up3_layer(layer3)                            # input: 128,256,256
        layer2 = torch.cat((features[2], f2_third, f2_layer,), dim=1)         # output: 128*256*256

        # The first layer
        up1 = self.Up2(layer2)  # input: 32*256*256,  output: 128*512*512
        layer1 = self.Up_conv1(up1)  # input: 48*512*512, output: 64*128*128

        out = self.Conv(layer1)

        return out

class RegDoubleMLPUNetlite(nn.Module):
    """
    UNet - Unet with down-upsampling operation
    Paper : https://arxiv.org/abs/1505.04597
    """
    def __init__(self, in_ch=1, out_ch=2):
        super(RegDoubleMLPUNetlite, self).__init__()

        self.in_chanel = in_ch
        self.out_chanel = out_ch

        filters = [32, 64, 128, 256, 512, 1024]

        embed_dims = [64, 128, 256, 512, 1024]

        drop_rate = 0.2
        img_size = 512

        self.hybrid_model = ResNetV2(block_units=(3, 4, 9), width_factor=0.5)

        self.mlp_block1 = LightMLPBlockv2(in_dim=embed_dims[3], out_dim=embed_dims[2], mlp_ratio=1, drop=drop_rate,
                                   img_size=img_size // 16, patch_size=3, stride=1)
        self.mlp_block2 = LightMLPBlockv2(in_dim=embed_dims[2], out_dim=embed_dims[3], mlp_ratio=1, drop=drop_rate,
                                          img_size=img_size // 16, patch_size=3, stride=1)

        self.conv5 = conv_block(filters[4], filters[4])
        self.Up5 = up_conv(filters[4], filters[3], 2)

        self.Up4_first = up_conv(filters[4], filters[2], 4)
        self.Up4_layer = up_conv(filters[4], filters[3], 2)

        self.Up3_first = up_conv(filters[4], filters[0], 8)
        self.Up3_layer = up_conv(filters[4], filters[2], 2)

        self.Up2 = up_conv(filters[2]+filters[1], filters[1], 2)
        self.Up_conv1 = conv_block(filters[1], filters[0])

        self.Conv = nn.Conv2d(filters[0], out_ch, kernel_size=1, stride=1, padding=0)
        self.conv = nn.Conv2d(filters[4], filters[3], kernel_size=1, stride=1,
                              padding=0,)
        self.do_ds = True

        # self.active = torch.nn.Sigmoid()self.MConv0(x)

    def forward(self, x):
        if x.size()[1] == 1:
            x = x.repeat(1,3,1,1)

        x, features = self.hybrid_model(x)

        e4 = self.mlp_block1(x)  # 512, 32, 32
        e5 = self.mlp_block2(e4) # 512, 32, 32

        d5 = self.conv5(e5)  # 512, 32, 32

        layer4_up = self.Up5(d5)                  #input: 512, 32, 32,  output: 256*64*64
        f4 = torch.sigmoid(layer4_up)*features[0]          # 256*64*64
        layer4 = torch.cat((features[0],f4), dim=1)        #output: 512*64*64

        #layer 3
        layer3_up_first = self.Up4_first(e5)                 #input: 512, 32, 32,  output: 128,128,128
        f3 = torch.sigmoid(layer3_up_first)*features[1]          #input: 128,128,128
        f3_layer = self.Up4_layer(layer4)                     # input: 512*64*64, 128*128*128
        layer3 = torch.cat((features[1], f3,f3_layer,), dim=1)     # output: (256+128+128)*128*128

        # layer 2
        layer2_up_first = self.Up3_first(e5)                         # input: 512, 32, 32 output: 32, 256, 256
        f2 = torch.sigmoid(layer2_up_first) * features[2]               # input: 32, 256, 256
        f2_layer = self.Up3_layer(layer3)                            # input: 128,256,256
        layer2 = torch.cat((features[2], f2, f2_layer,), dim=1)         # output: (128+32+32)*256*256

        # The first layer
        up1 = self.Up2(layer2)  # input: 32*256*256,  output: 128*512*512
        layer1 = self.Up_conv1(up1)  # input: 48*512*512, output: 64*128*128

        out = self.Conv(layer1)

        return out

class RegDoubleMLPUNetv2(nn.Module):
    def __init__(self, in_ch=1, out_ch=2):
        super(RegDoubleMLPUNetv2, self).__init__()

        self.in_chanel = in_ch
        self.out_chanel = out_ch

        filters = [32, 64, 128, 256, 512, 1024]

        embed_dims = [64, 128, 256, 512, 1024]

        drop_rate = 0.2
        img_size = 512

        self.hybrid_model = ResNetV2(block_units=(3, 4, 9), width_factor=0.5)

        self.mlp_block1 = LightMLPBlockv4(in_dim=embed_dims[3], out_dim=embed_dims[2], mlp_ratio=1, drop=drop_rate,
                                   img_size=img_size // 16, patch_size=3, stride=1)
        self.mlp_block2 = LightMLPBlockv4(in_dim=embed_dims[2], out_dim=embed_dims[3], mlp_ratio=1, drop=drop_rate,
                                          img_size=img_size // 16, patch_size=3, stride=1)

        self.conv5 = conv_block(filters[4], filters[4])
        self.Up5 = up_conv(filters[4], filters[3], 2)

        self.Up4_second = up_conv(filters[3], filters[2], 2)
        self.Up4_layer = up_conv(filters[4], filters[2], 2)

        self.Up3_third = up_conv(filters[2], filters[0], 2)
        self.Up3_layer = up_conv(filters[3]+filters[2], filters[2], 2)

        self.Up2 = up_conv(filters[2]+filters[1], filters[1], 2)
        self.Up_conv1 = conv_block(filters[1], filters[0])

        self.Conv = nn.Conv2d(filters[0], out_ch, kernel_size=1, stride=1, padding=0)
        self.conv = nn.Conv2d(filters[4], filters[3], kernel_size=1, stride=1,
                              padding=0,)
        self.do_ds = True

        # self.active = torch.nn.Sigmoid()self.MConv0(x)

    def forward(self, x):
        if x.size()[1] == 1:
            x = x.repeat(1,3,1,1)

        x, features = self.hybrid_model(x)

        e4 = self.mlp_block1(x)  # 512, 32, 32
        e5 = self.mlp_block2(e4) # 512, 32, 32

        d5 = self.conv5(e5)  # 512, 32, 32

        layer4_up = self.Up5(d5)                  #input: 512, 32, 32,  output: 256*64*64
        f4 = torch.sigmoid(layer4_up)*features[0]          # 256*64*64
        layer4 = torch.cat((features[0],f4), dim=1)        #output: 512*64*64

        #layer 3
        layer3_up_second = self.Up4_second(f4)                # input: 256*64*64,  output: 128*128*128
        f3_second = torch.sigmoid(layer3_up_second) * features[1]    # input: 128*128*128
        f3_layer = self.Up4_layer(layer4)                     # input: 512*64*64, 128*128*128
        layer3 = torch.cat((features[1], f3_second,f3_layer), dim=1)     # output: 512*128*128

        # layer 2
        layer2_up_third = self.Up3_third(f3_second)                  # input: 128*128*128,  output: 32, 256, 256
        f2_third = torch.sigmoid(layer2_up_third) * features[2]        # input: 32, 256, 256
        f2_layer = self.Up3_layer(layer3)                            # input: 128,256,256
        layer2 = torch.cat((features[2], f2_third, f2_layer), dim=1)         # output: 128*256*256

        # The first layer
        up1 = self.Up2(layer2)  # input: 32*256*256,  output: 128*512*512
        layer1 = self.Up_conv1(up1)  # input: 48*512*512, output: 64*128*128

        out = self.Conv(layer1)

        return out

class RegDouble1MLPUNetv2(nn.Module):
    """
    UNet - Unet with down-upsampling operation
    Paper : https://arxiv.org/abs/1505.04597
    """
    def __init__(self, in_ch=1, out_ch=2):
        super(RegDouble1MLPUNetv2, self).__init__()

        self.in_chanel = in_ch
        self.out_chanel = out_ch

        filters = [32, 64, 128, 256, 512, 1024]

        embed_dims = [64, 128, 256, 512, 1024]

        drop_rate = 0.2
        img_size = 512

        self.hybrid_model = ResNetV2(block_units=(3, 4, 9), width_factor=0.5)

        self.mlp_block1 = LightMLPBlockv4(in_dim=embed_dims[3], out_dim=embed_dims[3], mlp_ratio=1, drop=drop_rate,
                                   img_size=img_size // 16, patch_size=3, stride=1)


        self.conv5 = conv_block(filters[4], filters[4])
        self.Up5 = up_conv(filters[4], filters[3], 2)

        self.Up4_first = up_conv(filters[4], filters[2], 4)
        self.Up4_second = up_conv(filters[3], filters[2], 2)
        self.Up4_layer = up_conv(filters[4], filters[2], 2)

        self.Up3_first = up_conv(filters[4], filters[0], 8)
        self.Up3_second = up_conv(filters[3], filters[0], 4)
        self.Up3_third = up_conv(filters[2], filters[0], 2)
        self.Up3_layer = up_conv(filters[3]+filters[2], filters[2], 2)

        self.Up2 = up_conv(filters[2]+filters[1], filters[1], 2)
        self.Up_conv1 = conv_block(filters[1], filters[0])

        self.Conv = nn.Conv2d(filters[0], out_ch, kernel_size=1, stride=1, padding=0)
        self.conv = nn.Conv2d(filters[4], filters[3], kernel_size=1, stride=1,
                              padding=0,)
        self.do_ds = True

        # self.active = torch.nn.Sigmoid()self.MConv0(x)

    def forward(self, x):
        if x.size()[1] == 1:
            x = x.repeat(1,3,1,1)

        x, features = self.hybrid_model(x)

        e4 = self.mlp_block1(x)  # 512, 32, 32

        d5 = self.conv5(e4)  # 512, 32, 32

        layer4_up = self.Up5(d5)                  #input: 512, 32, 32,  output: 256*64*64
        f4 = torch.sigmoid(layer4_up)*features[0]          # 256*64*64
        layer4 = torch.cat((features[0],f4), dim=1)        #output: 512*64*64

        #layer 3
        layer3_up_first = self.Up4_first(d5)                 #input: 512, 32, 32,  output: 128,128,128
        f3_first = torch.sigmoid(layer3_up_first)*features[1]          #input: 128,128,128
        layer3_up_second = self.Up4_second(f4)                # input: 256*64*64,  output: 128*128*128
        f3_second = torch.sigmoid(layer3_up_second) * f3_first    # input: 128*128*128
        f3_layer = self.Up4_layer(layer4)                     # input: 512*64*64, 128*128*128
        layer3 = torch.cat((features[1], f3_second,f3_layer,), dim=1)     # output: 512*128*128

        # layer 2
        layer2_up_first = self.Up3_first(d5)                         # input: 512, 32, 32 output: 32, 256, 256
        f2_first = torch.sigmoid(layer2_up_first) * features[2]      # input: 32, 256, 256
        layer2_up_second = self.Up3_second(f4)                       # input: 128*128*128,  output:  32, 256, 256
        f2_second = torch.sigmoid(layer2_up_second) * f2_first       # input:  32, 256, 256
        layer2_up_third = self.Up3_third(f3_second)                  # input: 128*128*128,  output: 32, 256, 256
        f2_third = torch.sigmoid(layer2_up_third) * f2_second        # input: 32, 256, 256
        f2_layer = self.Up3_layer(layer3)                            # input: 128,256,256
        layer2 = torch.cat((features[2], f2_third, f2_layer,), dim=1)         # output: 128*256*256

        # The first layer
        up1 = self.Up2(layer2)  # input: 32*256*256,  output: 128*512*512
        layer1 = self.Up_conv1(up1)  # input: 48*512*512, output: 64*128*128

        out = self.Conv(layer1)

        return out


class DRMLPUNet_lite2(nn.Module):
    """
    UNet - Unet with down-upsampling operation
    Paper : https://arxiv.org/abs/1505.04597
    """
    def __init__(self, in_ch=1, out_ch=2):
        super(DRMLPUNet_lite2, self).__init__()

        self.in_chanel = in_ch
        self.out_chanel = out_ch

        filters = [32, 64, 128, 256, 512, 1024]

        embed_dims = [64, 128, 256, 512, 1024]

        drop_rate = 0.2
        img_size = 512

        self.hybrid_model = ResNetV2(block_units=(3, 4, 9), width_factor=0.5)

        self.mlp_block1 = LightMLPBlockv4(in_dim=embed_dims[3], out_dim=embed_dims[2], mlp_ratio=1, drop=drop_rate,
                                          img_size=img_size // 16, patch_size=3, stride=1)
        self.mlp_block2 = LightMLPBlockv4(in_dim=embed_dims[2], out_dim=embed_dims[3], mlp_ratio=1, drop=drop_rate,
                                          img_size=img_size // 16, patch_size=3, stride=1)

        self.mlp_block3 = LightMLPBlockv4(in_dim=embed_dims[2], out_dim=embed_dims[1], mlp_ratio=1, drop=drop_rate,
                                          img_size=img_size // 8, patch_size=3, stride=1)
        self.mlp_block4 = LightMLPBlockv4(in_dim=embed_dims[1], out_dim=embed_dims[2], mlp_ratio=1, drop=drop_rate,
                                          img_size=img_size // 8, patch_size=3, stride=1)

        self.mlp_block5 = LightMLPBlockv4(in_dim=embed_dims[1], out_dim=embed_dims[0], mlp_ratio=1, drop=drop_rate,
                                          img_size=img_size // 4, patch_size=3, stride=1)
        self.mlp_block6 = LightMLPBlockv4(in_dim=embed_dims[0], out_dim=embed_dims[1], mlp_ratio=1, drop=drop_rate,
                                          img_size=img_size // 4, patch_size=3, stride=1)

        self.mlp_block7 = LightMLPBlockv4(in_dim=filters[0], out_dim=filters[0], mlp_ratio=1, drop=drop_rate,
                                          img_size=img_size // 2, patch_size=3, stride=1)
        self.mlp_block8 = LightMLPBlockv4(in_dim=filters[0], out_dim=filters[0], mlp_ratio=1, drop=drop_rate,
                                          img_size=img_size // 2, patch_size=3, stride=1)

        self.conv5 = conv_block(filters[4], filters[4])
        self.Up5 = up_conv(filters[4], filters[3], 2)

        self.Up4_first = up_conv(filters[4], filters[2], 4)
        self.Up4_second = up_conv(filters[3], filters[2], 2)
        self.Up4_layer = up_conv(filters[4], filters[2], 2)

        self.Up3_first = up_conv(filters[4], filters[0], 8)
        self.Up3_second = up_conv(filters[3], filters[0], 4)
        self.Up3_third = up_conv(filters[2], filters[0], 2)
        self.Up3_layer = up_conv(filters[3]+filters[2], filters[2], 2)

        self.Up2 = up_conv(filters[2]+filters[1], filters[1], 2)
        self.Up_conv1 = conv_block(filters[1], filters[0])

        self.Conv = nn.Conv2d(filters[0], out_ch, kernel_size=1, stride=1, padding=0)
        self.conv = nn.Conv2d(filters[4], filters[3], kernel_size=1, stride=1,
                              padding=0,)
        self.do_ds = True

        # self.active = torch.nn.Sigmoid()self.MConv0(x)

    def forward(self, x):
        if x.size()[1] == 1:
            x = x.repeat(1,3,1,1)

        x, features = self.hybrid_model(x)

        e4 = self.mlp_block1(x)  # 512, 32, 32
        e5 = self.mlp_block2(e4) # 512, 32, 32


        d5 = self.conv5(e5)  # 512, 32, 32

        layer4_up = self.Up5(d5)                  #input: 512, 32, 32,  output: 256*64*64
        f4 = self.mlp_block3(features[0])
        f4 = self.mlp_block4(f4)
        f4 = torch.sigmoid(layer4_up)*f4          # 256*64*64
        layer4 = torch.cat((features[0],f4), dim=1)        #output: 512*64*64

        #layer 3
        layer3_up_first = self.Up4_first(e5)                 #input: 512, 32, 32,  output: 128,128,128
        f3 = self.mlp_block5(features[1])
        f3 = self.mlp_block6(f3)
        f3_first = torch.sigmoid(layer3_up_first)*f3          #input: 128,128,128
        # f3_first = torch.sigmoid(layer3_up_first) * features[1]
        layer3_up_second = self.Up4_second(f4)                # input: 256*64*64,  output: 128*128*128
        f3_second = torch.sigmoid(layer3_up_second) * f3_first    # input: 128*128*128
        f3_layer = self.Up4_layer(layer4)                     # input: 512*64*64, 128*128*128
        layer3 = torch.cat((features[1], f3_second,f3_layer,), dim=1)     # output: 512*128*128

        # layer 2
        layer2_up_first = self.Up3_first(e5)                         # input: 512, 32, 32 output: 32, 256, 256
        # f2 = self.mlp_block7(features[2])
        # f2 = self.mlp_block8(f2)
        # f2_first = torch.sigmoid(layer2_up_first) * f2               # input: 32, 256, 256
        f2_first = torch.sigmoid(layer2_up_first) * features[2]
        layer2_up_second = self.Up3_second(f4)                       # input: 128*128*128,  output:  32, 256, 256
        f2_second = torch.sigmoid(layer2_up_second) * f2_first       # input:  32, 256, 256
        layer2_up_third = self.Up3_third(f3_second)                  # input: 128*128*128,  output: 32, 256, 256
        f2_third = torch.sigmoid(layer2_up_third) * f2_second        # input: 32, 256, 256
        f2_layer = self.Up3_layer(layer3)                            # input: 128,256,256
        layer2 = torch.cat((features[2], f2_third, f2_layer,), dim=1)         # output: 128*256*256

        # The first layer
        up1 = self.Up2(layer2)  # input: 32*256*256,  output: 128*512*512
        layer1 = self.Up_conv1(up1)  # input: 48*512*512, output: 64*128*128

        out = self.Conv(layer1)

        return out

class DRMLPUNet_C16(nn.Module):
    """
    UNet - Unet with down-upsampling operation
    Paper : https://arxiv.org/abs/1505.04597
    """
    def __init__(self, in_ch=1, out_ch=2):
        super(DRMLPUNet_C16, self).__init__()

        self.in_chanel = in_ch
        self.out_chanel = out_ch

        filters = [16,32, 64, 128, 256, 512, 1024]

        embed_dims = [32,64, 128, 256, 512, 1024]

        drop_rate = 0.2
        img_size = 512

        self.hybrid_model = ResNetlite(block_units=(3, 4, 9), width_factor=0.25)

        self.mlp_block1 = LightMLPBlockv5(in_dim=embed_dims[3], out_dim=embed_dims[2], mlp_ratio=1, drop=drop_rate,
                                   img_size=img_size // 16, patch_size=3, stride=1)
        self.mlp_block2 = LightMLPBlockv5(in_dim=embed_dims[2], out_dim=embed_dims[3], mlp_ratio=1, drop=drop_rate,
                                          img_size=img_size // 16, patch_size=3, stride=1)

        self.conv5 = conv_block(filters[4], filters[4])
        self.Up5 = up_conv(filters[4], filters[3], 2)

        self.Up4_first = up_conv(filters[4], filters[2], 4)
        self.Up4_second = up_conv(filters[3], filters[2], 2)
        self.Up4_layer = up_conv(filters[4], filters[2], 2)

        self.Up3_first = up_conv(filters[4], filters[0], 8)
        self.Up3_second = up_conv(filters[3], filters[0], 4)
        self.Up3_third = up_conv(filters[2], filters[0], 2)
        self.Up3_layer = up_conv(filters[3]+filters[2], filters[2], 2)

        self.Up2 = up_conv(filters[2]+filters[1], filters[1], 2)
        self.Up_conv1 = conv_block(filters[1], filters[0])

        self.Conv = nn.Conv2d(filters[0], out_ch, kernel_size=1, stride=1, padding=0)
        self.conv = nn.Conv2d(filters[4], filters[3], kernel_size=1, stride=1,
                              padding=0,)
        self.do_ds = True

        # self.active = torch.nn.Sigmoid()self.MConv0(x)

    def forward(self, x):
        if x.size()[1] == 1:
            x = x.repeat(1,3,1,1)

        x, features = self.hybrid_model(x)


        e4 = self.mlp_block1(x)  # 512, 32, 32
        e5 = self.mlp_block2(e4)

        d5 = self.conv5(e5)  # 512, 32, 32

        layer4_up = self.Up5(d5)                  #input: 512, 32, 32,  output: 256*64*64
        f4 = torch.sigmoid(layer4_up)*features[0]          # 256*64*64
        layer4 = torch.cat((features[0],f4), dim=1)        #output: 512*64*64

        #layer 3
        layer3_up_first = self.Up4_first(d5)                 #input: 512, 32, 32,  output: 128,128,128
        f3_first = torch.sigmoid(layer3_up_first)*features[1]          #input: 128,128,128
        layer3_up_second = self.Up4_second(f4)                # input: 256*64*64,  output: 128*128*128
        f3_second = torch.sigmoid(layer3_up_second) * f3_first    # input: 128*128*128
        f3_layer = self.Up4_layer(layer4)                     # input: 512*64*64, 128*128*128
        layer3 = torch.cat((features[1], f3_second,f3_layer,), dim=1)     # output: 512*128*128

        # layer 2
        layer2_up_first = self.Up3_first(d5)                         # input: 512, 32, 32 output: 32, 256, 256
        f2_first = torch.sigmoid(layer2_up_first) * features[2]      # input: 32, 256, 256
        layer2_up_second = self.Up3_second(f4)                       # input: 128*128*128,  output:  32, 256, 256
        f2_second = torch.sigmoid(layer2_up_second) * f2_first       # input:  32, 256, 256
        layer2_up_third = self.Up3_third(f3_second)                  # input: 128*128*128,  output: 32, 256, 256
        f2_third = torch.sigmoid(layer2_up_third) * f2_second        # input: 32, 256, 256
        f2_layer = self.Up3_layer(layer3)                            # input: 128,256,256
        layer2 = torch.cat((features[2], f2_third, f2_layer,), dim=1)         # output: 128*256*256

        # The first layer
        up1 = self.Up2(layer2)  # input: 32*256*256,  output: 128*512*512
        layer1 = self.Up_conv1(up1)  # input: 48*512*512, output: 64*128*128

        out = self.Conv(layer1)

        return out

class DRMLPUNet_C16_lite(nn.Module):
    """
    UNet - Unet with down-upsampling operation
    Paper : https://arxiv.org/abs/1505.04597
    """
    def __init__(self, in_ch=1, out_ch=2):
        super(DRMLPUNet_C16_lite, self).__init__()

        self.in_chanel = in_ch
        self.out_chanel = out_ch

        filters = [8, 16, 32, 64, 128, 256, 512, 1024]

        embed_dims = [16, 32, 64, 128, 256, 512, 1024]

        drop_rate = 0.2
        img_size = 512

        self.hybrid_model = ResNetlite(block_units=(3, 4, 9), width_factor=0.125)

        self.mlp_block1 = LightMLPBlockv5(in_dim=embed_dims[3], out_dim=embed_dims[2], mlp_ratio=1, drop=drop_rate,
                                   img_size=img_size // 16, patch_size=3, stride=1)
        self.mlp_block2 = LightMLPBlockv5(in_dim=embed_dims[2], out_dim=embed_dims[3], mlp_ratio=1, drop=drop_rate,
                                          img_size=img_size // 16, patch_size=3, stride=1)

        self.conv5 = conv_block(filters[4], filters[4])
        self.Up5 = up_conv(filters[4], filters[3], 2)

        self.Up4_first = up_conv(filters[4], filters[2], 4)
        self.Up4_second = up_conv(filters[3], filters[2], 2)
        self.Up4_layer = up_conv(filters[4], filters[2], 2)

        self.Up3_first = up_conv(filters[4], filters[0], 8)
        self.Up3_second = up_conv(filters[3], filters[0], 4)
        self.Up3_third = up_conv(filters[2], filters[0], 2)
        self.Up3_layer = up_conv(filters[3]+filters[2], filters[2], 2)

        self.Up2 = up_conv(filters[2]+filters[1], filters[1], 2)
        self.Up_conv1 = conv_block(filters[1], filters[0])

        self.Conv = nn.Conv2d(filters[0], out_ch, kernel_size=1, stride=1, padding=0)
        self.conv = nn.Conv2d(filters[4], filters[3], kernel_size=1, stride=1,
                              padding=0,)
        self.do_ds = True

        # self.active = torch.nn.Sigmoid()self.MConv0(x)

    def forward(self, x):
        if x.size()[1] == 1:
            x = x.repeat(1,3,1,1)

        x, features = self.hybrid_model(x)


        e4 = self.mlp_block1(x)  # 512, 32, 32
        e5 = self.mlp_block2(e4)

        d5 = self.conv5(e5)  # 512, 32, 32

        layer4_up = self.Up5(d5)                  #input: 512, 32, 32,  output: 256*64*64
        f4 = torch.sigmoid(layer4_up)*features[0]          # 256*64*64
        layer4 = torch.cat((features[0],f4), dim=1)        #output: 512*64*64

        #layer 3
        layer3_up_first = self.Up4_first(d5)                 #input: 512, 32, 32,  output: 128,128,128
        f3_first = torch.sigmoid(layer3_up_first)*features[1]          #input: 128,128,128
        layer3_up_second = self.Up4_second(f4)                # input: 256*64*64,  output: 128*128*128
        f3_second = torch.sigmoid(layer3_up_second) * f3_first    # input: 128*128*128
        f3_layer = self.Up4_layer(layer4)                     # input: 512*64*64, 128*128*128
        layer3 = torch.cat((features[1], f3_second,f3_layer,), dim=1)     # output: 512*128*128

        # layer 2
        layer2_up_first = self.Up3_first(d5)                         # input: 512, 32, 32 output: 32, 256, 256
        f2_first = torch.sigmoid(layer2_up_first) * features[2]      # input: 32, 256, 256
        layer2_up_second = self.Up3_second(f4)                       # input: 128*128*128,  output:  32, 256, 256
        f2_second = torch.sigmoid(layer2_up_second) * f2_first       # input:  32, 256, 256
        layer2_up_third = self.Up3_third(f3_second)                  # input: 128*128*128,  output: 32, 256, 256
        f2_third = torch.sigmoid(layer2_up_third) * f2_second        # input: 32, 256, 256
        f2_layer = self.Up3_layer(layer3)                            # input: 128,256,256
        layer2 = torch.cat((features[2], f2_third, f2_layer,), dim=1)         # output: 128*256*256

        # The first layer
        up1 = self.Up2(layer2)  # input: 32*256*256,  output: 128*512*512
        layer1 = self.Up_conv1(up1)  # input: 48*512*512, output: 64*128*128

        out = self.Conv(layer1)

        return out

class RegDoubleMLPUNetv2(nn.Module):
    """
    UNet - Unet with down-upsampling operation
    Paper : https://arxiv.org/abs/1505.04597
    """
    def __init__(self, in_ch=1, out_ch=2):
        super(RegDoubleMLPUNetv2, self).__init__()

        self.in_chanel = in_ch
        self.out_chanel = out_ch

        filters = [32, 64, 128, 256, 512, 1024]

        embed_dims = [64, 128, 256, 512, 1024]

        drop_rate = 0.2
        img_size = 512

        self.hybrid_model = ResNetV2(block_units=(3, 4, 9), width_factor=0.5)

        self.mlp_block1 = LightMLPBlockv4(in_dim=embed_dims[3], out_dim=embed_dims[2], mlp_ratio=1, drop=drop_rate,
                                   img_size=img_size // 16, patch_size=3, stride=1)
        self.mlp_block2 = LightMLPBlockv4(in_dim=embed_dims[2], out_dim=embed_dims[3], mlp_ratio=1, drop=drop_rate,
                                          img_size=img_size // 16, patch_size=3, stride=1)


        self.conv5 = conv_block(filters[4], filters[4])
        self.Up5 = up_conv(filters[4], filters[3], 2)

        self.Up4_first = up_conv(filters[4], filters[2], 4)
        self.Up4_second = up_conv(filters[3], filters[2], 2)
        self.Up4_layer = up_conv(filters[4], filters[2], 2)

        self.Up3_first = up_conv(filters[4], filters[0], 8)
        self.Up3_second = up_conv(filters[3], filters[0], 4)
        self.Up3_third = up_conv(filters[2], filters[0], 2)
        self.Up3_layer = up_conv(filters[3]+filters[2], filters[2], 2)

        self.Up2 = up_conv(filters[2]+filters[1], filters[1], 2)
        self.Up_conv1 = conv_block(filters[1], filters[0])

        self.Conv = nn.Conv2d(filters[0], out_ch, kernel_size=1, stride=1, padding=0)
        self.conv = nn.Conv2d(filters[4], filters[3], kernel_size=1, stride=1,
                              padding=0,)
        self.do_ds = True

        # self.active = torch.nn.Sigmoid()self.MConv0(x)

    def forward(self, x):
        if x.size()[1] == 1:
            x = x.repeat(1,3,1,1)

        x, features = self.hybrid_model(x)

        e4 = self.mlp_block1(x)  # 512, 32, 32
        e5 = self.mlp_block2(e4)

        d5 = self.conv5(e5)  # 512, 32, 32

        layer4_up = self.Up5(d5)                  #input: 512, 32, 32,  output: 256*64*64
        f4 = torch.sigmoid(layer4_up)*features[0]          # 256*64*64
        layer4 = torch.cat((features[0],f4), dim=1)        #output: 512*64*64

        #layer 3
        layer3_up_first = self.Up4_first(d5)                 #input: 512, 32, 32,  output: 128,128,128
        f3_first = torch.sigmoid(layer3_up_first)*features[1]          #input: 128,128,128
        layer3_up_second = self.Up4_second(f4)                # input: 256*64*64,  output: 128*128*128
        f3_second = torch.sigmoid(layer3_up_second) * f3_first    # input: 128*128*128
        f3_layer = self.Up4_layer(layer4)                     # input: 512*64*64, 128*128*128
        layer3 = torch.cat((features[1], f3_second,f3_layer,), dim=1)     # output: 512*128*128

        # layer 2
        layer2_up_first = self.Up3_first(d5)                              # input: 512, 32, 32 output: 32, 256, 256
        f2_first = torch.sigmoid(layer2_up_first) * features[2]           # input: 32, 256, 256
        layer2_up_second = self.Up3_second(f4)                            # input: 128*128*128,  output:  32, 256, 256
        f2_second = torch.sigmoid(layer2_up_second) * f2_first            # input:  32, 256, 256
        layer2_up_third = self.Up3_third(f3_second)                       # input: 128*128*128,  output: 32, 256, 256
        f2_third = torch.sigmoid(layer2_up_third) * f2_second             # input: 32, 256, 256
        f2_layer = self.Up3_layer(layer3)                                 # input: 128,256,256
        layer2 = torch.cat((features[2], f2_third, f2_layer,), dim=1)     # output: 128*256*256

        # The first layer
        up1 = self.Up2(layer2)  # input: 32*256*256,  output: 128*512*512
        layer1 = self.Up_conv1(up1)  # input: 48*512*512, output: 64*128*128

        out = self.Conv(layer1)

        return out


# def init(module):
#     if isinstance(module, nn.Conv2d) or isinstance(module, nn.ConvTranspose2d):
#         nn.init.kaiming_normal_(module.weight.data, 0.25)
#         if module.bias is not None:
#             nn.init.constant_(module.bias.data, 0)
#
# def model_unet(model_input, in_channel=3, out_channel=1):
#     model_test = model_input(in_channel, out_channel)
#     return model_test
#
# net = model_unet(UNet_small,1,3)
# net.apply(init)
#
# # # 输出数据维度检查
# # net = net.cuda()
# # data = torch.randn((1, 1, 512, 512)).cuda()#B*C*W*H
# # res = net(data)
# # for item in res:
# #     print(item.size())
#
# # 计算网络参数
# print('net total parameters:', sum(param.numel() for param in net.parameters()))
# print('print net parameters finish!')
net = RegDoubleMLPUNetv2(in_ch=1, out_ch=2)
x = torch.randn(1, 1, 512, 512)
out = net(x)
print(out.shape)
total = sum([param.nelement() for param in net.parameters()])
print("Number of parameter: %.2fM" % (total/1e6))
print(total)

x = torch.randn(2, 256, 512//8, 512//8)
net = LightMLPBlockv4(in_dim=256, out_dim=128, mlp_ratio=1, drop=0.2,
                                          img_size=512 // 8, patch_size=3, stride=1)
out = net(x)
print(out.shape)
total = sum([param.nelement() for param in net.parameters()])
print("Number of parameter: %.2fM" % (total/1e6))
print(total)



