# -*- coding: utf-8 -*-
import math
import torch
import torch.nn as nn
import numpy as np
from thop import profile
from einops import rearrange
from einops.layers.torch import Rearrange, Reduce
from timm.models.layers import trunc_normal_, DropPath

class WMSA(nn.Module):
    """ Self-attention module in Swin Transformer
    """

    def __init__(self, input_dim, output_dim, head_dim, window_size, type):
        super(WMSA, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.head_dim = head_dim
        self.scale = self.head_dim ** -0.5
        self.n_heads = input_dim//head_dim
        self.window_size = window_size
        self.type=type
        self.embedding_layer = nn.Linear(self.input_dim, 3*self.input_dim, bias=True)

        # TODO recover
        # self.relative_position_params = nn.Parameter(torch.zeros(self.n_heads, 2 * window_size - 1, 2 * window_size -1))
        self.relative_position_params = nn.Parameter(torch.zeros((2 * window_size - 1)*(2 * window_size -1), self.n_heads))

        self.linear = nn.Linear(self.input_dim, self.output_dim)

        trunc_normal_(self.relative_position_params, std=.02)
        self.relative_position_params = torch.nn.Parameter(self.relative_position_params.view(2*window_size-1, 2*window_size-1, self.n_heads).transpose(1,2).transpose(0,1))

    def generate_mask(self, h, w, p, shift):
        """ generating the mask of SW-MSA
        Args:
            shift: shift parameters in CyclicShift.
        Returns:
            attn_mask: should be (1 1 w p p),
        """
        # supporting sqaure.
        attn_mask = torch.zeros(h, w, p, p, p, p, dtype=torch.bool, device=self.relative_position_params.device)
        if self.type == 'W':
            return attn_mask

        s = p - shift
        attn_mask[-1, :, :s, :, s:, :] = True
        attn_mask[-1, :, s:, :, :s, :] = True
        attn_mask[:, -1, :, :s, :, s:] = True
        attn_mask[:, -1, :, s:, :, :s] = True
        attn_mask = rearrange(attn_mask, 'w1 w2 p1 p2 p3 p4 -> 1 1 (w1 w2) (p1 p2) (p3 p4)')
        return attn_mask

    def forward(self, x):
        """ Forward pass of Window Multi-head Self-attention module.
        Args:
            x: input tensor with shape of [b h w c];
            attn_mask: attention mask, fill -inf where the value is True;
        Returns:
            output: tensor shape [b h w c]
        """
        if self.type!='W':
            x = torch.roll(x, shifts=(-(self.window_size//2), -(self.window_size//2)), dims=(1,2))
        x = rearrange(x, 'b (w1 p1) (w2 p2) c -> b w1 w2 p1 p2 c', p1=self.window_size, p2=self.window_size)
        h_windows = x.size(1)
        w_windows = x.size(2)
        # sqaure validation
        # assert h_windows == w_windows

        x = rearrange(x, 'b w1 w2 p1 p2 c -> b (w1 w2) (p1 p2) c', p1=self.window_size, p2=self.window_size)
        qkv = self.embedding_layer(x)
        q, k, v = rearrange(qkv, 'b nw np (threeh c) -> threeh b nw np c', c=self.head_dim).chunk(3, dim=0)
        sim = torch.einsum('hbwpc,hbwqc->hbwpq', q, k) * self.scale
        # Adding learnable relative embedding
        sim = sim + rearrange(self.relative_embedding(), 'h p q -> h 1 1 p q')
        # Using Attn Mask to distinguish different subwindows.
        if self.type != 'W':
            attn_mask = self.generate_mask(h_windows, w_windows, self.window_size, shift=self.window_size//2)
            sim = sim.masked_fill_(attn_mask, float("-inf"))

        probs = nn.functional.softmax(sim, dim=-1)
        output = torch.einsum('hbwij,hbwjc->hbwic', probs, v)
        output = rearrange(output, 'h b w p c -> b w p (h c)')
        output = self.linear(output)
        output = rearrange(output, 'b (w1 w2) (p1 p2) c -> b (w1 p1) (w2 p2) c', w1=h_windows, p1=self.window_size)

        if self.type!='W':
            output = torch.roll(output, shifts=(self.window_size//2, self.window_size//2), dims=(1,2))
        return output

    def relative_embedding(self):
        cord = torch.tensor(np.array([[i, j] for i in range(self.window_size) for j in range(self.window_size)]))
        relation = cord[:, None, :] - cord[None, :, :] + self.window_size -1
        # negative is allowed
        return self.relative_position_params[:, relation[:,:,0].long(), relation[:,:,1].long()]


class Block(nn.Module):
    def __init__(self, input_dim, output_dim, head_dim, window_size, drop_path, type='W', input_resolution=None):
        """ SwinTransformer Block
        """
        super(Block, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        assert type in ['W', 'SW']
        self.type = type
        if input_resolution <= window_size:
            self.type = 'W'

        print("Block Initial Type: {}, drop_path_rate:{:.6f}".format(self.type, drop_path))
        self.ln1 = nn.LayerNorm(input_dim)
        self.msa = WMSA(input_dim, input_dim, head_dim, window_size, self.type)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.ln2 = nn.LayerNorm(input_dim)
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, 4 * input_dim),
            nn.GELU(),
            nn.Linear(4 * input_dim, output_dim),
        )

    def forward(self, x):
        x = x + self.drop_path(self.msa(self.ln1(x)))
        x = x + self.drop_path(self.mlp(self.ln2(x)))
        return x


class ConvTransBlock(nn.Module):
    def __init__(self, conv_dim, trans_dim, head_dim, window_size, drop_path, type='W', input_resolution=None):
        """ SwinTransformer and Conv Block
        """
        super(ConvTransBlock, self).__init__()
        self.conv_dim = conv_dim
        self.trans_dim = trans_dim
        self.head_dim = head_dim
        self.window_size = window_size
        self.drop_path = drop_path
        self.type = type
        self.input_resolution = input_resolution

        assert self.type in ['W', 'SW']
        if self.input_resolution <= self.window_size:
            self.type = 'W'

        self.trans_block = Block(self.trans_dim, self.trans_dim, self.head_dim, self.window_size, self.drop_path, self.type, self.input_resolution)
        self.conv1_1 = nn.Conv2d(self.conv_dim+self.trans_dim, self.conv_dim+self.trans_dim, 1, 1, 0, bias=True)
        self.conv1_2 = nn.Conv2d(self.conv_dim+self.trans_dim, self.conv_dim+self.trans_dim, 1, 1, 0, bias=True)

        self.conv_block = nn.Sequential(
                nn.Conv2d(self.conv_dim, self.conv_dim, 3, 1, 1, bias=False),
                nn.ReLU(True),
                nn.Conv2d(self.conv_dim, self.conv_dim, 3, 1, 1, bias=False)
                )

    def forward(self, x):
        conv_x, trans_x = torch.split(self.conv1_1(x), (self.conv_dim, self.trans_dim), dim=1)
        conv_x = self.conv_block(conv_x) + conv_x
        trans_x = Rearrange('b c h w -> b h w c')(trans_x)
        trans_x = self.trans_block(trans_x)
        trans_x = Rearrange('b h w c -> b c h w')(trans_x)
        res = self.conv1_2(torch.cat((conv_x, trans_x), dim=1))
        del conv_x, trans_x
        x = x + res

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


class SCUNet(nn.Module):

    def __init__(self, in_nc=3, config=[2,2,2,2,2,2,2], dim=64, drop_path_rate=0.4, input_resolution=256):
        super(SCUNet, self).__init__()
        self.config = config
        self.dim = dim
        self.head_dim = 8
        self.window_size = 8

        # drop path rate for each layer
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(config))]

        self.m_head = [nn.Conv2d(in_nc, dim, 3, 1, 1, bias=False)]

        begin = 0
        self.m_down1 = [ConvTransBlock(dim // 2, dim // 2, self.head_dim, self.window_size, dpr[i + begin],
                                       'W' if not i % 2 else 'SW', input_resolution)
                        for i in range(config[0])] + [nn.Conv2d(dim, 2 * dim, 2, 2, 0, bias=False)]

        begin += config[0]
        self.m_down2 = [ConvTransBlock(dim, dim, self.head_dim, self.window_size, dpr[i + begin],
                                       'W' if not i % 2 else 'SW', input_resolution // 2)
                        for i in range(config[1])] + [nn.Conv2d(2 * dim, 4 * dim, 2, 2, 0, bias=False)]

        begin += config[1]
        self.m_down3 = [ConvTransBlock(2 * dim, 2 * dim, self.head_dim, self.window_size, dpr[i + begin],
                                       'W' if not i % 2 else 'SW', input_resolution // 4)
                        for i in range(config[2])] + [nn.Conv2d(4 * dim, 8 * dim, 2, 2, 0, bias=False)]

        begin += config[2]
        self.m_body = [ConvTransBlock(4 * dim, 4 * dim, self.head_dim, self.window_size, dpr[i + begin],
                                      'W' if not i % 2 else 'SW', input_resolution // 8)
                       for i in range(config[3])]

        begin += config[3]
        self.m_up3 = [nn.ConvTranspose2d(8 * dim, 4 * dim, 2, 2, 0, bias=False), ] + \
                     [ConvTransBlock(2 * dim, 2 * dim, self.head_dim, self.window_size, dpr[i + begin],
                                     'W' if not i % 2 else 'SW', input_resolution // 4)
                      for i in range(config[4])]

        begin += config[4]
        self.m_up2 = [nn.ConvTranspose2d(4 * dim, 2 * dim, 2, 2, 0, bias=False), ] + \
                     [ConvTransBlock(dim, dim, self.head_dim, self.window_size, dpr[i + begin],
                                     'W' if not i % 2 else 'SW', input_resolution // 2)
                      for i in range(config[5])]

        begin += config[5]
        self.m_up1 = [nn.ConvTranspose2d(2 * dim, dim, 2, 2, 0, bias=False), ] + \
                     [ConvTransBlock(dim // 2, dim // 2, self.head_dim, self.window_size, dpr[i + begin],
                                     'W' if not i % 2 else 'SW', input_resolution)
                      for i in range(config[6])]

        self.m_tail = [nn.Conv2d(dim, 1, 3, 1, 1, bias=False)]

        self.m_head = nn.Sequential(*self.m_head)
        self.m_down1 = nn.Sequential(*self.m_down1)
        self.m_down2 = nn.Sequential(*self.m_down2)
        self.m_down3 = nn.Sequential(*self.m_down3)
        self.m_body = nn.Sequential(*self.m_body)
        self.m_up3 = nn.Sequential(*self.m_up3)
        self.m_up2 = nn.Sequential(*self.m_up2)
        self.m_up1 = nn.Sequential(*self.m_up1)
        self.m_tail = nn.Sequential(*self.m_tail)
        # self.apply(self._init_weights)

    def forward(self, x0):
        h, w = x0.size()[-2:]
        paddingBottom = int(np.ceil(h / 64) * 64 - h)
        paddingRight = int(np.ceil(w / 64) * 64 - w)
        x0 = nn.ReplicationPad2d((0, paddingRight, 0, paddingBottom))(x0)

        x1 = self.m_head(x0)
        x2 = self.m_down1(x1)
        x3 = self.m_down2(x2)
        x4 = self.m_down3(x3)
        x = self.m_body(x4)
        x = self.m_up3(x + x4)
        x = self.m_up2(x + x3)
        x = self.m_up1(x + x2)
        x = self.m_tail(x + x1)

        x = x[..., :h, :w]

        return x

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)



class BAF_UNet(nn.Module):

    def __init__(self, in_nc=3, config=[2, 2, 2, 2, 2, 2, 2], dim=64, drop_path_rate=0.4, input_resolution=256):
        super(BAF_UNet, self).__init__()
        self.config = config
        self.dim = dim
        self.head_dim = 8
        self.window_size = 8

        # drop path rate for each layer
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(config))]

        self.m_head = [nn.Conv2d(in_nc, dim, 3, 1, 1, bias=False)]

        self.m_down1 = [Multiscaleconv_small(dim, 2 * dim)]
        self.down = [nn.MaxPool2d(kernel_size=2, stride=2)]
        self.m_down2 = [Multiscaleconv_small(2 * dim, 4 * dim)]
        self.up = [nn.Upsample(scale_factor=2)]

        begin = 0
        # self.m_down1 = [ConvTransBlock(dim//2, dim//2, self.head_dim, self.window_size, dpr[i+begin], 'W' if not i%2 else 'SW', input_resolution)
        #               for i in range(config[0])] + [nn.Conv2d(dim, 2*dim, 2, 2, 0, bias=False)]

        begin += config[0]
        self.m_down3 = [ConvTransBlock(2 * dim, 2 * dim, self.head_dim, self.window_size, dpr[i + begin],
                                       'W' if not i % 2 else 'SW', input_resolution // 2)
                        for i in range(config[1])] + [nn.Conv2d(4 * dim, 8 * dim, 2, 2, 0, bias=False)]

        begin += config[1]
        self.m_down4 = [ConvTransBlock(4 * dim, 4 * dim, self.head_dim, self.window_size, dpr[i + begin],
                                       'W' if not i % 2 else 'SW', input_resolution // 4)
                        for i in range(config[2])] + [nn.Conv2d(8 * dim, 16 * dim, 2, 2, 0, bias=False)]

        begin += config[2]
        self.m_body = [ConvTransBlock(8 * dim, 8 * dim, self.head_dim, self.window_size, dpr[i + begin],
                                      'W' if not i % 2 else 'SW', input_resolution // 8)
                       for i in range(config[3])]

        begin += config[3]
        self.m_up4 = [nn.ConvTranspose2d(16 * dim, 8 * dim, 2, 2, 0, bias=False), ] + \
                     [ConvTransBlock(4 * dim, 4 * dim, self.head_dim, self.window_size, dpr[i + begin],
                                     'W' if not i % 2 else 'SW', input_resolution // 4)
                      for i in range(config[4])]

        begin += config[4]
        self.m_up3 = [nn.ConvTranspose2d(8 * dim, 4 * dim, 2, 2, 0, bias=False), ] + \
                     [ConvTransBlock(2 * dim, 2 * dim, self.head_dim, self.window_size, dpr[i + begin],
                                     'W' if not i % 2 else 'SW', input_resolution // 2)
                      for i in range(config[5])]

        # begin += config[5]
        # self.m_up1 = [nn.ConvTranspose2d(2*dim, dim, 2, 2, 0, bias=False),] + \
        #             [ConvTransBlock(dim//2, dim//2, self.head_dim, self.window_size, dpr[i+begin], 'W' if not i%2 else 'SW', input_resolution)
        #               for i in range(config[6])]

        self.m_up2 = [up_conv(4 * dim, 4 * dim, 2)]
        self.Up_conv2 = [conv_block(4 * dim, 2 * dim)]

        self.m_up1 = [up_conv(2 * dim, dim, 2)]
        self.Up_conv1 = [conv_block( dim, dim)]

        self.m_tail = [nn.Conv2d(dim, 1, 3, 1, 1, bias=False)]

        self.m_head = nn.Sequential(*self.m_head)
        self.m_down1 = nn.Sequential(*self.m_down1)
        self.down1 = nn.Sequential(*self.down)
        self.m_down2 = nn.Sequential(*self.m_down2)
        self.down2 = nn.Sequential(*self.down)
        self.m_down3 = nn.Sequential(*self.m_down3)
        self.m_down4 = nn.Sequential(*self.m_down4)
        self.m_body = nn.Sequential(*self.m_body)
        self.m_up4 = nn.Sequential(*self.m_up4)
        self.m_up3 = nn.Sequential(*self.m_up3)
        self.m_up2 = nn.Sequential(*self.m_up2)
        self.Up_conv2 = nn.Sequential(*self.Up_conv2)
        self.m_up1 = nn.Sequential(*self.m_up1)
        self.Up_conv1 = nn.Sequential(*self.Up_conv1)
        self.m_tail = nn.Sequential(*self.m_tail)
        # self.apply(self._init_weights)

    def forward(self, x0):

        h, w = x0.size()[-2:]
        paddingBottom = int(np.ceil(h / 64) * 64 - h)
        paddingRight = int(np.ceil(w / 64) * 64 - w)
        x0 = nn.ReplicationPad2d((0, paddingRight, 0, paddingBottom))(x0)

        x1 = self.m_head(x0)
        x2 = self.m_down1(x1)
        x2 = self.down1(x2)
        x3 = self.m_down2(x2)
        x3 = self.down2(x3)
        x4 = self.m_down3(x3)
        x5 = self.m_down4(x4)
        x = self.m_body(x5)
        x = self.m_up4(x + x5)
        x = self.m_up3(x + x4)

        x = self.m_up2(x + x3)
        x = self.Up_conv2(x)
        # x = torch.cat([x, x2])
        x = self.m_up1(x)
        x = self.Up_conv1(x)
        x = self.m_tail(x + x1)

        x = x[..., :h, :w]

        return x

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)


class Attention_block(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        super(Attention_block, self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )

        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )

        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        # 下采样的gating signal 卷积
        g1 = self.W_g(g)
        # 上采样的 l 卷积
        x1 = self.W_x(x)
        # concat + relu
        psi = self.relu(g1 + x1)
        # channel 减为1，并Sigmoid,得到权重矩阵
        psi = self.psi(psi)
        # 返回加权的 x
        return x * psi


class FusingFullResolution(nn.Module):
    """
    Convolution Block
    """
    def __init__(self, in_up, out_ch,in_down, up_rate = 2, down_rate = 2):
        super(FusingFullResolution, self).__init__()

        self.Maxpool = nn.MaxPool2d(kernel_size=2, stride=down_rate)
        self.W_m = BN_Conv2d(out_ch + in_up, in_up, 3, 1, 1, bias=False)
        self.W_g = BN_Conv2d(in_up, in_up, 3, 1, 1, bias=False)
        self.m_psi = nn.Sequential(
            nn.Conv2d(out_ch, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )

        self.f_g = BN_Conv2d(in_up, out_ch, 1, 1, 0, bias=False)
        self.up = nn.Upsample(scale_factor=up_rate, mode='bilinear', align_corners=True)
        self.f_conv = BN_Conv2d(in_down, out_ch, 3, 1, 1, bias=False)
        self.g_psi = nn.Sequential(
            nn.Conv2d(out_ch, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )

    def forward(self, full, fg):
        if full.shape[2] == fg.shape[2]:
            fm = full
        else:
            fm = self.Maxpool(full)
            # fm = self.W_m(fm)
        fuse = torch.cat([fm, fg],dim=1)
        new_fg = self.W_m(fuse)

        # new_fg = self.W_g(fm)
        psi = self.m_psi(fm)
        multi_f = new_fg * psi

        f_g = self.f_g(fg)
        f_g = self.up(f_g)
        fuse_g = f_g + full
        new_fg = self.f_conv(fuse_g)
        g_psi = self.g_psi(f_g)
        global_f = new_fg * g_psi

        return multi_f, global_f


class Sim_Net(nn.Module):

    def __init__(self, in_nc=3, config=[2, 2, 2, 2, 2, 2, 2], dim=64, drop_path_rate=0.4, input_resolution=256):
        super(Sim_Net, self).__init__()
        self.config = config
        self.dim = dim
        self.head_dim = 8
        self.window_size = 8

        # drop path rate for each layer
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(config))]

        self.m_head = [nn.Conv2d(in_nc, dim, 3, 1, 1, bias=False)]

        self.m_down1 = [nn.Conv2d(dim, 2 * dim, 3, 1, 1, bias=False)]
        self.down = [nn.MaxPool2d(kernel_size=2, stride=2)]
        self.m_down2 = [nn.Conv2d(2 * dim, 4 * dim, 3, 1, 1, bias=False)]
        self.up = [nn.Upsample(scale_factor=2)]

        self.m_down3 = [nn.Conv2d(4 * dim, 8 * dim, 3, 1, 1, bias=False)]
        self.m_down4 = [nn.Conv2d(8 * dim, 16 * dim, 3, 1, 1, bias=False)]

        self.m_down5 = [nn.Conv2d(16 * dim, 16 * dim, 3, 1, 1, bias=False)]

        self.m_up4 = [up_conv(16 * dim, 8 * dim, 2)]
        self.Up_conv4 = [conv_block(8 * dim, 8 * dim)]

        self.m_up3 = [up_conv(8 * dim, 4 *dim, 2)]
        self.Up_conv3 = [conv_block(4 *dim, 4 *dim)]

        self.m_up2 = [up_conv(4 * dim, 2 * dim, 2)]
        self.Up_conv2 = [conv_block(2 * dim, 2 * dim)]

        self.m_up1 = [up_conv(2 * dim, dim, 2)]
        self.Up_conv1 = [conv_block( dim, dim)]

        self.m_tail = [nn.Conv2d(dim, 1, 3, 1, 1, bias=False)]

        self.m_head = nn.Sequential(*self.m_head)
        self.m_down1 = nn.Sequential(*self.m_down1)
        self.down1 = nn.Sequential(*self.down)
        self.m_down2 = nn.Sequential(*self.m_down2)
        self.down2 = nn.Sequential(*self.down)
        self.m_down3 = nn.Sequential(*self.m_down3)
        self.m_down4 = nn.Sequential(*self.m_down4)
        self.m_down5 = nn.Sequential(* self.m_down5)
        self.m_up4 = nn.Sequential(*self.m_up4)
        self.m_up3 = nn.Sequential(*self.m_up3)
        self.m_up2 = nn.Sequential(*self.m_up2)
        self.Up_conv2 = nn.Sequential(*self.Up_conv2)
        self.m_up1 = nn.Sequential(*self.m_up1)
        self.Up_conv1 = nn.Sequential(*self.Up_conv1)
        self.Up_conv3 = nn.Sequential(*self.Up_conv3)
        self.Up_conv4 = nn.Sequential(*self.Up_conv4)
        self.m_tail = nn.Sequential(*self.m_tail)
    def forward(self, x0):

        h, w = x0.size()[-2:]
        paddingBottom = int(np.ceil(h / 64) * 64 - h)
        paddingRight = int(np.ceil(w / 64) * 64 - w)
        x0 = nn.ReplicationPad2d((0, paddingRight, 0, paddingBottom))(x0)

        x1 = self.m_head(x0)  #1 ,32, 512, 512
        x2 = self.m_down1(x1)
        x2 = self.down1(x2)   #1 ,64, 256, 256
        x3 = self.m_down2(x2)
        x3 = self.down2(x3)   #1 ,128, 128, 128
        x4 = self.m_down3(x3)  #1 ,256, 64, 64
        x4 = self.down1(x4)  # 1 ,64, 256, 256
        x5 = self.m_down4(x4)    #1 ,512, 32, 32
        x5 = self.down1(x5)  # 1 ,64, 256, 256
        x = self.m_down5(x5)
        x = x5 + x

        x = self.m_up4(x)
        x = self.Up_conv4(x)

        x = self.m_up3(x)
        x = self.Up_conv3(x)

        x = self.m_up2(x)
        x = self.Up_conv2(x)

        x = self.m_up1(x)
        x = self.Up_conv1(x)
        x = self.m_tail(x)

        x = x[..., :h, :w]

        return x

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

class Sim_Net_MFF(nn.Module):

    def __init__(self, in_nc=3, config=[2, 2, 2, 2, 2, 2, 2], dim=64, drop_path_rate=0.4, input_resolution=256):
        super(Sim_Net_MFF, self).__init__()
        self.config = config
        self.dim = dim
        self.head_dim = 8
        self.window_size = 8

        # drop path rate for each layer
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(config))]

        self.m_head = [nn.Conv2d(in_nc, dim, 3, 1, 1, bias=False)]

        # self.m_down1 = [Multiscaleconv_small(dim, 2 * dim)]
        self.down = [nn.MaxPool2d(kernel_size=2, stride=2)]
        # self.m_down2 = [Multiscaleconv_small(2 * dim, 4 * dim)]

        self.up = [nn.Upsample(scale_factor=2)]

        begin = 0
        begin += config[0]
        self.m_down1 = [ConvTransBlock(dim//2, dim//2, self.head_dim, self.window_size, dpr[i+begin],
                                       'W' if not i%2 else 'SW', input_resolution * 2)
                      for i in range(config[0])] + [nn.Conv2d(dim, 2*dim, 2, 2, 0, bias=False)]

        begin += config[1]
        self.m_down2 = [ConvTransBlock(dim, dim, self.head_dim, self.window_size, dpr[i + begin],
                                       'W' if not i % 2 else 'SW', input_resolution)
                        for i in range(config[1])] + [nn.Conv2d(2 * dim, 4 * dim, 2, 2, 0, bias=False)]

        begin += config[2]
        self.m_down3 = [ConvTransBlock(2 * dim, 2 * dim, self.head_dim, self.window_size, dpr[i + begin],
                                       'W' if not i % 2 else 'SW', input_resolution // 2)
                        for i in range(config[1])] + [nn.Conv2d(4 * dim, 8 * dim, 2, 2, 0, bias=False)]

        begin += config[3]
        self.m_down4 = [ConvTransBlock(4 * dim, 4 * dim, self.head_dim, self.window_size, dpr[i + begin],
                                       'W' if not i % 2 else 'SW', input_resolution // 4)
                        for i in range(config[2])] + [nn.Conv2d(8 * dim, 16 * dim, 2, 2, 0, bias=False)]

        begin += config[4]
        self.m_body = [ConvTransBlock(8 * dim, 8 * dim, self.head_dim, self.window_size, dpr[i + begin],
                                      'W' if not i % 2 else 'SW', input_resolution // 8)
                       for i in range(config[3])]

        self.m_up1 = [up_conv(2 * dim, dim, 2)]
        self.Up_conv1 = [conv_block(dim, dim)]

        self.m_up2 = [up_conv(4 * dim, 2 * dim, 2)]
        self.Up_conv2 = [conv_block(2 * dim, 2 * dim)]

        self.m_up3 = [up_conv(8 * dim, 4 * dim, 2)]
        self.Up_conv3 = [conv_block(4 * dim, 4 * dim)]

        self.m_up4 = [up_conv(16 * dim, 8 * dim, 2)]
        self.Up_conv4 = [conv_block(8 * dim, 8 * dim)]

        self.m_tail = [nn.Conv2d(dim, 1, 3, 1, 1, bias=False)]

        self.m_head = nn.Sequential(*self.m_head)
        self.m_down1 = nn.Sequential(*self.m_down1)
        self.down1 = nn.Sequential(*self.down)
        self.m_down2 = nn.Sequential(*self.m_down2)
        self.down2 = nn.Sequential(*self.down)
        self.m_down3 = nn.Sequential(*self.m_down3)
        self.m_down4 = nn.Sequential(*self.m_down4)
        self.m_body = nn.Sequential(*self.m_body)
        self.m_up4 = nn.Sequential(*self.m_up4)
        self.m_up3 = nn.Sequential(*self.m_up3)
        self.m_up2 = nn.Sequential(*self.m_up2)
        self.Up_conv2 = nn.Sequential(*self.Up_conv2)
        self.m_up1 = nn.Sequential(*self.m_up1)
        self.Up_conv1 = nn.Sequential(*self.Up_conv1)
        self.Up_conv3 = nn.Sequential(*self.Up_conv3)
        self.Up_conv4 = nn.Sequential(*self.Up_conv4)
        self.m_tail = nn.Sequential(*self.m_tail)

        # self.apply(self._init_weights)

    def forward(self, x0):

        h, w = x0.size()[-2:]
        paddingBottom = int(np.ceil(h / 64) * 64 - h)
        paddingRight = int(np.ceil(w / 64) * 64 - w)
        x0 = nn.ReplicationPad2d((0, paddingRight, 0, paddingBottom))(x0)

        x1 = self.m_head(x0)            #[2, 32,512, 512]
        x2 = self.m_down1(x1)
        x3 = self.m_down2(x2)
        x4 = self.m_down3(x3)
        x5 = self.m_down4(x4)
        x = self.m_body(x5)

        x = self.m_up4(x)
        x = self.Up_conv4(x)

        x = self.m_up3(x)
        x = self.Up_conv3(x)

        x = self.m_up2(x)
        x = self.Up_conv2(x)

        x = self.m_up1(x)
        x = self.Up_conv1(x)
        x = self.m_tail(x)

        x = x[..., :h, :w]

        return x

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)


class Sim_Net_MFF_MRF(nn.Module):

    def __init__(self, in_nc=3, config=[2, 2, 2, 2, 2, 2, 2], dim=64, drop_path_rate=0.4, input_resolution=256):
        super(Sim_Net_MFF_MRF, self).__init__()
        self.config = config
        self.dim = dim
        self.head_dim = 8
        self.window_size = 8

        # drop path rate for each layer
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(config))]

        self.m_head = [nn.Conv2d(in_nc, dim, 3, 1, 1, bias=False)]

        # self.m_down1 = [Multiscaleconv_small(dim, 2 * dim)]
        self.down = [nn.MaxPool2d(kernel_size=2, stride=2)]
        # self.m_down2 = [Multiscaleconv_small(2 * dim, 4 * dim)]

        self.up = [nn.Upsample(scale_factor=2)]

        begin = 0
        begin += config[0]
        self.m_down1 = [ConvTransBlock(dim//2, dim//2, self.head_dim, self.window_size, dpr[i+begin],
                                       'W' if not i%2 else 'SW', input_resolution * 2)
                      for i in range(config[0])] + [nn.Conv2d(dim, 2*dim, 2, 2, 0, bias=False)]

        begin += config[1]
        self.m_down2 = [ConvTransBlock(dim, dim, self.head_dim, self.window_size, dpr[i + begin],
                                       'W' if not i % 2 else 'SW', input_resolution)
                        for i in range(config[1])] + [nn.Conv2d(2 * dim, 4 * dim, 2, 2, 0, bias=False)]

        begin += config[2]
        self.m_down3 = [ConvTransBlock(2 * dim, 2 * dim, self.head_dim, self.window_size, dpr[i + begin],
                                       'W' if not i % 2 else 'SW', input_resolution // 2)
                        for i in range(config[1])] + [nn.Conv2d(4 * dim, 8 * dim, 2, 2, 0, bias=False)]

        begin += config[3]
        self.m_down4 = [ConvTransBlock(4 * dim, 4 * dim, self.head_dim, self.window_size, dpr[i + begin],
                                       'W' if not i % 2 else 'SW', input_resolution // 4)
                        for i in range(config[2])] + [nn.Conv2d(8 * dim, 16 * dim, 2, 2, 0, bias=False)]

        begin += config[4]
        self.m_body = [ConvTransBlock(8 * dim, 8 * dim, self.head_dim, self.window_size, dpr[i + begin],
                                      'W' if not i % 2 else 'SW', input_resolution // 8)
                       for i in range(config[3])]

        self.m_up1 = [up_conv(2 * dim, dim, 2)]
        self.Up_conv1 = [conv_block(dim, dim)]

        self.m_up2 = [up_conv(4 * dim, 2 * dim, 2)]
        self.Up_conv2 = [conv_block(2 * dim, 2 * dim)]

        self.m_up3 = [up_conv(8 * dim, 4 * dim, 2)]
        self.Up_conv3 = [conv_block(4 * dim, 4 * dim)]

        self.m_up4 = [up_conv(16 * dim, 8 * dim, 2)]
        self.Up_conv4 = [conv_block(8 * dim, 8 * dim)]

        self.full_up5 = [up_conv(16 * dim, dim, 16)]
        self.full_up4 = [up_conv(8 * dim, dim, 8)]
        self.full_up3 = [up_conv(4 * dim, dim, 4)]
        self.full_up2 = [up_conv(2 * dim, dim, 2)]
        self.full_up1 = [up_conv(dim, dim, 1)]

        self.fuse4 = FusingFullResolution(8 * dim, dim, dim, 8, 8)
        self.fuse3 = FusingFullResolution(4 * dim, dim, dim, 4, 4)
        self.fuse2 = FusingFullResolution(2 * dim, dim, dim, 2, 2)
        self.fuse1 = FusingFullResolution(dim, dim, dim, 1, 1)

        self.m_tail = [nn.Conv2d(dim, 1, 3, 1, 1, bias=False)]

        self.m_head = nn.Sequential(*self.m_head)
        self.m_down1 = nn.Sequential(*self.m_down1)
        self.down1 = nn.Sequential(*self.down)
        self.m_down2 = nn.Sequential(*self.m_down2)
        self.down2 = nn.Sequential(*self.down)
        self.m_down3 = nn.Sequential(*self.m_down3)
        self.m_down4 = nn.Sequential(*self.m_down4)
        self.m_body = nn.Sequential(*self.m_body)
        self.m_up4 = nn.Sequential(*self.m_up4)
        self.m_up3 = nn.Sequential(*self.m_up3)
        self.m_up2 = nn.Sequential(*self.m_up2)
        self.Up_conv2 = nn.Sequential(*self.Up_conv2)
        self.m_up1 = nn.Sequential(*self.m_up1)
        self.Up_conv1 = nn.Sequential(*self.Up_conv1)
        self.Up_conv3 = nn.Sequential(*self.Up_conv3)
        self.Up_conv4 = nn.Sequential(*self.Up_conv4)
        self.m_tail = nn.Sequential(*self.m_tail)

        self.full_up1 = nn.Sequential(*self.full_up1)
        self.full_up2 = nn.Sequential(*self.full_up2)
        self.full_up3 = nn.Sequential(*self.full_up3)
        self.full_up4 = nn.Sequential(*self.full_up4)
        self.full_up5 = nn.Sequential(*self.full_up5)
        # self.apply(self._init_weights)

    def forward(self, x0):

        h, w = x0.size()[-2:]
        paddingBottom = int(np.ceil(h / 64) * 64 - h)
        paddingRight = int(np.ceil(w / 64) * 64 - w)
        x0 = nn.ReplicationPad2d((0, paddingRight, 0, paddingBottom))(x0)

        x1 = self.m_head(x0)            #[2, 32,512, 512]
        x2 = self.m_down1(x1)
        x3 = self.m_down2(x2)
        x4 = self.m_down3(x3)
        x5 = self.m_down4(x4)
        x = self.m_body(x5)

        # full resolution image
        x_fuse5 = self.full_up5(x)
        x_fuse4 = self.full_up4(x4)
        x_fuse3 = self.full_up3(x3)
        x_fuse2 = self.full_up2(x2)
        x_fuse1 = self.full_up1(x1)
        x_fuse = x_fuse5 + x_fuse4 + x_fuse3 + x_fuse2 + x_fuse1
        del x_fuse1, x_fuse2, x_fuse3, x_fuse4, x_fuse5   #[2, 32, 512, 512]

        x = self.m_up4(x)
        x = self.Up_conv4(x)

        x = self.m_up3(x)
        x = self.Up_conv3(x)

        x = self.m_up2(x)
        x = self.Up_conv2(x)

        x = self.m_up1(x)
        x = self.Up_conv1(x)

        x =x + x_fuse

        x = self.m_tail(x)
        x = x[..., :h, :w]

        return x

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

class BAF_Net(nn.Module):

    def __init__(self, in_nc=3, config=[2, 2, 2, 2, 2, 2, 2], dim=64, drop_path_rate=0.4, input_resolution=256):
        super(BAF_Net, self).__init__()
        self.config = config
        self.dim = dim
        self.head_dim = 8
        self.window_size = 8

        # drop path rate for each layer
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(config))]

        self.m_head = [nn.Conv2d(in_nc, dim, 3, 1, 1, bias=False)]

        # self.m_down1 = [Multiscaleconv_small(dim, 2 * dim)]
        self.down = [nn.MaxPool2d(kernel_size=2, stride=2)]
        # self.m_down2 = [Multiscaleconv_small(2 * dim, 4 * dim)]

        self.up = [nn.Upsample(scale_factor=2)]

        begin = 0
        begin += config[0]
        self.m_down1 = [ConvTransBlock(dim//2, dim//2, self.head_dim, self.window_size, dpr[i+begin],
                                       'W' if not i%2 else 'SW', input_resolution * 2)
                      for i in range(config[0])] + [nn.Conv2d(dim, 2*dim, 2, 2, 0, bias=False)]

        begin += config[1]
        self.m_down2 = [ConvTransBlock(dim, dim, self.head_dim, self.window_size, dpr[i + begin],
                                       'W' if not i % 2 else 'SW', input_resolution)
                        for i in range(config[1])] + [nn.Conv2d(2 * dim, 4 * dim, 2, 2, 0, bias=False)]

        begin += config[2]
        self.m_down3 = [ConvTransBlock(2 * dim, 2 * dim, self.head_dim, self.window_size, dpr[i + begin],
                                       'W' if not i % 2 else 'SW', input_resolution // 2)
                        for i in range(config[1])] + [nn.Conv2d(4 * dim, 8 * dim, 2, 2, 0, bias=False)]

        begin += config[3]
        self.m_down4 = [ConvTransBlock(4 * dim, 4 * dim, self.head_dim, self.window_size, dpr[i + begin],
                                       'W' if not i % 2 else 'SW', input_resolution // 4)
                        for i in range(config[2])] + [nn.Conv2d(8 * dim, 16 * dim, 2, 2, 0, bias=False)]

        begin += config[4]
        self.m_body = [ConvTransBlock(8 * dim, 8 * dim, self.head_dim, self.window_size, dpr[i + begin],
                                      'W' if not i % 2 else 'SW', input_resolution // 8)
                       for i in range(config[3])]

        self.m_up1 = [up_conv(2 * dim, dim, 2)]
        self.Up_conv1 = [conv_block(2 * dim, dim)]

        self.m_up2 = [up_conv(4 * dim, 2 * dim, 2)]
        self.Up_conv2 = [conv_block(4 * dim, 2 * dim)]

        self.m_up3 = [up_conv(8 * dim, 4 * dim, 2)]
        self.Up_conv3 = [conv_block(8 * dim, 4 * dim)]

        self.m_up4 = [up_conv(16 * dim, 8 * dim, 2)]
        self.Up_conv4 = [conv_block(16 * dim, 8 * dim)]

        self.full_up5 = [up_conv(16 * dim, dim, 16)]
        self.full_up4 = [up_conv(8 * dim, dim, 8)]
        self.full_up3 = [up_conv(4 * dim, dim, 4)]
        self.full_up2 = [up_conv(2 * dim, dim, 2)]
        self.full_up1 = [up_conv(dim, dim, 1)]

        self.fuse4 = FusingFullResolution(8 * dim, dim, dim, 8, 8)
        self.fuse3 = FusingFullResolution(4 * dim, dim, dim, 4, 4)
        self.fuse2 = FusingFullResolution(2 * dim, dim, dim, 2, 2)
        self.fuse1 = FusingFullResolution(dim, dim, dim, 1, 1)

        self.m_tail = [nn.Conv2d(dim, 1, 3, 1, 1, bias=False)]

        self.m_head = nn.Sequential(*self.m_head)
        self.m_down1 = nn.Sequential(*self.m_down1)
        self.down1 = nn.Sequential(*self.down)
        self.m_down2 = nn.Sequential(*self.m_down2)
        self.down2 = nn.Sequential(*self.down)
        self.m_down3 = nn.Sequential(*self.m_down3)
        self.m_down4 = nn.Sequential(*self.m_down4)
        self.m_body = nn.Sequential(*self.m_body)
        self.m_up4 = nn.Sequential(*self.m_up4)
        self.m_up3 = nn.Sequential(*self.m_up3)
        self.m_up2 = nn.Sequential(*self.m_up2)
        self.Up_conv2 = nn.Sequential(*self.Up_conv2)
        self.m_up1 = nn.Sequential(*self.m_up1)
        self.Up_conv1 = nn.Sequential(*self.Up_conv1)
        self.m_tail = nn.Sequential(*self.m_tail)

        self.full_up1 = nn.Sequential(*self.full_up1)
        self.full_up2 = nn.Sequential(*self.full_up2)
        self.full_up3 = nn.Sequential(*self.full_up3)
        self.full_up4 = nn.Sequential(*self.full_up4)
        self.full_up5 = nn.Sequential(*self.full_up5)
        # self.apply(self._init_weights)

    def forward(self, x0):

        h, w = x0.size()[-2:]
        paddingBottom = int(np.ceil(h / 64) * 64 - h)
        paddingRight = int(np.ceil(w / 64) * 64 - w)
        x0 = nn.ReplicationPad2d((0, paddingRight, 0, paddingBottom))(x0)

        x1 = self.m_head(x0)            #[2, 32,512, 512]
        x2 = self.m_down1(x1)
        x3 = self.m_down2(x2)
        x4 = self.m_down3(x3)
        x5 = self.m_down4(x4)
        x = self.m_body(x5)

        # full resolution image
        x_fuse5 = self.full_up5(x)
        x_fuse4 = self.full_up4(x4)
        x_fuse3 = self.full_up3(x3)
        x_fuse2 = self.full_up2(x2)
        x_fuse1 = self.full_up1(x1)
        x_fuse = x_fuse5 + x_fuse4 + x_fuse3 + x_fuse2 + x_fuse1
        del x_fuse1, x_fuse2, x_fuse3, x_fuse4, x_fuse5   #[2, 32, 512, 512]

        x = self.m_up4(x + x5)  #[2, 32, 512, 512]
        del x5

        multi_f4, global_f = self.fuse4(x_fuse, x)
        del x_fuse

        x = self.m_up3(multi_f4 + x4)
        multi_f3, global_f = self.fuse3(global_f, x)
        del multi_f4, x4

        x = self.m_up2(multi_f3 + x3)
        multi_f2, global_f = self.fuse2(global_f, x)
        del multi_f3, x3

        x = self.m_up1(multi_f2 + x2)
        multi_f1, global_f = self.fuse1(global_f, x)
        del multi_f2, x2

        x = torch.cat([multi_f1, global_f], 1)
        x = self.Up_conv1(x)
        del multi_f1, global_f
        x = x + x1
        del x1

        x = self.m_tail(x)
        x = x[..., :h, :w]

        return x

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)



class MBFUNet_B(nn.Module):

    def __init__(self, in_nc=3, config=[2, 2, 2, 2, 2, 2, 2], dim=64, drop_path_rate=0.4, input_resolution=256):
        super(MBFUNet_B, self).__init__()
        self.config = config
        self.dim = dim
        self.head_dim = 8
        self.window_size = 8

        # drop path rate for each layer
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(config))]

        self.m_head = [nn.Conv2d(in_nc, dim, 3, 1, 1, bias=False)]

        self.m_down1 = [Multiscaleconv_small(dim, 2 * dim)]
        self.down = [nn.MaxPool2d(kernel_size=2, stride=2)]
        self.m_down2 = [Multiscaleconv_small(2 * dim, 4 * dim)]
        self.up = [nn.Upsample(scale_factor=2)]

        begin = 0
        # self.m_down1 = [ConvTransBlock(dim//2, dim//2, self.head_dim, self.window_size, dpr[i+begin], 'W' if not i%2 else 'SW', input_resolution)
        #               for i in range(config[0])] + [nn.Conv2d(dim, 2*dim, 2, 2, 0, bias=False)]

        begin += config[0]
        self.m_down3 = [ConvTransBlock(2 * dim, 2 * dim, self.head_dim, self.window_size, dpr[i + begin],
                                       'W' if not i % 2 else 'SW', input_resolution // 2)
                        for i in range(config[1])] + [nn.Conv2d(4 * dim, 8 * dim, 2, 2, 0, bias=False)]

        begin += config[1]
        self.m_down4 = [ConvTransBlock(4 * dim, 4 * dim, self.head_dim, self.window_size, dpr[i + begin],
                                       'W' if not i % 2 else 'SW', input_resolution // 4)
                        for i in range(config[2])] + [nn.Conv2d(8 * dim, 16 * dim, 2, 2, 0, bias=False)]

        begin += config[2]
        self.m_body = [ConvTransBlock(8 * dim, 8 * dim, self.head_dim, self.window_size, dpr[i + begin],
                                      'W' if not i % 2 else 'SW', input_resolution // 8)
                       for i in range(config[3])]

        begin += config[3]
        self.m_up4 = [nn.ConvTranspose2d(16 * dim, 8 * dim, 2, 2, 0, bias=False), ] + \
                     [ConvTransBlock(4 * dim, 4 * dim, self.head_dim, self.window_size, dpr[i + begin],
                                     'W' if not i % 2 else 'SW', input_resolution // 4)
                      for i in range(config[4])]

        begin += config[4]
        self.m_up3 = [nn.ConvTranspose2d(8 * dim, 4 * dim, 2, 2, 0, bias=False), ] + \
                     [ConvTransBlock(2 * dim, 2 * dim, self.head_dim, self.window_size, dpr[i + begin],
                                     'W' if not i % 2 else 'SW', input_resolution // 2)
                      for i in range(config[5])]

        # begin += config[5]
        # self.m_up1 = [nn.ConvTranspose2d(2*dim, dim, 2, 2, 0, bias=False),] + \
        #             [ConvTransBlock(dim//2, dim//2, self.head_dim, self.window_size, dpr[i+begin], 'W' if not i%2 else 'SW', input_resolution)
        #               for i in range(config[6])]

        self.m_up2 = [up_conv(4 * dim, 2 * dim, 2)]
        self.Up_conv2 = [conv_block(4 * dim, 2 * dim)]

        self.m_up1 = [up_conv(2 * dim, dim, 2)]
        self.Up_conv1 = [conv_block(2 * dim, dim)]

        self.full_up5 = [up_conv(16 * dim, dim, 16)]
        self.full_up4 = [up_conv(8 * dim, dim, 8)]
        self.full_up3 = [up_conv(4 * dim, dim, 4)]
        self.full_up2 = [up_conv(2 * dim, dim, 2)]
        self.full_up1 = [up_conv(dim, dim, 1)]

        self.fuse4 = FusingFullResolution(8 * dim, dim, dim, 8, 8)
        self.fuse3 = FusingFullResolution(4 * dim, dim, dim, 4, 4)
        self.fuse2 = FusingFullResolution(2 * dim, dim, dim, 2, 2)
        self.fuse1 = FusingFullResolution(dim, dim, dim, 1, 1)

        self.m_tail = [nn.Conv2d(dim, 1, 3, 1, 1, bias=False)]

        self.m_head = nn.Sequential(*self.m_head)
        self.m_down1 = nn.Sequential(*self.m_down1)
        self.down1 = nn.Sequential(*self.down)
        self.m_down2 = nn.Sequential(*self.m_down2)
        self.down2 = nn.Sequential(*self.down)
        self.m_down3 = nn.Sequential(*self.m_down3)
        self.m_down4 = nn.Sequential(*self.m_down4)
        self.m_body = nn.Sequential(*self.m_body)
        self.m_up4 = nn.Sequential(*self.m_up4)
        self.m_up3 = nn.Sequential(*self.m_up3)
        self.m_up2 = nn.Sequential(*self.m_up2)
        self.Up_conv2 = nn.Sequential(*self.Up_conv2)
        self.m_up1 = nn.Sequential(*self.m_up1)
        self.Up_conv1 = nn.Sequential(*self.Up_conv1)
        self.m_tail = nn.Sequential(*self.m_tail)

        self.full_up1 = nn.Sequential(*self.full_up1)
        self.full_up2 = nn.Sequential(*self.full_up2)
        self.full_up3 = nn.Sequential(*self.full_up3)
        self.full_up4 = nn.Sequential(*self.full_up4)
        self.full_up5 = nn.Sequential(*self.full_up5)
        # self.apply(self._init_weights)

    def forward(self, x0):

        h, w = x0.size()[-2:]
        paddingBottom = int(np.ceil(h / 64) * 64 - h)
        paddingRight = int(np.ceil(w / 64) * 64 - w)
        x0 = nn.ReplicationPad2d((0, paddingRight, 0, paddingBottom))(x0)

        x1 = self.m_head(x0)
        x2 = self.m_down1(x1)
        x2 = self.down1(x2)
        x3 = self.m_down2(x2)
        x3 = self.down2(x3)
        x4 = self.m_down3(x3)
        x5 = self.m_down4(x4)
        x = self.m_body(x5)

        # full resolution image
        x_fuse5 = self.full_up5(x)
        x_fuse4 = self.full_up4(x4)
        x_fuse3 = self.full_up3(x3)
        x_fuse2 = self.full_up2(x2)
        x_fuse1 = self.full_up1(x1)
        x_fuse = x_fuse5 + x_fuse4 + x_fuse3 + x_fuse2 + x_fuse1
        del x_fuse1, x_fuse2, x_fuse3, x_fuse4, x_fuse5

        x = self.m_up4(x + x5)

        multi_f4, global_f = self.fuse4(x_fuse, x)
        del x_fuse

        x = self.m_up3(multi_f4 + x4)
        multi_f3, global_f = self.fuse3(global_f, x)
        del multi_f4

        x = self.m_up2(multi_f3 + x3)
        multi_f2, global_f = self.fuse2(global_f, x)
        del multi_f3

        # x = self.Up_conv2(multi_f2)
        x = self.m_up1(multi_f2 + x2)
        multi_f1, global_f = self.fuse1(global_f, x)
        del multi_f2

        # x = self.m_up4(x + x5)
        # x = self.m_up3(x + x4)
        #
        # x = self.m_up2(x + x3)
        # x = self.Up_conv2(x)
        x = torch.cat([multi_f1, global_f], 1)
        # x = self.m_up1(x)
        x = self.Up_conv1(x)
        x = self.m_tail(x + x1)

        x = x[..., :h, :w]

        return x

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)




if __name__ == '__main__':

    # torch.cuda.empty_cache()
    net = SCUNet()

    x = torch.randn((2, 3, 64, 128))
    x = net(x)
    print(x.shape)
