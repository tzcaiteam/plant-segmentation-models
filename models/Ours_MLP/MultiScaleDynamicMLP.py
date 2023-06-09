from __future__ import print_function, division
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
import torch
#from nnunet.network_architecture.neural_network import SegmentationNetwork
import math
from timm.models.layers import DropPath, to_2tuple, trunc_normal_



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


class DynamicBlock(nn.Module):
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
        self.proj = nn.Conv2d(in_dim, in_dim, kernel_size= 3, stride=stride, padding=1)
        self.norm = nn.BatchNorm2d(out_dim)
        self.fc1 = nn.Conv2d(in_dim, self.H * self.W * in_dim, kernel_size= 1, stride=1, padding=0)
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
        x = self.norm(x)
        y, z = torch.split(x, C // 2, dim= 1)

        x_shift_r = self.fc1(y.permute(0, 2, 3, 1))
        x_shift_r = self.sublayernorm(x_shift_r)
        x_shift_r = self.drop(x_shift_r)

        x_shift_c = self.fc1(z.permute(0, 2, 3, 1))
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


class Attention_block(nn.Module):
    '''
      Attention block/mechanism
    '''
    def __init__(self, in_dims, out_dims):
        super(Attention_block, self).__init__()

        self.conv_x = nn.Conv2d(out_dims, out_dims, kernel_size=1, stride=1, padding=0, bias=True)
        self.pool_x = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv_g = nn.Conv2d(in_dims, out_dims, kernel_size=1, stride=1, padding=0, bias=True)
        self.relu = nn.ReLU(inplace=True)

        self.Up = nn.Upsample(scale_factor=2, mode='bilinear')
        self.psi = nn.Conv2d(out_dims, 1, kernel_size=1, stride=1, padding=0, bias=True)

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

        # upsample_psi = upsample_psi.view(-1, H, W)
        y = torch.matmul(x,upsample_psi)
        # x_psi = x.reshape(-1, H, W)
        # upsample_psi = torch.repeat_interleave(upsample_psi.unsqueeze(dim=1), repeats=C, dim=1).squeeze(2)
        # # upsample_psi = upsample_psi.expand(shape=shape_x)
        # y = torch.bmm(upsample_psi, x_psi)
        # y = y.reshape(B, C, H, W)
        return y


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
        B, C, H, W = x.shape
        shape_x = x.shape
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

        # self.patch_embed3 = OverlapPatchEmbed(img_size=img_size // 4, patch_size=3, stride=2, in_chans=embed_dims[0],
        #                                       embed_dim=embed_dims[1])
        # self.patch_embed4 = OverlapPatchEmbed(img_size=img_size // 8, patch_size=3, stride=2, in_chans=embed_dims[1],
        #                                       embed_dim=embed_dims[2])
        #
        # self.block1 = shiftedBlock(dim=embed_dims[1],  mlp_ratio=1, drop=drop_rate)
        # self.block2 = shiftedBlock(dim=embed_dims[2], mlp_ratio=1,drop=drop_rate)
        # self.dblock1 = shiftedBlock(dim=embed_dims[1], mlp_ratio=1, drop=drop_rate)
        # self.dblock2 = shiftedBlock(dim=embed_dims[0], mlp_ratio=1,drop=drop_rate)

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

        self.mlp_block1 = MLPBlock(in_dim=embed_dims[0], out_dim=embed_dims[1], mlp_ratio=1, drop=drop_rate,
                                   img_size=img_size // 4, patch_size=3, stride=2)
        self.mlp_block2 = MLPBlock(in_dim=embed_dims[1], out_dim=embed_dims[2], mlp_ratio=1, drop=drop_rate,
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

        self.mlp_block1 = MLPBlock(in_dim=embed_dims[0], out_dim=embed_dims[1], mlp_ratio=1, drop=drop_rate,
                                   img_size=img_size // 4, patch_size=3, stride=2)
        self.mlp_block2 = MLPBlock(in_dim=embed_dims[1], out_dim=embed_dims[2], mlp_ratio=1, drop=drop_rate,
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
        pool3 = self.Maxpool3(e3)     #64*64*64

        e4 = self.MConv3(pool3)
        # e4 = self.mlp_block1(e3)        # 4, 128,64,64

        e5 = self.mlp_block2(e4)

        d5 = self.Up_conv5(e5)  # input: 256*32*32,  output: 256*32*32
        att_5 = self.att_block1(e4, d5)
        up4 = self.Up5(d5)                  #input: 256*32*32,  output: 256*64*64
        d4 = torch.cat((att_5, up4), dim=1)  # input: 384*64*64

        d4 = self.Up_conv4(d4)  # input: 384*64*64,  output: 128*64*64
        # att_4 = self.att_block2(e3, d4)
        up3 = self.Up4(d4)  # input: 128*64*64,  output: 128*128*128
        d3 = torch.cat((e3, up3), dim=1)  # input: 192*128*128

        d3 = self.Up_conv3(d3)  # input: 192*128*128, output: 64*128*128
        # att_3 = self.att_block3(e2, d3)
        up2 = self.Up3(d3)  # input: 64*128*128,  output: 64*256*256
        d2 = torch.cat((up2, e2), dim=1)  # input: 96*256*256

        d2 = self.Up_conv2(d2)  # input: 96*256*256, output: 32*256*256
        # att_2 = self.att_block4(e1, d2)
        up1 = self.Up2(d2)  # input: 32*256*256,  output: 128*512*512
        d1 = torch.cat((up1, e1), dim=1)  # input: 48*512*512

        d1 = self.Up_conv1(d1)  # input: 48*512*512, output: 64*128*128
        out = self.Conv(d1)

        del d5, d4, d3, d2, d1, e4, e3, e2, e1, up1, up2, up3, up4, pool1, pool2, att_5

        return out

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

def conv_bn(in_channels, out_channels, kernel_size, stride, padding, groups=1):
    result = nn.Sequential()
    result.add_module('conv', nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                                                  kernel_size=kernel_size, stride=stride, padding=padding, groups=groups, bias=False))
    result.add_module('bn', nn.BatchNorm2d(num_features=out_channels))
    return result



class MSMLPBlock(nn.Module):
    """ Image to Patch Embedding
    """
    def __init__(self, in_dim, out_dim=768, mlp_ratio=4., drop = 0.4, img_size=224, patch_size=7, num_patches = 2,
                  stride=4, globalperceptron_reduce = 4):
        super().__init__()

        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        self.drop = drop
        self.num_patches = num_patches
        self.in_channel = out_dim

        self.h, self.w = patch_size
        self.h_parts, self.w_parts = img_size[0] // patch_size[0], img_size[1] // patch_size[1]
        # self.num_patches = self.h_parts * self.w_parts

        self.proj = nn.Conv2d(in_dim, out_dim, kernel_size=3, stride=1, padding=1)
        self.gp = GlobalPerceptron(input_channels=out_dim, internal_neurons=out_dim // globalperceptron_reduce)

        self.fc3 = nn.Conv2d(self.h * self.w * self.num_patches, self.h * self.w * self.num_patches, 1, 1, 0, bias=True,
                             groups=self.num_patches)

        self.fc3_bn = nn.BatchNorm2d(self.num_patches)
        self.conv_branch1 = conv_bn(self.num_patches, self.num_patches, kernel_size=1, stride=1, padding=0,
                      groups=self.num_patches)
        self.conv_branch2 = conv_bn(self.num_patches, self.num_patches, kernel_size=3, stride=1, padding=1,
                      groups=self.num_patches)

        self.act = nn.GELU()

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

    def partition(self, x, h_parts, w_parts):
        x = x.reshape(-1, self.in_channel, h_parts, self.h, w_parts, self.w)
        x = x.permute(0, 2, 4, 1, 3, 5)
        return x

    def partition_affine(self, x, h_parts, w_parts):
        fc_inputs = x.reshape(-1, self.num_patches * self.h * self.w, 1, 1)
        out = self.fc3(fc_inputs)
        out = out.reshape(-1, self.num_patches, self.h, self.w)
        out = self.fc3_bn(out)
        out = out.reshape(-1, h_parts, w_parts, self.num_patches, self.h, self.w)
        return out

    def forward(self, x):
        x = self.proj(x)
        B, C, H, W = x.shape

        global_vec = self.gp(x)

        origin_shape = x.size()
        h_parts = origin_shape[2] // self.h
        w_parts = origin_shape[3] // self.w

        partitions = self.partition(x, self.h_parts, self.w_parts)

        #   Channel Perceptron
        fc3_out = self.partition_affine(partitions, self.h_parts, self.w_parts)

        #   Local Perceptron
        conv_inputs = partitions.reshape(-1, self.num_patches, self.h, self.w)
        conv_out = 0

        conv_out += self.conv_branch1(conv_inputs)
        conv_out += self.conv_branch2(conv_out)
        conv_out = conv_out.reshape(-1, h_parts, w_parts, self.num_patches, self.h, self.w)
        fc3_out += conv_out

        fc3_out = fc3_out.permute(0, 3, 1, 4, 2, 5)  # N, O, h_parts, out_h, w_parts, out_w
        out = fc3_out.reshape(*origin_shape)
        out = out * global_vec
        return out


class MSD_MLPNet(nn.Module):
    """
    UNet - Unet with down-upsampling operation
    Paper : https://arxiv.org/abs/1505.04597    """

    def __init__(self, in_ch=1, out_ch=3):
        super(MSD_MLPNet, self).__init__()

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

        self.att_block1 = Attention_block(in_dims=filters[4],out_dims=filters[3])
        self.att_block2 = Attention_block(in_dims=filters[3], out_dims=filters[2])
        self.att_block3 = Attention_block(in_dims=filters[2], out_dims=filters[1])
        self.att_block4 = Attention_block(in_dims=filters[1], out_dims=filters[0])
        self.branch = MSMLPBlock(in_dim=embed_dims[2], out_dim=embed_dims[2], mlp_ratio=1, drop=drop_rate,
                                   img_size=img_size // 16, patch_size=8, num_patches=2, stride=1)

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

        pool3 = self.Maxpool3(e3)     #64*64*64
        e4 = self.MConv3(pool3)

        pool4 = self.Maxpool4(e4)     #128*32*32
        e5 = self.MConv4(pool4)                #256*32*32

        d5 = self.branch(e5)

        # d5 = self.Up_conv5(e5)  # input: 256*32*32,  output: 256*32*32
        # att_5 = self.att_block1(e4, d5)

        up4 = self.Up5(d5)                  #input: 256*32*32,  output: 256*64*64
        d4 = torch.cat((up4, e4), dim=1)  # input: 384*64*64
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
