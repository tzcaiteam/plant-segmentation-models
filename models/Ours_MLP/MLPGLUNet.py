
import torch
from torch import nn
import torch.nn.functional as F
import einops
import numpy as np
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
import math


def conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=1, bias=False)

def unblock_images_einops(x, grid_size, patch_size):
  """patches to images."""
  x = einops.rearrange(
      x, "n (gh gw) (fh fw) c -> n (gh fh) (gw fw) c",
      gh=grid_size[0], gw=grid_size[1], fh=patch_size[0], fw=patch_size[1])
  return x

def block_images_einops(x, patch_size):
    """Image to patches."""
    batch, height, width, channels = x.shape
    grid_height = height // patch_size[0]
    grid_width = width // patch_size[1]
    x = einops.rearrange(
        x, "n (gh fh) (gw fw) c -> n (gh gw) (fh fw) c",
      gh=grid_height, gw=grid_width, fh=patch_size[0], fw=patch_size[1])
    return x

class BlockGatingUnit(nn.Module):
  """A SpatialGatingUnit as defined in the gMLP paper.

  The 'spatial' dim is defined as the **second last**.
  If applied on other dims, you should swapaxes first.
  """
  def __init__(self, num_channels,num_axis):
      super().__init__()

      self.dim = num_channels
      self.layernorm = nn.LayerNorm(num_channels)
      self.fc = nn.Linear(num_axis, num_axis)

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

  def forward(self,x):
      # u, v = np.split(x, 2, axis=-1)
      n, size_block, num_block, num_channels = x.shape

      u = x
      v = self.layernorm(x)
      # n = x.shape[-3]   # get spatial dim

      v = v.permute(0, 3, 1, 2)
      v = self.fc(v)

      v = v.permute(0, 2, 3, 1)
      # v = v.permute(0, 1, 2, 3)
      return u * (v + 1.)

class GridGatingUnit(nn.Module):
    """A SpatialGatingUnit as defined in the gMLP paper.
      The 'spatial' dim is defined as the second last.
        If applied on other dims, you should swapaxes first.
    """
    def __init__(self, num_channels, num_axis):
        super().__init__()

        self.num_channels = num_channels
        self.num_axis = num_axis
        self.fc = nn.Linear(num_axis, num_axis)
        self.layernorm = nn.LayerNorm(num_channels)

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

    def forward(self,x):
        # u, v = np.split(x, 2, axis=-1)
        n, size_block, num_block, num_channels = x.shape

        u = x
        v = self.layernorm(x)
        # n = x.shape[-3]   # get spatial dim

        v = v.permute(0, 3, 1, 2)
        v = self.fc(v)

        v = v.permute(0, 2, 3, 1)
        # v = v.permute(0, 1, 2, 3)
        return x * (v + 1.)


class GridGmlpLayer(nn.Module):
    """Grid gMLP layer that performs global mixing of tokens."""
    def __init__(self, num_channels, grid_size, num_axis, factor=2, dropout_rate = 0.4,act_layer=nn.GELU):
        super().__init__()
        self.grid_size = grid_size

        self.factor = factor
        self.dropout_rate = dropout_rate
        self.num_channels = num_channels
        self.act_layer = act_layer()

        self.layernorm = nn.LayerNorm(num_channels//2)
        self.fc1 = nn.Linear(num_channels//2, num_channels)
        self.fc2 = nn.Linear( num_channels, num_channels//2)
        self.drop = nn.Dropout(dropout_rate)

        self.gridunit = GridGatingUnit(num_channels,num_axis)

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

    def forward(self,x, h, w):
        n, _, num_channels = x.shape
        gh, gw = self.grid_size
        fh, fw = h // gh, w // gw
        x = block_images_einops(x.reshape(n, h, w, num_channels), patch_size=(fh, fw))
        # gMLP1: Global (grid) mixing part, provides global grid communication.
        y = self.layernorm(x)
        y = self.fc1(y)
        y = self.act_layer(y)
        y = self.gridunit(y)
        y = self.fc2(y)
        y = self.drop(y)
        x = x + y
        x = unblock_images_einops(x, grid_size=(gh, gw), patch_size=(fh, fw))
        return x

class BlockGmlpLayer(nn.Module):
    def __init__(self, num_channels, block_size, num_axis, factor=2, dropout_rate=0.0,act_layer=nn.GELU):
        super().__init__()
        """Block gMLP layer that performs local mixing of tokens."""
        self.block_size = block_size
        self.use_bias = True
        self.factor = factor
        self.dropout_rate = dropout_rate
        self.act_layer = act_layer()
        self.layernorm = nn.LayerNorm(num_channels // 2)

        self.fc1 = nn.Linear(num_channels//2, num_channels)
        self.drop = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(num_channels , num_channels //2)
        self.blockgate = BlockGatingUnit(num_channels, num_axis)

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

    def forward(self,x, h, w):
        n,_,  num_channels = x.shape
        fh, fw = self.block_size
        gh, gw = h // fh, num_channels // fw
        x = block_images_einops(x.reshape(n, h, w, num_channels).permute(0, 3, 1, 2), patch_size=(fh, fw)) # batch_size, num_channels, hight, width
        # MLP2: Local (block) mixing part, provides within-block communication.
        y = x.permute(0, 3, 2, 1)
        y = self.layernorm(y)
        y = self.fc1(y)
        y = self.act_layer(y)
        y = self.blockgate(y) #y: batch_size, width, hight, channels
        y = self.fc2(y)
        y = self.drop(y)
        x = x + y.permute(0, 3, 2, 1)
        x = unblock_images_einops(x, grid_size=(gh, gw), patch_size=(fh, fw))
        return x.reshape(n, -1, num_channels)


class MultiAxisGLmlpBlock(nn.Module):
    def __init__(self, num_channels, grid_num_axis, block_num_axis, dropout_rate = 0.0,  grid_size=(16, 16), block_size=(8, 8), grid_gmlp_factor=2, block_gmlp_factor=2, input_proj_factor=2, use_bias=True,act_layer=nn.GELU):
        super().__init__()
        # grid_num_axis represents the number of channels.
        # block_num_axis represents the number of width.
        """The multi-axis gated MLP block."""
        self.block_size = block_size
        self.grid_size = grid_size
        self.block_gmlp_factor = block_gmlp_factor
        self.grid_gmlp_factor = grid_gmlp_factor
        self.input_proj_factor = input_proj_factor
        self.use_bias = use_bias
        self.dropout_rate = dropout_rate

        self.layernorm = nn.LayerNorm(num_channels)
        self.fc1 = nn.Linear(num_channels, num_channels * self.input_proj_factor)
        self.act = act_layer()
        self.gridgmlp = GridGmlpLayer(num_channels, grid_size, grid_num_axis, factor=self.grid_gmlp_factor, dropout_rate=self.dropout_rate)
        self.blockgmlp = BlockGmlpLayer(num_channels, block_size, block_num_axis, factor=self.block_gmlp_factor, dropout_rate=self.dropout_rate)
        self.fc2 = nn.Linear(num_channels, num_channels)
        self.drop = nn.Dropout(self.dropout_rate)

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

    def forward(self,x):
        shortcut = x
        n, num_channels, h, w = x.shape

        x = x.reshape(n, num_channels, -1).permute(0, 2, 1) # 8, 64, 32, 32

        x = self.layernorm(x) # 8, 64, 32, 32
        x = self.fc1(x)
        x = self.act(x)

        # u, v = np.split(x, 2, axis=1)
        u = torch.narrow(x, 2, 0, num_channels//2)  # 个函数是返回tensor的第dim维切片start: start+length的数, 针对例子，
        v = torch.narrow(x, 2, num_channels//2, num_channels//2) #4, 1024, 64

        # GridGMLPLayer
        u = self.gridgmlp(u, h, w)
        u = u.reshape(n,-1, num_channels//2)

        # BlockGMLPLayer
        v = self.blockgmlp(v, h, w)

        x = torch.cat([u,v], axis=2)

        x = self.fc2(x)
        x = self.drop(x)
        x = x.reshape(n, h, w, num_channels).permute(0, 3, 1, 2)
        x = x + shortcut
        return x


class shiftmlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0., shift_size=5):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.dim = in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.dwconv = DWConv(hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

        self.shift_size = shift_size
        self.pad = shift_size // 2

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

    def forward(self, x, H, W):
        # pdb.set_trace()
        B, N, C = x.shape  # x: # 8, 1024, 64

        xn = x.transpose(1, 2).view(B, C, H, W).contiguous()# 8, 64, 32, 32
        xn = F.pad(xn, (self.pad, self.pad, self.pad, self.pad), "constant", 0)
        xs = torch.chunk(xn, self.shift_size, 1)  # torch.chunk(tensor, chunk_num, dim) 将tensor按dim（行或列）分割成chunk_num个tensor块，返回的是一个元组。

        x1_shift = [torch.roll(x_c, shift, 2) for x_c, shift in zip(xs, range(-self.pad, self.pad + 1))]
        x1_cat = torch.cat(x1_shift, 1)
        x1_cat = torch.narrow(x1_cat, 2, self.pad, H)  # 个函数是返回tensor的第dim维切片start: start+length的数, 针对例子，
        x1_s = torch.narrow(x1_cat, 3, self.pad, W)
        x1_s = x1_s.reshape(B, C, H * W).contiguous() # 8, 64, 1024
        x_shift_r = x1_s.transpose(1, 2)

        x2_shift = [torch.roll(x_c, shift, 3) for x_c, shift in zip(xs, range(-self.pad, self.pad + 1))]
        x2_cat = torch.cat(x2_shift, 1)
        x2_cat = torch.narrow(x2_cat, 2, self.pad, H)  # 个函数是返回tensor的第dim维切片start: start+length的数, 针对例子，
        x2_s = torch.narrow(x2_cat, 3, self.pad, W)
        x2_s = x2_s.reshape(B, C, H * W).contiguous() # 8, 64, 1024
        x_shift_c = x2_s.transpose(1, 2)

        # x_shift = torch.bmm(x_shift_r,x_shift_c.permute(0,2,1))
        x_shift = x_shift_r * x_shift_c
        x = self.fc1(x_shift)
        x = self.dwconv(x, H, W) # 8, 1024, 64
        x = self.act(x)
        x = self.drop(x) # 8, 1024, 64

        return x

class shiftedBlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, sr_ratio=1):
        super().__init__()

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = shiftmlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        # self.resAtt = SE_Block(mlp_hidden_dim)
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

    def forward(self, x, H, W):
        x =  x+ self.drop_path(self.mlp(self.norm2(x), H, W))
        return x

class SE_Block(nn.Module):
    def __init__(self, c, r=16):
        super().__init__()
        self.squeeze = nn.AdaptiveAvgPool2d(1)
        self.excitation = nn.Sequential(
            nn.Linear(c, c // r, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(c // r, c, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        bs, c, _, _ = x.shape
        y = self.squeeze(x).view(bs, c)
        y = self.excitation(y).view(bs, c, 1, 1)
        x = x + x * y.expand_as(x)
        return y

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


class mlp(nn.Module):
    def __init__(self, dim=768):
        super(DWConv, self).__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim)

    def forward(self, x, H, W):
        B, N, C = x.shape
        x = x.transpose(1, 2).view(B, C, H, W)
        x = self.dwconv(x)
        x = x.flatten(2).transpose(1, 2)

        return x

class OverlapPatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """

    def __init__(self, img_size=224, patch_size=7, stride=4, in_chans=3, embed_dim=768):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)

        self.img_size = img_size
        self.patch_size = patch_size
        self.H, self.W = img_size[0] // patch_size[0], img_size[1] // patch_size[1]
        self.num_patches = self.H * self.W
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=stride,
                              padding=(patch_size[0] // 2, patch_size[1] // 2))
        self.norm = nn.LayerNorm(embed_dim)

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
        x = self.proj(x) # 4 32 64 64, 4
        _, _, H, W = x.shape # 4, 128, 32, 32
        x = x.flatten(2).transpose(1, 2) # 4, 1024, 64
        x = self.norm(x)

        return x, H, W


class MLPGLUNet(nn.Module):

    ## Conv 3 + MLP 2 + shifted MLP

    def __init__(self, num_classes, input_channels=3, deep_supervision=False, img_size=512, patch_size=16, in_chans=3,
                 embed_dims=[32, 64, 128, 512],
                 num_heads=[1, 2, 4, 8], mlp_ratios=[4, 4, 4, 4], qkv_bias=False, qk_scale=None, drop_rate=0.,
                 attn_drop_rate=0., drop_path_rate=0., norm_layer=nn.LayerNorm,
                 depths=[1, 1, 1], sr_ratios=[8, 4, 2, 1], **kwargs):
        super().__init__()

        self.encoder1 = nn.Conv2d(3, 8, 3, stride=1, padding=1)
        self.encoder2 = nn.Conv2d(8, 16, 3, stride=1, padding=1)
        self.encoder3 = nn.Conv2d(16, 32, 3, stride=1, padding=1)

        self.ebn1 = nn.BatchNorm2d(8)
        self.ebn2 = nn.BatchNorm2d(16)
        self.ebn3 = nn.BatchNorm2d(32)

        self.norm3 = norm_layer(embed_dims[1])
        self.norm4 = norm_layer(embed_dims[2])

        self.dnorm3 = norm_layer(64)
        self.dnorm4 = norm_layer(32)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]

        # self.block1 = nn.ModuleList([shiftedBlock(
        #     dim=embed_dims[1], num_heads=num_heads[0], mlp_ratio=1, qkv_bias=qkv_bias, qk_scale=qk_scale,
        #     drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[0], norm_layer=norm_layer,
        #     sr_ratio=sr_ratios[0])])

        self.block1 = shiftedBlock(dim=embed_dims[1], num_heads=num_heads[0], mlp_ratio=1, qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[0], norm_layer=norm_layer, sr_ratio=sr_ratios[0])

        self.block2 = shiftedBlock(dim=embed_dims[2], num_heads=num_heads[0], mlp_ratio=1, qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[1], norm_layer=norm_layer,
            sr_ratio=sr_ratios[0])

        self.dblock1 = shiftedBlock(dim=embed_dims[1], num_heads=num_heads[0], mlp_ratio=1, qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[0], norm_layer=norm_layer,
            sr_ratio=sr_ratios[0])

        self.dblock2 = shiftedBlock(dim=embed_dims[0], num_heads=num_heads[0], mlp_ratio=1, qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[1], norm_layer=norm_layer,
            sr_ratio=sr_ratios[0])

        self.patch_embed3 = OverlapPatchEmbed(img_size=img_size // 4, patch_size=3, stride=2, in_chans=embed_dims[0],
                                              embed_dim=embed_dims[1])
        self.patch_embed4 = OverlapPatchEmbed(img_size=img_size // 8, patch_size=3, stride=2, in_chans=embed_dims[1],
                                              embed_dim=embed_dims[2])

        self.multaxis = MultiAxisGLmlpBlock(32, 16, 64, dropout_rate=0.5)

        self.decoder1 = nn.Conv2d(128, 64, 3, stride=1, padding=1)
        self.decoder2 = nn.Conv2d(64, 32, 3, stride=1, padding=1)
        self.decoder3 = nn.Conv2d(32, 16, 3, stride=1, padding=1)
        self.decoder4 = nn.Conv2d(16, 8, 3, stride=1, padding=1)
        self.decoder5 = nn.Conv2d(8, 8, 3, stride=1, padding=1)

        self.dbn1 = nn.BatchNorm2d(64)
        self.dbn2 = nn.BatchNorm2d(32)
        self.dbn3 = nn.BatchNorm2d(16)
        self.dbn4 = nn.BatchNorm2d(8)

        self.final = nn.Conv2d(8, num_classes, kernel_size=1)

        self.soft = nn.Softmax(dim=1)

    def forward(self, x):

        B = x.shape[0]
        ### Encoder
        ### Conv Stage

        ### Stage 1
        out = F.relu(F.max_pool2d(self.ebn1(self.encoder1(x)), 2, 2))
        t1 = out
        ### Stage 2
        out = F.relu(F.max_pool2d(self.ebn2(self.encoder2(out)), 2, 2))
        t2 = out
        ### Stage 3
        out = F.relu(F.max_pool2d(self.ebn3(self.encoder3(out)), 2, 2))
        t3 = out

        ### Tokenized MLP Stage
        ### Stage 4

        out = self.multaxis(out)

        out, H, W = self.patch_embed3(out)
        out = self.block1(out, H, W)
        out = self.norm3(out)
        out = out.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        t4 = out


        ### Bottleneck
        out, H, W = self.patch_embed4(out)
        out = self.block2(out, H, W)
        out = self.norm4(out)
        out = out.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()

        ### Stage 4
        out = F.relu(F.interpolate(self.dbn1(self.decoder1(out)), scale_factor=(2, 2), mode='bilinear'))
        out = torch.add(out, t4)
        _, _, H, W = out.shape
        out = out.flatten(2).transpose(1, 2)
        out = self.dblock1(out, H, W)

        ### Stage 3
        out = self.dnorm3(out)
        out = out.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        out = F.relu(F.interpolate(self.dbn2(self.decoder2(out)), scale_factor=(2, 2), mode='bilinear'))
        out = torch.add(out, t3)
        _, _, H, W = out.shape
        out = out.flatten(2).transpose(1, 2)

        out = self.dblock2(out, H, W)
        out = self.dnorm4(out)
        out = out.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()

        out = F.relu(F.interpolate(self.dbn3(self.decoder3(out)), scale_factor=(2, 2), mode='bilinear'))
        out = torch.add(out, t2)
        out = F.relu(F.interpolate(self.dbn4(self.decoder4(out)), scale_factor=(2, 2), mode='bilinear'))
        out = torch.add(out, t1)
        out = F.relu(F.interpolate(self.decoder5(out), scale_factor=(2, 2), mode='bilinear'))

        return self.final(out)

# EOF
