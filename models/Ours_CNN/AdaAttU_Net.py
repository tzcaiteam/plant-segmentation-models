from __future__ import print_function, division
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
import torch
from functools import reduce
# from nnunet.network_architecture.neural_network import SegmentationNetwork





class Multiscaleconv_block_v0(nn.Module):
    """
    Convolution Block
    """

    def __init__(self, in_ch, out_ch):
        super(Multiscaleconv_block_v0, self).__init__()

        # self.conv1 = nn.Sequential(
        #     nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=1, padding=0, bias=True),
        #     nn.BatchNorm2d(out_ch),
        #     nn.ReLU(inplace=True))
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True))
        self.conv5 = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=5, stride=1, padding=2, bias=True),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True))
    def forward(self, x):
        x = self.conv3(x)+ self.conv5(x)   #+self.conv1(x)
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
            BN_Conv2d(out_ch//2, out_ch, 3, 1, 1, bias=False),
            BN_Conv2d(out_ch, out_ch, 1, 1, 0, bias=False)
        )
    def forward(self, x):
        x = self.conv3(x)+ self.conv5(x)+self.conv1(x)
        return x

class inception_block(nn.Module):
    """
    Convolution Block
    """
    def __init__(self, in_ch, out_ch):
        super(inception_block, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(out_ch))
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_ch))
        self.conv1_3 = nn.Sequential(
            nn.Conv2d(in_ch, out_ch//2, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(out_ch//2),
            nn.Conv2d(out_ch//2, out_ch, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_ch) )
        self.avgpool = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.AvgPool2d(kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_ch))
        self.relu = nn.ReLU(inplace=True)
    def forward(self, x):
        conv3 = self.conv3(x)
        conv13=self.conv1_3(x)
        conv1 = self.conv1(x)
        pool = self.avgpool(x)
        x = self.relu(self.conv3(x)+ self.conv1_3(x)+self.conv1(x)+self.avgpool(x))
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

class Resconv_block(nn.Module):
    """
    Convolution Block
    """

    def __init__(self, in_ch, out_ch):
        super(Resconv_block, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True))

    def forward(self, x):
        x = self.conv(x)

        return x

class MultiscaleFuse(nn.Module):
    """
    Convolution Block
    """

    def __init__(self):
        super(MultiscaleFuse, self).__init__()

        self.up =  nn.Upsample(scale_factor=2)
        self.down = nn.MaxPool2d(kernel_size=2, stride=2)


    def forward(self, x):
        x1 = self.down(x)
        x  = self.up(x)

        x1 = self.up(x1)
        x1 = self.up(x1)
        return x + x1


class BoundaryAware_UNet(nn.Module):
    """
    UNet - Unet with down-upsampling operation
    Paper : https://arxiv.org/abs/1505.04597
    """

    def __init__(self, n_channels=1, n_classes=3, deep_supervision = False):
        super(BoundaryAware_UNet, self).__init__()

        n1 = 1
        self.deep_supervision = deep_supervision
        self.in_chanel = n_channels
        self.out_chanel = n_classes

        filters = [16, 32, 64, 128, 256, 512]

        self.up =  nn.Upsample(scale_factor=2)
        self.down = nn.MaxPool2d(kernel_size=2, stride=2)

        self.MConv0 = Multiscaleconv_small(n_channels, filters[0])
        self.MConv1 = Multiscaleconv_small(filters[0], filters[1])
        self.MConv2 = Multiscaleconv_small(filters[1], filters[2])
        self.MConv3 = Multiscaleconv_small(filters[2], filters[3])
        self.MConv4 = Multiscaleconv_small(filters[3], filters[4])

        self.inconv0 = conv_block(n_channels, filters[0])
        self.inconv1 = Resconv_block(filters[0], filters[0])
        self.inconv2 = Resconv_block(filters[1], filters[1])
        self.inconv3 = Resconv_block(filters[2], filters[2])
        self.inconv4 = Resconv_block(filters[3], filters[3])
        # self.inconv5 = Resconv_block(filters[4], filters[4])

        self.Maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool4 = nn.MaxPool2d(kernel_size=2, stride=2)
        # self.Maxpool5 = nn.MaxPool2d(kernel_size=2, stride=2)
        #
        self.mup = MultiscaleFuse()

        self.Up5 = up_conv(filters[4], filters[3], 2)
        self.Up_conv5 = conv_block(filters[4], filters[3])

        self.Up4 = up_conv(filters[3], filters[2], 2)
        self.Up_conv4 = conv_block(filters[3], filters[2])

        self.Up3 = up_conv(filters[2], filters[1], 2)
        self.Up_conv3 = conv_block(filters[2], filters[1])

        self.Up2 = up_conv(filters[1], filters[0], 2)
        self.Up_conv2 = conv_block(filters[1], filters[0])

        self.Conv = nn.Conv2d(filters[0], n_classes, kernel_size=1, stride=1, padding=0)
        self.do_ds = True

        if self.deep_supervision:
            self.final1 = nn.Conv2d(filters[1], n_classes, kernel_size=1, stride=1, padding=0)
            self.final2 = nn.Conv2d(filters[2], n_classes, kernel_size=1, stride=1, padding=0)
            self.final3 = nn.Conv2d(filters[3], n_classes, kernel_size=1, stride=1, padding=0)
            self.final4 = nn.Conv2d(filters[3], n_classes, kernel_size=1, stride=1, padding=0)

        # self.active = torch.nn.Sigmoid()

    def forward(self, x):

        # e1 = self.inconv0(x)   #16*512*512
        e1 = self.MConv0(x)
        # rese1 = self.resconv1(e1)  #32*512*512
        e2 = self.Maxpool1(e1)    #16*256*256
        # e2 = self.MConv1(e2)
        m2 = self.mup(e2)   #16*512*512
        m2 = m2 + e1
        m2 =self.inconv1(m2)#16*512*512

        # e2 = self.dconv1(e2)#32*256*256
        e2 = self.MConv1(e2)
        # rese2 = self.resconv2(e2)
        e3 = self.Maxpool2(e2)
        # e3 = self.MConv2(e3)
        m3 = self.mup(e3)
        m3 = m3 + e2
        m3 = self.inconv2(m3)

        # e3 = self.dconv2(e3)#64*128*128
        e3 = self.MConv2(e3)
        # rese3 = self.resconv3(e3)
        e4 = self.Maxpool3(e3)   #64*64*64
        # e4 = self.MConv3(e4)
        m4 = self.mup(e4)
        m4 = m4 + e3
        m4 = self.inconv3(m4)

        # e4 = self.dconv3(e4)#128*64*64
        e4 = self.MConv3(e4)
        # rese4 = self.resconv4(e4)
        e5 = self.Maxpool4(e4)
        # e5 = self.MConv4(e5)
        m5 = self.mup(e5)
        m5 = m5 +e4
        m5 = self.inconv4(m5)

        e6 = self.MConv4(e5)  # 256*32*32

        d5 = self.Up5(e6)
        d5 = torch.cat((m5, d5), dim=1)
        d5 = self.Up_conv5(d5)

        d4 = self.Up4(d5)
        d4 = torch.cat((m4, d4), dim=1)
        d4 = self.Up_conv4(d4)

        d3 = self.Up3(d4)
        d3 = torch.cat((m3, d3), dim=1)
        d3 = self.Up_conv3(d3)

        d2 = self.Up2(d3)
        d2 = torch.cat((m2, d2), dim=1)
        d2 = self.Up_conv2(d2)


        if self.deep_supervision:
            out0 = self.Conv(d2)
            out1 = F.interpolate(self.final1(d3), x.shape[2:], mode='bilinear')
            out2 = F.interpolate(self.final2(d4), x.shape[2:], mode='bilinear')
            out3 = F.interpolate(self.final3(d5), x.shape[2:], mode='bilinear')
            out4 = F.interpolate(self.final4(e5), x.shape[2:], mode='bilinear')
            return [out0, out1, out2, out3, out4]

        else:
            out = self.Conv(d2)
            return out

        # out = self.Conv(d2)
        #
        # return out

class selective_layer(nn.Module):
    """
    Convolution Block
    """
    def __init__(self, in_ch, out_ch):
        super(selective_layer, self).__init__()
        self.top_conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, dilation=2, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True))

        mid_features = out_ch // 2
        self.up = nn.Upsample(scale_factor=2)

        self.down_conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True))
        self.globalpool = nn.AdaptiveAvgPool2d(1)

        self.fc1 = nn.Sequential(nn.Conv2d(out_ch, mid_features, 1, bias=False),
                                 nn.BatchNorm2d(mid_features),
                                 nn.ReLU(inplace=True))

        self.fc2 = nn.Conv2d(mid_features, out_ch, 1, 1, bias=False)

        # self.fc1 = nn.Linear(out_ch,int(out_ch * compression))
        # self.bn = nn.BatchNorm2d(int(out_ch * compression))
        # self.relu = nn.ReLU(inplace=True)
        # self.fc2 = nn.Linear(int(out_ch * compression), out_ch)

        self.sigmoid = nn.Sigmoid()

   # l_res low resolution features, h_res: high resolution features

    def forward(self,l_res, h_res):

        up = self.up(h_res)

        x  = up +l_res

        x1 = self.top_conv(x)
        x2 = self.down_conv(x)
        x3 = x1 + x2
        x3 = self.globalpool(x3)
        x3 = self.fc1(x3)

        x3 = self.relu(self.bn(x3))
        x3 = self.fc2(x3)

        x3p = self.sigmoid(x3)
        x3m = map(lambda x: 1 - x, x3p)


        x4 = torch.bmm(up, x3p)
        x5 = torch.bmm(l_res, x3m)
        out = x4 + x5

        return out

class SK_Att(nn.Module):
    def __init__(self,in_channels,out_channels,stride=1, r=16, L=32):
        '''
        :param in_channels:  输入通道维度
        :param out_channels: 输出通道维度   原论文中 输入输出通道维度相同
        :param stride:  步长，默认为1
        :param M:  分支数
        :param r: 特征Z的长度，计算其维度d 时所需的比率（论文中 特征S->Z 是降维，故需要规定 降维的下界）
        :param L:  论文中规定特征Z的下界，默认为32
        '''
        super(SK_Att,self).__init__()
        d=max(in_channels//r,L)   # 计算向量Z 的长度d

        self.out_channels=out_channels
        self.conv1 = nn.Sequential(nn.Conv2d(in_channels,out_channels,3,stride,padding=1,dilation=1,groups=32,bias=False),
                                           nn.BatchNorm2d(out_channels),
                                           nn.ReLU(inplace=True))
        self.conv2 = nn.Sequential(nn.Conv2d(in_channels,out_channels,3,stride,padding=2,dilation=2,groups=32,bias=False),
                                           nn.BatchNorm2d(out_channels),
                                           nn.ReLU(inplace=True))

        self.global_pool=nn.AdaptiveAvgPool2d(1) # 自适应pool到指定维度    这里指定为1，实现 GAP
        self.fc1=nn.Sequential(nn.Conv2d(out_channels,d,1,bias=False),
                               nn.BatchNorm2d(d),
                               nn.ReLU(inplace=True))   # 降维
        self.fc2=nn.Conv2d(d,out_channels*2,1,1,bias=False)  # 升维
        self.softmax=nn.Softmax(dim=1) # 指定dim=1  使得两个全连接层对应位置进行softmax,保证 对应位置a+b+..=1
    def forward(self, input):
        batch_size=input.size(0)
        output=[]
        #the part of split
        output.append(self.conv1(input))
        output.append(self.conv2(input))
        #the part of fusion
        U=reduce(lambda x,y:x+y,output) # 逐元素相加生成 混合特征U
        s=self.global_pool(U)
        z=self.fc1(s)  # S->Z降维
        a_b=self.fc2(z) # Z->a，b 升维  论文使用conv 1x1表示全连接。结果中前一半通道值为a,后一半为b
        a_b=a_b.reshape(batch_size,2,self.out_channels,-1) #调整形状，变为 两个全连接层的值
        a_b=self.softmax(a_b) # 使得两个全连接层对应位置进行softmax
        #the part of selection
        a_b=list(a_b.chunk(2,dim=1))#split to a and b   chunk为pytorch方法，将tensor按照指定维度切分成 几个tensor块
        a_b=list(map(lambda x:x.reshape(batch_size,self.out_channels,1,1),a_b)) # 将所有分块调整形状，即扩展两维
        V=list(map(lambda x,y:x*y,output,a_b)) # 权重与对应  不同卷积核输出的U 逐元素相乘
        V=reduce(lambda x,y:x+y,V) # 两个加权后的特征逐元素相加
        return V

class SKU_Net(nn.Module):
    """
    UNet - Unet with down-upsampling operation
    Paper : https://arxiv.org/abs/1505.04597
    """

    def __init__(self, n_channels=1, n_classes=3, deep_supervision = False):
        super(SKU_Net, self).__init__()

        n1 = 1

        self.in_chanel = n_channels
        self.out_chanel = n_classes
        self.deep_supervision = deep_supervision

        filters = [32, 64, 128, 256, 512]

        self.up =  nn.Upsample(scale_factor=2)
        self.down = nn.MaxPool2d(kernel_size=2, stride=2)

        self.MConv0 = Multiscaleconv_small(n_channels, filters[0])
        self.MConv1 = Multiscaleconv_small(filters[0], filters[1])
        self.MConv2 = Multiscaleconv_small(filters[1], filters[2])
        self.MConv3 = Multiscaleconv_small(filters[2], filters[3])
        self.MConv4 = Multiscaleconv_small(filters[3], filters[4])

        self.inconv0 = conv_block(n_channels, filters[0])
        self.inconv1 = Resconv_block(filters[0], filters[0])
        self.inconv2 = Resconv_block(filters[1], filters[1])
        self.inconv3 = Resconv_block(filters[2], filters[2])
        self.inconv4 = Resconv_block(filters[3], filters[3])
        # self.inconv5 = Resconv_block(filters[4], filters[4])

        self.Maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool4 = nn.MaxPool2d(kernel_size=2, stride=2)
        # self.Maxpool5 = nn.MaxPool2d(kernel_size=2, stride=2)
        #
        self.mup = MultiscaleFuse()

        self.att45 = SK_Att(filters[4], filters[4])
        self.att34 = SK_Att(filters[3], filters[3])
        self.att23 = SK_Att(filters[2], filters[2])
        self.att12 = SK_Att(filters[1], filters[1])

        self.Up5 = up_conv(filters[4], filters[3], 2)
        self.Up_conv5 = conv_block(filters[4], filters[3])

        self.Up4 = up_conv(filters[3], filters[2], 2)
        self.Up_conv4 = conv_block(filters[3] + filters[2], filters[2])

        self.Up3 = up_conv(filters[2], filters[1], 2)
        self.Up_conv3 = conv_block(filters[2] + filters[1], filters[1])

        self.Up2 = up_conv(filters[1], filters[0], 2)
        self.Up_conv2 = conv_block(filters[1]+ filters[0], filters[0])

        self.Conv = nn.Conv2d(filters[0], n_classes, kernel_size=1, stride=1, padding=0)
        self.do_ds = True

        self.up = nn.Upsample(scale_factor=2, mode='bilinear')

        if self.deep_supervision:
            self.final1 = nn.Conv2d(filters[1], n_classes, kernel_size=1, stride=1, padding=0)
            self.final2 = nn.Conv2d(filters[2], n_classes, kernel_size=1, stride=1, padding=0)
            self.final3 = nn.Conv2d(filters[3], n_classes, kernel_size=1, stride=1, padding=0)
            self.final4 = nn.Conv2d(filters[3], n_classes, kernel_size=1, stride=1, padding=0)


        # self.active = torch.nn.Sigmoid()

    def forward(self, x):

        # e1 = self.inconv0(x)   #16*512*512
        e1 = self.MConv0(x)
        # rese1 = self.resconv1(e1)  #32*512*512
        e2 = self.Maxpool1(e1)    #16*256*256
        # e2 = self.MConv1(e2)
        m2 = self.mup(e2)   #16*512*512
        m2 = m2 + e1
        m2 = self.inconv1(m2)#16*512*512

        # e2 = self.dconv1(e2)#32*256*256
        e2 = self.MConv1(e2)
        # rese2 = self.resconv2(e2)
        e3 = self.Maxpool2(e2)
        # e3 = self.MConv2(e3)
        m3 = self.mup(e3)
        m3 = m3 + e2
        m3 = self.inconv2(m3)

        # e3 = self.dconv2(e3)   #64*128*128
        e3 = self.MConv2(e3)
        # rese3 = self.resconv3(e3)
        e4 = self.Maxpool3(e3)   #64*64*64
        # e4 = self.MConv3(e4)
        m4 = self.mup(e4)
        m4 = m4 + e3
        m4 = self.inconv3(m4)

        # e4 = self.dconv3(e4)   #128*64*64
        e4 = self.MConv3(e4)
        # rese4 = self.resconv4(e4)
        e5 = self.Maxpool4(e4)
        # e5 = self.MConv4(e5)
        m5 = self.mup(e5)
        m5 = m5 +e4
        m5 = self.inconv4(m5)
        e6 = self.MConv4(e5)     # 256*32*32

        # d5 = self.Up5(e6)
        d5 = self.att45(e6)
        d5 = self.up(d5)
        d5 = self.Up_conv5(d5)

        d5 = torch.cat((m5, d5), dim=1)
        d5 = self.Up_conv5(d5)

        # d4 = self.Up4(d5)
        d4 = self.att34(d5)
        d4 = self.up(d4)
        d4 = torch.cat((m4, d4), dim=1)
        d4 = self.Up_conv4(d4)

        # d3 = self.Up3(d4)
        d3 = self.att23(d4)
        d3 = self.up(d3)
        d3 = torch.cat((m3, d3), dim=1)
        d3 = self.Up_conv3(d3)

        # d2 = self.Up2(d3)
        d2 = self.att12(d3)
        d2 = self.up(d2)
        d2 = torch.cat((m2, d2), dim=1)
        d2 = self.Up_conv2(d2)

        if self.deep_supervision:
            out0 = self.Conv(d2)
            out1 = F.interpolate(self.final1(d3), x.shape[2:], mode='bilinear')
            out2 = F.interpolate(self.final2(d4), x.shape[2:], mode='bilinear')
            out3 = F.interpolate(self.final3(d5), x.shape[2:], mode='bilinear')
            out4 = F.interpolate(self.final4(e5), x.shape[2:], mode='bilinear')
            return [out0, out1, out2, out3, out4]
        else:
            out = self.Conv(d2)
            return out

class Hybrid_Att(nn.Module):
    def __init__(self,in_channels,out_channels,stride=1, r=16, L=32):
        '''
        :param in_channels:  输入通道维度
        :param out_channels: 输出通道维度   原论文中 输入输出通道维度相同
        :param stride:  步长，默认为1
        :param M:  分支数
        :param r: 特征Z的长度，计算其维度d 时所需的比率（论文中 特征S->Z 是降维，故需要规定 降维的下界）
        :param L:  论文中规定特征Z的下界，默认为32
        '''
        super(Hybrid_Att,self).__init__()
        d=max(in_channels//r,L)   # 计算向量Z 的长度d

        self.out_channels=out_channels
        self.conv1 = nn.Sequential(nn.Conv2d(in_channels,out_channels,3,stride,padding=1,dilation=1,groups=32,bias=False),
                                           nn.BatchNorm2d(out_channels),
                                           nn.ReLU(inplace=True))
        self.conv2 = nn.Sequential(nn.Conv2d(in_channels,out_channels,3,stride,padding=2,dilation=2,groups=32,bias=False),
                                           nn.BatchNorm2d(out_channels),
                                           nn.ReLU(inplace=True))

        self.global_pool=nn.AdaptiveAvgPool2d(1) # 自适应pool到指定维度    这里指定为1，实现 GAP
        self.fc1=nn.Sequential(nn.Conv2d(out_channels,d,1,bias=False),
                               nn.BatchNorm2d(d),
                               nn.ReLU(inplace=True))   # 降维
        self.fc2=nn.Conv2d(d,out_channels*2,1,1,bias=False)  # 升维
        self.softmax=nn.Softmax(dim=1) # 指定dim=1  使得两个全连接层对应位置进行softmax,保证 对应位置a+b+..=1

        self.g_conv1 = nn.Sequential(nn.Conv2d(in_channels,out_channels,3,stride,padding=2,dilation=2,groups=32,bias=False),
                                           nn.BatchNorm2d(out_channels),
                                           nn.ReLU(inplace=True))

    def forward(self, input, low_res):
        batch_size=input.size(0)
        output=[]
        #the part of split
        output.append(self.conv1(input))
        output.append(self.conv2(input))
        #the part of fusion
        U=reduce(lambda x,y:x+y,output) # 逐元素相加生成 混合特征U
        s=self.global_pool(U)
        z=self.fc1(s)  # S->Z降维
        a_b=self.fc2(z) # Z->a，b 升维  论文使用conv 1x1表示全连接。结果中前一半通道值为a,后一半为b
        a_b=a_b.reshape(batch_size,2,self.out_channels,-1) #调整形状，变为 两个全连接层的值
        a_b=self.softmax(a_b) # 使得两个全连接层对应位置进行softmax
        #the part of selection
        a_b=list(a_b.chunk(2,dim=1))#split to a and b   chunk为pytorch方法，将tensor按照指定维度切分成 几个tensor块
        a_b=list(map(lambda x:x.reshape(batch_size,self.out_channels,1,1),a_b)) # 将所有分块调整形状，即扩展两维
        V=list(map(lambda x,y:x*y,output,a_b)) # 权重与对应  不同卷积核输出的U 逐元素相乘
        V=reduce(lambda x,y:x+y,V) # 两个加权后的特征逐元素相加

        # low_res + V
        return V


class SHAU_Net(nn.Module):
    """
    UNet - Unet with down-upsampling operation
    Paper : https://arxiv.org/abs/1505.04597
    """

    def __init__(self, n_channels=1, n_classes=3, deep_supervision = False):
        super(SHAU_Net, self).__init__()

        n1 = 1

        self.in_chanel = n_channels
        self.out_chanel = n_classes
        self.deep_supervision = deep_supervision

        filters = [32, 64, 128, 256, 512]

        self.up =  nn.Upsample(scale_factor=2)
        self.down = nn.MaxPool2d(kernel_size=2, stride=2)

        self.MConv0 = Multiscaleconv_small(n_channels, filters[0])
        self.MConv1 = Multiscaleconv_small(filters[0], filters[1])
        self.MConv2 = Multiscaleconv_small(filters[1], filters[2])
        self.MConv3 = Multiscaleconv_small(filters[2], filters[3])
        self.MConv4 = Multiscaleconv_small(filters[3], filters[4])

        self.inconv0 = conv_block(n_channels, filters[0])
        self.inconv1 = Resconv_block(filters[0], filters[0])
        self.inconv2 = Resconv_block(filters[1], filters[1])
        self.inconv3 = Resconv_block(filters[2], filters[2])
        self.inconv4 = Resconv_block(filters[3], filters[3])
        # self.inconv5 = Resconv_block(filters[4], filters[4])

        self.Maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool4 = nn.MaxPool2d(kernel_size=2, stride=2)
        # self.Maxpool5 = nn.MaxPool2d(kernel_size=2, stride=2)
        #
        self.mup = MultiscaleFuse()

        self.att45 = Hybrid_Att(filters[4], filters[4])
        self.att34 = Hybrid_Att(filters[3], filters[3])
        self.att23 = Hybrid_Att(filters[2], filters[2])
        self.att12 = Hybrid_Att(filters[1], filters[1])

        self.Up5 = up_conv(filters[4], filters[3], 2)
        self.Up_conv5 = conv_block(filters[4], filters[3])

        self.Up4 = up_conv(filters[3], filters[2], 2)
        self.Up_conv4 = conv_block(filters[3] + filters[2], filters[2])

        self.Up3 = up_conv(filters[2], filters[1], 2)
        self.Up_conv3 = conv_block(filters[2] + filters[1], filters[1])

        self.Up2 = up_conv(filters[1], filters[0], 2)
        self.Up_conv2 = conv_block(filters[1]+ filters[0], filters[0])

        self.Conv = nn.Conv2d(filters[0], n_classes, kernel_size=1, stride=1, padding=0)
        self.do_ds = True

        if self.deep_supervision:
            self.final1 = nn.Conv2d(filters[1], n_classes, kernel_size=1, stride=1, padding=0)
            self.final2 = nn.Conv2d(filters[2], n_classes, kernel_size=1, stride=1, padding=0)
            self.final3 = nn.Conv2d(filters[3], n_classes, kernel_size=1, stride=1, padding=0)
            self.final4 = nn.Conv2d(filters[4], n_classes, kernel_size=1, stride=1, padding=0)

        self.up = nn.Upsample(scale_factor=2, mode='bilinear')

        # self.active = torch.nn.Sigmoid()

    def forward(self, x):

        # e1 = self.inconv0(x)   #16*512*512
        e1 = self.MConv0(x)
        # rese1 = self.resconv1(e1)  #32*512*512
        e2 = self.Maxpool1(e1)    #16*256*256
        # e2 = self.MConv1(e2)
        m2 = self.mup(e2)   #16*512*512
        m2 = m2 + e1
        m2 = self.inconv1(m2)#16*512*512

        # e2 = self.dconv1(e2)#32*256*256
        e2 = self.MConv1(e2)
        # rese2 = self.resconv2(e2)
        e3 = self.Maxpool2(e2)
        # e3 = self.MConv2(e3)
        m3 = self.mup(e3)
        m3 = m3 + e2
        m3 = self.inconv2(m3)

        # e3 = self.dconv2(e3)   #64*128*128
        e3 = self.MConv2(e3)
        # rese3 = self.resconv3(e3)
        e4 = self.Maxpool3(e3)   #64*64*64
        # e4 = self.MConv3(e4)
        m4 = self.mup(e4)
        m4 = m4 + e3
        m4 = self.inconv3(m4)

        # e4 = self.dconv3(e4)   #128*64*64
        e4 = self.MConv3(e4)
        # rese4 = self.resconv4(e4)
        e5 = self.Maxpool4(e4)
        # e5 = self.MConv4(e5)
        m5 = self.mup(e5)
        m5 = m5 +e4
        m5 = self.inconv4(m5)
        e6 = self.MConv4(e5)     # 256*32*32

        # d5 = self.Up5(e6)
        d5 = self.att45(e6, m5)
        d5 = self.up(d5)
        d5 = self.Up_conv5(d5)

        d5 = torch.cat((m5, d5), dim=1)
        d5 = self.Up_conv5(d5)

        # d4 = self.Up4(d5)
        d4 = self.att34(d5)
        d4 = self.up(d4)
        d4 = torch.cat((m4, d4), dim=1)
        d4 = self.Up_conv4(d4)

        # d3 = self.Up3(d4)
        d3 = self.att23(d4)
        d3 = self.up(d3)
        d3 = torch.cat((m3, d3), dim=1)
        d3 = self.Up_conv3(d3)

        # d2 = self.Up2(d3)
        d2 = self.att12(d3)
        d2 = self.up(d2)
        d2 = torch.cat((m2, d2), dim=1)
        d2 = self.Up_conv2(d2)
        if self.deep_supervision:
            out0 = self.Conv(d2)
            out1 = F.interpolate(self.final1(d3), x.shape[2:], mode='bilinear')
            out2 = F.interpolate(self.final2(d4), x.shape[2:], mode='bilinear')
            out3 = F.interpolate(self.final3(d5), x.shape[2:], mode='bilinear')
            out4 = F.interpolate(self.final4(e5), x.shape[2:], mode='bilinear')
            return [out0, out1, out2, out3, out4]
        else:
            out = self.Conv(d2)
            return out

        # out = self.Conv(d2)
        #
        # return out
