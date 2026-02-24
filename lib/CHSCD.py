import torch
import torch.nn as nn
import torchvision.models as models
from torch.nn import functional as F
import warnings
from MDRI import MDRI
from DCA import ChannelAttentionAKConv
warnings.filterwarnings('ignore')


def initialize_weights(*models):
    for model in models:
        for module in model.modules():
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight)
                if module.bias is not None:
                    module.bias.data.zero_()
            elif isinstance(module, nn.BatchNorm2d):
                module.weight.data.fill_(1)
                module.bias.data.zero_()


class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.resnet = models.resnet34(pretrained=True)
        for n, m in self.resnet.layer3.named_modules():
            if 'conv1' in n or 'downsample.0' in n:
                m.stride = (1, 1)
        for n, m in self.resnet.layer4.named_modules():
            if 'conv1' in n or 'downsample.0' in n:
                m.stride = (1, 1)

        self.conv = nn.Sequential(nn.Conv2d(512 + 256 + 128, 128, 1), nn.BatchNorm2d(128), nn.ReLU())
        initialize_weights(self.conv)

        self.DCA3 = ChannelAttentionAKConv(inc=128, outc=256, num_param=3)  # 条状物体，道路。河流
        self.DCA2 = ChannelAttentionAKConv(inc=64, outc=128, num_param=3)
        self.DCA4 = ChannelAttentionAKConv(inc=256, outc=512, num_param=3)
        self.C1 = CBAM(64)
        self.downsample = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),  # 或者用 AvgPool2d
        )

    def forward(self, x):
        x0 = self.resnet.relu(self.resnet.bn1(self.resnet.conv1(x)))#(64 256 256)
        xm = self.resnet.maxpool(x0)
        x1 = self.resnet.layer1(xm) # (64 128 128)

        x2_1 = self.resnet.layer2(x1) #（128 64 64）
        x2_2 = self.DCA2(x1)
        # x2 = torch.cat((x2_1, x2_2), 1)
        x2_2 = self.downsample(x2_2)
        x2 = x2_1 + x2_2

        x3_1 = self.resnet.layer3(x2) #（256 64 64）
        x3_2 = self.DCA3(x2)
        # x3_2 = self.downsample(x3_2)
        x3 = x3_1 + x3_2

        x4_1 = self.resnet.layer4(x3)#(512 64 64)
        x4_2 = self.DCA4(x3)
        # x4_2 = self.downsample(x4_2)
        x4 = x4_1 + x4_2

        x4 = torch.cat([x2, x3, x4], 1)
        x4 = self.conv(x4)
        x1 = self.C1(x1)

        return x0, x1, x4


class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ConvBlock(nn.Module):
    def __init__(self, channels):
        super(ConvBlock, self).__init__()
        self.conv = nn.Sequential(nn.Conv2d(channels, channels, 3, 1, 2, dilation=2, bias=False),
                                  nn.BatchNorm2d(channels), nn.ReLU(inplace=True))

    def forward(self, x):
        out = self.conv(x)
        return out


class DecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DecoderBlock, self).__init__()
        self.block = self._make_layers_(in_channels, out_channels)
        self.cb = ConvBlock(out_channels)

    def _make_layers_(self, in_channels, out_channels, blocks=2, stride=1):
        downsample = None
        if stride != 1 or in_channels != out_channels:
            downsample = nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                                       nn.BatchNorm2d(out_channels))
        layers = [ResBlock(in_channels, out_channels, stride, downsample)]
        for i in range(1, blocks):
            layers.append(ResBlock(out_channels, out_channels))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.block(x)
        x = self.cb(x)
        return x


class LCMDecoder(nn.Module):
    def __init__(self):
        super(LCMDecoder, self).__init__()
        self.db4_1 = DecoderBlock(128 + 64, 128)
        self.db1_0 = DecoderBlock(128 + 64, 128)

    def decode(self, x1, x2, db):
        x1 = F.upsample(x1, x2.shape[2:], mode='bilinear')
        x = torch.cat([x1, x2], 1)
        x = db(x)
        return x

    def forward(self, x0, x1, x4):
        x1 = self.decode(x4, x1, self.db4_1)
        x0 = self.decode(x1, x0, self.db1_0)
        return x0, x1

class GCN(nn.Module):
    def __init__(self, num_state, num_node, bias=False):
        super(GCN, self).__init__()
        self.conv1 = nn.Conv1d(num_node, num_node, kernel_size=1)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv1d(num_state, num_state, kernel_size=1, bias=bias)

    def forward(self, x):
        h = self.conv1(x)
        h = h - x
        h = self.relu(self.conv2(h))
        return h


class BasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1):
        super(BasicConv2d, self).__init__()

        self.conv = nn.Conv2d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x
class PDFGA(nn.Module):
    def __init__(self):
        super(PDFGA, self).__init__()
        self.db4 = DecoderBlock(256 + 128, 128)
        self.db1 = DecoderBlock(256, 128)
        self.db0 = DecoderBlock(256, 128)
        self.gcn = GCN(32, 32)
        self.Translayer6_1 = BasicConv2d(128, 32, 3, padding=1)
        self.Translayer7_1 = BasicConv2d(32, 384, 3, padding=1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.mlp = nn.Sequential(
            nn.Conv2d(32, 32 // 16, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(32 // 16, 32, 1, bias=False)
        )

    def decode(self, x1, x2, db):
        x1 = db(x1)
        x1 = F.upsample(x1, x2.shape[2:], mode='bilinear')
        x = torch.cat([x1, x2], 1)
        return x

    def forward(self, x0, x1, x1_4, x2_4): #x0 和 x1是差值
        n, c, h, w = x1.size()

        x1_diffA = torch.abs(x1_4 - x2_4)
        x4 = torch.cat([x1_4, x2_4, x1_diffA], 1)
        x1_diffA = self.Translayer6_1(x1_diffA)
        x_out1 = x1_diffA.view(n, 32, -1)  # x_out (16, 32, 4096)
        map_1 = self.gcn(x_out1)  # map (16, 32, 4096)
        maps_1 = map_1.view(n, 32, *x1_diffA.size()[2:])  # maps (16 32 64 64)
        change1 = (torch.sigmoid(self.mlp(self.max_pool(x1_diffA)))) * maps_1  # change (16 32 64 64)
        x4 = x4 + self.Translayer7_1(change1)

        x1 = self.decode(x4, x1, self.db4)
        x0 = self.decode(x1, x0, self.db1)
        x0 = self.db0(x0)

        return x0

class CBAM(nn.Module):
    def __init__(self, channel, reduction=16, spatial_kernel=7):
        super(CBAM, self).__init__()
        # channel attention 压缩H,W为1
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # shared MLP
        self.mlp = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // reduction, channel, 1, bias=False)
        )
        # spatial attention
        self.conv = nn.Conv2d(2, 1, kernel_size=spatial_kernel,
                              padding=spatial_kernel // 2, bias=False)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        max_out = self.mlp(self.max_pool(x))
        avg_out = self.mlp(self.avg_pool(x))
        channel_out = self.sigmoid(max_out + avg_out)
        x = channel_out * x
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        avg_out = torch.mean(x, dim=1, keepdim=True)
        spatial_out = self.sigmoid(self.conv(torch.cat([max_out, avg_out], dim=1)))
        x = spatial_out * x
        return x
class CHSCD(nn.Module):
    def __init__(self, channels=3, num_classes=5,drop_rate=0.4):
        super(CHSCD, self).__init__()
        self.HSFE_Encoder = Encoder()
        self.lcm_decoder = LCMDecoder()
        self.cd_branch = PDFGA()
        self.lcm_classifier1 = nn.Sequential(nn.Conv2d(128, 64, kernel_size=1), nn.BatchNorm2d(64), nn.ReLU(),
                                             nn.Conv2d(64, num_classes, kernel_size=1))
        self.lcm_classifier2 = nn.Sequential(nn.Conv2d(128, 64, kernel_size=1), nn.BatchNorm2d(64), nn.ReLU(),
                                             nn.Conv2d(64, num_classes, kernel_size=1))
        self.cd_classifier = nn.Sequential(nn.Conv2d(128, 64, kernel_size=1), nn.BatchNorm2d(64), nn.ReLU(),
                                           nn.Conv2d(64, 1, kernel_size=1))

        initialize_weights(self.lcm_decoder, self.cd_branch, self.lcm_classifier1, self.lcm_classifier2,
                           self.cd_classifier)
        self.MDRI_t1_t2 = MDRI(128)

        self.drop = nn.Dropout2d(drop_rate)

    def forward(self, x1, x2):
        x_size = x1.size()

        x1_0, x1_1, x1_4_ = self.HSFE_Encoder(x1) # 剩余三个的拼接（x1_4_）
        x2_0, x2_1, x2_4_ = self.HSFE_Encoder(x2)

        x1_4_, x2_4_ = self.MDRI_t1_t2(x1_4_, x2_4_)#(2 128 52 52)

        x1_4 = x1_4_ + self.drop(x1_4_)
        x2_4 = x2_4_ + self.drop(x2_4_)

        x1_0, x1_1 = self.lcm_decoder(x1_0, x1_1, x1_4)
        x2_0, x2_1 = self.lcm_decoder(x2_0, x2_1, x2_4)

        cd_map = self.cd_branch(torch.abs(x1_0 - x2_0), torch.abs(x1_1 - x2_1), x1_4, x2_4)# (2 128)

        change = self.cd_classifier(cd_map)
        out1 = self.lcm_classifier1(x1_0)
        out2 = self.lcm_classifier2(x2_0)

        return F.upsample(change, x_size[2:], mode='bilinear'), \
               F.upsample(out1, x_size[2:], mode='bilinear'), \
               F.upsample(out2, x_size[2:], mode='bilinear')

# net = CHSCD(3, 7).cuda()
# H = W = 512
# C = 3
# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# x1 = torch.randn(2, C,H, W).to(device)
# x2 = torch.randn(2, C,H, W).to(device)
# out,out1,out2 = net(x1,x2)
# print(out.shape, out1.shape, out2.shape)