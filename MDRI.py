import torch
import torch.nn as nn

"Rotate to Attend: Convolutional Triplet Attention Module"

class BasicConv(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, relu=True, bn=True, bias=False):
        super(BasicConv, self).__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_planes,eps=1e-5, momentum=0.01, affine=True) if bn else None
        self.relu = nn.ReLU() if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x
class MDRI(nn.Module):
    # ##
    def __init__(self, in_dim):
        super(MDRI, self).__init__()
        self.chanel_in = in_dim

        self.query_conv1 = nn.Conv2d(in_channels=in_dim, out_channels=in_dim*2 , kernel_size=1)
        self.key_conv1 = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 2, kernel_size=1)
        self.value_conv1 = nn.Conv2d(in_channels=in_dim, out_channels=in_dim//2, kernel_size=1)
        self.conv1 = nn.Conv2d(in_channels=in_dim// 2, out_channels=in_dim, kernel_size=1)

        self.query_conv2 = nn.Conv2d(in_channels=in_dim, out_channels=in_dim*2 , kernel_size=1)
        self.key_conv2 = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 2, kernel_size=1)
        self.value_conv2 = nn.Conv2d(in_channels=in_dim, out_channels=in_dim//2, kernel_size=1)
        self.conv2 = nn.Conv2d(in_channels=in_dim // 2, out_channels=in_dim, kernel_size=1)

        self.priors = nn.AdaptiveAvgPool2d(output_size=(6, 6))

        self.gamma1 = nn.Parameter(torch.zeros(1))
        self.gamma2 = nn.Parameter(torch.zeros(1))

        self.softmax = nn.Softmax(dim=-1)

        self.q_k_v = TripletAttention()

    def forward(self, x1, x2):
        ''' inputs :
                x1 : input feature maps( B X C X H X W)
                x2 : input feature maps( B X C X H X W)
            returns :
                out : attention value + input feature
                attention: B X (HxW) X (HxW) '''
        q1, k1, v1 = self.q_k_v(x1) # TripleAttention x_hw, x_cw, x_hc
        q2, k2, v2 = self.q_k_v(x2) #(, 128 52 52)

        m_batchsize, C, height, width = x1.size()
        q1 = self.priors(self.query_conv1(q1))[:, :, 1:-1, 1:-1].reshape(m_batchsize, self.chanel_in//2, -1).permute(0, 2, 1) #（16 16 32）
        k1 = self.key_conv1(k1).view(m_batchsize, -1, width * height) #（16 32 1024） #(12 ,128, 52, 52)
        v1 = self.value_conv1(v1).view(m_batchsize, -1, width * height)  #（16 32 1024）

        q2 = self.priors(self.query_conv2(q2))[:, :, 1:-1, 1:-1].reshape(m_batchsize, self.chanel_in//2, -1).permute(0, 2, 1)# #（16 16 32）
        k2 = self.key_conv2(k2).view(m_batchsize, -1, width * height) # k1=k2=(16 256 1024) #（16 32 1024）
        v2 = self.value_conv2(v2).view(m_batchsize, -1, width * height) # v1 = v2 =(16 256 1024)#（16 32 1024）

        energy2 = torch.bmm(q1, k2)
        attention2 = self.softmax(energy2)
        out1 = torch.mul(v1, attention2) #(16 16 1024)（32）  （64）(16 32 1024)
        out1 = out1.view(m_batchsize, C//2, height, width)
        out1 = self.conv2(out1)

        energy1 = torch.bmm(q2, k1)
        attention1 = self.softmax(energy1)
        out2 = torch.mul(v2, attention1)
        out2 = out2.view(m_batchsize, C//2, height, width)
        out2 = self.conv1(out2)

        out1 = x1 + self.gamma1 * out1
        out2 = x2 + self.gamma2 * out2

        return out1, out2
class ZPool(nn.Module):
    def forward(self, x):
        # 以建立CW之间的交互为例, x:(B, H, C, W)
        a = torch.max(x,1)[0].unsqueeze(1) # 全局最大池化: (B, H, C, W)->(B, 1, C, W);  torch.max返回的是数组:[最大值,对应索引]
        b = torch.mean(x,1).unsqueeze(1)   # 全局平均池化: (B, H, C, W)->(B, 1, C, W);
        c = torch.cat((a, b), dim=1)       # 在对应维度拼接最大和平均特征: (B, 2, C, W)
        return c

class AttentionGate(nn.Module):
    def __init__(self):
        super(AttentionGate, self).__init__()
        kernel_size = 7
        self.compress = ZPool()
        self.conv = BasicConv(2, 1, kernel_size, stride=1, padding=(kernel_size-1) // 2, relu=False)
    def forward(self, x):
        # 以建立CW之间的交互为例, x:(B, H, C, W)
        x_compress = self.compress(x) # 在对应维度上执行最大池化和平均池化,并将其拼接: (B, H, C, W) --> (B, 2, C, W);
        x_out = self.conv(x_compress) # 通过conv操作将最大池化和平均池化特征映射到一维: (B, 2, C, W) --> (B, 1, C, W);
        scale = torch.sigmoid_(x_out) # 通过sigmoid函数生成权重: (B, 1, C, W);
        return x * scale              # 对输入进行重新加权表示: (B, H, C, W) * (B, 1, C, W) = (B, H, C, W)

class TripletAttention(nn.Module):
    def __init__(self, no_spatial=True):
        super(TripletAttention, self).__init__()
        self.cw = AttentionGate()
        self.hc = AttentionGate()
        self.no_spatial=no_spatial
        # if not no_spatial:
        self.hw = AttentionGate()
    def forward(self, x):
        # 建立C和W之间的交互:
        x_perm1 = x.permute(0,2,1,3).contiguous() # (B, C, H, W)--> (B, H, C, W);  执行“旋转操作”,建立C和W之间的交互,所以要在H维度上压缩
        x_out1 = self.cw(x_perm1) # (B, H, C, W)-->(B, H, C, W);  在H维度上进行压缩、拼接、Conv、sigmoid操作, 然后通过权重重新加权
        x_cw = x_out1.permute(0,2,1,3).contiguous() # 恢复与输入相同的shape,也就是重新旋转回来: (B, H, C, W)-->(B, C, H, W)

        # 建立H和C之间的交互:
        x_perm2 = x.permute(0,3,2,1).contiguous() # (B, C, H, W)--> (B, W, H, C); 执行“旋转操作”,建立H和C之间的交互,所以要在W维度上压缩
        x_out2 = self.hc(x_perm2) # (B, W, H, C)-->(B, W, H, C);  在W维度上进行压缩、拼接、Conv、sigmoid操作, 然后通过权重重新加权
        x_hc = x_out2.permute(0,3,2,1).contiguous() # 恢复与输入相同的shape,也就是重新旋转回来: (B, W, H, C)-->(B, C, H, W)
        x_hw = self.hw(x)
        return x_hw, x_cw, x_hc
#
if __name__ == '__main__':
    # (B, C, H, W)
    input1=torch.randn(12,128,52,52)
    input2=torch.randn(12,128,52,52)
    Model = MDRI(128)
    out1, out2=Model(input1, input2)
    print(out1.shape)
