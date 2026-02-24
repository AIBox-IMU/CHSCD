# Reproduced version according to the original paper description "ChangeMask: Deep multi-task encoder-transformer-decoder architecture for semantic change detection"
import torch
import torch.nn as nn
import torchvision

from segmentation_models_pytorch.base import SegmentationHead
from segmentation_models_pytorch.decoders.unet.decoder import UnetDecoder
from segmentation_models_pytorch.encoders.efficientnet import (EfficientNetEncoder, efficient_net_encoders)

# pip install segmentation-models-pytorch
class Squeeze2(nn.Module):
    def forward(self, x):
        return x.squeeze(dim=2)


class TSTBlock(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size=3):
        super().__init__()        
        self.block = nn.Sequential(
            nn.Conv3d(in_channel, out_channel, [2, kernel_size, kernel_size], stride=1, padding=(0, kernel_size//2, kernel_size//2), bias=False),
            Squeeze2(),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(True)
        )
        
    def forward(self, x):
        x = self.block(x)
        return x


class TST(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(TST, self).__init__()   
        
        self.tst_list = nn.ModuleList([TSTBlock(in_channel, out_channel) for in_channel, out_channel in zip(in_channels, out_channels)])
        
    def forward(self, features_A, features_B):
        features_AB = [tst(torch.stack([fa, fb], dim=2)) for tst, fa, fb in zip(self.tst_list, features_A, features_B)]
        features_BA = [tst(torch.stack([fb, fa], dim=2)) for tst, fb, fa in zip(self.tst_list, features_B, features_A)]
        tst_features = [fab * fba for fab, fba in zip(features_AB, features_BA)]
        return tst_features
    

class ChangeMask(nn.Module):
    def __init__(self, num_classes=7,seg_pretrain=True,pretrained = True):
        super(ChangeMask, self).__init__()
        self.seg_pretrain = seg_pretrain
        encoder_params = efficient_net_encoders['efficientnet-b0']['params']
        self.encoder = EfficientNetEncoder(**encoder_params)
        
        self.seg_decoder = UnetDecoder(
            encoder_channels = encoder_params['out_channels'],
            decoder_channels = (256, 128, 64, 32, 16),
            n_blocks = 5
        )

        self.bcd_decoder = UnetDecoder(
            encoder_channels = encoder_params['out_channels'],
            decoder_channels = (256, 128, 64, 32, 16),
            n_blocks = 5
        )
        
        self.bcd_head = SegmentationHead(in_channels=16, out_channels=1, kernel_size=1)
        self.seg_head = SegmentationHead(in_channels=16, out_channels=num_classes, kernel_size=1)
        
        self.tst = TST(
            in_channels = encoder_params['out_channels'],
            out_channels = encoder_params['out_channels'],
        )
        
        if pretrained:
            self._init_weighets()
            
    def _init_weighets(self):
        efficientnet_b0 = torchvision.models.efficientnet_b0(weights='EfficientNet_B0_Weights.IMAGENET1K_V1')                  
        pretrained_dict = efficientnet_b0.state_dict()
        encoder_dict = self.encoder.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in encoder_dict}
        encoder_dict.update(pretrained_dict)    
                         
    def forward(self, img_A, img_B):
        # img_A = imgs[:, 0:3, :, :]
        # img_B = imgs[:, 3::, :, :]
        
        features_A = self.encoder(img_A)
        features_B = self.encoder(img_B)

        tst_features = self.tst(features_A, features_B)
        # tst_feature = tst_features[-1]
        logits_BCD = self.bcd_decoder(tst_features)
        logits_BCD = self.bcd_head(logits_BCD)
            
        seg_A = self.seg_decoder(features_A)
        seg_B = self.seg_decoder(features_B)
        logits_A = self.seg_head(seg_A)
        logits_B = self.seg_head(seg_B)
        
        outputs = {}
        outputs['bcl_loss'] = torch.tensor(0)
        outputs['seg_A'] = logits_A
        outputs['seg_B'] = logits_B
        outputs['BCD'] = logits_BCD     
        return outputs['BCD'], outputs['seg_A'], outputs['seg_B']

#
# if __name__ == '__main__':
#     import torch
#     model = ChangeMask(num_classes=5).cuda()
#     H = W = 32
#     C = 3
#     device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#     x1 = torch.randn(12, C,H, W).to(device)
#     x2 = torch.randn(12, C,H, W).to(device)
#
#     output, output1, output2 = model(x1, x2)
#     print(output.shape, output1.shape, output2.shape)