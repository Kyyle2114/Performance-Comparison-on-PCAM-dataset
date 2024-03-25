import torch
import torch.nn as nn
import torchvision.models as models 

class VGG16_Backbone(nn.Module):
    def __init__(self, pretrain=True):
        """
        Backbone : VGG16
        
        returns feature map, size ([batch_size, channels, width, height])
        
        e.g. tensor size ([1, 3, 224, 224]) -> ([1, 512, 7, 7])
        
        Args:
            pretrain (bool, optional): if True, use ImageNet weights. if False, use kaiming_normal initialize in Conv layer. Defaults to True.
        """
        super(VGG16_Backbone, self).__init__()
        
        if pretrain:
            self.backbone = models.vgg16(weights='IMAGENET1K_V1')
        else:
            self.backbone = models.vgg16(weights=None)
            self._initialize_weights()
            
        self.backbone_features = nn.Sequential(*list(self.backbone.features.children()))

    def forward(self, x):
        x = self.backbone_features(x)
        return x 
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)    