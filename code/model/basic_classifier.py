import torch
import torch.nn as nn
import torchvision.models as models

class BasicClassifier(nn.Module):
    def __init__(self, backbone, freezing=False, num_classes=1):
        """
        Basic Classifier with Global Average Pooling
        
        Args:
            backbone (torch backbone)
            freezing (bool, optional): if True, freeze weight of backbone. Defaults to False.
            num_classes (int, optional): number of classes. Defaults to 1(binary classification).
        """
        super(BasicClassifier, self).__init__()
        
        self.backbone = backbone.backbone_features
        
        if freezing:
            for param in self.backbone.parameters():
                param.requires_grad = False
        
        self.avg = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Sequential(nn.Flatten(),
                                nn.Linear(512, 64),
                                nn.ReLU(),
                                nn.Dropout(0.5),
                                nn.Linear(64, num_classes))                   

    def forward(self, x):
        x = self.backbone(x)
        x = self.avg(x)
        output = self.fc(x)
        return output