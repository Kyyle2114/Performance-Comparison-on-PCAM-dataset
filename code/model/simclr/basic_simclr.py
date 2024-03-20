import torch
import torch.nn as nn

class BasicSimCLR(nn.Module):
    def __init__(self, backbone, num_classes=128):
        """
        Basic SimCLR model 

        Args:
            backbone (torch backbone): torch backbone 
            num_classes (int, optional): dimension of z. Defaults to 128.
        """

        super(BasicSimCLR, self).__init__()
        
        self.backbone = backbone.backbone_features
        self.avg = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Sequential(nn.Flatten(),
                                nn.Linear(512, 512),
                                nn.ReLU(),
                                nn.Linear(512, num_classes))

    def forward(self, x):
        x = self.backbone(x)
        x = self.avg(x)
        output = self.fc(x)
        return output