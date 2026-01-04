import torch
import torch.nn as nn
from torchvision import models


class EyeModel(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()

        model = models.efficientnet_b0(weights="IMAGENET1K_V1")

        # same classifier as training
        in_features = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(in_features, num_classes)

        # making same architechture as traning 
        self.features = model.features
        self.avgpool = model.avgpool
        self.classifier = model.classifier

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)          
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


class YawnModel(EyeModel):
    pass
