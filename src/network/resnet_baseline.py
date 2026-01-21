import torchvision.models as models
import torch.nn as nn
from torchvision.models import ResNet50_Weights

class ResNetBaseline(nn.Module):
    def __init__(self, num_classes: int, pretrained: bool = True):
        super(ResNetBaseline, self).__init__()
        self.model = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V1 if pretrained else None)
        # Replace the final fully connected layer
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)

    def forward(self, x):
        # expects input shape (B, C, H, W) https://docs.pytorch.org/vision/main/models/generated/torchvision.models.resnet50.html
        return self.model(x)