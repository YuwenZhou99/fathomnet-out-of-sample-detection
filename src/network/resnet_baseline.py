import torchvision.models as models
import torch.nn as nn
from torchvision.models import ResNet50_Weights

class ResNetBaseline(nn.Module):
    def __init__(self, num_classes: int, pretrained: bool = True, layer_size = None):
        super(ResNetBaseline, self).__init__()
        self.model = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V1 if pretrained else None)
        if layer_size is not None:
            last_dim = self.model.fc.in_features
            self.model.fc = nn.Sequential(
                nn.Linear(last_dim, layer_size),
                nn.ReLU(),
                nn.Linear(layer_size, num_classes)
            )
        else:
            # Replace the final fully connected layer
            self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)


    def forward(self, x):
        # expects input shape (B, C, H, W) https://docs.pytorch.org/vision/main/models/generated/torchvision.models.resnet50.html
        # images are resized to resize_size=[256] using interpolation=InterpolationMode.BILINEAR
        # followed by a central crop of crop_size=[224]. finally values are rescaled to [0.0, 1.0]
        # and than normalized using mean=[0.485, 0.456, 0.406] and std=[0.229, 0.224, 0.225]
        return self.model(x)