import torch
import torchvision.models as models
import torch.nn as nn


class ViTBaseline(nn.Module):
    def __init__(self, num_classes: int, pretrained: bool = True, layer_size = None):
        super(ViTBaseline, self).__init__()
        self.model = models.vit_b_16(weights=models.ViT_B_16_Weights.IMAGENET1K_V1 if pretrained else None)
        # Replace the final fully connected layer
        self.model.heads.head = nn.Linear(self.model.heads.head.in_features, num_classes)

    def forward(self, x):
        # expects input shape (B, C, H, W) https://pytorch.org/vision/stable/models/generated/torchvision.models.vit_b_16.html
        # images are resized to resize_size[256] using interpolation=InterpolationMode.BILINEAR
        # followed by a central crop of crop_size=[224]. finally values are rescaled to [0.0, 1.0]
        # and than normalized using mean=[0.485, 0.456, 0.406] and std=[0.229, 0.224, 0.225]
        return self.model(x)
