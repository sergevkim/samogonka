import torch.nn as nn

from einops.layers.torch import Rearrange
from torchvision.models import resnet50, ResNet50_Weights


class ResNet50Model(nn.Module):
    def __init__(self, pretrained: bool = False):
        super().__init__()
        weights = None if pretrained is True else ResNet50_Weights.IMAGENET1K_V1
        model = resnet50(weights=weights)
        self.feature_extractor = nn.Sequential(*(list(model.children())[:-2]))
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(output_size=(1, 1)),
            Rearrange('bs c 1 1 -> bs c'),
            nn.Linear(in_features=2048, out_features=10, bias=True),
        )

    def forward(self, x, inputs: str = 'images'):
        features = None
        if inputs == 'images':
            features = self.extract_features(x)
        elif inputs == 'features':
            features = x

        outputs = self.head(features)

        return outputs

    def extract_features(self, x):
        x = self.feature_extractor(x)

        return x
