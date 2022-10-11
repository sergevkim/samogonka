import torch.nn as nn

from einops.layers.torch import Rearrange
from torchvision.models import resnet18


class ResNet18Model(nn.Module):
    def __init__(self, ):
        super().__init__()
        model = resnet18(weights=None)
        self.feature_extractor = nn.Sequential(*(list(model.children())[:-2]))
        self.neck = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=2048, kernel_size=1),
            nn.BatchNorm2d(2048),
        )
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
        x = self.neck(x)

        return x
