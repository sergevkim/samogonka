import torch

from torchvision.models.resnet import (
    ResNet, Bottleneck, BasicBlock, conv1x1
)
from torch import nn, Tensor
from torch.nn import functional as F
from typing import Type, List, Union, Optional, Callable, Dict, Any


class PredictionHead(nn.Module):
    def __init__(self, channels: int, num_classes: int):
        super(PredictionHead, self).__init__()
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(channels, num_classes)

    def forward(self, feats: Tensor) -> Tensor:
        feats = self.avgpool(feats).view(feats.shape[0], -1)
        return self.fc(feats)


class ResNetSelfDistillation(ResNet):
    @staticmethod
    def last_channels(layer):
        if isinstance(layer[-1], BasicBlock):
            return layer[-1].conv2.out_channels
        return layer[-1].conv3.out_channels

    def __init__(
            self,
            num_classes: int = 100,
            **kwargs
    ):
        super(ResNetSelfDistillation, self).__init__(num_classes=num_classes, **kwargs)
        self.fc = nn.Identity()
        self.avgpool = nn.Identity()

        block = type(self.layer4[0])

        pred_feats = self.last_channels(self.layer4)
        out_planes = pred_feats // block.expansion

        self.bottleneck1 = block(
            inplanes=self.last_channels(self.layer1),
            planes=out_planes, stride=8,
            downsample=nn.Sequential(
                conv1x1(self.last_channels(self.layer1), pred_feats, stride=8),
                nn.BatchNorm2d(pred_feats)
            )
        )
        self.bottleneck2 = block(
            inplanes=self.last_channels(self.layer2),
            planes=out_planes, stride=4,
            downsample=nn.Sequential(
                conv1x1(self.last_channels(self.layer2), pred_feats, stride=4),
                nn.BatchNorm2d(pred_feats)
            )
        )
        self.bottleneck3 = block(
            inplanes=self.last_channels(self.layer3),
            planes=out_planes, stride=2,
            downsample=nn.Sequential(
                conv1x1(self.last_channels(self.layer3), pred_feats, stride=2),
                nn.BatchNorm2d(pred_feats)
            )
        )

        self.phead4 = PredictionHead(pred_feats, num_classes)
        self.phead3 = PredictionHead(pred_feats, num_classes)
        self.phead2 = PredictionHead(pred_feats, num_classes)
        self.phead1 = PredictionHead(pred_feats, num_classes)

    def _forward_backbone(self, x: Tensor, regime: str = 'full') -> Dict[str, Tensor]:
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        result: Dict[str, Tensor] = dict()

        last_layer = 4
        if 'until_' in regime or 'only_' in regime:
            # 'until_3', 'until_4', 'until_2', 'until_1'
            last_layer = int(regime[-1])

        for layer_idx in range(1, last_layer + 1):
            x = getattr(self, f'layer{layer_idx}')(x)
            if regime == 'full' or layer_idx == last_layer or 'until_' in regime:
                result[f'layer{layer_idx}'] = x
        return result

    def forward(self, x: Tensor, regime: str = 'full') -> Dict[str, Tensor]:
        result = self._forward_backbone(x, regime)
        layers_names = list(result.keys())
        for layer_name in layers_names:
            feats = result[layer_name]
            num_layer = int(layer_name[-1])
            if num_layer != 4:
                bottleneck = getattr(self, f'bottleneck{num_layer}')
                feats = bottleneck(feats)
                result[layer_name] = feats
            logits: Tensor = getattr(self, f'phead{num_layer}')(feats)
            result[f'logits{num_layer}'] = logits
        return result


def _resnet(
    block: Type[Union[BasicBlock, Bottleneck]],
    layers: List[int],
    **kwargs: Any,
) -> ResNetSelfDistillation:
    model = ResNetSelfDistillation(block=block, layers=layers, **kwargs)
    return model


def resnet18(**kwargs: Any) -> ResNetSelfDistillation:
    return _resnet(BasicBlock, [2, 2, 2, 2], **kwargs)


def resnet34(**kwargs: Any) -> ResNetSelfDistillation:
    return _resnet(BasicBlock, [3, 4, 6, 3], **kwargs)


def resnet50(**kwargs: Any) -> ResNetSelfDistillation:
    return _resnet(Bottleneck, [3, 4, 6, 3], **kwargs)