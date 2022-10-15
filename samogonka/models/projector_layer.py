from collections import OrderedDict

import torch
import torch.nn as nn
from torch import Tensor


class ProjectorLayer(nn.Module):
    def __init__(self, channels_num: int, blocks_num: int = 2) -> None:
        super().__init__()
        ordered_dict = OrderedDict()
        for i in range(blocks_num):
            ordered_dict[f'conv_{i}'] = nn.Conv2d(
                in_channels=channels_num,
                out_channels=channels_num,
                kernel_size=3,
                padding=1,
            )
            if i != blocks_num - 1:
                ordered_dict[f'relu_{i}'] = nn.ReLU()
        self.projector_layer = nn.Sequential(ordered_dict)

    def forward(self, x: Tensor) -> Tensor:
        return self.projector_layer(x)


if __name__ == '__main__':
    channels_num = 64
    layer = ProjectorLayer(channels_num=channels_num, blocks_num=3)
    print(layer)
    x = torch.randn(8, channels_num, 224, 224)
    print(x.shape, layer(x).shape)
