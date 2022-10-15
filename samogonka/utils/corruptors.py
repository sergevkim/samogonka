import torch
from torch import Tensor


class MaskCorruptor:
    def __init__(self, lambda_coef: float = 0.5) -> None:
        self.lambda_coef = lambda_coef

    def __call__(self, x: Tensor) -> Tensor:
        noise = torch.rand_like(x)
        mask = torch.where(noise < self.lambda_coef, 1, 0)
        corrupted_x = mask * x

        return corrupted_x


class GaussianNoiseCorruptor:
    def __init__(self, mean: float = 0, std: float = 1) -> None:
        self.mean = mean
        self.std = std

    def __call__(self, x: Tensor) -> Tensor:
        noise = torch.normal(mean=self.mean, std=self.std, size=x.shape)
        corrupted_x = x + noise

        return corrupted_x
