from typing import Any, Dict

from pytorch_lightning import LightningModule
from torch import Tensor
from torch.nn import CrossEntropyLoss, Module
from torch.optim import Adam
from torchmetrics import Accuracy


class ClassificationModule(LightningModule):
    def __init__(self, model: Module, learning_rate: float = 3e-4) -> None:
        super().__init__()
        self.model = model
        self.criterion = CrossEntropyLoss()
        self.accuracy_metric = Accuracy()
        self.learning_rate = learning_rate

    def forward(self, x: Tensor) -> Tensor:
        return self.model(x)

    def _step(self, batch: Any, batch_idx: int) -> Dict[str, Tensor]:
        images, labels = batch
        predicts = self.model(images)
        loss = self.criterion(predicts, labels)
        accuracy = self.accuracy_metric(predicts, labels)

        info = {'loss': loss, 'accuracy': accuracy}

        return info

    def training_step(self, batch: Any, batch_idx: int) -> Dict[str, Tensor]:
        info = self._step(batch, batch_idx)
        for metric in info:
            self.log(metric, info[metric])

        return info

    def validation_step(self, batch: Any, batch_idx: int) -> Dict[str, Tensor]:
        info = self._step(batch, batch_idx)
        for metric in info:
            self.log(f'val_{metric}', info[metric])

        return info

    def test_step(self, batch: Any, batch_idx: int) -> Dict[str, Tensor]:
        info = self._step(batch, batch_idx)
        for metric in info:
            self.log(f'test_{metric}', info[metric])

        return info

    def configure_optimizers(self):
        return Adam(params=self.model.parameters(), lr=self.learning_rate)
