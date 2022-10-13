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

    def _step(self, batch: Any, batch_idx: int, mode: str) -> Dict[str, Tensor]:
        images, labels = batch
        predicts = self.model(images)
        loss = self.criterion(predicts, labels)
        accuracy = self.accuracy_metric(predicts, labels)

        if mode == 'train':
            info = {f'loss': loss, f'accuracy': accuracy}
        else:
            info = {f'{mode}_loss': loss, f'{mode}_accuracy': accuracy}

        return info

    def training_step(self, batch: Any, batch_idx: int) -> Dict[str, Tensor]:
        info = self._step(batch, batch_idx, mode='train')
        self.log('train', info)

        return info

    def validation_step(self, batch: Any, batch_idx: int) -> Dict[str, Tensor]:
        info = self._step(batch, batch_idx, mode='val')
        self.log('val', info)

        return info

    def test_step(self, batch: Any, batch_idx: int) -> Dict[str, Tensor]:
        info = self._step(batch, batch_idx, mode='test')
        self.log('test', info)

        return info

    def configure_optimizers(self):
        return Adam(params=self.model.parameters(), lr=self.learning_rate)
