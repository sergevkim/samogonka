import torch

from torch import Tensor
from torch.nn import CrossEntropyLoss, Module
from typing import Any, Dict

from .classification_module import ClassificationModule


class SelfDistillationModule(ClassificationModule):
    def __init__(
            self,
            model: Module,
            learning_rate: float = 3e-4,
            temp: float = 2.0,
            alpha_coef: float = 0.35,
            lambda_coef: float = 0.05
    ) -> None:
        super().__init__(model, learning_rate)
        self.T = temp
        self.lambda_coef = lambda_coef
        self.alpha_coef = alpha_coef
        self.mse_loss = torch.nn.MSELoss()

    def forward(self, x: Tensor, regime: str = 'full') -> Dict[str, Tensor]:
        # assume, that currently we work only with resnets, "full" we may change later
        # we don't use others regimes in code
        return self.model(x, regime)

    def _step(self, batch: Any, batch_idx: int) -> Dict[str, Tensor]:
        images, labels = batch
        predicts: Dict[str, Tensor] = self.model(images)

        info: Dict[str, Tensor] = dict()

        for pred_name, pred in predicts.items():
            layer_num = int(pred_name[-1])
            if 'logits' in pred_name:
                logits = pred / self.T
                info[f'CE-{layer_num}'] = self.criterion(logits, labels)
                info[f'ACC-{layer_num}'] = self.accuracy_metric(logits, labels)
                continue
            if layer_num == 4:
                continue
            assert 'layer' in pred_name
            deepest_feats = predicts['layer4']
            info[f'MSE-F{layer_num}'] = self.mse_loss(pred, deepest_feats)
            info[f'KL-{layer_num}'] = self._kl(pred / self.T, deepest_feats / self.T)

        info['KL'] = info['KL-1'] + info['KL-2'] + info['KL-3']
        info['MSE-F'] = info['MSE-F1'] + info['MSE-F2'] + info['MSE-F3']
        info['CE'] = info['CE-1'] + info['CE-2'] + info['CE-3'] + info['CE-4']

        info['loss'] = self.alpha_coef * info['KL'] + \
                       (1 - self.alpha_coef) * (info['CE-1'] + info['CE-2'] + info['CE-3']) + \
                       self.lambda_coef * info['MSE-F'] + info['CE-4']
        info['accuracy'] = info['ACC-4']
        mean_logits = (predicts['logits1'] + predicts['logits2'] + predicts['logits3'] + predicts['logits4']) / 4
        info['accuracy_ensemble'] = self.accuracy_metric(mean_logits, labels)

        return info

    @staticmethod
    def _kl(student_logits, teacher_logits) -> Tensor:
        # KL(q_s || q_t) = sum [q_s log q_s - q_s log p_t]
        log_q_s = torch.nn.functional.log_softmax(student_logits, -1)
        log_q_t = torch.nn.functional.log_softmax(teacher_logits, -1)

        q_s = torch.exp(log_q_s)

        return torch.mean(torch.sum(q_s * (log_q_s - log_q_t), dim=-1))

    def _log_info(self, info, prefix):
        for k, v in info.items():
            pp = prefix
            if 'MSE' in k:
                pp += '/MSE'
            elif 'KL' in k:
                pp += '/KL'
            elif 'CE' in k:
                pp += '/CE'
            elif 'ACC' in k:
                pp += '/ACC'
            self.log(pp + '/' + k, v.item())

    def training_step(self, batch: Any, batch_idx: int) -> Dict[str, Tensor]:
        info = self._step(batch, batch_idx)
        self._log_info(info, 'train')

        return info

    def validation_step(self, batch: Any, batch_idx: int) -> Dict[str, Tensor]:
        info = self._step(batch, batch_idx)
        self._log_info(info, 'val')

        return info

    def test_step(self, batch: Any, batch_idx: int) -> Dict[str, Tensor]:
        info = self._step(batch, batch_idx)
        self._log_info(info, 'test')

        return info

