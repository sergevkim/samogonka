from ssl import ALERT_DESCRIPTION_HANDSHAKE_FAILURE
from typing import Any, Dict, Optional

from pytorch_lightning import LightningModule
from torch import Tensor
from torch.nn import CrossEntropyLoss, Module, MSELoss
from torch.optim import Adam
from torchmetrics import Accuracy


class DistillationModule(LightningModule):
    def __init__(
        self,
        student,
        teacher,
        alpha_coef: float = 7e-5, # distillation coef
        lambda_coef: float = 0.5, # mask coef
        corruptor: Optional[Module] = None,
        generator: Optional[Module] = None,
    ) -> None:
        self.student = student
        self.teacher = teacher
        # TODO corruptor architecture
        self.corruptor = corruptor
        self.generator = generator

        self.alpha_coef = alpha_coef
        self.lambda_coef = lambda_coef

        self.original_criterion = CrossEntropyLoss()
        self.distillation_criterion = MSELoss()

    def training_step(self, batch: Any, batch_idx: int) -> Dict[str, Tensor]:
        images, labels = batch
        student_features = self.student.extract_features(images)
        teacher_features = self.teacher.extract_features(images)

        #corrupted_student_features = self.corruptor(student_features)
        #student_features = self.generator(corrupted_student_features)
        distillation_loss = \
            self.distillation_criterion(student_features, teacher_features)

        student_predictions = self.student(student_features, inputs='features')
        original_loss = self.original_criterion(student_predictions, labels)

        loss = original_loss + distillation_loss * self.alpha_coef

        info = {
            'loss': loss,
            'distillation_loss': distillation_loss,
            'original_loss': original_loss,
        }

        return info

    def configure_optimizers(self):
        return Adam(self.student.parameters(), lr=3e-4)
