import typing as tp

import torch
import pytorch_lightning as ptl


class Model(ptl.LightningModule):
    def __init__(self, learning_rate, is_distiling: bool, training_model_type: str, teacher_path: tp.Optional[str] = None,
                 temperature=1):
        super().__init__()
        self.save_hyperparameters()
        self.learning_rate = learning_rate
        self.is_distiling = is_distiling
        self.temperature = temperature
        if not is_distiling:
            if training_model_type not in ["resnet18", "resnet50"]:
                raise ValueError
            self.model = torch.hub.load('pytorch/vision:v0.10.0', training_model_type, pretrained=False)
        else:
            if teacher_path is None:
                raise ValueError
            checkpoint_teacher = torch.load(teacher_path)['state_dict']
            self.teacher = torch.hub.load('pytorch/vision:v0.10.0', "resnet50", pretrained=False)
            new_checkpoint_dict = {}
            for key, value in checkpoint_teacher.items():
                new_checkpoint_dict[key[6:]] = value
            self.teacher.load_state_dict(new_checkpoint_dict)

            self.student = torch.hub.load('pytorch/vision:v0.10.0', "resnet18", pretrained=False)

            self.kl_div_loss = torch.nn.KLDivLoss(log_target=True)

        self.loss = torch.nn.CrossEntropyLoss()

    def configure_optimizers(self):
        if self.is_distiling:
            return torch.optim.Adam(self.student.parameters(), self.learning_rate)
        else:
            return torch.optim.Adam(self.model.parameters(), self.learning_rate)

    def forward(self, batch):
        images, target_classes = batch

        if self.is_distiling:
            teacher_logits = self.teacher(images)
            teacher_log_distribution = torch.nn.functional.log_softmax(teacher_logits / self.temperature, dim=1)

            logits = self.student(images)
            student_distribution = torch.nn.functional.softmax(logits / self.temperature, dim=1)
            kl_loss = self.temperature ** 2 + self.kl_div_loss(student_distribution, teacher_log_distribution)

            loss = self.loss(logits, target_classes)
        else:
            logits = self.model(images)
            loss = self.loss(logits, target_classes)
            kl_loss = None

        pred_class = torch.argmax(logits, dim=1)
        accuracy = (pred_class == target_classes).float().mean()
        return loss, accuracy, kl_loss

    def training_step(self, batch, batch_idx):
        loss, accuracy, kl_loss = self(batch)

        self.log("train loss", loss)
        self.log("train accuracy", accuracy)
        if kl_loss is not None:
            self.log("train kl loss", kl_loss)
            loss += kl_loss

        return loss

    def validation_step(self, batch, batch_idx):
        loss, accuracy, _ = self(batch)

        self.log("val loss", loss)
        self.log("val accuracy", accuracy)
