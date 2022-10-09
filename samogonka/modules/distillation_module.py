import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.optim as optim


class DistillationModule(pl.LightningModule):
    def __init__(self, student, teacher, generator, corruptor):
        self.student = student
        self.teacher = teacher
        self.generator = generator
        self.corruptor = corruptor
        # TODO corruptor architecture
        self.original_criterion = nn.CrossEntropyLoss()
        self.distillation_criterion = nn.MSELoss()

    def training_step(self, batch):
        images, labels = batch
        student_features = self.student.extract_features(images)
        teacher_features = self.teacher.extract_features(images)

        corrupted_student_features = self.corruptor(student_features)
        recovered_student_features = self.generator(corrupted_student_features)
        distillation_loss = self.distillation_criterion(
            recovered_student_features,
            teacher_features,
        )

        student_logits = self.student.head(student_features)
        original_loss = self.original_criterion(student_logits, labels)

        loss = distillation_loss + original_loss

        return loss

    def configure_optimizers(self):
        optimizer = optim.Adam(self.student.parameters(), lr=3e-4)
        return optimizer
