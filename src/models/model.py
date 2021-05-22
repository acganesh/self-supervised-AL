import pytorch_lightning as pl
import torch

from ..byol_pytorch.byol_pytorch import BYOL


class SelfSupervisedLearner(pl.LightningModule):
    def __init__(self, net, lr, **kwargs):
        super().__init__()
        self.learner = BYOL(net, **kwargs)
        self.lr = lr

    def forward(self, images):
        return self.learner(images)

    def training_step(self, images, _):
        loss = self.forward(images)
        return {'loss': loss}

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)

    def on_before_zero_grad(self, _):
        if self.learner.use_momentum:
            self.learner.update_moving_average()


class LogisticRegression(torch.nn.Module):
    def __init__(self, input_dim, output_dim):
        super(LogisticRegression, self).__init__()
        self.linear = torch.nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return self.linear(x)
