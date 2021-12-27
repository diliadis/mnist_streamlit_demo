import torch
import pytorch_lightning as pl
from torchvision.models import resnet18
from torch import nn

class ResNetMNIST(pl.LightningModule):
    def __init__(self):
        super().__init__()
        # define model and loss
        self.model = resnet18(num_classes=10)
        self.model.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        self.loss = nn.CrossEntropyLoss()

    # this decorator automatically handles moving your tensors to GPU if required
    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_no):
        # implement single training step
        x, y = batch
        logits = self(x)
        loss = self.loss(logits, y)
        return loss

    def configure_optimizers(self):
        # choose your optimizer
        #return torch.optim.RMSprop(self.parameters(), lr=0.005)
        optimizer = torch.optim.Adam(self.parameters(), lr=0.005)
        return optimizer

class CustomMNIST(pl.LightningModule):
    def __init__(self):
        super().__init__()
        # define model and loss

        self.l1 = torch.nn.Linear(28 * 28, 10)

        self.loss = nn.CrossEntropyLoss()

    # this decorator automatically handles moving your tensors to GPU if required
    def forward(self, x):
        return torch.relu(self.l1(x.view(x.size(0), -1)))

    def training_step(self, batch, batch_no):
        # implement single training step
        x, y = batch
        logits = self(x)
        loss = self.loss(logits, y)
        return loss

    def configure_optimizers(self):
        # choose your optimizer
        #return torch.optim.RMSprop(self.parameters(), lr=0.005)
        optimizer = torch.optim.Adam(self.parameters(), lr=0.005)

        return optimizer