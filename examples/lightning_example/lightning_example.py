# imports
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
import pytorch_lightning as ptl
from pytorch_lightning import Trainer
from pathlib import Path

# import the checkpointer callback
from checkpointer.lightning_callback import LightningCheckpointerCallback


# Download training data from open datasets.
training_data = datasets.FashionMNIST(
    root="data",
    train=True,
    download=True,
    transform=ToTensor(),
)

# Download test data from open datasets.
test_data = datasets.FashionMNIST(
    root="data",
    train=False,
    download=True,
    transform=ToTensor(),
)

batch_size = 256

# Create data loaders.
train_dataloader = DataLoader(training_data, batch_size=batch_size, num_workers=4)
test_dataloader = DataLoader(test_data, batch_size=batch_size, num_workers=4)

for X, y in test_dataloader:
    print(f"Shape of X [N, C, H, W]: {X.shape}")
    print(f"Shape of y: {y.shape} {y.dtype}")
    break

# create the lightning module


class LightningModuleExample(ptl.LightningModule):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10)
        )
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

    def training_step(self, batch, batch_idx):
        X, y = batch
        logits = self.forward(X)
        loss = self.loss_fn(logits, y)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        X, y = batch
        logits = self.forward(X)
        loss = self.loss_fn(logits, y)
        self.log('val_loss', loss)

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(self.parameters(), lr=1e-3)
        return optimizer


model = LightningModuleExample()


# create the checkpointer callback
checkpoint_path = 'trainer_checkpoint.ckpt'

checkpointer = LightningCheckpointerCallback(
    local_checkpoint_file=Path(checkpoint_path),
    checkpoint_every=1
)

# create the lightning trainer instance
epochs = 500
trainer = Trainer(max_epochs=epochs, accelerator='gpu', callbacks=[checkpointer])
# do the fitting.
# the callbacks restor fuunction takes care of checking for checkpoints
# and transfering them. The trainers fit function takes care of the
# restoring.
trainer.fit(model, train_dataloader, test_dataloader, ckpt_path=checkpointer.restore())
