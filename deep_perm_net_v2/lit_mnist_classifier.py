import os
import sys
from collections import defaultdict
from typing import List, Callable
from numbers import Number
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import cvxpy as cp
from cvxpylayers.torch.cvxpylayer import CvxpyLayer
import pytorch_lightning as pl

DIR_PATH = Path(__file__).parent
sys.path.append(str(DIR_PATH))
import mnist_classifier as mnist_clf


class LitMNISTClassifier(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.model = mnist_clf.MNISTClassifier()
        # self.save_hyperparameters()

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        self.log('train_loss', loss)
        return loss

    def training_epoch_end(self, outputs):
        losses = torch.as_tensor([o['loss'] for o in outputs])
        self.log('avg_train_loss', losses.mean(), prog_bar=True)

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)

        loss = F.cross_entropy(y_hat, y)
        probs = F.softmax(y_hat, dim=1)
        classes_hat = torch.argmax(probs, dim=1)
        acc = torch.mean((classes_hat == y).type(torch.float32))

        output = {
            'val_loss': loss
            , 'val_acc': acc
        }
        self.log_dict(output, prog_bar=True)

        return output

    def validation_epoch_end(self, outputs):
        losses = []
        accs = []

        for o in outputs:
            losses.append(o['val_loss'])
            accs.append(o['val_acc'])

        self.log_dict({
            'avg_val_loss': torch.as_tensor(losses).mean()
            , 'avg_val_acc': torch.as_tensor(accs).mean()
        }, prog_bar=True)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.003)

    def train_dataloader(self):
        train_dataset = datasets.MNIST(
            'mnist'
            , train=True
            , download=False
            , transform=transforms.Compose([
                transforms.ToTensor()
                , transforms.Normalize((0.1307,), (0.3081,))
            ])
        )

        train_loader = torch.utils.data.DataLoader(
            train_dataset
            , batch_size=64
            , num_workers=2
            , shuffle=True
        )

        return train_loader

    def val_dataloader(self):
        val_dataset = datasets.MNIST(
            'mnist'
            , train=False
            , download=False
            , transform=transforms.Compose([
                transforms.ToTensor()
                , transforms.Normalize((0.1307,), (0.3081,))
            ])
        )

        val_loader = torch.utils.data.DataLoader(
            val_dataset
            , batch_size=64
            , num_workers=2
        )

        return val_loader
        

if __name__ == "__main__":
    model = LitMNISTClassifier()
    logger = pl.loggers.TensorBoardLogger(save_dir='./logs_mnist_clf', name='')
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        monitor='avg_val_acc'
        , save_top_k=1
        , mode='max'
        , filepath=str(Path(logger.log_dir, 'checkpoints', '{epoch}-{avg_val_loss:.4f}-{avg_val_acc:.4f}'))
    )
    trainer = pl.Trainer(
        max_epochs=15
        , checkpoint_callback=checkpoint_callback
        , logger=logger
    )
    trainer.fit(model)
