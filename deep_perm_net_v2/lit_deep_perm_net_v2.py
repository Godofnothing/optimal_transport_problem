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
import utils
import _deep_perm_net_v2 as deep_perm_net_v2
import permuted_mnist_dataset as pmnist
import lit_mnist_classifier


class LitDeepPermNet_v2(deep_perm_net_v2.DeepPermNet_v2, pl.LightningModule):
    def __init__(
        self
        , feature_extractor: nn.Module
        , feature_size: int
        , head_size: int
        , loss_func: str
        , batch_size: int
        , lr: float
        , mnist_path: str
        , download: bool
        , duplicates_multiplier: int
        , seed: int
        , move_data_to_cuda: bool
        , use_binary_optimization_accuracy_metric: bool
        , transform: Callable = transforms.Compose([
            transforms.ToTensor()
            , transforms.Normalize((0.1307,), (0.3081,))
        ])
        , **kwargs
    ):

        super().__init__(feature_extractor, feature_size, head_size, **kwargs)

        if loss_func.lower() == 'l1':
            self.loss_func = nn.L1Loss()
        elif loss_func.lower() == 'l2':
            self.loss_func = nn.MSELoss()
        else:
            raise NotImplementedError()
    
        self.use_binary_optimization_accuracy_metric = use_binary_optimization_accuracy_metric
        self.transform = transform
        self.move_data_to_cuda = move_data_to_cuda
        self.batch_size = batch_size
        self.download = download
        self.feature_size = feature_size
        self.head_size = head_size
        self.lr = lr
        self.duplicates_multiplier = duplicates_multiplier
        self.seed = seed
        self.mnist_path = mnist_path
        self.save_hyperparameters(
            {
                'loss_func': loss_func.lower()
                , 'batch_size': batch_size
                , 'feature_size': feature_size
                , 'head_size': head_size
                , 'lr': lr
                , 'duplicates_multiplier': duplicates_multiplier
                , 'seed': seed
                , **kwargs
            }
        )

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.loss_func(y_hat.flatten(1), y.flatten(1)).type(torch.float32)
        self.log('train_loss', loss)
        return loss

    def training_epoch_end(self, outputs):
        losses = torch.as_tensor([o['loss'] for o in outputs])
        self.log('avg_train_loss', losses.mean(), prog_bar=True)

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)

        loss = self.loss_func(y_hat.flatten(1), y.flatten(1)).type(torch.float32)
        output = {'val_loss': loss}

        if self.use_binary_optimization_accuracy_metric:
            acc = utils.get_mean_accuracy(y_hat, y)
            output['val_acc'] = acc

        self.log_dict(output, prog_bar=True)

        return output

    def validation_epoch_end(self, outputs):
        losses = []
        accs = []

        for o in outputs:
            losses.append(o['val_loss'])
            if o.get('val_acc', None) is not None:
                accs.append(o['val_acc'])

        log = {'avg_val_loss': torch.as_tensor(losses).mean()}
        if len(accs) > 0:
            log['avg_val_acc'] = torch.as_tensor(accs).mean()

        self.log_dict(log, prog_bar=True)

    def configure_optimizers(self):
        return torch.optim.Adam(filter(lambda p: p.requires_grad, self.parameters()), lr=self.lr)

    def train_dataloader(self):
        train_dataset = pmnist.PermutedMnistDataset(
            self.mnist_path
            , download=self.download
            , train=True
            , transform=self.transform
            , duplicates_multiplier=self.duplicates_multiplier
            , head_size=self.head_size
            , seed=self.seed
            , move_data_to_cuda=self.move_data_to_cuda
        )

        train_loader = torch.utils.data.DataLoader(
            train_dataset
            , batch_size=self.batch_size
            , shuffle=True
        )

        return train_loader

    def val_dataloader(self):
        val_dataset = pmnist.PermutedMnistDataset(
            self.mnist_path
            , download=self.download
            , train=False
            , transform=self.transform
            , duplicates_multiplier=self.duplicates_multiplier
            , head_size=self.head_size
            , seed=self.seed
            , move_data_to_cuda=self.move_data_to_cuda
        )

        val_loader = torch.utils.data.DataLoader(
            val_dataset
            , batch_size=self.batch_size
        )

        return val_loader
        

if __name__ == "__main__":
    feature_extractor = lit_mnist_classifier.LitMNISTClassifier.load_from_checkpoint(
        checkpoint_path='logs_mnist_clf/version_1/checkpoints/epoch=14-avg_val_loss=0.0394-avg_val_acc=0.9876.ckpt'
    )
    feature_extractor = feature_extractor.float()

    model = LitDeepPermNet_v2(
        feature_extractor=feature_extractor
        , feature_size=10
        , head_size=10
        , loss_func='l1'
        , batch_size=16
        , lr=0.0003
        , mnist_path='mnist'
        , download=False
        , duplicates_multiplier=2
        , seed=42
        , disable_feature_extractor_training=True
        , use_binary_optimization_accuracy_metric=False
        , permutation_extractor_version='v5'
        , bottleneck_features_num=64
        , move_data_to_cuda=False
        , entropy_reg=1e-1
        , max_iter=1e2
        , tol=1e-6
        , log=False
        , verbose=False
        , log_interval=10
        , tau=1e3
    )
    # model.on_validation_model_eval()
    
    logger = pl.loggers.TensorBoardLogger(save_dir='./logs_deep_perm_net_v2_1', name='')
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        monitor='avg_val_loss'
        , save_top_k=1
        , mode='min'
        , filepath=str(Path(logger.log_dir, 'checkpoints', '{epoch}-{avg_val_loss:.4f}'))
    )
    trainer = pl.Trainer(
        max_epochs=100
        , checkpoint_callback=checkpoint_callback
        , logger=logger
        # , gradient_clip_val=0.5
        # , precision=16  # On GPU can be beneficial
        , track_grad_norm=2
        # , gpus=[0]
        # , fast_dev_run=True
    )
    trainer.fit(model)
