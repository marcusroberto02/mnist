from pytorch_lightning import LightningModule
from torch import nn, optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchmetrics.functional import accuracy
import torch
import wandb

class MyAwesomeModel(LightningModule):
    def __init__(self):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(784, 256),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(64, 10),
            nn.LogSoftmax(dim=1)
        )

        self.criterium = nn.CrossEntropyLoss()

    def forward(self, x):
        return self.classifier(x)
    
    def training_step(self, batch, batch_idx):
        data, target = batch
        preds = self(data)
        loss = self.criterium(preds,target)
        acc = (target == preds.argmax(dim=-1)).float().mean()
        self.log('train_loss', loss)
        self.log('train_acc', acc)
        self.logger.experiment.log({'logits': wandb.Histogram(preds.detach().numpy())})
        return loss
    
    def test_step(self, batch, batch_idx):
        loss, acc = self._shared_eval_step(batch, batch_idx)
        metrics = {"test_acc": acc, "test_loss": loss}
        self.log_dict(metrics)
        return metrics

    def _shared_eval_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.criterium(y_hat, y)
        acc = accuracy(y_hat, y, task="multiclass",num_classes=10)
        return loss, acc
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.criterium(y_hat, y)
        self.log("val_loss", loss)
    
    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=1e-2)

    def train_dataloader(self):
        trainset = torch.load("data/processed/train.pt")
        return DataLoader(trainset,batch_size=64,shuffle=True)

    def test_dataloader(self):
        testset = torch.load("data/processed/test.pt")
        return DataLoader(testset,batch_size=64,shuffle=True)

    def val_dataloader(self):
        valset = torch.load("data/processed/val.pt")
        print(len(valset))
        return DataLoader(valset,batch_size=64,shuffle=True)