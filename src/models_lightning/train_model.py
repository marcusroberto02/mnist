# import torch
import click

from model import MyAwesomeModel
# from tqdm import tqdm

# from torch import nn, optim

# import matplotlib.pyplot as plt

from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import WandbLogger


@click.group()
def cli():
    pass


@click.command()
@click.option("--lr", default=1e-3, help="learning rate to use for training")
@click.option("--epochs", default=10)
@click.option("--model_checkpoint", default="models/model.pt")
def train(lr, epochs, model_checkpoint):
    print("Training day and night")
    print(lr)

    # TODO: Implement training loop here
    model = MyAwesomeModel()

    checkpoint_callback = ModelCheckpoint(
        dirpath="./models/lightning/", monitor="val_loss", mode="min"
    )

    early_stopping_callback = EarlyStopping(
        monitor="val_loss", patience=3, verbose=True, mode="min"
    )

    trainer = Trainer(
        max_epochs=10,
        callbacks=[checkpoint_callback, early_stopping_callback],
        logger=WandbLogger(project="mnist-project"),
    )
    trainer.fit(
        model,
        train_dataloaders=model.train_dataloader(),
        val_dataloaders=model.val_dataloader(),
    )
    trainer.test(model, dataloaders=model.test_dataloader())

    # plt.plot(train_losses, label="Training loss")
    # plt.legend(frameon=True)
    # plt.savefig("reports/figures/training_curve.png")
    # torch.save(model, model_checkpoint)


cli.add_command(train)


if __name__ == "__main__":
    cli()
