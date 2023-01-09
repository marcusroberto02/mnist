import torch
import click

from src.data.mnist import mnist
from model import MyAwesomeModel
from tqdm import tqdm

from torch import nn, optim
import torch.nn.functional as F
from torchvision import datasets, transforms

import matplotlib.pyplot as plt


@click.group()
def cli():
    pass


@click.command()
@click.option("--lr", default=1e-3, help='learning rate to use for training')
@click.option("--epochs",default = 10)
@click.option("--model_checkpoint",default = "models/model.pt")
def train(lr,epochs,model_checkpoint):
    print("Training day and night")
    print(lr)

    # TODO: Implement training loop here
    model = MyAwesomeModel()
    trainset, _ = mnist()
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)
    
    criterion = nn.NLLLoss()

    optimizer = optim.Adam(model.parameters(), lr=0.003)

    epochs = epochs

    train_losses = []
    for e in tqdm(range(epochs)):
        running_loss = 0
        for images, labels in trainloader:
            
            optimizer.zero_grad()
            
            log_ps = model(images)
            loss = criterion(log_ps, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
        else:
            train_losses.append(running_loss / len(trainloader))
    
    plt.plot(train_losses,label="Training loss")
    plt.legend(frameon=True)
    plt.savefig("reports/figures/training_curve.png")
    torch.save(model,model_checkpoint)

cli.add_command(train)


if __name__ == "__main__":
    cli()


    
    
    
    