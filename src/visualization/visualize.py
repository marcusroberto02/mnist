import sys

import numpy as np

import torch
import click

from src.data.mnist import mnist

from sklearn.manifold import TSNE

import matplotlib.pyplot as plt

# give path to model definition
sys.path.append("src/models/")


@click.group()
def cli():
    pass


@click.command()
@click.argument("model_checkpoint")
def visualize(model_checkpoint):
    print("Evaluating until hitting the ceiling")
    print(model_checkpoint)

    # TODO: Implement evaluation logic here
    model = torch.load(model_checkpoint)

    _, testset = mnist()
    testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=True)
    features = np.array([]).reshape(0, 128)
    with torch.no_grad():
        model.eval()
        for images, labels in testloader:
            _ = model(images)
            ft = (features, model.features_fc2.numpy())
            features = np.concatenate(ft, axis=0)
    model.train()

    # embed to 2d space
    X_embedded = TSNE(
        n_components=2, learning_rate="auto", init="random", perplexity=3
    ).fit_transform(features)

    plt.scatter(X_embedded[:, 0], X_embedded[:, 1])
    plt.savefig("reports/figures/2d_embeddings.png")


cli.add_command(visualize)


if __name__ == "__main__":
    cli()
