import torch
import click

from src.data.mnist import mnist


@click.group()
def cli():
    pass


@click.command()
@click.argument("model_checkpoint")
def evaluate(model_checkpoint):
    print("Evaluating until hitting the ceiling")
    print(model_checkpoint)

    # TODO: Implement evaluation logic here
    model = torch.load(model_checkpoint)

    _, testset = mnist()
    testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=True)
    accuracy = 0
    with torch.no_grad():
        model.eval()
        for images, labels in testloader:
            ps = torch.exp(model(images))

            _, top_class = ps.topk(1, dim=1)

            equals = top_class == labels.view(*top_class.shape)

            accuracy += torch.mean(equals.type(torch.FloatTensor))

    accuracy /= len(testloader)

    print(f"Accuracy: {accuracy.item()*100}%")
    model.train()


cli.add_command(evaluate)


if __name__ == "__main__":
    cli()
