import click
import matplotlib.pyplot as plt
import torch
from model import MyAwesomeModel
from torch import nn, optim


@click.group()
def cli():
    pass


@click.command()
@click.option("--lr", default=1e-3, help="learning rate to use for training")
def train(lr):
    print("Training day and night")
    print(lr)

    model = MyAwesomeModel()
    data = torch.load("data/processed/processed_data.pth")

    # train = TensorDataset(train_images, train_labels)

    train_set = data["train"]
    trainloader = torch.utils.data.DataLoader(train_set, batch_size=64, shuffle=True)

    criterion = nn.NLLLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr)

    epochs = 30
    loss_list = []
    for e in range(epochs):
        running_loss = 0
        for images, labels in trainloader:
            # Flatten MNIST images into a 784 long vector
            images = images.view(images.shape[0], -1)

            # Clear the gradients, do this because gradients are accumulated
            optimizer.zero_grad()

            # Forward pass, then backward pass, then update weights
            print(images.shape)
            output = model(images.float())
            # TODO: Training pass
            loss = criterion(output, labels)

            loss.backward()
            running_loss += loss.item()

            optimizer.step()
        else:
            loss_list.append(running_loss / len(trainloader))
            # print(f"Training loss_epoch_{e}: {running_loss/len(trainloader)}")
    torch.save(model.state_dict(), "models/trained_model.pth")

    plt.plot(loss_list, marker="o")
    plt.savefig("reports/figures/train_loss.png")


cli.add_command(train)

if __name__ == "__main__":
    cli()
