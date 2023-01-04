import argparse
import sys

import torch
import click

from data import mnist
from model import MyAwesomeModel

from torch import optim
from torch import nn

import matplotlib.pyplot as plt

@click.group()
def cli():
    pass


@click.command()
@click.option("--lr", default=1e-3, help='learning rate to use for training')
def train(lr):
    print("Training day and night")
    print(lr)

    model = MyAwesomeModel()
    train_set, _ = mnist()
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
            output = model(images.float())
            # TODO: Training pass
            loss = criterion(output, labels)
            
            loss.backward()
            running_loss += loss.item()
            
            optimizer.step()
        else:
            loss_list.append(running_loss/len(trainloader))
            # print(f"Training loss_epoch_{e}: {running_loss/len(trainloader)}")
    torch.save(model.state_dict(), 'trained_model.pt')
    plt.plot(loss_list, marker='o')
    plt.show()

@click.command()
@click.argument("model_checkpoint")
def evaluate(model_checkpoint):
    print("Evaluating until hitting the ceiling")
    print(model_checkpoint)

    # TODO: Implement evaluation logic here
    model = MyAwesomeModel()
    model.load_state_dict(torch.load(model_checkpoint))
    _, test_set = mnist()
    testloader = torch.utils.data.DataLoader(test_set, batch_size=64, shuffle=True)

    criterion = nn.NLLLoss()

    with torch.no_grad():
        val_running_loss = 0
        # set model to evaluation mode
        model.eval()

        # validation pass here
        for images, labels in testloader:
    
            log_ps = model(images.float())
            loss = criterion(log_ps, labels)

            val_running_loss += loss.item()

        ps = torch.exp(model(images.float()))
        _, top_class = ps.topk(1, dim=1)
        equals = top_class == labels.view(*top_class.shape)
        accuracy = torch.mean(equals.type(torch.FloatTensor))
        print(f'Validation accuracy: {accuracy.item()*100}%')
        
        # set model back to train mode
        model.train()

cli.add_command(train)
cli.add_command(evaluate)


if __name__ == "__main__":
    cli()
