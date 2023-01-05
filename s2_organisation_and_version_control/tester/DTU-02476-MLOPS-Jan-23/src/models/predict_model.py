import click
import torch
from model import MyAwesomeModel
from torch import nn


@click.group()
def cli():
    pass


@click.command()
@click.argument("model_checkpoint")
def evaluate(model_checkpoint):
    print("Evaluating until hitting the ceiling")
    print(model_checkpoint)

    # TODO: Implement evaluation logic here
    model = MyAwesomeModel()
    state_dict = torch.load(model_checkpoint)
    model.load_state_dict(state_dict)

    data = torch.load("data/processed/processed_data.pth")

    test_set = data["test"]
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
        print(f"Validation accuracy: {accuracy.item()*100}%")

        # set model back to train mode
        model.train()


cli.add_command(evaluate)


if __name__ == "__main__":
    cli()
