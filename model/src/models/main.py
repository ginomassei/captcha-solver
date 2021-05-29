import torch
from Network import NeuralNetwork
from Digits import DigitsDataset
from torchvision import transforms
from torch import nn

# Hyperparameters
learning_rate = 0.1
batch_size = 128
num_epochs = 100


def main():
    # Detect the device being used.
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Using {device}.\n')

    print(f'Current hyperparamters:\n'
          f'Learning rate: {learning_rate}\n'
          f'Batch Size: {batch_size}\n'
          f'Epochs: {num_epochs}\n')
    print('-' * 80 + '\n')

    # Instanciate the model.
    model = NeuralNetwork()
    model.to(device)  # Pass the model to the device.

    # Instanciate the dataset object.
    dataset = DigitsDataset(
        '../../data/landmarks.csv',
        '../../data/digits/',
        transform=transforms.ToTensor()
    )

    # Divide the dataset into train and test set.
    train_set, test_set = torch.utils.data.random_split(dataset, [3659, 1000])

    # Creating the loaders.
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=True)

    # Initialize the loss function
    loss_fn = nn.CrossEntropyLoss()

    # Initialize the optimizer, i'll use SGD since i know how it works.
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

    # Handle the number of epochs.
    for epoch in range(num_epochs):
        print(f"Epoch {epoch + 1}\n-------------------------------")
        training_loop(train_loader, model, loss_fn, optimizer)
        testing_loop(test_loader, model, loss_fn)

    # Save the model.
    model_path = '../../trained/trainedmodel.pth'
    torch.save(model.state_dict(), model_path)
    print(f"Done! Model saved as: {model_path}")


def training_loop(dataloader, model, loss_function, optimizer):
    """ Iterate over the training dataset and try to converge to optimal parameters. """
    size = len(dataloader.dataset)

    # The dataloader handles the batch size.
    for batch, (X, y) in enumerate(dataloader):
        # Compute prediction, passing to the model an image.
        prediction = model(X)

        # Compute the loss, bu passing the data to the loss function.
        loss = loss_function(prediction, y)

        # Use backpropagation to update the weights.
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(X)
            # This format have been taken from a pytorch example.
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


def testing_loop(dataloader, model, loss_function):
    """ Iterate over the test dataset to check if model performance is improving. """
    size = len(dataloader.dataset)
    test_loss, correct = 0, 0

    with torch.no_grad():
        for X, y in dataloader:
            pred = model(X)
            test_loss += loss_function(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    test_loss /= size
    correct /= size
    print(f"Test Error: \nAccuracy: {(100 * correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")


if __name__ == '__main__':
    main()
