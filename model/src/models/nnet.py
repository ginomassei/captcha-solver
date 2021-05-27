import torch
import torch.nn.functional as F
import torch.optim as optim

from torch import nn
from torchvision import transforms
from digits import DigitsDataset


class NeuralNet(nn.Module):
    def __init__(self, num_classes, input_channels=1):
        super(NeuralNet, self).__init__()

        self.conv1 = nn.Conv2d(input_channels, out_channels=32, kernel_size=(3, 3))
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3, 3))

        self.pool = nn.MaxPool2d(kernel_size=3, stride=(2, 2))
        self.fc1 = nn.Linear(6912, num_classes)

    
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)

        x = F.relu(self.conv2(x))

        x = x.reshape(x.shape[0], -1)
        x = self.fc1(x)

        return x
    

def check_accuracy(loader, model):
    num_correct = 0
    num_samples = 0
    model.eval()

    with torch.no_grad():
        for x, y in loader:
            x = x.float().to(device=device)
            y = y.to(device=device)

            scores = model(x)
            scores = scores.double()
            
            _, predictions = scores.max(1)
            num_correct += (predictions == y).sum()
            num_samples += predictions.size(0)
        
        print(f'Got {num_correct} of {num_samples} --> {float(num_correct)/float(num_samples)*100:.2f}%')

    model.train()


if __name__ == "__main__":
    # Instanciate the dataset object.
    dataset = DigitsDataset(
        '../data/landmarks.csv', 
        '../data/digits/', 
        transform=transforms.ToTensor()
    )
    
    # Divide the dataset into train and test set.
    train_set, test_set = torch.utils.data.random_split(dataset, [3659, 1000])
    
    # Hyperparameters
    output_size = 36
    learning_rate = 0.01 # Before --> 0.01
    batch_size = 32 # Before --> 32
    num_epochs = 100

    # Creating the loaders.
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=True)
    
    # Check for the device.
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Initialize the network.
    model = NeuralNet(output_size)
    model.float()
    model.to(device)
    
    # Loss and optimizer.
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)

    # Train the network
    for epoch in range(num_epochs):
        print(f'Epoch: {epoch}')
        for batch_index, (data, targets) in enumerate(train_loader):
            # Get data to the device.
            data = data.float()
            data.to(device=device)

            # Labels for the data comparison.
            targets.to(device=device)
            
            # Forward.
            scores = model(data)
            loss = criterion(scores, targets)

            # Backprop.
            optimizer.zero_grad()  # Set all the gradients to zero.
            loss.backward()

            # Do an optimizer step.
            optimizer.step()

    print('\nTrained accuracy: ')
    check_accuracy(train_loader, model)
    check_accuracy(test_loader, model)

    torch.save(model.state_dict(), 'captcha-recognizer.pth')
