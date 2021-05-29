from torch import nn


class NeuralNetwork(nn.Module):
    """
        Neural networ class. Contains a Stack with the different layers to be used.
        The first layer size is fixed to the size of a digit image.
        The output is fixed to 36. 26 letters and 10 digits.
    """
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Conv2d(1, 128, (3, 3)),
            nn.ReLU(),
            nn.AvgPool2d((3, 3)),
            nn.Conv2d(128, 32, (3, 3)),
            nn.ReLU(),
            nn.Conv2d(32, 32, (3, 3)),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(320, 36),
        )

    def forward(self, x):
        x = self.linear_relu_stack(x)  # Pass the image into the ordered container of modules
        return x
