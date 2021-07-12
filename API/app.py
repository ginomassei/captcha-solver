import cv2
from flask import Flask, request, jsonify
from torchvision import transforms
from methods import clean_image, crop_digits
from inversed_labels import labels
import torch
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


app = Flask(__name__)

# Instanciate the model.
model = NeuralNetwork()

# Load the state dict of the model.
model_path = 'trainedmodel.pth'
model.load_state_dict(torch.load(model_path))

# Set the model into evaluation state.
model.eval()


@app.route('/ping')
def ping():
    return "Hello"


@app.route('/solve', methods=['POST'])
def solve():
    # Get the image from the request, and store it.
    file = request.files['captcha']
    # TODO: Implement a saver, for recolecting more images to train the net.
    file.save('image.jpeg')

    # With CV2 read the image stored, and clean it.
    image = cv2.imread('image.jpeg')

    image = clean_image(image)

    # Crop each digit into a digits array.
    digits = crop_digits(image)

    digits_tensors = []
    for digit in digits:
        # Transform each digit into a tensor.
        current = transforms.ToTensor()(digit)
        # Remove the first dimention.
        current = current.unsqueeze(0)

        digits_tensors.append(current)

    captcha = []
    for digit in digits_tensors:
        # Pass the digit to the model.
        predictions = model(digit)
        # Filter the predictions.
        predictions = torch.nn.Softmax(dim=1)(predictions)
        predicted_value = predictions.argmax(1).item()
        captcha.append(labels[predicted_value])

    body = {
        'captcha': ''.join([str(elem) for elem in captcha])
    }
    return jsonify(body)


if __name__ == '__main__':
    app.run(host='0.0.0.0')
