import torch
from torchvision import transforms
from PIL import Image
from model.src.models.Network import NeuralNetwork
from inversed_labels import labels


if __name__ == '__main__':
    # Instanciate the model.
    model = NeuralNetwork()

    # Load the state dict of the model.
    model_path = 'trainedmodel.pth'
    model.load_state_dict(torch.load(model_path))

    # Set the model into evaluation state.
    model.eval()

    # Load an image.
    image_path = '../data/digits/5-2333.jpeg'
    X = Image.open(image_path).convert('L')

    # Transform the ndarray into a tensor.
    X = transforms.ToTensor()(X)

    # Remove the first dimention.
    X = X.unsqueeze(0)

    # Pass the image to the model.
    predictions = model(X)

    predictions = torch.nn.Softmax(dim=1)(predictions)
    predicted_value = predictions.argmax(1).item()
    print('The model predicted: ')
    print(labels[predicted_value])
