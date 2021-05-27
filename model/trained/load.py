import torch
from nnet import NeuralNet
from labels import labels
import functions
from skimage import io
from torchvision import transforms


if __name__ == '__main__':
    model = NeuralNet(36)
    model.load_state_dict(torch.load('captcha-recognizer.pth'))
    model.eval()

    
    image = io.imread('1-82480.jpeg', as_gray=True)
    
    image = torch.tensor(image)
    print(image.shape)
    
    exit()
    result = model.forward(image)
