import torch
import torch.nn as nn

class Resnet18(nn.Module):
    def __init__(self, output_size, device=None):
        super(Resnet18, self).__init__()
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.resnet = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=False)
        self.resnet.fc = torch.nn.Linear(512, output_size)
        self.resnet.to(device)
        print("Initialized Resnet18 on", device)
    
    def forward(self, x):
        return self.resnet(x)

if __name__ == "__main__":
    # Define the input tensor
    inputs = torch.randn(1, 3, 224, 224)

    model = Resnet18(output_size=6, device='cpu')
    
    print(model(inputs))