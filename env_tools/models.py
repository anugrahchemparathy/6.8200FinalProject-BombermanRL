import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.categorical import Categorical

def init_params(m):
    classname = m.__class__.__name__
    if classname.find("Linear") != -1:
        m.weight.data.normal_(0, 1)
        m.weight.data *= 1 / torch.sqrt(m.weight.data.pow(2).sum(1, keepdim=True))
        if m.bias is not None:
            m.bias.data.fill_(0)

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

class ActorModel(nn.Module):
    
    def __init__(self, num_actions, device=None):
        super().__init__()
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.image_conv_actor = nn.Sequential(
            nn.Conv2d(1, 4, (2, 2)),
            nn.ReLU(),
            nn.MaxPool2d((2, 2)),
            nn.Conv2d(4, 8, (2, 2)),
            nn.ReLU(),
            nn.Conv2d(8, 16, (2, 2)),
            nn.ReLU()
        )
        self.actor = nn.Sequential(
            nn.Linear(576, 64),

        )
        self.apply(init_params)

    def forward(self, obs):
        conv_in = obs.unsqueeze(0)

        x = self.image_conv_actor(conv_in)
        embedding = x.reshape(x.shape[0], -1)

        x = self.actor(embedding)

        return x



if __name__ == "__main__":
    # Define the input tensor
    inputs = torch.randn(1, 3, 224, 224)

    model = Resnet18(output_size=6, device='cpu')
    
    print(model(inputs))