import torch

if __name__ == "__main__":
    # Define the input tensor
    inputs = torch.randn(1, 3, 224, 224)

    model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=False)
    print(model)
    outputs = model(inputs)
    print(outputs)
    print(outputs.shape)
    # By default outputs 1000 dimensional vector
    # can change to size of our action space?