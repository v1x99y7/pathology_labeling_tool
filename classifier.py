import torch
from torch import nn
from torchvision import transforms, models

from PIL import Image

device = 'cuda' if torch.cuda.is_available() else 'cpu'

label_names = ['ADI', 'BACK', 'DEB', 'LYM', 'MUC', 'MUS', 'NORM', 'STR', 'TUM']

# transform
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# model
model = models.resnet50()
model.fc = nn.Sequential(
    nn.Linear(2048, 1024),
    nn.Dropout(0.5),
    nn.Linear(1024, 9)
)
model.load_state_dict(torch.load('resnet50_weights.pth'))
model = model.to(device)
model.eval()

# classify function
def classify(input_path):
    input = Image.open(input_path)
    input = transform(input).unsqueeze(0)
    input = input.to(device)

    output = model(input)

    _, index = torch.max(output, 1)
    result = label_names[index]

    return result