import torch
from torch import nn, optim
from torchvision import transforms, datasets, models
from torch.utils.data import DataLoader, Subset

from sklearn.model_selection import train_test_split

import os
import time
import copy
import numpy as np
from tqdm import tqdm

# print(torch.cuda.is_available())
# print(torch.__version__)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Using {device} device')
print('-' * 30)

# dataset
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

NCT_CRC_HE_100K = datasets.ImageFolder('D:/Open_data/NCT-CRC-HE-100K', transform=transform)
# print(NCT_CRC_HE_100K.class_to_idx)

train_idx, val_idx = train_test_split(np.arange(len(NCT_CRC_HE_100K)),
                                             test_size=0.2,
                                             random_state=0,
                                             shuffle=True,
                                             stratify=NCT_CRC_HE_100K.targets)
train_dataset = Subset(NCT_CRC_HE_100K, train_idx)
val_dataset = Subset(NCT_CRC_HE_100K, val_idx)

# dataloader
train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=64, shuffle=False)

# model
model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)

for name, param in model.named_parameters():
    param.requires_grad = False
    
model.fc = nn.Sequential(
    nn.Linear(2048, 1024),
    nn.Dropout(0.5),
    nn.Linear(1024, 9)
)

# loss
loss_fn = nn.CrossEntropyLoss()

# optimizer
params_to_update = []
for name, param in model.named_parameters():
    if param.requires_grad == True:
        params_to_update.append(param)
            
optimizer = optim.Adam(params_to_update, lr=0.0001)

# training
def train_loop(train_dataloader, val_dataloader, model, loss_fn, optimizer, num_epochs):
    since = time.time()
    
    train_acc_history, val_acc_history = [], []
    best_train_acc, best_val_acc = 0, 0
    best_model_wts = copy.deepcopy(model.state_dict())
    
    model = model.to(device)
    
    for epoch in range(num_epochs):
        print(f'Epoch {epoch + 1}/{num_epochs}')
        print('-' * 30)
        time.sleep(1)
        
        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
                dataloader = train_dataloader
            else:
                model.eval()
                dataloader = val_dataloader
        
            running_loss = 0
            running_corrects = 0
            
            # Iterate over data
            for inputs, labels in tqdm(dataloader):
                inputs, labels = inputs.to(device), labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()
            
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    loss = loss_fn(outputs, labels)
                    _, preds = torch.max(outputs, 1)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.shape[0]
                running_corrects += torch.sum(preds == labels)

            epoch_loss = running_loss / len(dataloader.dataset)
            epoch_acc = running_corrects / len(dataloader.dataset)
            print('{} loss: {:.4f} acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

            if phase == 'train' and epoch_acc > best_train_acc:
                best_train_acc = epoch_acc
            if phase == 'train':
                train_acc_history.append(epoch_acc)

            if phase == 'val' and epoch_acc > best_val_acc:
                best_val_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
            if phase == 'val':
                val_acc_history.append(epoch_acc)
                
            time.sleep(1)

        print('-' * 30)
    
    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best train acc: {:.4f}'.format(best_train_acc))
    print('Best val acc: {:.4f}'.format(best_val_acc))
    print('-' * 30)
    print(f'Train acc history: {[round(a.item(), 4) for a in train_acc_history]}')
    print(f'Val acc history: {[round(a.item(), 4) for a in val_acc_history]}')
    print('-' * 30)

    # load best model weights
    model.load_state_dict(best_model_wts)

    return model, train_acc_history, val_acc_history

print('training')
print('-' * 60)

model_best, train_acc_history, val_acc_history = train_loop(train_dataloader, val_dataloader, model, loss_fn, optimizer, 20)
torch.save(model_best.state_dict(), 'resnet50_weights.pth')

# testing
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
CRC_VAL_HE_7K = datasets.ImageFolder('D:/Open_data/CRC-VAL-HE-7K', transform=transform)
test_dataloader = DataLoader(CRC_VAL_HE_7K, batch_size=64, shuffle=False)

model_load = models.resnet50()
model_load.fc = nn.Sequential(
    nn.Linear(2048, 1024),
    nn.Dropout(0.5),
    nn.Linear(1024, 9)
)
model_load.load_state_dict(torch.load('resnet50_weights.pth'))
model_load = model_load.to(device)
model_load.eval()

running_loss = 0.0
running_corrects = 0

print('testing')
print('-' * 60)

for inputs, labels in tqdm(test_dataloader):
    inputs, labels = inputs.to(device), labels.to(device)
                
    with torch.set_grad_enabled(False):
        outputs = model_load(inputs)
        loss = loss_fn(outputs, labels)
        _, preds = torch.max(outputs, 1)
                    
        # statistics
        running_loss += loss.item() * inputs.shape[0]
        running_corrects += torch.sum(preds == labels)
                
epoch_loss = running_loss / len(test_dataloader.dataset)
epoch_acc = running_corrects / len(test_dataloader.dataset)
            
time.sleep(1)
    
print('test loss: {:.4f} acc: {:.4f}'.format(epoch_loss, epoch_acc))