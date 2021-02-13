%reload_ext autoreload
%autoreload 2
%matplotlib inline

from IPython.core.interactiveshell import InteractiveShell
import seaborn as sns
# PyTorch
from torchvision import transforms, datasets, models
import torch
from torch import optim, cuda
from torch.utils.data import DataLoader, sampler, random_split
import torch.nn as nn
import torch.nn.functional as F
from torchvision.datasets import ImageFolder
from torch.utils.data.dataloader import DataLoader
from torchsummary import summary

import warnings
warnings.filterwarnings('ignore', category=FutureWarning)

# Data science tools
import numpy as np
import pandas as pd
import os

# Image manipulations
from PIL import Image
# Useful for examining network
from torchsummary import summary
# Timing utility
from timeit import default_timer as timer
import pickle

# Visualizations
import matplotlib.pyplot as plt
%matplotlib inline
plt.rcParams['font.size'] = 20
plt.rcParams['font.family'] = ['sans-serif']
plt.rcParams['font.sans-serif'] = ['SimHei']

# Printing out all outputs
InteractiveShell.ast_node_interactivity = 'all'

data_dir  = './data/'

# print(os.listdir(data_dir))
classes = os.listdir(data_dir + '/train')
# print(classes)
n_classes=len(classes)

# checkpoint_path = '/content/drive/MyDrive/Github/Chinese Handwriting Recognition/weights/resnet50-transfer-4.pth'
best_model_path = './weights/resnet50-transfer-4-bestmodel.pth'
# Change to fit hardware
batch_size = 64

# Whether to train on a gpu
train_on_gpu = cuda.is_available()
print(f'Train on gpu: {train_on_gpu}')

# Number of gpus
multi_gpu = False
if train_on_gpu:
    gpu_count = cuda.device_count()
    print(f'{gpu_count} gpus detected.')
    if gpu_count > 1:
        multi_gpu = True
    else:
        multi_gpu = False

# Image transformations
image_transforms = {
    # Train uses data augmentation
    'train':
    transforms.Compose([
        transforms.Resize(size=(224,224)),
        # transforms.RandomResizedCrop(size=256, scale=(0.8, 1.0)),
        # transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(contrast=[5,5]),
        # transforms.RandomHorizontalFlip(),
        # transforms.CenterCrop(size=224),  # Image net standards
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])  # Imagenet standards
    ]),
    # Validation does not use augmentation
    'val':
    transforms.Compose([
        transforms.Resize(size=(224,224)),
        # transforms.Resize(size=256),
        # transforms.CenterCrop(size=224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    # Test does not use augmentation
    'test':
    transforms.Compose([
        transforms.Resize(size=(224,224)),
        # transforms.Resize(size=256),
        # transforms.CenterCrop(size=224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}


def imshow(image):
    """Display image"""
    plt.figure(figsize=(6, 6))
    plt.imshow(image)
    plt.axis('off')
    plt.show()


def imshow_tensor(image, ax=None, title=None):
    """Imshow for Tensor."""

    if ax is None:
        fig, ax = plt.subplots()

    # Set the color channel as the third dimension
    image = image.numpy().transpose((1, 2, 0))

    # Reverse the preprocessing steps
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = std * image + mean

    # Clip the image pixel values
    image = np.clip(image, 0, 1)

    ax.imshow(image)
    plt.axis('off')

    return ax, image


dataset = ImageFolder(data_dir+'/train', transform=image_transforms['train'])
# len(dataset)


random_seed = 42
torch.manual_seed(random_seed);

val_size = int(len(dataset) * 0.2)
train_size = len(dataset) - val_size

train_ds, val_ds = torch.utils.data.random_split(dataset,[train_size, val_size])
len(train_ds), len(val_ds)


batch_size=32

train_dl = DataLoader(train_ds, batch_size, shuffle=True, num_workers=4, pin_memory=True)
val_dl = DataLoader(val_ds, batch_size*2, num_workers=4, pin_memory=True)

def get_pretrained_model(model_name):
    """Retrieve a pre-trained model from torchvision

    Params
    -------
        model_name (str): name of the model (currently only accepts vgg16 and resnet50)

    Return
    --------
        model (PyTorch model): cnn

    """

    if model_name == 'resnet50':
        model = models.resnet50(pretrained=True)

        # Make sure to set parameters as not trainable
        for param in model.parameters():
            param.requires_grad = False

        n_inputs = model.fc.in_features
        # model.fc = nn.Sequential(
        #     nn.Linear(n_inputs, 256), nn.ReLU(), nn.Dropout(0.2),
        #     nn.Linear(256, n_classes), nn.LogSoftmax(dim=1))
        model.fc = nn.Linear(n_inputs, n_classes)

    # Move to gpu and parallelize
    if train_on_gpu:
        model = model.to('cuda')

    if multi_gpu:
        model = nn.DataParallel(model)

    return model


model = get_pretrained_model('resnet50')
# if multi_gpu:
#     summary(
#         model.module,
#         input_size=(3, 224, 224),
#         batch_size=batch_size,
#         device='cuda')
# else:
#     summary(
#         model, input_size=(3, 224, 224), batch_size=batch_size, device='cuda')


model.class_to_idx = dataset.class_to_idx
model.idx_to_class = {
    idx: class_
    for class_, idx in model.class_to_idx.items()
}

list(model.idx_to_class.items())[:10]


#define loss function and optimizer
criterion = nn.CrossEntropyLoss()
# optimizer = optim.Adam(model.parameters(), lr=0.001)

optimizer = optim.Adam(
    [
        {"params": model.conv1.parameters(), "lr": 1e-5},
        {"params": model.bn1.parameters(), "lr": 2e-4},
        {"params": model.layer1.parameters(), "lr": 3e-4},
        {"params": model.layer2.parameters(), "lr": 4e-4},
        {"params": model.layer3.parameters(), "lr": 5e-4},
        {"params": model.layer4.parameters(), "lr": 6e-4},
        {"params": model.fc.parameters(), "lr": 1e-3},
    ],
    lr=1e-3,
)


def save_checkpoint(state, is_best, checkpoint_path, best_model_path):
    """
    state: checkpoint we want to save
    is_best: is this the best checkpoint; min validation loss
    checkpoint_path: path to save checkpoint
    best_model_path: path to save best model
    """
    # save checkpoint data to the path given, checkpoint_path
    torch.save(state, checkpoint_path)
    # if it is a best model, min validation loss
    if is_best:
        torch.save(state, best_model_path)


def load_checkpoint(checkpoint_fpath, optimizer):
    """
    checkpoint_fpath: path to save checkpoint
    optimizer: optimizer we defined in previous training
    """

    # Load in checkpoint
    checkpoint = torch.load(checkpoint_fpath,
                            # Load from CPU, uncomment if training on GPU
                            map_location=torch.device('cpu'))

    model = models.resnet50(pretrained=True)
    model.fc = checkpoint['fc']

    # Load in the state dict
    model.load_state_dict(checkpoint['state_dict'])

    # Move to gpu
    if multi_gpu:
        model = nn.DataParallel(model)

    if train_on_gpu:
        model = model.to('cuda')

    # Model basics
    model.class_to_idx = checkpoint['class_to_idx']
    model.idx_to_class = checkpoint['idx_to_class']
    epoch = checkpoint['epoch']

    # Optimizer
    optimizer = optim.Adam(
          [
            {"params": model.conv1.parameters(), "lr": 1e-5},
            {"params": model.bn1.parameters(), "lr": 2e-4},
            {"params": model.layer1.parameters(), "lr": 3e-4},
            {"params": model.layer2.parameters(), "lr": 4e-4},
            {"params": model.layer3.parameters(), "lr": 5e-4},
            {"params": model.layer4.parameters(), "lr": 6e-4},
            {"params": model.fc.parameters(), "lr": 1e-3},
        ],
        lr=1e-3,
    )
    optimizer.load_state_dict(checkpoint['optimizer'])

    # History
    history = checkpoint['history']
    valid_loss_min = checkpoint['valid_loss_min']
    valid_best_acc = checkpoint['valid_best_acc']

    return model, optimizer, history, valid_loss_min, valid_best_acc, epoch

    from tqdm.auto import tqdm

def train(model,
          criterion,
          optimizer,
          train_loader,
          valid_loader,
          checkpoint_path,
          best_model_path,
          freeze_layers=True,
          max_epochs_stop=3,
          n_epochs=20,
          print_every=2):
    """Train a PyTorch Model

    Params
    --------
        model (PyTorch model): cnn to train
        criterion (PyTorch loss): objective to minimize
        optimizer (PyTorch optimizier): optimizer to compute gradients of model parameters
        train_loader (PyTorch dataloader): training dataloader to iterate through
        valid_loader (PyTorch dataloader): validation dataloader used for early stopping
        checkpoint_path: path to save checkpoint
        best_model_path: path to save best model
        freeze_layers: true => only train fc layers, false => train all layers
        max_epochs_stop (int): maximum number of epochs with no improvement in validation loss for early stopping
        n_epochs (int): maximum number of training epochs
        print_every (int): frequency of epochs to print training stats

    Returns
    --------
        model (PyTorch model): trained cnn with best weights
        history (DataFrame): history of train and validation loss and accuracy
    """

    # Early stopping intialization
    epochs_no_improve = 0
    valid_loss_min = np.Inf

    valid_best_acc = 0
    history = []

    # Load the best state dict
    try:
      model, optimizer, history, valid_loss_min, valid_best_acc, epoch = load_checkpoint(checkpoint_fpath=checkpoint_path,optimizer=optimizer)
    except:
      print("Start from scratch")

    # Parameters set as not trainable in get_pre_trained_model, set to trainable here
    if freeze_layers==False:
      for param in model.parameters():
          param.requires_grad = True

    # Number of epochs already trained (if using loaded in model weights)
    try:
        print(f'Model has been trained for: {epoch} epochs.\n')
    except:
        print(f'Starting Training from Scratch.\n')

    overall_start = timer()

    # Main loop
    for epoch in range(n_epochs):

        # keep track of training and validation loss each epoch
        train_loss = 0.0
        valid_loss = 0.0

        train_acc = 0
        valid_acc = 0

        # Set to training
        model.train()
        start = timer()

        # Training loop
        for ii, (data, target) in enumerate(tqdm(train_loader)):
            # Tensors to gpu
            if train_on_gpu:
                data, target = data.cuda(), target.cuda()

            # Clear gradients
            optimizer.zero_grad()
            # Predicted outputs
            output = model(data)

            # Loss and backpropagation of gradients
            loss = criterion(output, target)
            loss.backward()

            # Update the parameters
            optimizer.step()

            ## record the average training loss, using something like
            ## train_loss = train_loss + ((1 / (ii + 1)) * (loss.data - train_loss))
            # train_loss = train_loss + ((1 / (ii + 1)) * (loss.data - train_loss))

            # Track train loss by multiplying average loss by number of examples in batch
            train_loss += loss.item() * data.size(0)

            # Calculate accuracy by finding max log probability
            _, pred = torch.max(output, dim=1)
            correct_tensor = pred.eq(target.data.view_as(pred))
            # Need to convert correct tensor from int to float to average
            accuracy = torch.mean(correct_tensor.type(torch.FloatTensor))
            # Multiply average accuracy times the number of examples in batch
            train_acc += accuracy.item() * data.size(0)

            # Track training progress
            print(
                f'Epoch: {epoch}\t{100 * (ii + 1) / len(train_loader):.2f}% complete. {timer() - start:.2f} seconds elapsed in epoch.',
                end='\r')

        # After training loops ends, start validation
        else:

            # Don't need to keep track of gradients
            with torch.no_grad():
                # Set to evaluation mode
                model.eval()

                # Validation loop
                for data, target in tqdm(valid_loader):
                    # Tensors to gpu
                    if train_on_gpu:
                        data, target = data.cuda(), target.cuda()

                    # Forward pass
                    output = model(data)

                    # Validation loss
                    loss = criterion(output, target)


                    # # update average validation loss
                    # valid_loss = valid_loss + ((1 / (ii + 1)) * (loss.data - valid_loss))
                    # Multiply average loss times the number of examples in batch
                    valid_loss += loss.item() * data.size(0)

                    # Calculate validation accuracy
                    _, pred = torch.max(output, dim=1)
                    correct_tensor = pred.eq(target.data.view_as(pred))
                    accuracy = torch.mean(
                    correct_tensor.type(torch.FloatTensor))
                    # Multiply average accuracy times the number of examples
                    valid_acc += accuracy.item() * data.size(0)

                # Calculate average losses
                train_loss = train_loss / len(train_loader.dataset)
                valid_loss = valid_loss / len(valid_loader.dataset)

                # Calculate average accuracy
                train_acc = train_acc / len(train_loader.dataset)
                valid_acc = valid_acc / len(valid_loader.dataset)

                history.append([train_loss, valid_loss, train_acc, valid_acc])

                # Print training and validation results
                if (epoch + 1) % print_every == 0:
                    print(
                        f'\nEpoch: {epoch} \tTraining Loss: {train_loss:.4f} \tValidation Loss: {valid_loss:.4f}'
                    )
                    print(
                        f'\t\tTraining Accuracy: {100 * train_acc:.2f}%\t Validation Accuracy: {100 * valid_acc:.2f}%'
                    )

                # Save the model if validation loss decreases
                if valid_loss < valid_loss_min:
                    # Attach the optimizer
                    model.optimizer = optimizer

                    # Track improvement
                    epochs_no_improve = 0
                    valid_loss_min = valid_loss
                    valid_best_acc = valid_acc
                    best_epoch = epoch

                    # create checkpoint variable and add important data
                    checkpoint = {
                        'epoch': epoch + 1,
                        'valid_loss_min': valid_loss,
                        'valid_best_acc': valid_acc,
                        'state_dict': model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'class_to_idx': model.class_to_idx,
                        'idx_to_class': model.idx_to_class,
                        'history': history,
                        'fc': model.fc,
                    }

                    # Save model
                    # torch.save(model.state_dict(), save_file_name)
                    save_checkpoint(checkpoint, True, checkpoint_path, best_model_path)

                # Otherwise increment count of epochs with no improvement
                else:
                    epochs_no_improve += 1
                    # Trigger early stopping
                    if epochs_no_improve >= max_epochs_stop:
                        print(
                            f'\nEarly Stopping! Total epochs: {epoch}. Best epoch: {best_epoch} with loss: {valid_loss_min:.2f} and acc: {100 * valid_acc:.2f}%'
                        )
                        total_time = timer() - overall_start
                        print(
                            f'{total_time:.2f} total seconds elapsed. {total_time / (epoch+1):.2f} seconds per epoch.'
                        )

                        # create checkpoint variable and add important data
                        checkpoint = {
                          'epoch': epoch + 1,
                          'valid_loss_min': valid_loss,
                          'valid_best_acc': valid_acc,
                          'state_dict': model.state_dict(),
                          'optimizer': optimizer.state_dict(),
                          'class_to_idx': model.class_to_idx,
                          'idx_to_class': model.idx_to_class,
                          'history': history,
                          'fc': model.fc,
                        }

                        # save checkpoint
                        save_checkpoint(checkpoint, False, checkpoint_path, best_model_path)


train(
    model,
    criterion,
    optimizer,
    train_dl,
    val_dl,
    checkpoint_path,
    best_model_path,
    freeze_layers=False,
    max_epochs_stop=5,
    n_epochs=30,
    print_every=1)
