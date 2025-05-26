import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from torch import nn
import numpy as np
import torch
import os
import sys
import random
from tqdm import tqdm as tqdm
from IPython import display

project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.append(project_root)

from models.vgg import VGG_A
from models.vgg import VGG_A_BatchNorm # you need to implement this network
from data.loaders import get_cifar_loader

# ## Constants (parameters) initialization
device_id = [0,1,2,3]
num_workers = 4
batch_size = 128

# add our package dir to path 
module_path = os.path.dirname(os.getcwd())
home_path = module_path
figures_path = os.path.join(home_path, 'reports', 'figures')
models_path = os.path.join(home_path, 'reports', 'models')

# Make sure you are using the right device.

# 修改设备选择的代码
if torch.cuda.is_available():
    device = torch.device("cuda")  # 使用默认的可用 GPU
    print(f"使用 GPU: {torch.cuda.get_device_name(0)}")
else:
    device = torch.device("cpu")
    print("使用 CPU")

# Initialize your data loader and
# make sure that dataloader works
# as expected by observing one
# sample from it.
train_loader = get_cifar_loader(train=True)
val_loader = get_cifar_loader(train=False)
for X,y in train_loader:
    ## --------------------
    # Add code as needed
    #
    #
    #
    #
    ## --------------------
    break



# This function is used to calculate the accuracy of model classification
def get_accuracy():
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data in val_loader:
            images, labels = data
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return 100 * correct / total

# Set a random seed to ensure reproducible results
def set_random_seeds(seed_value=0, device='cpu'):
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    random.seed(seed_value)
    if device != 'cpu': 
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


# We use this function to complete the entire
# training process. In order to plot the loss landscape,
# you need to record the loss value of each step.
# Of course, as before, you can test your model
# after drawing a training round and save the curve
# to observe the training
def train(model, optimizer, criterion, train_loader, val_loader, scheduler=None, epochs_n=100, best_model_path=None):
    model.to(device)
    learning_curve = [np.nan] * epochs_n
    train_accuracy_curve = [np.nan] * epochs_n
    val_accuracy_curve = [np.nan] * epochs_n
    max_val_accuracy = 0
    max_val_accuracy_epoch = 0

    batches_n = len(train_loader)
    losses_list = []
    grads = []
    for epoch in tqdm(range(epochs_n), unit='epoch'):
        if scheduler is not None:
            scheduler.step()
        model.train()

        loss_list = []  # use this to record the loss value of each step
        grad = []  # use this to record the loss gradient of each step
        learning_curve[epoch] = 0  # maintain this to plot the training curve

        for data in train_loader:
            x, y = data
            x = x.to(device)
            y = y.to(device)
            optimizer.zero_grad()
            prediction = model(x)
            loss = criterion(prediction, y)
            # You may need to record some variable values here
            # if you want to get loss gradient, use
            # grad = model.classifier[4].weight.grad.clone()
            ## --------------------
            # Add your code
            #
            #
            #
            #
            ## --------------------


            loss.backward()
            optimizer.step()

            loss_list.append(loss.item())
            # if you want to get loss gradient, use
            if model.classifier[4].weight.grad is not None:
                grad.append(model.classifier[4].weight.grad.clone().cpu().numpy())

            learning_curve[epoch] += loss.item()

        losses_list.append(loss_list)
        grads.append(grad)
        display.clear_output(wait=True)
        f, axes = plt.subplots(1, 2, figsize=(15, 3))

        learning_curve[epoch] /= batches_n
        axes[0].plot(learning_curve)

        # Test your model and save figure here (not required)
        # remember to use model.eval()
        ## --------------------
        # Add code as needed
        #
        #
        #
        #
        ## --------------------
        model.eval()
        train_accuracy = get_accuracy(model, train_loader)
        val_accuracy = get_accuracy(model, val_loader)
        train_accuracy_curve[epoch] = train_accuracy
        val_accuracy_curve[epoch] = val_accuracy
        axes[1].plot(train_accuracy_curve, label='Train Accuracy')
        axes[1].plot(val_accuracy_curve, label='Validation Accuracy')
        axes[1].legend()
        plt.show()
        plt.savefig(os.path.join(figures_path, f'epoch_{epoch}.png'))
        plt.close()


    return losses_list, grads


# Train your model
# feel free to modify
epo = 20
loss_save_path = './results/loss'
grad_save_path = './results/grads'

set_random_seeds(seed_value=2020, device=device)
model = VGG_A()
lr = 0.001
optimizer = torch.optim.Adam(model.parameters(), lr = lr)
criterion = nn.CrossEntropyLoss()
losses, grads = train(model, optimizer, criterion, train_loader, val_loader, epochs_n=epo)
flattened_losses = [item for sublist in losses for item in sublist]
np.savetxt(os.path.join(loss_save_path, 'loss.txt'), losses, fmt='%s', delimiter=' ')
flattened_grads = [item for sublist in grads for item in sublist]
np.savetxt(os.path.join(grad_save_path, 'grads.txt'), grads, fmt='%s', delimiter=' ')

# Maintain two lists: max_curve and min_curve,
# select the maximum value of loss in all models
# on the same step, add it to max_curve, and
# the minimum value to min_curve
min_curve = []
max_curve = []
## --------------------
# Add your code
#
#
#
#
## --------------------
num_steps = max([len(loss_list) for loss_list in losses])
for step in range(num_steps):
    step_losses = [loss_list[step] for loss_list in losses if step < len(loss_list)]
    min_curve.append(np.nanmin(step_losses))
    max_curve.append(np.nanmax(step_losses))

# Use this function to plot the final loss landscape,
# fill the area between the two curves can use plt.fill_between()
def plot_loss_landscape():
    ## --------------------
    # Add your code
    #
    #
    #
    #
    ## --------------------
    plt.figure(figsize=(10, 6))
    steps = np.arange(len(min_curve))
    plt.fill_between(steps, min_curve, max_curve, alpha=0.5)
    plt.xlabel('Training Steps')
    plt.ylabel('Loss')
    plt.title('Loss Landscape')
    plt.savefig(os.path.join(figures_path, 'loss_landscape.png'))
    plt.close()

plot_loss_landscape()