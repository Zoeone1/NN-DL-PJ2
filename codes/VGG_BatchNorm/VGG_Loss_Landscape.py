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

from models.vgg import VGG_A, VGG_A_BatchNorm, VGG_A_Dropout, VGG_A_Light
from data.loaders import get_cifar_loader


# ## Constants (parameters) initialization
device_id = [0, 1, 2, 3]
num_workers = 4
batch_size = 128

# add our package dir to path
module_path = os.path.dirname(os.getcwd())
home_path = module_path
figures_path = os.path.join(home_path, 'reports', 'figures')
models_path = os.path.join(home_path, 'reports', 'models')


# Make sure you are using the right device.
# 修改设备选择的代码
def get_device():
    if torch.cuda.is_available():
        device = torch.device("cuda")  # 使用默认的可用 GPU
        print(f"使用 GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        print("使用 CPU")
    return device


# This function is used to calculate the accuracy of model classification
def get_accuracy(model, loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data in loader:
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
def train(model, model_name, optimizer, criterion, train_loader, val_loader, scheduler=None, epochs_n=100, best_model_path=None, save_all_epochs=False):
    model.to(device)
    learning_curve = [np.nan] * epochs_n
    train_accuracy_curve = [np.nan] * epochs_n
    val_accuracy_curve = [np.nan] * epochs_n
    max_val_accuracy = 0
    max_val_accuracy_epoch = 0

    batches_n = len(train_loader)
    losses_list = []
    grads_stats = []  # 修改为保存梯度统计信息
    
    # 创建模型保存目录
    if best_model_path:
        os.makedirs(os.path.dirname(best_model_path), exist_ok=True)

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
            #loss = criterion(prediction, y)
            #loss = criterion(prediction, y) + 1e-6 * sum(p.abs().sum() for p in model.parameters())
            loss = criterion(prediction, y) + 1e-7 * sum(p.pow(2).sum() for p in model.parameters())
            loss.backward()
            optimizer.step()

            loss_list.append(loss.item())
            # 收集梯度信息
            if model.classifier[4].weight.grad is not None:
                # 将梯度张量展平并转换为numpy数组
                grad_flat = model.classifier[4].weight.grad.clone().cpu().numpy().flatten()
                grad.append(grad_flat)

            learning_curve[epoch] += loss.item()

        losses_list.append(loss_list)
        
        # 每轮结束后计算梯度统计信息
        if grad:
            epoch_grads = np.array(grad)
            grad_mean = np.mean(epoch_grads, axis=0)
            grad_std = np.std(epoch_grads, axis=0)
            # 保存每轮的梯度均值和标准差的均值作为代表值
            grads_stats.append([epoch, np.mean(grad_mean), np.mean(grad_std)])
        
        display.clear_output(wait=True)
        f, axes = plt.subplots(1, 2, figsize=(15, 3))

        learning_curve[epoch] /= batches_n
        axes[0].plot(learning_curve)
        axes[0].set_title(f'{model_name} - Training Loss')

        # Test your model and save figure here (not required)
        # remember to use model.eval()
        model.eval()
        train_accuracy = get_accuracy(model, train_loader)
        val_accuracy = get_accuracy(model, val_loader)
        train_accuracy_curve[epoch] = train_accuracy
        val_accuracy_curve[epoch] = val_accuracy
        axes[1].plot(train_accuracy_curve, label='Train Accuracy')
        axes[1].plot(val_accuracy_curve, label='Validation Accuracy')
        axes[1].legend()
        axes[1].set_title(f'{model_name} - Accuracy')
        
        
        # 保存每轮的训练曲线
        if epoch == 19:
            plt.savefig(os.path.join(figures_path, f'{model_name}_training_epoch_{epoch}.png'))
            plt.close()
        

        if val_accuracy > max_val_accuracy and best_model_path:
            max_val_accuracy = val_accuracy
            max_val_accuracy_epoch = epoch
            torch.save(model.state_dict(), best_model_path)
            print(f"Epoch {epoch}: {model_name} 验证集准确率提高，保存模型到 {best_model_path}")
        

    print(f"{model_name} 训练完成！最佳验证准确率: {max_val_accuracy:.2f}% (Epoch {max_val_accuracy_epoch})")
    
    final_train_accuracy = train_accuracy_curve[-1]
    final_val_accuracy = val_accuracy_curve[-1]
    print(f"{model_name} 最终训练准确率: {final_train_accuracy:.2f}%")
    print(f"{model_name} 最终验证准确率: {final_val_accuracy:.2f}%")
    
    return losses_list, grads_stats, train_accuracy_curve, val_accuracy_curve

def plot_loss_landscape(losses, model_name):
    min_curve = []
    max_curve = []
    num_steps = max([len(loss_list) for loss_list in losses])
    for step in range(num_steps):
        step_losses = [loss_list[step] for loss_list in losses if step < len(loss_list)]
        min_curve.append(np.nanmin(step_losses))
        max_curve.append(np.nanmax(step_losses))

    plt.figure(figsize=(10, 6))
    steps = np.arange(len(min_curve))
    plt.fill_between(steps, min_curve, max_curve, alpha=0.5)
    plt.xlabel('Training Steps')
    plt.ylabel('Loss')
    plt.title(f'{model_name} - Loss Landscape')
    plt.savefig(os.path.join(figures_path, f'{model_name}_loss_landscape.png'))
    plt.close()

 
def compare_accuracy_curves(all_train_acc, all_val_acc, model_names, epochs):
    plt.figure(figsize=(12, 6))
    
    plt.subplot(1, 2, 1)
    for i, model_name in enumerate(model_names):
        plt.plot(range(epochs), all_train_acc[i], label=model_name)
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.title('Training Accuracy')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    for i, model_name in enumerate(model_names):
        plt.plot(range(epochs), all_val_acc[i], label=model_name)
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.title('Validation Accuracy')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(figures_path, 'model_comparison.png'))
    plt.close()


if __name__ == '__main__':
    device = get_device()

    # Initialize your data loader
    train_loader = get_cifar_loader(train=True)
    val_loader = get_cifar_loader(train=False)
    
    models = {
        #'VGG_A':VGG_A,
        'VGG_A_BatchNorm': VGG_A_BatchNorm
        #'VGG_A_Dropout': VGG_A_Dropout
    }
    
    epo = 20
    lr = 0.001
    all_train_acc = []
    all_val_acc = []
    model_names = list(models.keys())
    
    os.makedirs('./results/loss', exist_ok=True)
    os.makedirs('./results/grads', exist_ok=True)
    os.makedirs(models_path, exist_ok=True)
    os.makedirs(figures_path, exist_ok=True)

    set_random_seeds(seed_value=2020, device=device)

    for model_name, model_class in models.items():
        print(f"\n开始训练 {model_name}...")
        
        model = model_class()
        
        loss_save_path = f'./results/loss/{model_name}_loss.txt'
        grad_save_path = f'./results/grads/{model_name}_grads_stats.txt'
        best_model_path = os.path.join(models_path, f'{model_name}_best_model.pth')
        
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        #optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
        #optimizer = torch.optim.RMSprop(model.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss()
        
        loss, grads, train_acc, val_acc = train(
            model, model_name, optimizer, criterion, train_loader, val_loader,
            epochs_n=epo,
            best_model_path=best_model_path,
            save_all_epochs=True
        )
        
        np.savetxt(loss_save_path, loss, fmt='%s', delimiter=' ')
        
        if grads:
            grads_array = np.array(grads)
            np.savetxt(grad_save_path, 
                      grads_array, 
                      fmt='%s', 
                      delimiter=' ',
                      header='Epoch, Gradient_Mean, Gradient_Std')
            print(f"{model_name} 梯度统计信息已保存到 {grad_save_path}")
        
        plot_loss_landscape(loss, model_name)
        
        all_train_acc.append(train_acc)
        all_val_acc.append(val_acc)
    
    compare_accuracy_curves(all_train_acc, all_val_acc, model_names, epo)
    print(f"所有模型训练完成！模型比较图已保存到 {os.path.join(figures_path, 'model_comparison.png')}")