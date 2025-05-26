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


# 训练函数 - 修改为记录每个学习率下的损失波动
def train(model, model_name, optimizer, criterion, train_loader, val_loader, scheduler=None, epochs_n=100, best_model_path=None, save_all_epochs=False):
    model.to(device)
    learning_curve = [np.nan] * epochs_n
    train_accuracy_curve = [np.nan] * epochs_n
    val_accuracy_curve = [np.nan] * epochs_n
    max_val_accuracy = 0
    max_val_accuracy_epoch = 0

    batches_n = len(train_loader)
    losses_list = []  # 记录每个step的loss，维度：[epoch数, step数]
    grads_stats = []  # 保存梯度统计信息
    
    # 创建模型保存目录
    if best_model_path:
        os.makedirs(os.path.dirname(best_model_path), exist_ok=True)

    for epoch in tqdm(range(epochs_n), unit='epoch'):
        if scheduler is not None:
            scheduler.step()
        model.train()

        loss_list = []  # 记录每个step的loss值，维度：[step数]
        grad = []  # 记录每个step的梯度值

        for data in train_loader:
            x, y = data
            x = x.to(device)
            y = y.to(device)
            optimizer.zero_grad()
            prediction = model(x)
            loss = criterion(prediction, y) + 1e-7 * sum(p.pow(2).sum() for p in model.parameters())
            loss.backward()
            optimizer.step()

            loss_list.append(loss.item())
            # 收集梯度信息
            if hasattr(model, 'classifier') and len(model.classifier) > 4 and model.classifier[4].weight.grad is not None:
                grad_flat = model.classifier[4].weight.grad.clone().cpu().numpy().flatten()
                grad.append(grad_flat)

        losses_list.append(loss_list)  # 保存每个epoch的step损失列表
        
        # 计算epoch级平均损失
        learning_curve[epoch] = np.mean(loss_list) if loss_list else np.nan
        
        # 计算梯度统计信息
        if grad:
            epoch_grads = np.array(grad)
            grad_mean = np.mean(epoch_grads)
            grad_std = np.std(epoch_grads)
            grads_stats.append([epoch, grad_mean, grad_std])
        
        # 验证集评估
        model.eval()
        train_accuracy = get_accuracy(model, train_loader)
        val_accuracy = get_accuracy(model, val_loader)
        train_accuracy_curve[epoch] = train_accuracy
        val_accuracy_curve[epoch] = val_accuracy
        
        # 保存最后一轮曲线
        if epoch == epochs_n - 1 and save_all_epochs:
            f, axes = plt.subplots(1, 2, figsize=(15, 3))
            axes[0].plot(learning_curve)
            axes[0].set_title(f'{model_name} - Training Loss')
            axes[1].plot(train_accuracy_curve, label='Train Accuracy')
            axes[1].plot(val_accuracy_curve, label='Validation Accuracy')
            axes[1].legend()
            axes[1].set_title(f'{model_name} - Accuracy')
            plt.savefig(os.path.join(figures_path, f'{model_name}_training_lr_{lr}.png'))
            plt.close()

        # 保存最佳模型
        if val_accuracy > max_val_accuracy and best_model_path:
            max_val_accuracy = val_accuracy
            max_val_accuracy_epoch = epoch
            torch.save(model.state_dict(), best_model_path)
            print(f"Epoch {epoch}: {model_name} 验证集准确率提高，保存模型到 {best_model_path}")

    print(f"{model_name} 训练完成！最佳验证准确率: {max_val_accuracy:.2f}% (Epoch {max_val_accuracy_epoch})")
    
    return losses_list, grads_stats, train_accuracy_curve, val_accuracy_curve


# 绘制不同学习率下的损失波动比较图
def plot_loss_fluctuation_by_lr(all_losses, model_names, learning_rates):
    plt.figure(figsize=(12, 8))
    
    for i, model_name in enumerate(model_names):
        model_losses = all_losses[i]  # 维度：[学习率数, epoch数, step数]
        plt.subplot(len(model_names), 1, i+1)
        
        # 创建用于图例的代理对象列表
        legend_handles = []
        
        for j, lr in enumerate(learning_rates):
            epoch_losses = model_losses[j]  # 维度：[epoch数, step数]
            if not epoch_losses:
                continue
            
            # 展平所有step损失（跨epoch）
            flat_losses = np.concatenate(epoch_losses)
            steps = np.arange(len(flat_losses))
            
            # 计算移动平均和标准差（窗口大小为总步数的5%）
            window_size = max(2, int(len(steps) * 0.05))
            avg_loss = np.convolve(flat_losses, np.ones(window_size)/window_size, mode='valid')
            std_loss = np.array([
                np.std(flat_losses[max(0, k-window_size+1):k+1]) 
                for k in range(len(flat_losses))
            ])[window_size-1:]  # 对齐平均数组长度
            
            # 绘制置信区间并保存返回的PolyCollection对象
            fill = plt.fill_between(steps[window_size-1:], 
                                   avg_loss - std_loss, 
                                   avg_loss + std_loss, 
                                   alpha=0.2, edgecolor='none')
            
            # 创建代理对象用于图例
            from matplotlib.patches import Patch
            legend_handles.append(Patch(facecolor=fill.get_facecolor()[0], 
                                       label=f'LR={lr}'))
        
        plt.title(f'{model_name} - Loss Fluctuation by Learning Rate')
        plt.xlabel('Training Steps')
        plt.ylabel('Loss')
        plt.legend(handles=legend_handles)  # 使用代理对象设置图例
        plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(figures_path, 'loss_fluctuation_by_lr.png'))
    plt.close()

# 比较不同模型在相同学习率下的性能
def compare_models_at_lr(all_train_acc, all_val_acc, model_names, learning_rates, epochs):
    for j, lr in enumerate(learning_rates):
        plt.figure(figsize=(12, 6))
        
        plt.subplot(1, 2, 1)
        for i, model_name in enumerate(model_names):
            plt.plot(range(epochs), all_train_acc[i][j], label=model_name)
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy (%)')
        plt.title(f'Training Accuracy (LR = {lr})')
        plt.legend()
        plt.grid(True)
        
        plt.subplot(1, 2, 2)
        for i, model_name in enumerate(model_names):
            plt.plot(range(epochs), all_val_acc[i][j], label=model_name)
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy (%)')
        plt.title(f'Validation Accuracy (LR = {lr})')
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig(os.path.join(figures_path, f'model_comparison_lr_{lr}.png'))
        plt.close()


if __name__ == '__main__':
    device = get_device()
    train_loader = get_cifar_loader(train=True)
    val_loader = get_cifar_loader(train=False)
    
    models = {
        'VGG_A_BatchNorm': VGG_A_BatchNorm,
    }
    learning_rates = [1e-3, 2e-3, 1e-4, 5e-4]
    epo = 20
    all_losses = []
    all_train_acc = []
    all_val_acc = []
    model_names = list(models.keys())
    
    os.makedirs('./results/loss', exist_ok=True)
    os.makedirs('./results/grads', exist_ok=True)
    os.makedirs(models_path, exist_ok=True)
    os.makedirs(figures_path, exist_ok=True)

    for model_name, model_class in models.items():
        model_losses = []
        model_train_acc = []
        model_val_acc = []
        
        for lr in learning_rates:
            print(f"\n开始训练 {model_name} (学习率 = {lr})...")
            set_random_seeds(seed_value=2020, device=device)
            model = model_class()
            
            loss_save_path = f'./results/loss/{model_name}_loss_lr_{lr}.txt'
            grad_save_path = f'./results/grads/{model_name}_grads_stats_lr_{lr}.txt'
            best_model_path = os.path.join(models_path, f'{model_name}_best_model_lr_{lr}.pth')
            
            optimizer = torch.optim.Adam(model.parameters(), lr=lr)
            criterion = nn.CrossEntropyLoss()
            
            # 调用训练函数，获取每个epoch的step损失列表（losses_list）
            losses_list, grads, train_acc, val_acc = train(
                model, model_name, optimizer, criterion,
                train_loader, val_loader,
                epochs_n=epo,
                best_model_path=best_model_path,
                save_all_epochs=False
            )
            
            # 保存loss（每个epoch的平均损失）
            epoch_avg_losses = [np.mean(epoch_loss) for epoch_loss in losses_list]
            np.savetxt(loss_save_path, epoch_avg_losses, fmt='%.4f')
            
            if grads:
                grads_array = np.array(grads)
                np.savetxt(grad_save_path, 
                          grads_array, 
                          fmt='%d %.6f %.6f', 
                          header='Epoch Gradient_Mean Gradient_Std')
                print(f"{model_name} (LR={lr}) 梯度统计信息已保存到 {grad_save_path}")
            
            model_losses.append(losses_list)  # 保存每个学习率的step损失列表
            model_train_acc.append(train_acc)
            model_val_acc.append(val_acc)
        
        all_losses.append(model_losses)
        all_train_acc.append(model_train_acc)
        all_val_acc.append(model_val_acc)
    
    # 绘制损失波动图（修正维度问题）
    plot_loss_fluctuation_by_lr(all_losses, model_names, learning_rates)
    
    # 比较模型性能
    compare_models_at_lr(all_train_acc, all_val_acc, model_names, learning_rates, epo)
    
    print(f"所有模型训练完成！结果已保存至 {figures_path}")