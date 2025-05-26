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

from models.vgg import VGG_A_Dropout
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
        device = torch.device("cuda")
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


# We use this function to complete the entire training process
def train(model, model_name, optimizer, criterion, train_loader, val_loader, scheduler=None, epochs_n=100, best_model_path=None):
    model.to(device)
    learning_curve = [np.nan] * epochs_n
    train_accuracy_curve = [np.nan] * epochs_n
    val_accuracy_curve = [np.nan] * epochs_n
    max_val_accuracy = 0
    max_val_accuracy_epoch = 0

    batches_n = len(train_loader)
    losses_list = []

    for epoch in tqdm(range(epochs_n), unit='epoch'):
        if scheduler is not None:
            scheduler.step()
        model.train()

        loss_list = []
        learning_curve[epoch] = 0

        for data in train_loader:
            x, y = data
            x = x.to(device)
            y = y.to(device)
            optimizer.zero_grad()
            prediction = model(x)
            
            # 计算基础损失
            base_loss = criterion(prediction, y)
            
            # 添加正则化项（根据模型名称判断）
            if "L1_Regularized" in model_name:
                # L1 正则化
                l1_reg = 0.0001 * sum(p.abs().sum() for p in model.parameters())
                loss = base_loss + l1_reg
            elif "L2_Regularized" in model_name:
                # L2 正则化 (等同于weight decay)
                l2_reg = 0.0001 * sum(p.pow(2).sum() for p in model.parameters())
                loss = base_loss + l2_reg
            else:
                loss = base_loss  # 无正则化
            
            loss.backward()
            optimizer.step()

            loss_list.append(loss.item())
            learning_curve[epoch] += loss.item()

        losses_list.append(loss_list)
        
        # 每轮结束后评估模型
        model.eval()
        train_accuracy = get_accuracy(model, train_loader)
        val_accuracy = get_accuracy(model, val_loader)
        train_accuracy_curve[epoch] = train_accuracy
        val_accuracy_curve[epoch] = val_accuracy
        
        # 保存最佳模型
        if val_accuracy > max_val_accuracy and best_model_path:
            max_val_accuracy = val_accuracy
            max_val_accuracy_epoch = epoch
            torch.save(model.state_dict(), best_model_path)
            print(f"Epoch {epoch}: {model_name} 验证集准确率提高，保存模型到 {best_model_path}")

    print(f"{model_name} 训练完成！最佳验证准确率: {max_val_accuracy:.2f}% (Epoch {max_val_accuracy_epoch})")
    return losses_list, train_accuracy_curve, val_accuracy_curve, max_val_accuracy


# Plot the final loss landscape
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


# Compare accuracy curves of different models
def compare_accuracy_curves(all_train_acc, all_val_acc, model_names, epochs, save_path):
    plt.figure(figsize=(12, 6))
    
    # Plot training accuracy
    plt.subplot(1, 2, 1)
    for i, model_name in enumerate(model_names):
        plt.plot(range(epochs), all_train_acc[i], label=model_name)
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.title('Training Accuracy')
    plt.legend()
    plt.grid(True)
    
    # Plot validation accuracy
    plt.subplot(1, 2, 2)
    for i, model_name in enumerate(model_names):
        plt.plot(range(epochs), all_val_acc[i], label=model_name)
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.title('Validation Accuracy')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


# Save experiment results to CSV
def save_experiment_results(results, save_path):
    with open(save_path, 'w') as f:
        f.write('Model Name,Best Validation Accuracy (%)\n')
        for name, acc in results.items():
            f.write(f'{name},{acc:.4f}\n')
    print(f"实验结果已保存到 {save_path}")


if __name__ == '__main__':
    device = get_device()

    # Initialize data loader
    train_loader = get_cifar_loader(train=True)
    val_loader = get_cifar_loader(train=False)
    
    # 训练参数
    epo = 20
    lr = 0.001
    all_results = {}
    
    # 创建保存目录
    os.makedirs('./results/loss', exist_ok=True)
    os.makedirs('./results/grads', exist_ok=True)
    os.makedirs(models_path, exist_ok=True)
    os.makedirs(figures_path, exist_ok=True)

    # 设置随机种子
    set_random_seeds(seed_value=2020, device=device)

    # 定义要测试的不同参数组合
    experiments = {
        # (b) 不同损失函数（含正则化）
        'loss_function': [
            ('CrossEntropy', nn.CrossEntropyLoss()),
            ('L1_Regularized', nn.CrossEntropyLoss()),  # 仅基础损失函数
            ('L2_Regularized', nn.CrossEntropyLoss()),  # 仅基础损失函数
        ],
        # (d) 不同优化器
        #'optimizer': [
            #('Adam', lambda params: torch.optim.Adam(params, lr=lr)),
            #('SGD', lambda params: torch.optim.SGD(params, lr=lr, momentum=0.9)),
            #('RMSprop', lambda params: torch.optim.RMSprop(params, lr=lr)),
        #],
    }

    # 为每个实验类别创建子目录
    for exp_type in experiments.keys():
        exp_figures_path = os.path.join(figures_path, exp_type)
        exp_models_path = os.path.join(models_path, exp_type)
        os.makedirs(exp_figures_path, exist_ok=True)
        os.makedirs(exp_models_path, exist_ok=True)
        
        all_train_acc = []
        all_val_acc = []
        model_names = []
        exp_results = {}
        
        print(f"\n=== 开始 {exp_type} 实验 ===")
        
        for name, param in experiments[exp_type]:
            model = VGG_A_Dropout()
            # 根据实验类型设置不同参数
            if exp_type == 'loss_function':
                criterion = param
                optimizer = torch.optim.Adam(model.parameters(), lr=lr)
                full_name = f"{exp_type}_{name}"
            elif exp_type == 'optimizer':
                criterion = nn.CrossEntropyLoss()
                optimizer = param(model.parameters())
                full_name = f"{exp_type}_{name}"
            
            # 定义保存路径
            best_model_path = os.path.join(exp_models_path, f'{full_name}_best_model.pth')
            loss_save_path = f'./results/loss/{full_name}_loss.txt'
            
            # 训练模型
            print(f"\n开始训练: {full_name}")
            losses, train_acc, val_acc, best_acc = train(
                model, full_name, optimizer, criterion, train_loader, val_loader,
                epochs_n=epo, best_model_path=best_model_path
            )
            
            # 保存损失
            np.savetxt(loss_save_path, losses, fmt='%s', delimiter=' ')
            
            # 保存损失景观图
            plot_loss_landscape(losses, full_name)
            
            # 记录结果
            all_train_acc.append(train_acc)
            all_val_acc.append(val_acc)
            model_names.append(full_name)
            exp_results[full_name] = best_acc
            all_results[full_name] = best_acc
        
        # 比较当前实验类别下的所有模型
        compare_accuracy_curves(
            all_train_acc, all_val_acc, model_names, epo,
            os.path.join(exp_figures_path, f'{exp_type}_comparison.png')
        )
        
        # 保存当前实验类别的结果
        save_experiment_results(
            exp_results, 
            os.path.join(exp_figures_path, f'{exp_type}_results.csv')
        )
    
    # 保存所有实验的总结果
    save_experiment_results(
        all_results, 
        os.path.join(figures_path, 'all_experiments_results.csv')
    )
    
    # 找出表现最好的模型
    best_model = max(all_results, key=all_results.get)
    print(f"\n最佳模型: {best_model} (准确率: {all_results[best_model]:.4f}%)")