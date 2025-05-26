import numpy as np
import matplotlib.pyplot as plt
import os

# 定义文件路径
batchnorm_path = './results/loss_stats/VGG_A_BatchNorm_loss_stats.txt'
vgg_a_path = './results/loss_stats/VGG_A_loss_stats.txt'
output_path = 'loss_landscape_comparison.png'

# 检查文件是否存在
for file_path in [batchnorm_path, vgg_a_path]:
    if not os.path.exists(file_path):
        print(f"错误: 文件 {file_path} 不存在")
        exit()

try:
    # 读取VGG_A_BatchNorm_loss_stats.txt文件数据
    with open(batchnorm_path, 'r') as f:
        header = f.readline().strip().split('\t')
        data_VGG_A_BatchNorm = np.loadtxt(f, skiprows=0)
        mean_loss_VGG_A_BatchNorm = data_VGG_A_BatchNorm[:, 0]
        std_loss_VGG_A_BatchNorm = data_VGG_A_BatchNorm[:, 1]

    # 读取VGG_A_loss_stats.txt文件数据 
    with open(vgg_a_path, 'r') as f:
        header = f.readline().strip().split('\t')
        data_VGG_A = np.loadtxt(f, skiprows=0)
        mean_loss_VGG_A = data_VGG_A[:, 0]
        std_loss_VGG_A = data_VGG_A[:, 1]

    # 设置图形参数
    plt.figure(figsize=(10, 6))

    # 绘制VGG_A_BatchNorm的损失景观
    steps = np.arange(len(mean_loss_VGG_A_BatchNorm))
    # plt.plot(steps, mean_loss_VGG_A_BatchNorm, label='VGG_A_BatchNorm', color='green', linewidth=1.5) 
    patch1 = plt.fill_between(steps, 
                              mean_loss_VGG_A_BatchNorm - std_loss_VGG_A_BatchNorm, 
                              mean_loss_VGG_A_BatchNorm + std_loss_VGG_A_BatchNorm, 
                              alpha=0.1, facecolor='green', interpolate=True)

    # 绘制VGG_A的损失景观
    steps = np.arange(len(mean_loss_VGG_A))
    # plt.plot(steps, mean_loss_VGG_A, label='VGG_A', color='red', linewidth=1.5) 
    patch2 = plt.fill_between(steps, 
                              mean_loss_VGG_A - std_loss_VGG_A, 
                              mean_loss_VGG_A + std_loss_VGG_A, 
                              alpha=0.1, facecolor='red', interpolate=True)

    # 添加图形标签
    plt.xlabel('Steps')
    plt.ylabel('Loss')
    plt.title('Loss Landscape Comparison')
    plt.legend([patch1, patch2], ['VGG_A_BatchNorm', 'VGG_A'], loc='upper right')

    # 保存图形
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"图形已成功保存到: {os.path.abspath(output_path)}")

    # 显示图形（在交互式环境中有效）
    plt.show()

except Exception as e:
    print(f"发生错误: {e}")
finally:
    plt.close()