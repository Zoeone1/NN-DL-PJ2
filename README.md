# NN-DL-PJ2

代码功能解释：

VGG_Loss_Landscape.py: 可选择参数进行训练，调整优化器和损失函数类型；

VGG_BNoptim.py：遍历学习率参数空间，训练VGG_BN模型，并绘制相应的对比图；

filter-visualization.py: 导入最优模型，可视化滤波器权重；

gradients: 训练VGG_BN模型，根据训练历史，绘制gradient-predictability和gradient-consistency图像

landscape-contrast：导入训练历史，绘制VGG_A模型和VGG_A_BatchNorm模型的landscape对比图；

VGG_Loss_Landscape_contrast.py：导入训练历史，绘制VGG_A模型和VGG_A_BatchNorm模型的landscape对比图.

模型权重解释：

VGG_A_Dropout_best_model-L2loss.pth: 为任务一最优参数组合模型训练的权重。

VGG_A_BatchNorm_best_model.pth: 为任务二最优学习率带BN模型训练的权重。
