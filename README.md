# NN-DL-PJ2

代码功能解释：

VGG_Loss_Landscape.py: 可选择模型进行训练，调整优化器和损失函数类型；

VGG_BNoptim.py：遍历学习率参数空间，训练VGG_BN模型，并绘制相应的对比图；

filter-visualization.py: 导入最优模型，可视化滤波器权重；

gradients: 训练VGG_BN模型，根据训练历史，绘制gradient-predictability和gradient-consistency图像

landscape-contrast：导入训练历史，绘制VGG_A模型和VGG_A_BatchNorm模型的landscape对比图；



