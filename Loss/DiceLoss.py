#损失函数文件
import torch
import torch.nn as nn
import torch.nn.functional as F

class DiceLoss(nn.Module):
    def __init__(self):
        super(DiceLoss, self).__init__()

    #定义DiceLoss类的前向传播逻辑——计算损失值的具体计算过程——用于衡量模型预测结果于真实目标之间的差异
    def forward(self, predicted, target):
        #predicted是预测值（通常是模型的输出，在图像分割任务中可能是每个像素属于某一类别的预测概率）
        #target是目标值   （真实的标签值，在图像分割中就是每个像素实际所属的类别，通常会进行适当的编码处理）

        #1.计算交集
        intersection = torch.sum(predicted * target)

        #2.计算并集
        union = torch.sum(predicted) + torch.sum(target)

        #3.计算Dice系数
        dice = (2.0 * intersection + 1e-5) / (union + 1e-5)

        #.4.计算Dice损失值
        return 1 - dice
