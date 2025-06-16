'''创建一个可用于神经网络训练的组合损失函数，
它结合了二进制交叉熵损失（BCE Loss）和 Dice 损失，
通过为这两种损失分配不同的权重来综合衡量模型预测结果与真实目标之间的差异，
从而在训练过程中引导模型朝着更优的方向优化参数，提高预测的准确性。'''

import torch
import torch.nn as nn
import torch.nn.functional as F

class CombinedLoss(nn.Module):
    def __init__(self, weight_bce=0.5, weight_dice=0.5):
        super(CombinedLoss, self).__init__()
        self.weight_bce = weight_bce
        self.weight_dice = weight_dice

    def forward(self, y_pred, y_true):
        # 二进制交叉熵损失（注意：不应用Sigmoid函数）
        bce_loss = F.binary_cross_entropy(y_pred, y_true)

        # Dice 损失
        smooth = 1e-5
        intersection = torch.sum(y_pred * y_true)
        union = torch.sum(y_pred) + torch.sum(y_true)
        dice_loss = 1 - (2 * intersection + smooth) / (union + smooth)

        # 结合两个损失
        combined_loss = self.weight_bce * bce_loss + self.weight_dice * dice_loss

        return combined_loss

