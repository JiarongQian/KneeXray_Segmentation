#导入包
import torch
from torch.utils.data import DataLoader, random_split
import os
#导入写好的函数
from utils.dataset import MedicalImageDataset
from model.unet import Model
from Loss.DiceLoss import DiceLoss
from train_val.train_and_vail_change import train_and_validate
from predict.predict_display import predict_and_display
from predict.evaluate_predictions import evaluate_predictions

#1.数据准备
#设置数据集路径
current_dir = os.path.dirname(os.path.abspath(__file__))
images_dir = os.path.join(current_dir,  "dataset", "croped_images") #原图像文件夹
masks_dir = os.path.join(current_dir, "dataset", "croped_masks")    #掩膜mask文件夹
# 初始化数据集
dataset = MedicalImageDataset(images_dir, masks_dir)


# 划分训练集和验证集
train_size = int(0.7 * len(dataset))               # 70%用于训练
val_size   = int(0.15 * len(dataset))               # 15%用于验证
pred_size  = len(dataset) - train_size - val_size  # 15%用于测试
train_dataset, val_dataset ,pred_dataset= random_split(dataset, [train_size, val_size,pred_size])

# 创建数据加载器
train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
val_loader   = DataLoader(val_dataset,   batch_size=1, shuffle=False)
pred_loader  = DataLoader(pred_dataset,  batch_size=1, shuffle=False)

print("-------------------train_loader-------------------")
for batch_images, batch_masks in train_loader:
    print(f"Batch Images shape = {batch_images.shape}, Batch Masks shape = {batch_masks.shape}")

print("-------------------val_loader--------------------")
for batch_images, batch_masks in val_loader:
    print(f"Batch Images shape = {batch_images.shape}, Batch Masks shape = {batch_masks.shape}")

print("-------------------pred_loader--------------------")
for batch_images, batch_masks in pred_loader:
    print(f"Batch Images shape = {batch_images.shape}, Batch Masks shape = {batch_masks.shape}")

#2.创建模型
train_on_gpu = torch.cuda.is_available()      # 是否使用GPU
model = Model(in_features=1, out_features=1, init_features=32).to(device=torch.device("cuda" if train_on_gpu else "cpu"))

#3.定义训练相关参数
loss_fn = DiceLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)


#4.训练模型
num_epochs=20
patience=10
train_and_validate(model, train_loader, val_loader, optimizer, loss_fn, num_epochs, patience, train_on_gpu)

#5.对测试集进行预测
predict_and_display(model_path=os.path.join(current_dir, "./results/model.pt"),pred_loader=pred_loader)

#6.计算预测指标
#evaluate_predictions()
