#数据集创建
import os
import nibabel as nib
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split
import SimpleITK as sitk
#这个数据集的原图像和掩膜图像都只有一张切片————（可以理解为是二维图像，或者是只有一张切片的三维图像）
#该项目文件将其理解为二维图像，并使用2D_UNET进行图像分割

#定义数据集类，其继承自Dataset
class MedicalImageDataset(Dataset):
    def __init__(self, images_dir, masks_dir, transform=None):#transform初始化为None,表示默认情况下不进行数据增强
        #定义函数内部变量
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        self.image_files = sorted(os.listdir(images_dir))  # 获取图像文件列表
        self.mask_files = sorted(os.listdir(masks_dir))  # 获取掩膜文件列表
        self.transform = transform  # 数据增强操作（如果有）

    #返回数据集的长度，即返回图像文件列表的长度
    def __len__(self):
        return len(self.image_files)

    #__getitem__方法用于根据给定的索引idx获取数据集中的一个样本
    def __getitem__(self, idx):
        #1.构建具体图像文件路径
        image_path = os.path.join(self.images_dir, self.image_files[idx])
        mask_path = os.path.join(self.masks_dir, self.mask_files[idx])

        #2.使用nibabel加载nii文件，通过get_fdata获取文件中的数据内容
        image = nib.load(image_path).get_fdata()
        mask = nib.load(mask_path).get_fdata()

        #3.转换为numpy数组——原nibabael库加载后的数据类型属于医学图像数据特有格式，转换为Numpy数组后可以提供一个统一地，在python科学计算领域广泛使用的数据格式
        image = np.array(image, dtype=np.float32)
        mask = np.array(mask, dtype=np.float32)

        #4.标准化图像：将其像素值映射到0-1的范围内——优化模型训练（数值不会过大或过小：加速收敛；提高模型性能和泛化能力；防止数据溢出和梯度消失/爆炸）
        # 通过减去最小值并除以最大值和最小值的差值来实现（最大最小值归一化）
        image = (image - np.min(image)) / (np.max(image) - np.min(image))

        #5.转换为torch张量并添加通道维度——pytorch是一个深度学习框架，其数据结构torch.Tensor是专门为了在深度学习模型中高效地进行计算和梯度计算而设计的。
        # ①通过unsqueeze(0)操作在第0维上添加一个维度（通常是通道维度，这里是单通道）
        image = torch.tensor(image).unsqueeze(0)
        mask = torch.tensor(mask).unsqueeze(0)
        #（1，736，576，1）——（通道数，长，宽，切片数）

        # ②通过squeeze(-1)操作去掉最后一维，在这里属于多余维度，因为是2D-UNET，一张切片就是一例数据
        image= image.squeeze(-1)
        mask = mask.squeeze(-1)
        #（1，736，576）——（通道数，长，宽）

        #6.应用数据增强（如果有） 不过一般只对image进行数据增强，不需要对mask进行数据增强
        if self.transform:
            image, mask = self.transform(image, mask)

        return image, mask

