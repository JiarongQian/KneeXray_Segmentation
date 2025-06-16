import nibabel as nib
import numpy as np
"检测文件是否可以打开"
# 加载.nii.gz文件
file_path =r"D:\医学影像处理课题组\膝关节X图像分割\2D_unet膝关节图像分割\dataset\croped_masks\ST000000_T-1.nii.gz"
img = nib.load(file_path)
data = img.get_fdata()

# 打印数据形状，例如（x,y,z）维度
print(data.shape)