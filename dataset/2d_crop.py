#为减小内存占用，对原图像进行中心裁剪，裁剪出中间大小为736*576的大小部分（因为要作为UNET的输入，所以长和宽需要是32的倍数）


import nibabel as nib
import numpy as np
import os
from scipy.ndimage import zoom

#定义对单个图像文件进行中心裁剪的函数
def center_crop_nii(input_path, output_path, target_shape=(736, 576)):
    img = nib.load(input_path)
    data = img.get_fdata()
    original_shape = data.shape

    # 计算裁剪的起始坐标
    start_x = (original_shape[0] - target_shape[0]) // 2
    start_y = (original_shape[1] - target_shape[1]) // 2
    # 进行中心裁剪
    cropped_data = data[start_x:start_x + target_shape[0], start_y:start_y + target_shape[1]]
    # 创建新的Nifti1Image对象
    new_img = nib.Nifti1Image(cropped_data, img.affine)
    # 保存新的.nii文件
    nib.save(new_img, output_path)

#images原始文件夹和目标文件夹
current_dir=current_dir = os.path.dirname(os.path.abspath(__file__))

#使用相对路径
#original_folder = os.path.join(current_dir,  "data")
#cropped_folder= os.path.join(current_dir,  "croped_images")

#使用绝对路径做数据源
original_folder = r"D:\医学影像处理课题组\膝关节X图像分割\knee\原始数据格式"
cropped_folder= os.path.join(current_dir,  "croped_images")

if not os.path.exists(cropped_folder):
    os.makedirs(cropped_folder)

for root,dirs,files in os.walk(original_folder):
    for file in files:
        if file.endswith('.nii'):
            input_path = os.path.join(root, file)
            output_path = os.path.join(cropped_folder, file)
            center_crop_nii(input_path, output_path)

# masks原始文件夹和目标文件夹
current_dir=current_dir = os.path.dirname(os.path.abspath(__file__))

#使用相对路径
#original_folder =  os.path.join(current_dir,  "data")

#使用绝对路径做数据源
original_folder = r"D:\医学影像处理课题组\膝关节X图像分割\knee\原始数据格式"
cropped_folder =os.path.join(current_dir,  "croped_masks")

if not os.path.exists(cropped_folder):
    os.makedirs(cropped_folder)

for root,dirs,files in os.walk(original_folder):
    for file in files:
        if file.endswith('T-1.nii.gz'):
           input_path = os.path.join(root, file)
           subfolder_name=os.path.basename(root)
           new_file_name=f"{subfolder_name}_{file}"
           output_path = os.path.join(cropped_folder, new_file_name)
           center_crop_nii(input_path, output_path)
