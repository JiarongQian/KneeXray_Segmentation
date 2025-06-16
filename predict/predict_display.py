import torch
import matplotlib.pyplot as plt
import numpy as np
import os
from model.unet import Model

def predict_and_display(model_path, pred_loader):
    # 加载模型
    model=Model(in_features=1,out_features=1,init_features=32)
    state_dict = torch.load(model_path)
    model.load_state_dict(state_dict)
    
    if torch.cuda.is_available():
         model=model.cuda()
         
    model.eval()  # 设置模型为评估模式

    predictions = []
    save_dir=os.path.join(os.getcwd(),"prediction_images")
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        
    for batch in pred_loader:
        images, _ = batch  # 假设你的预测加载器返回的数据格式是 (图像数据, 其他可能的信息)，这里我们只需要图像数据进行预测

        # 将图像数据移动到合适的设备（如果模型在GPU上训练，这里也需要将数据放到GPU上）
        if torch.cuda.is_available():
            images = images.cuda()
        # 使用模型进行预测
        outputs = model(images)

        # 对模型输出进行处理，比如根据你的任务进行适当的转换（例如二值化等）
        prediction = (outputs.detach().cpu().numpy() > 0.5).astype(int)

        predictions.append(prediction)

        # 显示当前批次的预测图像
        for i in range(prediction.shape[0]):
            plt.imshow(prediction[i, 0], cmap='gray')
            plt.title(f'Prediction Image {i}')
            
            save_path = os.path.join(save_dir, f'prediction_image_{i}.png')
            plt.savefig(save_path)
            plt.close()


    return predictions

