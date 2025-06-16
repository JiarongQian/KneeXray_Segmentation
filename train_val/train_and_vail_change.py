import torch
import os
import numpy as np
from tqdm import tqdm
from train_val.save_metrics import save_metrics
from train_val.Animator import plot_training_progress

def train_and_validate(model, train_loader, val_loader, optimizer, loss_fn, num_epochs, patience, train_on_gpu=True):
    device = torch.device("cuda" if train_on_gpu and torch.cuda.is_available() else "cpu")
    model.to(device)

    #创建一系列空列表
    train_losses = []
    val_losses = []
    ious = []
    recalls = []
    f1_scores = []
    precisions = []
    best_iou = 0.0
    best_epoch = 0
    best_model_state = None

    for epoch in range(num_epochs):
        # 训练模式
        print(f"------------------第{epoch+1}轮训练开始------------------")
        model.train()
        train_loss = 0.0

        for batch_idx, (images, masks) in enumerate(train_loader):
            images, masks = images.to(device), masks.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = loss_fn(outputs, masks)

            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        # 计算每个 epoch 的平均训练损失
        train_loss /= len(train_loader)
        train_losses.append(train_loss)
        print(f"第{epoch+1}轮训练 train_loss为{train_loss}")

        # 验证模式
        model.eval()
        val_loss = 0.0
        val_iou = 0.0
        val_recall = 0.0
        val_f1 = 0.0
        val_precision = 0.0

        with torch.no_grad():
            for batch_idx, (images, masks) in enumerate(val_loader):
                images, masks = images.to(device), masks.to(device)

                if train_on_gpu:
                    images, masks = images.cuda(), masks.cuda()

                outputs = model(images)
                loss = loss_fn(outputs, masks.float())
                val_loss += loss.item()


                true =masks.cpu().numpy().astype(np.bool_)
                prediction = (outputs.detach().cpu().numpy() > 0.5).astype(np.bool_)

                true_positive = np.sum(prediction& true)          #计算真正例的数量
                true_negative = np.sum((~prediction) & (~true))   #计算真反例的数量
                false_positive = np.sum(prediction & (~true))     #计算假正例的数量
                false_negative = np.sum((~prediction) & true)     #计算假反例的数量

                epsilon = 1e-7                                                                   #设置小数值 epsilon，为了防止在后续计算某些指标（如准确率、F1 分数）时出现除以零的情况。
                precision = (true_positive) / (true_positive + false_positive + epsilon)         #计算准确率
                recall = true_positive / (true_positive + false_negative)                        #计算召回率
                f1 = 2 * (precision * recall) / (precision + recall + epsilon)                   #计算F1分数
                iou = true_positive / (true_positive + false_positive + false_negative)          #计算交并比（iou）

                #指标累加
                val_precision += precision
                val_recall += recall
                val_f1 += f1
                val_iou += iou

        # 计算每个 epoch 的平均验证损失和平均评估指标（指标平均值）
        val_loss /= len(val_loader)
        val_iou /= len(val_loader)
        val_recall /= len(val_loader)
        val_f1 /= len(val_loader)
        val_precision /= len(val_loader)

        #将每一轮得到的指标记录在列表中
        val_losses.append(val_loss)
        ious.append(val_iou)
        recalls.append(val_recall)
        f1_scores.append(val_f1)
        precisions.append(val_precision)

        print(f"第{epoch+1}轮训练 val_loss为{val_loss}")
        print(f"第{epoch+1}轮训练 val_iou为{val_iou}")

        #早停机制及最佳模型保存
        if epoch >= num_epochs - patience:
            if val_iou > best_iou:
                best_iou = val_iou
                best_epoch = epoch + 1
                best_model_state = model.state_dict()         #获取当前模型的状态字典，状态字典包含了模型的所有可训练参数的值。
                
                save_path="./results/model.pt"
                torch.save(best_model_state,save_path)
        '''
        patience：整数，
        用于早停机制。当训练轮次达到 num_epochs - patience 之后，
        如果在验证集上的交并比（IoU）有所提高，
        就会保存当前模型为最佳模型。
        '''

        #保存指标数据
        metrics_path='./results/metrics.txt'
        save_metrics(train_losses, val_losses, ious, recalls, f1_scores, precisions,metrics_path=metrics_path)

        #绘制训练进度图表：
        plot_path="./results/training_profress.png"
        if(epoch==num_epochs-1):
           plot_training_progress(train_losses, val_losses, ious, recalls, f1_scores, precisions,
                               num_epochs,plot_path=plot_path)  # 在每个 epoch 结束后动态更新图表


    #最终输出最佳模型信息
    print(f"Best model saved at epoch {best_epoch} with IoU {best_iou:.4f}")


'''
关于训练损失、验证损失以及一些验证阶段的评估指标（如交并比 IoU、召回率 Recall、F1 分数、准确率 Precision）说明
1.训练损失（train_losses)
一个列表或者数组，其中包含了在每个训练轮次（Epoch）中计算得到的训练损失值。
用于衡量模型在训练数据上的拟合程度。

2.验证损失（val_losses)
一个列表或者数组，存储了每个训练轮次在验证数据集上计算得到的验证损失值。
帮助判断模型在未见过的数据（即验证集数据）上的泛化能力。

3.交并比（IoU)
一个列表或数组，存放着每个训练轮次在验证数据集上计算得到的交并比（IoU）值。
交并比是一种用于衡量图像分割等任务中预测结果与真实结果之间重叠程度的指标，取值范围在 0 到 1 之间，值越高表示预测结果与真实结果的匹配度越高。

4.召回率（recalls)
一个列表或数组，包含每个训练轮次在验证数据集上计算得到的召回率（Recall）值。
召回率用于衡量在所有真实的正例中，模型能够正确预测出的比例，也是一个介于 0 到 1 之间的指标，常用于评估分类任务或检测任务中模型对正例的捕捉能力。

5.F1分数（f1_scores)
一个列表或数组，存放每个训练轮次在验证数据集上计算得到的 F1 分数
F1 分数是综合考虑了召回率和准确率的一个指标，它通过调和平均数的方式将召回率和准确率结合起来，同样取值范围在 0 到 1 之间，能够更全面地评估模型的性能。

6.准确率值（precisions)
一个列表或数组，里面是每个训练轮次在验证数据集上计算得到的准确率（Precision）值。
准确率用于衡量在所有预测为正例的结果中，真正为正例的比例，是评估模型预测准确性的重要指标之一。

'''