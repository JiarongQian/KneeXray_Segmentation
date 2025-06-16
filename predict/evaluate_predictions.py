import os
import cv2
import numpy as np

#用来计算预测指标的文件 （感觉用处不大 这样去比较的话就和验证集差不多了）
def evaluate_predictions(prediction_dir, true_dir):
    # 创建存储评估结果的列表
    precision_list = []
    iou_list = []
    recall_list = []
    f1_list = []
    dice_list = []
    mcc_list = []

    # 获取预测结果和真实标注文件列表并排序
    prediction_files = sorted(os.listdir(prediction_dir))
    true_files = sorted(os.listdir(true_dir))

    # 遍历每个预测结果文件
    for pred_file, true_file in zip(prediction_files, true_files):
        # 加载预测结果和真实标注图像
        prediction = cv2.imread(os.path.join(prediction_dir, pred_file), cv2.IMREAD_GRAYSCALE)
        true = cv2.imread(os.path.join(true_dir, true_file), cv2.IMREAD_GRAYSCALE)  
        
        # 对标签数据进行二值化处理
        true = np.where(true > 128, 1, 0)
        prediction = np.where(prediction > 128, 1, 0)
        
        # 计算各类数值（转为浮点数以防止溢出）
        true_positive = np.sum((prediction == 1) & (true == 1)).astype(np.float64)
        true_negative = np.sum((prediction == 0) & (true == 0)).astype(np.float64)
        false_positive = np.sum((prediction == 1) & (true == 0)).astype(np.float64)
        false_negative = np.sum((prediction == 0) & (true == 1)).astype(np.float64)

        epsilon = 1e-7
        
        # 计算 precision, recall, f1, iou
        precision = true_positive / (true_positive + false_positive + epsilon)
        recall = true_positive / (true_positive + false_negative + epsilon)
        f1 = 2 * (precision * recall) / (precision + recall + epsilon)
        iou = true_positive / (true_positive + false_positive + false_negative + epsilon)

        # 计算 Dice 系数
        dice = 2 * true_positive / (2 * true_positive + false_positive + false_negative + epsilon)

        # 计算 MCC
        mcc_numerator = (true_positive * true_negative) - (false_positive * false_negative)
        mcc_denominator = np.sqrt(
            (true_positive + false_positive) * (true_positive + false_negative) *
            (true_negative + false_positive) * (true_negative + false_negative) + epsilon
        )

        # 如果 mcc_denominator 出现溢出或无效结果，用 epsilon 替代以避免 NaN
        if np.isinf(mcc_denominator) or np.isnan(mcc_denominator):
            mcc = 0.0
        else:
            mcc = mcc_numerator / (mcc_denominator + epsilon)

        # 将评估结果添加到列表中
        precision_list.append(precision)
        iou_list.append(iou)
        recall_list.append(recall)
        f1_list.append(f1)
        dice_list.append(dice)
        mcc_list.append(mcc)

        # 打印评估结果
        print(f"图像: {pred_file}")
        print(f"   precision: {precision:.4f}", f"   Recall: {recall:.4f}", f"   F1 Score: {f1:.4f}", f"   IOU: {iou:.4f}", f"   Dice: {dice:.4f}", f"   MCC: {mcc:.4f}")

    # 计算平均评估指标
    avg_precision = np.mean(precision_list)
    avg_iou = np.mean(iou_list)
    avg_recall = np.mean(recall_list)
    avg_f1 = np.mean(f1_list)
    avg_dice = np.mean(dice_list)
    avg_mcc = np.mean(mcc_list)

    # 打印平均评估结果
    print("平均值")
    print(f"   precision: {avg_precision:.4f}", f"   Recall: {avg_recall:.4f}", f"   F1 Score: {avg_f1:.4f}", f"   IOU: {avg_iou:.4f}", f"   Dice: {avg_dice:.4f}", f"   MCC: {avg_mcc:.4f}")

# 使用该函数时，请提供预测文件和真实标注文件的路径
# evaluate_predictions2('path_to_prediction_dir', 'path_to_true_dir')


