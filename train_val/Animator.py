import matplotlib.pyplot as plt
from IPython.display import clear_output
'''这个函数的主要目的是绘制神经网络训练过程中的各种指标曲线，以便直观地观察模型在训练和验证阶段的表现。
它接收训练损失、验证损失以及一些验证阶段的评估指标（如交并比 IoU、召回率 Recall、F1 分数、准确率 Precision）等数据作为输入，
并将这些数据绘制成两条曲线：
一条是训练损失和验证损失曲线，(train_losses,val_losses)
另一条是包含多个验证指标的曲线。
最后，函数将绘制好的图形保存为 SVG 文件，
并在运行环境中进行展示（通过 plt.draw() 和 plt.pause(0.001)）。
'''

def plot_training_progress(train_losses, val_losses, ious, recalls, f1_scores, precisions, num_epochs,plot_path):
    clear_output(wait=True)
    fig, axes = plt.subplots(2, 1, figsize=(8, 10))

    # 设定颜色
    color = 'green'  # 可以选择你想要的颜色

    # Plot loss curve
    axes[0].plot(train_losses, label='Train Loss', color=color, linestyle='--')  # 虚线表示训练损失
    axes[0].plot(val_losses, label='Validation Loss', color=color, linewidth=2)  # 粗线表示验证损失
    axes[0].set_title('Training and Validation Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_ylim(0, 1)  # 根据需要调整 y 轴范围
    axes[0].set_xlim(0, num_epochs - 1)
    axes[0].legend()
    axes[0].grid(True)

    # Plot metrics curves
    axes[1].plot(ious, label='IoU')  # 可选择其他颜色
    axes[1].plot(recalls, label='Recall')  # 可选择其他颜色
    axes[1].plot(f1_scores, label='F1 Score')  # 可选择其他颜色
    axes[1].plot(precisions, label='Precision')  # 可选择其他颜色
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Metrics')
    axes[1].set_title('Validation Metrics')
    axes[1].set_ylim(0, 1)  # 根据需要调整 y 轴范围
    axes[1].set_xlim(0, num_epochs - 1)
    axes[1].legend()
    axes[1].grid(True)

    plt.tight_layout()
    plt.savefig('training_progress.svg')  # 保存为 SVG 文件
    plt.draw()
    plt.pause(0.001)  # 暂停以更新图形
    plt.savefig(plot_path)
    plt.close()
# 示例用法:
# plot_training_progress(train_losses, val_losses, ious, recalls, f1_scores, precisions, num_epochs)

