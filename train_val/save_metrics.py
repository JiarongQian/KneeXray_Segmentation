def save_metrics(train_losses, val_losses, ious, recalls, f1_scores, precisions,metrics_path):
    with open(metrics_path, 'w') as f:
        f.write("Epoch\tTrain Loss\tVal Loss\tIoU\tRecall\tF1 Score\tPrecision\n")
        for i in range(len(train_losses)):
            f.write(f"{i+1}\t{train_losses[i]}\t{val_losses[i]}\t{ious[i]}\t{recalls[i]}\t{f1_scores[i]}\t{precisions[i]}\n")