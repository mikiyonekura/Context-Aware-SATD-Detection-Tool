import matplotlib
matplotlib.use('Agg')  # バックエンドを指定
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
from sklearn.metrics import precision_recall_curve

# ファイルからのデータを読み込み、数値型に変換
with open('/work/miki-yo/true_labels-1.txt', 'r') as f:
    true_labels = [int(line.strip()) for line in f.readlines()]  # int型に変換

with open('/work/miki-yo/predicted_probs-1.txt', 'r') as f:
    predicted_probs = [float(line.strip()) for line in f.readlines()]  # float型に変換

with open('/work/miki-yo/true_labels-2.txt', 'r') as f:
    true_labels2 = [int(line.strip()) for line in f.readlines()]  # int型に変換

with open('/work/miki-yo/predicted_probs-2.txt', 'r') as f:
    predicted_probs2 = [float(line.strip()) for line in f.readlines()]  # float型に変換

#ROC曲線を描画
fpr, tpr, thresholds = roc_curve(true_labels, predicted_probs, pos_label=1)
roc_auc = auc(fpr, tpr)
fpr2, tpr2, thresholds2 = roc_curve(true_labels2, predicted_probs2, pos_label=1)
roc_auc2 = auc(fpr2, tpr2)
plt.figure(figsize=(6, 5))
plt.plot(fpr, tpr, label='1-input (area = %.2f)'%roc_auc)
plt.plot(fpr2, tpr2, label='2-input (area = %.2f)'%roc_auc2)
plt.legend()
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.savefig("roc_curveeeee.png")
plt.show()


#PR曲線を描画
precision, recall, thresholds = precision_recall_curve(true_labels, predicted_probs, pos_label=1)
plt.figure(figsize=(6, 5))
plt.plot(recall, precision, label='1-input')
precision2, recall2, thresholds2 = precision_recall_curve(true_labels2, predicted_probs2, pos_label=1)
plt.plot(recall2, precision2, label='2-input')
plt.legend()
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.savefig("pr_curveeeee.png")
plt.show()







