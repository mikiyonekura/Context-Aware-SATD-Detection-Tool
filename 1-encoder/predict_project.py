import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import torch
import preprocess as pre
from sklearn.metrics import roc_curve, auc

def predict_satd(data, model, tokenizer):
    true_labels = []
    predicted_labels = []
    predicted_probs = []  # 予測された確率を保存するためのリスト

    for d in data:
        text = d["comment"]
        #変更点
        # preprocessed_text = pre.stemmerLemmatizer(text)

        inputs = tokenizer(text, truncation=True, padding=True, return_tensors="pt")
        #変更点
        # inputs = tokenizer(preprocessed_text, truncation=True, padding=True, return_tensors="pt")
        
        # If available use GPU
        if torch.cuda.is_available():
            inputs = {k: v.to('cuda') for k, v in inputs.items()}
            model.to('cuda')
        
        # Set model in evaluation mode and get output
        model.eval()
        with torch.no_grad():
            outputs = model(**inputs)
        
        # Get probabilities from output
        probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1)
        
        # Get probability for non-SATD and SATD
        non_satd_prob = probabilities[0][0].item()
        satd_prob = probabilities[0][1].item()

        # Save the probability of SATD
        predicted_probs.append(satd_prob)


        # Predicted label is the one with the highest probability
        predicted_label = 0 if non_satd_prob > satd_prob else 1

        # Save true and predicted labels
        true_labels.append(d["label"])
        predicted_labels.append(predicted_label)

        print("---------------------------------")
        print(f"comment: {text}")
        print(f"true label: {d['label']}")
        print(f"predicted label: {predicted_label}")
        print(f"non_satd_prob: {non_satd_prob}",f"satd_prob: {satd_prob}")

    # Create confusion matrix
    cm = confusion_matrix(true_labels, predicted_labels, labels=[0, 1])

    # Plot confusion matrix as a heatmap
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion matrix')
    plt.xlabel('Predicted labels')
    plt.ylabel('True labels')
    plt.xticks([0.5,1.5], ['non-SATD', 'SATD'], rotation=0, va='center')
    plt.yticks([0.5,1.5], ['non-SATD', 'SATD'], rotation=90, va='center')
    plt.savefig("confusion_matrix.png")
    # plt.show()

    # 2. ROC曲線のためのデータを取得
    fpr, tpr, thresholds = roc_curve(true_labels, predicted_probs, pos_label=1)
    roc_auc = auc(fpr, tpr)
    #true_labels, predicted_probsをprint
    print(true_labels)
    print(predicted_probs)
    #txtとして保存
    with open('true_labels-1.txt', 'w') as f:
        for item in true_labels:
            f.write("%s\n" % item)
    
    with open('predicted_prob-1.txt', 'w') as f:
        for item in predicted_probs:
            f.write("%s\n" % item)

    # 3. ROC曲線をプロット
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.savefig("roc_curve.png")
    plt.show()

if __name__ == "__main__":
    predict_satd(data, model, tokenizer)
