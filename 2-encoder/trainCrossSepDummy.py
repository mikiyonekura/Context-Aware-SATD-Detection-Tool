from transformers import RobertaTokenizerFast, RobertaForSequenceClassification, Trainer, TrainingArguments
import torch
from sklearn.model_selection import train_test_split
import numpy as np

def calculate_weights(labels):
    count_0 = labels.count(0)
    count_1 = labels.count(1)
    
    weight_0 = 1. / count_0
    weight_1 = 1. / count_1
    
    total = weight_0 + weight_1
    weight_0 /= total
    weight_1 /= total
    
    return [weight_0, weight_1]

def train():
    model_name = 'microsoft/codebert-base'
    tokenizer = RobertaTokenizerFast.from_pretrained(model_name)
    model = RobertaForSequenceClassification.from_pretrained(model_name)

    def prepare_data(data):
        # テキストとコードの間にセパレータを追加
        dummy = "x=1"  

        combined_texts = [x['comment'] + " </s> " + dummy for x in data]
        encodings = tokenizer(combined_texts, truncation=True, padding=True, max_length=512, return_tensors='pt')
        labels = [x['label'] for x in data]
        return encodings, labels

    class SATDDataset(torch.utils.data.Dataset):
        def __init__(self, encodings, labels):
            self.encodings = encodings
            self.labels = labels

        def __getitem__(self, idx):
            item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
            item['labels'] = torch.tensor(self.labels[idx])
            return item

        def __len__(self):
            return len(self.labels)

    with open('/work/miki-yo/MySatdDetector/datasetNew/1-original/data--Merge--9.txt', 'r') as f:
        comments = [line.strip() for line in f.readlines()]

    with open('/work/miki-yo/MySatdDetector/datasetNew/1-under/1-under--Merge--9.txt', 'r') as f:
        codes = [line.strip() for line in f.readlines()]

    with open('/work/miki-yo/MySatdDetector/datasetNew/1-label/label--Merge--9.txt', 'r') as f:
        labels = [1 if line.strip() == 'positive' else 0 for line in f.readlines()]  

    data = [{'comment': c, 'code': d, 'label': l} for c, d, l in zip(comments, codes, labels)]
    train_data, valid_data = train_test_split(data, test_size=0.2, random_state=42)

    train_encodings, train_labels = prepare_data(train_data)
    valid_encodings, valid_labels = prepare_data(valid_data)
    train_dataset = SATDDataset(train_encodings, train_labels)
    valid_dataset = SATDDataset(valid_encodings, valid_labels)

    weights = calculate_weights(train_labels)
    weights = torch.tensor(weights).to('cuda' if torch.cuda.is_available() else 'cpu')

    model.to('cuda' if torch.cuda.is_available() else 'cpu')
    model.loss_fct = torch.nn.CrossEntropyLoss(weight=weights)

    training_args = TrainingArguments(
        output_dir='./results',
        num_train_epochs=3,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=16,
        warmup_steps=500,
        weight_decay=0.01
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset
    )

    trainer.train()
    trainer.save_model("./trainedNew/trained_model-Merge--9--2input--SEP--Dummy")

if __name__ == '__main__':
    train()
