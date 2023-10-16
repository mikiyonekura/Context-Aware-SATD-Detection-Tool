from transformers import RobertaTokenizerFast, RobertaForSequenceClassification, Trainer, TrainingArguments
import torch
from sklearn.model_selection import train_test_split

def train():
    # モデルとトークナイザーの初期化
    model_name = 'microsoft/codebert-base'
    tokenizer = RobertaTokenizerFast.from_pretrained(model_name)
    model = RobertaForSequenceClassification.from_pretrained(model_name)

    # 学習データの準備
    def prepare_data(data):
        # ここでは、コメントとコードを連結して1つのテキストとして扱います
        texts = [f"{x['comment']} {x['code']}" for x in data]
        labels = [x['label'] for x in data]  # ラベルは0（非SATD）と1（SATD）の二値
        encodings = tokenizer(texts, truncation=True, padding=True)
        return encodings, labels

    # データセットの作成
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

    # ファイルからデータを読み込む
    with open('dataset/data--Argo-Hive-re.txt', 'r') as f:
        comments = [line.strip() for line in f.readlines()]  # 各行を読み込み、改行文字を取り除く

    with open('dataset/under--Argo-Hive-re.txt', 'r') as f:
        codes = [line.strip() for line in f.readlines()]  # 各行を読み込み、改行文字を取り除く

    with open('dataset/label--Argo-Hive-re.txt', 'r') as f:
        # "positive" を 1 に、"false" を 0 にマッピング
        labels = [1 if line.strip() == 'positive' else 0 for line in f.readlines()]  

    data = [{'comment': c,'code': d, 'label': l} for c, d, l in zip(comments, codes, labels)]

    # trainデータとvalidationデータに分ける（この例では8:2に分ける）
    train_data, valid_data = train_test_split(data, test_size=0.2, random_state=42)

    train_encodings, train_labels = prepare_data(train_data)
    valid_encodings, valid_labels = prepare_data(valid_data)
    train_dataset = SATDDataset(train_encodings, train_labels)
    valid_dataset = SATDDataset(valid_encodings, valid_labels)

    # 学習設定
    training_args = TrainingArguments(
        output_dir='./results',
        num_train_epochs=3,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=16,
        warmup_steps=500,
        weight_decay=0.01,
        # logging_dir='./logs',
    )

    # トレーナーの初期化と学習の開始
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
    )

    trainer.train()

    # モデルの保存
    trainer.save_model("./trained/trained_model-Argo-Hive-re-2")

    # # 学習したモデルをロードします
    # trained_model = RobertaForSequenceClassification.from_pretrained("trained_model")

    

if __name__ == '__main__':
    train()