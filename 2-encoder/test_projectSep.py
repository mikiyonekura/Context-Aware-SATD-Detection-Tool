import train
import predict_projectSep as predict
from transformers import RobertaTokenizerFast, RobertaForSequenceClassification, Trainer, TrainingArguments


# trained_model, tokenizer = train.train()
tokenizer = RobertaTokenizerFast.from_pretrained("microsoft/codebert-base")
#自作のtrained_model
trained_model = RobertaForSequenceClassification.from_pretrained("/work/miki-yo/trainedNew/trained_model-Merge--9--2input--SEP")

# コメントを用意します

# ファイルからデータを読み込む
with open('/work/miki-yo/MySatdDetector/datasetNew/1-original/data--Ant-Hivernate-SQ.txt', 'r') as f:
    comments = [line.strip() for line in f.readlines()]  # 各行を読み込み、改行文字を取り除く

with open('/work/miki-yo/MySatdDetector/datasetNew/1-under/under--Ant-Hivernate-SQ.txt', 'r') as f:
    codes = [line.strip() for line in f.readlines()]  # 各行を読み込み、改行文字を取り除く

with open('/work/miki-yo/MySatdDetector/datasetNew/1-label/label--Ant-Hivernate-SQ.txt', 'r') as f:
    # "positive" を 1 に、"false" を 0 にマッピング
    labels = [1 if line.strip() == 'positive' else 0 for line in f.readlines()]  

data = [{'comment': c, 'code':d, 'label': l} for c, d, l in zip(comments,codes, labels)]
for i in data:
    print(i)

# SATDかどうか判定します
predict.predict_satd(data, trained_model, tokenizer)

