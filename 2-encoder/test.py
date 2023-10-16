import train
import predict

train.trainer.train()

# コメントとコードを用意します
comment = "This is a hack, need to fix in future"
code = "int x = y / 0;"

# SATDかどうか判定します
result = predict.predict_satd(comment, code)

print(result)
