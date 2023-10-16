import train
import predict
from transformers import RobertaTokenizerFast, RobertaForSequenceClassification, Trainer, TrainingArguments

# trained_model, tokenizer = train.train()
tokenizer = RobertaTokenizerFast.from_pretrained("microsoft/codebert-base")
trained_model = RobertaForSequenceClassification.from_pretrained("trained_model")

# コメントを用意します
comment = "//removed item in last position"
# SATDかどうか判定します
result = predict.predict_satd(comment, trained_model, tokenizer)
print(result)

# コメントとコードを用意します
comment = "//assert Arrays.asList(properties).contains( //    event.getPropertyName()) //  : event.getPropertyName(); // TODO: Do we really always need to do this or only if // notationProvider is null?"
# SATDかどうか判定します
result = predict.predict_satd(comment, trained_model, tokenizer)
print(result)

# コメントとコードを用意します
comment = "// Add an invisible button to be used when everything is off"
# SATDかどうか判定します
result = predict.predict_satd(comment, trained_model, tokenizer)
print(result)

# コメントとコードを用意します
comment = "// TODO: If this turns out to be a performance bottleneck, we can  // probably optimize the common case by caching our iterator and current // position, assuming that the next request will be for a greater index"
# SATDかどうか判定します
result = predict.predict_satd(comment, trained_model, tokenizer)
print(result)

# print(torch.backends.mps.is_available())