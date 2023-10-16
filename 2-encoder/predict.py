from train import tokenizer, model, prepare_data, SATDDataset, training_args, trainer 
import torch

def predict_satd(comment, code):
    text = comment + " " + code
    inputs = tokenizer(text, truncation=True, padding=True, return_tensors="pt").to('mps')
    
    # GPUが利用可能であれば使用します
    if torch.cuda.is_available():
        inputs = {k: v.to('mps') for k, v in inputs.items()}
        model.to('mps')
    
    # モデルの推論モードを設定し、出力を得ます
    model.eval()
    with torch.no_grad():
        outputs = model(**inputs)
    
    # 出力からsoftmaxを取り、確率を得ます
    probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1)
    
    # 非SATDとSATDの確率を得ます
    non_satd_prob = probabilities[0][0].item()
    satd_prob = probabilities[0][1].item()
    
    return {"non_satd_prob": non_satd_prob, "satd_prob": satd_prob}


if __name__ == "__main__":
    print("import predict")