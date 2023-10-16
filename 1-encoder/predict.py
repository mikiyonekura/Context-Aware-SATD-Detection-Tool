import torch

def predict_satd(comment, model, tokenizer):
    text = comment
    inputs = tokenizer(text, truncation=True, padding=True, return_tensors="pt")
    
    # GPUが利用可能であれば使用します
    if torch.cuda.is_available():
        inputs = {k: v.to('cuda') for k, v in inputs.items()}
        model.to('cuda')
    
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
    predict_satd("This is a hack, need to fix in future", model, tokenizer)