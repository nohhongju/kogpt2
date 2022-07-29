import torch
from transformers import PreTrainedTokenizerFast
from transformers import PreTrainedTokenizerFast, GPT2LMHeadModel
from transformers import GPT2LMHeadModel
from transformers import PreTrainedTokenizerFast


Q_TKN = "<usr>"
A_TKN = "<sys>"
BOS = '</s>'
EOS = '</s>'
MASK = '<unused0>'
SENT = '<unused1>'
PAD = '<pad>'

tokenizer = PreTrainedTokenizerFast.from_pretrained("skt/kogpt2-base-v2", 
    bos_token=BOS, eos_token=EOS, unk_token='<unk>',  pad_token=PAD, mask_token=MASK)

#모델 불러오기
device = torch.device('cpu')
PATH = 'C:/MyProject/kogpt/kogpt/save/chatbot_v80.pt'
model = torch.load(PATH, map_location=device)

model.eval()

with torch.no_grad():
    while 1:
        q = input("user > ").strip()
        if q == "바이":
            print("chatbot > 또 만나요^^")
            break
        elif q == "책추천해줘":
            Emotion.test(self)
            continue
        a = ""
        while 1:
            input_ids = torch.LongTensor(tokenizer.encode(Q_TKN + q + SENT+ A_TKN + a)).unsqueeze(dim=0)
            pred = model(input_ids)
            pred = pred.logits
            gen = tokenizer.convert_ids_to_tokens(torch.argmax(pred, dim=-1).squeeze().numpy().tolist())[-1]
            if gen == EOS:
                break
            a += gen.replace("▁", " ")
        print("chatbot > {}".format(a.strip()))


if __name__ == '__main__':
    torch.no_grad()