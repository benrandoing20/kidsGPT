from model import GPT2Simple
from tokenizers import Tokenizer
import torch

model = GPT2Simple(vocab_size=50257)
model.load_state_dict(torch.load("kidgpt.pt"))
model.eval()
tok = Tokenizer.from_file("tokenizer.json")

prompt = "<s_grade=3> What is gravity?"
ids = tok.encode(prompt).ids
x = torch.tensor(ids).unsqueeze(0)

with torch.no_grad():
    out = model(x)[:, -1, :].argmax(dim=-1)
    print(tok.decode(out.tolist()))
