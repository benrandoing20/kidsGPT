import torch, json
from torch.utils.data import Dataset, DataLoader
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group
from scheduler.readability_sampler import CurriculumSampler
from model import GPT2Simple
from tokenizers import Tokenizer

class KidDataset(Dataset):
    def __init__(self, path, tok, block_size=512):
        self.data, self.grades = [], []
        for ln in open(path):
            o=json.loads(ln)
            tokens=tok.encode(f"<s_grade={o['grade']}> "+o['text']).ids
            if len(tokens)>block_size: tokens=tokens[:block_size]
            self.data.append(tokens); self.grades.append(o['grade'])
    def __len__(self): return len(self.data)
    def __getitem__(self,i):
        x=self.data[i]; return torch.tensor(x[:-1]), torch.tensor(x[1:])

def main():
    init_process_group(backend="nccl")
    rank, world = int(os.environ['RANK']), int(os.environ['WORLD_SIZE'])
    tok=Tokenizer.from_file("tokenizer.json")
    ds=KidDataset("data.jsonl", tok)
    sampler=CurriculumSampler(ds.grades, epoch=int(os.environ['LOCAL_RANK']), total_epochs=10)
    dl=DataLoader(ds, batch_size=4, sampler=sampler)
    model=GPT2Simple(tok.get_vocab_size()).cuda()
    model=DDP(model, device_ids=[rank])

    opt=torch.optim.AdamW(model.parameters(), lr=3e-4)
    scaler=torch.cuda.amp.GradScaler()

    for ep in range(10):
        sampler=CurriculumSampler(ds.grades, ep, 10)
        for xb, yb in dl:
            xb,yb=xb.cuda(),yb.cuda()
            with torch.cuda.amp.autocast():
                logits=model(xb)
                loss=nn.CrossEntropyLoss()(logits.view(-1, logits.size(-1)), yb.view(-1))
            scaler.scale(loss).backward()
            scaler.step(opt); scaler.update(); opt.zero_grad()
        if rank == 0:
            print(f"Epoch {ep} loss:", loss.item(), "| max_grade:", sampler.allowed_grade)

if __name__ == "__main__":
    main()
