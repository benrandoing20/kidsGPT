import torch, json, os
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from scheduler.readability_sampler import CurriculumSampler
from model.model import GPT2Simple
from tokenizers import Tokenizer

# Disable distributed training to avoid initialization errors
os.environ['MASTER_ADDR'] = 'localhost'
os.environ['MASTER_PORT'] = '12355'
os.environ['WORLD_SIZE'] = '1'
os.environ['RANK'] = '0'

def collate_fn(batch):
    """Custom collate function to pad sequences to the same length within a batch"""
    # Separate x and y from the batch
    x_batch, y_batch = zip(*batch)
    
    # Find the maximum length in this batch
    max_len = max(len(x) for x in x_batch)
    
    # Pad sequences to max_len
    x_padded = []
    y_padded = []
    
    for x, y in zip(x_batch, y_batch):
        # Pad x
        if len(x) < max_len:
            x_pad = torch.cat([x, torch.zeros(max_len - len(x), dtype=x.dtype)])
        else:
            x_pad = x
        x_padded.append(x_pad)
        
        # Pad y
        if len(y) < max_len:
            y_pad = torch.cat([y, torch.zeros(max_len - len(y), dtype=y.dtype)])
        else:
            y_pad = y
        y_padded.append(y_pad)
    
    # Stack the padded tensors
    x_batch = torch.stack(x_padded)
    y_batch = torch.stack(y_padded)
    
    return x_batch, y_batch

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
    # Check if CUDA is available and set device
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Using CUDA device: {torch.cuda.get_device_name()}")
    else:
        device = torch.device("cpu")
        print("Using CPU")
    
    # Load tokenizer and dataset
    tok = Tokenizer.from_file("tokenizer/tokenizer.json")
    ds = KidDataset("data/data.jsonl", tok)
    
    # Create sampler and dataloader
    sampler = CurriculumSampler(ds.grades, epoch=0, total_epochs=10)
    dl = DataLoader(ds, batch_size=4, sampler=sampler, collate_fn=collate_fn)
    
    # Create model
    model = GPT2Simple(tok.get_vocab_size()).to(device)
    
    # Optimizer and scaler
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4)
    scaler = torch.amp.GradScaler('cuda') if device.type == "cuda" else None

    for ep in range(10):
        sampler = CurriculumSampler(ds.grades, ep, 10)
        dl = DataLoader(ds, batch_size=4, sampler=sampler, collate_fn=collate_fn)
        
        for xb, yb in dl:
            xb, yb = xb.to(device), yb.to(device)
            
            if device.type == "cuda":
                with torch.cuda.amp.autocast():
                    logits = model(xb)
                    loss = nn.CrossEntropyLoss()(logits.view(-1, logits.size(-1)), yb.view(-1))
                scaler.scale(loss).backward()
                scaler.step(opt)
                scaler.update()
            else:
                logits = model(xb)
                loss = nn.CrossEntropyLoss()(logits.view(-1, logits.size(-1)), yb.view(-1))
                loss.backward()
                opt.step()
            
            opt.zero_grad()
        
        print(f"Epoch {ep} loss: {loss.item():.4f} | max_grade: {sampler.max_grade}")

if __name__ == "__main__":
    main()
