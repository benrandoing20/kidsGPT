import torch
import json
from torch.utils.data import Dataset

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