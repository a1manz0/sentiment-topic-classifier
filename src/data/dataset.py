import torch
from torch.utils.data import Dataset
import pandas as pd

class TextClassificationDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=512):
        # Преобразуем тексты в список если это pandas Series
        self.texts = texts
        
        self.labels = labels
            
        self.encodings = tokenizer(
            texts, 
            truncation=True,
            padding=True,
            max_length=max_length,
            return_tensors="pt"
        )

    def __getitem__(self, idx):
        item = {key: val[idx] for key, val in self.encodings.items()}
        # Просто преобразуем число в тензор
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels) 