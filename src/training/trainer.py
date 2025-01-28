import torch
from tqdm import tqdm
from ..utils.helpers import save_checkpoint

class Trainer:
    def __init__(self, model, train_loader, val_loader, optimizer, 
                 sentiment_criterion, topic_criterion, device):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.sentiment_criterion = sentiment_criterion
        self.topic_criterion = topic_criterion
        self.device = device
        
    def train_epoch(self):
        self.model.train()
        total_loss = 0
        
        for batch in tqdm(self.train_loader):
            self.optimizer.zero_grad()
            
            outputs = self.model(
                input_ids=batch['input_ids'].to(self.device),
                attention_mask=batch['attention_mask'].to(self.device)
            )
            
            loss = self.calculate_loss(outputs, batch['labels'].to(self.device))
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            
        return total_loss / len(self.train_loader) 