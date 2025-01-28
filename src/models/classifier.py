import torch.nn as nn
from transformers import AutoModel

class MultiTaskClassifier(nn.Module):
    def __init__(self, model_name, num_sentiment_labels, num_topic_labels):
        super().__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(0.1)
        self.sentiment_classifier = nn.Linear(768, num_sentiment_labels)
        self.topic_classifier = nn.Linear(768, num_topic_labels)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, 
                          attention_mask=attention_mask)
        pooled_output = outputs.last_hidden_state[:, 0]
        pooled_output = self.dropout(pooled_output)
        
        return {
            'sentiment': self.sentiment_classifier(pooled_output),
            'topic': self.topic_classifier(pooled_output)
        } 