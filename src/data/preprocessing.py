import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer

class DataPreprocessor:
    def __init__(self, tokenizer_name="DeepPavlov/rubert-base-cased"):
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    
    def preprocess_text(self, text):
        # Базовая предобработка текста
        text = text.lower()
        text = text.strip()
        return text
    
    def prepare_dataset(self, df, test_size=0.2, val_size=0.2):
        # Разделение данных
        train_df, temp_df = train_test_split(df, test_size=(test_size + val_size))
        val_df, test_df = train_test_split(temp_df, test_size=0.5)
        
        return train_df, val_df, test_df 