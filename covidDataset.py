# -*- coding: utf-8 -*-

"""

Turn #Covid-19 of ig posts into Bert input format, which contains:

- tokens_tensor
- segments_tensor
- label_tensor

"""

import torch
from torch.utils.data import Dataset
    
class CovidDataset(Dataset):
    # Init
    def __init__(self, df, tokenizer):
        self.df = df
        self.len = len(self.df)
        self.tokenizer = tokenizer  # BERT tokenizer
    
    # Return data for train/test
    def __getitem__(self, idx):
        if self.df.shape[0] == 618:   # test set
            _, text = self.df.iloc[idx, :].values
            label_tensor = None
        else:
            label, text = self.df.iloc[idx, :].values
            label_tensor = torch.tensor(label)
    
        # Establish tokens_tensor
        word_pieces = ['[CLS]']
        text = self.tokenizer.tokenize(text)
        word_pieces += text
        len_w = len(word_pieces)

        # Covert token series to id series
        ids = self.tokenizer.convert_tokens_to_ids(word_pieces)
        tokens_tensor = torch.tensor(ids)
        
        # We have only a sentence so, segments_tensor contains only 1
        segments_tensor = torch.tensor([1] * len_w, dtype=torch.long)
        
        return (tokens_tensor, segments_tensor, label_tensor)
    
    def __len__(self):
        return self.len
