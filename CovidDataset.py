# -*- coding: utf-8 -*-

"""

Turn #Covid-19 of ig posts into Bert input format, which contains:

- tokens_tensor
- segments_tensor
- label_tensor

"""

import torch
import pandas as pd
from torch.utils.data import Dataset
    
class CovidDataset(Dataset):
    # Init
    def __init__(self, mode, tokenizer):
        self.mode = mode
        self.df = pd.read_csv(mode + '.tsv', sep = '\t').fillna('')
        self.len = len(self.df)
        self.tokenizer = tokenizer  # BERT tokenizer
    
    # Return data for train/test
    def __getitem__(self, idx):
        if self.mode == 'test':
            comment, = self.df.iloc[idx, :].values
            label_tensor = None
        else:
            comment, label = self.df.iloc[idx, :].values
            label_tensor = torch.tensor(label)
            
        # Establish tokens_tensor
        word_pieces = ['[CLS]']
        comment = self.tokenizer.tokenize(comment)
        word_pieces += comment
        len_w = len(word_pieces)
               
        # Covert token series to id series
        ids = self.tokenizer.convert_tokens_to_ids(word_pieces)
        tokens_tensor = torch.tensor(ids)
        
        # We have only a sentence so, segments_tensor contains only 1
        segments_tensor = torch.tensor([1] * len_w, dtype=torch.long)
        
        return (tokens_tensor, segments_tensor, label_tensor)
    
    def __len__(self):
        return self.len
