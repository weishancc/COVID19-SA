# -*- coding: utf-8 -*-

"""

Return what bert model need as a batch

- tokens_tensors  : (batch_size, max_seq_len_in_batch)
- segments_tensors: (batch_size, max_seq_len_in_batch)
- masks_tensors   : (batch_size, max_seq_len_in_batch)
- label_ids       : (batch_size)

"""

import torch
from torch.nn.utils.rnn import pad_sequence

# "samples" is a list whose elements are sample from "CovidDataset"
#  where each sample has:
#
# - tokens_tensor
# - segments_tensor
# - label_tensor
#
# We adopted zero padding to "tokens_tensor", "label_tensor", and
# finally generated "mask_tensor"

def create_mini_batch(samples):
    tokens_tensors = [s[0] for s in samples]
    segments_tensors = [s[1] for s in samples]
    
    if samples[0][2] is not None:
        label_ids = torch.stack([s[2] for s in samples]) # s[2] is label_tensor
    else:
        label_ids = None
    
    # Zero padding
    tokens_tensors = pad_sequence(tokens_tensors, 
                                  batch_first=True)
    segments_tensors = pad_sequence(segments_tensors, 
                                    batch_first=True)
    
    # Let Bert attends the position of tokens which is not '0'
    masks_tensors = torch.zeros(tokens_tensors.shape, 
                                dtype=torch.long)
    masks_tensors = masks_tensors.masked_fill(
        tokens_tensors != 0, 1)
    
    return tokens_tensors, segments_tensors, masks_tensors, label_ids


