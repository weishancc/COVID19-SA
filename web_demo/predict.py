# -*- coding: utf-8 -*-
import torch

"""
Get the inference ouput given a DataLoader or a single sentence

"""

def get_predictions(model, data_loader, device, compute_acc=False):
    predictions = None
    correct = total = 0

    with torch.no_grad():
        for data in data_loader:
            if next(model.parameters()).is_cuda:
                data = [t.to(device) for t in data if t is not None]           
            tokens_tensors, segments_tensors, masks_tensors = data[:3]
            
            # ouputs: (logits), (hidden_states), (attentions)
            outputs = model(input_ids = tokens_tensors, 
                            token_type_ids = segments_tensors, 
                            attention_mask =masks_tensors)
            logits = outputs[0]
            _, pred = torch.max(logits.data, 1)      
            
            if compute_acc:
                labels = data[3]
                total += labels.size(0)
                correct += (pred == labels).sum().item()         
            if predictions is None:
                predictions = pred
            else:
                predictions = torch.cat((predictions, pred))
    
    return predictions, correct/total if compute_acc else predictions


# Inference with single sentence(text) instead of batch, which is used in web demo :)
def get_prediction_with_single(tokenizer, model, text, device):
    
    # Establish tokens_tensor
    word_pieces = ['[CLS]']
    text = tokenizer.tokenize(text)
    word_pieces += text
    len_w = len(word_pieces)
    

    #| tokens_tensors |#
    # Covert token series to id series, then unsqueeze tensor in dim-0(e.g., size[x, y] -> size[1, x, y])
    ids = tokenizer.convert_tokens_to_ids(word_pieces)
    tokens_tensor = torch.tensor(ids).unsqueeze(0)
    
    
    #| segments_tensor |#
    # We have only a sentence so, segments_tensor contains only 1, and also, unsqueeze the tensor
    segments_tensor = torch.tensor([1] * len_w, dtype=torch.long)
    segments_tensor = segments_tensor.unsqueeze(0)
    
               
    #| masks_tensors |#
    # Let Bert attends the position of tokens which is not '0'
    masks_tensor = torch.zeros(tokens_tensor.shape, 
                                dtype=torch.long)
    masks_tensor = masks_tensor.masked_fill(tokens_tensor != 0, 1)

    # To cuda
    tokens_tensor = tokens_tensor.to(device)
    segments_tensor = segments_tensor.to(device)
    masks_tensor = masks_tensor.to(device)

    # Now we have all input tensors, let's feed into model and then predict  
    # ouputs: (logits), (hidden_states), (attentions)
    outputs = model(input_ids = tokens_tensor, 
                    token_type_ids = segments_tensor, attention_mask = masks_tensor)
    logits = outputs[0]
    _, predtion = torch.max(logits.data, 1)      
    
    return predtion
