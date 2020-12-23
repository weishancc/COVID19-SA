# -*- coding: utf-8 -*-
import torch
import predict
import numpy as np
import torch.nn as nn
from transformers import AdamW

def train_epoch(model, data_loader, device):
    model.train()
    #optimizer = torch.optim.AdamW(model.parameters(), lr=1e-6)    
    train_loss = 0.0    
    optimizer = AdamW(model.parameters(), lr=1e-5)

    for data in data_loader:       
        tokens_tensors, segments_tensors, masks_tensors, labels = [t.to(device) for t in data]
        optimizer.zero_grad()
             
        # Forwarding, ouputs: (loss), (logits), (hidden_states), (attentions)
        outputs = model(input_ids=tokens_tensors, 
                        token_type_ids=segments_tensors, 
                        attention_mask=masks_tensors,
                        labels=labels)
        loss = outputs[0]
        train_loss += loss.item()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

    # Compute accuray and average loss
    _, train_acc = predict.get_predictions(model, data_loader, device, compute_acc=True)
    train_avg_loss = np.mean(train_loss) / len(data_loader)

    return train_acc, train_avg_loss


def eval_epoch(model, data_loader, device):
    model.eval()  
    val_loss = 0.0
    
    with torch.no_grad():
        for data in data_loader:       
            tokens_tensors, segments_tensors, masks_tensors, labels = [t.to(device) for t in data]
            
            # Forwarding, ouput: (loss), (logits), (hidden_states), (attentions)
            outputs = model(input_ids=tokens_tensors, 
                            token_type_ids=segments_tensors, 
                            attention_mask=masks_tensors, 
                            labels=labels)
            loss = outputs[0]
            val_loss += loss.item()
        
    # Compute accuray and average loss
    _, val_acc = predict.get_predictions(model, data_loader, device, compute_acc=True)
    val_avg_loss = np.mean(val_loss) / len(data_loader)

    return val_acc, val_avg_loss
