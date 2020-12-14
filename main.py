import torch
import pandas as pd
import os
import Batch
import predict
import preprocessing
from transformers import BertTokenizer
from CovidDataset import CovidDataset
from torch.utils.data import DataLoader
from transformers import BertForSequenceClassification

os.environ['CUDA_VISIBLE_DEVICE'] = '3'

# Preprocess training datea 
filepath = os.path.join('./', 'fake_data', 'train.csv')
df_train = preprocessing.preprocess(filepath, 'train')

# Preprocess testing data
filepath = os.path.join('./', 'fake_data', 'test.csv')
df_train = preprocessing.preprocess(filepath, 'test')

PRETRAINED_MODEL_NAME = 'bert-base-uncased'
tokenizer = BertTokenizer.from_pretrained(PRETRAINED_MODEL_NAME, do_lower_case = True)
train_set = CovidDataset('train', tokenizer = tokenizer)

sample_idx = 1
comment, label = train_set.df.iloc[sample_idx].values
tokens_tensor, segments_tensor, label_tensor = train_set[sample_idx]
tokens = tokenizer.convert_ids_to_tokens(tokens_tensor)
combined_text = " ".join(tokens)

print(f"""[Original]\n
Comment: {comment}
Label: {label}

--------------------

[Coverted tensors]\n
tokens_tensor  ：{tokens_tensor}

segments_tensor：{segments_tensor}

label_tensor   ：{label_tensor}

--------------------

[Original tokens_tensors]\n
{combined_text}
\n""")

# DataLoader returned 64 samples at a time
# "collate_fn" parameter defined the batch output
BATCH_SIZE = 64
train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, 
                         collate_fn = Batch.create_mini_batch)
data = next(iter(train_loader))
tokens_tensors, segments_tensors, masks_tensors, label_ids = data

print(f"""
tokens_tensors.shape = {tokens_tensors.shape} 
{tokens_tensors}
------------------------
segments_tensors.shape = {segments_tensors.shape}
{segments_tensors}
------------------------
masks_tensors.shape = {masks_tensors.shape}
{masks_tensors}
------------------------
label_ids.shape = {label_ids.shape}
{label_ids}
""")


NUM_LABELS = 3
model = BertForSequenceClassification.from_pretrained(
    PRETRAINED_MODEL_NAME, num_labels = NUM_LABELS)

device = torch.device('cuda:3' if torch.cuda.is_available() else 'cpu')
print('device:', device)
model = model.to(device)

# Numbers of parameters
model_params = [p for p in model.parameters() if p.requires_grad]
clf_params = [p for p in model.classifier.parameters() if p.requires_grad]

print(f"""
Parameters of total classifier(Bert + Linear)：{sum(p.numel() for p in model_params)}
Parameters of linear classifier：{sum(p.numel() for p in clf_params)}
""")


# Let's begin to train and fine-tune
model.train()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
EPOCHS  = 6

for epoch in range(EPOCHS): 
    running_loss = 0.0 
    
    for data in train_loader:       
        tokens_tensors, segments_tensors, masks_tensors, labels = [t.to(device) for t in data]
        optimizer.zero_grad()    
        
        # forward pass
        outputs = model(input_ids = tokens_tensors, 
                        token_type_ids = segments_tensors, 
                        attention_mask = masks_tensors, 
                        labels = labels)
        # ouput: (loss), (logits), (hidden_states), (attentions)
        loss = outputs[0]
        loss.backward()
        optimizer.step()

        # Record current batch loss
        running_loss += loss.item()
        
    # Compute acc
    _, acc = predict.get_predictions(model, train_loader, compute_acc = True)
    
    print(f"{'Epoch':^7} | {'Training Loss':^12} | {'Acc':^9}")
    print("-" * 70)   
    print(f"{epoch + 1:^7} | {running_loss:^12.6f} | {acc:^9.2f}")
    
print('\nTraining complete!')

# inference
test_set = CovidDataset('test', tokenizer = tokenizer)
test_loader = DataLoader(test_set, batch_size = 64, 
                        collate_fn = Batch.create_mini_batch)
predictions = predict.get_predictions(model, test_loader)

# Ouput testing retult to csv
df = pd.DataFrame({'Category': predictions.tolist()})
df_pred = pd.concat([test_set.df.loc[:, ['Comment']], 
                          df.loc[:, 'Category']], axis = 1)
df_pred.to_csv('bert_1_prec_training_samples.csv', index = False)

