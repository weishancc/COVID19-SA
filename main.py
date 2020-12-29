import os
import batch
import preprocessing
import train
import predict
import pandas as pd
from matplotlib import pyplot as plt
from covidDataset import CovidDataset
from torch.utils.data import DataLoader
from transformers import BertForSequenceClassification, BertTokenizer
from sklearn.model_selection import train_test_split
from torch import device, cuda, save
from collections import defaultdict

os.environ['CUDA_VISIBLE_DEVICE'] = '0'

# Preprocess training data 
train_file = os.path.join('./', 'TwitterPost', 'train.csv')
df_train, map_en = preprocessing.preprocess(train_file)


# Preprocess testing data
test_file = os.path.join('./', 'TwitterPost', 'test.csv')
df_test, map_en = preprocessing.preprocess(test_file)

# Load bert model and tokenizer
PRETRAINED_MODEL_NAME = 'bert-base-uncased'
tokenizer = BertTokenizer.from_pretrained(PRETRAINED_MODEL_NAME, do_lower_case=True)


# Self define Covid-train-Dataset and check the first sample
# i.e., converted (tokens_tensor, segments_tensor, label_tensor)
train_set = CovidDataset(df_train, tokenizer=tokenizer)
label, text = train_set.df.iloc[0].values
tokens_tensor, segments_tensor, label_tensor = train_set[0]


# Deduction to original text
tokens = tokenizer.convert_ids_to_tokens(tokens_tensor)
combined_text = ' '.join(tokens)

print(f"""[Original]\n
Text: {text}
Label: {label}

--------------------

[Coverted tensors]\n
tokens_tensor  ：{tokens_tensor}

segments_tensor：{segments_tensor}

label_tensor   ：{label_tensor}

#--------------------
#
#[Original tokens_tensors]\n
#{combined_text}
#\n""")


# DataLoader returned 64 samples at a time,
# "collate_fn" parameter defined the batch output
BATCH_SIZE = 64
train_set, val_set = train_test_split(train_set, test_size=0.1, random_state=2000)
train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, collate_fn=batch.create_mini_batch)
val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, collate_fn=batch.create_mini_batch)
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


# Fine-tune task is "BertForSequenceClassification"
model = BertForSequenceClassification.from_pretrained(
    PRETRAINED_MODEL_NAME, num_labels=4)


# Numbers of parameters
model_params = [p for p in model.parameters() if p.requires_grad]
clf_params = [p for p in model.classifier.parameters() if p.requires_grad]

print(f"""
Parameters of total classifier(Bert + Linear)：{sum(p.numel() for p in model_params)}
Parameters of linear classifier：{sum(p.numel() for p in clf_params)}
""")


## Let's begin to train and fine-tune
device = device('cuda:0' if cuda.is_available() else 'cpu')
print('device:', device)
model = model.to(device)
print('\n###Start training###\n')
print(f"{'Epoch':^7} | {'Train loss':^12} | {'Train accuracy':^9} |{'Val loss':^12} | {'Val accuracy':^9} |")
print("-" * 70)   

EPOCHS = 4
history = defaultdict(list)
for epoch in range(EPOCHS):   
    best_accuracy = 0

    # Training
    train_acc, train_loss = train.train_epoch(model, train_loader, device)
    print(f"{epoch + 1:^7} | {train_loss:^12.6f} | {train_acc:^15.2f}", end='')     
    
    # Evaluating
    val_acc, val_loss = train.eval_epoch(model, val_loader, device)
    print(f"| {val_loss:^11.6f} | {val_acc:^14.2f}")     
    print()

    history['train_acc'].append(train_acc)
    history['train_loss'].append(train_loss)
    history['val_acc'].append(val_acc)
    history['val_loss'].append(val_loss)

    # Save the best model
    if val_acc > best_accuracy:
        save(model.state_dict(), 'best_model_state.bin')

print('Training complete!')


# Plot the result
plt.plot(history['train_acc'], label='train acc')
plt.plot(history['val_acc'], label='val acc')
plt.title('Accuracy history')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend()
plt.ylim([0, 1])
plt.grid()
plt.savefig('acc_history.png')
plt.clf()

plt.plot(history['train_loss'], label='train loss')
plt.plot(history['val_loss'], label='val loss')
plt.title('Loss history')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend()
plt.grid()
plt.savefig('loss_history.png')

# Inference with test set
test_set = CovidDataset(df_test, tokenizer=tokenizer)
test_loader = DataLoader(test_set, batch_size=256, 
                        collate_fn=batch.create_mini_batch)
predictions = predict.get_predictions(model, test_loader, device)   ### Currently have a bug here, why prediction get tuple wite 2 same predicted results?


# Concat predition to .csv
df_pred = df_test
df_pred['prediction'] = predictions[0].tolist()
df_pred.to_csv('predict.csv', index=False)
