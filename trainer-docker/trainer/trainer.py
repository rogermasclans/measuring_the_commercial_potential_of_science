import transformers
from transformers import BertModel, BertTokenizer, AdamW, get_linear_schedule_with_warmup, AutoModel, AutoTokenizer
import torch
from torch import nn, optim
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader, TensorDataset
import torch.nn.functional as F
import json
import os
import numpy as np
import pandas as pd
from pylab import rcParams
from matplotlib import rc
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.utils import shuffle
from collections import defaultdict
from textwrap import wrap
from google.cloud import storage
import subprocess
import time
import sys

MAX_LEN = 512
TEST_SIZE = 0.25 

# Dynamic data: based on command line argument
args = sys.argv
dynamic_data = "training_data_small_test.csv"
local_model_name = 'small_test_model.bin'
bucket_name = 'scicompot'
epochs = 5
drop_out_rate = 0.3
lr = 2e-5 
model_used = 'scibert'
training_data_folder = 'training-data'
batch_size = 16

if len(args) > 1:
    dynamic_data = args[1]

if len(args) > 2:
    local_model_name = args[2]

if len(args) > 3:
    bucket_name = args[3]

if len(args) > 4:
    epochs = int(args[4])

if len(args) > 5:
    drop_out_rate = float(args[5])

if len(args) > 6:
    lr = float(args[6])

if len(args) > 7:
    model_used = args[7]

if len(args) > 8:
    training_data_folder = args[8]

if len(args) > 9:
    batch_size = int(args[9])

data_path = 'gs://' + bucket_name + '/' + training_data_folder + '/'
final_data = data_path + dynamic_data

rcParams['figure.figsize'] = 12, 8
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

# read the data
df = pd.read_csv(final_data, encoding='utf-8', on_bad_lines='skip', engine="python")

print(sum(df['sentiment']==0))
print(sum(df['sentiment']==1))

if model_used == 'bert':
  pre_trained_model_name = 'bert-base-uncased'
  print("Using BERT base model.")
  tokenizer = BertTokenizer.from_pretrained(pre_trained_model_name)
elif model_used == 'scibert':
  pre_trained_model_name = 'scibert_scivocab_uncased'
  print("Using SciBERT model.")
  tokenizer = AutoTokenizer.from_pretrained('allenai/scibert_scivocab_uncased')
else:
  pre_trained_model_name= 'errorReturn'
  print("Unsupported model. Please choose either 'bert' or 'scibert'.")
  tokenizer = AutoTokenizer.from_pretrained('error')
  
class GPReviewDataset(Dataset):

  def __init__(self, reviews, targets, tokenizer, max_len): #create constructor/methods. assign variables:
    self.reviews = reviews
    self.targets = targets
    self.tokenizer = tokenizer
    self.max_len = max_len
  
  def __len__(self): #returns number of reviews, length of data set
    return len(self.reviews) 
  
  def __getitem__(self, item): #takes the index of the element from the data set 
    review = str(self.reviews[item])[:MAX_LEN]
    target = self.targets[item]
    
    # Replace problematic character with a space
    review = review.replace('\x85', ' ')

    encoding = self.tokenizer.encode_plus(
      review,
      add_special_tokens=True,
      truncation=True,
      max_length=self.max_len,
      return_token_type_ids=False,
      padding='max_length',
      return_attention_mask=True,
      return_tensors='pt'
    )

    return {
      'review_text': review,
      'input_ids': encoding['input_ids'].flatten(),
      'attention_mask': encoding['attention_mask'].flatten(),
      'targets': torch.tensor(target, dtype=torch.long) #sentiment for the review, converted into tensor of type long bc is a classification problem
    }

df_train, df_test = train_test_split(df, test_size=TEST_SIZE, random_state=RANDOM_SEED)
df_val, df_test = train_test_split(df_test, test_size=0.5, random_state=RANDOM_SEED)

print(df_train.shape)
print(df_val.shape)
print(df_test.shape)

def create_data_loader(df, tokenizer, max_len, batch_size): #take dataframe, tokenizer and other vars and  
  ds = GPReviewDataset(
    reviews=df.content.to_numpy(), #get numpy values of content
    targets=df.sentiment.to_numpy(), #get targets/sentiments values - integers
    tokenizer=tokenizer, #pass tokenizer
    max_len=max_len
  )

  return DataLoader(
    ds,
    batch_size=batch_size,
    num_workers=4
  )

train_data_loader = create_data_loader(df_train, tokenizer, MAX_LEN, batch_size)
val_data_loader = create_data_loader(df_val, tokenizer, MAX_LEN, batch_size)
test_data_loader = create_data_loader(df_test, tokenizer, MAX_LEN, batch_size)

data = next(iter(train_data_loader))
data.keys()

print(data['input_ids'].shape)
print(data['attention_mask'].shape)
print(data['targets'].shape)

if model_used == 'bert':
  bert_model = BertModel.from_pretrained(pre_trained_model_name, return_dict=False) #return_dict=False needed for compatibility. Check https://huggingface.co/docs/transformers/migration
else: 
  bert_model = AutoModel.from_pretrained('allenai/scibert_scivocab_uncased', return_dict=False)

if model_used == "bert":
  class SentimentClassifier(nn.Module):

    def __init__(self, n_classes):
      super(SentimentClassifier, self).__init__()
      self.bert = BertModel.from_pretrained(pre_trained_model_name, return_dict=False)
      self.drop = nn.Dropout(p=drop_out_rate)
      self.out = nn.Linear(self.bert.config.hidden_size, n_classes)
    
    def forward(self, input_ids, attention_mask):
      _, pooled_output = self.bert(
        input_ids=input_ids,
        attention_mask=attention_mask
      )
      output = self.drop(pooled_output)
      return self.out(output)
else:
  class SentimentClassifier(nn.Module):

    def __init__(self, n_classes):
      super(SentimentClassifier, self).__init__()
      self.bert = AutoModel.from_pretrained('allenai/scibert_scivocab_uncased', return_dict=False)
      self.drop = nn.Dropout(p=drop_out_rate)
      self.out = nn.Linear(self.bert.config.hidden_size, n_classes)
    
    def forward(self, input_ids, attention_mask):
      _, pooled_output = self.bert(
        input_ids=input_ids,
        attention_mask=attention_mask
      )
      output = self.drop(pooled_output)
      return self.out(output)

class_names = ['No patent renewal', 'Patent renewal']

model = SentimentClassifier(len(class_names))
model = model.to(device)

input_ids = data['input_ids'].to(device)
attention_mask = data['attention_mask'].to(device)

print(input_ids.shape) # batch size x seq length
print(attention_mask.shape) # batch size x seq length

F.sigmoid(model(input_ids, attention_mask))

optimizer = AdamW(model.parameters(), lr=lr, correct_bias=False)
total_steps = len(train_data_loader) * epochs

scheduler = get_linear_schedule_with_warmup(
  optimizer,
  num_warmup_steps=0,
  num_training_steps=total_steps
)

loss_fn = nn.CrossEntropyLoss().to(device)

def train_epoch(
  model, 
  data_loader, 
  loss_fn, 
  optimizer, 
  device, 
  scheduler, 
  n_examples
):
  model = model.train()

  losses = []
  correct_predictions = 0
  
  for d in data_loader:
    print('train epoch loop')
    
    input_ids = d["input_ids"].to(device)
    attention_mask = d["attention_mask"].to(device)
    targets = d["targets"].to(device)

    outputs = model(
      input_ids=input_ids,
      attention_mask=attention_mask
    )

    _, preds = torch.max(outputs, dim=1)
    loss = loss_fn(outputs, targets)

    correct_predictions += torch.sum(preds == targets)
    losses.append(loss.item())

    loss.backward()
    nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()
    scheduler.step()
    optimizer.zero_grad()

  return correct_predictions.double() / n_examples, np.mean(losses)

def eval_model(model, data_loader, loss_fn, device, n_examples):
  model = model.eval()

  losses = []
  correct_predictions = 0

  with torch.no_grad():
    for d in data_loader:
      input_ids = d["input_ids"].to(device)
      attention_mask = d["attention_mask"].to(device)
      targets = d["targets"].to(device)

      outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask
      )
      _, preds = torch.max(outputs, dim=1)

      loss = loss_fn(outputs, targets)

      correct_predictions += torch.sum(preds == targets)
      losses.append(loss.item())

  return correct_predictions.double() / n_examples, np.mean(losses)

print(model_used, MAX_LEN, TEST_SIZE, batch_size, epochs, lr, drop_out_rate)

history = defaultdict(list)
best_accuracy = 0

for epoch in range(epochs):
    print(f'Epoch {epoch + 1}/{epochs}')
    print('-' * 10)

    start_time = time.time()

    train_acc, train_loss = train_epoch(model, train_data_loader, loss_fn, optimizer, device, scheduler, len(df_train))

    print(f'Train loss {train_loss} accuracy {train_acc}')

    val_acc, val_loss = eval_model(model, val_data_loader, loss_fn, device, len(df_val))

    print(f'Val loss {val_loss} accuracy {val_acc}')

    history['train_acc'].append(train_acc)
    history['train_loss'].append(train_loss)
    history['val_acc'].append(val_acc)
    history['val_loss'].append(val_loss)

    if val_acc > best_accuracy:
      torch.save(model.state_dict(), local_model_name)
      best_accuracy = val_acc

    end_time = time.time()
    epoch_time = end_time - start_time
    print(f"Time taken for epoch {epoch + 1}: {epoch_time:.2f} seconds")

print("Saving in Google Cloud Buckets...")
client = storage.Client()
bucket = client.bucket(bucket_name)
blob = bucket.blob('vertex_custom_models/' + local_model_name)
blob.upload_from_filename(local_model_name)
print("Saved completed.")
