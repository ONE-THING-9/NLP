import os
os.system("pip install transformers")
os.system("pip install datasets")

import transformers
from datasets import load_dataset
from tqdm.auto import tqdm
import numpy as np
import math
from dataclasses import dataclass
import torch
import torch.nn as nn
from torch.nn import functional as F
import GPT

class GPTConfig:
  n_head = 8
  d_model = 512
  n_layer = 2
  dropout_attn = .1
  dropout_ffn = .1
  d_inter = 512*4
  vocab_size = 50258
  max_len  = 300
  device = "cuda" if torch.cuda.is_available() else "cpu"
  pad_token = 50257
  loss_fn = torch.nn.CrossEntropyLoss(ignore_index = 50257)

class TrainConfig:
  dataset = 'wikitext'
  batch_size = 16
  device = "cuda" if torch.cuda.is_available() else "cpu"
  epoch = 2
  tokenizer = tokenizer = transformers.AutoTokenizer.from_pretrained("gpt2")
  tokenizer.add_special_tokens({'pad_token': '[PAD]'})
  max_len = 128
  train_batch_size = 32
  val_batch_size = 16
  save_dir = "./"

class AverageMeter:

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def preprocess(text , tokenizer , max_len):
  inp = tokenizer(text , padding = 'max_length' , max_length = max_len , truncation = True)

  for i , j in inp.items():
    inp[i] = torch.tensor(j)
  return inp

def collate(inputs):
      mask_len = int(inputs['attention_mask'].sum(axis=1).max())
      for k,v in inputs.items():
          inputs[k] = inputs[k][:,:mask_len]
      return inputs

class customdataset:
  def __init__(self , src ):
    self.src = src

  def __len__(self):
    return len(self.src)

  def __getitem__(self , item):

    inp = preprocess(self.src[item] , TrainConfig.tokenizer , TrainConfig.max_len)

    ids = inp['input_ids'][:-1]

    target = inp['input_ids'][1:]

    return {"ids" :ids ,  "target":target}

@torch.no_grad()
def generate(model , idx , max_token , temp = 1.0 , top_k = None):
  model.eval()
  for _ in range(max_token):
    idx_cond = idx if idx.size(1) < TrainConfig.max_len else idx[: , -TrainConfig.max_len:]
    logit , _  = model(idx_cond)
    logit  = logit[:,-1 , :]/temp
    if top_k is not None:
      v , _ = torch.topk(logit , min(top_k , logits.size(-1)))
      logit[logit < v[: , [-1]]] = float('-inf')

    probs = F.softmax(logit , dim=-1)
    idx_next = torch.multinomial(probs , num_samples = 1)
    idx = torch.cat([idx , idx_next] , dim=1)
  v = idx.detach().cpu().numpy().tolist()
  out = TrainConfig.tokenizer.decode(v[0])
  return out

def train(model , train_loader , val_loader  , device):
  losses_train = AverageMeter()
  losses_val = AverageMeter()


  best_score = np.inf
  for i in range(TrainConfig.epoch):
    model.train()
    tk0 = tqdm(train_loader, total=len(train_loader))

    for bi , data in enumerate(tk0):

      for i,j in data.items():
        data[i] = data[i].to(device)

      model.zero_grad()
      logit , loss = model(data['ids']  , data['target'])

      loss.backward()
      opt.step()

      losses_train.update(loss.item(), data['ids'].size(0))
      tk0.set_postfix(l=losses_train.avg)
    print(f"training loss - {losses_train.avg}")
    
    with torch.no_grad():
      model.eval()
      tk1 = tqdm(val_loader, total=len(val_loader))
      for bi , data in enumerate(tk1):

        for i,j in data.items():
          data[i] = data[i].to(device)

        logit , loss = model(data['ids'],data['target'])
        losses_val.update(loss.item(), data['ids'].size(0))
        tk0.set_postfix(l=losses_val.avg)
      
      xx = torch.tensor([[18165, 1893]] , dtype = torch.int).to(TrainConfig.device)
      out = generate(model , xx,15)
      
      print(f"best Validation loss - {losses_val.avg}")
    
      print("\npromt: America president ")
      print(f"generated : {out}")

      if(best_score > losses_val.avg ):
        best_score = losses_val.avg
        torch.save(model.state_dict(), os.path.join(TrainConfig.save_dir , "best_model.pth"))

      print(f"best Validation loss - {best_score}\n\n")

dataset = load_dataset("wikitext",'wikitext-103-v1')
df = dataset['train'].to_pandas()
df = df.iloc[:10000]
df['len'] = df['text'].apply(lambda x :len(x))
df_train = df[df['len'] > 20]

total = len(df.index)
df_train = df.iloc[:int(total*.9)]
df_val = df.iloc[int(total*.9):]

train_dataset = customdataset(df_train['text'].values.tolist())
val_dataset = customdataset(df_val['text'].values.tolist())

train_loader = torch.utils.data.DataLoader(train_dataset , TrainConfig.train_batch_size)
val_loader = torch.utils.data.DataLoader(val_dataset , TrainConfig.val_batch_size)

model = GPT(Config)

opt = torch.optim.AdamW(lr = 1e-3 , params=model.parameters())
loss  = torch.nn.CrossEntropyLoss(ignore_index = TrainConfig.tokenizer.pad_token_id)

model.train()
model = model.to(TrainConfig.device)
train(model ,train_loader , val_loader , TrainConfig.device)


