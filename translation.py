import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math, copy,time

from torch.autograd import Variable
import matplotlib.pyplot as plt
import seaborn
from tqdm.auto import tqdm

from Transformer import CustomTransformer , subsequent_mask

import os
os.system("pip install transformers")
from transformers import AutoTokenizer
device = "cuda" if torch.cuda.is_available() else "cpu"

class Config:
  tokenizer_src = AutoTokenizer.from_pretrained("bert-base-uncased")
  tokenizer_tgt = AutoTokenizer.from_pretrained("monsoon-nlp/hindi-bert")
  pad_token_src = tokenizer_src.convert_tokens_to_ids((tokenizer_src.pad_token))
  pad_token_tgt = tokenizer_tgt.convert_tokens_to_ids((tokenizer_tgt.pad_token))
  
  decoder_layer = 2
  encoder_layer = 2
  max_len = 128
  d_model = 128
  d_ff = 256
  head = 8
  train_batch_size = 64
  val_batch_size = 32
  save_dir = "./"
  epoch = 20

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

src_vocab = Config.tokenizer_src.vocab_size
tgt_vocab =Config.tokenizer_tgt.vocab_size 
Transformer = CustomTransformer(src_vocab , tgt_vocab , Config.d_model  , Config.d_ff , Config.head , Config.encoder_layer , Config.decoder_layer)

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

class CustomDataset:
  def __init__(self , src , tgt):
    self.src = src
    self.tgt = tgt

  def __len__(self):
    return len(self.src)

  def __getitem__(self , item):
    s_inp = preprocess(self.src[item] , Config.tokenizer_src , Config.max_len)
    t_inp = preprocess(self.tgt[item] , Config.tokenizer_tgt , Config.max_len)
    s_ids = s_inp['input_ids']
    t_ids = t_inp['input_ids'][:-1]
    s_mask = (s_ids != Config.pad_token_src).unsqueeze(-2)
    t_mask = (t_ids != Config.pad_token_tgt).unsqueeze(-2)
    t_mask = t_mask & subsequent_mask(t_ids.size(-1))
    target = t_inp['input_ids'][1:]

    return {"s_ids" : s_ids , "t_ids":t_ids ,
            "s_mask":s_mask , "t_mask":t_mask , "target":target}


os.system("pip install datasets")
from datasets import load_dataset
dataset = load_dataset("cfilt/iitb-english-hindi")

df = dataset['train'].to_pandas()
df = df.iloc[:110000]
df['english'] = df['translation'].apply(lambda x : x['en'])
df['hindi'] = df['translation'].apply(lambda x : x['hi'])
eng = df['english'].values.tolist()
hin = df['hindi'].values.tolist()

from sklearn.model_selection import train_test_split
eng_train , eng_val , hin_train , hin_val = train_test_split(eng ,hin , test_size = .1)

dataset_train = CustomDataset(eng_train , hin_train)
dataset_val = CustomDataset(eng_val , hin_val)

train_loader = torch.utils.data.DataLoader(dataset_train , Config.train_batch_size , shuffle=True)
val_loader = torch.utils.data.DataLoader(dataset_val , Config.val_batch_size)

opt = torch.optim.AdamW(lr = 1e-3 , params=Transformer.parameters())

#ignoring the pad tokens in loss calculation
loss  = torch.nn.CrossEntropyLoss(ignore_index = Config.pad_token_tgt)

def greedy_decode(model , src , src_mask , max_len , start_symbol , device):
  model.eval()
  with torch.no_grad():
    memory = model.encode(src,src_mask)

    ys = torch.ones(1,1).fill_(start_symbol).type_as(src.data).to(device)
    for i in range(max_len -1):
      t_mask = Variable(subsequent_mask(ys.size(1)).type_as(src.data)).unsqueeze(0).to(device)
      tgt = Variable(ys).to(device)
      out = model.decode(memory , src_mask  ,tgt , t_mask )

      prob = model.generator(out[:,-1] , softmax = True)

      _ , next_word = torch.max(prob , dim=1)

      next_word = next_word.data[0]


      ys =torch.cat([ys , torch.ones(1,1).type_as(src.data).fill_(next_word)] , dim=1)

  return ys

def pre(text , device):
  s_inp = preprocess(text , Config.tokenizer_src , Config.max_len)
  s_ids = s_inp['input_ids'].view(1,-1).to(device)
  s_mask = (s_ids != Config.pad_token_src).unsqueeze(-2).to(device)

  return s_ids,s_mask

def inference(text , model , device):
  inp ,mask = pre(text,device)

  out_ids = greedy_decode(model , inp ,mask , 15 ,3  , device)

  out = Config.tokenizer_tgt.decode(out_ids[0])
  print("*"*20 , "Example" , '*'*20)
  print("original --- " , text)
  print("Translated --- " , out)

def train(model , train_loader , val_loader  , device):
  losses_train = AverageMeter()
  losses_val = AverageMeter()


  best_score = np.inf
  for i in range(Config.epoch):
    model.train()
    tk0 = tqdm(train_loader, total=len(train_loader))

    for bi , data in enumerate(tk0):

      for i,j in data.items():
        data[i] = data[i].to(device)

      model.zero_grad()
      yp = model(data['s_ids'] , data['t_ids'] , data['s_mask'] , data['t_mask']).permute(0,2,1)

      
      l = loss(yp , data['target'])
      l.backward()
      opt.step()

      losses_train.update(l.item(), data['s_ids'].size(0))
      tk0.set_postfix(l=losses_train.avg)
    print(f"Epoch {i} : Training loss - {losses_train.avg}")
    
    with torch.no_grad():
      tk1 = tqdm(val_loader, total=len(val_loader))
      for bi , data in enumerate(tk1):

        for i,j in data.items():
          data[i] = data[i].to(device)

        yp = model(data['s_ids'] , data['t_ids'] , data['s_mask'] , data['t_mask']).permute(0,2,1)
        l = loss(yp , data['target'])
        losses_val.update(l.item(), data['s_ids'].size(0))
        tk0.set_postfix(l=losses_val.avg)

      if(best_score > losses_val.avg ):
        best_score = losses_val.avg
        #save the best score model
        torch.save(model.state_dict(), os.path.join(Config.save_dir , "best_model.pth"))

      print(f"Epoch {i} : Validation loss - {best_score}")
      inference(eng[12] , model , device)

model = Transformer.to(device)
train(Transformer ,train_loader , val_loader , device)   
