import os 
import sys
import re
import gc
import html
import unicodedata
import random
import pandas as pd
from pathlib import Path
from collections import namedtuple
import numpy as np
from typing import *
from tqdm.auto import tqdm
from scipy import stats
from sklearn.utils.extmath import softmax
from sklearn import model_selection
import warnings
warnings.filterwarnings("ignore")

os.environ['CURL_CA_BUNDLE'] = ''

#os.system("pip install iterative-stratification -q")
os.system("pip install transformers -q")
os.system("pip install accelerate -q")
os.system('pip install iterative-stratification -q')
os.system("pip install seqeval")
os.sytem("!pip install datasets")

from seqeval.metrics import classification_report
from accelerate import Accelerator
import torch
import torch.nn as nn
import torch.optim as optim

import transformers
from transformers import AdamW
from torch.cuda.amp import autocast , GradScaler
from transformers import get_cosine_schedule_with_warmup , get_cosine_with_hard_restarts_schedule_with_warmup , get_linear_schedule_with_warmup
from transformers import AutoModel , AutoTokenizer
from datasets import load_dataset



def seed_all(seed = 42):
    """
    Fix seed for reproducibility
    """
    # python RNG
    import random
    random.seed(seed)

    # pytorch RNGs
    import torch
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)

    import numpy as np
    np.random.seed(seed)

class config:
    SEED = 42
    SAVE_DIR = './output'
    MAX_LEN = 192
    MODEL = 'bert-base-uncased' #'microsoft/mdeberta-v3-base'
    EPOCHS = 5
    TRAIN_BATCH_SIZE = 8*2
    VALID_BATCH_SIZE = 4*2
    NUM_EVAL = 5000
    ENC_LR = 5e-6
    DEC_LR = 1e-5 
    WEIGHT_DECAY = 0.1
    NUM_CLASS = 9
    LOSS = nn.CrossEntropyLoss(ignore_index=-100)   

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



def preprocess_tag_1(text , label ,max_len,tokenizer,tag2id):  # with subword as "ign" tag

    tokens =tokenizer.encode_plus(text,padding = 'max_length' , truncation = True ,max_length = 128, is_split_into_words=True)
    word_ids = tokens.word_ids()

    label_ids = []
    prev_id = None
    for i in word_ids:
        if(i is None or prev_id == i):
            label_ids.append(-100)
        elif(prev_id != i):
            label_ids.append(label[i])

        # else:
        #     if(label[i][0] == 'B'):
        #         new_label = 'I' + label[i][1:]
        #         label_ids.append(new_label)
        #     else:
        #         label_ids.append(label[i])
        prev_id = i

    return {"input_ids":torch.tensor(tokens['input_ids'],dtype = torch.long) , 
            "attention_mask":torch.tensor(tokens['attention_mask'] , dtype = torch.long),
            "label":torch.tensor(label_ids , dtype = torch.long)}

def preprocess_tag_single(text  ,max_len,tokenizer,tag2id):  # with subword as "ign" tag
   
    tokens = tokenizer.encode_plus(text,padding = 'max_length' ,  truncation = True,max_length = max_len , is_split_into_words=True)
    word_ids = tokens.word_ids()
    
    label_ids = []
    prev_id = None
    return {"input_ids":torch.reshape(torch.tensor(tokens['input_ids'],dtype = torch.long) , (1,-1)) , 
            "attention_mask":torch.reshape(torch.tensor(tokens['attention_mask'] , dtype = torch.long) , (1,-1)),
            "label":label_ids,
           "word_ids":word_ids}

class Dataset:
    def __init__(self, texts, labels, tokenizer , tag2id):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.tag2id = tag2id
    
    def __len__(self):
        return len(self.texts)
        
    def __getitem__(self, idx):
        data = preprocess_tag_1(self.texts[idx], self.labels[idx] , config.MAX_LEN,self.tokenizer,self.tag2id)
        return data

class Head(nn.Module):
    def __init__(self, d0, d1):
        super().__init__()
        self.W = nn.Linear(d0, d1)
        self.V = nn.Linear(d1, config.NUM_CLASS)

    def forward(self, x):
        x = self.W(x)
        x = nn.Tanh()(x)
        x = self.V(x)
        # x = nn.Softmax(dim=1)(x)
        return x

class CustomModel(nn.Module):
    def __init__(self, model_name, config,vocab_sz,):
        super().__init__()
        self.config = config
        self.model_name = model_name
        self.model = transformers.AutoModel.from_pretrained(model_name)

        self.model.resize_token_embeddings(vocab_sz)

        self.head = Head(config.hidden_size, config.hidden_size)      
    def forward(
        self, input_ids=None,attention_mask=None,):
        outputs = self.model(input_ids,attention_mask=attention_mask,)
        sequence_output = outputs[0]
        out = self.head(sequence_output)
        return out

class Trainer:
    def __init__(self, model, train_data_loader, valid_data_loader, optimizer, accelerator):
        
        model, train_data_loader, valid_data_loader, optimizer = accelerator.prepare(model, train_data_loader, valid_data_loader, optimizer)
        
        self.model = model
        self.train_data_loader = train_data_loader 
        self.valid_data_loader = valid_data_loader 
        self.optimizer = optimizer
        self.device = accelerator.device
        self.accelerator = accelerator
        self.loss_fn = config.LOSS
    
      
        
    def train_eval_fn(self, epoch):
        self.model.train()
        losses = AverageMeter()
        
        para_loader = self.train_data_loader
        tk0 = tqdm(para_loader, total=len(para_loader), disable=not self.accelerator.is_local_main_process)
        
        for bi, d in enumerate(tk0):
            
            self.model.zero_grad()
            output = self.model(input_ids=d['input_ids'], attention_mask=d['attention_mask'])
            output = torch.permute(output , (0,2,1))
            target = d['label']
#             print(target.size() , output.size())
            loss = self.loss_fn(output, target)
    
            self.accelerator.backward(loss)
            self.optimizer.step()

            losses.update(loss.item(), target.size(0))
            tk0.set_postfix(loss=losses.avg)
     
        valid_loss, _ = self.eval_fn()
        print(f'Epoch : {epoch + 1} | Validation Score : {valid_loss}')
        self.accelerator.wait_for_everyone()
            
    def eval_fn(self):
        self.model.eval()
        losses = AverageMeter()
        para_loader = self.valid_data_loader
        
        tk0 = tqdm(para_loader, total=len(para_loader), leave=True, disable=not self.accelerator.is_local_main_process)
        yt, yp = [], []

        with torch.no_grad():
            for bi, d in enumerate(tk0):
                ids = d['input_ids']
                mask = d['attention_mask']
                target = d['label']
                output = self.model(input_ids=ids, attention_mask=mask)
                output = torch.permute(output , (0,2,1))
                
                loss = self.loss_fn(output, target)
                output = torch.sigmoid(output)
                yt = yt + (target.cpu().numpy()*4 + 1).tolist()
                yp = yp + (output.detach().cpu().numpy()*4 + 1).tolist()
                losses.update(loss.item(), ids.size(0))
               
                tk0.set_postfix(loss=losses.avg)

        return losses.avg, yp

def run(df_train, df_val):  
    labels = ['B-LOC', 'B-MISC', 'B-ORG', 'B-PER', 'I-LOC', 'I-MISC', 'I-ORG', 'I-PER', 'O']
    tag2id = {j:i for i, j in enumerate(labels)}
    id2tag = {i:j for i, j in enumerate(labels)}
 
    tokenizer = transformers.AutoTokenizer.from_pretrained(config.MODEL)
    tokenizer.save_pretrained(config.SAVE_DIR)

    train_dataset = Dataset(texts = df_train.tokens.values.tolist(),labels = df_train.label.values.tolist(),tokenizer=tokenizer,tag2id=tag2id)

    valid_dataset = Dataset(texts = df_val.tokens.values.tolist(),labels = df_val.label.values.tolist(),tokenizer=tokenizer,tag2id=tag2id)

    train_data_loader = torch.utils.data.DataLoader(train_dataset,batch_size= config.TRAIN_BATCH_SIZE,)

    valid_data_loader = torch.utils.data.DataLoader(valid_dataset,batch_size=config.VALID_BATCH_SIZE,num_workers=2)

    model_config = transformers.AutoConfig.from_pretrained(config.MODEL)
    model_config.update({'num_labels': config.NUM_CLASS,"output_hidden_states":True})

    model = CustomModel(config.MODEL, model_config, len(tokenizer))

    optimizer = AdamW(model.parameters(), lr=config.ENC_LR)

    accelerator = Accelerator()
    
    trainer_obj = Trainer(model, train_data_loader, valid_data_loader, optimizer, accelerator)

    accelerator.print('Starting training....')

    for epoch in range(config.EPOCHS):
        trainer_obj.train_eval_fn(epoch)

def infer(text,model , tokenizer , tag2id ,id2tag):
    data = preprocess_tag_single(text ,192,tokenizer , tag2id )
    ids = data['input_ids']
    mask = data['attention_mask']
    word_id = data['word_ids']
    out = model(ids , mask)
    out = torch.softmax(out,dim=2)
    y_pred = torch.argmax(out , dim=2)
    
    yp = []
    pre = None
    for i , ids in enumerate(word_id):
        if(ids is None or ids == pre):
            continue
        yp.append(id2tag[y_pred[0][i].item()])
        pre = ids
    return yp

def infer_val(df):
    texts = df.text.values.tolist()
    tags = df.tag.values.tolist()
    labels = ['B-LOC', 'B-MISC', 'B-ORG', 'B-PER', 'I-LOC', 'I-MISC', 'I-ORG', 'I-PER', 'O']
    tag2id = {j:i for i, j in enumerate(labels)}
    id2tag = {i:j for i, j in enumerate(labels)}
    model_config = transformers.AutoConfig.from_pretrained(config.MODEL)
    model_config.update({'num_labels': config.NUM_CLASS,"output_hidden_states":True})
    tokenizer = transformers.AutoTokenizer.from_pretrained(config.MODEL)
    model = CustomModel(config.MODEL, model_config, len(tokenizer))
    model.load_state_dict(torch.load(os.path.join(config.SAVE_DIR , "model.bin")))
    y_pred = []
    y_true = [tag.split(" ") for tag in tags]
    for line in tqdm(texts):
        y_pred.append(infer(line,model,tokenizer , tag2id,id2tag))
    print(classification_report(y_true,y_pred))
    
    
def run_model(df ):
    seed_all(config.SEED)

    df['mlabel'] = (df.ner_tags)
    val_size  = int(df.shape[0]*.1)
    df_train = df
    df_val = df.iloc[:-val_size]

    print('Training examples ', len(df_train))
    print('Validation examples ', len(df_val))
    
    print("\ntraining started....")
    run(df_train, df_val)
    print("training completed....")
    
    infer_val(df_val)



dataset = load_dataset("conll2003")
df= dataset['train'].to_pandas()

run_model(df)