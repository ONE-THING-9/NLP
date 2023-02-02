import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math, copy,time

from torch.autograd import Variable
import matplotlib.pyplot as plt
import seaborn
from tqdm.auto import tqdm

def clones(module,n):
  return nn.ModuleList([copy.deepcopy(module) for _ in range(n)])

def subsequent_mask(size):
  attn_shape = (size,size)
  sub_mask = np.triu(np.ones(attn_shape) , k=1).astype(int)
  return torch.from_numpy(sub_mask) == 0

def attention(query , key , value ,mask = None, dropout = None):
  #query = key = value = (batch , num_head , seq_len , d_model/num_head )   ---> (2, 4, 8, 16)

  dk = query.size()[-1]
  #dk = 8
  scores = torch.matmul(query , key.transpose(-2,-1)) / math.sqrt(dk)
  #  matmul( (2,4,8,16) , (2,4,16,8))   -> (2,4,8,8)

  if mask is not None:
    scores = scores.masked_fill(mask ==0  , -1e-9)
    # mask torch.Size([2, 1, 1, 8]) in case of encoder and decoder cross attention 
    #scores = (2,4,8,8).masked_fill( (2,1,1,8) ) = this is done by broadcasting
    # mask torch.Size([2, 1, 8, 8]) in case of decoder self attetnion 
    #scores = (2,4,8,8).masked_fill( (2,1,8,8) ) = this is done by broadcasting

  p_attn = F.softmax(scores , dim=-1)
  # p_attn -> (2,4,8,8)

  if dropout is not None:
    p_attn = dropout(p_attn)

  result =  torch.matmul(p_attn , value) , p_attn
  #matmul((2,4,8,8) , (2,4,8,16) )   -> (2,4,8,16)
  return result

class MultiheadAttention(nn.Module):
  def __init__(self,h,d_model ,dropout = .1):
    super(MultiheadAttention , self).__init__()

    self.dk = int(d_model/h)
    self.h = h
    self.d_model = d_model
    self.linear = clones(nn.Linear(d_model , d_model),4)
    self.attn = None
    self.dropout = nn.Dropout(p = dropout)

  def forward(self , query ,key,value , mask = None):

      #query = key = value = (batch , seq_len , d_model)   --- let's take   (2,8 ,64)
      # h = 4 , d_model = 64
      #mask  = (2,1,8) in case of encoder

      if mask is not  None:
        mask = mask.unsqueeze(1)
        #mask = (2,1,1,8)  in case of encoder and decoder cross attention
        #mask = (2,1,8,8) in case of decoder self attention
      nbatch = key.size(0) 
       # nbatch -> 2

      #1. pass through the linear layer and split into h heads
      query , key , value = [l(x).view(nbatch , -1 , self.h , self.dk).transpose(1,2)  for l,x in zip(self.linear , (query , key ,value))]
      # l(x) -> (2 , 8 ,64)
      # l(x).view(nbatch , -1 , self.h , self.dk)    -> (2 , 8 , 4 ,16)
      #l(x).view(nbatch , -1 , self.h , self.dk).transpose(1,2)  -> (2 , 4, 8, 16)

      #2. calculate attetnion 
      x , self.attn = attention(query , key,value , mask =mask, dropout = self.dropout)
      #x = (2,4,8,16) 

      #3. concat the head and pass through linear layer
      x = x.transpose(1,2).contiguous().view(nbatch , -1 , self.d_model)
      #x.transpose(1,2) -> (2,8,4,16)
      #x.transpose(1,2).contiguous().view(nbatch , -1 , self.d_model)  -> (2,8,64)

      return self.linear[-1](x)

class PositionwiseFeedforward(nn.Module):
  def __init__(self , d_model , d_inter , dropout = .1):
    super(PositionwiseFeedforward , self).__init__()
    self.l1 = nn.Linear(d_model , d_inter)
    self.l2 = nn.Linear(d_inter , d_model) 
    self.dropout = nn.Dropout( p = dropout)

  def forward(self , x):
    #x = (batch ,seq_len , d_model)
    out = F.relu(self.l1(x))
    #out = (batch ,seq_len , d_inter)

    out = self.dropout(out)
    out = self.l2(out)
    #out = (batch ,seq_len , d_model)
    return out

class Embedding(nn.Module):
  def __init__(self , d_model ,vocab):
    super(Embedding , self).__init__()
    self.emb = nn.Embedding(vocab , d_model)
    self.d_model = d_model

  def forward(self,x):
    out = self.emb(x) / math.sqrt(self.d_model)
    return out

class PositionalEmbedding(nn.Module):
  def __init__(self , d_model , dropout=.1 , max_len = 5000):
    super(PositionalEmbedding , self).__init__()
    self.dropout = nn.Dropout(p = dropout)
    pe = torch.zeros(max_len , d_model)
    #pe  = (max_len , d_model)
    position = torch.arange(0 ,max_len).unsqueeze(1)
    #position = (max_len , 1)
    div_term = torch.exp(torch.arange(0 ,d_model ,2) * -(math.log(10000.0) / d_model) )
    #div_term = d_model/2

    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    #pe = (max_len , d_model)
    pe = pe.unsqueeze(0)
    #pe = (1 , max_len , d_model)
    self.register_buffer('pe' , pe)

  def forward(self , x):
    x = x + Variable(self.pe[: , :x.size(1)] , requires_grad = False)
    #x = (batch , seq_len , d_model)
    return self.dropout(x)

class LayerNorm(nn.Module):
  def __init__(self , feature , eps = 1e-6):
    super(LayerNorm , self).__init__()
    self.a2 = nn.Parameter(torch.ones(feature))
    self.b2 = nn.Parameter(torch.ones(feature))
    self.e = eps

  def forward(self,x):
    mean = x.mean( -1 , keepdim = True)
    std = x.std(-1 , keepdim = True)
    return self.a2 * (x -mean) / (std + self.e) + self.b2

class SubLayerConn(nn.Module):
  #residual connection followed by layer norm
  def __init__(self , size , dropout):
    super(SubLayerConn , self).__init__()
    self.norm = LayerNorm(size)
    self.dropout = nn.Dropout(p = dropout)

  def forward(self , x , sublayer):
    return x + self.dropout(sublayer(self.norm(x)))
    # x +    dropout (  self_attn (LayerNorm (x))) 
    # x +    dropout (  feedforward (LayerNorm (x))) 

class EncoderLayer(nn.Module):
  def __init__(self , size , self_attn , feed_forward , dropout):
    super(EncoderLayer , self).__init__()
    self.self_attn = self_attn
    self.feed_forward = feed_forward
    self.sublayer = clones(SubLayerConn(size , dropout) , 2)
    self.size = size

  def forward(self , x , mask):
    x = self.sublayer[0](x , lambda x : self.self_attn(x ,x,x,mask))
    return self.sublayer[1](x , self.feed_forward)
# x -> layernorm -> self_attn -> dropout -> residual connection -> Layernorm -> positional_feedforward -> dropout -> residual_connection

class Encoder(nn.Module):
  def __init__(self , layer ,N):
    super(Encoder , self).__init__()
    self.layers = clones(layer , N)
    self.norm = LayerNorm(layer.size)

  def forward(self ,x, mask):
    for layer in self.layers:
      x = layer(x ,mask)
    return self.norm(x)
# x ->  (encoderLayer)* N ->  LayerNorm

class DecoderLayer(nn.Module):
  def __init__(self , size , self_attn , src_attn , feed_forward , dropout):
    super(DecoderLayer , self).__init__()
    self.size=  size
    self.self_attn = self_attn
    self.src_attn = src_attn
    self.feed_forward = feed_forward
    self.sublayer = clones(SubLayerConn(size , dropout) ,3 )

  def forward(self , x , memory , src_mask , target_mask):
    m = memory
    x = self.sublayer[0](x , lambda x : self.self_attn(x,x,x,target_mask))
    #self attn on target tokens     
    x = self.sublayer[1](x , lambda x : self.src_attn(x , m,m,src_mask))
    #cross attention of decoder embedding and encoder output 
    return self.sublayer[2](x , self.feed_forward)

class Decoder(nn.Module):
  def __init__(self , layer  , N):
    super(Decoder , self).__init__()
    self.layers = clones(layer , N)
    self.norm = LayerNorm(layer.size)
  def forward(self , x , memory , src_mask , tgt_mask):
    for layer in self.layers:
       x = layer(x , memory , src_mask , tgt_mask)
    return self.norm(x)

class Generator(nn.Module):
  def __init__(self, d_model , vocab):
    super(Generator , self).__init__()
    self.proj = nn.Linear(d_model , vocab)

  def forward(self , x , softmax = None):
    if(softmax is not None):
      return F.log_softmax(self.proj(x) , dim = -1)

    else:
      return self.proj(x)


class EncoderDecoder(nn.Module):
  def __init__(self,encoder , decoder , src_emb ,tgt_emb , generator):
    super(EncoderDecoder , self).__init__()

    self.encoder = encoder
    self.decoder = decoder
    self.src_emb = src_emb
    self.tgt_emb = tgt_emb
    self.generator = generator

  def forward(self , src , tgt , src_mask , tgt_mask):


    out = self.encode(src , src_mask)
    out = self.decode(out , src_mask , tgt , tgt_mask)
    out = self.generator(out)
    return out

  def encode(self , src , src_mask):
    return self.encoder(self.src_emb(src) , src_mask)

  def decode(self , memory , src_mask ,tgt ,tgt_mask):
    return self.decoder(self.tgt_emb(tgt) , memory , src_mask , tgt_mask)

class Transformer(nn.Module):
    def __init__(self , src_vocab , tgt_vocab , d_model , d_ff ,head,encoder_layer,decoder_layer, dropout =.1):
        super(Transformer , self).__init__()
        c = copy.deepcopy

        attn = MultiheadAttention(head , d_model)
        ff = PositionwiseFeedforward(d_model , d_ff)
        position = PositionalEmbedding(d_model)

        encoder = Encoder(EncoderLayer(d_model , c(attn) , c(ff) , dropout=dropout) , encoder_layer)
        decoder = Decoder(DecoderLayer(d_model , c(attn) ,c(attn) , c(ff) , dropout = dropout ) , decoder_layer)

        enc_emb = nn.Sequential(Embedding(d_model , src_vocab) , c(position))
        dec_emb = nn.Sequential(Embedding(d_model , tgt_vocab) , c(position))
        generator = Generator(d_model , tgt_vocab)
        
        self.model = EncoderDecoder(encoder , decoder , enc_emb ,dec_emb , generator)

        for p in self.model.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    def forward(self , src , tgt , src_mask , tgt_mask):
        return self.model(src , tgt , src_mask , tgt_mask)


