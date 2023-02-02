import math
from dataclasses import dataclass
import torch
import torch.nn as nn
from torch.nn import functional as F

def gelu(x):
  return .5*x*(1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + .044715 * torch.pow(x , 3.))))

class LayerNorm(nn.Module):
  def __init__(self ,config , eps = 1e-6):
    super(LayerNorm , self).__init__()
    self.a2 = nn.Parameter(torch.ones(config.d_model))
    self.b2 = nn.Parameter(torch.ones(config.d_model))
    self.e = eps

  def forward(self,x):
    mean = x.mean( -1 , keepdim = True)
    std = x.std(-1 , keepdim = True)

    return self.a2 * (x -mean) / (std + self.e) + self.b2

class CausalSelfAttention(nn.Module):
  def __init__(self , config):
    super(CausalSelfAttention, self).__init__()
    assert config.d_model % config.n_head == 0
    self.config = config
 
    self.head_dim = self.config.d_model//self.config.n_head 
    self.key = nn.Linear(config.d_model , config.d_model)
    self.query = nn.Linear(config.d_model , config.d_model)
    self.value = nn.Linear(config.d_model , config.d_model)
    self.proj = nn.Linear(config.d_model , config.d_model)
    self.attn_drop = nn.Dropout(config.dropout_attn)
    self.proj_drop = nn.Dropout(config.dropout_attn)
    

  def forward(self ,x):
    B,T,C = x.shape
    k = self.key(x).view(B ,T , self.config.n_head , self.head_dim).transpose(2,1)
    q = self.query(x).view(B ,T , self.config.n_head , self.head_dim).transpose(2,1)
    v = self.value(x).view(B ,T , self.config.n_head , self.head_dim).transpose(2,1)
    attn = (q@k.transpose(-2,-1)) * ( 1.0 / math.sqrt(k.size(-1)))

    tril = torch.tril(torch.ones((1,1,T,T))).to(self.config.device)
    attn = attn.masked_fill(tril == 0.0, float("-inf"))
    
    attn = F.softmax(attn , dim=-1)
    attn = self.attn_drop(attn)
    out  = attn @v
    out = out.transpose(-2,-1).contiguous().view(B,T,C)
    out = self.proj(out)
    out = self.proj_drop(out)

    return out

class PositionwiseFeedforward(nn.Module):
  def __init__(self ,config):
    super(PositionwiseFeedforward , self).__init__()

    self.l1 = nn.Linear(config.d_model , config.d_inter)
    self.proj = nn.Linear(config.d_inter , config.d_model) 

    self.dropout = nn.Dropout( p = config.dropout_ffn)

  def forward(self , x):
    #x = (batch ,seq_len , d_model)
    out = gelu(self.l1(x))
    #out = (batch ,seq_len , d_inter)
    out = self.proj(out)
    #out = (batch ,seq_len , d_model)
    out = self.dropout(out)
    return out

class Block(nn.Module):
  def __init__(self , Config):
    super(Block , self).__init__()
    self.layernorm1 = LayerNorm(Config)
    self.attn = CausalSelfAttention(Config)
    self.layernorm2 = LayerNorm(Config)
    self.ffn = PositionwiseFeedforward(Config)

  def forward(self , x):
    x = x + self.attn(self.layernorm1(x))
    x = x + self.ffn(self.layernorm2(x))
    return x

class Transformer(nn.Module):
  def __init__(self , config):
    super(Transformer , self).__init__()
    self.config = config
    self.token_emb = nn.Embedding(config.vocab_size , config.d_model)
    self.pos_emb = nn.Embedding(config.max_len , config.d_model)
    self.dropout = nn.Dropout( p =.1)
    self.layernorm = LayerNorm(Config)
    self.blocks = nn.ModuleList([Block(config) for _ in range(config.n_layer)])

  def forward(self,x):
    B,T = x.shape
    pos = torch.arange(0,T , dtype =torch.long).to(self.config.device).unsqueeze(0)

    tok = self.token_emb(x)
    pos_e = self.pos_emb(pos)
    x = self.dropout(tok + pos_e)

    for i in range(self.config.n_layer):
      x = self.blocks[i](x)
    x = self.layernorm(x)

    return x

class GPT(nn.Module):
  def __init__(self , config):
    super(GPT , self).__init__()
    self.config = config
    self.transformer = Transformer(config)
    self.lm_head = nn.Linear(config.d_model , config.vocab_size)
    
    #weight tying
    #self.transformer.token_emb.weight = self.lm_head.weight
    self.apply(self.init_weights)

    for np , p in self.named_parameters():
      if np.endswith('proj.weight'):
        torch.nn.init.normal_(p , mean = 0.0 , std = .02 / math.sqrt(2 * config.n_layer))

    n_param = sum(p.numel() for p in self.parameters())
    print(f"Total number of parameter : {n_param}")
        
  def init_weights(self,module):
    if isinstance(module , nn.Linear):
      torch.nn.init.normal_(module.weight , mean = 0.0 , std = .02)
      if module.bias is not None:
        torch.nn.init.zeros_(module.bias)

    elif isinstance(module , nn.Embedding):
      torch.nn.init.normal_(module.weight , mean = 0.0 , std = .02)
    
  def forward(self , x , target = None):
    x = self.transformer(x)

    if target is not None:
      logits = self.lm_head(x)
      loss = self.config.loss_fn(logits.view(-1 ,logits.size(-1) ) , target.view(-1))
    else:
      logits = self.lm_head(x)
      loss = None

    return logits  , loss
