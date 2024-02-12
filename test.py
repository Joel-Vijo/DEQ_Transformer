# import random 

# import torch 
# import torch.nn as nn 

# from dataloader_utils import MyCollate
# from torch.utils.data import DataLoader
# from transformers import AutoTokenizer
# from transformers import BertModel
# from datasets import load_dataset
# from decoder import decoder
# from model import Model
# tokenizer = AutoTokenizer.from_pretrained("bert-base-multilingual-cased")

# #setting seed 
# MANUAL_SEED = 3407
# random.seed(MANUAL_SEED)
# torch.manual_seed(MANUAL_SEED)
# torch.backends.cudnn.deterministic = True


# # ARGS: 

# ## TRAIN-ARGS: 
# ### batch_size = 32
# ### epochs = 10
# ### tol = 1e-3

# ## DATASET SPECS:  
# ### dataset_name = "bentrevett/multi30k"

# ## MODEL ARGS: 

 
# #initializing dataset
# dataset = load_dataset("bentrevett/multi30k")

# #initializing tokenizer
# tokenizer = AutoTokenizer.from_pretrained("bert-base-multilingual-cased")
# print(tokenizer.vocab_size)

# #initializing dataloader 
# ##train_loader
# train_loader = DataLoader(dataset=dataset['train'], batch_size=16, collate_fn=MyCollate(tokenizer)) 
# val_loader = DataLoader(dataset=dataset['validation'], batch_size=16, collate_fn=MyCollate(tokenizer)) 

# #defining encoder 
# enc = BertModel.from_pretrained("bert-base-multilingual-cased")

# #defining decoder
# dec_layer = nn.TransformerDecoderLayer(768,nhead=4)
# dec = decoder(dec_layer=dec_layer,num_layers=3)

# # model = Model(enc,dec)
# pad_idx=0
# print(len(train_loader))
# for batch in enumerate(train_loader):
#     # print(batch)
#     src = batch[1][0]
#     tgt = batch[1][1]
#     print(src)
#     #ensuring src is of the right size: 
#     if len(src.shape)>2: 
#         src = src.squeeze(-1)
    

#     #defining att_mask
#     att_mask = torch.ones(src.shape).masked_fill(src == pad_idx,0)
#     att_mask = att_mask.squeeze(-1) #att_mask.shape = (batch_size, src_len)

#     att_mask2 = torch.ones(tgt.shape).masked_fill(tgt == pad_idx,0)
#     att_mask2 = att_mask2.squeeze(-1)

#     print(f'src.shape = {src.shape}, att_mask.shape = {att_mask.shape}')
    
#     #getting src embeddings 
#     mem = enc(src, attention_mask = att_mask)['last_hidden_state'] 
#     # print(mem)
#     print(f'mem.shape= {mem.shape}')

#     #getting tgt embeddings 
#     tgt = enc(tgt, attention_mask = att_mask2)['last_hidden_state'] 
#     print("TGT",tgt.shape)
#     # print(tgt)
#     tgt_mask = torch.nn.Transformer.generate_square_subsequent_mask(tgt.shape[1])

#     pred = dec(tgt=tgt, memory=mem, memory_mask=att_mask, tgt_mask=tgt_mask,
#                tgt_is_causal=True, memory_is_causal=True)
    
#     print(f'pred.shape= {pred.shape}')

#     break


import jax.numpy as jnp
def anderson_solver(f, z_init, m=5, lam=1e-4, max_iter=50, tol=1e-5, beta=1.0):
  x0 = z_init
  x1 = f(x0)
  x2 = f(x1)
  X = jnp.concatenate([jnp.stack([x0, x1]), jnp.zeros((m - 2, *jnp.shape(x0)))])
  F = jnp.concatenate([jnp.stack([x1, x2]), jnp.zeros((m - 2, *jnp.shape(x0)))])

  res = []
  for k in range(2, max_iter):
    n = min(k, m)
    G = F[:n] - X[:n]
    GTG = jnp.tensordot(G, G, [list(range(1, G.ndim))] * 2)
    print(G.shape)
    print(GTG.shape)
    H = jnp.block([[jnp.zeros((1, 1)), jnp.ones((1, n))],
                   [ jnp.ones((n, 1)), GTG]]) + lam * jnp.eye(n + 1)
    print(H.shape)
    alpha = jnp.linalg.solve(H, jnp.zeros(n+1).at[0].set(1))[1:]
    print(alpha.shape)
    print(F[:n].shape)
    xk = beta * jnp.dot(alpha, F[:n]) + (1-beta) * jnp.dot(alpha, X[:n])
    X = X.at[k % m].set(xk)
    F = F.at[k % m].set(f(xk))

    res = jnp.linalg.norm(F[k % m] - X[k % m]) / (1e-5 + jnp.linalg.norm(F[k % m]))
    if res < tol:
      break
  return xk

def fixed_point_layer(solver, f, params, x):
  z_star = solver(lambda z: f(params, x, z), z_init=jnp.zeros_like(x))
  return z_star

f = lambda W, x, z: jnp.tanh(jnp.dot(W, z) + x)

from jax import random

ndim = 10
W = random.normal(random.PRNGKey(0), (ndim, ndim)) / jnp.sqrt(ndim)
x = random.normal(random.PRNGKey(1), (ndim,))
# print(x.shape)
z_star = fixed_point_layer(anderson_solver, f, W, x)
print(z_star)