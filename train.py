import random 
import argparse

import torch 
import torch.nn as nn 
import wandb

from dataloader_utils import MyCollate
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from transformers import BertModel
from datasets import load_dataset
from decoder import decoder
from model import Model
from solvers import anderson_solver
tokenizer = AutoTokenizer.from_pretrained("bert-base-multilingual-cased")
wandb.login()
wandb.init(project="transformerDEQ")
#setting seed 
MANUAL_SEED = 3407
random.seed(MANUAL_SEED)
torch.manual_seed(MANUAL_SEED)
torch.backends.cudnn.deterministic = True


# ARGS: 
parser = argparse.ArgumentParser(description="Training script for seq2seq model")

## TRAIN-ARGS: 
### primary train-args
parser.add_argument('--batch_size', type=int, default=16)
parser.add_argument('--epochs', type=int, default=10)

### secondary train-args
parser.add_argument('--device', type=str, default='cuda')
parser.add_argument('--tol', type=float, default=1e-3)

## DATASET SPECS:  
parser.add_argument('--dataset_name', type=str, default='bentrevett/multi30k')

## MODEL ARGS: 

args = parser.parse_args()
 
#initializing dataset
dataset = load_dataset(args.dataset_name)

#initializing tokenizer
tokenizer = AutoTokenizer.from_pretrained("bert-base-multilingual-cased")

#initializing dataloader 
##train_loader
train_loader = DataLoader(dataset=dataset['train'], batch_size=args.batch_size, collate_fn=MyCollate(tokenizer))
val_loader = DataLoader(dataset=dataset['validation'], batch_size=args.batch_size, collate_fn=MyCollate(tokenizer)) 

# for batch in enumerate(train_loader):

#         src=batch[1][0] #src.size = (batch_size,seq_len)
#         trg=batch[1][1]
#         print(src.shape)
#         print(trg.shape)

#defining encoder 
enc = BertModel.from_pretrained("bert-base-multilingual-cased")
# TODO: add parameters to this and the final FC layer
dec_layer = nn.TransformerDecoderLayer(768,nhead=4)
dec = decoder(dec_layer=dec_layer,num_layers=3)
model = Model(enc,dec).to(args.device)

# TODO: define loss_function 
criterion = nn.CrossEntropyLoss(ignore_index = 0)

#defining optimizer 
optimizer = torch.optim.Adam(model.parameters())
device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)
for epoch in range(args.epochs): 
    epoch_loss = 0 

    for batch in enumerate(train_loader):

        src=batch[1][0].to(device) #src.size = (batch_size,seq_len)
        trg=batch[1][1].to(device) #trg.size = (batch_size, seq_len)
        # print("SRC",src.shape)
        optimizer.zero_grad()
        output = anderson_solver(model,src,trg,device)
        # output = torch.from_numpy(output).to(device)
        # print("Output",output.shape)
        # output=output.permute(0,2,1)
        # print("Trg",trg.shape)
        output_size=output.shape[-1]
        output=output.contiguous().view(-1,output_size)
        trg=trg.contiguous().view(-1)
        # print("Output",output.shape)
        # output=output.permute(0,2,1)
        # print("Trg",trg.shape)
        loss = criterion(output, trg)
        # print(f"Epoch: {epoch} | Loss: {epoch_loss/len(train_loader)}")
            
        epoch_loss += loss.item()
        loss.backward()
        optimizer.step()
        # print("Iterations Needed:", iterations)
    
    print(f"Epoch: {epoch} | Loss: {epoch_loss/len(train_loader)}")
            
            
        
      