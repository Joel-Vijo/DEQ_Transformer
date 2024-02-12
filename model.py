import torch.nn as nn 
import torch

class Model(nn.Module):
    
    def __init__(self,encoder,decoder):
        super(Model,self).__init__()
        self.enc = encoder
        self.dec = decoder
        # TODO: dimension values
        self.fc = nn.Linear(768, 119547)
        # self.d2e = nn.Linear(768, 1)

    def forward(self,src, tgt, pad_idx=0):
        # print("src",src.type)
        if len(src.shape)>2: 
            src = src.squeeze(-1)
        # print("src",src.shape)

        mask = torch.ones(src.shape).masked_fill(src == pad_idx,0)
        # print("mask1",mask.shape)
        # mask = mask.squeeze(-1)
        # print("src",src.shape)
        # print("mask",mask.shape)
        mask2 = torch.ones(tgt.shape).masked_fill(tgt == pad_idx,0)
        # mask2 = mask2.squeeze(-1)

        tol = 0.19
        out = None
        iterations = 0
        # if out == None:
        
        # print("Mask",mask.shape)
        enc_output = self.enc(src, attention_mask=mask)
        enc_output = enc_output['last_hidden_state']
        tgt = self.enc(tgt, attention_mask = mask2)['last_hidden_state'] 
        tgt_mask = torch.nn.Transformer.generate_square_subsequent_mask(tgt.shape[1])
        out = self.dec(tgt=tgt, memory=enc_output, memory_mask=mask,tgt_mask = tgt_mask,tgt_is_causal=True, memory_is_causal=True)
        out_fin=self.fc(out)
        iterations += 1
        # out=self.d2e(out)
        # print(src)
        # out=out.squeeze(-1)
        # print(out.shape)
        trg_new=[]
        for sentence in out_fin:
            y=[]
            for word in sentence:
                y.append(torch.argmax(word))
            trg_new.append(y)
        # print(len(trg_new[0][0]))
        trg_new = torch.tensor(trg_new)
        # # print(trg_new)
        # enc_output = self.enc(trg_new, attention_mask=mask2)
        # enc_output = enc_output['last_hidden_state']
        # # tgt = self.enc(tgt, attention_mask = mask2)['last_hidden_state'] 
        # tgt_mask = torch.nn.Transformer.generate_square_subsequent_mask(tgt.shape[1])
        # out_new = self.dec(tgt=out, memory=enc_output, memory_mask=mask,tgt_mask = tgt_mask,tgt_is_causal=True, memory_is_causal=True)
        # out_fin=self.fc(out_new)
        # # print(sum(sum(sum(abs(out_new - out))))/(out_new.shape[0]*out_new.shape[1]*out_new.shape[2]))
        #     # out_new = self.dec(tgt, enc_output, tgt_mask = mask)
        # while sum(sum(sum(abs(out_new - out)))).item()/(out_new.shape[0]*out_new.shape[1]*out_new.shape[2]) > tol:
        #     if(iterations>13):
        #         break
        #     trg_new=[]
        #     for sentence in out_fin:
        #         y=[]
        #         for word in sentence:
        #             y.append(torch.argmax(word))
        #         trg_new.append(y)
        #     # print(len(trg_new[0][0]))
        #     trg_new = torch.tensor(trg_new)
        #     enc_output = self.enc(trg_new, attention_mask=mask2)
        #     enc_output = enc_output['last_hidden_state']
        #     # enc_output = self.enc(out, attention_mask=mask2)
        #     # enc_output = enc_output['last_hidden_state']
        #     out=out_new
        #     out_new = self.dec(tgt=out, memory=enc_output, memory_mask=mask,tgt_mask = tgt_mask,tgt_is_causal=True, memory_is_causal=True)
        #     out_fin=self.fc(out_new)
        #     # out=out_new
        #     iterations += 1
        #     print(iterations)
            # print(sum(sum(sum(abs(out_new - out))))/(out_new.shape[0]*out_new.shape[1]*out_new.shape[2]))
            # print("Iterations Needed:", iterations)
            

        return trg_new,out_fin
    
    # def forward(self,src, memory, pad_idx=0):
    #     if len(src.shape)>2: 
    #         src = src.squeeze(-1)
    #     mask = torch.ones(src.shape).masked_fill(src == pad_idx,0)
    #     mask = mask.squeeze(-1).unsqueeze(1)
    
    #     rep = self.enc(src, attention_mask=mask)
    #     rep = rep['last_hidden_state']
    #     print(rep.shape)

    #     out = self.dec(rep,memory)

    #     return out