# import datasets 
# from datasets import load_dataset


'''
train split: 29000 example 
val split: 1014
test: 1000
'''

#initializing dataset 
# dataset = load_dataset("bentrevett/multi30k")

# def tokenization(example):
#     return tokenizer(example["en"],padding=True)

# def tokenization2(example):
#     return tokenizer(example["de"])

# # train = train.map(tokenization, batched=True)
# eng = dataset.map(tokenization, batched=True)
# de = dataset.map(tokenization2,batched=True)

class MyCollate(): 

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.pad_idx = self.tokenizer.pad_token_id

    def __call__(self, batch): 
        
        eng = [item['en'].lower() for item in batch]
        de = [item['de'].lower() for item in batch]

        eng_batch =  self.tokenizer(eng , 
                       max_length=100, padding = 'max_length',return_tensors='pt', truncation=True)['input_ids'].T 

        de_batch =  self.tokenizer(de ,
                       max_length=100, padding = 'max_length',return_tensors='pt', truncation=True)['input_ids'].T 
        
        return eng_batch, de_batch
                    
        
