import torch 
from torch.utils.data import Dataset, DataLoader
from tqdm.auto import tqdm



class MyDataset(Dataset):
    def __init__(self, max_len):
        self.data = ["apple","grava","banana","orange","mango","dog","cat","pig","rabbit","elephant"]
        self.max_len = max_len
        #print(max_len - len(self.data)%max_len)
        self.data += ["<pad>"]*(max_len - (len(self.data)% self.max_len))
    def __len__(self):
        return len(self.data)
    def __getitem__(self, i):
        return self.data[i]

s = MyDataset(max_len=3)   #['apple', 'grava', 'banana', 'orange', 'mango', 'dog', 'cat', 'pig', 'rabbit', 'elephant', '<pad>', '<pad>']


vocab = {'apple':0,'grava':1,'banana':2,'orange':3,'mango':4,'dog':5,'cat':6,'pig':7,'rabbit':8,'elephant':9,'<pad>':10}
def collate_fn(examples): #each batch
    examples = [vocab[ex] for ex in examples]
    examples = torch.tensor(examples,dtype=torch.long)
    return examples




myDataLoader = DataLoader(dataset=s,batch_size=3, shuffle=False, collate_fn=collate_fn)
for batch in tqdm(myDataLoader,desc="testing~~~"):
    print(batch)