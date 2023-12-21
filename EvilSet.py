import torch
from torch.utils.data import DataLoader
import json

# 数据集类
class EvilSet():
    def __init__(self,data,label=None) -> None:
        self.data = data
        self.label = label
    def __getitem__(self,index):
        if self.label is None:
            return torch.tensor(self.data[index],dtype=torch.float32)
        else:
            return torch.tensor(self.data[index],dtype=torch.float32),\
                torch.tensor(self.label[index],dtype=torch.float32)
    def __len__(self):
        return len(self.data)

def LoadEvilSet(path:str,set_choice:str,batch_size:int):
    with open(path) as f:
        DS = json.load(f)
    USE = DS[set_choice]
    uses = EvilSet(USE['input'],USE['output'])
    return DataLoader(uses,batch_size,shuffle=True)

