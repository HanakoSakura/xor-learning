import torch
import torch.nn as nn
import torch.nn.functional as F

# 模型类
class Circle(nn.Module):
    def __init__(self):
        super(Circle,self).__init__()
        self.l1 = nn.Linear(2,4,dtype=torch.float32)
        self.l2 = nn.Linear(4,4,dtype=torch.float32)
        self.l3 = nn.Linear(4,1,dtype=torch.float32)

    # 正向传播函数
    def forward(self,x):
        y = F.sigmoid(self.l1(x))
        y = F.sigmoid(self.l2(y))
        y = F.sigmoid(self.l3(y))
        return y
    
