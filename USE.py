import torch
from EvilCircle import Circle

hate_name = input('Choose a model:')

model = torch.load('evil/'+hate_name+'.evil')

while True:
    IN = input(hate_name+' ').split()
    tmp = []
    for i in IN:
        tmp.append(float(i))
    x = torch.tensor(tmp,dtype=torch.float32)
    y = model(x)
    tmp = y.tolist()
    print('{}  ({:.3f})'.format(round(tmp[0]),tmp[0]))
