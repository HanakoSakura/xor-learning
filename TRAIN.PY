import torch
import torch.nn as nn

import EvilCircle
import EvilSet
import trainEvil

import sys

def mian():
    if len(sys.argv) != 2:
        print('Error when read value')
        return
    hate =  EvilCircle.Circle()
    train_loader = EvilSet.LoadEvilSet('XOR-TEST.json',sys.argv[1],4)
    optimizer = torch.optim.SGD(hate.parameters(),lr=0.02,momentum=0.9)
    criterion = nn.BCELoss()
    print('Training '+sys.argv[1]+' model')
    trainEvil.train(20000,hate,optimizer,criterion,train_loader,1000)
    trainEvil.validation(hate,criterion,train_loader)
    trainEvil.Save(hate,'evil/'+sys.argv[1]+'.evil')
    return

if __name__ == '__main__':
    mian()
