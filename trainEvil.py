import torch

import EvilCircle as EC
import EvilSet as ES

def train(times,evil:EC.Circle,optimizer,criterion,train_loader,show_time):
    # 调整模式
    evil.train()
    for t in range(times+1):
        for batch_index,(data,target) in enumerate(train_loader):
            y = evil(data)
            loss = criterion(y,target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if t % show_time == 0:
                print('Train times:{} loss: {:.3f}'.format(t,loss.data))
        if loss.data < 0.002 :
            return

def validation(evil,criterion,train_loader):
    val_loss = 0
    num_batches = len(train_loader)
    # 调整模式
    evil.eval()
    with torch.no_grad():
        for x,y in train_loader:
            p = evil(x)
            val_loss += criterion(p,y).item()
    val_loss /= num_batches
    print('Loss: {}'.format(val_loss))

def Save(evil,path):
    torch.save(evil,path)
