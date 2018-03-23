
# coding: utf-8

# import libraries

# In[16]:


import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.utils.data as Data
import torchvision
import matplotlib.pyplot as plt
from functools import reduce


# define hyperparameters

# In[22]:


EPOCH = 1000
BATCH_SIZE = 64
LR = 0.01


# download and import mnist data

# In[18]:


train_data = torchvision.datasets.MNIST(root='./',train=False,download=True,transform=torchvision.transforms.ToTensor())
test_data = torchvision.datasets.MNIST(root='./',train=False,transform=torchvision.transforms.ToTensor())
train_loader = Data.DataLoader(dataset=train_data,batch_size=BATCH_SIZE,shuffle=True,num_workers=2)
test_x = Variable(test_data.test_data.view(-1, 28*28)).type(torch.FloatTensor)
test_y = test_data.test_labels


# Define Fully Connected Neural Network

# In[19]:


class FCNets(nn.Module):
    def __init__(self,nlist):
        super(FCNets,self).__init__()
        if len(nlist) < 2:
            print('error:not enough layers')
        else:
            self.fc = nn.Sequential()
            for n in range(len(nlist)-1):
                self.fc.add_module('linear' + str(n+1), nn.Linear(in_features=nlist[n], out_features=nlist[n+1]))
                self.fc.add_module('relu' + str(n+1), nn.ReLU())
            #self.fc.add_module('softmax', nn.Softmax())
    def forward(self,x):
        return self.fc(x)


# Define Neural Net instance and optimizer, loss function

# In[20]:


fc = FCNets([28*28,32,10])
print(fc)
optimizer = torch.optim.SGD(params=fc.parameters(), lr=LR)
loss_func = nn.CrossEntropyLoss()


# In[21]:


for epoch in range(EPOCH):
    for step,(x,y) in enumerate(train_loader):
        x = x.view(-1,28*28)
        bx = Variable(x)
        by = Variable(y)
        output = fc(bx)
        loss = loss_func(output, by)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
    if epoch%5 == 0:
        y_predict = torch.max(fc(test_x),1)[1].data
        accuracy = sum(y_predict == test_y)/len(test_y)
        print('Epoch:',epoch,'train loss:%.4f' %loss.data[0],'test accuracy is:%.3f' %accuracy)


# if neural nets size is 784 10
# with lr = 0.01 
# achieve accuracy of 92%
# if neural nets size is 784 32 10
# with lr = 0.01 epoch=600
# achieve accuracy of 98.2%

# In[ ]:


for param in fc.parameters():
    print(type(param.data), param.size())

