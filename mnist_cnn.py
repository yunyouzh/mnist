
# coding: utf-8

# import libraries

# In[17]:


import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.utils.data as Data
import torchvision
import matplotlib.pyplot as plt


# define hyperparameters

# In[18]:


EPOCH = 5
BATCH_SIZE = 64
LR = 0.0001


# download and import mnist data

# In[19]:


train_data = torchvision.datasets.MNIST(root='./',train=True,download=False,transform=torchvision.transforms.ToTensor())
test_data = torchvision.datasets.MNIST(root='./',train=False,transform=torchvision.transforms.ToTensor())
train_loader = Data.DataLoader(dataset=train_data,batch_size=BATCH_SIZE,shuffle=True,num_workers=2)
train_x = Variable(train_data.test_data.view(-1,1,28,28)).type(torch.FloatTensor)
train_y = train_data.test_labels
alltest_x = Variable(test_data.test_data.view(-1,1, 28,28)).type(torch.FloatTensor)
alltest_y = test_data.test_labels
val_x = alltest_x[:5000,:,:,:]
val_y = alltest_y[:5000]
test_x = alltest_x[5000:,:,:,:]
test_y = alltest_y[5000:]


# Define Fully Connected Neural Network

# In[20]:


class ConvNets(nn.Module):
    def __init__(self):
        super(ConvNets,self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1,16,5,1,2),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16,32,5,1,2),
            nn.ReLU(),
            nn.MaxPool2d(2),
            )
        self.out = nn.Linear(32*7*7, 10)
        
    def forward(self,x):
        x = self.conv(x)
        x = x.view(-1,32*7*7)
        x = self.out(x)
        return x


# Define Neural Net instance and optimizer, loss function

# In[21]:


cnn = ConvNets()
optimizer = torch.optim.Adam(params=cnn.parameters(), lr=LR)
loss_func = nn.CrossEntropyLoss()


# In[22]:


train_acc =  []
val_acc = []
epochs = []
train_loss = []


# In[23]:


for epoch in range(EPOCH):
    for step,(x,y) in enumerate(train_loader):
        x = x.view(-1,1,28,28)
        bx = Variable(x)
        by = Variable(y)
        output = cnn(bx)
        loss = loss_func(output, by)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if step%50 == 0:
            val_y_predict = torch.max(cnn(val_x),1)[1].data
            train_y_predict = torch.max(cnn(train_x[:5000,:,:,:]),1)[1].data
            val_accuracy = sum(val_y_predict == val_y)/len(val_y)
            train_accuracy = sum(train_y_predict == train_y[:5000])/len(train_y[:5000])
            train_acc.append(train_accuracy)
            val_acc.append(val_accuracy)
            epochs.append(epoch)
            train_loss.append(loss)
            print('Epoch:',epoch,'train loss:%.4f' %loss.data[0],'train accuracy is:%.4f' %train_accuracy,'validation accuracy is:%.4f' %val_accuracy)



# In[ ]:


test_y_predict = torch.max(cnn(test_x),1)[1].data
test_accuracy = sum(test_y_predict == test_y)/len(test_y)
train_y_predict = torch.max(cnn(train_x),1)[1].data
train_accuracy = sum(train_y_predict == train_y)/len(train_y)
print('Final train loss:%.4f' %loss.data[0],'Final train accuracy is:%.4f' %train_accuracy,'Final test accuracy is:%.4f' %test_accuracy)
        


# for param in fc.parameters():
#     print(type(param.data), param.size())

# In[ ]:


torch.save(cnn,'cnn.pkl')


# plt.plot(epochs,train_acc,'r',label = 'training accuracy')
# plt.plot(epochs,val_acc,'b',label= 'validation accuracy')
# plt.xlabel('epochs')
# plt.ylabel('accuracy')
# plt.legend()
# plt.show()

# plt.plot(epochs,[i.data[0] for i in train_loss])
# plt.xlabel('epochs')
# plt.ylabel('loss')
# plt.show()
