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