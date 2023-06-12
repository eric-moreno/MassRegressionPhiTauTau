import torch 
import torch.nn as nn 
import torch.nn.functional as F 
import torch.optim as optim 
import torch.utils.data as utils
import torch.nn.init as init
from torch.autograd import Variable


class SimpleGru(nn.Module):
    def __init__(self):
        super(SimpleGru, self).__init__()
        
        self.gru1 = nn.GRU(10,4,2,batch_first=True,dropout=0.5)
        self.gru2 = nn.GRU(13,4,2,batch_first=True,dropout=0.5)
        self.linear1  = nn.Linear(20,40)
        self.linear2  = nn.Linear(40,20)
        self.linear3  = nn.Linear(20,10)
        self.linear4  = nn.Linear(10,8)
        self.linear5  = nn.Linear(8,4)
        #self.linear6  = nn.Linear(6,3)
        #self.linear7  = nn.Linear(3,1)
        
        self.dropout = nn.Dropout(p=0.5)
        self.batchnorm1 = nn.BatchNorm1d(40)
        self.batchnorm2 = nn.BatchNorm1d(20)
        self.batchnorm3 = nn.BatchNorm1d(10)
        self.batchnorm4 = nn.BatchNorm1d(8)
        #self.batchnorm5 = nn.BatchNorm1d(6)
        #self.batchnorm6 = nn.BatchNorm1d(3)

    def forward(self, x1,x2,x3):
        #print(x1.shape)
        out1, hidden1 = self.gru1(x1)
        out2, hidden2 = self.gru2(x2)
        #print(out[:,-1].shape)
        #print(hidden.shape)
        #print(torch.flatten(out,start_dim=1).shape)
        #x = torch.flatten(out,start_dim=1)
        #print(x.shape)
        #x2 = self.gru2(x2)
        #print(type(x1[0]))
        #print(x1[0][:,-1].shape)
        x = torch.cat((out1[:,-1],out2[:,-1],x3),1)

        x = self.batchnorm1(self.linear1(x))
        x = F.leaky_relu(x)
        x = self.dropout(x)
        x = self.batchnorm2(self.linear2(x))
        x = F.leaky_relu(x)
        x = self.dropout(x)
        x = self.batchnorm3(self.linear3(x))
        x = F.leaky_relu(x)
        x = self.dropout(x)
        x = self.batchnorm4(self.linear4(x))
        x = F.leaky_relu(x)
        #x = self.dropout(x)
        #x = self.batchnorm5(self.linear5(x))
        #x = F.leaky_relu(x)
        #x = self.dropout(x)
        #x = self.batchnorm6(self.linear6(x))
        #x = F.leaky_relu(x)
        #x = self.linear7(x)
        x = self.linear5(x)
        return x

class gruTestData(utils.Dataset):
    
    def __init__(self, X1, X2, X3):
        self.X1 = X1
        self.X2 = X2
        self.X3 = X3
        
    def __getitem__(self, index):
        return self.X1[index], self.X2[index],self.X3[index]
        
    def __len__ (self):
        return len(self.X1)
    
class gruTrainData(utils.Dataset):
    
    def __init__(self, X1,X2,X3, y):
        self.X1 = X1
        self.X2 = X2
        self.X3 = X3
        self.y = y
        
    def __getitem__(self, index):
        return self.X1[index], self.X2[index],self.X3[index],self.y[index]
        
    def __len__ (self):
        return len(self.X1)    
    
class trainData(utils.Dataset):
    
    def __init__(self, X_data, y_data):
        self.X_data = X_data
        self.y_data = y_data
        
    def __getitem__(self, index):
        return self.X_data[index], self.y_data[index]
        
    def __len__ (self):
        return len(self.X_data)



## test data    
class testData(utils.Dataset):
    
    def __init__(self, X_data):
        self.X_data = X_data
        
    def __getitem__(self, index):
        return self.X_data[index]
        
    def __len__ (self):
        return len(self.X_data)
    

class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        
        self.linear1 = nn.Linear(377,400)
        self.linear2 = nn.Linear(400,200)
        self.linear3 = nn.Linear(200,100)
        self.linear4 = nn.Linear(100,50)
        self.linear5 = nn.Linear(50,25)
        self.linear6 = nn.Linear(25,12)
        self.linear_out = nn.Linear(12,1)
        
        self.dropout = nn.Dropout(p=0.1)
        self.batchnorm1 = nn.BatchNorm1d(400)
        self.batchnorm2 = nn.BatchNorm1d(200)
        self.batchnorm3 = nn.BatchNorm1d(100)
        self.batchnorm4 = nn.BatchNorm1d(50)
        self.batchnorm5 = nn.BatchNorm1d(25)
        self.batchnorm6 = nn.BatchNorm1d(12)
        

    def forward(self, x):
        x = self.batchnorm1(self.linear1(x))
        x = F.leaky_relu(x)
        x = self.dropout(x)
        x = self.batchnorm2(self.linear2(x))
        x = F.leaky_relu(x)
        x = self.dropout(x)
        x = self.batchnorm3(self.linear3(x))
        x = F.leaky_relu(x)
        x = self.dropout(x)
        x = self.batchnorm4(self.linear4(x))
        x = F.leaky_relu(x)
        x = self.dropout(x)
        x = self.batchnorm5(self.linear5(x))
        x = F.leaky_relu(x)
        x = self.dropout(x)
        x = self.batchnorm6(self.linear6(x))
        x = F.leaky_relu(x)
        x = self.linear_out(x)
        return x
