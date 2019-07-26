import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import pickle
import random
import numpy as np
from torch.autograd import Variable


#data = pickle.load(open("synthfacade.npy"))

third = sorted(list(os.walk("train_third_prog"))[0][2])
full  = sorted(list(os.walk("train_full_prog"))[0][2])

third = [map(toMat, pickle.load("train_third_prog/" + i) for i in third)]
full = [map(toMat, pickle.load("train_full_prog/" + i) for i in third)]

data = [(third(i), full(i)) for i in range(len(third))]

def toMat(for_tup):
    x = np.zeros(8)
    x[0] = for_tup.i_offset/9.0
    x[1] = for_tup.j_offset/9.0
    x[2] = for_tup.i_n/9.0
    x[3] = for_tup.j_n/9.0
    x[4] = for_tup.i_size/5.0
    x[5] = for_tup.j_size/5.0
    x[6] = for_tup.i_mul/9.0
    x[7] = for_tup.j_mul/9.0
    return x

train_loader = torch.utils.data.DataLoader([(torch.from_numpy(data[i][0]).float(), torch.from_numpy(data[i][1]).float()) for i in range(len(data) - 100)],
    batch_size=64, shuffle=True)
test_loader = torch.utils.data.DataLoader([(torch.from_numpy(data[i][0]).float(), torch.from_numpy(data[i][1]).float()) for i in range(len(data) - 100, len(data))],
    batch_size=64, shuffle=True)

learning_rate = 1e-4


class NeuralNet(nn.Module):
    def __init__(self, input_size=8, hidden_size=8, output_size = 8):
        super(NeuralNet, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size) 
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.relu(out)
        out = self.fc3(out)
        return self.sigmoid(out)


model = NeuralNet()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)  
loss_function = nn.MSELoss()

outf = "prog2prog-models-facade"

def train(epoch):
    model.train()
    train_loss = 0
    for batch_idx, (data_x, data_y) in enumerate(train_loader):
        data = Variable(data_x)
        optimizer.zero_grad()

        recon = model(data)
        #print(recon_batch.shape)
        loss = loss_function(recon, data_y)
        loss.backward()
        train_loss += loss.data[0]
        optimizer.step()
        if False: #batch_idx % 5 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader),
                loss.data[0] / len(data)))
    print('====> Epoch: {} Average loss: {:.6f}'.format(
          epoch, train_loss / len(train_loader.dataset)))

    if epoch % 50 == 0:
        torch.save(model.state_dict(), '%s/nn_epoch_%d.pth' % (outf, epoch))


def test(epoch):
    model.eval()
    test_loss = 0
    for i, (data_x, data_y) in enumerate(test_loader):
        data = Variable(data_x, volatile=True)
        if torch.cuda.is_available():
            data = data.cuda()
            data_y = data_y.cuda()
        recon_batch = model(data)
            #print(a.shape)
    test_loss /= len(test_loader.dataset)
    print('====> Test set loss: {:.4f}'.format(test_loss))


for epoch in range(1, 10001):
    train(epoch)
