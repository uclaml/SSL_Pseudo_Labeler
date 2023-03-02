import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

# 
def dataloader(d, n, sigma, feature, seed=3100, shuffle=True):
    
    np.random.seed(seed)
    
    # positive samples
    x_pos = np.zeros([2, n, d])
    x_pos[0, :, :] = np.tile(feature, (n, 1))
    x_pos[1, :, :] = sigma * np.random.randn(n, d)
    
    y_pos = np.zeros([n])
    
    # negative samples
    x_neg = np.zeros([2, n, d])
    x_neg[0, :, :] = -1 * np.tile(feature, (n, 1))
    x_neg[1, :, :] = sigma * np.random.randn(n, d)
    
    y_neg = np.ones([n])
    
    x = np.concatenate([x_pos.transpose(1, 0, 2), x_neg.transpose(1, 0, 2)], axis=0)
    y = np.concatenate([y_pos, y_neg], axis=0)

    return x, np.int64(y)

class CNN(nn.Module):
    
    def __init__(self, m, d):
        super().__init__()
        
        self.lin1 = nn.Linear(d, m, bias=False)
        self.lin2 = nn.Linear(d, m, bias=False)
        
    def activation(self, x):
        return (F.relu(x))**3
        
    def log(self, x):
        return torch.sum(self.activation(self.lin1(x[0, 0, :])), dim=-1), torch.sum(self.activation(self.lin1(x[0, 1, :])), dim=-1)
    
    def status(self, x):
        feature_learning = self.lin1(x[:, 0, :])
        noise_memorization = self.lin1(x[:, 1, :])
        
        return feature_learning, noise_memorization
    
    def forward(self, x):
        output1 = torch.sum(self.activation(self.lin1(x[:, 0, :]))
                            + self.activation(self.lin1(x[:, 1, :])), dim=-1, keepdim=True)
        output2 = torch.sum(self.activation(self.lin2(x[:, 0, :]))
                            + self.activation(self.lin2(x[:, 1, :])), dim=-1, keepdim=True)
        output = torch.cat([output1, output2], dim=-1)
        
        return output

def evaluate(model, x, y):
    model.eval()
    x, y = torch.FloatTensor(x).to(device), torch.from_numpy(y).to(device)
    output = model(x)
    loss = criteria(output, y)
    pred = torch.argmax(output, dim=-1)
    err = torch.mean((y!=pred).float())
    return loss.item(), err.item()

d = 10000
n = 10
n_test = 500
m = 20
lr = 1e-4
epochs = 200
device = "cuda:0"

model = CNN(m=m, d=d).to("cuda:0")

# generate feature vector v with ||v||=\Theta(d^{1/2})
feature = np.zeros([1, d])
feature[0, :] = np.random.randn(1, d)
print(feature)

sigma_p = 10 * (d ** 0.01)
data_x, data_y = dataloader(d=d, n=n, sigma=sigma_p, feature=feature)


# optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
# lambda_= d ** 0.7
lambda_ = 0
optimizer = torch.optim.SGD(model.parameters(), lr=lr)

model.train()
criteria = nn.CrossEntropyLoss()

# initialize all weights from N(0,sigma_0^2)
sigma_0 = d ** -0.75 / 10

# print all the parameters
print('Dimension d is', d)
print('Sample size n is', n)
print('Number of neurons m is', m)
print('Learning rate eta is', lr)
print('regularizer lambda is', lambda_)
print('STD of noise patch sigma_p is', sigma_p)
print('STD of weight initialization sigma_0 is', sigma_0)

def init_weights(m):
    for param in m.parameters():
        nn.init.normal_(param, mean=0.0, std=sigma_0)

model.apply(init_weights)

log_feat, log_noise = [], []

for epoch in range(epochs):
    x, y = torch.FloatTensor(data_x).to(device), torch.from_numpy(data_y).to(device)
    
    output = model(x)
    loss = criteria(output, y)
    
    print (loss.item(),model.log(x))

    a, b = model.status(x)
    log_feat += [a.detach().cpu().numpy()]
    log_noise += [b.detach().cpu().numpy()]
    
    
    optimizer.zero_grad()
    loss.backward()
    
    optimizer.step()
    
import matplotlib.pyplot as plt

log_a, log_b = np.array(log_feat), np.array(log_noise)
# print(log_a, log_b)
max_feat1 = log_a[:,0:epochs,:].max(axis=(1,2))
# max_feat2 = -log_a[:,50:,:].max(axis=(1,2))

max_noise1 = log_b[:, 0:epochs, :].max(axis=(1,2))
print(max_feat1)
print(max_noise1)

import matplotlib.pyplot as plt

plt.plot(max_feat1)
plt.plot(max_noise1)
plt.xlabel('Iterations')
plt.ylabel('Value')
plt.legend([r'Feature Learning: $\max_j\  \langle  w_{j},  v\rangle$', 
            r'Noise Memorization: $\max_i\ \max_j\  \langle  w_{j},  \xi_i\rangle$'])

plt.savefig('test8.pdf')

model.eval()
x, y = torch.FloatTensor(data_x).to(device), torch.from_numpy(data_y).to(device)
output = model(x)
loss = criteria(output, y)
pred = torch.argmax(output, dim=-1)
err = torch.mean((y!=pred).float())
print('training error is ', err)

model.eval()
test_data_x, test_data_y = dataloader(d=d, n=n_test, sigma=sigma_p, feature=feature, seed=1210)
test_x, test_y = torch.FloatTensor(test_data_x).to(device), torch.from_numpy(test_data_y).to(device)
test_output = model(test_x)
test_loss = criteria(test_output, test_y)
test_pred = torch.argmax(test_output, dim=-1)
test_err = torch.mean((test_y!=test_pred).float())
print('test loss is', test_loss,', test error is ', test_err)

model.eval()
test_data_x, test_data_y = dataloader(d=d, n=n_test, sigma=sigma_p, feature=feature, seed=1200)
test_x, test_y = torch.FloatTensor(test_data_x).to(device), torch.from_numpy(test_data_y).to(device)
test_output = model(test_x)
test_loss = criteria(test_output, test_y)
test_pred = torch.argmax(test_output, dim=-1)
test_err = torch.mean((test_y!=test_pred).float())
print('test loss is', test_loss,', test error is ', test_err)

model.eval()
test_data_x, test_data_y = dataloader(d=d, n=n_test, sigma=sigma_p, feature=feature, seed=1110)
test_x, test_y = torch.FloatTensor(test_data_x).to(device), torch.from_numpy(test_data_y).to(device)
test_output = model(test_x)
test_loss = criteria(test_output, test_y)
test_pred = torch.argmax(test_output, dim=-1)
test_err = torch.mean((test_y!=test_pred).float())
print('test loss is', test_loss,', test error is ', test_err)

model.eval()
test_data_x, test_data_y = dataloader(d=d, n=n_test, sigma=sigma_p, feature=feature, seed=1100)
test_x, test_y = torch.FloatTensor(test_data_x).to(device), torch.from_numpy(test_data_y).to(device)
test_output = model(test_x)
test_loss = criteria(test_output, test_y)
test_pred = torch.argmax(test_output, dim=-1)
test_err = torch.mean((test_y!=test_pred).float())
print('test loss is', test_loss,', test error is ', test_err)

model.eval()
test_data_x, test_data_y = dataloader(d=d, n=n_test, sigma=sigma_p, feature=feature, seed=1210)
test_x, test_y = torch.FloatTensor(test_data_x).to(device), torch.from_numpy(test_data_y).to(device)
test_output = model(test_x)
test_loss = criteria(test_output, test_y)
test_pred = torch.argmax(test_output, dim=-1)
test_err = torch.mean((test_y!=test_pred).float())
print('test loss is', test_loss,', test error is ', test_err)