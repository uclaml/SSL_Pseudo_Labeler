import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import random

class LinearModel(nn.Module):
    
    def __init__(self, d):
        super().__init__()
        self.linear = nn.Linear(d, 1, bias=False)
        
    def forward(self, x):
        return self.linear(x)

def dataloader1(d, n, sigma, feature, seed=4340):
    
    np.random.seed(seed)
    
    # positive samples
    x_pos = np.zeros([n, d * 2])
    x_pos[ :, :d] = np.tile(feature, (n, 1))
    x_pos[ :, d:] = sigma * np.random.randn(n, d)
    
    y_pos = np.zeros([n])
    
    # negative samples
    x_neg = np.zeros([n, d * 2])
    x_neg[ :, :d] = -1 * np.tile(feature, (n, 1))
    x_neg[ :, d:] = sigma * np.random.randn(n, d)
    
    y_neg = np.ones([n])
    
    x = np.concatenate([x_pos, x_neg], axis=0)
    y = np.concatenate([y_pos, y_neg], axis=0)
    
    return x, np.int64(y)

# generate both labeled data and unlabeled data
def dataloader2(d, n_l, n_u, sigma, feature, seed=1141):
    
    np.random.seed(seed)
    
    # positive samples
    x_pos = np.zeros([2, n_l + n_u, d])
    y_pos = np.zeros([n_l+n_u])
    
    # generate labeled data
    x_pos[0, :n_l, :] = np.tile(feature, (n_l, 1))
    x_pos[1, :n_l, :] = sigma * np.random.randn(n_l, d)
    
    y_pos[:n_l] = np.zeros([n_l])
    
    # generate pseudo-labeled data
    x_pos[0, n_l:n_l+n_u, :] = np.tile(feature, (n_u, 1))
    x_pos[1, n_l:n_l+n_u, :] = sigma * np.random.randn(n_u, d)
    x_input_pos = np.zeros([n_u, d * 2])
    x_input_pos[:, :d] = x_pos[0, n_l:n_l+n_u, :].copy()
    x_input_pos[:, d:] = x_pos[1, n_l:n_l+n_u, :].copy()
    # print(x_input_pos)
    # print(model(torch.tensor(x_input_pos).float().to(device)))
    y_pos[n_l:n_l+n_u] = (model(torch.tensor(x_input_pos).float().to(device))>0).cpu().numpy()[:,0]
    
    # negative samples
    x_neg = np.zeros([2, n_l+n_u, d])
    y_neg = np.zeros([n_l+n_u])
    
    # generate labeled data
    x_neg[0, :n_l, :] = -1 * np.tile(feature, (n_l, 1))
    x_neg[1, :n_l, :] = sigma * np.random.randn(n_l, d)

    y_neg[:n_l] = np.ones([n_l])
    
    # generate pseudo-labeled data
    x_neg[0, n_l:n_l+n_u, :] = -1 * np.tile(feature, (n_u, 1))
    x_neg[1, n_l:n_l+n_u, :] = sigma * np.random.randn(n_u, d)
    x_input_neg = np.zeros([n_u, d * 2])
    x_input_neg[:, :d] = x_neg[0, n_l:n_l+n_u, :].copy()
    x_input_neg[:, d:] = x_neg[1, n_l:n_l+n_u, :].copy()
    y_neg[n_l:n_l+n_u] = (model(torch.tensor(x_input_neg).float().to(device))[:,0] > 0).cpu().numpy()
    # print(x_input_neg)
    # print(model(torch.tensor(x_input_neg).float().to(device)))
    
    x = np.concatenate([x_pos.transpose(1, 0, 2), x_neg.transpose(1, 0, 2)], axis=0)
    y = np.concatenate([y_pos, y_neg], axis=0)

    return x, np.int64(y)

def dataloader3(d, n, sigma, feature, seed=1244):
    
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

d = 10000
n = 100
n_test = 100
lr = 1e-4
epochs = 100
device = "cuda:0"

# generate feature vector v with ||v||=\Theta(d^{1/2})
feature = np.zeros([1, d])
feature[0, :] = np.random.randn(1, d)
# print(feature)

sigma_p = 10 * (d ** 0.01)
data_x, data_y = dataloader1(d=d, n=n, sigma=sigma_p, feature=feature)
test_data_x, test_data_y = dataloader1(d=d, n=n_test, sigma=sigma_p, feature=feature,seed=1230)
# print(data_x.shape, data_y.shape)

model = LinearModel(d * 2).to("cuda:0")

optimizer = torch.optim.SGD(model.parameters(), lr=lr)

model.train()
criteria = nn.BCEWithLogitsLoss()

for epoch in range(epochs):
    x, y = torch.FloatTensor(data_x).to(device), torch.from_numpy(data_y).to(device)
    
    output = model(x)

    loss = criteria(output[:,0], y.float())
    y_pred = torch.tensor(output[:,0]>0)
    
    err = torch.mean((y!=y_pred).float())
    print(loss.item(),err)
    
    optimizer.zero_grad()
    loss.backward()
    
    optimizer.step()

model.eval()
test_x, test_y = torch.FloatTensor(test_data_x).to(device), torch.from_numpy(test_data_y).to(device)
test_output = model(test_x)
test_y_pred = torch.tensor(test_output[:,0]>0)
test_loss = criteria(test_output[:,0], test_y.float())
test_err = torch.mean((test_y!=test_y_pred).float())
print(test_loss, test_err)

d = 10000
n_l = 10
n_u = 10000
n_test = 1000
m = 20
lr = 1e-4
epochs = 200
device = "cuda:0"

data_x, data_y = dataloader2(d=d, n_l=n_l, n_u=n_u, sigma=sigma_p, feature=feature, seed=1230)

CNNmodel=CNN(m=m, d=d).to("cuda:0")

# optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
lambda_= 0.1
optimizer = torch.optim.SGD(CNNmodel.parameters(), lr=lr, weight_decay=lambda_)

CNNmodel.train()
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

CNNmodel.apply(init_weights)

log_feat, log_noise = [], []

for epoch in range(epochs):
    x, y = torch.FloatTensor(data_x).to(device), torch.from_numpy(data_y).to(device)
    
    output = CNNmodel(x)
    loss = criteria(output, y)
    
    print (loss.item(),CNNmodel.log(x))

    a, b = CNNmodel.status(x)
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

# import matplotlib.pyplot as plt

# plt.plot(max_feat1)
# plt.plot(max_noise1)
# plt.xlabel('Iterations')
# plt.ylabel('Value')
# plt.legend([r'Feature Learning: $\max_j\  \langle  w_{j},  v\rangle$', 
#             r'Noise Memorization: $\max_i\ \max_j\  \langle  w_{j},  \xi_i\rangle$'])

# plt.savefig('test11.pdf')

CNNmodel.eval()
x, y = torch.FloatTensor(data_x).to(device), torch.from_numpy(data_y).to(device)
output = CNNmodel(x)
loss = criteria(output, y)
pred = torch.argmax(output, dim=-1)
err = torch.mean((y!=pred).float())
print('training loss is', loss,', training error is ', err)

test_data_x, test_data_y = dataloader3(d=d, n=n_test, sigma=sigma_p, feature=feature, seed=1000)
test_x, test_y = torch.FloatTensor(test_data_x).to(device), torch.from_numpy(test_data_y).to(device)
test_output = CNNmodel(test_x)
test_loss = criteria(test_output, test_y)
test_pred = torch.argmax(test_output, dim=-1)
test_err = torch.mean((test_y!=test_pred).float())
print('test loss is', test_loss,', test error is ', test_err)

class LogisticRegression(nn.Module):
     def __init__(self, input_dim, output_dim):
         super(LogisticRegression, self).__init__()
         self.linear = nn.Linear(input_dim, output_dim)
     def forward(self, x):
         outputs = torch.sigmoid(self.linear(x))
         return outputs
     
epochs = 50
input_dim = 2 # Two inputs x1 and x2 
output_dim = 1 # Single binary output 
learning_rate = 0.5

data_x_l=np.zeros((n_l*2,2,d))
data_x_l[:n_l,:,:]=data_x[:n_l,:,:].copy()
data_x_l[n_l:,:,:]=data_x[n_u+n_l:n_u+2*n_l,:,:].copy()
data_y_l=np.zeros(n_l*2)
data_y_l[:n_l]=data_y[:n_l]
data_y_l[n_l:]=data_y[n_u+n_l:n_u+2*n_l].copy()
x_l, y_l = torch.FloatTensor(data_x_l), torch.from_numpy(data_y_l)
CNNoutput = CNNmodel(torch.FloatTensor(x_l).to(device)).to('cpu')
print(CNNoutput.shape,y_l.shape)

test_data_x, test_data_y = dataloader3(d=d, n=n_test, sigma=sigma_p, feature=feature, seed=1100)
test_x, test_y = torch.FloatTensor(test_data_x).to(device), torch.from_numpy(test_data_y).to(device)
test_X=CNNmodel(test_x)

downstream_model = LogisticRegression(input_dim,output_dim).to(device)
criterion = nn.BCELoss()
optimizer = torch.optim.SGD(downstream_model.parameters(), lr=learning_rate)

def init_weights_2(m):
    for param in m.parameters():
        nn.init.constant_(param,0)

downstream_model.apply(init_weights_2)

losses = []
losses_test = []
Iterations = []
iter = 0

for epoch in range(epochs):
    x, y = torch.FloatTensor(CNNoutput).to(device), torch.from_numpy(data_y_l).to(device)
    optimizer.zero_grad()
    outputs = downstream_model(x)
    loss = criterion(torch.squeeze(outputs),y.to(torch.float32))
    loss.backward(retain_graph=True)
    optimizer.step()
    print(loss.item())
    iter+=1
    if iter%10==0:
        with torch.no_grad():
            # Calculating the loss and accuracy for the test dataset
            correct_test = 0
            total_test = 0
            # print(test_X.shape)
            outputs_test = torch.squeeze(downstream_model(test_X))
            loss_test = criterion(outputs_test, test_y.to(torch.float32))
            
            predicted_test = outputs_test.round().detach().cpu().numpy()
            total_test += test_y.size(0)
            correct_test += np.sum(predicted_test == test_y.detach().cpu().numpy())
            accuracy_test = 100 * correct_test/total_test
            losses_test.append(loss_test.item())
            
            # Calculating the loss and accuracy for the train dataset
            total = 0
            correct = 0
            total += y.size(0)
            correct += np.sum(torch.squeeze(outputs).round().detach().cpu().numpy() == y.detach().cpu().numpy())
            accuracy = 100 * correct/total
            losses.append(loss.item())
            Iterations.append(iter)
            
            print(f"Iteration: {iter}. \nTest - Loss: {loss_test.item()}. Accuracy: {accuracy_test}")
            print(f"Train -  Loss: {loss.item()}. Accuracy: {accuracy}\n")