import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

# Load data
df = pd.read_csv(r'C:\Users\jatha\Desktop\College\Sem6\EDI\fraud\data.csv')
df = df.rename(columns={'oldbalanceOrg': 'oldBalanceOrig', 'newbalanceOrig': 'newBalanceOrig',
                        'oldbalanceDest': 'oldBalanceDest', 'newbalanceDest': 'newBalanceDest'})

# Sample data
sampled_data = df.sample(n=900000, random_state=42)
sampled_data.reset_index(drop=True, inplace=True)

# Data cleaning
X = sampled_data.loc[(sampled_data.type == 'TRANSFER') | (sampled_data.type == 'CASH_OUT')]
randomState = 5
np.random.seed(randomState)
Y = X['isFraud']
del X['isFraud']

X = X.drop(['nameOrig', 'nameDest', 'isFlaggedFraud'], axis=1)
X.loc[X.type == 'TRANSFER', 'type'] = 0
X.loc[X.type == 'CASH_OUT', 'type'] = 1
X.type = X.type.astype(int)

X.loc[(X.oldBalanceDest == 0) & (X.newBalanceDest == 0) & (X.amount != 0),
['oldBalanceDest', 'newBalanceDest']] = -1

X['errorBalanceOrig'] = X.newBalanceOrig + X.amount - X.oldBalanceOrig
X['errorBalanceDest'] = X.oldBalanceDest + X.amount - X.newBalanceDest


# Deep Learning Model
class Neural_Network(nn.Module):
    def __init__(self):
        super(Neural_Network, self).__init__()
        self.fc1 = nn.Linear(9, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 32)
        self.fc4 = nn.Linear(32, 10)
        self.fc5 = nn.Linear(10, 2)  # Output layer with 2 classes

    def forward(self, x):
        x = F.sigmoid(self.fc1(x))
        x = F.sigmoid(self.fc2(x))
        x = F.sigmoid(self.fc3(x))
        x = F.sigmoid(self.fc4(x))
        x = self.fc5(x)  # Output layer without activation
        return x


# Parameters
learning_rate = 0.1
split_ratio = 0.2
batch_size = 64
num_epochs = 2

# Split data
train_x, test_x, train_y, test_y = train_test_split(X, Y, test_size=split_ratio, random_state=randomState)

# Convert types
train_x = torch.Tensor(np.array(train_x))
test_x = torch.Tensor(np.array(test_x))
train_y = torch.Tensor(np.array(train_y)).type(torch.LongTensor)
test_y = torch.Tensor(np.array(test_y)).type(torch.LongTensor)

# DataLoader
train_dataset = torch.utils.data.TensorDataset(train_x, train_y)
test_dataset = torch.utils.data.TensorDataset(test_x, test_y)

trainloader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
testloader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

# Initialize the model
model = Neural_Network()

# Loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)

# Train the neural network
for epoch in range(num_epochs):
    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        inputs, labels = data
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        if i % 1000 == 999:  # Print every 1000 mini-batches
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 1000))
            running_loss = 0.0

print('Finished Training')

# Test the model
correct = 0
total = 0
with torch.no_grad():
    for data in testloader:
        inputs, labels = data
        outputs = model(inputs)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print('Accuracy of the network on the test data: %f %%' % (
        100 * correct / total))
