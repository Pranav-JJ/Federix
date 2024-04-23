import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import flwr as fl
from typing import List, Tuple

# Load the finance dataset
df = pd.read_csv(r'C:\Users\jatha\Desktop\College\Sem6\EDI\fraud\data.csv')
df = df.rename(columns={'oldbalanceOrg': 'oldBalanceOrig', 'newbalanceOrig': 'newBalanceOrig',
                        'oldbalanceDest': 'oldBalanceDest', 'newbalanceDest': 'newBalanceDest'})
sampled_data = df.sample(n=500000, random_state=42)
sampled_data.reset_index(drop=True, inplace=True)

# DATA CLEANING
X = sampled_data.loc[(sampled_data.type == 'TRANSFER') | (sampled_data.type == 'CASH_OUT')]
randomState = 5
np.random.seed(randomState)
Y = X['isFraud']
del X['isFraud']
X = X.drop(['nameOrig', 'nameDest', 'isFlaggedFraud'], axis=1)
X.loc[X.type == 'TRANSFER', 'type'] = 0
X.loc[X.type == 'CASH_OUT', 'type'] = 1
X.type = X.type.astype(int)
X.loc[(X.oldBalanceDest == 0) & (X.newBalanceDest == 0) & (X.amount != 0), \
    ['oldBalanceDest', 'newBalanceDest']] = - 1
X['errorBalanceOrig'] = X.newBalanceOrig + X.amount - X.oldBalanceOrig
X['errorBalanceDest'] = X.oldBalanceDest + X.amount - X.newBalanceDest


# Define your Neural Network class
class Neural_Network(nn.Module):
    def __init__(self):
        super(Neural_Network, self).__init__()
        self.fc1 = nn.Linear(9, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 32)
        self.fc4 = nn.Linear(32, 10)
        self.fc5 = nn.Linear(10, 5)
        self.fc6 = nn.Linear(5, 1)

    def forward(self, x):
        x = self.fc1(x)
        x = F.sigmoid(self.fc2(x))
        x = F.sigmoid(self.fc3(x))
        x = F.sigmoid(self.fc4(x))
        x = F.sigmoid(self.fc5(x))
        x = F.sigmoid(self.fc6(x))
        return x


# Define the client's model using Flower client
class FlowerClient(fl.client.NumPyClient):
    def __init__(self, model: nn.Module):
        super(FlowerClient, self).__init__()
        self.model = model

    def get_parameters(self, config: dict) -> List[np.ndarray]:
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]

    def set_parameters(self, parameters: List[np.ndarray]) -> None:
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = {k: torch.tensor(v) for k, v in params_dict}
        self.model.load_state_dict(state_dict, strict=True)

    def fit(self, parameters: List[np.ndarray], config: dict) -> Tuple[List[np.ndarray], int, dict]:
        self.set_parameters(parameters)
        self.model.cuda()  # Move model to GPU

        # Convert data to tensors and move to GPU
        train_x, test_x, train_y, test_y = train_test_split(X, Y, test_size=0.2, random_state=42)
        train_x = torch.Tensor(np.array(train_x)).cuda()
        train_y = torch.Tensor(np.array(train_y)).type(torch.LongTensor).cuda()
        train_dataset = torch.utils.data.TensorDataset(train_x, train_y)
        trainloader = DataLoader(dataset=train_dataset, batch_size=64, shuffle=True)

        # Loss function and optimizer
        criterion = nn.BCEWithLogitsLoss()
        optimizer = optim.SGD(self.model.parameters(), lr=0.1, momentum=0.9)

        # Train the model
        for epoch in range(2):  # Number of epochs
            for inputs, labels in trainloader:
                optimizer.zero_grad()
                outputs = self.model(inputs)
                labels = labels.type_as(outputs)
                loss = criterion(outputs.squeeze(), labels)
                loss.backward()
                optimizer.step()

        return self.get_parameters(config={}), len(trainloader.dataset), {}

    def evaluate(self, parameters: List[np.ndarray], config: dict) -> Tuple[float, int, dict]:
        self.set_parameters(parameters)
        self.model.cuda()  # Move model to GPU

        # Convert data to tensors and move to GPU
        test_x, test_y = X, Y  # Use entire dataset for evaluation
        test_x = torch.Tensor(np.array(test_x)).cuda()
        test_y = torch.Tensor(np.array(test_y)).type(torch.LongTensor).cuda()
        test_dataset = torch.utils.data.TensorDataset(test_x, test_y)
        testloader = DataLoader(dataset=test_dataset, batch_size=64, shuffle=False)

        # Evaluate the model
        total = 0
        correct = 0
        with torch.no_grad():
            for inputs, labels in testloader:
                outputs = self.model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        accuracy = correct / total
        return 0.0, len(testloader.dataset), {"accuracy": float(accuracy)}


# Initialize your model and move to GPU
model = Neural_Network().cuda()

# Start Flower client
client = FlowerClient(model)
fl.client.start_client(server_address="127.0.0.1:8080", client=client.to_client())
