import argparse
import warnings
from collections import OrderedDict

import flwr as fl
import torch
import pandas as pd
from evaluate import load as load_metric
from torch.optim import AdamW
from flwr_datasets import FederatedDataset
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
from transformers import BertForSequenceClassification, BertTokenizer
from transformers import DataCollatorWithPadding

warnings.filterwarnings("ignore", category=UserWarning)
DEVICE = torch.device("cpu")
CHECKPOINT = "bert-base-uncased"  # transformer model checkpoint

def load_data(node_id):
    """Load synthetic-fraud-detection data (training and eval)"""
    fds = FederatedDataset(dataset="V12X-ksr/Fraud.csv", partitioners={"train": 1_000})
    partition = fds.load_partition(node_id)

    # Preprocess data
    partition = partition.map(preprocess_data, batched=True)

    # Split data into train and test
    partition_train_test = partition.train_test_split(test_size=0.2)

    trainloader = DataLoader(
        partition_train_test["train"], shuffle=True, batch_size=32
    )

    testloader = DataLoader(
        partition_train_test["test"], batch_size=32
    )

    return trainloader, testloader

def preprocess_data(examples):
    data = examples.copy()
    data['errorBalanceOrig'] = [x + y - z for x, y, z in zip(data['newbalanceOrig'], data['amount'], data['oldbalanceOrg'])]
    data['errorBalanceDest'] = [x + y - z for x, y, z in zip(data['oldbalanceDest'], data['amount'], data['newbalanceDest'])]
    data = data.remove_columns(['step', 'type', 'amount', 'nameOrig', 'oldbalanceOrg', 'newbalanceOrig', 'nameDest', 'oldbalanceDest', 'newbalanceDest', 'isFlaggedFraud'])
    data = data.to_tensor('float32')
    return data


def train(net, trainloader, epochs):
    optimizer = AdamW(net.parameters(), lr=5e-5)
    net.train()
    for _ in range(epochs):
        for batch in trainloader:
            batch = batch.to(DEVICE)
            outputs = net(batch[:, :-1])
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

def test(net, testloader):
    metric = load_metric("accuracy")
    loss = 0
    net.eval()
    for batch in testloader:
        batch = batch.to(DEVICE)
        with torch.no_grad():
            outputs = net(batch[:, :-1])
        loss += outputs.loss.item()
        predictions = torch.argmax(outputs.logits, dim=-1)
        metric.add_batch(predictions=predictions, references=batch[:, -1].long())
    loss /= len(testloader.dataset)
    accuracy = metric.compute()["accuracy"]
    return loss, accuracy

def main(node_id):
    net = BertForSequenceClassification.from_pretrained(CHECKPOINT, num_labels=2).to(DEVICE)

    trainloader, testloader = load_data(node_id)

    # Flower client
    class FraudClient(fl.client.NumPyClient):
        def get_parameters(self, config):
            return [val.cpu().numpy() for _, val in net.state_dict().items()]

        def set_parameters(self, parameters):
            params_dict = zip(net.state_dict().keys(), [torch.Tensor(param) for param in parameters])
            state_dict = OrderedDict({k: v for k, v in params_dict})
            net.load_state_dict(state_dict, strict=True)

        def fit(self, parameters, config):
            self.set_parameters(parameters)
            print("Training Started...")
            train(net, trainloader, epochs=1)
            print("Training Finished.")
            return self.get_parameters(config={}), len(trainloader), {}

        def evaluate(self, parameters, config):
            self.set_parameters(parameters)
            loss, accuracy = test(net, testloader)
            return float(loss), len(testloader), {"accuracy": float(accuracy)}

    # Start client
    fl.client.start_client(
        server_address="127.0.0.1:8080", client=FraudClient().to_client()
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Flower")
    parser.add_argument(
        "--node-id",
        choices=list(range(1_000)),
        required=True,
        type=int,
        help="Partition of the dataset divided into 1,000 iid partitions created "
        "artificially.",
    )
    node_id = parser.parse_args().node_id
    main(node_id)