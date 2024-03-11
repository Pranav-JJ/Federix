import argparse
import warnings
from collections import OrderedDict

import flwr as fl
import torch
from torch.optim import AdamW

from transformers import AutoModelForSeq2SeqLM
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from datasets import load_dataset

warnings.filterwarnings("ignore", category=UserWarning)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CHECKPOINT = "t5-small"  # transformer model checkpoint
MAX_INPUT_LENGTH = 512  # Maximum length of the input to the model
MIN_TARGET_LENGTH = 5  # Minimum length of the output by the model
MAX_TARGET_LENGTH = 128  # Maximum length of the output by the model
BATCH_SIZE = 16  # Batch-size for training our model
LEARNING_RATE = 1e-5  # Learning-rate for training our model
MAX_EPOCHS = 5  # Maximum number of epochs we will train the model for


def load_data(node_id):
    """Load dataset (training and eval)"""
    dataset = load_dataset("lighteval/legal_summarization", "BillSum")
    train_dataset = dataset["train"]
    eval_dataset = dataset["test"]

    tokenizer = AutoTokenizer.from_pretrained(CHECKPOINT)

    train_dataset = train_dataset.map(
        lambda x: tokenizer.prepare_seq2seq_batch(x["article"], x["summary"], max_length=MAX_INPUT_LENGTH, truncation=True, padding="max_length"),
        batched=True,
    )

    eval_dataset = eval_dataset.map(
        lambda x: tokenizer.prepare_seq2seq_batch(x["article"], x["summary"], max_length=MAX_INPUT_LENGTH, truncation=True, padding="max_length"),
        batched=True,
    )

    trainloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
    evalloader = DataLoader(eval_dataset, batch_size=BATCH_SIZE, collate_fn=collate_fn)

    return trainloader, evalloader


def collate_fn(data):
    """Collate function to convert data into tensors"""
    tokenized_articles = [torch.tensor(d["input_ids"]) for d in data]
    tokenized_summaries = [torch.tensor(d["labels"]) for d in data]

    tokenized_articles = torch.stack(tokenized_articles).squeeze(dim=1)
    tokenized_summaries = torch.stack(tokenized_summaries).squeeze(dim=1)

    return {"input_ids": tokenized_articles, "labels": tokenized_summaries}



def train(net, trainloader, epochs):
    optimizer = AdamW(net.parameters(), lr=LEARNING_RATE)
    net.train()
    total_batches = len(trainloader)
    print("Training started...")
    for epoch in range(epochs):
        for i, batch in enumerate(trainloader, start=1):
            inputs = {k: v.to(DEVICE) for k, v in batch.items()}  # Move all tensors to the appropriate device
            labels = inputs.pop("labels", None)  # Remove labels from inputs
            outputs = net(**inputs, labels=labels) if labels is not None else net(**inputs)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            # Print progress within the single epoch
            print(f"\rEpoch {epoch + 1}/{epochs}, Batch {i}/{total_batches} - Loss: {loss.item():.4f}", end="", flush=True)
    print("\nTraining finished.")


def main(node_id):
    net = AutoModelForSeq2SeqLM.from_pretrained(
        CHECKPOINT,
    ).to(DEVICE)

    trainloader, _ = load_data(node_id)

    # Flower client
    class PlaceholderClient(fl.client.NumPyClient):
        def get_parameters(self, config):
            return [val.cpu().numpy() for _, val in net.state_dict().items()]

        def set_parameters(self, parameters):
            params_dict = zip(net.state_dict().keys(), parameters)
            state_dict = OrderedDict({k: torch.Tensor(v) for k, v in params_dict})
            net.load_state_dict(state_dict, strict=True)

        def fit(self, parameters, config):
            self.set_parameters(parameters)
            print("Training Started...")
            train(net, trainloader, epochs=MAX_EPOCHS)
            print("Training Finished.")
            return self.get_parameters(config={}), len(trainloader), {}

        def evaluate(self, parameters, config):
            # Add evaluation functionality here if needed
            pass

    # Start client
    fl.client.start_client(
        server_address="192.168.11.235:8080", client=PlaceholderClient().to_client()
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Flower")
    parser.add_argument(
        "--node-id",
        choices=list(range(2)),
        required=True,
        type=int,
        help="Partition of the dataset divided into 1,000 iid partitions created "
             "artificially.",
    )
    node_id = parser.parse_args().node_id
    main(node_id)
