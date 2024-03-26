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
DEVICE = torch.device("cpu")
CHECKPOINT = "t5-small"  # transformer model checkpoint


def load_data(node_id):
    """Load dataset (training and eval)"""
    dataset = load_dataset("lighteval/legal_summarization", "BillSum")
    full_train_dataset = dataset["train"]
    eval_dataset = dataset["test"]

    tokenizer = AutoTokenizer.from_pretrained(CHECKPOINT)

    # Split the full training dataset into two halves
    train_dataset_size = len(full_train_dataset)
    train_dataset_1 = full_train_dataset.select(range(0, train_dataset_size // 30))
    train_dataset_2 = full_train_dataset.select(range(train_dataset_size // 2, train_dataset_size))

    # Choose one half as the training data
    train_dataset = train_dataset_1

    train_dataset = train_dataset.map(
        lambda x: tokenizer.prepare_seq2seq_batch(x["article"], x["summary"]),
        batched=True,
    )

    eval_dataset = eval_dataset.map(
        lambda x: tokenizer.prepare_seq2seq_batch(x["article"], x["summary"]),
        batched=True,
    )

    trainloader = DataLoader(train_dataset, batch_size=4, collate_fn=lambda data: collate_fn(data, tokenizer))
    evalloader = DataLoader(eval_dataset, batch_size=4, collate_fn=lambda data: collate_fn(data, tokenizer))

    return trainloader, evalloader


def collate_fn(data, tokenizer):
    """Collate function to convert data into tensors"""
    # Initialize lists to store tokenized articles and summaries
    tokenized_articles = []
    tokenized_summaries = []

    # Iterate over each dictionary in the list
    for item in data:
        # Tokenize the article and summary
        tokenized_item = tokenizer(item["article"], item["summary"], truncation=True, padding=True, return_tensors="pt")

        # Append tokenized article to the list
        tokenized_articles.append(tokenized_item["input_ids"])

        # Check if "labels" key is present in the tokenized item
        if "labels" in tokenized_item and "labels" in tokenized_item:
            # If "labels" key is present, append tokenized summary to the list
            tokenized_summaries.append(tokenized_item["labels"])
        else:
            # If "labels" key is not present, use "input_ids" as a placeholder for the summary
            # You may need to adjust this logic based on the tokenizer's behavior
            tokenized_summaries.append(tokenized_item["input_ids"])

    # Convert lists to tensors
    tokenized_articles = torch.stack(tokenized_articles).squeeze(dim=1)  # Remove singleton dimension
    tokenized_summaries = torch.stack(tokenized_summaries).squeeze(dim=1)  # Remove singleton dimension

    return {"input_ids": tokenized_articles, "labels": tokenized_summaries}


def train(net, trainloader, epochs):
    optimizer = AdamW(net.parameters(), lr=5e-5)
    net.train()
    total_batches = len(trainloader)
    print("Training started...")
    for i, batch in enumerate(trainloader, start=1):
        inputs = {k: v.to(torch.device("cuda")) for k, v in batch.items()}  # Move all tensors to GPU
        labels = inputs.pop("labels", None)  # Remove labels from inputs
        outputs = net(*inputs, labels=labels) if labels is not None else net(*inputs)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        # Print progress within the single epoch
        print(f"\rBatch {i}/{total_batches} - Loss: {loss.item():.4f}", end="", flush=True)
    print("\nTraining finished.")


def main(node_id):
    net = AutoModelForSeq2SeqLM.from_pretrained(
        CHECKPOINT,
    ).to("cuda")

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
            train(net, trainloader, epochs=1)
            print("Training Finished.")
            return self.get_parameters(config={}), len(trainloader), {}

        def evaluate(self, parameters, config):
            # Add evaluation functionality here if needed
            pass

    # Start client
    fl.client.start_client(
        server_address="172.19.0.179:8089", client=PlaceholderClient().to_client()
    )


if __name__ == "_main_":
    parser = argparse.ArgumentParser(description="Flower")
    parser.add_argument(
        "--node-id",
        choices=list(range(3)),
        required=True,
        type=int,
        help="Partition of the dataset divided into 1,000 iid partitions created "
             "artificially.",
    )
    node_id = parser.parse_args().node_id
    main(node_id)