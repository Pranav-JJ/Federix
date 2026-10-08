import argparse
import warnings
from collections import OrderedDict
import os
import flwr as fl
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    Trainer,
    TrainingArguments,
    Seq2SeqTrainingArguments,
    AutoConfig,
)

from transformers import logging as hf_logging

from datasets import load_dataset

warnings.filterwarnings("ignore", category=UserWarning)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CHECKPOINT = "t5-small"  # transformer model checkpoint


def load_data(partition_id):
    """Load legal summarization dataset"""
    dataset = load_dataset("joelniklaus/legal_case_document_summarization")
    partition = dataset["train"].train_test_split(test_size=0.2, seed=42)["train"]
    eval_partition = dataset["test"]

    tokenizer = AutoTokenizer.from_pretrained(CHECKPOINT)

    def tokenize_function(examples):
        return tokenizer(
            examples["judgement"],
            examples["summary"],
            padding="max_length",
            truncation=True,
            max_length=512,
            return_tensors="pt",
        )

    partition = partition.map(tokenize_function, batched=True)
    eval_partition = eval_partition.map(tokenize_function, batched=True)

    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=CHECKPOINT)
    trainloader = DataLoader(
        partition, batch_size=4, shuffle=True, collate_fn=data_collator
    )
    evalloader = DataLoader(
        eval_partition, batch_size=4, shuffle=False, collate_fn=data_collator
    )

    return trainloader, evalloader


def train(net, trainloader, epochs):
    optimizer = AdamW(net.parameters(), lr=1e-3)
    net.train()
    for _ in range(epochs):
        for batch in trainloader:
            batch = {k: v.to(DEVICE) for k, v in batch.items()}
            outputs = net(**batch)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()


def evaluate(net, evalloader):
    net.eval()
    eval_loss = 0.0
    for batch in evalloader:
        batch = {k: v.to(DEVICE) for k, v in batch.items()}
        with torch.no_grad():
            outputs = net(**batch)
            eval_loss += outputs.loss.item()
    eval_loss /= len(evalloader.dataset)
    return eval_loss


def main(partition_id):
    net = AutoModelForSeq2SeqLM.from_pretrained(CHECKPOINT).to(DEVICE)

    trainloader, evalloader = load_data(partition_id)

    # Login to Hugging Face Hub
    hf_login_command = "transformers-cli login"
    print("Please login to the Hugging Face Hub...")
    os.system(hf_login_command)

    # Flower client
    class LegalSummClient(fl.client.NumPyClient):
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
            config = AutoConfig.from_pretrained(CHECKPOINT)
            net.config = config
            net.save_pretrained("phoen1x/federated-legal-summarisation")
            return self.get_parameters(config={}), len(trainloader), {}

        def evaluate(self, parameters, config):
            self.set_parameters(parameters)
            print("Evaluating model...")
            eval_loss = evaluate(net, evalloader)
            return float(eval_loss), len(evalloader), {}

    # Start client
    fl.client.start_client(
        server_address="127.0.0.1:8080", client=LegalSummClient().to_client()
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Flower")
    parser.add_argument(
        "--partition-id",
        required=True,
        type=int,
        help="Partition of the dataset divided into 1,000 iid partitions created "
        "artificially.",
    )
    partition_id = parser.parse_args().partition_id
    main(partition_id)
