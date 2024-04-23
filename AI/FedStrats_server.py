import flwr as fl
import numpy as np


# Define metric aggregation function
def weighted_average(metrics):
    accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]
    return {"accuracy": sum(accuracies) / sum(examples)}


# Define strategies
fedavg_strategy = fl.server.strategy.FedAvg(
    fraction_fit=1.0,  # Use all available clients for training
    min_fit_clients=2,  # Minimum number of clients for training
    evaluate_metrics_aggregation_fn=weighted_average,
)

trimmed_avg_strategy = fl.server.strategy.FedTrimmedAvg(
    fraction_fit=1.0,
    min_fit_clients=2,
    beta=0.1,  # Fraction to cut off of both tails of the distribution
    evaluate_metrics_aggregation_fn=weighted_average,
)

# Start Flower server with selected strategies
fl.server.start_server(
    server_address="0.0.0.0:8080",
    config=fl.server.ServerConfig(num_rounds=3),
    strategy=trimmed_avg_strategy,  # Choose the desired strategies here
)
