import flwr as fl

# Define metric aggregation function
def aggregate_metrics(metrics):
    """Aggregate metrics from multiple clients by calculating mean averages."""
    num_examples_total = sum(num_examples for num_examples, _ in metrics)

    accuracy_total = sum(num_examples * metric["accuracy"] for num_examples, metric in metrics)
    accuracy = accuracy_total / num_examples_total

    # Additional metrics
    rec_total = sum(num_examples * metric["recall"] for num_examples, metric in metrics)
    rec = rec_total / num_examples_total

    prec_total = sum(num_examples * metric["precision"] for num_examples, metric in metrics)
    prec = prec_total / num_examples_total

    f1_total = sum(num_examples * metric["f1"] for num_examples, metric in metrics)
    f1 = f1_total / num_examples_total

    return {
        "accuracy": accuracy,
        "f1": f1,
        "precision": prec,
        "recall": rec,
    }


strategy = fl.server.strategy.FedAvg(
    fraction_fit=1.0, fraction_evaluate=1.0,
    evaluate_metrics_aggregation_fn=aggregate_metrics,
)

# Start server
fl.server.start_server(
    server_address="0.0.0.0:8089",
    config=fl.server.ServerConfig(num_rounds=1),
    strategy=strategy,
)