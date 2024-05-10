import flwr as fl
from datasets import load_dataset
from rouge import Rouge
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

def calculate_rouge(reference, prediction):
    rouge = Rouge()
    scores = rouge.get_scores(reference, prediction)
    return scores[0]["rouge-1"]["f"], scores[0]["rouge-2"]["f"], scores[0]["rouge-l"]["f"]

# Define strategy
strategy = fl.server.strategy.FedAvg(
    fraction_fit=1.0,
    fraction_evaluate=1.0,
    min_fit_clients=1,
    min_evaluate_clients=1,
)

# Start server
fl.server.start_server(
    server_address="0.0.0.0:8089",
    config=fl.server.ServerConfig(num_rounds=1),
    strategy=strategy,
    grpc_max_message_length=1024 * 1024 * 1024,
)

# Get the final model parameters and evaluation results from the strategy
final_parameters = strategy.parameters
eval_results = strategy.evaluation_results

if final_parameters:
    # Load the initial model
    initial_model = AutoModelForSeq2SeqLM.from_pretrained("t5-small")

    # Create a new model instance with the final parameters
    trained_model = AutoModelForSeq2SeqLM.from_pretrained("t5-small", state_dict=final_parameters)

    if eval_results:
        # Aggregate evaluation results
        total_loss = sum(result.loss * result.num_examples for result in eval_results) / sum(result.num_examples for result in eval_results)
        total_examples = sum(result.num_examples for result in eval_results)
        rouge_1 = sum(result.metrics["rouge-1"] * result.num_examples for result in eval_results) / total_examples
        rouge_2 = sum(result.metrics["rouge-2"] * result.num_examples for result in eval_results) / total_examples
        rouge_l = sum(result.metrics["rouge-l"] * result.num_examples for result in eval_results) / total_examples

        print(f"Evaluation Results:")
        print(f"Average Loss: {total_loss:.4f}")
        print(f"ROUGE-1 Score: {rouge_1:.4f}")
        print(f"ROUGE-2 Score: {rouge_2:.4f}")
        print(f"ROUGE-L Score: {rouge_l:.4f}")
    else:
        print("No evaluation results received from clients.")
else:
    print("No parameters received from clients.")