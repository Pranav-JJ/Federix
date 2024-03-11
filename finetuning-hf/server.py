import flwr as fl
from datasets import load_dataset
from rouge import Rouge
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from typing import Any


def calculate_rouge(reference, prediction):
    rouge = Rouge()
    scores = rouge.get_scores(reference, prediction)
    return scores[0]["rouge-1"]["f"], scores[0]["rouge-2"]["f"], scores[0]["rouge-l"]["f"]


# Initialize variables to store reference and prediction for ROUGE calculation
reference = []
prediction = []


def aggregate_reference_and_prediction(reference_batch: Any,
                                       prediction_batch: Any) -> None:
    global reference
    global prediction
    reference.extend(reference_batch)
    prediction.extend(prediction_batch)


if __name__ == "_main_":
    # Load test dataset
    test_dataset = load_dataset("lighteval/legal_summarization", "BillSum")["test"]
    references = [example["summary"] for example in test_dataset]

    # Define strategy
    strategy = fl.server.strategy.FedAvg(
        fraction_fit=1.0,
        fraction_evaluate=1.0,
    )

    # Start server
    fl.server.start_server(
        server_address="0.0.0.0:8089",
        config=fl.server.ServerConfig(num_rounds=3),
        strategy=strategy,
    )

    # Load the trained model for inference
    net = AutoModelForSeq2SeqLM.from_pretrained("trained_model")

    # After training is complete, save the model
    net.save_pretrained("trained_model")

    # Instantiate tokenizer
    tokenizer = AutoTokenizer.from_pretrained("t5-small")  # Use the appropriate tokenizer here

    # Generate summaries for a few articles from the test dataset
    generated_summaries = []
    for example in test_dataset[:5]:  # Example: Generate summaries for the first 5 articles
        input_ids = tokenizer(example["article"], truncation=True, padding=True, return_tensors="pt")["input_ids"]
        outputs = net.generate(input_ids.to("cuda"))
        generated_summary = tokenizer.decode(outputs[0], skip_special_tokens=True)
        generated_summaries.append(generated_summary)

    # Calculate ROUGE scores
    rouge_1, rouge_2, rouge_l = calculate_rouge(references[:5], generated_summaries)
    print("ROUGE-1 Score:", rouge_1)
    print("ROUGE-2 Score:", rouge_2)
    print("ROUGE-L Score:", rouge_l)