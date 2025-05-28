from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import numpy as np
if __name__ == "__main__":
    # Load model and tokenizer
    model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    print(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    print("loading tokenizer")
    model = AutoModelForCausalLM.from_pretrained(model_name)
    print("loading model")
    model.eval()

    # Example: context and ending
    context = "What is the capital of france? "

    endings = ["The capital is Paris", "The capital is Berlin", "I don't know !!"]
    scores = []
    for ending in endings:
        # Combine and tokenize
        input_text = context + ending
        inputs = tokenizer(input_text, return_tensors="pt")
        with torch.no_grad():
            outputs = model(**inputs, labels=inputs["input_ids"])
            # CrossEntropyLoss averaged over all tokens
            neg_log_likelihood = outputs.loss.item()

        # Convert loss to log-likelihood score (optional: multiply by length for total LL)
        log_likelihood_score = -neg_log_likelihood
        scores.append(log_likelihood_score)

        print(f"Negative Log-Likelihood Score for the QA {input_text}: {log_likelihood_score}")
    print(f"The best answer (completion) for {context} is : {endings[np.argmax(scores)]}")