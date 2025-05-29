from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from datasets import load_dataset
import numpy as np
import random
from tqdm import tqdm
from loguru import logger
if __name__ == "__main__":
    ds = load_dataset("Rowan/hellaswag",split="validation")
    N = ds.shape[0]
    k = 5000
    indices = [random.randint(1, N-1) for _ in range(k)]
    ds_sample = ds.select(indices=indices)
    logger.info(f"Loaded dataset with shape = {ds.shape} and selected randomly {k} samples")
    # Load model and tokenizer
    model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    logger.info(f"Loaded tokenizer")
    model = AutoModelForCausalLM.from_pretrained(model_name)
    logger.info(f"Successfully loaded model {model_name}")
    model.eval()
    correct_selection_count = 0
    for sample in tqdm(list(ds_sample.iter(1)),desc="Scoring samples"):
        context = sample["ctx"][0]
        endings = sample["endings"][0]
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
        top_scored_index = np.argmax(scores)
        if top_scored_index == int(sample["label"][0]):
            correct_selection_count +=1
    accuracy = float(correct_selection_count)/k
    logger.info(f"Accuracy = {accuracy}")
