"""
Plan
-----------
Experiment: Cost-Benefit Comparison of Tiny LLMs

Models:

TinyLLaMA-1.1B

Phi-2 (2.7B)

Task:

Text classification (binary sentiment analysis on IMDb reviews)

Metric:

Accuracy per $1 inference cost

Steps:

Load both models (e.g. via Hugging Face or vLLM)

Run inference on 500–1000 IMDb samples

Record accuracy on the test set

Measure or estimate inference cost (time, memory, hardware cost)

Compute: accuracy / cost

Compare and analyze trade-offs


==========================================================================================================
Links
https://www.datacamp.com/blog/top-small-language-models
This script is to compare a set of small language models against different text generation tasks

Pareto Frontier
https://www.databricks.com/blog/efficiently-estimating-pareto-frontiers

Pareto optimality, economy–effectiveness trade-offs and ion channel degeneracy: improving population modelling for single neurons
https://royalsocietypublishing.org/doi/10.1098/rsob.220073


The Fairness-Accuracy Pareto Front
https://arxiv.org/abs/2008.10797
"""
from argparse import ArgumentParser
from transformers import pipeline
import torch
from datetime import datetime

parser = ArgumentParser()
parser.add_argument("--model-names", nargs='+')
args = parser.parse_args()
from loguru import logger
import sys
from transformers import AutoTokenizer, AutoModelForCausalLM

RED = "\033[91m"
BLUE = "\033[94m"
RESET = "\033[0m"

logger.remove()
logger.add(sys.stderr, level="INFO")

if __name__ == "__main__":
    # Load model directly

    tokenizer = AutoTokenizer.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    model = AutoModelForCausalLM.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0")

    model_names = args.model_names
    torch.cuda.empty_cache()
    pipe = pipeline(
        task="text-generation",
        device_map="auto",
        model="TinyLlama/TinyLlama-1.1B-Chat-v1.0", torch_dtype="auto")
    logger.info(f"Successfully loaded the model with device = {pipe.device} and torch_dtype = {pipe.torch_dtype}")
    while True:
        try:
            user_input = input(f"{RED}. Please enter your prompt: {RESET}")

            messages = [
                {"role": "user", "content": user_input},
            ]
            start_time = datetime.now()
            responses = pipe(messages)
            end_time = datetime.now()
            print(f"Answer generated in {(end_time - start_time).seconds} seconds")
            answer = None
            for entry in responses[0]["generated_text"]:
                if entry["role"] == "assistant":
                    answer = entry["content"]
            print(f"{BLUE}Answer : {answer}{RESET}")
        except (EOFError, KeyboardInterrupt):
            print(f"{RED}\nExiting...{RESET}")
            break
