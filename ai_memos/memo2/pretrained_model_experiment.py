"""

References:
    1. https://www.geeksforgeeks.org/deep-learning/how-to-use-hugging-face-pretrained-model/
    2.https://huggingface.co/google-bert/bert-base-uncased
    3. https://huggingface.co/docs/transformers/en/tasks/masked_language_modeling
    4. https://www.geeksforgeeks.org/machine-learning/next-sentence-prediction-using-bert/
"""
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch
from loguru import logger
if __name__=="__main__":
    # Load pretrained model and tokenizer
    model_name = "bert-base-uncased"
    logger.info(f"Loading pretrained model {model_name}")
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    logger.info(f"Loading tokenizer!")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    # Sample text for classification
    text = "I am very happy and lucky!"
    # Tokenize the input text
    inputs = tokenizer(text, return_tensors="pt")
    # Perform inference
    with torch.no_grad():
        outputs = model(**inputs)

    # Get the predicted class
    predictions = torch.argmax(outputs.logits, dim=-1)
    print(f"Predicted class: {predictions.item()}")
