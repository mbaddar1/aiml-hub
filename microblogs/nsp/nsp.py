"""
This script shows a small example for Next-Sentence Prediction (NSP) using BERT
The base of this code is taken from here
    https://www.geeksforgeeks.org/machine-learning/next-sentence-prediction-using-bert/
"""
# SEED = 42
import nltk

nltk.download("punkt_tab")
from transformers import BertTokenizer, BertForNextSentencePrediction, Trainer, TrainingArguments  # ,set_seed
# set_seed(seed=SEED)
from datasets import Dataset, DatasetDict
from sklearn.metrics import roc_auc_score
import requests
import random
from loguru import logger
from tqdm import tqdm


def tokenize_function(examples):
    return tokenizer(examples['sentence1'], examples['sentence2'], truncation=True, padding='max_length',
                     max_length=128)


def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    acc = roc_auc_score(labels, preds)  # original code use accuracy
    return {"auc": acc}


def build_nsp_dataset(url, min_words=3, seed=None):
    """
    Build a dataset for next-sentence prediction (NSP).

    Args:
        url (str): Link to a plain text file (e.g., from Project Gutenberg).
        min_words (int, optional): Minimum number of words a sentence must have
            to be included. Defaults to 3.
        seed (int, optional): Random seed for reproducibility. Defaults to 42.

    Returns:
        List[Tuple[str, str, int]]: A shuffled dataset where each item is a tuple:
            (sentence1, sentence2, label), with label=1 for adjacent sentences
            and label=0 for random negative pairs.
    """
    # Fetch text
    text = requests.get(url).text
    logger.info(f"Retrieving text from {url}")
    # Naive sentence splitting (period-based)
    # sentences = [s.strip() for s in text.split(".") if len(s.strip().split()) > min_words]
    sentences = nltk.sent_tokenize(text)
    logger.info(f"Building NSP dataset with {len(sentences)} sentences")

    dataset = []

    # Positive examples (adjacent pairs)
    for i in tqdm(range(len(sentences) - 1), desc=f"Building Positive Samples for NSP dataset"):
        dataset.append((sentences[i], sentences[i + 1], 1))

    # Negative examples (random pairs, non-adjacent)
    num_neg = len(dataset)
    for _ in tqdm(range(num_neg), desc=f"Building Negative Samples for NSP dataset"):
        s1, s2 = random.sample(sentences, 2)
        dataset.append((s1, s2, 0))

    # Shuffle dataset
    random.shuffle(dataset)

    return dataset


if __name__ == "__main__":
    input_url = "https://www.gutenberg.org/cache/epub/27039/pg27039.txt"
    model_name = "bert-base-uncased"
    sentences = build_nsp_dataset(url=input_url)

    dataset = Dataset.from_dict({
        "sentence1": [s[0] for s in sentences],
        "sentence2": [s[1] for s in sentences],
        "label": [s[2] for s in sentences]
    })

    train_test_split: DatasetDict = dataset.train_test_split(test_size=0.2)
    train_dataset: Dataset = train_test_split['train']
    test_dataset: Dataset = train_test_split['test']

    tokenizer = BertTokenizer.from_pretrained(pretrained_model_name_or_path=model_name)

    train_dataset = train_dataset.map(tokenize_function, batched=True)
    test_dataset = test_dataset.map(tokenize_function, batched=True)
    """
    Args:
        input_ids (List[int]): Token IDs for the input sequence. Each token is mapped to an
            integer index in the model's embedding matrix. This is not the embedding itself, 
            but the index used to retrieve the corresponding embedding vector. In practice, 
            it provides the numerical representation of each token for the model.

        attention_mask (List[int], optional): Attention mask to distinguish between real tokens 
            and padding. A value of 1 indicates that the token should be attended to, while 
            0 indicates padding and should be ignored. For example, given a batch with 
            sentences of lengths 3 and 5, the masks would be `[1, 1, 1, 0, 0]` and 
            `[1, 1, 1, 1, 1]` respectively. Not all models strictly require this argument, 
            but it is commonly used to ensure correct handling of variable-length sequences.
    """
    train_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])
    test_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])
    """
    You can use variations of base-models from here
        https://huggingface.co/google-bert/bert-base-uncased#model-variations
    """
    model = BertForNextSentencePrediction.from_pretrained(model_name)

    training_args = TrainingArguments(
        output_dir="results",
        learning_rate=1e-5,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        num_train_epochs=3,
        weight_decay=0.001,
        logging_dir="./logs",
        logging_steps=10,
        save_strategy="epoch"
    )
    logger.info(f"Training Arguments: {training_args}")
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        compute_metrics=compute_metrics
    )

    trainer.train()

    eval_results = trainer.evaluate()
    print(f"\nEvaluation Results: {eval_results}")

"""
Experimental results
Experiment 1
training_args = TrainingArguments(
        output_dir="results",
        learning_rate=2e-5,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        num_train_epochs=3,
        weight_decay=0.01,
        logging_dir="./logs",
        logging_steps=10,
        save_strategy="epoch",
    )
Evaluation Results: {'eval_loss': 0.7426218390464783, 'eval_auc': 0.6498062015503876, 'eval_runtime': 13.1524, 'eval_samples_per_second': 20.453, 'eval_steps_per_second': 2.585, 'epoch': 3.0}
.....
Experiment 2
training_args = TrainingArguments(
        output_dir="results",
        learning_rate=1e-5,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        num_train_epochs=3,
        weight_decay=0.01,
        logging_dir="./logs",
        logging_steps=10,
        save_strategy="epoch",
    )
Evaluation Results: {'eval_loss': 0.6371045708656311, 'eval_auc': 0.6772425249169436, 'eval_runtime': 9.0709, 'eval_samples_per_second': 29.655, 'eval_steps_per_second': 3.748, 'epoch': 3.0}
....
Experiment 3
training_args = TrainingArguments(
        output_dir="results",
        learning_rate=1e-4,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        num_train_epochs=3,
        weight_decay=0.01,
        logging_dir="./logs",
        logging_steps=10,
        save_strategy="epoch",
    )
Evaluation Results: {'eval_loss': 1.1804970502853394, 'eval_auc': 0.5236710963455149, 'eval_runtime': 5.2087, 'eval_samples_per_second': 51.644, 'eval_steps_per_second': 6.528, 'epoch': 3.0}
training_args = TrainingArguments(
        output_dir="results",
        learning_rate=1e-4,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        num_train_epochs=3,
        weight_decay=0.01,
        logging_dir="./logs",
        logging_steps=10,
        save_strategy="epoch",
    )
------------------ 
LR      auc for NSO
2e-5    0.65
1e-5    0.68
1e-4    0.52
-------------------------------
weight_decay    roc_auc
0.1             0.68
0.1             0.67             
0.1             0.6476
---
0.01           0.6427
0.01           0.6999
0.01           0.653
-----------------------------
0.001           0.652
0.001           0.654
0.001           0.626
"""
