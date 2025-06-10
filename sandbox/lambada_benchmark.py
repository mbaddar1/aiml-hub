"""
Relevant Links
https://medium.com/nlplanet/two-minutes-nlp-keeping-track-of-information-and-the-lambada-benchmark-b808dd5af15c
https://stackoverflow.com/a/76318628
"""
from datasets import load_dataset
if __name__=="__main__":
    ds = load_dataset("cimec/lambada")
    train_ds = ds["train"]
    one_sample = train_ds.select([1])
    query_text = one_sample["text"]
    pass