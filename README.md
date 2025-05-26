# LLM-Arena

**_The open benchmarking arena for local LLMs._**

---

## 🎯 Purpose

LLM-Arena provides a lightweight and reproducible framework to **benchmark, compare, and analyze local large language models**. Whether you're evaluating Mistral, LLaMA, TinyLLaMA, or Phi, this repo gives you a head-to-head testing ground for instruction-following, summarization, chat, and more — all on your machine.

---

## 📁 Repo Structure

```
LLM-Arena/
├── models/           # Loaders and wrappers for various local LLMs
├── prompts/          # Prompt templates for evaluation tasks
├── evaluate.py       # Main script to run evaluations across models
├── metrics.py        # Utility functions for computing metrics (BLEU, ROUGE, etc.)
├── results/          # Store results in JSON/CSV for reproducibility
├── demo.ipynb        # Interactive notebook for exploring outputs
├── README.md         # Project overview and getting started
└── requirements.txt  # Required Python packages
```

---

## 🛣 Roadmap

1. Create test set group given this paper : https://arxiv.org/html/2412.01020v1

