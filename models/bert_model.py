from datasets import load_dataset
from transformers import BertForQuestionAnswering, BertTokenizer

# import evaluate
# import numpy as np
# import torch

# Load model and tokenizer
model_name = "bert-base-uncased"
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertForQuestionAnswering.from_pretrained(model_name)

dataset = load_dataset("squad")
print("Dataset loaded:", dataset)
