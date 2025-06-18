import os

import evaluate
import mlflow
import mlflow.transformers
import numpy as np
import torch
from datasets import load_dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    logging,
)

# fix warning
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Configuration
mlflow_link = "http://localhost:5228"
model_name = "distilbert-base-uncased"
model_save_path = "./saved_model"
max_length = 512
dataset_name = "stanfordnlp/sst2"
metric_name = "accuracy"

# Setup MlFlow
mlflow.set_tracking_uri(mlflow_link)
mlflow.set_experiment("SST2-DistilBERT")
mlflow.transformers.autolog()

# Load dataset SST2
dataset = load_dataset(dataset_name)

train_texts = dataset["train"]["sentence"]
train_labels = dataset["train"]["label"]

val_texts = dataset["validation"]["sentence"]
val_labels = dataset["validation"]["label"]

# Tokenize dataset
tokenizer = AutoTokenizer.from_pretrained(model_name)

train_encodings = tokenizer(
    train_texts, truncation=True, padding=True, max_length=max_length
)
val_encodings = tokenizer(
    val_texts, truncation=True, padding=True, max_length=max_length
)


# Create dataset class
class SSTDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {k: torch.tensor(v[idx]) for k, v in self.encodings.items()}
        item["labels"] = torch.tensor([self.labels[idx]])
        return item

    def __len__(self):
        return len(self.labels)


train_dataset = SSTDataset(train_encodings, train_labels)
eval_dataset = SSTDataset(val_encodings, val_labels)


# Ignore warning on model loading
logging.set_verbosity_error()

model = AutoModelForSequenceClassification.from_pretrained(
    model_name,
    num_labels=2,
)

metric = evaluate.load(metric_name)


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)


training_args = TrainingArguments(
    output_dir="./logs/bert_sst2_classification",
    eval_strategy="steps",
    eval_steps=100,
    save_strategy="steps",
    # learning_rate=2e-5,
    # per_device_train_batch_size=32,
    # per_device_eval_batch_size=64,
    num_train_epochs=1,
    weight_decay=0.01,
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
    dataloader_pin_memory=False,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    compute_metrics=compute_metrics,
)


# Check untrained model quality
accuracy_score = trainer.evaluate()["eval_accuracy"]
print(f"\nPre-training Accuracy : {round(accuracy_score, 3)}\n")

print("### Model training started ###")

# Train model
trainer.train()

# Final accuracy
accuracy_score = trainer.evaluate()["eval_accuracy"]
print(f"\nPost-training Accuracy : {round(accuracy_score, 3)}\n")

# Save model
trainer.save_model(model_save_path)
tokenizer.save_pretrained(model_save_path)
