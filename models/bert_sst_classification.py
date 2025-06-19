import os
from typing import Optional

import evaluate
import fire
import hydra
import mlflow
import numpy as np
import torch
from datasets import load_dataset
from omegaconf import OmegaConf
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    logging,
)


def main(num_epochs: Optional[int], train_fraction: Optional[float]):
    # fix warning
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    with hydra.initialize(config_path="configs"):
        hydra_cfg = hydra.compose(config_name="config")

    cli_conf = OmegaConf.create(
        {"training": {"num_epochs": num_epochs, "train_fraction": train_fraction}}
    )
    cfg = OmegaConf.merge(hydra_cfg, cli_conf)
    print("Config finished")

    # Configuration
    mlflow_tracking_uri = "http://localhost:5228"
    mlflow_experiment_name = "SST2-DistilBERT"
    model_name = "distilbert-base-uncased"
    model_save_path = "./saved_models/distilbert_sst2"
    max_length = 512
    num_epochs = 1
    dataset_name = "stanfordnlp/sst2"
    eval_metric = "accuracy"

    # Setup MlFlow
    mlflow.set_tracking_uri(mlflow_tracking_uri)
    mlflow.set_experiment(mlflow_experiment_name)
    mlflow.transformers.autolog()

    # Load dataset SST2
    dataset = load_dataset(dataset_name)

    # len(train_text) == 67349
    # train_fraction is used to take only part of train dataset, not all
    # 0.0 < train_fraction <= 1.0
    n_rows = int(len(dataset["train"]["sentence"]) * train_fraction)

    train_texts = dataset["train"]["sentence"][:n_rows]
    train_labels = dataset["train"]["label"][:n_rows]

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

    metric = evaluate.load(eval_metric)

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        return metric.compute(predictions=predictions, references=labels)

    training_args = TrainingArguments(
        output_dir="./logs/bert_sst2_classification",
        eval_strategy="steps",
        eval_steps=100,
        save_strategy="steps",
        num_train_epochs=num_epochs,
        weight_decay=0.01,
        load_best_model_at_end=True,
        metric_for_best_model=eval_metric,
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

    # Train model
    print("### Model training started ###\n")
    trainer.train()

    # Final accuracy
    accuracy_score = trainer.evaluate()["eval_accuracy"]
    print(f"\nPost-training Accuracy : {round(accuracy_score, 3)}\n")

    # Save model
    trainer.save_model(model_save_path)
    tokenizer.save_pretrained(model_save_path)


if __name__ == "__main__":
    fire.Fire(main)
