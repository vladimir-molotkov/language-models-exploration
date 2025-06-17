from pathlib import Path

from datasets import load_dataset, load_from_disk
from transformers import BertTokenizerFast

dataset_path = Path(__file__).parent.parent / "data/tokenized_squad"
model_name = "bert-base-uncased"

# Load tokenizer
tokenizer = BertTokenizerFast.from_pretrained(model_name)

# Load SQuAD v1
dataset = load_dataset("squad")


def preprocess_function(examples):
    """
    Prepare and tokenize SQuAD v1 dataset

    Args:
        examples (Dataset): raw SQuAD dataset

    Returns:
        Dataset: tokenized dataset
    """
    tokenized_examples = tokenizer(
        examples["question"],
        examples["context"],
        # Only truncate the context (second sequence)
        truncation="only_second",
        max_length=384,  # Maximum length for BERT
        stride=128,  # Overlap between chunks when splitting long contexts
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding="max_length",
    )

    # Since one example might give us several features,
    # we need a map from feature to example
    sample_mapping = tokenized_examples.pop("overflow_to_sample_mapping")
    offset_mapping = tokenized_examples.pop("offset_mapping")

    tokenized_examples["start_positions"] = []
    tokenized_examples["end_positions"] = []

    for i, offsets in enumerate(offset_mapping):
        # Get the corresponding example index
        sample_index = sample_mapping[i]

        # Get the answer
        answers = examples["answers"][sample_index]
        start_char = answers["answer_start"][0]
        end_char = start_char + len(answers["text"][0])

        # Find the start and end token positions
        sequence_ids = tokenized_examples.sequence_ids(i)

        # Find the start and end of the context
        context_start = 0
        while sequence_ids[context_start] != 1:
            context_start += 1
        context_end = len(sequence_ids) - 1
        while sequence_ids[context_end] != 1:
            context_end -= 1

        # If the answer is out of the span, set to the CLS token
        if (start_char < offsets[context_start][0]) or (
            end_char > offsets[context_end][1]
        ):
            tokenized_examples["start_positions"].append(0)
            tokenized_examples["end_positions"].append(0)
        else:
            # Find the start and end tokens
            start_token = context_start
            while (start_token <= context_end) and (
                offsets[start_token][0] <= start_char
            ):
                start_token += 1
            tokenized_examples["start_positions"].append(start_token - 1)

            end_token = context_end
            while (
                (end_token >= context_start) and (offsets[end_token][1])
            ) >= end_char:
                end_token -= 1
            tokenized_examples["end_positions"].append(end_token + 1)

    return tokenized_examples


tokenized_dataset = dataset.map(
    preprocess_function,
    batched=True,
    remove_columns=dataset["train"].column_names,  # nbqa: E501
)
tokenized_dataset.set_format(
    "torch",
    columns=[
        "input_ids",
        "attention_mask",
        "token_type_ids",
        "start_positions",
        "end_positions",
    ],
)
tokenized_dataset.save_to_disk(dataset_path)
tokenized_dataset = load_from_disk(dataset_path)

print(f"\nTrain: {tokenized_dataset['train'].num_rows:,}")
print(f"Validation: {tokenized_dataset['validation'].num_rows:,}\n")
