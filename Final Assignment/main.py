from datasets import load_dataset
from transformers import AutoTokenizer
import datasets
import chardet  
from datasets import Dataset, DatasetDict, Sequence, ClassLabel, Value
import tensorflow as tf
from transformers import DataCollatorForTokenClassification
from transformers import TFAutoModelForTokenClassification
from transformers import create_optimizer
import tensorflow as tf
import evaluate
import numpy as np
from seqeval.metrics import classification_report as seqeval_classification_report
import pandas as pd
from collections import Counter
import random
import os
import re

model_checkpoint = "bert-base-cased"
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

# Define label names and mappings
label_names = [
    "O", "B-ADR", "I-ADR", "B-Drug", "I-Drug",
    "B-Disease", "I-Disease", "B-Symptom", "I-Symptom",
    "B-Finding", "I-Finding"
]
label_mapping = {label: idx for idx, label in enumerate(label_names)}


def label_to_id(label):
    """Map a label to its corresponding ID."""
    return label_mapping.get(label, -100)


def detect_encoding(filepath):
    """Detect file encoding using chardet."""
    with open(filepath, 'rb') as f:
        raw_data = f.read()
        result = chardet.detect(raw_data)
    return result['encoding']


def read_bio_file(filepath):
    """Read a BIO file and extract tokens, NER tags, and ADR codes."""
    encoding = detect_encoding(filepath)
    print(f"Detected encoding for {filepath}: {encoding}")
    
    sentences = []
    current_sentence = {"tokens": [], "ner_tags": [], "adr_codes": []}

    with open(filepath, "r", encoding=encoding) as f:
        for line in f:
            line = line.strip()
            if line == "":
                if current_sentence["tokens"]:
                    sentences.append(current_sentence)
                    current_sentence = {"tokens": [], "ner_tags": [], "adr_codes": []}
            else:
                parts = line.split(maxsplit=2)
                if len(parts) == 3:
                    token, label, code = parts
                    current_sentence["tokens"].append(token)
                    current_sentence["ner_tags"].append(label_to_id(label))
                    current_sentence["adr_codes"].append(code)
                elif len(parts) == 2:
                    token, label = parts
                    current_sentence["tokens"].append(token)
                    current_sentence["ner_tags"].append(label_to_id(label))
                    current_sentence["adr_codes"].append(None)
                else:
                    print(f"Skipping malformed line: {line}")
    
    if current_sentence["tokens"]:
        sentences.append(current_sentence)
    
    return sentences


def create_dataset(train_file, val_file, test_file):
    """Create a DatasetDict from BIO files."""
    train_data = read_bio_file(train_file)
    val_data = read_bio_file(val_file)
    test_data = read_bio_file(test_file)
    
    dataset = DatasetDict({
        "train": Dataset.from_dict({
            "tokens": [d["tokens"] for d in train_data],
            "ner_tags": [d["ner_tags"] for d in train_data],
            "adr_codes": [d["adr_codes"] for d in train_data]
        }),
        "validation": Dataset.from_dict({
            "tokens": [d["tokens"] for d in val_data],
            "ner_tags": [d["ner_tags"] for d in val_data],
            "adr_codes": [d["adr_codes"] for d in val_data]
        }),
        "test": Dataset.from_dict({
            "tokens": [d["tokens"] for d in test_data],
            "ner_tags": [d["ner_tags"] for d in test_data],
            "adr_codes": [d["adr_codes"] for d in test_data]
        })
    })

    ner_feature = Sequence(ClassLabel(names=label_names))
    adr_feature = Sequence(Value("string"))

    dataset = dataset.cast_column("ner_tags", ner_feature)
    dataset = dataset.cast_column("adr_codes", adr_feature)
    
    return dataset

def align_labels_with_tokens(labels, word_ids):
    """Align labels with tokens after tokenization."""
    new_labels = []
    current_word = None
    for word_id in word_ids:
        if word_id != current_word:
            current_word = word_id
            label = -100 if word_id is None else labels[word_id]
            new_labels.append(label)
        elif word_id is None:
            new_labels.append(-100)
        else:
            label = labels[word_id]
            if label % 2 == 1:
                label += 1
            new_labels.append(label)
    return new_labels


def tokenize_and_align_labels(dataset, tokenizer):
    """Tokenize dataset and align labels."""
    def tokenize_fn(examples):
        tokenized_inputs = tokenizer(
            examples["tokens"], truncation=True, is_split_into_words=True
        )
        all_labels = examples["ner_tags"]
        new_labels = []
        for i, labels in enumerate(all_labels):
            word_ids = tokenized_inputs.word_ids(i)
            new_labels.append(align_labels_with_tokens(labels, word_ids))
        tokenized_inputs["labels"] = new_labels
        return tokenized_inputs
    
    return dataset.map(
        tokenize_fn,
        batched=True,
        remove_columns=dataset["train"].column_names,
    )

def prepare_tf_datasets(tokenized_datasets, tokenizer):
    """Convert tokenized datasets to TensorFlow datasets."""
    data_collator = DataCollatorForTokenClassification(
        tokenizer=tokenizer,
        return_tensors="tf"
    )
    
    tf_train_dataset = tokenized_datasets["train"].to_tf_dataset(
        columns=["attention_mask", "input_ids", "labels", "token_type_ids"],
        collate_fn=data_collator,
        shuffle=True,
        batch_size=16,
    )
    
    tf_eval_dataset = tokenized_datasets["validation"].to_tf_dataset(
        columns=["attention_mask", "input_ids", "labels", "token_type_ids"],
        collate_fn=data_collator,
        shuffle=False,
        batch_size=16,
    )
    
    tf_test_dataset = tokenized_datasets["test"].to_tf_dataset(
        columns=["attention_mask", "input_ids", "labels", "token_type_ids"],
        collate_fn=data_collator,
        shuffle=False,
        batch_size=16,
    )
    
    return tf_train_dataset, tf_eval_dataset, tf_test_dataset

def initialize_model(model_checkpoint):
    """Initialize the Token Classification model."""
    id2label = {i: label for i, label in enumerate(label_names)}
    label2id = {v: k for k, v in id2label.items()}
    
    model = TFAutoModelForTokenClassification.from_pretrained(
        model_checkpoint,
        id2label=id2label,
        label2id=label2id,
    )
    return model

def compile_model(model, train_dataset, num_epochs=1):
    """Compile the model with optimizer and learning rate schedule."""
    num_train_steps = len(train_dataset) * num_epochs
    
    optimizer, schedule = create_optimizer(
        init_lr=2e-5,
        num_warmup_steps=0,
        num_train_steps=num_train_steps,
        weight_decay_rate=0.01,
    )
    
    model.compile(optimizer=optimizer)
    return model

# Load and preprocess datase
# edit training set here
dataset = create_dataset("Train-test-split/train_Org.txt", "Train-test-split/validation_Org.txt", "Train-test-split/test_Org.txt")
tokenized_datasets = tokenize_and_align_labels(dataset, tokenizer)

# Prepare TensorFlow datasets
tf_train, tf_val, tf_test = prepare_tf_datasets(tokenized_datasets, tokenizer)

# Initialize and compile model
model = initialize_model('bert-base-cased')
model = compile_model(model, tf_train)

# Training (fit separately)
model.fit(tf_train, validation_data=tf_val, epochs=3)

# Save the trained model and tokenizer
# edit name for train set
output_dir = "saved_model/Org/"
model.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)

print(f"Model and tokenizer saved to {output_dir}")


# Load and preprocess datase
# edit training set here
dataset = create_dataset("Train-test-split/train_Meddra.txt", "Train-test-split/validation_Org.txt", "Train-test-split/test_Org.txt")
tokenized_datasets = tokenize_and_align_labels(dataset, tokenizer)

# Prepare TensorFlow datasets
tf_train, tf_val, tf_test = prepare_tf_datasets(tokenized_datasets, tokenizer)

# Initialize and compile model
model = initialize_model('bert-base-cased')
model = compile_model(model, tf_train)

# Training (fit separately)
model.fit(tf_train, validation_data=tf_val, epochs=3)

# Save the trained model and tokenizer
# edit name for train set
output_dir = "saved_model/Meddra/"
model.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)

print(f"Model and tokenizer saved to {output_dir}")

# Load and preprocess datase
# edit training set here
dataset = create_dataset("Train-test-split/train_Sct.txt", "Train-test-split/validation_Org.txt", "Train-test-split/test_Org.txt")
tokenized_datasets = tokenize_and_align_labels(dataset, tokenizer)

# Prepare TensorFlow datasets
tf_train, tf_val, tf_test = prepare_tf_datasets(tokenized_datasets, tokenizer)

# Initialize and compile model
model = initialize_model('bert-base-cased')
model = compile_model(model, tf_train)

# Training (fit separately)
model.fit(tf_train, validation_data=tf_val, epochs=3)

# Save the trained model and tokenizer
# edit name for train set
output_dir = "saved_model/Sct/"
model.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)

print(f"Model and tokenizer saved to {output_dir}")

# Load and preprocess datase
# edit training set here
dataset = create_dataset("Train-test-split/train_All.txt", "Train-test-split/validation_Org.txt", "Train-test-split/test_Org.txt")
tokenized_datasets = tokenize_and_align_labels(dataset, tokenizer)

# Prepare TensorFlow datasets
tf_train, tf_val, tf_test = prepare_tf_datasets(tokenized_datasets, tokenizer)

# Initialize and compile model
model = initialize_model('bert-base-cased')
model = compile_model(model, tf_train)

# Training (fit separately)
model.fit(tf_train, validation_data=tf_val, epochs=3)

# Save the trained model and tokenizer
# edit name for train set
output_dir = "saved_model/Org/"
model.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)

print(f"Model and tokenizer saved to {output_dir}")