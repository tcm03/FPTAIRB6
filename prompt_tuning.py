import os
import ast
import logging
import torch
import pandas as pd
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    default_data_collator,
    get_linear_schedule_with_warmup,
)
from datasets import Dataset
from peft import (
    get_peft_model,
    PromptTuningConfig,
    PromptTuningInit,
    TaskType,
)

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(filename)s:%(lineno)d - %(funcName)s - %(levelname)s - %(message)s"
)

# Hyperparameters and settings
device = "cuda" if torch.cuda.is_available() else "cpu"
model_name_or_path = "checkpoints/Qwen2.5-Coder-7B"  # adjust if needed
batch_size = 8
num_epochs = 5
max_length = 128  # maximum sequence length for training examples
lr = 3e-2

# Define prompt tuning configuration with 20 virtual tokens
peft_config = PromptTuningConfig(
    task_type=TaskType.CAUSAL_LM,
    prompt_tuning_init=PromptTuningInit.TEXT,
    num_virtual_tokens=20,  # using 20 virtual tokens
    prompt_tuning_init_text="Select the correct answer:",
    tokenizer_name_or_path=model_name_or_path,
)

# Initialize tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

def load_and_process_csv(csv_path: str) -> Dataset:
    """
    Loads a CSV file and converts it to a Hugging Face Dataset.
    Assumes the CSV has columns: task_id, question, choices, answer.
    The choices column is parsed from its string representation.
    """
    df = pd.read_csv(csv_path)
    # Parse the choices column (assumed to be stored as string representation of list)
    df["choices"] = df["choices"].apply(ast.literal_eval)
    if "answer" in df.columns:
        df = df[df['answer'].notna() & (df['answer'] != '')]
        df = df.reset_index(drop=True)
    return Dataset.from_pandas(df)

def preprocess_function(examples):
    """
    For each example, construct an input sequence by concatenating the question and choices,
    and set the target as the answer (we use the last character of the answer string).
    The final sequence is the concatenation of the prompt (input) and target,
    with the input portion masked out in the labels.
    """
    inputs = []
    targets = []
    for q, choices, ans in zip(examples["question"], examples["choices"], examples["answer"]):
        # Construct the prompt – note the choices are joined by a comma.
        prompt_text = f"Question: {q}\nChoices: {', '.join(choices)}\nAnswer: "
        print(f"prompt_text: {prompt_text}")
        return
        inputs.append(prompt_text)
        # Assume the answer letter is the last nonempty character in the answer string
        ans_letter = ans.strip()[-1] if ans.strip() else ""
        targets.append(ans_letter)
    
    # Tokenize prompt and target separately
    tokenized_inputs = tokenizer(inputs, truncation=True, max_length=max_length, padding=False)
    tokenized_targets = tokenizer(targets, truncation=True, max_length=16, padding=False)
    
    input_ids_list = []
    labels_list = []
    for i in range(len(inputs)):
        inp_ids = tokenized_inputs["input_ids"][i]
        # Append a pad token at the end of the target tokens if needed
        target_ids = tokenized_targets["input_ids"][i] + [tokenizer.pad_token_id]
        # Concatenate prompt and target tokens
        combined_ids = inp_ids + target_ids
        # For labels, mask the prompt part (set to -100) so that loss is computed only on target tokens.
        labels = [-100] * len(inp_ids) + target_ids

        # Pad the sequences to max_length
        if len(combined_ids) < max_length:
            pad_len = max_length - len(combined_ids)
            combined_ids = combined_ids + [tokenizer.pad_token_id] * pad_len
            labels = labels + ([-100] * pad_len)
        else:
            combined_ids = combined_ids[:max_length]
            labels = labels[:max_length]

        input_ids_list.append(combined_ids)
        labels_list.append(labels)
    return {"input_ids": input_ids_list, "labels": labels_list}

def main():
    # Load training and evaluation datasets from CSV
    train_dataset = load_and_process_csv("data/b6_train_data.csv")
    eval_dataset = load_and_process_csv("data/b6_test_data.csv")
    
    # Apply the preprocessing to tokenize and format the examples
    train_dataset = train_dataset.map(
        preprocess_function,
        batched=True,
        remove_columns=train_dataset.column_names,
        load_from_cache_file=False,
        desc="Tokenizing train dataset",
    )
    return
    eval_dataset = eval_dataset.map(
        preprocess_function,
        batched=True,
        remove_columns=eval_dataset.column_names,
        load_from_cache_file=False,
        desc="Tokenizing eval dataset",
    )

    # Create DataLoaders for training and evaluation
    train_dataloader = DataLoader(
        train_dataset, shuffle=True, batch_size=batch_size, collate_fn=default_data_collator, pin_memory=True
    )
    eval_dataloader = DataLoader(
        eval_dataset, batch_size=batch_size, collate_fn=default_data_collator, pin_memory=True
    )

    # Load the base Qwen2.5-Coder-7B model with trust_remote_code enabled.
    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        trust_remote_code=True,
        device_map="auto"
    )
    # Wrap the model with PEFT for prompt tuning; only the prompt tokens will be trainable.
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()
    model = model.to(device)

    # Set up the optimizer and learning rate scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    total_steps = len(train_dataloader) * num_epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)

    # Training loop
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0
        for batch in tqdm(train_dataloader, desc=f"Training Epoch {epoch}"):
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            total_loss += loss.item()
        avg_train_loss = total_loss / len(train_dataloader)
        logging.info(f"Epoch {epoch} average training loss: {avg_train_loss:.4f}")

        # Evaluation loop
        model.eval()
        eval_loss = 0.0
        with torch.no_grad():
            for batch in tqdm(eval_dataloader, desc="Evaluating"):
                batch = {k: v.to(device) for k, v in batch.items()}
                outputs = model(**batch)
                loss = outputs.loss
                eval_loss += loss.item()
        avg_eval_loss = eval_loss / len(eval_dataloader)
        logging.info(f"Epoch {epoch} average evaluation loss: {avg_eval_loss:.4f}")

    # Save the prompt tuning model and tokenizer
    output_dir = "outputs/prompt_tuning"
    os.makedirs(output_dir, exist_ok=True)
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    logging.info(f"Prompt tuning complete. Model saved at {output_dir}")

if __name__ == "__main__":
    main()
