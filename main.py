from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import pandas as pd
from sklearn.metrics import accuracy_score
import ast

def format_prompt(question, choices):
    prompt = f"{question}\n"
    choice_labels = ['A', 'B', 'C', 'D']
    for label, choice in zip(choice_labels, choices):
        prompt += f"{label}. {choice}\n"
    prompt += "Answer:"
    return prompt

def run_model1():
    codeqwen1d5_name = "Qwen/CodeQwen1.5-7B"
    codeqwen1d5_path = 'checkpoints'
    device = "cuda" if torch.cuda.is_available() else "cpu"

    codeqwen1d5 = AutoModelForCausalLM.from_pretrained(
        codeqwen1d5_path,
        device_map="auto",
        trust_remote_code=True
    )

    codeqwen1d5_tokenizer = AutoTokenizer.from_pretrained(
        codeqwen1d5_path,
        trust_remote_code=True
    )

    generated_answers = []
    results = []

    for index, row in df_val.iterrows():
        print(f"Processing question {index}...")
        prompt = format_prompt(row['question'], row['choices'])
        inputs = codeqwen1d5_tokenizer(prompt, return_tensors="pt", return_token_type_ids=False).to(device)

        outputs = codeqwen1d5.generate(**inputs, max_new_tokens=1)
        answer_tokens = outputs[:, inputs['input_ids'].shape[1]:]
        generated_answer = codeqwen1d5_tokenizer.decode(answer_tokens[0], skip_special_tokens=True).strip()

        generated_answers.append(generated_answer)

        results.append({
            'task_id': row['task_id'],
            'answer': generated_answer
        })

    # write results to a csv
    df_results = pd.DataFrame(results)
    df_results.to_csv('submission/codeqwen1d5_7b_submission.csv', index=False)

if __name__ == "__main__":
    df_train = pd.read_csv('data/b6_train_data.csv')
    # Ensure the 'choices' column is properly parsed as a list
    df_train['choices'] = df_train['choices'].apply(ast.literal_eval)
    df_val = pd.read_csv('data/b6_test_data.csv')
    df_val['choices'] = df_val['choices'].apply(ast.literal_eval)

    run_model1()